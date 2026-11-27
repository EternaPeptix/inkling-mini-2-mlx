"""Scheduler & BatchGenerator decode optimizations for Inkling 2 on vllm-mlx.

Installed by `vllm_mlx_inkling2.serve` alongside `patches.install()` and `stream_opt.install()`:
1. Per-request `samplers` in `BatchGenerator.insert()` without tearing down/recreating
   `BatchGenerator` when consecutive or concurrent requests use different sampling params.
2. `need_final_cache=[False]` for non-trimmable Inkling 2 `CacheList` models so
   `GenerationBatch` skips the wasted `max_tokens+1` forward pass and synchronous
   `flush()`/`unpack_b1()` on completion (vllm-mlx discards completion caches for
   non-trimmable topologies anyway).
3. Raise `Scheduler._clear_cache_interval` from 32 -> 512 (configurable via
   `LOOPLING_CLEAR_CACHE_INTERVAL`) to eliminate every-32-step `+1..5.5ms`
   `mx.clear_cache()` jitter spikes on 512 GB unified memory.
4. Eagerly pre-load DFlash 2 Q4 drafter weights during scheduler initialization when
   `LOOPLING_MTP=dflash2` so Request #1 pays 0 ms drafter load latency.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_INSTALLED = False
_SAMPLER_CACHE: dict[tuple[float, float, float], Any] = {}


def _get_cached_sampler(sampling_params: Any):
    from mlx_lm.sample_utils import make_sampler

    key = (
        float(getattr(sampling_params, "temperature", 0.0) or 0.0),
        float(getattr(sampling_params, "top_p", 1.0) if getattr(sampling_params, "top_p", None) is not None else 1.0),
        float(getattr(sampling_params, "min_p", 0.0) or 0.0),
    )
    samp = _SAMPLER_CACHE.get(key)
    if samp is None:
        if len(_SAMPLER_CACHE) >= 128:
            _SAMPLER_CACHE.clear()
        samp = _SAMPLER_CACHE[key] = make_sampler(temp=key[0], top_p=key[1], min_p=key[2])
    return samp, key


def _is_inkling2(model: Any) -> bool:
    inner = getattr(model, "model", None)
    lm = getattr(inner, "language_model", None) if inner is not None else None
    return lm is not None and hasattr(lm, "layer_visits") and hasattr(model, "args")


def _build_dflash2_status(model: Any, *, preloaded: bool = False) -> dict[str, Any]:
    raw = getattr(model, "dflash2_stats", None) or {}
    cycles = int(raw.get("cycles", 0) or 0)
    accepted = int(raw.get("accepted", 0) or 0)
    tokens = int(raw.get("tokens", 0) or 0)
    draft_ms = float(raw.get("draft_ms", 0.0) or 0.0)
    verify_ms = float(raw.get("verify_ms", 0.0) or 0.0)
    return {
        "enabled": True,
        "backend": "dflash2",
        "drafter_loaded": bool(preloaded or raw),
        "cycles": cycles,
        "accepted": accepted,
        "tokens": tokens,
        "mean_tpc": round(tokens / cycles, 3) if cycles > 0 else 0.0,
        "draft_ms": round(draft_ms, 2),
        "verify_ms": round(verify_ms, 2),
    }


def install() -> None:
    """Apply idempotent Scheduler decode & batching optimizations."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from mlx_lm.generate import StopSequences
    from vllm_mlx import scheduler as vsched

    clear_interval = max(32, int(os.environ.get("LOOPLING_CLEAR_CACHE_INTERVAL", "512")))

    orig_init = vsched.Scheduler.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._clear_cache_interval = clear_interval
        self._pending_insert_sampler = None
        self._pending_insert_stop_seq = None
        self._base_stop_tokens = None
        self._dflash2_preloaded = False
        self._is_inkling2_model = _is_inkling2(self.model)
        if self._is_inkling2_model and os.environ.get("LOOPLING_MTP") == "dflash2":
            try:
                from mlx_lm.models.inkling2 import dflash2 as DF2

                if os.path.isdir(DF2.DFLASH2_DIR):
                    DF2._get_drafter(self.model, DF2.DFLASH2_DIR, reset=False)
                    self._dflash2_preloaded = True
                    logger.info("[sched_opt] Pre-loaded DFlash 2 Q4 drafter from %s", DF2.DFLASH2_DIR)
            except Exception as exc:
                logger.warning("[sched_opt] DFlash 2 pre-load skipped: %s", exc)

    vsched.Scheduler.__init__ = _patched_init

    orig_get_stats = vsched.Scheduler.get_stats

    def _patched_get_stats(self) -> dict[str, Any]:
        stats = orig_get_stats(self)
        if getattr(self, "_is_inkling2_model", False) and os.environ.get("LOOPLING_MTP") == "dflash2":
            mtp_info = stats.get("mtp")
            if not isinstance(mtp_info, dict) or not mtp_info.get("enabled"):
                stats["mtp"] = _build_dflash2_status(
                    self.model, preloaded=bool(getattr(self, "_dflash2_preloaded", False))
                )
        return stats

    vsched.Scheduler.get_stats = _patched_get_stats

    orig_ensure = vsched.Scheduler._ensure_batch_generator

    def _patched_ensure(self, sampling_params):
        samp, key = _get_cached_sampler(sampling_params)
        self._pending_insert_sampler = samp
        extra_stops = getattr(sampling_params, "stop_token_ids", None)
        if extra_stops:
            if self._base_stop_tokens is None:
                self._base_stop_tokens = set(self._get_stop_tokens())
            all_stops = self._base_stop_tokens | set(extra_stops)
            self._pending_insert_stop_seq = StopSequences([[t] for t in all_stops])
        else:
            self._pending_insert_stop_seq = None

        if self.batch_generator is None:
            orig_ensure(self, sampling_params)
            self._current_sampler_params = key
            return

        # Keep existing BatchGenerator alive across sampling-param changes:
        # per-request `samplers` are passed directly into `BatchGenerator.insert()`.
        self._current_sampler_params = key

    vsched.Scheduler._ensure_batch_generator = _patched_ensure

    orig_create = vsched.Scheduler._create_batch_generator

    def _patched_create_bg(self, sampling_params):
        bg = orig_create(self, sampling_params)
        sig_params = inspect.signature(type(bg).insert).parameters
        has_samplers = "samplers" in sig_params
        has_stops = "stop_sequences" in sig_params
        has_nfc = "need_final_cache" in sig_params
        is_ink2 = getattr(self, "_is_inkling2_model", _is_inkling2(self.model))
        orig_insert = bg.insert

        def _insert_opt(prompts, *args, **kwargs):
            n = len(prompts)
            if has_samplers and ("samplers" not in kwargs or kwargs["samplers"] is None):
                s = getattr(self, "_pending_insert_sampler", None)
                if s is not None:
                    kwargs["samplers"] = [s] * n
            if has_stops and ("stop_sequences" not in kwargs or kwargs["stop_sequences"] is None):
                ss = getattr(self, "_pending_insert_stop_seq", None)
                if ss is not None:
                    kwargs["stop_sequences"] = [ss] * n
            if has_nfc and is_ink2 and ("need_final_cache" not in kwargs or kwargs["need_final_cache"] is None):
                kwargs["need_final_cache"] = [False] * n
            return orig_insert(prompts, *args, **kwargs)

        bg.insert = _insert_opt
        if is_ink2 and os.environ.get("LOOPLING_MTP") == "dflash2" and not hasattr(bg, "get_mtp_stats"):
            bg.get_mtp_stats = lambda: _build_dflash2_status(
                self.model,
                preloaded=bool(
                    getattr(self, "_dflash2_preloaded", False)
                    or getattr(self.model, "_dflash2_preloaded", False)
                ),
            )
        return bg

    vsched.Scheduler._create_batch_generator = _patched_create_bg

    def _run_dflash2_and_2pass_warmup(model):
        if not _is_inkling2(model):
            return
        if os.environ.get("LOOPLING_STARTUP_WARMUP", "1") in ("0", "", "false", "False"):
            return
        if getattr(model, "_dflash2_2pass_warmed", False):
            return
        import time

        t0 = time.perf_counter()
        try:
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.models.inkling2.batchdecode import BatchDecoder

            # 1. Warm S=1024 (2-pass SDPA) + coupled/typical sampler (t=0.7, p=0.95)
            c = make_prompt_cache(model)
            prompt_1k = mx.ones((1, 1024), dtype=mx.int32)
            _ = model(prompt_1k, cache=c)
            mx.eval([x.state for x in c])

            bd = BatchDecoder(model, [c])
            try:
                t_tok = mx.array([1], dtype=mx.int32)
                y_g = bd.step(t_tok)
                mx.eval(y_g)
                samp_07 = make_sampler(temp=0.7, top_p=0.95)
                y_s = bd.sample_step(t_tok, samp_07)
                mx.eval(y_s)
            finally:
                bd.release_stage()
            del bd, c

            # 2. Warm DFlash2 drafter + verify graphs (W=4,6,8 greedy; W=4,6 coupled)
            if os.environ.get("LOOPLING_MTP") == "dflash2":
                from mlx_lm.models.inkling2 import dflash2 as DF2
                from mlx_lm.models.inkling2.mtp import (
                    _feats_from_dict,
                    default_taps,
                    parse_taps,
                )

                dr = DF2._get_drafter(model, DF2.DFLASH2_DIR, reset=True)
                tm = model.model.language_model
                passes = tm.args.loop_config["num_passes"]
                taps = parse_taps(
                    os.environ.get("LOOPLING_MTP_TAPS", default_taps(passes)),
                    passes,
                )
                tap_keys = [t if t in ("F", "f", "o") else int(t[1:]) for t in taps]
                tap_set = {k for k in tap_keys if isinstance(k, int)}

                def _pad_cache_for_warmup(base_cache, target_len: int):
                    c_out = copy.deepcopy(base_cache)
                    pad_evals = []
                    for cl in c_out:
                        kv = cl[0]
                        if getattr(kv, "_packed", None) is not None:
                            kv.unpack_b1()
                        cur_k = kv._keys
                        cur_v = kv._values
                        if cur_k is None or cur_v is None:
                            continue
                        is_rot = hasattr(kv, "_idx")
                        want_len = min(target_len, int(kv.max_size)) if is_rot else target_len
                        need = want_len - int(cur_k.shape[2])
                        if need > 0:
                            ek = mx.zeros((1, cur_k.shape[1], need, cur_k.shape[3]), cur_k.dtype)
                            ev = mx.zeros((1, cur_v.shape[1], need, cur_v.shape[3]), cur_v.dtype)
                            kv._keys = mx.concatenate([cur_k, ek], axis=2)
                            kv._values = mx.concatenate([cur_v, ev], axis=2)
                        kv.offset = target_len
                        if is_rot:
                            kv._idx = want_len
                        pad_evals.extend((kv._keys, kv._values))
                    if pad_evals:
                        mx.eval(pad_evals)
                    return c_out

                def _run_df2_warmup_suite(c_base, ctx_len: int):
                    for w_val, samp, pa_adapt_override in configs:
                        os.environ["LOOPLING_DFLASH2_W"] = str(w_val)
                        if pa_adapt_override is not None:
                            os.environ["LOOPLING_DFLASH2_PA_ADAPT"] = pa_adapt_override
                        elif orig_adapt_env is None:
                            os.environ.pop("LOOPLING_DFLASH2_PA_ADAPT", None)
                        else:
                            os.environ["LOOPLING_DFLASH2_PA_ADAPT"] = orig_adapt_env
                        c_i = copy.deepcopy(c_base)
                        mx.eval([x.state for x in c_i])
                        bd_i = BatchDecoder(model, [c_i])
                        try:
                            if w_val == 4:
                                tail_ws = (2, 3, 4, 5, 6, 7) if samp is None else (2, 3, 4, 5)
                                for w_tail in tail_ws:
                                    t_ret, *_ = bd_i.verify_w(
                                        mx.ones((w_tail,), dtype=mx.int32),
                                        tap_visits=tap_set | {"F"},
                                        sampler=samp,
                                    )
                                    mx.eval(t_ret)
                                # Also warm shallow tree shapes (max_d=1, 2) that draft_hybrid_tree emits at tail
                                for pars, deps in (
                                    ((-1, 0, 0), (0, 1, 1)),
                                    ((-1, 0, 0, 1), (0, 1, 1, 2)),
                                ):
                                    t_ret, *_ = bd_i.verify_w(
                                        mx.ones((len(pars),), dtype=mx.int32),
                                        tap_visits=tap_set | {"F"},
                                        parents=pars,
                                        depths=deps,
                                        sampler=samp,
                                    )
                                    mx.eval(t_ret)
                            st = dr.new_stream_state(allocate=True)
                            dr.append_ctx_batch([st], [f0])
                            runner = DF2._DFlash2C1Runner(
                                bd_i,
                                dr,
                                st,
                                y0,
                                int(y0.item()),
                                all_tokens=[1] * ctx_len + [int(y0.item())],
                                tap_keys=tap_keys,
                                tap_set=tap_set,
                                sampler=samp,
                                max_tokens=100,
                                num_tokens=1,
                                track_cache_rb=False,
                            )
                            runner.step()
                            runner.discard()
                        finally:
                            bd_i.release_stage()
                        del bd_i, c_i

                orig_w_env = os.environ.get("LOOPLING_DFLASH2_W")
                orig_adapt_env = os.environ.get("LOOPLING_DFLASH2_PA_ADAPT")
                try:
                    import copy

                    c2 = make_prompt_cache(model)
                    _ = model(mx.ones((1, 64), dtype=mx.int32), cache=c2)
                    mx.eval([x.state for x in c2])
                    bd_init = BatchDecoder(model, [c2])
                    try:
                        t_tok2 = mx.array([1], dtype=mx.int32)
                        res_taps = bd_init.step_with_taps(
                            t_tok2, tap_visits=tap_set, sampler=None
                        )
                        y0, taps0 = res_taps[0], res_taps[1]
                        mx.eval(y0)
                        f0 = _feats_from_dict(taps0, tap_keys)
                    finally:
                        bd_init.release_stage()
                    del bd_init

                    configs = [
                        (4, None, None),
                        (6, None, None),
                        (8, None, "0"),
                        (4, samp_07, None),
                        (6, samp_07, None),
                    ]
                    # 2a. Single-pass packed (< _TWO_PASS_N)
                    _run_df2_warmup_suite(c2, 64)

                    # 2b. Two-pass packed (_TWO_PASS_N <= S < _KV_STACK_MAX_CAP)
                    c_2p = _pad_cache_for_warmup(c2, 1200)
                    _run_df2_warmup_suite(c_2p, 1200)
                    del c_2p

                    # 2c. Unpacked-full long SDPA (> _KV_STACK_MAX_CAP)
                    c_unp = _pad_cache_for_warmup(c2, 32769)
                    _run_df2_warmup_suite(c_unp, 32769)
                    del c_unp, c2
                finally:
                    if orig_w_env is None:
                        os.environ.pop("LOOPLING_DFLASH2_W", None)
                    else:
                        os.environ["LOOPLING_DFLASH2_W"] = orig_w_env
                    if orig_adapt_env is None:
                        os.environ.pop("LOOPLING_DFLASH2_PA_ADAPT", None)
                    else:
                        os.environ["LOOPLING_DFLASH2_PA_ADAPT"] = orig_adapt_env

                dr.reset()
                dr._budget = None
                dr._costs = None
                model.dflash2_stats = {
                    "cycles": 0,
                    "accepted": 0,
                    "tokens": 0,
                    "draft_ms": 0.0,
                    "verify_ms": 0.0,
                }
                model._dflash2_preloaded = True

            mx.clear_cache()
            model._dflash2_2pass_warmed = True
            logger.info(
                "[vllm_mlx_inkling2] 2-pass SDPA + DFlash2 warmup complete in %.2fs",
                time.perf_counter() - t0,
            )
        except Exception as e:
            logger.warning(
                "[vllm_mlx_inkling2] 2-pass SDPA / DFlash2 warmup skipped: %s", e
            )

    try:
        from vllm_mlx.engine import batched as engine_batched

        orig_prepare = engine_batched.BatchedEngine.prepare_for_start

        def _patched_prepare_for_start(self, *args, **kwargs):
            res = orig_prepare(self, *args, **kwargs)
            _run_dflash2_and_2pass_warmup(getattr(self, "_model", None))
            return res

        engine_batched.BatchedEngine.prepare_for_start = _patched_prepare_for_start
    except Exception as e:
        logger.debug("[vllm_mlx_inkling2] BatchedEngine warmup hook skipped: %s", e)

    logger.info(
        "[vllm_mlx_inkling2] Scheduler decode opts installed (clear_cache_interval=%d, per-req samplers=ON)",
        clear_interval,
    )
