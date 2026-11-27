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
        self._is_inkling2_model = _is_inkling2(self.model)
        if self._is_inkling2_model and os.environ.get("LOOPLING_MTP") == "dflash2":
            try:
                from mlx_lm.models.inkling2 import dflash2 as DF2

                if os.path.isdir(DF2.DFLASH2_DIR):
                    DF2._get_drafter(self.model, DF2.DFLASH2_DIR, reset=False)
                    logger.info("[sched_opt] Pre-loaded DFlash 2 Q4 drafter from %s", DF2.DFLASH2_DIR)
            except Exception as exc:
                logger.warning("[sched_opt] DFlash 2 pre-load skipped: %s", exc)

    vsched.Scheduler.__init__ = _patched_init

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
        return bg

    vsched.Scheduler._create_batch_generator = _patched_create_bg
    logger.info(
        "[vllm_mlx_inkling2] Scheduler decode opts installed (clear_cache_interval=%d, per-req samplers=ON)",
        clear_interval,
    )
