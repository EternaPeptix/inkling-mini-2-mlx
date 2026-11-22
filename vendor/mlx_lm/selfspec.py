"""Self-speculative decoding for looped models (Loopling).

Draft with few passes of the shared core, verify k drafts with the target pass count in one forward.
Enabled by LOOPLING_SELFSPEC=<draft_passes>:<k> (default off). Greedy only. One parameter set; two caches
(different layer-visit layouts). The target cache is rolled back by tail truncation (+ recorded conv windows),
the draft cache by per-step snapshots.
"""
import os
import time

import mlx.core as mx

from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

STATS = {"rounds": []}  # (tokens emitted, accepted drafts, k used, seconds) per round, for benchmarks


def parse_env():
    v = os.environ.get("LOOPLING_SELFSPEC")
    if not v:
        return None
    d, k = v.split(":")
    return int(d), int(k)


def applicable(model, d, k):
    lm = getattr(getattr(model, "model", None), "language_model", None)
    cfg = getattr(getattr(lm, "args", None), "loop_config", None)
    return cfg is not None and 1 <= d < cfg["num_passes"] and k >= 1


def _alias(x):
    # a new array object over the same data: later in-place slice writes cannot touch it
    return x[:, :, :, :] if x is not None else None


def _snap(caches):
    out = []
    for c in caches:
        kv, conv = c[0], c[1]
        if isinstance(kv, RotatingKVCache):
            s = ("rot", _alias(kv.keys), _alias(kv.values), kv.offset, kv._idx)
        else:
            s = ("kv", kv.offset)
        out.append((s, list(conv.cache)))
    return out


def _restore(caches, snap):
    for c, (s, conv) in zip(caches, snap):
        kv = c[0]
        if s[0] == "rot":
            kv.keys, kv.values, kv.offset, kv._idx = _alias(s[1]), _alias(s[2]), s[3], s[4]
        else:
            kv.offset = s[1]
        c[1].cache = list(conv)


def _rollback_target(caches, r, m, S):
    """Drop the last r of the S tokens just verified; m = S - r tokens stay."""
    for c in caches:
        kv, conv = c[0], c[1]
        if isinstance(kv, RotatingKVCache):
            # S > 1 went through the concat path: storage is in temporal order and _idx == length
            L = kv.keys.shape[2]
            kv.keys, kv.values = kv.keys[..., : L - r, :], kv.values[..., : L - r, :]
            kv.offset -= r
            kv._idx = L - r
        else:
            kv.trim(r)
        for i, w in enumerate(conv.windows):
            n = w.shape[1] - S  # kernel_size - 1
            conv.cache[i] = w[:, m : m + n, :]


def _bi_on():
    return os.environ.get("LOOPLING_BATCH_INVARIANT") == "1"


def _rollback_target_bi(caches, snap, r, m, S):
    """LOOPLING_BATCH_INVARIANT: the verify forward wrote its rows with 1-token in-place cache updates. Rotating caches go
    back to the pre-verify snapshot and replay the m kept rows (exactly the writes sequential decode would have made)."""
    for c, (s, _conv) in zip(caches, snap):
        kv, conv = c[0], c[1]
        if isinstance(kv, RotatingKVCache):
            rows = kv._bi_rows
            kv.keys, kv.values, kv.offset, kv._idx = _alias(s[1]), _alias(s[2]), s[3], s[4]
            for j in range(m):
                kv.update_and_fetch(*rows[j])
        else:
            kv.trim(r)
        for i, w in enumerate(conv.windows):
            n = w.shape[1] - S
            conv.cache[i] = w[:, m : m + n, :]


def _lp_argmax(logits):
    if _bi_on() and logits.shape[0] > 1:  # per-row reductions, as the 1-token path does them
        parts = [_lp_argmax(logits[j : j + 1]) for j in range(logits.shape[0])]
        return mx.concatenate([p[0] for p in parts]), mx.concatenate([p[1] for p in parts])
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return lp, mx.argmax(lp, axis=-1)


def selfspec_generate_step(prompt, model, d, k, stream, *, max_tokens=256, prefill_step_size=2048, **_):
    cfg = model.model.language_model.args.loop_config
    T = cfg["num_passes"]
    STATS["rounds"] = []

    def fwd(cache, y, p):
        cfg["num_passes"] = p
        return model(y[None], cache=cache)

    try:
        with mx.stream(stream):
            cfg["num_passes"] = T
            tcache = make_prompt_cache(model)
            cfg["num_passes"] = d
            dcache = make_prompt_cache(model)
            cfg["num_passes"] = T
            y = prompt.astype(mx.uint32)
            for cache, p in ((dcache, d), (tcache, T)):
                yy = y
                while yy.size > 1:
                    n = min(prefill_step_size, yy.size - 1)
                    fwd(cache, yy[:n], p)
                    mx.eval([c.state for c in cache])
                    yy = yy[n:]
                    mx.clear_cache()
            for c in tcache:
                c[1].record = True
            pending = y[-1:]
            draft_in = pending
            ntoks = 0
            clear_mark = 0
            while ntoks < max_tokens:
                t0 = time.perf_counter()
                ke = min(k, max_tokens - ntoks - 1)
                if ke == 0:
                    for c in tcache:
                        c[1].record = False
                    lp, g = _lp_argmax(fwd(tcache, pending, T)[0, -1:, :])
                    mx.eval(g)
                    yield g.item(), lp[0], False
                    return
                ys, snaps, yin = [], {}, draft_in
                for i in range(ke):
                    if i >= 1:
                        snaps[i] = _snap(dcache)
                    lg = fwd(dcache, yin, d)[0, -1, :]
                    yin = mx.argmax(lg, axis=-1, keepdims=True).astype(mx.uint32)
                    mx.async_eval(yin)
                    ys.append(yin)
                dcat = mx.concatenate(ys)
                vin = mx.concatenate([pending, dcat])
                S = ke + 1
                if _bi_on():
                    from mlx_lm.models.inkling2 import batch_invariant as bi
                    tsnap = _snap(tcache)
                    with bi.active():
                        lg_v = fwd(tcache, vin, T)[0]
                else:
                    lg_v = fwd(tcache, vin, T)[0]
                lp, g = _lp_argmax(lg_v)
                mx.eval(g, dcat)
                gl, dl = g.tolist(), dcat.tolist()
                n = 0
                while n < ke and gl[n] == dl[n]:
                    n += 1
                for i in range(n):
                    yield dl[i], lp[i], True
                yield gl[n], lp[n], False
                ntoks += n + 1
                STATS["rounds"].append((n + 1, n, ke, time.perf_counter() - t0))
                if ntoks >= max_tokens:
                    return
                pending = g[n : n + 1].astype(mx.uint32)
                if n < ke:
                    if _bi_on():
                        _rollback_target_bi(tcache, tsnap, ke - n, n + 1, S)
                    else:
                        _rollback_target(tcache, ke - n, n + 1, S)
                    if n + 1 < ke:
                        _restore(dcache, snaps[n + 1])
                    draft_in = pending
                else:
                    draft_in = mx.concatenate([ys[-1], pending])
                if ntoks // 256 != clear_mark:
                    clear_mark = ntoks // 256
                    mx.clear_cache()
    finally:
        cfg["num_passes"] = T
