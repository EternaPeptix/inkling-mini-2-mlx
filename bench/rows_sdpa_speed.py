"""Speed of rows/sdpa.py vs fastdecode._core (mask build + sdpa), median us over REPS timed reps.
Cache update is excluded from both sides (the reference's mask/sdpa part is replicated from _core with a no-op cache).
run: LANE=e3 RHOST=512S2 scripts/r --gpu 'python bench/rows_sdpa_speed.py'
"""
import statistics
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.inkling2 import fastdecode, layers as L
from mlx_lm.models.inkling2.rows.sdpa import row_meta, sdpa_rows

H, HKV, D, WIN, R = 24, 8, 128, 1024, 1024
SC = D**-0.5
REPS = 200


class Frozen:
    """Wraps the real cache; update_and_fetch returns a fixed pair so only mask + sdpa are timed."""

    def __init__(self, kv, k, v):
        self.kv, self.k, self.v = kv, k, v
        kv.update_and_fetch = lambda a, b: (k, v)


def make(kind, S):
    kv = KVCache() if kind == "f" else RotatingKVCache(max_size=WIN)
    mx.random.seed(S)
    n = S - 1 if kind == "f" else S
    kv.update_and_fetch(
        mx.random.normal((1, HKV, n, D)).astype(mx.bfloat16), mx.random.normal((1, HKV, n, D)).astype(mx.bfloat16)
    )
    if kind == "s":  # wrap the ring
        for _ in range(37):
            kv.update_and_fetch(mx.random.normal((1, HKV, 1, D)).astype(mx.bfloat16), mx.random.normal((1, HKV, 1, D)).astype(mx.bfloat16))
    mx.eval(kv.keys, kv.values)
    return kv


def timeit(fn):
    for _ in range(5):
        mx.eval(fn())
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t) * 1e6)
    return statistics.median(ts)


def case(kind, S, B):
    window = WIN if kind == "s" else 0
    layer = type("X", (), {})()
    layer.self_attn = type("A", (), {"sliding_window": window, "scaling": SC})()
    kvs = [make(kind, S) for _ in range(B)]
    k1 = k2 = None
    # for the full cache, the "append" step grows it by one; emulate by using the existing arrays as-is
    qs = [mx.random.normal((1, H, 1, D)).astype(mx.bfloat16) for _ in range(B)]
    pbs = [mx.random.normal((1, 1, H, R)).astype(mx.bfloat16) for _ in range(B)]
    sks = [mx.random.normal((1, 1, H)).astype(mx.bfloat16) for _ in range(B)]
    kn = mx.zeros((1, HKV, 1, D), mx.bfloat16)
    frozen = [Frozen(kv, kv.keys[..., : kv.keys.shape[2] if kind == "s" else S, :], kv.values[..., : kv.values.shape[2] if kind == "s" else S, :]) for kv in kvs]
    mx.eval(qs + pbs + sks)
    fastdecode._ctx.clear()
    # frozen caches: _end() uses kv.offset (full) / kv._offset+d; state is whatever the cache holds
    ks = [f.k for f in frozen]
    vs = [f.v for f in frozen]
    meta = [row_meta(f.kv) for f in frozen]
    if kind == "f":  # row_meta(kv) uses kv.offset == S-1 ; the frozen pair has S keys => treat offset as S
        meta = [(S, 0, 0, ks[i].shape[2], vs[i].shape[2]) for i in range(B)]
        for f in frozen:
            f.kv.offset = S
    qc, pbc, skc = mx.concatenate(qs), mx.concatenate(pbs), mx.concatenate(sks)

    def ref():
        return [fastdecode._core(layer, qs[i], kn, kn, pbs[i], sks[i], frozen[i].kv) for i in range(B)]

    def rows():
        return sdpa_rows(qc, ks, vs, pbc, skc, meta, scale=SC, window=window)

    # note: ref() reuses cached mask contexts (_ctx) like the real decode does between identical states
    return timeit(ref), timeit(rows)


def main():
    print(f"{'kind':8s} {'S':>6s} {'B':>2s} {'ref us':>9s} {'rows us':>9s} {'speedup':>8s}  (ref = B x _core calls evaluated together)")
    for kind, S in (("f", 512), ("s", 1024), ("f", 1024), ("f", 4096), ("f", 16384)):
        for B in (1, 8):
            r, w = case(kind, S, B)
            print(f"{'full' if kind == 'f' else 'sliding':8s} {S:6d} {B:2d} {r:9.0f} {w:9.0f} {r / w:7.2f}x", flush=True)


if __name__ == "__main__":
    main()
