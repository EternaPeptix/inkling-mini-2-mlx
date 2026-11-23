"""Per-visit us for KV update + decode attention: RowKV vs (per-row update_and_fetch + sdpa_rows) vs per-row _core.
Also: in-place check (update time must not scale with Cap).
run: LANE=f5 RHOST=512S1 scripts/r --gpu 'python bench/fused_kv_speed.py'"""
import time
from types import SimpleNamespace

import numpy as np

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.inkling2 import fastdecode
from mlx_lm.models.inkling2.rows import sdpa as RS
from mlx_lm.models.inkling2.rows.kvstore import RowKV

H, HKV, D, WIN, R = 24, 8, 128, 1024, 1024
SCALE = D**-0.5
REPS, VIS = 20, 12  # VIS visits chained per eval (amortize sync), time per visit


def rnd(*s):
    return (mx.random.normal(s) * 1.5).astype(mx.bfloat16)


def mk(kind, n):
    kv = KVCache() if kind == "f" else RotatingKVCache(max_size=WIN)
    kv.update_and_fetch(rnd(1, HKV, n, D), rnd(1, HKV, n, D))
    mx.eval(kv.keys, kv.values)
    return kv


def timeit(fn):
    fn()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) / VIS * 1e6


def case(kind, B, n):
    window = WIN if kind == "s" else 0
    layer = SimpleNamespace(self_attn=SimpleNamespace(sliding_window=window, scaling=SCALE))
    lens = [n + 7 * i for i in range(B)]
    q = rnd(B, H, 1, D)
    pb = rnd(B, 1, H, R)
    sk = rnd(B, 1, H)
    k, v = rnd(B, HKV, 1, D), rnd(B, HKV, 1, D)
    out = {}
    caches = [mk(kind, l) for l in lens]

    def core():
        acc = []
        for _ in range(VIS):
            o = [fastdecode._core(layer, q[i : i + 1], k[i : i + 1], v[i : i + 1], pb[i : i + 1], sk[i : i + 1], c)
                 for i, c in enumerate(caches)]
            acc += o
        mx.eval(acc)

    def per_row():
        acc = []
        for _ in range(VIS):
            ks, vs, meta = [], [], []
            for i, c in enumerate(caches2):
                ka, va = c.update_and_fetch(k[i : i + 1], v[i : i + 1])
                ks.append(ka), vs.append(va), meta.append(RS.row_meta(c))
            acc.append(RS.sdpa_rows(q, ks, vs, pb, sk, meta, scale=SCALE, window=window))
        mx.eval(acc)

    caches2 = [mk(kind, l) for l in lens]
    store = RowKV([mk(kind, l) for l in lens])
    kk, vv = k.reshape(B, HKV, D), v.reshape(B, HKV, D)

    def rowkv():
        acc = []
        for _ in range(VIS):
            store.update(kk, vv)
            acc.append(store.sdpa(q, pb, sk, scale=SCALE, window=window))
        mx.eval(acc)

    out["core"] = timeit(core) if B <= 8 else None
    out["rows"] = timeit(per_row) if B <= 11 else None
    out["rowkv"] = timeit(rowkv)
    return out


def inplace():
    # NOTE: eval after EVERY update can defeat donation (the previous command buffer's completion handler may still hold the
    # buffer -> scatter copies the whole store); a decode step chains the updates of many visits in one graph, so it is fine.
    print("in-place check: RowKV.update only (B=8), us per update vs Cap (must be flat)")
    for n in (300, 2000, 8000, 30000):
        store = RowKV([mk("f", n) for _ in range(8)])
        kk, vv = rnd(8, HKV, D), rnd(8, HKV, D)

        def f():
            for _ in range(VIS):
                store.update(kk, vv)
            mx.eval(store.K, store.V)

        print(f"  cap {store.cap:6d} ({8 * HKV * store.cap * D * 2 / 1e6:7.0f} MB per array): {timeit(f):7.1f} us", flush=True)


def main():
    print(f"{'visit':8s} {'B':>2s} {'S':>6s} {'B x _core':>10s} {'upd_and_fetch+sdpa_rows':>24s} {'RowKV':>8s}  (us / visit, VIS visits chained per eval, as in a decode step)")
    for kind in ("f", "s"):
        for B in (1, 8):
            for n in ((300, 2000, 8000) if kind == "f" else (300, 1500)):
                o = case(kind, B, n)
                f = lambda x: "-" if x is None else f"{x:.0f}"
                print(f"{'full' if kind == 'f' else 'sliding':8s} {B:2d} {n:6d} {f(o['core']):>10s} {f(o['rows']):>24s} {f(o['rowkv']):>8s}", flush=True)
    inplace()


if __name__ == "__main__":
    main()
