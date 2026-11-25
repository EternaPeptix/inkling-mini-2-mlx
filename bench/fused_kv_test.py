"""Exactness of rows/kvstore.py (RowKV) against per-row real caches + fastdecode._core, bit patterns.

B streams with different prompt lengths (rows below 1024, rows crossing 1024 during the test so the ring wraps / the
2-pass regime starts, rows > 4096, chunk-prefilled rows), real KVCache / RotatingKVCache(1024) classes. T decode steps
both ways: (a) per row `_core` (cache update + mask + sdpa), (b) RowKV.update + RowKV.sdpa. Compares the attention
output of every step and, at the end, the flushed caches (keys/values buffers incl. shape, offset, _idx).
run: LANE=f5 RHOST=512S1 scripts/r --gpu 'python bench/fused_kv_test.py [quick]'
"""
import sys
import time
from types import SimpleNamespace

import numpy as np

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.inkling2 import fastdecode
from mlx_lm.models.inkling2.rows.kvstore import RowKV

H, HKV, D, WIN, R = 24, 8, 128, 1024, 1024
SCALE = D**-0.5
T = 48
# prompt lengths; (length, chunk) chunk = prefill chunk size (0 = one shot)
LENS = [(1010, 0), (300, 0), (5000, 0), (1500, 512), (100, 0), (1023, 0), (4200, 700), (700, 0), (1024, 0),
        (2048, 512), (10, 0), (1000, 0), (3000, 0)]
LENS_FULL = [(1010, 0), (300, 0), (5000, 0), (1500, 512), (100, 0), (255, 0), (4200, 700), (256, 0), (1024, 0),
             (2048, 512), (10, 0), (1000, 0), (3000, 0)]


def build(kind, spec, seed):
    n, chunk = spec
    rs = np.random.RandomState(seed)
    rnd = lambda m: mx.array((rs.randn(1, HKV, m, D) * 1.5).astype(np.float32)).astype(mx.bfloat16)
    out = []
    pk, pv = rnd(n), rnd(n)
    for _ in range(2):
        kv = KVCache() if kind == "f" else RotatingKVCache(max_size=WIN)
        if chunk:
            for s in range(0, n, chunk):
                kv.update_and_fetch(pk[:, :, s : s + chunk], pv[:, :, s : s + chunk])
        else:
            kv.update_and_fetch(pk, pv)
        mx.eval(kv.keys, kv.values)
        out.append(kv)
    return out


def bits(a):
    return np.array(a.view(mx.uint16)).reshape(-1)


def run(kind, B, seed, lens, steps=T, pbdt=mx.bfloat16, gated=False, paged=None):
    window = WIN if kind == "s" else 0
    layer = SimpleNamespace(self_attn=SimpleNamespace(sliding_window=window, scaling=SCALE))
    specs = lens[:B]
    rows = [build(kind, sp, seed * 100 + i) for i, sp in enumerate(specs)]
    ref = [r[0] for r in rows]
    store = RowKV([r[1] for r in rows], paged=paged)
    rs = np.random.RandomState(seed + 5)
    bad = 0
    nel = 0
    for step in range(steps):
        q = mx.array(rs.randn(B, H, 1, D).astype(np.float32)).astype(mx.bfloat16)
        k = mx.array((rs.randn(B, HKV, 1, D) * 1.5).astype(np.float32)).astype(mx.bfloat16)
        v = mx.array((rs.randn(B, HKV, 1, D) * 1.5).astype(np.float32)).astype(mx.bfloat16)
        pb = mx.array((2 * rs.randn(B, 1, H, R)).astype(np.float32)).astype(pbdt)
        sk = mx.array(rs.randn(B, 1, H).astype(np.float32)).astype(mx.bfloat16)
        gate = mx.array(rs.randn(B, 1, H * D).astype(np.float32)).astype(mx.bfloat16) if gated else None
        outs = [fastdecode._core(layer, q[i : i + 1], k[i : i + 1], v[i : i + 1], pb[i : i + 1], sk[i : i + 1], ref[i])
                for i in range(B)]
        if gated:
            g4 = gate.reshape(B, 1, H, D).astype(mx.float32)
            outs = [(outs[i].astype(mx.float32) * mx.sigmoid(g4[i : i + 1])).astype(mx.bfloat16) for i in range(B)]
        store.update(k.reshape(B, HKV, D), v.reshape(B, HKV, D))
        o = store.sdpa(q, pb, sk, scale=SCALE, window=window, gate=gate)
        mx.eval(outs + [o])
        for i in range(B):
            a, b = bits(outs[i]), bits(o[i : i + 1])
            assert a.shape == b.shape
            bad += int((a != b).sum())
            nel += a.size
    store.to_caches()
    cbad = 0
    for i in range(B):
        a, c = ref[i], rows[i][1]
        assert a.offset == c.offset, (a.offset, c.offset)
        if kind == "s":
            assert a._idx == c._idx, (a._idx, c._idx)
        for x, y in ((a.keys, c.keys), (a.values, c.values)):
            assert x.shape == y.shape, (kind, i, x.shape, y.shape)
            cbad += int((bits(x) != bits(y)).sum())
    return bad, cbad, nel, [s[0] for s in specs]


STAGGERED = [(17, 0), (255, 0), (256, 0), (257, 0), (1023, 0), (1024, 0), (2049, 0), (8192, 0)]


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    t0 = time.time()
    allok = True
    print(f"{'visit':8s} {'B':>2s} {'attn mismatches':>16s} {'cache mismatches':>17s}  prompts")
    seed = 1
    for kind in ("f", "s"):
        for B in (1, 2, 3, 5, 8, 13):
            if quick and B in (3, 13):
                continue
            seed += 1
            lens = LENS_FULL if kind == "f" else LENS
            bad, cbad, nel, pl = run(kind, B, seed, lens)
            ok = bad == 0 and cbad == 0
            allok &= ok
            print(f"{'full' if kind == 'f' else 'sliding':8s} {B:2d} {bad:10d} / {nel // 1000}k {cbad:12d}  {'PASS' if ok else 'FAIL'} {pl}", flush=True)
    # f32 pb
    bad, cbad, nel, pl = run("s", 3, 99, LENS, pbdt=mx.float32)
    allok &= bad == 0 and cbad == 0
    print(f"sliding  3 f32 pb: attn {bad} cache {cbad}", flush=True)
    # all rows already in the 2-pass regime (narrow threadgroup geometry)
    for kind in ("f", "s"):
        bad, cbad, nel, pl = run(kind, 4, 77, [(1500, 0), (5000, 0), (1100, 0), (2048, 512)])
        allok &= bad == 0 and cbad == 0
        print(f"{kind} all>=1024 B=4: attn {bad} cache {cbad}", flush=True)
    # staggered page (256) and 2-pass (1024) boundary crossing over T=300 decode steps, gated & ungated
    if not quick:
        for kind in ("f", "s"):
            for g in (False, True):
                bad, cbad, nel, _ = run(kind, 8, 123 + int(g), STAGGERED, steps=300, gated=g)
                allok &= bad == 0 and cbad == 0
                print(f"{kind} staggered-300 gate={g}: attn {bad} / {nel // 1000}k cache {cbad}", flush=True)
    # a plan shared by two stores with identical row state (as the lead would use it: one plan per visit kind)
    shared()
    print(f"SUMMARY {'ALL PASS' if allok else 'FAIL'} ({time.time() - t0:.0f}s)")
    sys.exit(0 if allok else 1)


def shared():
    B = 4
    specs = [(1010, 0), (300, 0), (5000, 0), (100, 0)]
    a = [build("f", sp, 11 + i) for i, sp in enumerate(specs)]
    b = [build("f", sp, 11 + i) for i, sp in enumerate(specs)]
    s1, s2 = RowKV([r[1] for r in a]), RowKV([r[1] for r in b])
    rs = np.random.RandomState(3)
    for _ in range(300):  # includes growth reallocation for the 300 and 5000-row (cap 5120 -> row 100..)
        k = mx.array(rs.randn(B, HKV, D).astype(np.float32)).astype(mx.bfloat16)
        p = s1.plan()
        s1.update(k, k, p)
        s2.update(k, k, p)
    if s1.paged:
        mx.eval(s1.base_K, s1.ov_K, s1.act_K, s2.base_K, s2.ov_K, s2.act_K)
        assert (bits(s1.base_K) == bits(s2.base_K)).all()
        assert (bits(s1.ov_K) == bits(s2.ov_K)).all()
        assert (bits(s1.act_K) == bits(s2.act_K)).all()
        s1.to_caches()
        s2.to_caches()
        for c1, c2 in zip(s1.caches, s2.caches):
            assert (bits(c1.keys) == bits(c2.keys)).all()
    else:
        mx.eval(s1.K, s2.K)
        assert (bits(s1.K) == bits(s2.K)).all()
    print("shared plan ok", flush=True)


if __name__ == "__main__":
    main()
