"""Exactness test of rows/sdpa.py against fastdecode._core, row by row (bit patterns).

Caches are the real KVCache / RotatingKVCache(max_size=1024) filled with random bf16 keys/values through their own
update_and_fetch (one prefill chunk, then single-token decode steps, so the ring layout and `_idx` are the cache's own).
Per row two identical caches are built from the same data: one goes through `_core` (the reference B=1 path, which also
does the cache update), the other through `update_and_fetch` + `sdpa_rows`.
run: LANE=e3 RHOST=512S2 scripts/r --gpu 'python bench/rows_sdpa_test.py [quick]'
"""
import sys
import time
from types import SimpleNamespace

import numpy as np

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.inkling2 import fastdecode
from mlx_lm.models.inkling2.rows.sdpa import row_meta, sdpa_rows

H, HKV, D, WIN = 24, 8, 128, 1024
SCALE = D**-0.5
R_FULL = 1024  # rel_extent (also the sliding window size)

# (kind, prefill tokens, decode steps incl. the final one)
FULL = {
    "full S=1": ("f", 0, 1),
    "full S=2": ("f", 1, 1),
    "full S=300": ("f", 299, 1),
    "full S=1023": ("f", 1022, 1),
    "full S=1024": ("f", 1023, 1),
    "full S=1025": ("f", 1024, 1),
    "full S=1500": ("f", 1000, 500),
    "full S=5000": ("f", 4999, 1),
    "full S=20000": ("f", 19000, 1000),
    "full S=70000": ("f", 69999, 1),
}
SLIDE = {
    "slide S=300": ("s", 200, 100),
    "slide S=1023": ("s", 1022, 1),
    "slide S=1024 full ring": ("s", 1024, 1),
    "slide S=1024 idx~1": ("s", 1500, 1),
    "slide wrap +37": ("s", 1024, 37),
    "slide wrap +300": ("s", 1500, 300),
    "slide wrap +1023": ("s", 3000, 1023),
    "slide wrap +1024": ("s", 3000, 1024),
    "slide wrap +2000": ("s", 5000, 2000),
    "slide prefill 70000": ("s", 70000, 3),
}


def build(spec, seed):
    """-> (kvA_ref, kvB, last k, last v)"""
    kind, p, d = spec
    rs = np.random.RandomState(seed)

    def rnd(n):
        return mx.array(rs.randn(1, HKV, n, D).astype(np.float32)).astype(mx.bfloat16)

    pk, pv = (rnd(p), rnd(p)) if p else (None, None)
    dk, dv = rnd(d), rnd(d)
    caches = []
    for _ in range(2):
        kv = KVCache() if kind == "f" else RotatingKVCache(max_size=WIN)
        if p:
            kv.update_and_fetch(pk, pv)
        for i in range(d - 1):
            kv.update_and_fetch(dk[:, :, i : i + 1], dv[:, :, i : i + 1])
        caches.append(kv)
    mx.eval([c.keys for c in caches] + [c.values for c in caches])
    return caches[0], caches[1], dk[:, :, d - 1 : d], dv[:, :, d - 1 : d]


def run_case(name, specs, window, seed, pbdt=mx.bfloat16, sdt=mx.bfloat16):
    B = len(specs)
    rs = np.random.RandomState(seed + 999)
    layer = SimpleNamespace(self_attn=SimpleNamespace(sliding_window=window, scaling=SCALE))
    rows = [build(sp, seed * 100 + i) for i, sp in enumerate(specs)]
    qs = [mx.array(rs.randn(1, H, 1, D).astype(np.float32)).astype(mx.bfloat16) for _ in range(B)]
    pbs = [mx.array((2 * rs.randn(1, 1, H, R_FULL)).astype(np.float32)).astype(pbdt) for _ in range(B)]
    sks = [mx.array(rs.randn(1, 1, H).astype(np.float32)).astype(sdt) for _ in range(B)]
    refs = [
        fastdecode._core(layer, qs[i], rows[i][2], rows[i][3], pbs[i], sks[i], rows[i][0]) for i in range(B)
    ]
    ks, vs, meta = [], [], []
    for i in range(B):
        k, v = rows[i][1].update_and_fetch(rows[i][2], rows[i][3])
        ks.append(k)
        vs.append(v)
        meta.append(row_meta(rows[i][1]))
    out = sdpa_rows(
        mx.concatenate(qs), ks, vs, mx.concatenate(pbs), mx.concatenate(sks), meta, scale=SCALE, window=window
    )
    mx.eval(refs + [out])
    bad = 0
    for i in range(B):
        a = np.array(refs[i].view(mx.uint16)).reshape(-1)
        b = np.array(out[i : i + 1].view(mx.uint16)).reshape(-1)
        assert a.shape == b.shape
        bad += int((a != b).sum())
    # layout contract: head stride of the returned arrays == cap*D
    for i in range(B):
        st = memoryview(ks[i].view(mx.uint16)).strides
        assert st[1] == meta[i][3] * D * 2 and st[2] == D * 2, (name, i, st, meta[i])
    desc = " ".join(f"S={k.shape[2]}{'r' + str(m[1]) if m[2] else ''}" for k, m in zip(ks, meta))
    print(f"{'PASS' if bad == 0 else 'FAIL'} {name:34s} B={B:2d} win={window:4d} [{desc}] mismatches={bad}", flush=True)
    return bad == 0


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    t0 = time.time()
    ok = []
    seed = 1
    for name, sp in FULL.items():
        if quick and "70000" in name:
            continue
        seed += 1
        ok.append(run_case(name, [sp], 0, seed))
    for name, sp in SLIDE.items():
        if quick and "70000" in name:
            continue
        seed += 1
        ok.append(run_case(name, [sp], WIN, seed))
    fl = [v for k, v in FULL.items() if "70000" not in k]
    sl = [v for k, v in SLIDE.items() if "70000" not in k]
    rng = np.random.RandomState(7)
    for B in (2, 3, 5, 8):
        for rep in range(3 if not quick else 1):
            seed += 1
            ok.append(run_case(f"mixed full rep{rep}", [fl[j] for j in rng.choice(len(fl), B)], 0, seed))
            seed += 1
            ok.append(run_case(f"mixed slide rep{rep}", [sl[j] for j in rng.choice(len(sl), B)], WIN, seed))
    # explicit mixed regimes in one launch
    seed += 1
    ok.append(run_case("mix: 300/wrapped1024/full5000", [FULL["full S=300"], FULL["full S=5000"], FULL["full S=1024"]], 0, seed))
    seed += 1
    ok.append(run_case("mix: 300/ring/ring+37", [SLIDE["slide S=300"], SLIDE["slide S=1024 full ring"], SLIDE["slide wrap +37"]], WIN, seed))
    seed += 1
    ok.append(run_case("mix full incl 70000", [FULL["full S=70000"], FULL["full S=1"], FULL["full S=20000"], FULL["full S=1023"]], 0, seed))
    # B > MAX_ROWS (chunked)
    seed += 1
    ok.append(run_case("B=13 slide chunked", [sl[j % len(sl)] for j in range(13)], WIN, seed))
    seed += 1
    ok.append(run_case("B=16 full chunked", [fl[(3 * j) % len(fl)] for j in range(16)], 0, seed))
    # window narrower than the cache (visibility cut inside the kernel) on full-attention caches
    seed += 1
    ok.append(run_case("window=700 on full caches", [FULL["full S=300"], FULL["full S=1500"], FULL["full S=5000"], FULL["full S=1023"]], 700, seed))
    # f32 position bias / sinks
    seed += 1
    ok.append(run_case("f32 pb+sinks full", [FULL["full S=1500"], FULL["full S=300"]], 0, seed, mx.float32, mx.float32))
    seed += 1
    ok.append(run_case("f32 pb+sinks slide", [SLIDE["slide wrap +300"], SLIDE["slide S=300"]], WIN, seed, mx.float32, mx.float32))
    print(f"SUMMARY {sum(ok)}/{len(ok)} cases PASS ({time.time() - t0:.0f}s)")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
