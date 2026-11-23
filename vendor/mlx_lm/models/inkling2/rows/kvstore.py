"""Batched KV store for B decode streams of ONE layer visit (lane F5: kernels K7 KV + K8 SDPA of docs/FUSED.md).

Today every stream owns a KVCache / RotatingKVCache per visit: 2 slice_updates per row per visit, then rows/sdpa.py binds
up to 11 per-row arrays. `RowKV` keeps the B rows' keys / values in ONE pair of flat arrays

    K, V : (B * Hkv * Cap, D) bf16      row b, kv head h, slot s  ->  flat row ((b * Hkv + h) * Cap + s)

with each row's EXACT physical slot layout (same slot numbers as that row's own cache buffer; slots >= its logical
capacity are zero), and per-row metadata (offset, ring index, logical capacity) kept on the host (they advance by +1 per
step, deterministically) and handed to the kernels as ARRAYS:

  plan = store.plan()                       # host: advance every row's cache state one token, upload 2 tiny arrays
  store.update(k, v, plan)                  # k, v (B, Hkv, D) bf16: 2 scatter launches (K and V), in place (donated)
  o = store.sdpa(q, pb, sinks, plan, scale=..., window=...)   # (B, 1, Hq, D), 1 launch (<1024 keys) / 2 launches

`update(k, v)` / `sdpa(...)` without a plan use the store's own (`update` plans, `sdpa` reuses the last plan). A plan is
a plain pair of mx arrays plus the host states it leads to, so ONE plan can be shared by every store whose rows have the
same cache states (all full-attention visits of a stream set share one, all sliding visits another; per token a model
makes 2 plans instead of 48): `update` of a store that is given a plan checks its own state equals the plan's `prev` and
advances it.

Exactness: the update emulates `update_and_fetch(S=1)` of each row's cache (KVCache: append at offset, logical capacity
`prev + step` when full; RotatingKVCache: grow by min(step, max_size - prev) until max_size, then ring write at `_idx`,
wrap to `keep`), the sdpa kernel is the arithmetic of rows/sdpa.py (itself bitwise equal to `fastdecode._core`) reading
the store: key length S = min(offset, logical capacity) (what `update_and_fetch` returns), ring rotation and masks from the
plan's per-row [S, end, idx, rot] int32 table.

Compile safety: nothing per-step is baked into a kernel source or template. Template parameters are B, Cap (a shape),
`has2` (Cap >= 1024: a 2-pass row is possible) and the threadgroup width (fixed at construction: narrow = GQA simdgroups only if every row starts in the 2-pass regime, else 32), all constants of
the store; the grid is derived from them only. Growth beyond Cap reallocates (changes shapes: retraces).

In-place update: `self.K[widx] = k` is MLX's scatter with a donatable input: the buffer is reused, traffic is
O(B * Hkv * D) (checked in bench/fused_kv_speed.py). The store arrays must therefore not be referenced elsewhere while
an update is pending (do not keep `store.K` views across `update`).

Memory: Cap is the maximum logical capacity over the rows, so a 70000-token row makes every row's slab that big
(B * Hkv * Cap * D * 2 bytes each for K and V). Use one store per group of rows of similar length if that matters.
"""
from types import SimpleNamespace

import numpy as np

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.inkling2.rows.sdpa import _HEADER, _STAGE2, _TWO_PASS_N, _blocks

_MW = 4  # per-row table columns: S, end, idx, rot
D = 128

_STAGE1 = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;  // threadgroups per kv head in the 2-pass stage 1
    constexpr int NTG2 = HKV * TGPK;
    constexpr int NG = HAS2 ? NTG2 : HQ;                // threadgroups reserved per row (static)
    threadgroup float outputs[1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = (int)t / NG;
    int tl = (int)t % NG;
    const int S = meta[b * MW + 0];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[b * MW + 1]; msk.idx = meta[b * MW + 2]; msk.rot = meta[b * MW + 3];
    msk.R = R; msk.window = WIN;
    const float sc = scale[0];
    if (S < 1024) {
        if (tl >= HQ) return;
        int h = tl;
        int kvh = h / GQA;
        int bh = b * HQ + h;
        size_t base = ((size_t)(b * HKV + kvh) * CAP) * D;
        msk.pb = pb + (size_t)bh * R;
        sdpa1_head<T, PBT, D>(
            q + (size_t)bh * D, kk + base, vv + base, o1 + (size_t)bh * D, S, sc,
            float(static_cast<T>(sinks[bh])), msk, outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
    } else {
        int kvh = tl / TGPK;
        int f = (tl % TGPK) * NS + (int)simd_gid;
        if (f < BLOCKS * GQA) {
            int block = f / GQA;
            int h = kvh * GQA + f % GQA;
            int bh = b * HQ + h;
            size_t base = ((size_t)(b * HKV + kvh) * CAP) * D;
            msk.pb = pb + (size_t)bh * R;
            sdpa2_block<T, PBT, D, BLOCKS>(
                q + (size_t)bh * D, kk + base, vv + base,
                partials + ((size_t)bh * BLOCKS + block) * D, sums + (size_t)bh * BLOCKS + block,
                maxs + (size_t)bh * BLOCKS + block, S, sc, float(static_cast<T>(sinks[bh])), msk, block, simd_lid);
        }
    }
"""

_kernels: dict = {}


def _kernel_pair():
    ks = _kernels.get(0)
    if ks is None:
        s1 = mx.fast.metal_kernel(
            name="kvstore_sdpa_s1", input_names=["q", "pb", "sinks", "meta", "scale", "kk", "vv"],
            output_names=["o1", "partials", "sums", "maxs"], source=_STAGE1, header=_HEADER,
            ensure_row_contiguous=False,
        )
        s2 = mx.fast.metal_kernel(
            name="rows_sdpa_s2", input_names=["partials", "sums", "maxs", "o1", "meta"], output_names=["out"],
            source=_STAGE2, header=_HEADER, ensure_row_contiguous=False,
        )
        ks = _kernels[0] = (s1, s2)
    return ks


def _advance_full(off, idx, cap, step):
    """KVCache.update_and_fetch(S=1) on host state -> (write slot, off', idx', cap')."""
    if off + 1 > cap:
        cap = off + step  # n_steps = 1: truncate to `off` if not a multiple of step, then append `step` zeros
    return off, off + 1, 0, cap


def _advance_ring(off, idx, cap, step, max_size, keep):
    """RotatingKVCache._update_in_place(S=1) on host state -> (write slot, off', idx', cap')."""
    if off >= cap and cap < max_size:
        cap += min(step, max_size - off)
        idx = off
    assert cap <= max_size
    if idx == max_size:
        idx = keep
    return idx, off + 1, idx + 1, cap


class RowKV:
    """B streams' keys/values of one layer visit; see the module docstring."""

    def __init__(self, caches):
        assert len(caches) >= 1
        kinds = {type(c) for c in caches}
        assert kinds <= {KVCache} or kinds <= {RotatingKVCache}, f"mixed or unsupported cache classes {kinds}"
        self.caches = list(caches)
        self.B = B = len(caches)
        self.ring = isinstance(caches[0], RotatingKVCache)
        self.step = caches[0].step
        assert all(c.step == self.step for c in caches)
        if self.ring:
            self.max_size = caches[0].max_size
            assert all(c.max_size == self.max_size and c.keep == 0 for c in caches), "ring rows need keep=0, equal max_size"
        ks, vs, st = [], [], []
        for c in caches:
            assert c.keys is not None and c.keys.dtype == mx.bfloat16 and c.values.dtype == mx.bfloat16
            kb, vb = c.keys, c.values
            assert kb.shape[0] == 1 and kb.shape[3] == D and vb.shape == kb.shape, (kb.shape, vb.shape)
            off = int(c.offset)
            if self.ring:
                idx = int(c._idx)
                if kb.shape[2] > self.max_size:
                    # exactly what the cache's own next update does first (keep=0): keep the last max_size slots
                    t = kb.shape[2] - self.max_size
                    kb, vb = kb[..., t:, :], vb[..., t:, :]
                    idx = self.max_size
            else:
                idx = 0
                assert off <= kb.shape[2]
            ks.append(kb)
            vs.append(vb)
            st.append((off, idx, kb.shape[2]))
        self.Hkv = ks[0].shape[1]
        assert all(k.shape[1] == self.Hkv for k in ks)
        self.cap = self.max_size if self.ring else max(s[2] for s in st)
        self.state = tuple(st)  # per row (offset, ring idx, logical capacity)
        self.K = self._pack(ks)
        self.V = self._pack(vs)
        mx.eval(self.K, self.V)
        # threadgroup width of stage 1: GQA simdgroups (MLX's 2-pass geometry) only if no row can ever be < 1024 keys
        self.narrow = min(min(o, c) for o, _, c in st) >= _TWO_PASS_N
        self._plan = None
        self._scale = {}

    # ---------------------------------------------------------------- layout
    def _pack(self, bufs):
        Hkv, cap = self.Hkv, self.cap
        rows = []
        for b in bufs:
            pad = cap - b.shape[2]
            if pad:
                b = mx.concatenate([b, mx.zeros((1, Hkv, pad, D), b.dtype)], axis=2)
            rows.append(b)
        a = rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=0)
        return a.reshape(len(bufs) * Hkv * cap, D)

    def _grow(self, new_cap):
        B, Hkv, cap = self.B, self.Hkv, self.cap
        z = mx.zeros((B, Hkv, new_cap - cap, D), mx.bfloat16)
        self.K = mx.concatenate([self.K.reshape(B, Hkv, cap, D), z], axis=2).reshape(-1, D)
        self.V = mx.concatenate([self.V.reshape(B, Hkv, cap, D), z], axis=2).reshape(-1, D)
        self.cap = new_cap
        mx.eval(self.K, self.V)

    # ---------------------------------------------------------------- per-step host metadata
    def plan(self):
        """Advance every row's cache state by one token; returns the plan (device arrays + the host states it
        leads to). Reallocates the store (rare) if a row outgrows Cap."""
        new, wpos, tbl = [], [], np.empty((self.B, _MW), np.int32)
        for i, (off, idx, cap) in enumerate(self.state):
            if self.ring:
                w, off2, idx2, cap2 = _advance_ring(off, idx, cap, self.step, self.max_size, 0)
            else:
                w, off2, idx2, cap2 = _advance_full(off, idx, cap, self.step)
            new.append((off2, idx2, cap2))
            wpos.append(w)
            tbl[i] = (min(off2, cap2), off2, idx2 if self.ring else 0, 1 if self.ring else 0)
        need = max(c for _, _, c in new)
        if need > self.cap:
            self._grow(need)
        base = (np.arange(self.B * self.Hkv, dtype=np.int64) * self.cap).reshape(self.B, self.Hkv)
        widx = (base + np.asarray(wpos, np.int64)[:, None]).reshape(-1).astype(np.int32)
        return SimpleNamespace(prev=self.state, new=tuple(new), cap=self.cap, mt=mx.array(tbl), widx=mx.array(widx),
                               lens=tuple(int(t) for t in tbl[:, 0]))

    def _apply(self, p):
        assert p.prev == self.state, "plan was made for a different row state"
        if p.cap > self.cap:
            self._grow(p.cap)
        assert p.cap == self.cap, "plan was made for a store with a different capacity"
        self.state = p.new
        self._plan = p

    # ---------------------------------------------------------------- K7: update
    def update(self, k, v, plan=None):
        """k, v: (B, Hkv, D) bf16 (the new token of every row). In place (donated scatter): 2 launches."""
        assert k.shape == (self.B, self.Hkv, D) and v.shape == k.shape
        assert k.dtype == mx.bfloat16 and v.dtype == mx.bfloat16
        if plan is None:
            plan = self.plan()
            self._apply_planned(plan)
        else:
            self._apply(plan)
        n = self.B * self.Hkv
        self.K[plan.widx] = k.reshape(n, D)
        self.V[plan.widx] = v.reshape(n, D)
        return plan

    def _apply_planned(self, p):  # plan() made for ourselves: states already match, only commit
        self.state = p.new
        self._plan = p

    # ---------------------------------------------------------------- K8: sdpa
    def sdpa(self, q, pb, sinks, plan=None, *, scale, window):
        """q (B, Hq, 1, D) bf16; pb (B, 1, Hq, R) or (B, Hq, R); sinks (B, 1, Hq) or (B, Hq). Returns (B, 1, Hq, D) bf16,
        row b bitwise == fastdecode._core on row b's own cache. `plan`: the plan used by this step's update
        (default: the store's last)."""
        plan = plan or self._plan
        assert plan is not None and plan.new == self.state, "call update() for this step first"
        B, Hkv, cap = self.B, self.Hkv, self.cap
        assert q.dtype == mx.bfloat16 and q.shape[0] == B and q.shape[2] == 1 and q.shape[3] == D
        Hq = q.shape[1]
        gqa = Hq // Hkv
        assert Hq == gqa * Hkv and 3 <= gqa <= 5, "block count rule implemented for n_simds in 3..5 (blocks=128)"
        blocks = _blocks(cap, gqa)
        assert blocks == 128
        R = pb.shape[-1]
        has2 = cap >= _TWO_PASS_N
        ns = gqa if (self.narrow and has2) else 32
        tgpk = (blocks * gqa + ns - 1) // ns
        ng = Hkv * tgpk if has2 else Hq
        s1, s2 = _kernel_pair()
        key = (float(scale),)
        sc = self._scale.get(key)
        if sc is None:
            sc = self._scale[key] = mx.array([scale], mx.float32)
        n2 = B * Hq * blocks if has2 else 1
        tmpl = [("T", mx.bfloat16), ("PBT", pb.dtype), ("MW", _MW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns), ("CAP", cap), ("HAS2", 1 if has2 else 0)]
        o1, partials, sums, maxs = s1(
            inputs=[mx.contiguous(q), mx.contiguous(pb), mx.contiguous(sinks), plan.mt, sc, self.K, self.V],
            template=tmpl, grid=(B * ng * 32 * ns, 1, 1), threadgroup=(32 * ns, 1, 1),
            output_shapes=[(B, 1, Hq, D), (n2 * D if has2 else 1,), (n2,), (n2,)],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
        )
        if not has2:
            return o1
        (out,) = s2(
            inputs=[partials, sums, maxs, o1, plan.mt],
            template=[("T", mx.bfloat16), ("MW", _MW), ("HQ", Hq), ("BLOCKS", blocks)],
            grid=(B * Hq * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(B, 1, Hq, D)],
            output_dtypes=[mx.bfloat16],
        )
        return out

    # ---------------------------------------------------------------- write back
    def flush(self, i):
        """Write row i's state into its own cache object, as if that cache had run alone."""
        c = self.caches[i]
        off, idx, cap = self.state[i]
        K4 = self.K.reshape(self.B, self.Hkv, self.cap, D)
        V4 = self.V.reshape(self.B, self.Hkv, self.cap, D)
        c.keys = mx.contiguous(K4[i : i + 1, :, :cap, :])
        c.values = mx.contiguous(V4[i : i + 1, :, :cap, :])
        c.offset = off
        if self.ring:
            c._idx = idx
        mx.eval(c.keys, c.values)
        return c

    def to_caches(self):
        return [self.flush(i) for i in range(self.B)]
