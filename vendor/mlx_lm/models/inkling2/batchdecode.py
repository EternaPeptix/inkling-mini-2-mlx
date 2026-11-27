"""Batched decode: C independent streams advance one token in ONE forward, and every row is bitwise equal to the B=1
fast decode path (fastdecode.py) of that stream. Exact by construction per op; checked end to end by
bench/batch_dec.py (tokens of every stream == its single-stream greedy tokens).

Which formulation keeps a C-row op bitwise equal to C separate B=1 calls (bench/rows_ops.py, bench/rows_f32.py):
  * elementwise, norms, l2 clips, depth attention, relative logits, router top-k kernel, experts (gather_qmv per row),
    dynamic short conv, softcap, argmax: plain batching is already row invariant.
  * 8-bit QuantizedLinear (attention projections, dynamic-conv weight_proj, lm_head): plain M=C dispatch changes the
    kernel (qmv -> qmv_wide/qmm). Batch the WEIGHTS instead (stride-0 broadcast, B=C, M=1 each): same qmv kernel per row.
  * f32 Linear (router, message-gate forget/message projections): every zero-copy form (stride-0 broadcast, gather_mm,
    addmm, einsum) collapses to an M=C GEMM. Only a materialized (C, N, K) copy gives a batched M=1 gemv per row.
    The copies are made once per (weight, C) and captured by the compiled segments (weights are frozen: call
    `reset()` after changing model weights).
  * attention core (KV cache update + sdpa): per stream (each stream has its own cache length).
  * log-softmax before argmax: per row, with the same all-axes reduction as generate_step.

Usage:
    dec = BatchDecoder(model, caches)      # caches[i] = prompt cache of stream i (prefilled, as generate_step leaves it)
    toks = dec.step(last_tokens)           # (C,) int -> (C,) next greedy tokens; caches advance in place
    dec.flush()                            # write the batched conv states back into the per-stream caches
"""
import inspect
import numpy as np
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import kernels as IK
from mlx_lm.models.inkling2 import layers as L
from mlx_lm.models.inkling2 import loopcd
from mlx_lm.models.inkling2.rows import gemv as RG
from mlx_lm.models.inkling2.rows import gather as RGa
from mlx_lm.models.inkling2.rows import head as RH
from mlx_lm.models.inkling2.rows import mid as RM
from mlx_lm.models.inkling2.rows import moe as RMo
from mlx_lm.models.inkling2.rows import proj as RP
from mlx_lm.models.inkling2.rows import qmv as RQ
from mlx_lm.models.inkling2.rows import rowlocal as RL
from mlx_lm.models.inkling2.rows import sdpa as RS
from mlx_lm.models.inkling2.rows import kvstore as RKV
from mlx_lm.models.inkling2.rows import visit as RV

_ROWS = False
# exact row kernels (lane E1/E3/E4): weights read once for all rows; each row bitwise = the B=1 op. Default all on.
_USE = set(filter(None, os.environ.get("LOOPLING_BD_ROWS", "qmv,gemv,sdpa,gather").split(",")))
# timing-only toggles (NOT exact): plain M=C dispatch instead of the invariant formulation
# fused visit stages (rows/visit.py, docs/FUSED.md): "pre" = K4 qkv multi-qmv + K5 mid + K6 rel in place of make_s1's ops
_FUSED = set(filter(None, os.environ.get("LOOPLING_BD_FUSED", "pre,k1,post,head").split(",")))
_PLAIN = set(filter(None, os.environ.get("LOOPLING_BD_PLAIN", "").split(",")))  # ql,lin,router,head
_mat: dict = {}
_tree_tbl_cache: dict = {}
_installed = False


def build_tree_tables(parents, depths):
    """Build cached `(depths_tuple, conv_anc (W,3) i32, tree_meta (W,9) i32)` for a depth-sorted tree."""
    p_t = tuple(int(p) for p in parents)
    d_t = tuple(int(d) for d in depths)
    key = (p_t, d_t)
    res = _tree_tbl_cache.get(key)
    if res is not None:
        return res
    W = len(p_t)
    assert 2 <= W <= 16 and p_t[0] == -1 and d_t[0] == 0
    assert max(d_t) < 16
    tm_w = 9 if W <= 8 else 17
    conv_rows = []
    meta_rows = []
    paths = []
    for b in range(W):
        pb = p_t[b]
        db = d_t[b]
        if b == 0:
            path = [0]
        else:
            assert 0 <= pb < b and d_t[b - 1] <= db and db == d_t[pb] + 1
            path = paths[pb] + [b]
        paths.append(path)
        a0 = path[db - 3] if db >= 3 else -(db + 1)
        a1 = path[db - 2] if db >= 2 else -(db + 2)
        a2 = pb if db >= 1 else -3
        conv_rows.append([a0, a1, a2])
        meta_rows.append([db] + path + [0] * ((tm_w - 1) - len(path)))
    if len(_tree_tbl_cache) >= 256:
        _tree_tbl_cache.clear()
    res = _tree_tbl_cache[key] = (d_t, mx.array(conv_rows, mx.int32), mx.array(meta_rows, mx.int32))
    return res


def _rows(x):
    return _ROWS and x.ndim == 3 and x.shape[0] > 1 and x.shape[1] == 1


def _matw(w, C):
    """Materialized (C, N, K) copy of a float weight (N, K), cached per (weight, C)."""
    k = (id(w), C)
    m = _mat.get(k)
    if m is None or m[0] is not w:
        c = mx.contiguous(mx.broadcast_to(w[None], (C,) + w.shape))
        mx.eval(c)
        m = _mat[k] = (w, c)
    return m[1]


def install():
    global _installed
    if _installed:
        return
    _installed = True
    ql_call, lin_call, router_call = nn.QuantizedLinear.__call__, nn.Linear.__call__, L.TopKRouter.__call__
    experts_call = L.Experts.__call__

    def _gq(m, x, idx):
        return RGa.gather_rows(x, m["weight"], m["scales"], None, idx, mode=m.mode, group_size=m.group_size,
                               bits=m.bits, global_scale=m.get("global_scale"))

    def experts(self, x, indices):
        gu_m, dn_m = self.gate_up_proj, self.down_proj
        ok = ("gather" in _USE and _rows(x) and x.dtype == mx.bfloat16 and gu_m.mode in ("nvfp4", "mxfp8", "mxfp4")
              and "biases" not in gu_m and "biases" not in dn_m and x.shape[0] * indices.shape[-1] <= 256)
        if not ok:
            return experts_call(self, x, indices)
        C, D = x.shape[0], x.shape[-1]
        idx = mx.stop_gradient(indices).reshape(C, -1).astype(mx.uint32)
        gu = _gq(gu_m, x.reshape(C, D), idx)  # (C, E, 2h)
        h = self.act_fn(gu[..., 0::2], gu[..., 1::2])
        y = _gq(dn_m, h, idx)  # (C, E, D)
        return y.reshape(C, 1, idx.shape[1], D)

    def qlinear(self, x):
        if not _rows(x) or "ql" in _PLAIN:
            return ql_call(self, x)
        C, K = x.shape[0], x.shape[-1]
        if ("qmv" in _USE and self.mode == "affine" and self.bits == 8 and self.group_size == 64
                and x.dtype == mx.bfloat16 and C <= RQ.MAX_B and K % 256 == 0 and K not in (64, 128)
                and "bias" not in self):
            return RQ.qmv_rows(x.reshape(C, K), self["weight"], self["scales"], self["biases"]).reshape(C, 1, -1)
        bc = lambda a: None if a is None else mx.broadcast_to(a[None], (C,) + a.shape)
        y = mx.quantized_matmul(x, bc(self["weight"]), scales=bc(self["scales"]), biases=bc(self.get("biases")),
                                transpose=True, group_size=self.group_size, bits=self.bits, mode=self.mode)
        if "bias" in self:
            y = y + self["bias"]
        return y

    def linear(self, x):
        if not _rows(x) or "lin" in _PLAIN:
            return lin_call(self, x)
        w = self["weight"]
        if "gemv" in _USE and x.dtype == w.dtype and w.dtype in (mx.float32, mx.bfloat16) and "bias" not in self:
            C, K = x.shape[0], x.shape[-1]
            return RG.gemv_rows(x.reshape(C, K), w).reshape(C, 1, -1)
        y = x @ _matw(w, x.shape[0]).transpose(0, 2, 1)
        if "bias" in self:
            y = y + self["bias"]
        return y

    def router(self, h):
        if not _rows(h) or "router" in _PLAIN:
            return router_call(self, h)
        from mlx_lm.kernels.expert_routing import expert_routing

        assert L._use_kernels()
        if "gemv" in _USE:
            C, K = h.shape[0], h.shape[-1]
            logits = RG.gemv_rows(h.astype(mx.float32).reshape(C, K), self.weight).reshape(C, 1, -1)
        else:
            logits = h.astype(mx.float32) @ _matw(self.weight, h.shape[0]).transpose(0, 2, 1)
        return expert_routing(logits, self.e_score_correction_bias, top_k=self.top_k, multiplier=1.0,
                              score_eps=L._ROUTER_SCORE_EPS, route_scale=self.routed_scaling_factor)

    nn.QuantizedLinear.__call__ = qlinear
    nn.Linear.__call__ = linear
    L.TopKRouter.__call__ = router
    L.Experts.__call__ = experts


class _RowsMode:
    def __enter__(self):
        global _ROWS
        self.prev, _ROWS = _ROWS, True

    def __exit__(self, *a):
        global _ROWS
        _ROWS = self.prev


def _wrap(layer, body):
    def run(*a):
        with _RowsMode():
            return body(layer, *a)

    return mx.compile(run) if FD._COMPILE else run


def _b1_cache_ok(lc) -> bool:
    kv, conv = lc[0], lc[1]
    return (
        type(kv) in (KVCache, RotatingKVCache)
        and getattr(kv, "keep", 0) == 0
        and getattr(kv, "_fd_delta", None) is None
        and (
            (
                kv._packed is not None
                and kv._packed.dtype == mx.bfloat16
                and kv._packed.shape[1] == 8
                and kv._packed.shape[3] == 128
            )
            or (
                kv._keys is not None
                and kv._keys.dtype == mx.bfloat16
                and kv._keys.shape[0] == 1
                and kv._keys.shape[1] == 8
                and kv._keys.shape[3] == 128
            )
        )
        and all(x is not None and x.dtype == mx.float32 for x in conv.cache)
    )


def _std_sampler_params(sampler):
    """Return `(temp, top_p, min_p)` if `sampler` matches standard `make_sampler` with GPU fast path, else None."""
    key = getattr(sampler, "_sampler_key", None)
    if (
        isinstance(key, tuple)
        and len(key) == 8
        and key[0] > 0.0
        and key[3] <= 1
        and key[4] == 0
        and key[5] == 0.0
    ):
        return float(key[0]), float(key[1]), float(key[2])
    return None


_cw_darr_cache: dict = {}
_cw_tau_lut_cache: dict = {}
_cw_tau_table_cache: dict = {}
_cw_ar_cache: dict = {}
_multi_conv_anc_cache: dict = {}
_batch_st_infos_cache: dict = {}
_update_kv_stage_k: dict = {}
_stage_commit_multi_k: dict = {}
_gather_stack_kv_k: dict = {}


def build_multi_chain_conv_anc(C: int, W: int):
    """Build cached `(C*W, 3) i32` causal conv ancestor table for C independent chains of width W."""
    key = (int(C), int(W))
    arr = _multi_conv_anc_cache.get(key)
    if arr is None:
        rows = []
        for c in range(C):
            base_r = c * W
            base_st = c * 3
            for w in range(W):
                a0 = (base_r + w - 3) if w >= 3 else -(base_st + w + 1)
                a1 = (base_r + w - 2) if w >= 2 else -(base_st + w + 2)
                a2 = (base_r + w - 1) if w >= 1 else -(base_st + 3)
                rows.append([a0, a1, a2])
        if len(_multi_conv_anc_cache) >= 64:
            _multi_conv_anc_cache.clear()
        arr = _multi_conv_anc_cache[key] = mx.array(rows, mx.int32)
    return arr
KV_STAGE_CAP = 32
_BATCH_KV_STAGE_CAP = int(os.environ.get("LOOPLING_KV_STAGE_CAP", "128"))
_KV_STACK_MAX_CAP = int(os.environ.get("LOOPLING_KV_STACK_MAX_CAP", "131072"))


def _kv_stack_max_cap() -> int:
    return int(os.environ.get("LOOPLING_KV_STACK_MAX_CAP", str(_KV_STACK_MAX_CAP)))


def _multi_long_min_s() -> int:
    return int(os.environ.get("LOOPLING_MULTI_LONG_MIN_S", "32768"))


def _cascade_min_tokens() -> int:
    return int(os.environ.get("LOOPLING_CASCADE_MIN_TOKENS", "32768"))


def _fast_sample_enabled() -> bool:
    return os.environ.get("LOOPLING_FAST_SAMPLE", "1") not in ("0", "", "false", "False")


def _batch_full_attn_long_or_cascade(
    q, kv_new_ml, pb, qkv_flat, gate_off, gate_stride, sink_off, sink_stride,
    stage, v, C, W, scale, *, ks_v=None, vs_v=None, stage_buf=None, tree_meta=None,
):
    """Route C=2..8 full-attention (W=1..8, C*W<=32) to Cascade or Multi-Stream Split-K when eligible."""
    if not isinstance(stage, _BatchKVStage) or not (2 <= C <= 8) or C * W > 32:
        return None
    st_buf = stage.buf if stage_buf is None else stage_buf
    sig = stage._sigs[v]
    st_lens = stage.st_lens
    c_groups = stage._cascade_groups
    valid_groups = [g for g in c_groups if 2 <= len(g[2]) <= 8 and len(g[2]) * W <= 32]
    if valid_groups:
        # Fast path: all C streams share one cascade prefix (zero row slicing).
        if len(valid_groups) == 1 and len(valid_groups[0][2]) == C:
            _, L_sh, _ = valid_groups[0]
            k_sh, v_sh = stage._casc_shared[v][0]
            ks_pr = stage._casc_k_priv[v]
            vs_pr = stage._casc_v_priv[v]
            S0s = [sig[b][0] + st_lens[b] for b in range(C)]
            pkcaps = [int(ks_pr[b].shape[2]) for b in range(C)]
            pvcaps = [int(vs_pr[b].shape[2]) for b in range(C)]
            slot_m_pr = [sig[b][0] - L_sh for b in range(C)]
            return RS.sdpa_cascade_long_full(
                q, k_sh, v_sh, L_sh, ks_pr, vs_pr, st_buf, kv_new_ml, pb, qkv_flat,
                gate_off, gate_stride, sink_off, sink_stride,
                S0s, pkcaps, pvcaps, slot_m_pr, st_lens, C, W,
                scale=scale, kv_layer_idx=v, scap=stage.scap, tree_meta=tree_meta,
            )
        # Mixed batch: run each cascade group via sdpa_cascade_long_full and remaining streams via multi/single long-full.
        B = C * W
        Hq_dim = int(q.shape[-1]) * int(q.shape[-2]) if q.ndim == 3 else int(q.shape[-1])
        Hq_n = Hq_dim // 128
        q_r = q.reshape(C, W, -1)
        pb_r = pb.reshape(C, W, pb.shape[-1]) if pb.ndim == 2 else pb.reshape(C, W, pb.shape[1], pb.shape[2])
        # Extract per-row gate (B, Hq*128) and sinks (B, Hq) from qkv_flat regardless of grouped/flat layout.
        qkv_1d = qkv_flat.reshape(-1)
        gate_rows = mx.stack(
            [qkv_1d[gate_off + r * gate_stride : gate_off + r * gate_stride + Hq_dim] for r in range(B)], axis=0
        ).reshape(C, W, Hq_dim)
        sink_rows = mx.stack(
            [qkv_1d[sink_off + r * sink_stride : sink_off + r * sink_stride + Hq_n] for r in range(B)], axis=0
        ).reshape(C, W, Hq_n)
        kvn_r = kv_new_ml.reshape(2, kv_new_ml.shape[1], C, W, 128)
        parts_by_stream = [None] * C
        covered = set()
        for g_idx, (_, L_sh, grp) in enumerate(valid_groups):
            cg = len(grp)
            g_arr = mx.array(grp, mx.int32)
            k_sh, v_sh = stage._casc_shared[v][g_idx]
            ks_pr = [stage._casc_k_priv[v][b] for b in grp]
            vs_pr = [stage._casc_v_priv[v][b] for b in grp]
            S0s = [sig[b][0] + st_lens[b] for b in grp]
            pkcaps = [int(k.shape[2]) for k in ks_pr]
            pvcaps = [int(x.shape[2]) for x in vs_pr]
            slot_m_pr = [sig[b][0] - L_sh for b in grp]
            st_l_g = [st_lens[b] for b in grp]
            st_g = mx.contiguous(st_buf[v : v + 1, g_arr])
            q_g = mx.contiguous(q_r[g_arr].reshape(cg * W, -1))
            pb_g = mx.contiguous(pb_r[g_arr].reshape(cg * W, *pb_r.shape[2:]))
            qkv_g = mx.concatenate([gate_rows[g_arr].reshape(-1), sink_rows[g_arr].reshape(-1)], axis=0)
            go_g, so_g = 0, cg * W * Hq_dim
            kvn_g = mx.contiguous(kvn_r[:, :, g_arr, :, :].reshape(2, kv_new_ml.shape[1], cg * W, 128))
            out_g = RS.sdpa_cascade_long_full(
                q_g, k_sh, v_sh, L_sh, ks_pr, vs_pr, st_g, kvn_g, pb_g, qkv_g,
                go_g, Hq_dim, so_g, Hq_n,
                S0s, pkcaps, pvcaps, slot_m_pr, st_l_g, cg, W,
                scale=scale, kv_layer_idx=0, scap=stage.scap, tree_meta=tree_meta,
            ).reshape(cg, W, -1)
            for j, b in enumerate(grp):
                parts_by_stream[b] = out_g[j]
                covered.add(b)
        rem = [b for b in range(C) if b not in covered]
        ks_all = stage.ks_main[v] if ks_v is None else ks_v
        vs_all = stage.vs_main[v] if vs_v is None else vs_v
        if rem:
            cr = len(rem)
            r_arr = mx.array(rem, mx.int32)
            S0s_r = [sig[b][0] + st_lens[b] for b in rem]
            st_r = mx.contiguous(st_buf[v : v + 1, r_arr])
            q_m = mx.contiguous(q_r[r_arr].reshape(cr * W, -1))
            pb_m = mx.contiguous(pb_r[r_arr].reshape(cr * W, *pb_r.shape[2:]))
            qkv_m = mx.concatenate([gate_rows[r_arr].reshape(-1), sink_rows[r_arr].reshape(-1)], axis=0)
            go_m, so_m = 0, cr * W * Hq_dim
            kvn_m = mx.contiguous(kvn_r[:, :, r_arr, :, :].reshape(2, kv_new_ml.shape[1], cr * W, 128))
            if W == 1 and max(S0s_r) + 1 < _multi_long_min_s():
                # Short non-cascade streams (<32K) use sdpa_rows_staged for bitwise parity with generate_step.
                kvn_b = mx.contiguous(kvn_m.reshape(2, kv_new_ml.shape[1], cr, 128).transpose(2, 0, 1, 3))
                mt_r = np.empty((cr, RS._SMW), np.int32)
                ss_r = []
                for j, b in enumerate(rem):
                    end_m, _, rot_i, kcap, vcap, slot_m, rot_w, max_s = sig[b]
                    end_now = end_m + st_lens[b] + 1
                    mt_r[j] = (end_now, end_now, 0, 0, kcap, vcap, slot_m, 0, 0)
                    ss_r.append(end_now)
                out_r = RS.sdpa_rows_staged(
                    q_m.reshape(cr, -1, 128), [ks_all[b] for b in rem], [vs_all[b] for b in rem],
                    st_r, kvn_b, pb_m, qkv_m, go_m, Hq_dim, so_m, Hq_n, None,
                    scale=scale, window=0, kv_layer_idx=0, st_len=st_lens[rem[0]], scap=stage.scap,
                    meta_arr=mx.array(mt_r), Ss=tuple(ss_r),
                ).reshape(cr, 1, -1)
                for j, b in enumerate(rem):
                    parts_by_stream[b] = out_r[j]
            elif cr >= 2:
                kcaps = [sig[b][3] for b in rem]
                vcaps = [sig[b][4] for b in rem]
                slot_ms = [sig[b][5] for b in rem]
                st_l_r = [st_lens[b] for b in rem]
                out_r = RS.sdpa_multi_long_full_staged(
                    q_m, [ks_all[b] for b in rem], [vs_all[b] for b in rem], st_r, kvn_m, pb_m, qkv_m,
                    go_m, Hq_dim, so_m, Hq_n,
                    S0s_r, kcaps, vcaps, slot_ms, st_l_r, cr, W,
                    scale=scale, kv_layer_idx=0, scap=stage.scap, tree_meta=tree_meta,
                ).reshape(cr, W, -1)
                for j, b in enumerate(rem):
                    parts_by_stream[b] = out_r[j]
            else:
                b = rem[0]
                s_pref = sig[b][0] + st_lens[b]
                st_b = mx.contiguous(st_buf[v, b])
                parts_by_stream[b] = RS.sdpa_long_full_staged_unpacked(
                    q_m, ks_all[b], vs_all[b], kvn_m, pb_m, qkv_m, go_m, so_m,
                    s_pref, s_pref + W, sig[b][3], sig[b][4], W,
                    scale=scale, kv_layer_idx=0, tree_meta=tree_meta,
                    kv_stage=st_b, slot_main=sig[b][5], st_len=st_lens[b],
                    gate_stride=Hq_dim, sink_stride=Hq_n,
                ).reshape(W, -1)
        return mx.concatenate(parts_by_stream, axis=0).reshape(B, -1)

    S0s = [sig[b][0] + st_lens[b] for b in range(C)]
    if max(S0s) + W < _multi_long_min_s():
        return None
    ks_use = stage.ks_main[v] if ks_v is None else ks_v
    vs_use = stage.vs_main[v] if vs_v is None else vs_v
    kcaps = [sig[b][3] for b in range(C)]
    vcaps = [sig[b][4] for b in range(C)]
    slot_mains = [sig[b][5] for b in range(C)]
    return RS.sdpa_multi_long_full_staged(
        q, ks_use, vs_use, st_buf, kv_new_ml, pb, qkv_flat,
        gate_off, gate_stride, sink_off, sink_stride,
        S0s, kcaps, vcaps, slot_mains, st_lens, C, W,
        scale=scale, kv_layer_idx=v, scap=stage.scap, tree_meta=tree_meta,
    )


def _batch_cache_ok(caches, nv) -> bool:
    for v in range(nv):
        k0 = caches[0][v][0]
        is_rot = type(k0) is RotatingKVCache
        if type(k0) not in (KVCache, RotatingKVCache) or getattr(k0, "keep", 0) != 0:
            return False
        for c in caches:
            kv, conv = c[v][0], c[v][1]
            if (
                type(kv) is not type(k0)
                or getattr(kv, "keep", 0) != 0
                or getattr(kv, "_fd_delta", None) is not None
                or (kv._packed is None and kv._keys is None)
                or (kv._packed is not None and (kv._packed.dtype != mx.bfloat16 or kv._packed.shape[1] != 8 or kv._packed.shape[3] != 128))
                or (kv._keys is not None and (kv._keys.dtype != mx.bfloat16 or kv._keys.shape[0] != 1 or kv._keys.shape[1] != 8 or kv._keys.shape[3] != 128))
                or (is_rot and kv.max_size != k0.max_size)
                or not all(x is not None and x.dtype == mx.float32 for x in conv.cache)
            ):
                return False
    return True


class _ScapOnly:
    __slots__ = ("scap",)

    def __init__(self, scap: int):
        self.scap = scap


def _tm_weight_sig(tm):
    return tuple(id(v) for _, v in tree_flatten(tm.parameters()))


_COMPILE_ENV_KEYS = (
    "MOE_AUTO_ROUTE", "MOE_TILE_MIN_B", "MOE_TILE_TMAX", "MOE_TILE_NSG", "MOE_TILE_ROWS",
    "MOE_TILE_SORTED_MIN_B",
    "MOE_ROWS_GU", "MOE_ROWS_DN", "MOE_NSG_GU", "MOE_NSG_DN", "MOE_BALLOT_SCAN",
    "MOE_ROUTER_FUSE", "MOE_ROUTER_TM1", "MOE_FP8_LUT", "MOE_TMAX_CAP",
    "LOOPLING_SDPA2_STAGED_NS", "LOOPLING_SDPA_MID_BATCH_OPASS", "LOOPLING_SDPA_MID_BATCH",
    "LOOPLING_FUSE_K3B_QKV", "LOOPLING_FUSE_K3B_QKV_MAX_B",
    "LOOPLING_FUSE_K3B_MLP", "LOOPLING_FUSE_K3B_MLP_MAX_B",
)


def _env_compile_sig():
    return tuple(os.environ.get(k) for k in _COMPILE_ENV_KEYS)


def _layer_batch_body(
    batch_meta,
    layer,
    v,
    h,
    bs,
    msg,
    stage,
    step_metas,
    step_Ss,
    full_offs,
    st_len,
    kv_news,
    tau_cache,
    conv_list,
    prev_moe=None,
    skip_combine=False,
    tau_override=None,
    st_info_arr=None,
    stage_buf=None,
    ks_v=None,
    vs_v=None,
):
    a, sliding, op_m, rel_proj, scale, window = batch_meta[id(layer)]
    cs = conv_list[v]
    if prev_moe is not None:
        r1 = RL.k1_from_down_batch_layer(layer, prev_moe, bs, msg, emit_h=True)
        h = r1["h"]
        conv_list[v - 1][3] = r1["w3"]
    else:
        r1 = RL.k1_pre_layer(layer, h, bs, msg)
    nr = r1["normed_residual"]
    if msg is None:
        qkv_flat, qkv_lay = RP.qkv_rows(a, r1["q_in"], r1["k_in"], r1["v_in"], flat=True)
    elif (
        os.environ.get("LOOPLING_FUSE_K3B_QKV", "1") not in ("0", "")
        and int(r1["s_q"].shape[0]) <= int(os.environ.get("LOOPLING_FUSE_K3B_QKV_MAX_B", "4"))
    ):
        qkv_flat, qkv_lay = RP.qkv_msg_rows(layer, r1, msg, flat=True)
    else:
        gs = (layer["q_message_gate"], layer["k_message_gate"], layer["v_message_gate"])
        mixed = RV._gates(gs, (r1["s_q"], r1["s_k"], r1["s_v"]), msg, (0, 1, 2), r1["qkv_gate_in"])
        q_in, k_in, v_in = RL.k3b_norm_layer(layer, mixed)
        qkv_flat, qkv_lay = RP.qkv_rows(a, q_in, k_in, v_in, flat=True)
    if qkv_lay[2][0] == "gate" and qkv_lay[3][0] == "sinks":
        gate_off, sink_off = qkv_lay[2][1], qkv_lay[3][1]
        gate_stride, sink_stride = qkv_lay[2][2], qkv_lay[3][2]
    else:
        dm = {item[0]: (item[1], item[2]) for item in qkv_lay}
        gate_off, gate_stride = dm["gate"]
        sink_off, sink_stride = dm["sinks"]
    tau_1d = None
    if not sliding:
        if tau_override is not None:
            tau_1d = tau_override
        else:
            oarr, max_p = full_offs if full_offs is not None else stage.full_off_for_visit(v, st_len)
            tkey = (id(stage), -1 if full_offs is not None else v, a.log_scaling_alpha, a.log_scaling_n_floor)
            tau_1d = tau_cache.get(tkey)
            if tau_1d is None:
                tau_1d = _get_tau_table(layer, max_p)[oarr]
                tau_cache[tkey] = tau_1d
    rel_w_bf = FD._rel_w_bf(rel_proj)
    if (
        os.environ.get("LOOPLING_SDPA_MID_BATCH", "1") not in ("0", "")
        and max(step_Ss[v]) < RS._TWO_PASS_N
    ):
        attn_out, kv_new, w0, w1 = RS.sdpa_mid_rows_staged(
            layer, qkv_flat, qkv_lay,
            stage.ks_main[v] if ks_v is None else ks_v,
            stage.vs_main[v] if vs_v is None else vs_v,
            stage.buf if stage_buf is None else stage_buf,
            cs[0], cs[1], rel_w_bf,
            scale=scale, window=window, kv_layer_idx=v, st_len=st_len, scap=stage.scap,
            meta_arr=step_metas[v], Ss=step_Ss[v], st_info_arr=st_info_arr, tau=tau_1d,
        )
        kv_news[v] = kv_new
    else:
        q, kv_new, pb, w0, w1 = RM.mid_rows_flat(
            layer, qkv_flat, qkv_lay, cs[0], cs[1], tau=tau_1d, rel_w=rel_w_bf, packed_kv_b=True
        )
        kv_news[v] = kv_new
        attn_out = None
        if not sliding and isinstance(stage, _BatchKVStage):
            kv_new_ml = mx.contiguous(kv_new.transpose(1, 2, 0, 3))
            attn_out = _batch_full_attn_long_or_cascade(
                q, kv_new_ml, pb, qkv_flat, gate_off, gate_stride, sink_off, sink_stride,
                stage, v, stage.C, 1, scale, ks_v=ks_v, vs_v=vs_v, stage_buf=stage_buf,
            )
        if attn_out is None:
            sdpa_kw = dict(
                scale=scale, window=window, kv_layer_idx=v, st_len=st_len, scap=stage.scap,
                meta_arr=step_metas[v], Ss=step_Ss[v],
            )
            if st_info_arr is not None:
                sdpa_kw["st_info_arr"] = st_info_arr
            attn_out = RS.sdpa_rows_staged(
                q,
                stage.ks_main[v] if ks_v is None else ks_v,
                stage.vs_main[v] if vs_v is None else vs_v,
                stage.buf if stage_buf is None else stage_buf,
                kv_new, pb, qkv_flat,
                gate_off, gate_stride, sink_off, sink_stride, None,
                **sdpa_kw,
            )
    op = RQ.qmv_rows(attn_out, op_m["weight"], op_m["scales"], op_m["biases"])
    r10 = RL.k10_post_layer(layer, op, h, nr, bs, cs[2], msg, debug=False)
    if layer.is_block_boundary:
        bs = [nr] if bs is None else (bs + [nr])
    if msg is None:
        x_mlp, mlp_mixed = r10["mlp_x"], None
    elif (
        os.environ.get("LOOPLING_FUSE_K3B_MLP", "1") not in ("0", "")
        and int(r10["mlp_src"].shape[0]) <= int(os.environ.get("LOOPLING_FUSE_K3B_MLP_MAX_B", "2"))
    ):
        x_mlp, mlp_mixed = None, RP.mlp_msg_part(layer, r10, msg)
    else:
        mixed = RV._gates((layer["mlp_message_gate"],), (r10["mlp_src"],), msg, (3,), (r10["mlp_gate_in"],))
        (x_mlp,) = RL.k3b_norm_layer(layer, mixed, mlp=True)
        mlp_mixed = None
    e = RMo.moe_rows(layer, x_mlp, r10["residual"], cs[3], route=RV.ROUTE, skip_combine=skip_combine, mlp_mixed=mlp_mixed)
    if skip_combine and "down" in e:
        conv_list[v] = [w0, w1, r10["w2"], None]
        return None, bs, e
    conv_list[v] = [w0, w1, r10["w2"], e["w3"]]
    return e["out"], bs, None


class _BatchKVStage:
    """Ping-pong KV staging buffers over unpacked per-stream KVCache/RotatingKVCache objects."""

    def __init__(self, caches, scap=None, pool=None):
        self.caches = caches
        self.C = len(caches)
        self.nv = len(caches[0])
        self.scap = int(_BATCH_KV_STAGE_CAP if scap is None else scap)
        self._pool = pool
        self.st_len = 0
        self._meta_cache = {}
        self._off_arr_cache = {}
        to_eval = []
        for v in range(self.nv):
            for b in range(self.C):
                kv = caches[b][v][0]
                if kv._packed is not None:
                    kv.unpack_b1()
                if kv._keys is None:
                    continue
                is_rot = hasattr(kv, "_idx")
                if is_rot and kv._keys.shape[2] > kv.max_size:
                    t = kv._keys.shape[2] - kv.max_size
                    kv._keys = kv._trim(t, kv._keys)
                    kv._values = kv._trim(t, kv._values)
                    kv._idx = kv.max_size
                    to_eval.append(kv._keys)
                    to_eval.append(kv._values)
                cap = int(kv._keys.shape[2])
                step = int(getattr(kv, "step", 256))
                target = ((max(cap, 1) + step - 1) // step) * step
                if is_rot:
                    target = min(target, int(kv.max_size))
                # Never copy/pad large full-attn caches or cascade-pinned caches (kernels take runtime kcap/vcap).
                if cap < target and (is_rot or (cap < 4096 and getattr(kv, "_cascade_ref", None) is None)):
                    add = target - cap
                    kv._keys = mx.concatenate(
                        [kv._keys, mx.zeros((1, kv._keys.shape[1], add, kv._keys.shape[3]), kv._keys.dtype)],
                        axis=2,
                    )
                    kv._values = mx.concatenate(
                        [kv._values, mx.zeros((1, kv._values.shape[1], add, kv._values.shape[3]), kv._values.dtype)],
                        axis=2,
                    )
                    to_eval.append(kv._keys)
                    to_eval.append(kv._values)
        if to_eval:
            mx.eval(to_eval)
        Hkv = caches[0][0][0]._keys.shape[1]
        shp = (self.nv, self.C, 2, Hkv, self.scap, 128)
        self._shp = shp
        reused = pool.pop(shp, None) if pool is not None else None
        if reused is not None:
            self._bufs = reused
        else:
            self._bufs = [mx.zeros(shp, mx.bfloat16), mx.zeros(shp, mx.bfloat16)]
            mx.eval(self._bufs[0], self._bufs[1])
        self._active = 0
        self.buf = self._bufs[0]
        self._prev_step = None
        self.st_len = 0
        self.st_lens = [0] * self.C
        self._verify_meta_cache = {}
        self._refresh_meta()

    def release(self):
        if self._pool is not None and self._bufs is not None and self._bufs[0] is not None and self._bufs[1] is not None:
            if len(self._pool) >= 4 and self._shp not in self._pool:
                self._pool.pop(next(iter(self._pool)))
            self._pool[self._shp] = self._bufs
        self._bufs = None
        self.buf = None
        self._prev_step = None

    def _refresh_meta(self):
        self.ks_main = []
        self.vs_main = []
        self._sigs = []
        self._rot_flags = []
        self._need_grow_check = []
        for v in range(self.nv):
            kv0 = self.caches[0][v][0]
            rot_i = 1 if hasattr(kv0, "_idx") else 0
            self._rot_flags.append(bool(rot_i))
            ks_v, vs_v, sig = [], [], []
            for b in range(self.C):
                kv = self.caches[b][v][0]
                ks_v.append(kv._keys)
                vs_v.append(kv._values)
                end_m = int(kv.offset)
                kcap = int(kv._keys.shape[2])
                vcap = int(kv._values.shape[2])
                if rot_i:
                    max_s = int(kv.max_size)
                    idx_m = int(kv._idx)
                    rot_w = 1 if end_m >= max_s else 0
                    slot_m = (0 if idx_m == max_s else idx_m) if rot_w else end_m
                    if not rot_w:
                        self._need_grow_check.append((v, b, end_m, min(kcap, max_s)))
                else:
                    max_s = 0
                    idx_m = 0
                    rot_w = 0
                    slot_m = end_m
                sig.append((end_m, idx_m, rot_i, kcap, vcap, slot_m, rot_w, max_s))
            self.ks_main.append(ks_v)
            self.vs_main.append(vs_v)
            self._sigs.append(tuple(sig))
        self._ks_flat_all = [x for ks_v in self.ks_main for x in ks_v]
        self._vs_flat_all = [x for vs_v in self.vs_main for x in vs_v]
        self._full_visits = [v for v in range(self.nv) if not self._rot_flags[v]]
        if self._full_visits:
            f0 = tuple(int(self.caches[b][self._full_visits[0]][0].offset) for b in range(self.C))
            unif = all(
                tuple(int(self.caches[b][v][0].offset) for b in range(self.C)) == f0
                for v in self._full_visits[1:]
            )
            self._full_end_main = f0 if unif else None
        else:
            self._full_end_main = None
        self._meta_cache.clear()
        self._off_arr_cache.clear()
        self._verify_meta_cache.clear()
        self._refresh_cascade_meta()

    def _refresh_cascade_meta(self):
        """Detect streams sharing a pinned `kv._cascade_ref` prefix and build tiny private suffix buffers (0 GiB copy)."""
        self._cascade_groups = []
        self._casc_shared = {}
        self._casc_k_priv = {}
        self._casc_v_priv = {}
        self._casc_priv_len = {}
        self._casc_stream_info = {}
        if not self._full_visits or not (2 <= self.C <= 8):
            return
        min_toks = _cascade_min_tokens()
        by_key = {}
        for b in range(self.C):
            ref0 = getattr(self.caches[b][self._full_visits[0]][0], "_cascade_ref", None)
            if ref0 is None or len(ref0) != 4:
                continue
            pid0, l_sh0, _, _ = ref0
            l_sh0 = int(l_sh0)
            if l_sh0 < min_toks or (l_sh0 % 256) != 0:
                continue
            ok = True
            for v in self._full_visits:
                kv = self.caches[b][v][0]
                rf = getattr(kv, "_cascade_ref", None)
                if (
                    rf is None
                    or len(rf) != 4
                    or rf[0] != pid0
                    or int(rf[1]) != l_sh0
                    or int(kv.offset) < l_sh0
                    or int(rf[2].shape[-2]) < l_sh0
                    or int(rf[3].shape[-2]) < l_sh0
                ):
                    ok = False
                    break
            if ok:
                by_key.setdefault((pid0, l_sh0), []).append(b)

        groups = [(pid, l_sh, grp) for (pid, l_sh), grp in by_key.items() if 2 <= len(grp) <= 8]
        if not groups:
            return
        Hkv = int(self.caches[0][self._full_visits[0]][0]._keys.shape[1])
        dummy_pr = mx.zeros((1, Hkv, 1, 128), mx.bfloat16)
        to_eval = []
        self._cascade_groups = groups
        for v in self._full_visits:
            self._casc_shared[v] = []
            self._casc_k_priv[v] = [None] * self.C
            self._casc_v_priv[v] = [None] * self.C
            self._casc_priv_len[v] = [0] * self.C
            for pid, l_sh, grp in groups:
                rf_g = self.caches[grp[0]][v][0]._cascade_ref
                self._casc_shared[v].append((rf_g[2], rf_g[3]))
                for b in grp:
                    self._casc_stream_info[b] = (pid, l_sh)
                    kv = self.caches[b][v][0]
                    priv_len = int(kv.offset) - l_sh
                    self._casc_priv_len[v][b] = priv_len
                    cp = getattr(kv, "_cascade_priv", None)
                    if cp is not None and cp[0] == pid and int(cp[1]) == l_sh and int(cp[4]) == priv_len:
                        self._casc_k_priv[v][b] = cp[2]
                        self._casc_v_priv[v][b] = cp[3]
                    elif priv_len <= 0:
                        self._casc_k_priv[v][b] = dummy_pr
                        self._casc_v_priv[v][b] = dummy_pr
                    else:
                        kp = mx.contiguous(kv._keys[:, :, l_sh : int(kv.offset), :])
                        vp = mx.contiguous(kv._values[:, :, l_sh : int(kv.offset), :])
                        self._casc_k_priv[v][b] = kp
                        self._casc_v_priv[v][b] = vp
                        to_eval.append(kp)
                        to_eval.append(vp)
        if to_eval:
            mx.eval(to_eval)

    def needs_long_or_cascade(self, W: int = 1) -> bool:
        if not self._full_visits or not (2 <= self.C <= 8) or self.C * int(W) > 32:
            return False
        if any(2 <= len(g[2]) <= 8 and len(g[2]) * int(W) <= 32 for g in self._cascade_groups):
            return True
        min_s = _multi_long_min_s()
        for v in self._full_visits:
            sig = self._sigs[v]
            if max(sig[b][0] + self.st_lens[b] for b in range(self.C)) + int(W) >= min_s:
                return True
        return False

    def maybe_flush_before_step(self):
        st_max = max(self.st_lens) if self.st_lens else self.st_len
        need_flush = (
            st_max >= self.scap
            or (st_max > 0 and (self._prev_step is None or any(s != self.st_lens[0] for s in self.st_lens)))
        )
        if not need_flush and self._need_grow_check:
            for _, b, end_m, lim in self._need_grow_check:
                if end_m + self.st_lens[b] + 1 > lim:
                    need_flush = True
                    break
        if need_flush and st_max > 0:
            self.flush(materialize_cascade=False)
        # Handle RotatingKVCache with cap < max_size when st_len == 0 (e.g. right after short prefill)
        grew = False
        for v, b, end_m, _ in list(self._need_grow_check):
            kv = self.caches[b][v][0]
            cap = int(kv._keys.shape[2])
            if cap < kv.max_size and end_m + 1 > cap:
                add = min(kv.step, kv.max_size - cap)
                ext_k = mx.zeros((1, kv._keys.shape[1], add, 128), kv._keys.dtype)
                ext_v = mx.zeros((1, kv._values.shape[1], add, 128), kv._values.dtype)
                kv._keys = mx.concatenate([kv._keys, ext_k], axis=2)
                kv._values = mx.concatenate([kv._values, ext_v], axis=2)
                kv._idx = end_m
                mx.eval(kv._keys, kv._values)
                grew = True
        if grew:
            self._refresh_meta()

    def maybe_flush_before_verify(self, W: int):
        st_max = max(self.st_lens)
        need_flush = (st_max + W > self.scap)
        if not need_flush and self._need_grow_check:
            for _, b, end_m, lim in self._need_grow_check:
                if end_m + self.st_lens[b] + W > lim:
                    need_flush = True
                    break
        if need_flush and st_max > 0:
            self.flush(materialize_cascade=False)
        grew = False
        for v, b, end_m, _ in list(self._need_grow_check):
            kv = self.caches[b][v][0]
            cap = int(kv._keys.shape[2])
            if cap < kv.max_size and end_m + W > cap:
                add = min(kv.step, kv.max_size - cap)
                ext_k = mx.zeros((1, kv._keys.shape[1], add, 128), kv._keys.dtype)
                ext_v = mx.zeros((1, kv._values.shape[1], add, 128), kv._values.dtype)
                kv._keys = mx.concatenate([kv._keys, ext_k], axis=2)
                kv._values = mx.concatenate([kv._values, ext_v], axis=2)
                kv._idx = end_m
                mx.eval(kv._keys, kv._values)
                grew = True
        if grew:
            self._refresh_meta()

    def verify_view(self, W: int):
        st_t = tuple(self.st_lens)
        step_metas = [None] * self.nv
        step_Ss = [None] * self.nv
        by_sig = {}
        for v in range(self.nv):
            sig = self._sigs[v]
            ent = by_sig.get(sig)
            if ent is None:
                mkey = (sig, st_t, int(W))
                ent = self._verify_meta_cache.get(mkey)
                if ent is None:
                    mt = np.empty((self.C, RS._MCW), np.int32)
                    Ss = []
                    for b, (end_m, _, rot_i, kcap, vcap, slot_m, rot_w, max_s) in enumerate(sig):
                        st_b = st_t[b]
                        mt[b] = (end_m, slot_m, rot_i, rot_w, max_s, kcap, vcap, st_b)
                        base_b = end_m + st_b
                        if rot_i:
                            for d in range(W):
                                Ss.append(min(base_b + d + 1, max_s))
                        else:
                            for d in range(W):
                                Ss.append(base_b + d + 1)
                    ent = (mx.array(mt), tuple(Ss))
                    if len(self._verify_meta_cache) >= 512:
                        self._verify_meta_cache.clear()
                    self._verify_meta_cache[mkey] = ent
                by_sig[sig] = ent
            step_metas[v], step_Ss[v] = ent
        if self._full_end_main is not None:
            offs = [e + st_t[b] + d for b, e in enumerate(self._full_end_main) for d in range(W)]
            full_offs = (mx.array(offs, mx.int32), max(offs) + 1)
        else:
            full_offs = None
        return step_metas, step_Ss, full_offs

    def full_off_for_verify_visit(self, v: int, W: int):
        offs = [int(self.caches[b][v][0].offset) + self.st_lens[b] + d for b in range(self.C) for d in range(W)]
        return mx.array(offs, mx.int32), max(offs) + 1

    def step_view(self):
        t = self.st_len
        step_metas = [None] * self.nv
        step_Ss = [None] * self.nv
        by_sig = {}
        for v in range(self.nv):
            sig = self._sigs[v]
            ent = by_sig.get(sig)
            if ent is None:
                mkey = (sig, t)
                ent = self._meta_cache.get(mkey)
                if ent is None:
                    mt = np.empty((self.C, RS._SMW), np.int32)
                    Ss = []
                    for b, (end_m, idx_m, rot_i, kcap, vcap, slot_m, rot_w, max_s) in enumerate(sig):
                        end_now = end_m + t + 1
                        if rot_i:
                            S_now = min(end_now, max_s)
                            idx_now = (((slot_m + t) % max_s) + 1) if rot_w else end_now
                        else:
                            S_now = end_now
                            idx_now = 0
                        mt[b] = (S_now, end_now, idx_now, rot_i, kcap, vcap, slot_m, rot_w, max_s)
                        Ss.append(S_now)
                    ent = (mx.array(mt), tuple(Ss))
                    if len(self._meta_cache) >= 512:
                        self._meta_cache.clear()
                    self._meta_cache[mkey] = ent
                by_sig[sig] = ent
            step_metas[v], step_Ss[v] = ent
        if self._full_end_main is not None:
            oarr = self._off_arr_cache.get(t)
            if oarr is None:
                oarr = self._off_arr_cache[t] = mx.array([e + t for e in self._full_end_main], mx.int32)
            full_offs = (oarr, max(self._full_end_main) + t + 1)
        else:
            full_offs = None
        return step_metas, step_Ss, full_offs, t

    def full_off_for_visit(self, v, t):
        offs = [int(self.caches[b][v][0].offset) + t for b in range(self.C)]
        return mx.array(offs, mx.int32), max(offs) + 1

    def commit_step(self, kv_news):
        cur = mx.stack(kv_news, axis=0)[:, :, :, :, None, :]
        nxt = 1 - self._active
        b = self._bufs[nxt]
        self._bufs[nxt] = None
        if self.st_len == 0:
            src = cur
            pos = 0
        else:
            src = mx.concatenate([self._prev_step, cur], axis=4)
            pos = self.st_len - 1
        b = mx.slice_update(b, src, FD._wpos(pos), axes=(4,))
        self._bufs[nxt] = b
        self._active = nxt
        self.buf = b
        self._prev_step = cur
        self.st_len += 1
        self.st_lens = [self.st_len] * self.C

    def commit_verify_w(self, kv_news, keeps, W: int):
        """Commit per-stream variable keep_c tokens `(1 <= keep_c <= W)` from `kv_news[v] (2, Hkv, C*W, 128)`."""
        assert len(kv_news) == self.nv and len(keeps) == self.C
        nxt = 1 - self._active
        self._bufs[nxt] = None
        old_b = self.buf
        self.buf = None
        Hkv = int(old_b.shape[3])
        nv = self.nv
        C = self.C
        scap = self.scap
        k_t = tuple(int(k) for k in keeps)
        st_t = tuple(int(s) for s in self.st_lens)
        k_arr = mx.array(k_t, mx.int32)
        st_arr = mx.array(st_t, mx.int32)
        kern = _stage_commit_multi_k.get(nv)
        if kern is None:
            src = """
  uint tid = thread_position_in_grid.x;
  const uint total_v4 = NV * CC * 2u * HKV * SCAP * 32u;
  if (tid >= total_v4) return;
  uint d4 = tid & 31u;
  uint q0 = tid >> 5;
  uint s = q0 % SCAP;
  uint q1 = q0 / SCAP;
  uint h = q1 % HKV;
  uint q2 = q1 / HKV;
  uint kv = q2 & 1u;
  uint q3 = q2 >> 1;
  uint c = q3 % CC;
  uint v = q3 / CC;
  int st0 = st_lens[c];
  int kc = keeps[c];
  int rel = int(s) - st0;
  if (rel >= 0 && rel < kc) {
    size_t s_off = (((((size_t(v) * 2u + kv) * HKV + h) * (CC * WW) + (c * WW + uint(rel))) * 128u) + d4 * 4u) >> 2;
    ((device uint2*)out)[tid] = ((const device uint2*)kv_all)[s_off];
  } else if (int(s) < st0) {
    ((device uint2*)out)[tid] = ((const device uint2*)st_old)[tid];
  } else {
    ((device uint2*)out)[tid] = uint2(0u, 0u);
  }
"""
            kern = _stage_commit_multi_k[nv] = mx.fast.metal_kernel(
                name=f"stage_commit_multi_{nv}",
                input_names=["st_old", "keeps", "st_lens", "kv_all"],
                output_names=["out"],
                source=src,
                ensure_row_contiguous=True,
            )
        kv_all = mx.stack(kv_news, axis=0)
        n_v4 = nv * C * 2 * Hkv * scap * 32
        (new_b,) = kern(
            inputs=[old_b, k_arr, st_arr, kv_all],
            template=[("NV", nv), ("CC", C), ("WW", int(W)), ("HKV", Hkv), ("SCAP", scap)],
            grid=(((n_v4 + 255) // 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[old_b.shape],
            output_dtypes=[mx.bfloat16],
        )
        self._bufs[nxt] = new_b
        self._active = nxt
        self.buf = new_b
        self._prev_step = None
        self.st_lens = [st_t[c] + k_t[c] for c in range(C)]
        self.st_len = max(self.st_lens)

    def flush(self, *, materialize_cascade: bool = True):
        has_st = max(self.st_lens) > 0
        if not has_st and not (materialize_cascade and self._casc_stream_info):
            return
        to_eval = []
        full_set = set(self._full_visits)
        for b in range(self.C):
            k = int(self.st_lens[b])
            c_info = self._casc_stream_info.get(b)
            if k <= 0 and not (materialize_cascade and c_info is not None):
                continue
            src = self.buf[:, b : b + 1, :, :, :k, :] if k > 0 else None
            for v in range(self.nv):
                kv = self.caches[b][v][0]
                if c_info is not None and v in full_set:
                    pid, l_sh = c_info
                    kp = self._casc_k_priv[v][b]
                    vp = self._casc_v_priv[v][b]
                    pl = int(self._casc_priv_len[v][b])
                    if k > 0:
                        k_add = src[v, :, 0]
                        v_add = src[v, :, 1]
                        if pl == 0:
                            cap_new = ((k + 255) // 256) * 256
                            kp_new = mx.zeros((1, k_add.shape[1], cap_new, 128), k_add.dtype)
                            vp_new = mx.zeros((1, v_add.shape[1], cap_new, 128), v_add.dtype)
                            kp = mx.slice_update(kp_new, k_add, FD._wpos(0), axes=(2,))
                            vp = mx.slice_update(vp_new, v_add, FD._wpos(0), axes=(2,))
                        else:
                            cap_cur = int(kp.shape[2])
                            if pl + k > cap_cur:
                                target = max(((pl + k + 255) // 256) * 256, cap_cur * 2)
                                ext_k = mx.zeros((1, kp.shape[1], target - cap_cur, 128), kp.dtype)
                                ext_v = mx.zeros((1, vp.shape[1], target - cap_cur, 128), vp.dtype)
                                kp = mx.concatenate([kp, ext_k], axis=2)
                                vp = mx.concatenate([vp, ext_v], axis=2)
                            kp = mx.slice_update(kp, k_add, FD._wpos(pl), axes=(2,))
                            vp = mx.slice_update(vp, v_add, FD._wpos(pl), axes=(2,))
                        pl += k
                        kv.offset = int(kv.offset) + k
                        self._casc_k_priv[v][b] = kp
                        self._casc_v_priv[v][b] = vp
                        self._casc_priv_len[v][b] = pl
                    if materialize_cascade:
                        rf = kv._cascade_ref
                        k_sh, v_sh = rf[2], rf[3]
                        if pl > 0:
                            kv._keys = mx.concatenate([k_sh[:, :, :l_sh, :], kp[:, :, :pl, :]], axis=2)
                            kv._values = mx.concatenate([v_sh[:, :, :l_sh, :], vp[:, :, :pl, :]], axis=2)
                        else:
                            kv._keys = k_sh
                            kv._values = v_sh
                        if hasattr(kv, "_cascade_priv"):
                            delattr(kv, "_cascade_priv")
                        to_eval.append(kv._keys)
                        to_eval.append(kv._values)
                    else:
                        kv._cascade_priv = (pid, l_sh, kp, vp, pl)
                        to_eval.append(kp)
                        to_eval.append(vp)
                elif k > 0:
                    kv.commit_unpacked_w(src[v, :, 0], src[v, :, 1])
                    to_eval.append(kv._keys)
                    to_eval.append(kv._values)
        if to_eval:
            mx.eval(to_eval)
        self.st_len = 0
        self.st_lens = [0] * self.C
        self._prev_step = None
        self._refresh_meta()


def _gather_stack_kv_chunk(kvs, path_pad, W, keep_arr=None):
    """Gather `kv[:, :, path_pad, :]` (clamping `w >= keep` to `path_pad[0]` when `keep_arr` is set) and stack <=24 tensors in 1 launch."""
    N = len(kvs)
    has_keep = keep_arr is not None
    assert 1 <= N <= 24 and kvs[0].dtype == mx.bfloat16 and kvs[0].shape[0] == 2 and kvs[0].shape[1] == 8 and kvs[0].shape[3] == 128
    key = (N, has_keep)
    k = _gather_stack_kv_k.get(key)
    if k is None:
        ptrs = ", ".join(f"(const device uint4*)k{i}" for i in range(N))
        src_w_expr = "int src_w = (int(w_i) < keep_arr[0]) ? path_pad[w_i] : path_pad[0];" if has_keep else "int src_w = path_pad[w_i];"
        src = f"""
  uint tid = thread_position_in_grid.x;
  const uint total_v8 = uint(NN) * 2u * 8u * uint(WW) * 16u;
  if (tid >= total_v8) return;
  uint d8 = tid & 15u;
  uint q0 = tid >> 4;
  uint w_i = q0 % uint(WW);
  uint q1 = q0 / uint(WW);
  uint h = q1 & 7u;
  uint q2 = q1 >> 3;
  uint kv = q2 & 1u;
  uint n = q2 >> 1;
  {src_w_expr}
  size_t s_off = (((size_t(kv) * 8u + h) * uint(WW) + uint(src_w)) * 16u) + d8;
  const device uint4* ptrs[NN] = {{{ptrs}}};
  ((device uint4*)out)[tid] = ptrs[n][s_off];
"""
        ins_names = [f"k{i}" for i in range(N)] + ["path_pad"] + (["keep_arr"] if has_keep else [])
        k = _gather_stack_kv_k[key] = mx.fast.metal_kernel(
            name=f"gather_stack_kv_{N}_{int(has_keep)}",
            input_names=ins_names,
            output_names=["out"],
            source=src,
            ensure_row_contiguous=True,
        )
    n_v8 = N * 2 * 8 * int(W) * 16
    ins = list(kvs) + [path_pad.reshape(-1)] + ([keep_arr.reshape(-1)] if has_keep else [])
    return k(
        inputs=ins,
        template=[("NN", N), ("WW", int(W))],
        grid=(((n_v8 + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(N, 2, 8, int(W), 128)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _gather_stack_kv_gpu(kvs, path_pad, W, keep_arr=None):
    if len(kvs) <= 24:
        return _gather_stack_kv_chunk(kvs, path_pad, W, keep_arr=keep_arr)
    parts = []
    for i in range(0, len(kvs), 20):
        parts.append(_gather_stack_kv_chunk(kvs[i : i + 20], path_pad, W, keep_arr=keep_arr))
    return mx.concatenate(parts, axis=0)


def update_kv_stage_gpu(st_old, kvs, path_pad, st_pos, W):
    """Append/scatter W-verified KV slots into small (NL, 2, 8, SCAP, 128) staging buffer in 1 kernel."""
    NL = len(kvs)
    _, _, HKV, SCAP, DD = st_old.shape
    assert HKV == 8 and DD == 128 and st_old.dtype == mx.bfloat16 and int(st_old.shape[0]) == NL
    k = _update_kv_stage_k.get(NL)
    if k is None:
        ptrs = ", ".join(f"(const device uint2*)k{i}" for i in range(NL))
        src = f"""
  uint tid = thread_position_in_grid.x;
  const uint total_v4 = NL * 2u * 8u * SCAP * 32u;
  if (tid >= total_v4) return;
  uint d4 = tid & 31u;
  uint q0 = tid >> 5;
  uint s = q0 % SCAP;
  uint q1 = q0 / SCAP;
  uint h = q1 & 7u;
  uint q2 = q1 >> 3;
  uint kv = q2 & 1u;
  uint l = q2 >> 1;
  int sp = st_pos[0];
  int rel = int(s) - sp;
  if (rel >= 0 && rel < WW) {{
    int src_w = path_pad[rel];
    size_t s_off = ((((size_t(kv) * 8u + h) * WW + uint(src_w)) * 128u) + d4 * 4u) >> 2;
    const device uint2* ptrs[NL] = {{{ptrs}}};
    ((device uint2*)out)[tid] = ptrs[l][s_off];
  }} else {{
    ((device uint2*)out)[tid] = ((const device uint2*)st_old)[tid];
  }}
"""
        k = _update_kv_stage_k[NL] = mx.fast.metal_kernel(
            name=f"update_kv_stage_{NL}",
            input_names=["st_old"] + [f"k{i}" for i in range(NL)] + ["path_pad", "st_pos"],
            output_names=["out"],
            source=src,
            ensure_row_contiguous=True,
        )
    n_v4 = NL * 2 * 8 * int(SCAP) * 32
    return k(
        inputs=[st_old] + list(kvs) + [path_pad.reshape(-1), st_pos.reshape(-1)],
        template=[("NL", NL), ("SCAP", int(SCAP)), ("WW", int(W))],
        grid=(((n_v4 + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[st_old.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def _get_cw_meta_arr(mkey):
    a = RS._CW_META_CACHE.get(mkey)
    if a is None:
        if len(RS._CW_META_CACHE) >= 256:
            RS._CW_META_CACHE.clear()
        a = RS._CW_META_CACHE[mkey] = mx.array(mkey, mx.int32)
    return a


def _get_tau_lut(layer, end0, max_depth):
    a = layer["self_attn"]
    key = (int(end0), int(max_depth), a.log_scaling_alpha, a.log_scaling_n_floor)
    lut = _cw_tau_lut_cache.get(key)
    if lut is None:
        if len(_cw_tau_lut_cache) >= 256:
            _cw_tau_lut_cache.clear()
        lut = _cw_tau_lut_cache[key] = mx.concatenate([FD._full_tau(layer, int(end0) + d).reshape(1) for d in range(int(max_depth) + 1)])
    return lut


def _get_tau_table(layer, max_pos):
    a = layer["self_attn"]
    size = max(4096, ((int(max_pos) + 4095) // 4096) * 4096)
    key = (a.log_scaling_alpha, a.log_scaling_n_floor, size)
    tbl = _cw_tau_table_cache.get(key)
    if tbl is None:
        if len(_cw_tau_table_cache) >= 16:
            _cw_tau_table_cache.clear()
        pos = mx.arange(size, dtype=mx.int32).astype(mx.float32)
        tbl = 1.0 + a.log_scaling_alpha * mx.log(mx.maximum((pos + 1) / a.log_scaling_n_floor, 1.0))
        mx.eval(tbl)
        _cw_tau_table_cache[key] = tbl
    return tbl


def _layer_causal_w_body(
    cw_meta, kv_vmap, kv_ring, kv_full, kv_ring_st, kv_full_st, conv_list,
    layer, v, h, bs, msg, W, tau_cache=None, depths=None, conv_anc=None, tree_meta=None,
    depths_arr=None, tau_lut=None, tau_table=None, meta_arr=None, s_bounds=None,
    prev_moe=None, skip_combine=False, emit_h=False, use_stage=False, w_act=None, kv=None,
    b1_unp_full=None,
):
    a, sliding, op_m, rel_proj, scale, window, rot, max_size = cw_meta[v]
    cs = conv_list[v]
    if prev_moe is not None:
        assert w_act is None
        r1 = RL.k1_from_down_causal_layer(layer, prev_moe, conv_anc, bs, msg, emit_h=emit_h)
        h = r1.get("h")
        prev_w3 = r1["w3"]
    else:
        r1 = RL.k1_pre_layer(layer, h, bs, msg, w_act=w_act)
        prev_w3 = None
    nr = r1["normed_residual"]
    if msg is None:
        qkv_flat, qkv_lay = RP.qkv_rows(a, r1["q_in"], r1["k_in"], r1["v_in"], flat=True)
    elif (
        os.environ.get("LOOPLING_FUSE_K3B_QKV", "1") not in ("0", "")
        and int(r1["s_q"].shape[0]) <= int(os.environ.get("LOOPLING_FUSE_K3B_QKV_MAX_B", "4"))
    ):
        qkv_flat, qkv_lay = RP.qkv_msg_rows(layer, r1, msg, flat=True)
    else:
        gs = (layer["q_message_gate"], layer["k_message_gate"], layer["v_message_gate"])
        mixed = RV._gates(gs, (r1["s_q"], r1["s_k"], r1["s_v"]), msg, (0, 1, 2), r1["qkv_gate_in"])
        q_in, k_in, v_in = RL.k3b_norm_layer(layer, mixed)
        qkv_flat, qkv_lay = RP.qkv_rows(a, q_in, k_in, v_in, flat=True)
    if qkv_lay[2][0] == "gate" and qkv_lay[3][0] == "sinks":
        gate_off, sink_off = qkv_lay[2][1], qkv_lay[3][1]
        gate_stride, sink_stride = qkv_lay[2][2], qkv_lay[3][2]
    else:
        dm = {item[0]: (item[1], item[2]) for item in qkv_lay}
        gate_off, gate_stride = dm["gate"]
        sink_off, sink_stride = dm["sinks"]
    tau_w = None
    if not sliding:
        if tau_table is not None and meta_arr is not None and depths_arr is not None:
            tau_w = tau_table[meta_arr[1] + depths_arr]
        elif tau_lut is not None and depths_arr is not None:
            tau_w = tau_lut[depths_arr]
        else:
            end0_t = kv.offset
            tkey = (end0_t, depths, a.log_scaling_alpha, a.log_scaling_n_floor)
            tau_w = tau_cache.get(tkey) if tau_cache is not None else None
            if tau_w is None:
                tau_w = mx.concatenate([FD._full_tau(layer, end0_t + d).reshape(1) for d in depths])
                if tau_cache is not None:
                    tau_cache[tkey] = tau_w
    rel_w_bf = FD._rel_w_bf(rel_proj)
    if not sliding and b1_unp_full is not None and kv is not None and kv._packed is None:
        q, kv_new, pb, w0, w1 = RM.mid_causal_w_flat(
            layer, qkv_flat, qkv_lay, cs[0], cs[1], W, tau=tau_w, rel_w=rel_w_bf, conv_anc=conv_anc, w_act=w_act,
        )
        unp_st, unp_lidx, unp_slot0, unp_len = b1_unp_full
        s_pref = int(kv.offset)
        attn_out = RS.sdpa_long_full_staged_unpacked(
            q, kv._keys, kv._values, kv_new, pb, qkv_flat, gate_off, sink_off,
            s_pref, s_pref + int(W), int(kv._keys.shape[2]), int(kv._values.shape[2]), int(W),
            scale=scale, kv_layer_idx=unp_lidx, tree_meta=tree_meta,
            kv_stage=unp_st, slot_main=unp_slot0, st_len=unp_len, w_act=w_act,
            gate_stride=gate_stride, sink_stride=sink_stride,
        )
    else:
        if kv_vmap is not None:
            is_ring, kv_layer_idx = kv_vmap[v]
            kv_buf = kv_ring if is_ring else kv_full
            kv_stage = (kv_ring_st if is_ring else kv_full_st) if use_stage else None
            cap = kv_buf.shape[3]
        else:
            kv_layer_idx = 0
            kv_buf = kv._packed
            kv_stage = None
            cap = kv_buf.shape[2]
        if meta_arr is not None and s_bounds is not None:
            S0 = end0 = slot0 = 0
            cw_smax = int(s_bounds[1])
        else:
            end0 = kv.offset
            S0 = min(end0, max_size) if rot else end0
            slot0 = (0 if kv._idx == max_size else kv._idx) if rot else end0
            if s_bounds is not None:
                cw_smax = int(s_bounds[1])
            else:
                max_d = max(depths) if depths is not None else (W - 1)
                cw_smax = min(end0 + max_d + 1, max_size) if rot else (S0 + max_d + 1)
        if (
            os.environ.get("LOOPLING_SDPA_MID_CW", "1") not in ("0", "")
            and kv_stage is None
            and 1 <= W <= 8
            and cw_smax < RS._TWO_PASS_N
            and hasattr(RS, "sdpa_mid_causal_w_packed")
        ):
            attn_out, kv_new, w0, w1 = RS.sdpa_mid_causal_w_packed(
                layer, qkv_flat, qkv_lay, kv_buf, cs[0], cs[1], rel_w_bf,
                S0, end0, slot0, rot, cap, max_size, W,
                scale=scale, window=window, tau=tau_w, kv_layer_idx=kv_layer_idx,
                tree_meta=tree_meta, conv_anc=conv_anc, depths=depths,
                meta_arr=meta_arr, s_bounds=s_bounds, w_act=w_act,
            )
        else:
            q, kv_new, pb, w0, w1 = RM.mid_causal_w_flat(
                layer, qkv_flat, qkv_lay, cs[0], cs[1], W, tau=tau_w, rel_w=rel_w_bf, conv_anc=conv_anc, w_act=w_act,
            )
            attn_out = RS.sdpa_causal_w_packed(
                q, kv_buf, kv_new, pb, qkv_flat, gate_off, sink_off,
                S0, end0, slot0, rot, cap, max_size, W,
                scale=scale, window=window, kv_layer_idx=kv_layer_idx, tree_meta=tree_meta, depths=depths,
                meta_arr=meta_arr, s_bounds=s_bounds, kv_stage=kv_stage, w_act=w_act,
                gate_stride=gate_stride, sink_stride=sink_stride,
            )
    op = RQ.qmv_rows(attn_out, op_m["weight"], op_m["scales"], op_m["biases"])
    r10 = RL.k10_post_layer(layer, op, h, nr, bs, cs[2], msg, debug=False, causal=True, conv_anc=conv_anc, w_act=w_act)
    if layer.is_block_boundary:
        bs = [nr] if bs is None else (bs + [nr])
    if msg is None:
        x_mlp, mlp_mixed = r10["mlp_x"], None
    elif (
        os.environ.get("LOOPLING_FUSE_K3B_MLP", "1") not in ("0", "")
        and int(r10["mlp_src"].shape[0]) <= int(os.environ.get("LOOPLING_FUSE_K3B_MLP_MAX_B", "2"))
    ):
        x_mlp, mlp_mixed = None, RP.mlp_msg_part(layer, r10, msg)
    else:
        mixed = RV._gates((layer["mlp_message_gate"],), (r10["mlp_src"],), msg, (3,), (r10["mlp_gate_in"],))
        (x_mlp,) = RL.k3b_norm_layer(layer, mixed, mlp=True)
        mlp_mixed = None
    e = RMo.moe_rows(
        layer, x_mlp, r10["residual"], cs[3], route=RV.ROUTE, causal=True, conv_anc=conv_anc,
        skip_combine=skip_combine, w_act=w_act, mlp_mixed=mlp_mixed,
    )
    if skip_combine:
        return e, bs, kv_new, (w0, w1, r10["w2"]), prev_w3, h
    return e["out"], bs, kv_new, (w0, w1, r10["w2"], e["w3"]), prev_w3, h


class BatchDecoder:
    def __init__(self, model, caches):
        install()
        self.model = model
        self.tm = model.model.language_model
        self.caches = caches
        self.C = len(caches)
        self.nv = len(caches[0])
        self.seg: dict = {}
        self._head = None
        # LOOPLING_BD_FUSED=kv: one RowKV store per visit (rows/kvstore.py, lane F5): 2 scatter launches per visit
        # instead of 2 slice_updates per row, sdpa over the packed store. The cache objects are written back by flush().
        self.kv = None
        if "kv" in _FUSED:
            self.kv = []
            for v in range(self.nv):
                self.kv.append(RKV.RowKV([c[v][0] for c in caches]))
                for c in caches:  # release source arrays immediately per visit (avoids 2x peak during __init__)
                    c[v][0].keys = c[v][0].values = None
        ref_codas = [getattr(c[-1], "_loopcd_ref_coda", None) for c in caches]
        has_ref = all(rc is not None for rc in ref_codas)
        lc = self.tm.args.loop_config
        cfg = self.tm.loopcd
        if lc is not None and cfg.mode == "logits" and cfg.enabled_for(lc["num_passes"]) and not has_ref:
            raise RuntimeError(
                "LoopCD-Logits requires prefilling caches with LoopCD-Logits enabled before constructing BatchDecoder."
            )
        self._ref_caches = ref_codas if has_ref else None
        self._ref_kv = None
        self._ref_conv = None
        if has_ref:
            self._n_coda = len(ref_codas[0])
            if self.kv is not None:
                self._ref_kv = []
                for j in range(self._n_coda):
                    self._ref_kv.append(RKV.RowKV([rc[j][0] for rc in ref_codas]))
                    for rc in ref_codas:
                        rc[j][0].keys = rc[j][0].values = None
        self._plans: dict = {}
        self._stage_main = None
        self._stage_ref = None
        self._batch_meta = None
        self._pending_stage_deps = ()
        self._stage_dep_mode = os.environ.get("LOOPLING_STAGE_DEP", "step")
        self._b1_fast = (
            self.C == 1
            and self.kv is None
            and {"pre", "k1", "post"}.issubset(_FUSED)
            and not _PLAIN
            and not RV.FREEZE
            and not RV.DUP
            and not RV.SYNC
            and RV.MOE
            and {"qmv", "gemv", "sdpa", "gather"}.issubset(_USE)
            and FD._FCONV
            and self.tm.args.hidden_size == 2304
            and all(_b1_cache_ok(caches[0][v]) for v in range(self.nv))
            and (not has_ref or all(_b1_cache_ok(ref_codas[0][j]) for j in range(self._n_coda)))
            and all(FD._fused_b1_ok_layer(layer) for layer in self.tm.layers)
        )
        self._batch_fast = (
            not self._b1_fast
            and os.environ.get("LOOPLING_BATCH_FAST", "1") not in ("0", "")
            and 2 <= self.C <= 32
            and self.kv is None
            and {"pre", "k1", "post"}.issubset(_FUSED)
            and not _PLAIN
            and not RV.FREEZE
            and not RV.DUP
            and not RV.SYNC
            and RV.MOE
            and {"qmv", "gemv", "sdpa", "gather"}.issubset(_USE)
            and FD._FCONV
            and self.tm.args.hidden_size == 2304
            and _batch_cache_ok(caches, self.nv)
            and (not has_ref or _batch_cache_ok(ref_codas, self._n_coda))
            and all(FD._fused_b1_ok_layer(layer) for layer in self.tm.layers)
        )
        self._kv_stacked = False
        self._kv_ring = None
        self._kv_full = None
        self._kv_ring_st = None
        self._kv_full_st = None
        self._kv_st_len = 0
        self._kv_main_slot_r = 0
        self._kv_main_end_r = 0
        self._kv_main_end_f = 0
        self._ring_visits = ()
        self._full_visits = ()
        self._kv_vmap = None
        self._cw_compiled = {}
        self._b1_st_cap = _BATCH_KV_STAGE_CAP
        self._b1_ring_bufs = None
        self._b1_full_bufs = None
        self._b1_st_active = 0
        self._b1_st_prev_r = None
        self._b1_st_prev_f = None
        self._b1_compiled = {}
        self._batch_compiled = {}
        self._b1_pending_kv = [None] * self.nv
        self._b1_pending_ref = [None] * self._n_coda if has_ref else None
        self._sdpa_staged_has_st_info = "st_info_arr" in inspect.signature(RS.sdpa_rows_staged).parameters
        self._cw_meta = None
        self._static_kv_vmap = None
        self._b1_unpacked_full = False
        self._b1_full_unp_visits = ()
        self._b1_full_unp_map = {}
        self._b1_full_unp_st = None
        self._b1_full_unp_len = 0
        self._b1_full_unp_slot0 = 0
        if self._b1_fast:
            self.conv = [[caches[0][v][1].cache[j] for j in range(4)] for v in range(self.nv)]
            rot_sig = tuple(
                (1 if hasattr(caches[0][v][0], "_idx") else 0, getattr(caches[0][v][0], "max_size", 0) if hasattr(caches[0][v][0], "_idx") else 0)
                for v in range(self.nv)
            )
            ring_v = tuple(v for v, (rot, _) in enumerate(rot_sig) if rot)
            full_v = tuple(v for v, (rot, _) in enumerate(rot_sig) if not rot)
            max_stack = _kv_stack_max_cap()
            use_unp_full = (
                bool(full_v)
                and max_stack > 0
                and all(
                    caches[0][v][0]._packed is None
                    and caches[0][v][0]._keys is not None
                    and caches[0][v][0]._values is not None
                    and int(caches[0][v][0].offset) == int(caches[0][full_v[0]][0].offset)
                    for v in full_v
                )
                and any(
                    int(caches[0][v][0]._keys.shape[2]) >= max_stack or int(caches[0][v][0].offset) >= max_stack
                    for v in full_v
                )
            )
            if use_unp_full:
                self._b1_unpacked_full = True
                self._b1_full_unp_visits = full_v
                self._b1_full_unp_map = {v: i for i, v in enumerate(full_v)}
                self._b1_full_unp_slot0 = int(caches[0][full_v[0]][0].offset)
                # Pack only the 40 sliding-window visits (~160 MiB); leave the 8 full visits unpacked (0 GiB copy!).
                for v in ring_v:
                    caches[0][v][0].pack_b1()
            else:
                for v in range(self.nv):
                    caches[0][v][0].pack_b1()
            v_map = [None] * self.nv
            for i, v in enumerate(ring_v):
                v_map[v] = (True, i)
            for i, v in enumerate(full_v):
                v_map[v] = (False, i)
            self._static_kv_vmap = v_map
            wsig = _tm_weight_sig(self.tm)
            hsig = (
                tuple(id(v) for _, v in tree_flatten(self.model.lm_head.parameters()))
                if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "parameters")
                else ()
            )
            esig = _env_compile_sig()
            shared = getattr(self.tm, "_cw_shared_cache", None)
            if (
                shared is not None
                and shared[0] == wsig
                and shared[1] == hsig
                and shared[2] == esig
                and shared[3] == rot_sig
            ):
                self._cw_compiled = shared[4]
                self._cw_meta = shared[5]
            else:
                cw_meta = [
                    (
                        layer["self_attn"],
                        "input_layernorm" in layer,
                        layer["self_attn"]["o_proj"],
                        layer["self_attn"]["rel_logits_proj"],
                        layer["self_attn"].scaling,
                        layer["self_attn"].sliding_window or 0,
                        rot_sig[v][0],
                        rot_sig[v][1],
                    )
                    for v, layer in enumerate(self.tm.layer_visits)
                ]
                cw_comp = {}
                self.tm._cw_shared_cache = (wsig, hsig, esig, rot_sig, cw_comp, cw_meta)
                self._cw_compiled = cw_comp
                self._cw_meta = cw_meta
            if self._ref_caches is not None:
                self._ref_conv = [
                    [self._ref_caches[0][j][1].cache[k] for k in range(4)] for j in range(self._n_coda)
                ]
                for j in range(self._n_coda):
                    self._ref_caches[0][j][0].pack_b1()
        else:
            # batched conv states per visit: 4 arrays (C, 3, dims) f32
            self.conv = [[mx.concatenate([c[v][1].cache[j] for c in caches]) for j in range(4)] for v in range(self.nv)]
            if self._ref_caches is not None:
                self._ref_conv = [
                    [mx.concatenate([rc[j][1].cache[k] for rc in self._ref_caches]) for k in range(4)]
                    for j in range(self._n_coda)
                ]
            if self._batch_fast:
                pool = getattr(self.tm, "_bd_stage_pool", None)
                if pool is None:
                    pool = self.tm._bd_stage_pool = {}
                self._stage_main = _BatchKVStage(caches, pool=pool)
                if self._ref_caches is not None:
                    self._stage_ref = _BatchKVStage(self._ref_caches)
                wsig = _tm_weight_sig(self.tm)
                shared = getattr(self.tm, "_bd_shared_cache", None)
                if shared is not None and shared[0] == wsig:
                    self._batch_compiled = shared[1]
                    self._batch_meta = shared[2]
                else:
                    bmeta = {
                        id(layer): (
                            layer["self_attn"],
                            "input_layernorm" in layer,
                            layer["self_attn"]["o_proj"],
                            layer["self_attn"]["rel_logits_proj"],
                            layer["self_attn"].scaling,
                            layer["self_attn"].sliding_window or 0,
                        )
                        for layer in self.tm.layers
                    }
                    bcomp = {}
                    self.tm._bd_shared_cache = (wsig, bcomp, bmeta)
                    self._batch_compiled = bcomp
                    self._batch_meta = bmeta

    def release_stage(self):
        if self._stage_main is not None:
            self._stage_main.release()
        if self._stage_ref is not None:
            self._stage_ref.release()

    def _flush_b1_pending(self, caches=None, pending=None):
        """Commit any deferred B=1 kv_new tensors (from RS.sdpa_mid_b1) into packed KV caches."""
        if not self._b1_fast:
            return
        if caches is None:
            caches = self.caches
            pending = self._b1_pending_kv
        elif pending is None:
            pending = self._b1_pending_ref if caches is self._ref_caches else self._b1_pending_kv
        if pending is None:
            return
        for v, kv_new in enumerate(pending):
            if kv_new is not None:
                caches[0][v][0].update_packed_b1(kv_new, FD._wpos)
                pending[v] = None

    def _ensure_stacked_kv(self):
        self._flush_b1_pending()
        if self._kv_stacked:
            return True
        if os.environ.get("LOOPLING_KV_STACK", "1") in ("0", ""):
            return False
        kvs = [self.caches[0][v][0] for v in range(self.nv)]
        ring_v = tuple(v for v, kv in enumerate(kvs) if hasattr(kv, "_idx"))
        full_v = tuple(v for v, kv in enumerate(kvs) if not hasattr(kv, "_idx"))

        def _group_uniform(vs):
            if not vs:
                return True
            k0 = kvs[vs[0]]
            if k0._packed is None:
                return False
            sig = (k0.offset, getattr(k0, "_idx", 0), getattr(k0, "max_size", 0), k0._packed.shape[2], k0.step)
            for v in vs[1:]:
                kv = kvs[v]
                if kv._packed is None or (
                    kv.offset,
                    getattr(kv, "_idx", 0),
                    getattr(kv, "max_size", 0),
                    kv._packed.shape[2],
                    kv.step,
                ) != sig:
                    return False
            return True

        if self._b1_unpacked_full:
            return False
        if not _group_uniform(ring_v) or not _group_uniform(full_v):
            return False
        max_stack = _kv_stack_max_cap()
        if full_v and max_stack > 0:
            kf0 = kvs[full_v[0]]
            if int(kf0._packed.shape[2]) >= max_stack or int(kf0.offset) >= max_stack:
                return False
        v_map = [None] * self.nv
        if ring_v:
            self._kv_ring = mx.contiguous(mx.stack([kvs[v]._packed for v in ring_v], axis=0))
            for i, v in enumerate(ring_v):
                kvs[v]._packed = None
                v_map[v] = (True, i)
        if full_v:
            self._kv_full = mx.contiguous(mx.stack([kvs[v]._packed for v in full_v], axis=0))
            for i, v in enumerate(full_v):
                kvs[v]._packed = None
                v_map[v] = (False, i)
        self._ring_visits = ring_v
        self._full_visits = full_v
        self._kv_vmap = v_map
        self._kv_stacked = True
        return True

    def unstack_kv(self):
        """Restore per-visit 4D `_packed` buffers onto caches if currently stacked."""
        self._flush_b1_pending()
        if not self._kv_stacked:
            return
        self.flush_kv_stage()
        if self._kv_ring is not None:
            buf = self._kv_ring
            self._kv_ring = None
            for i, v in enumerate(self._ring_visits):
                self.caches[0][v][0]._packed = mx.contiguous(buf[i])
        if self._kv_full is not None:
            buf = self._kv_full
            self._kv_full = None
            for i, v in enumerate(self._full_visits):
                self.caches[0][v][0]._packed = mx.contiguous(buf[i])
        self._kv_ring_st = None
        self._kv_full_st = None
        self._b1_ring_bufs = None
        self._b1_full_bufs = None
        self._b1_st_active = 0
        self._b1_st_prev_r = None
        self._b1_st_prev_f = None
        self._kv_vmap = None
        self._kv_stacked = False

    def init_kv_stage(self) -> bool:
        """Allocate small 32-slot KV staging buffers over stacked main KV buffers."""
        if not self._ensure_stacked_kv() or not self._ring_visits or not self._full_visits:
            return False
        self.flush_kv_stage()
        self._b1_ring_bufs = None
        self._b1_full_bufs = None
        self._b1_st_active = 0
        self._b1_st_prev_r = None
        self._b1_st_prev_f = None
        if self._kv_ring_st is None or int(self._kv_ring_st.shape[0]) != len(self._ring_visits) or int(self._kv_ring_st.shape[3]) != KV_STAGE_CAP:
            self._kv_ring_st = mx.zeros(
                (len(self._ring_visits), 2, self._kv_ring.shape[2], KV_STAGE_CAP, self._kv_ring.shape[4]),
                self._kv_ring.dtype,
            )
        if self._kv_full_st is None or int(self._kv_full_st.shape[0]) != len(self._full_visits) or int(self._kv_full_st.shape[3]) != KV_STAGE_CAP:
            self._kv_full_st = mx.zeros(
                (len(self._full_visits), 2, self._kv_full.shape[2], KV_STAGE_CAP, self._kv_full.shape[4]),
                self._kv_full.dtype,
            )
        kv_r = self.caches[0][self._ring_visits[0]][0]
        kv_f = self.caches[0][self._full_visits[0]][0]
        max_size_r = int(kv_r.max_size)
        self._kv_st_len = 0
        self._kv_main_end_r = int(kv_r.offset)
        self._kv_main_slot_r = 0 if kv_r._idx == max_size_r else int(kv_r._idx)
        self._kv_main_end_f = int(kv_f.offset)
        mx.eval(self._kv_ring_st, self._kv_full_st)
        return True

    def _ensure_b1_kv_stage(self) -> bool:
        """Ensure stacked 5D ping-pong KV staging buffers are ready for C=1 decode (opt-in via LOOPLING_B1_STAGED_KV=1)."""
        if os.environ.get("LOOPLING_B1_STAGED_KV", "0") in ("0", ""):
            return False
        if not self._ensure_stacked_kv() or not self._ring_visits or not self._full_visits:
            return False
        if self._b1_ring_bufs is None:
            if self._kv_st_len > 0:
                self.flush_kv_stage()
            scap = self._b1_st_cap
            shp_r = (len(self._ring_visits), 2, self._kv_ring.shape[2], scap, self._kv_ring.shape[4])
            shp_f = (len(self._full_visits), 2, self._kv_full.shape[2], scap, self._kv_full.shape[4])
            self._b1_ring_bufs = [mx.zeros(shp_r, self._kv_ring.dtype), mx.zeros(shp_r, self._kv_ring.dtype)]
            self._b1_full_bufs = [mx.zeros(shp_f, self._kv_full.dtype), mx.zeros(shp_f, self._kv_full.dtype)]
            self._b1_st_active = 0
            self._kv_ring_st = self._b1_ring_bufs[0]
            self._kv_full_st = self._b1_full_bufs[0]
            self._b1_st_prev_r = None
            self._b1_st_prev_f = None
            self._kv_st_len = 0
            mx.eval(self._b1_ring_bufs[0], self._b1_ring_bufs[1], self._b1_full_bufs[0], self._b1_full_bufs[1])
        elif self._kv_st_len >= self._b1_st_cap:
            self.flush_kv_stage()
        if self._kv_st_len == 0:
            kv_r = self.caches[0][self._ring_visits[0]][0]
            kv_f = self.caches[0][self._full_visits[0]][0]
            max_size_r = int(kv_r.max_size)
            self._kv_main_end_r = int(kv_r.offset)
            self._kv_main_slot_r = 0 if kv_r._idx == max_size_r else int(kv_r._idx)
            self._kv_main_end_f = int(kv_f.offset)
        return True

    def _commit_b1_kv_stage(self, kv_news):
        """Commit 1 token's kv_news into ping-pong 5D staging buffers via 2 donated slice_updates."""
        cur_r = mx.stack([kv_news[v] for v in self._ring_visits], axis=0)
        cur_f = mx.stack([kv_news[v] for v in self._full_visits], axis=0)
        nxt = 1 - self._b1_st_active
        br = self._b1_ring_bufs[nxt]
        bf = self._b1_full_bufs[nxt]
        self._b1_ring_bufs[nxt] = None
        self._b1_full_bufs[nxt] = None
        if self._kv_st_len == 0:
            src_r, src_f, pos = cur_r, cur_f, 0
        else:
            src_r = mx.concatenate([self._b1_st_prev_r, cur_r], axis=3)
            src_f = mx.concatenate([self._b1_st_prev_f, cur_f], axis=3)
            pos = self._kv_st_len - 1
        wpos = FD._wpos(pos)
        br = mx.slice_update(br, src_r, wpos, axes=(3,))
        bf = mx.slice_update(bf, src_f, wpos, axes=(3,))
        self._b1_ring_bufs[nxt] = br
        self._b1_full_bufs[nxt] = bf
        self._b1_st_active = nxt
        self._kv_ring_st = br
        self._kv_full_st = bf
        self._b1_st_prev_r = cur_r
        self._b1_st_prev_f = cur_f
        self.sync_kv_offsets(1, staged=True)

    def flush_kv_stage(self):
        """Flush staged KV tokens back into main stacked KV buffers if non-empty."""
        if self._kv_st_len <= 0 or not self._kv_stacked:
            self._kv_st_len = 0
            self._b1_st_prev_r = None
            self._b1_st_prev_f = None
            return
        k = int(self._kv_st_len)
        if self._ring_visits and self._kv_ring is not None and self._kv_ring_st is not None:
            br = self._kv_ring
            self._kv_ring = None
            kv_r = self.caches[0][self._ring_visits[0]][0]
            max_s = int(kv_r.max_size)
            s0 = int(self._kv_main_slot_r)
            cap_r = int(br.shape[3])
            if cap_r < max_s and s0 + k > cap_r:
                target_cap = min(max_s, ((min(s0 + k, max_s) + kv_r.step - 1) // kv_r.step) * kv_r.step)
                ext = mx.zeros((len(self._ring_visits), 2, br.shape[2], target_cap - cap_r, br.shape[4]), br.dtype)
                br = mx.concatenate([br, ext], axis=3)
            src_r = self._kv_ring_st[:, :, :, :k, :]
            if s0 + k <= max_s:
                br = mx.slice_update(br, src_r, FD._wpos(s0), axes=(3,))
            else:
                n1 = max_s - s0
                br = mx.slice_update(br, src_r[:, :, :, :n1, :], FD._wpos(s0), axes=(3,))
                br = mx.slice_update(br, src_r[:, :, :, n1:, :], FD._wpos(0), axes=(3,))
            self._kv_ring = br
            self._kv_main_end_r += k
            self._kv_main_slot_r = (s0 + k) % max_s
        if self._full_visits and self._kv_full is not None and self._kv_full_st is not None:
            bf = self._kv_full
            self._kv_full = None
            kv_f = self.caches[0][self._full_visits[0]][0]
            e0 = int(self._kv_main_end_f)
            cap_f = int(bf.shape[3])
            if e0 + k > cap_f:
                target_cap = ((e0 + k + kv_f.step - 1) // kv_f.step) * kv_f.step
                ext = mx.zeros((len(self._full_visits), 2, bf.shape[2], target_cap - cap_f, bf.shape[4]), bf.dtype)
                bf = mx.concatenate([bf, ext], axis=3)
            bf = mx.slice_update(bf, self._kv_full_st[:, :, :, :k, :], FD._wpos(e0), axes=(3,))
            self._kv_full = bf
            self._kv_main_end_f += k
        self._kv_st_len = 0
        self._b1_st_prev_r = None
        self._b1_st_prev_f = None

    def ensure_gpu_cap(self, extra: int) -> bool:
        """Pre-grow stacked ring/full KV buffers for `extra` upcoming tokens so GPU commits can donate in place."""
        if not self._ensure_stacked_kv() or not self._ring_visits or not self._full_visits:
            return False
        self.flush_kv_stage()
        grew = False
        ex = int(extra)
        kv_r = self.caches[0][self._ring_visits[0]][0]
        end0_r = int(kv_r.offset)
        cap_r = int(self._kv_ring.shape[3])
        max_size_r = int(kv_r.max_size)
        if cap_r < max_size_r and end0_r + ex > cap_r:
            buf = self._kv_ring
            self._kv_ring = None
            target_cap = min(max_size_r, ((end0_r + ex + kv_r.step - 1) // kv_r.step) * kv_r.step)
            ext = mx.zeros((len(self._ring_visits), 2, buf.shape[2], target_cap - cap_r, buf.shape[4]), buf.dtype)
            self._kv_ring = mx.concatenate([buf, ext], axis=3)
            grew = True

        kv_f = self.caches[0][self._full_visits[0]][0]
        end0_f = int(kv_f.offset)
        cap_f = int(self._kv_full.shape[3])
        if end0_f + ex > cap_f:
            buf = self._kv_full
            self._kv_full = None
            if end0_f % kv_f.step != 0:
                buf = buf[:, :, :, :end0_f, :]
            need_f = end0_f + ex
            align_f = 4096 if need_f < 16384 else 8192
            target_cap = ((need_f + align_f - 1) // align_f) * align_f
            ext = mx.zeros(
                (len(self._full_visits), 2, buf.shape[2], target_cap - buf.shape[3], buf.shape[4]), buf.dtype
            )
            self._kv_full = mx.concatenate([buf, ext], axis=3)
            grew = True
        if grew:
            mx.eval(*self.kv_eval_targets())
        return True

    def get_gpu_meta(self, staged: bool = False):
        """Return `(meta_r, meta_f)` 6-int32 (or 8-int32 when staged=True) GPU arrays matching current KV state."""
        assert self._kv_stacked and self._ring_visits and self._full_visits
        kv_r = self.caches[0][self._ring_visits[0]][0]
        kv_f = self.caches[0][self._full_visits[0]][0]
        end0_r = int(kv_r.offset)
        max_size_r = int(kv_r.max_size)
        cap_r = int(self._kv_ring.shape[3])
        slot0_r = 0 if kv_r._idx == max_size_r else int(kv_r._idx)
        s0_r = min(end0_r, max_size_r)
        end0_f = int(kv_f.offset)
        cap_f = int(self._kv_full.shape[3])
        if staged:
            st_len = int(self._kv_st_len)
            return _get_cw_meta_arr(
                (s0_r, end0_r, slot0_r, cap_r, max_size_r, 0, int(self._kv_main_slot_r), st_len)
            ), _get_cw_meta_arr((end0_f, end0_f, 0, cap_f, 0, 0, int(self._kv_main_end_f), st_len))
        return _get_cw_meta_arr((s0_r, end0_r, slot0_r, cap_r, max_size_r, 0)), _get_cw_meta_arr(
            (end0_f, end0_f, 0, cap_f, 0, 0)
        )

    def sync_kv_offsets(self, delta_keep: int, *, staged: bool = False):
        """Advance CPU KV `offset` and `_idx` counters by `delta_keep` after GPU commit."""
        d = int(delta_keep)
        assert d >= 1 and self._kv_stacked
        if staged:
            self._kv_st_len += d
        if self._ring_visits:
            kv0 = self.caches[0][self._ring_visits[0]][0]
            max_size = int(kv0.max_size)
            slot0 = 0 if kv0._idx == max_size else int(kv0._idx)
            new_off = int(kv0.offset) + d
            new_idx = ((slot0 + d - 1) % max_size) + 1
            for v in self._ring_visits:
                kv = self.caches[0][v][0]
                kv.offset = new_off
                kv._idx = new_idx
        if self._full_visits:
            new_off = int(self.caches[0][self._full_visits[0]][0].offset) + d
            for v in self._full_visits:
                self.caches[0][v][0].offset = new_off

    def kv_eval_targets(self, *, staged: bool = False):
        """Return currently active KV array(s) for mx.eval."""
        if self._kv_stacked:
            out = []
            if staged:
                if self._kv_ring_st is not None:
                    out.append(self._kv_ring_st)
                if self._kv_full_st is not None:
                    out.append(self._kv_full_st)
                return out
            if self._kv_ring is not None:
                out.append(self._kv_ring)
            if self._kv_full is not None:
                out.append(self._kv_full)
            return out
        if self._b1_unpacked_full:
            out = [c[0]._packed for c in self.caches[0] if c[0]._packed is not None]
            if self._b1_full_unp_st is not None:
                out.append(self._b1_full_unp_st)
            return out
        return [c[0]._packed for c in self.caches[0]]

    def _ensure_b1_full_unp_stage(self, extra_w: int = 1) -> None:
        """Ensure small 5D staging buffer for unpacked C=1 full-attn visits (grows 256..4096, 0 GiB copy of main KV)."""
        nf = len(self._b1_full_unp_visits)
        kv0 = self.caches[0][self._b1_full_unp_visits[0]][0]
        Hkv = int(kv0._keys.shape[1])
        if self._b1_full_unp_st is None:
            scap = max(256, ((int(extra_w) + 255) // 256) * 256)
            self._b1_full_unp_st = mx.zeros((nf, 2, Hkv, scap, 128), kv0._keys.dtype)
            self._b1_full_unp_len = 0
            self._b1_full_unp_slot0 = int(kv0.offset)
            mx.eval(self._b1_full_unp_st)
            return
        cur_cap = int(self._b1_full_unp_st.shape[3])
        if self._b1_full_unp_len + int(extra_w) <= cur_cap:
            return
        if cur_cap < 4096:
            new_cap = min(4096, max(cur_cap * 2, ((self._b1_full_unp_len + int(extra_w) + 255) // 256) * 256))
            buf = self._b1_full_unp_st
            self._b1_full_unp_st = None
            ext = mx.zeros((nf, 2, Hkv, new_cap - cur_cap, 128), buf.dtype)
            self._b1_full_unp_st = mx.concatenate([buf, ext], axis=3)
            mx.eval(self._b1_full_unp_st)
            return
        self._flush_b1_full_unpacked()

    def _commit_b1_full_unpacked(self, kv_news_full: dict, keep: int) -> None:
        """Commit `keep` tokens for all unpacked C=1 full-attn visits into tiny 5D stage (0 GiB copy)."""
        k = int(keep)
        self._ensure_b1_full_unp_stage(k)
        src = mx.stack([kv_news_full[v][:, :, :k, :] for v in self._b1_full_unp_visits], axis=0)
        buf = self._b1_full_unp_st
        self._b1_full_unp_st = None
        pos = int(self._b1_full_unp_len)
        self._b1_full_unp_st = mx.slice_update(buf, src, FD._wpos(pos), axes=(3,))
        self._b1_full_unp_len = pos + k
        new_off = self._b1_full_unp_slot0 + self._b1_full_unp_len
        for v in self._b1_full_unp_visits:
            self.caches[0][v][0].offset = new_off

    def _flush_b1_full_unpacked(self) -> None:
        """Flush staged tokens for unpacked C=1 full-attn visits into `kv._keys`/`kv._values`."""
        k = int(self._b1_full_unp_len)
        if k <= 0 or self._b1_full_unp_st is None:
            self._b1_full_unp_len = 0
            if self._b1_full_unp_visits:
                self._b1_full_unp_slot0 = int(self.caches[0][self._b1_full_unp_visits[0]][0].offset)
            return
        st = self._b1_full_unp_st
        to_eval = []
        for i, v in enumerate(self._b1_full_unp_visits):
            kv = self.caches[0][v][0]
            kv.offset = int(self._b1_full_unp_slot0)
            k_add = st[i, 0, :, :k, :][None]
            v_add = st[i, 1, :, :k, :][None]
            kv.commit_unpacked_w(k_add, v_add)
            to_eval.append(kv._keys)
            to_eval.append(kv._values)
        mx.eval(to_eval)
        self._b1_full_unp_len = 0
        self._b1_full_unp_slot0 = int(self.caches[0][self._b1_full_unp_visits[0]][0].offset)

    def kv_packed(self, v):
        """Return 4D packed KV buffer view (2, Hkv, cap, 128) for visit `v` (for tests/inspection)."""
        self._flush_b1_pending()
        if self._kv_stacked:
            if self._kv_st_len > 0:
                self.flush_kv_stage()
            is_ring, lidx = self._kv_vmap[v]
            return (self._kv_ring if is_ring else self._kv_full)[lidx]
        return self.caches[0][v][0]._packed

    def _flush_group(self, caches, conv, kv, stage=None, *, keep_cascade_split: bool = False):
        nv = len(conv)
        if self._b1_fast:
            self._flush_b1_pending(caches)
            if caches is self.caches:
                self.unstack_kv()
                if self._b1_unpacked_full:
                    self._flush_b1_full_unpacked()
            for v in range(nv):
                for j in range(4):
                    caches[0][v][1][j] = conv[v][j]
                if caches[0][v][0]._packed is not None:
                    caches[0][v][0].unpack_b1()
            return
        for v in range(nv):
            for i, c in enumerate(caches):
                for j in range(4):
                    c[v][1][j] = conv[v][j][i : i + 1]
        if stage is not None:
            stage.flush(materialize_cascade=not keep_cascade_split)
        elif kv is not None:
            for st in kv:
                st.to_caches()

    def flush(self, *, keep_cascade_split: bool = False):
        self._flush_group(
            self.caches, self.conv, self.kv, stage=self._stage_main, keep_cascade_split=keep_cascade_split
        )
        if self._ref_caches is not None:
            self._flush_group(self._ref_caches, self._ref_conv, self._ref_kv, stage=self._stage_ref)

    def snapshot_cache(self, idx: int):
        """Return a detached CacheList snapshot for stream `idx` without destroying live `_packed` or flushing other streams."""
        import copy as _copy

        if self._b1_fast:
            assert idx == 0
            self._flush_b1_pending()
            if self._kv_stacked and self._kv_st_len > 0:
                self.flush_kv_stage()
            out = []
            for v in range(self.nv):
                kv = self.caches[0][v][0]
                is_rot = hasattr(kv, "_idx")
                kv_c = RotatingKVCache(max_size=kv.max_size, keep=kv.keep) if is_rot else KVCache()
                kv_c.step = kv.step
                kv_c.offset = int(kv.offset)
                if is_rot:
                    kv_c._idx = int(kv._idx)
                if self._b1_unpacked_full and v in self._b1_full_unp_map:
                    kv_c._keys = kv._keys
                    kv_c._values = kv._values
                    if self._b1_full_unp_len > 0 and self._b1_full_unp_st is not None:
                        fi = self._b1_full_unp_map[v]
                        k = int(self._b1_full_unp_len)
                        kv_c.offset = int(self._b1_full_unp_slot0)
                        kv_c.commit_unpacked_w(
                            self._b1_full_unp_st[fi, 0, :, :k, :][None],
                            self._b1_full_unp_st[fi, 1, :, :k, :][None],
                        )
                else:
                    if self._kv_stacked:
                        is_ring, lidx = self._kv_vmap[v]
                        pk = (self._kv_ring if is_ring else self._kv_full)[lidx]
                    else:
                        pk = kv._packed
                    if pk is not None:
                        kv_c._keys = mx.contiguous(pk[0][None])
                        kv_c._values = mx.contiguous(pk[1][None])
                    else:
                        kv_c._keys = kv._keys
                        kv_c._values = kv._values
                ac = ArraysCache(4)
                ac.cache = [self.conv[v][j] for j in range(4)]
                out.append(CacheList(kv_c, ac))
            return out

        if self._batch_fast and self._stage_main is not None:
            st = self._stage_main
            k_st = int(st.st_lens[idx])
            src_all = st.buf[:, idx : idx + 1, :, :, :k_st, :] if k_st > 0 else None
            c_info = st._casc_stream_info.get(idx)
            full_set = set(st._full_visits)
            out = []
            for v in range(self.nv):
                kv = self.caches[idx][v][0]
                is_rot = hasattr(kv, "_idx")
                kv_c = RotatingKVCache(max_size=kv.max_size, keep=kv.keep) if is_rot else KVCache()
                kv_c.step = kv.step
                kv_c.offset = int(kv.offset)
                if is_rot:
                    kv_c._idx = int(kv._idx)
                if c_info is not None and v in full_set:
                    _, l_sh = c_info
                    rf = kv._cascade_ref
                    pl = int(st._casc_priv_len[v][idx])
                    if pl > 0:
                        kv_c._keys = mx.concatenate([rf[2][:, :, :l_sh, :], st._casc_k_priv[v][idx][:, :, :pl, :]], axis=2)
                        kv_c._values = mx.concatenate([rf[3][:, :, :l_sh, :], st._casc_v_priv[v][idx][:, :, :pl, :]], axis=2)
                    else:
                        kv_c._keys = rf[2]
                        kv_c._values = rf[3]
                else:
                    kv_c._keys = kv._keys
                    kv_c._values = kv._values
                if k_st > 0:
                    kv_c.commit_unpacked_w(src_all[v, :, 0], src_all[v, :, 1])
                ac = ArraysCache(4)
                ac.cache = [self.conv[v][j][idx : idx + 1] for j in range(4)]
                out.append(CacheList(kv_c, ac))
            return out

        self.flush()
        return _copy.deepcopy(self.caches[idx])

    def _segs(self, layer, k, gated=False):
        key = (id(layer), k, gated)
        s = self.seg.get(key)
        if s is None:
            fd = FD._get_fd(layer)
            w = lambda body: _wrap(layer, body)
            s1 = w(RV.make_pre(layer, *k)) if "pre" in _FUSED else fd["make_s1"](*k, wrap=w)
            s2 = w(RV.make_post(layer, *k, gated=gated)) if "post" in _FUSED else fd["make_s2"](*k, wrap=w)
            s = self.seg[key] = (s1, s2)
        return s

    def _layer(self, layer, v, h, bs, msg, tau_cache=None, caches=None, conv=None, kv=None):
        sliding = isinstance(layer, L.SlidingDecoderLayer)
        c_list = self.caches if caches is None else caches
        conv_list = self.conv if conv is None else conv
        kv_list = self.kv if kv is None else kv
        kvs = [c[v][0] for c in c_list]
        cs = conv_list[v]
        k = (bs is not None, msg is not None)
        st = kv_list[v] if kv_list is not None else None
        use_sdpa_rows = (st is None) and ("sdpa" in _USE) and not any(getattr(kv_i, "_fd_delta", None) is not None for kv_i in kvs)
        gated = ("post" in _FUSED) and (st is not None or use_sdpa_rows)
        s1, s2 = self._segs(layer, k, gated=gated)
        args = [h]
        if bs is not None:
            args.append(bs)
        if msg is not None:
            args.append(msg)
        if not sliding:
            offs = tuple(o for o, _, _ in st.state) if st is not None else tuple(FD._end(kv_i) for kv_i in kvs)
            a = layer.self_attn
            tkey = (offs, a.log_scaling_alpha, a.log_scaling_n_floor)
            tau_t = tau_cache.get(tkey) if tau_cache is not None else None
            if tau_t is None:
                tau_t = mx.concatenate([FD._full_tau(layer, o)[None] for o in offs])
                if tau_cache is not None:
                    tau_cache[tkey] = tau_t
            args.append(tau_t)
        args += [cs[0], cs[1]]
        q, kk, vv, pb, sinks, gate, w0, w1, nr = s1(*args)
        sdpa_gate = gate if gated else None
        if st is not None:
            pk = (st.ring, caches is not None)
            plan = self._plans.get(pk)
            if plan is None or plan.prev != st.state or getattr(plan, "prev_pages", None) != st.pages_sig:
                plan = self._plans[pk] = st.plan()
            C, Hkv = kk.shape[0], kk.shape[1]
            st.update(kk.reshape(C, Hkv, -1), vv.reshape(C, Hkv, -1), plan)
            a = layer.self_attn
            attn_out = RV.frz("sdpa", (id(layer), msg is not None), st.sdpa, q, pb, sinks, plan,
                              scale=a.scaling, window=a.sliding_window or 0, gate=sdpa_gate)
        elif use_sdpa_rows:
            ks, vs, meta = [], [], []
            for i, kv_i in enumerate(kvs):
                ka, va = kv_i.update_and_fetch(kk[i : i + 1], vv[i : i + 1])
                ks.append(ka)
                vs.append(va)
                meta.append(RS.row_meta(kv_i))
            a = layer.self_attn
            attn_out = RV.frz("sdpa", (id(layer), msg is not None), RS.sdpa_rows, q, ks, vs, pb, sinks, meta,
                              scale=a.scaling, window=a.sliding_window or 0, gate=sdpa_gate)
        else:
            outs = [FD._core(layer, q[i : i + 1], kk[i : i + 1], vv[i : i + 1], pb[i : i + 1], sinks[i : i + 1], kv_i)
                    for i, kv_i in enumerate(kvs)]
            attn_out = mx.concatenate(outs)
        args2 = [h, nr, attn_out, gate, w0, w1, cs[2], cs[3]]
        if bs is not None:
            args2.append(bs)
        if msg is not None:
            args2.append(msg)
        o, bs_new, *states = s2(*args2)
        conv_list[v] = states
        return o, bs_new

    def _layer_b1(self, layer, v, h, bs, msg, tau_cache=None, caches=None, conv=None, b1_stage=None, b1_unp_news=None):
        a = layer["self_attn"]
        sliding = "input_layernorm" in layer
        c_list = self.caches if caches is None else caches
        conv_list = self.conv if conv is None else conv
        kv = c_list[0][v][0]
        pending = self._b1_pending_ref if (caches is not None and caches is self._ref_caches) else self._b1_pending_kv
        if b1_stage is None and pending is not None and pending[v] is not None:
            kv.update_packed_b1(pending[v], FD._wpos)
            pending[v] = None
        cs = conv_list[v]
        r1 = RL.k1_pre_layer(layer, h, bs, msg)
        nr = r1["normed_residual"]
        tau_1d = None
        if not sliding:
            if b1_stage is not None and b1_stage[7] is not None:
                tau_1d = b1_stage[7]
            else:
                end = kv.offset
                tkey = (end, a.log_scaling_alpha, a.log_scaling_n_floor)
                tau_1d = tau_cache.get(tkey) if tau_cache is not None else None
                if tau_1d is None:
                    tau_1d = FD._full_tau(layer, end).reshape(-1)
                    if tau_cache is not None:
                        tau_cache[tkey] = tau_1d
        rel_w_bf = FD._rel_w_bf(a["rel_logits_proj"])
        if os.environ.get("LOOPLING_QKV_MID_B1", "0") not in ("0", ""):
            if msg is None:
                q, kv_new, pb, qkv_flat, w0, w1 = RP.qkv_mid_rel_b1(
                    layer, r1["q_in"], r1["k_in"], r1["v_in"], cs[0], cs[1], rel_w_bf, tau=tau_1d
                )
            else:
                q, kv_new, pb, qkv_flat, w0, w1 = RP.qkv_mid_rel_msg_b1(
                    layer, r1, msg, cs[0], cs[1], rel_w_bf, tau=tau_1d
                )
            gate_off, sink_off = 0, 3072
            gate_stride, sink_stride = 3072, 24
            if b1_stage is not None:
                is_ring, kv_layer_idx = self._kv_vmap[v]
                kv_buf = self._kv_ring if is_ring else self._kv_full
                kv_st = self._kv_ring_st if is_ring else self._kv_full_st
                meta_arr = b1_stage[0] if is_ring else b1_stage[1]
                s_bounds = b1_stage[2] if is_ring else b1_stage[3]
                tree_meta = b1_stage[4]
                rot = 1 if is_ring else 0
                max_size = b1_stage[5] if is_ring else 0
                cap = kv_buf.shape[3]
                attn_out = RS.sdpa_causal_w_packed(
                    q, kv_buf, kv_new, pb, qkv_flat, gate_off, sink_off,
                    0, 0, 0, rot, cap, max_size, 1,
                    scale=a.scaling, window=a.sliding_window or 0,
                    kv_layer_idx=kv_layer_idx, tree_meta=tree_meta, depths=(0,),
                    meta_arr=meta_arr, s_bounds=s_bounds, kv_stage=kv_st,
                    gate_stride=gate_stride, sink_stride=sink_stride,
                )
                b1_stage[6][v] = kv_new
            else:
                kv_buf, S, end, idx, rot, cap = kv.update_packed_b1(kv_new, FD._wpos)
                attn_out = RS.sdpa_b1_packed(
                    q, kv_buf, pb, qkv_flat, gate_off, sink_off, S, end, idx, rot, cap,
                    scale=a.scaling, window=a.sliding_window or 0,
                    gate_stride=gate_stride, sink_stride=sink_stride,
                )
        else:
            if msg is None:
                qkv_flat, qkv_lay = RP.qkv_rows(a, r1["q_in"], r1["k_in"], r1["v_in"], flat=True)
            elif (
                os.environ.get("LOOPLING_FUSE_K3B_QKV", "1") not in ("0", "")
                and int(r1["s_q"].shape[0]) <= int(os.environ.get("LOOPLING_FUSE_K3B_QKV_MAX_B", "4"))
            ):
                qkv_flat, qkv_lay = RP.qkv_msg_rows(layer, r1, msg, flat=True)
            else:
                gs = (layer["q_message_gate"], layer["k_message_gate"], layer["v_message_gate"])
                mixed = RV._gates(gs, (r1["s_q"], r1["s_k"], r1["s_v"]), msg, (0, 1, 2), r1["qkv_gate_in"])
                q_in, k_in, v_in = RL.k3b_norm_layer(layer, mixed)
                qkv_flat, qkv_lay = RP.qkv_rows(a, q_in, k_in, v_in, flat=True)
            if qkv_lay[2][0] == "gate" and qkv_lay[3][0] == "sinks":
                gate_off, sink_off = qkv_lay[2][1], qkv_lay[3][1]
                gate_stride, sink_stride = qkv_lay[2][2], qkv_lay[3][2]
            else:
                dm = {item[0]: (item[1], item[2]) for item in qkv_lay}
                gate_off, gate_stride = dm["gate"]
                sink_off, sink_stride = dm["sinks"]
            is_rot = hasattr(kv, "_idx")
            max_sz = int(kv.max_size) if is_rot else 0
            prev = int(kv.offset)
            S_now = min(prev + 1, max_sz) if is_rot else (prev + 1)
            if (
                b1_stage is None
                and FD._SDPA_MID_B1
                and kv._packed is not None
                and S_now < RS._TWO_PASS_N
            ):
                rot_wrap = 1 if (is_rot and prev >= max_sz) else 0
                slot_main = (0 if kv._idx == max_sz else int(kv._idx)) if rot_wrap else prev
                idx_now = (slot_main + 1) if is_rot else 0
                attn_out, kv_new, w0, w1 = RS.sdpa_mid_b1(
                    layer, qkv_flat, qkv_lay, kv._packed, cs[0], cs[1], rel_w_bf,
                    S_now, prev + 1, idx_now, 1 if is_rot else 0, int(kv._packed.shape[2]),
                    slot_main, rot_wrap, max_sz,
                    scale=a.scaling, window=a.sliding_window or 0, tau=tau_1d,
                )
                if pending is not None:
                    pending[v] = kv_new
                else:
                    kv.update_packed_b1(kv_new, FD._wpos)
            else:
                q, kv_new, pb, w0, w1 = RM.mid_rows_flat(
                    layer, qkv_flat, qkv_lay, cs[0], cs[1], tau=tau_1d, rel_w=rel_w_bf, packed_kv=True
                )
                if b1_stage is not None:
                    is_ring, kv_layer_idx = self._kv_vmap[v]
                    kv_buf = self._kv_ring if is_ring else self._kv_full
                    kv_st = self._kv_ring_st if is_ring else self._kv_full_st
                    meta_arr = b1_stage[0] if is_ring else b1_stage[1]
                    s_bounds = b1_stage[2] if is_ring else b1_stage[3]
                    tree_meta = b1_stage[4]
                    rot = 1 if is_ring else 0
                    max_size = b1_stage[5] if is_ring else 0
                    cap = kv_buf.shape[3]
                    attn_out = RS.sdpa_causal_w_packed(
                        q, kv_buf, kv_new, pb, qkv_flat, gate_off, sink_off,
                        0, 0, 0, rot, cap, max_size, 1,
                        scale=a.scaling, window=a.sliding_window or 0,
                        kv_layer_idx=kv_layer_idx, tree_meta=tree_meta, depths=(0,),
                        meta_arr=meta_arr, s_bounds=s_bounds, kv_stage=kv_st,
                        gate_stride=gate_stride, sink_stride=sink_stride,
                    )
                    b1_stage[6][v] = kv_new
                elif self._b1_unpacked_full and not sliding and b1_unp_news is not None and kv._packed is None:
                    fi = self._b1_full_unp_map[v]
                    s_pref = int(kv.offset)
                    attn_out = RS.sdpa_long_full_staged_unpacked(
                        q, kv._keys, kv._values, kv_new, pb, qkv_flat, gate_off, sink_off,
                        s_pref, s_pref + 1, int(kv._keys.shape[2]), int(kv._values.shape[2]), 1,
                        scale=a.scaling, kv_layer_idx=fi,
                        kv_stage=self._b1_full_unp_st, slot_main=self._b1_full_unp_slot0, st_len=self._b1_full_unp_len,
                        gate_stride=gate_stride, sink_stride=sink_stride,
                    )
                    b1_unp_news[v] = kv_new
                else:
                    kv_buf, S, end, idx, rot, cap = kv.update_packed_b1(kv_new, FD._wpos)
                    attn_out = RS.sdpa_b1_packed(
                        q, kv_buf, pb, qkv_flat, gate_off, sink_off, S, end, idx, rot, cap,
                        scale=a.scaling, window=a.sliding_window or 0,
                        gate_stride=gate_stride, sink_stride=sink_stride,
                    )
        op_m = a["o_proj"]
        op = mx.quantized_matmul(
            attn_out.reshape(1, 1, -1), op_m["weight"], scales=op_m["scales"], biases=op_m["biases"],
            transpose=True, group_size=64, bits=8, mode="affine",
        ).reshape(1, -1)
        r10 = RL.k10_post_layer(layer, op, h, nr, bs, cs[2], msg, debug=False)
        if layer.is_block_boundary:
            bs = [nr] if bs is None else (bs + [nr])
        if msg is None:
            x_mlp, mlp_mixed = r10["mlp_x"], None
        elif (
            os.environ.get("LOOPLING_FUSE_K3B_MLP", "1") not in ("0", "")
            and int(r10["mlp_src"].shape[0]) <= int(os.environ.get("LOOPLING_FUSE_K3B_MLP_MAX_B", "2"))
        ):
            x_mlp, mlp_mixed = None, RP.mlp_msg_part(layer, r10, msg)
        else:
            mixed = RV._gates((layer["mlp_message_gate"],), (r10["mlp_src"],), msg, (3,), (r10["mlp_gate_in"],))
            (x_mlp,) = RL.k3b_norm_layer(layer, mixed, mlp=True)
            mlp_mixed = None
        e = RMo.moe_rows(layer, x_mlp, r10["residual"], cs[3], route=RV.ROUTE, mlp_mixed=mlp_mixed)
        conv_list[v] = [w0, w1, r10["w2"], e["w3"]]
        return e["out"], bs

    def _layer_causal_w(
        self, layer, v, h, bs, msg, W, tau_cache=None, depths=None, conv_anc=None, tree_meta=None,
        depths_arr=None, tau_lut=None, tau_table=None, meta_arr=None, s_bounds=None,
        prev_moe=None, skip_combine=False, emit_h=False, use_stage=False, w_act=None,
    ):
        kv_vmap = self._kv_vmap if self._kv_stacked else None
        kv = self.caches[0][v][0]
        b1_unp = None
        if self._b1_unpacked_full and v in self._b1_full_unp_map:
            b1_unp = (self._b1_full_unp_st, self._b1_full_unp_map[v], self._b1_full_unp_slot0, self._b1_full_unp_len)
        return _layer_causal_w_body(
            self._cw_meta, kv_vmap, self._kv_ring, self._kv_full, self._kv_ring_st, self._kv_full_st,
            self.conv, layer, v, h, bs, msg, W, tau_cache=tau_cache, depths=depths, conv_anc=conv_anc,
            tree_meta=tree_meta, depths_arr=depths_arr, tau_lut=tau_lut, tau_table=tau_table,
            meta_arr=meta_arr, s_bounds=s_bounds, prev_moe=prev_moe, skip_combine=skip_combine,
            emit_h=emit_h, use_stage=use_stage, w_act=w_act, kv=kv, b1_unp_full=b1_unp,
        )

    def _layer_batch(
        self,
        layer,
        v,
        h,
        bs,
        msg,
        stage,
        step_metas,
        step_Ss,
        full_offs,
        st_len,
        kv_news,
        tau_cache,
        conv_list,
        prev_moe=None,
        skip_combine=False,
        tau_override=None,
        st_info_arr=None,
        stage_buf=None,
        ks_v=None,
        vs_v=None,
    ):
        return _layer_batch_body(
            self._batch_meta, layer, v, h, bs, msg, stage, step_metas, step_Ss, full_offs,
            st_len, kv_news, tau_cache, conv_list, prev_moe=prev_moe, skip_combine=skip_combine,
            tau_override=tau_override, st_info_arr=st_info_arr, stage_buf=stage_buf, ks_v=ks_v, vs_v=vs_v,
        )

    def _layer_batch_causal_w(
        self,
        layer,
        v,
        h,
        bs,
        msg,
        stage,
        step_metas,
        step_Ss,
        full_offs,
        kv_news,
        tau_cache,
        conv_list,
        conv_anc,
        C,
        W,
        prev_moe=None,
        skip_combine=False,
        tau_override=None,
        stage_buf=None,
        ks_v=None,
        vs_v=None,
    ):
        a, sliding, op_m, rel_proj, scale, window = self._batch_meta[id(layer)]
        cs = conv_list[v]
        if prev_moe is not None:
            r1 = RL.k1_from_down_causal_layer(layer, prev_moe, conv_anc, bs, msg, emit_h=True)
            h_before = r1["h"]
            conv_list[v - 1][3] = r1["w3"]
        else:
            h_before = h
            r1 = RL.k1_pre_layer(layer, h, bs, msg)
        nr = r1["normed_residual"]
        if msg is None:
            qkv_flat, qkv_lay = RP.qkv_rows(a, r1["q_in"], r1["k_in"], r1["v_in"], flat=True)
        elif (
            os.environ.get("LOOPLING_FUSE_K3B_QKV", "1") not in ("0", "")
            and int(r1["s_q"].shape[0]) <= int(os.environ.get("LOOPLING_FUSE_K3B_QKV_MAX_B", "4"))
        ):
            qkv_flat, qkv_lay = RP.qkv_msg_rows(layer, r1, msg, flat=True)
        else:
            gs = (layer["q_message_gate"], layer["k_message_gate"], layer["v_message_gate"])
            mixed = RV._gates(gs, (r1["s_q"], r1["s_k"], r1["s_v"]), msg, (0, 1, 2), r1["qkv_gate_in"])
            q_in, k_in, v_in = RL.k3b_norm_layer(layer, mixed)
            qkv_flat, qkv_lay = RP.qkv_rows(a, q_in, k_in, v_in, flat=True)
        if qkv_lay[2][0] == "gate" and qkv_lay[3][0] == "sinks":
            gate_off, sink_off = qkv_lay[2][1], qkv_lay[3][1]
            gate_stride, sink_stride = qkv_lay[2][2], qkv_lay[3][2]
        else:
            dm = {item[0]: (item[1], item[2]) for item in qkv_lay}
            gate_off, gate_stride = dm["gate"]
            sink_off, sink_stride = dm["sinks"]
        tau_1d = None
        if not sliding:
            if tau_override is not None:
                tau_1d = tau_override
            else:
                oarr, max_p = full_offs if full_offs is not None else stage.full_off_for_verify_visit(v, W)
                tkey = (id(stage), -1 if full_offs is not None else v, a.log_scaling_alpha, a.log_scaling_n_floor)
                tau_1d = tau_cache.get(tkey)
                if tau_1d is None:
                    tau_1d = _get_tau_table(layer, max_p)[oarr]
                    tau_cache[tkey] = tau_1d
        rel_w_bf = FD._rel_w_bf(rel_proj)
        B = C * W
        q, kv_new, pb, w0, w1 = RM.mid_causal_w_flat(
            layer, qkv_flat, qkv_lay, cs[0], cs[1], B, tau=tau_1d, rel_w=rel_w_bf, conv_anc=conv_anc,
        )
        kv_news[v] = kv_new
        attn_out = None
        if not sliding and isinstance(stage, _BatchKVStage):
            attn_out = _batch_full_attn_long_or_cascade(
                q, kv_new, pb, qkv_flat, gate_off, gate_stride, sink_off, sink_stride,
                stage, v, C, W, scale, ks_v=ks_v, vs_v=vs_v, stage_buf=stage_buf,
            )
        if attn_out is None:
            attn_out = RS.sdpa_multi_causal_w_staged(
                q,
                stage.ks_main[v] if ks_v is None else ks_v,
                stage.vs_main[v] if vs_v is None else vs_v,
                stage.buf if stage_buf is None else stage_buf,
                kv_new, pb, qkv_flat,
                gate_off, gate_stride, sink_off, sink_stride,
                step_metas[v], step_Ss[v], C, W,
                scale=scale, window=window, kv_layer_idx=v, scap=stage.scap,
            )
        op = RQ.qmv_rows(attn_out, op_m["weight"], op_m["scales"], op_m["biases"])
        r10 = RL.k10_post_layer(layer, op, h_before, nr, bs, cs[2], msg, debug=False, causal=True, conv_anc=conv_anc)
        if layer.is_block_boundary:
            bs = [nr] if bs is None else (bs + [nr])
        if msg is None:
            x_mlp, mlp_mixed = r10["mlp_x"], None
        elif (
            os.environ.get("LOOPLING_FUSE_K3B_MLP", "1") not in ("0", "")
            and int(r10["mlp_src"].shape[0]) <= int(os.environ.get("LOOPLING_FUSE_K3B_MLP_MAX_B", "2"))
        ):
            x_mlp, mlp_mixed = None, RP.mlp_msg_part(layer, r10, msg)
        else:
            mixed = RV._gates((layer["mlp_message_gate"],), (r10["mlp_src"],), msg, (3,), (r10["mlp_gate_in"],))
            (x_mlp,) = RL.k3b_norm_layer(layer, mixed, mlp=True)
            mlp_mixed = None
        e = RMo.moe_rows(layer, x_mlp, r10["residual"], cs[3], route=RV.ROUTE, causal=True, conv_anc=conv_anc, skip_combine=skip_combine, mlp_mixed=mlp_mixed)
        if skip_combine and "down" in e:
            conv_list[v] = [w0, w1, r10["w2"], None]
            return None, bs, e, h_before
        conv_list[v] = [w0, w1, r10["w2"], e["w3"]]
        return e["out"], bs, None, h_before

    def _get_b1_compiled(self, r_key, f_key, cap_r, cap_f, s_bounds_r, s_bounds_f, max_size_r, num_passes, cd_key, fuse_k3b):
        key = (r_key, f_key, cap_r, cap_f, num_passes, cd_key, fuse_k3b, self._b1_st_cap)
        ent = self._b1_compiled.get(key)
        if ent is not None:
            return ent
        tm = self.tm
        layers_seq = tm.layer_visits
        core = tm.args.loop_core
        cfg = tm.loopcd
        use_cd = cd_key is not None
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        nv = self.nv
        pre_n = sum(1 for v in range(0, core.start) if layers_seq[v].is_block_boundary)
        n_core = core.stop - core.start
        n_bs_core = pre_n + sum(1 for v in range(core.start, core.stop) if layers_seq[v].is_block_boundary)
        use_batch_msg = L._use_kernels() and getattr(tm, "message_input", None) is not None
        _, b1_tree_meta = RS._default_tree_meta(1)

        def make_chunk(v0, v1, *, is_first=False, is_last=False, use_msg=False, apply_cd=False):
            f_vs = [v for v in range(v0, v1) if not self._cw_meta[v][1]]
            f_pos = {v: i for i, v in enumerate(f_vs)}

            @mx.compile
            def fn(h_in, bs_in, h_aux, bs_aux, kv_r, kv_f, kv_st_r, kv_st_f, m_r, m_f, tau_1ds, *convs):
                old = (self._kv_ring, self._kv_full, self._kv_ring_st, self._kv_full_st, self.conv)
                try:
                    self._kv_ring = kv_r
                    self._kv_full = kv_f
                    self._kv_ring_st = kv_st_r
                    self._kv_full_st = kv_st_f
                    self.conv = [None] * nv
                    for j, v in enumerate(range(v0, v1)):
                        self.conv[v] = [convs[4 * j], convs[4 * j + 1], convs[4 * j + 2], convs[4 * j + 3]]
                    if is_first:
                        h = tm.embed(h_in.reshape(1, 1)).reshape(1, 2304)
                        bs = None
                        msgs = None
                    elif use_msg:
                        bank = (
                            mx.stack(list(bs_in[pre_n + 1 :]), axis=0),
                            L.unweighted_rms_norm(h_in.astype(mx.float32)),
                        )
                        if use_batch_msg:
                            msgs = IK.depth_attention_batch(bank[0], bank[1], tm.message_input.query, scale=2304 ** -0.5)
                        else:
                            msgs = [tm.message_input(bank, i) for i in range(v1 - v0)]
                        h = h_aux
                        bs = list(bs_aux)
                    else:
                        h = h_in
                        bs = list(bs_in) if bs_in else None
                        msgs = None
                        if apply_cd:
                            h, bs = loopcd.apply_hidden(
                                h, bs, h_aux, (list(bs_aux) if cfg.blend_bs else None), pre_n, cfg
                            )
                    kv_news = [None] * nv
                    for j, v in enumerate(range(v0, v1)):
                        msg_v = None if msgs is None else msgs[j]
                        tau_v = tau_1ds[f_pos[v]] if v in f_pos else None
                        b1_stage = (m_r, m_f, s_bounds_r, s_bounds_f, b1_tree_meta, max_size_r, kv_news, tau_v)
                        h, bs = self._layer_b1(layers_seq[v], v, h, bs, msg_v, b1_stage=b1_stage)
                    kvns = [kv_news[v] for v in range(v0, v1)]
                    cws = [x for v in range(v0, v1) for x in self.conv[v]]
                    if is_last:
                        h_out = RL.k1_pre(
                            h, bs, None, tm.final_input.query, [tm.norm.weight], sliding=True, eps_ln=tm.norm.eps
                        )["q_in"].reshape(1, 1, 2304)
                        return [h_out] + kvns + cws
                    return [h] + (list(bs) if bs is not None else []) + kvns + cws
                finally:
                    self._kv_ring, self._kv_full, self._kv_ring_st, self._kv_full_st, self.conv = old

            n_bs_out = 0 if is_last else (pre_n if is_first else n_bs_core)
            return fn, v0, v1, n_bs_out, f_vs

        chunks = [make_chunk(0, core.start, is_first=True)]
        for p in range(num_passes):
            v0 = core.start + p * n_core
            chunks.append(make_chunk(v0, v0 + n_core, use_msg=(p > 0)))
        coda_v0 = core.start + num_passes * n_core
        chunks.append(make_chunk(coda_v0, nv, is_last=True, apply_cd=use_cd))
        if len(self._b1_compiled) >= 64:
            self._b1_compiled.clear()
        ent = self._b1_compiled[key] = (chunks, pre_n, n_bs_core, use_cd, ref_pass, bool(cfg.blend_bs))
        return ent

    def _b1_hidden_pair(self, tokens, tap_set=None, tap_buf=None):
        tm = self.tm
        core = tm.args.loop_core
        layers = tm.layers
        num_passes = tm.args.loop_config["num_passes"]
        cfg = tm.loopcd
        use_cd = cfg.enabled_for(num_passes)
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        do_logits_cd = use_cd and cfg.mode == "logits"
        if do_logits_cd and self._ref_caches is None:
            raise RuntimeError("LoopCD-Logits shadow coda cache missing in BatchDecoder.")

        b1_staged = self._ensure_b1_kv_stage()
        if not b1_staged:
            self.unstack_kv()
        else:
            meta_r, meta_f = self.get_gpu_meta(staged=True)
            kv_r = self.caches[0][self._ring_visits[0]][0]
            kv_f = self.caches[0][self._full_visits[0]][0]
            max_size_r = int(kv_r.max_size)
            end0_r = int(kv_r.offset)
            end0_f = int(kv_f.offset)
            s_r = min(end0_r + 1, max_size_r)
            s_f = end0_f + 1
            s_bounds_r = (s_r, s_r)
            s_bounds_f = (s_f, s_f)
            r_key = self._regime_key(s_r, s_r)
            f_key = self._regime_key(s_f, s_f)

            use_compile = (
                tap_set is None
                and not do_logits_cd
                and os.environ.get("LOOPLING_COMPILE_B1", "1") not in ("0", "")
                and self.nv == len(tm.layer_visits)
                and r_key is not None
                and f_key is not None
            )
            if use_compile:
                cd_key = self._loopcd_cd_key(cfg, ref_pass, force_hidden=True) if (use_cd and cfg.mode == "hidden") else None
                fuse_k3b = (
                os.environ.get("LOOPLING_FUSE_K3B_QKV", "1") not in ("0", "")
                and self.C <= int(os.environ.get("LOOPLING_FUSE_K3B_QKV_MAX_B", "4"))
            )
                cap_r = int(self._kv_ring.shape[3])
                cap_f = int(self._kv_full.shape[3])
                chunks, pre_n, n_bs_core, cd_on, r_pass, blend_bs = self._get_b1_compiled(
                    r_key, f_key, cap_r, cap_f, s_bounds_r, s_bounds_f, max_size_r, num_passes, cd_key, fuse_k3b,
                )
                layers_seq = tm.layer_visits
                conv_flat = [x for cs in self.conv for x in cs]
                tau_cache = {}

                def _taus(fvs):
                    out = []
                    for v in fvs:
                        layer = layers_seq[v]
                        a = layer["self_attn"]
                        tk = (end0_f, a.log_scaling_alpha, a.log_scaling_n_floor)
                        tv = tau_cache.get(tk)
                        if tv is None:
                            tv = tau_cache[tk] = FD._full_tau(layer, end0_f).reshape(-1)
                        out.append(tv)
                    return out

                outs_all = []
                fn0, v0, v1, nbs_pre, fvs0 = chunks[0]
                o_pre = fn0(
                    tokens, [], [], [], self._kv_ring, self._kv_full, self._kv_ring_st, self._kv_full_st,
                    meta_r, meta_f, _taus(fvs0), *conv_flat[4 * v0 : 4 * v1],
                )
                mx.async_eval(o_pre[0])
                outs_all.append((o_pre, v1 - v0, nbs_pre))
                h_pre = o_pre[0]
                bs_pre = o_pre[1 : 1 + nbs_pre]
                o_prev = o_pre
                ref_o = None
                for p in range(num_passes):
                    fn_p, pv0, pv1, nbs_p, pfvs = chunks[1 + p]
                    if p == 0:
                        o_p = fn_p(
                            h_pre, bs_pre, [], [], self._kv_ring, self._kv_full, self._kv_ring_st, self._kv_full_st,
                            meta_r, meta_f, _taus(pfvs), *conv_flat[4 * pv0 : 4 * pv1],
                        )
                    else:
                        o_p = fn_p(
                            o_prev[0], o_prev[1 : 1 + n_bs_core], h_pre, bs_pre,
                            self._kv_ring, self._kv_full, self._kv_ring_st, self._kv_full_st,
                            meta_r, meta_f, _taus(pfvs), *conv_flat[4 * pv0 : 4 * pv1],
                        )
                    mx.async_eval(o_p[0])
                    outs_all.append((o_p, pv1 - pv0, nbs_p))
                    o_prev = o_p
                    if cd_on and p == r_pass:
                        ref_o = o_p
                fn_c, cv0, cv1, _, cfvs = chunks[-1]
                h_aux = ref_o[0] if cd_on else []
                bs_aux = ref_o[1 : 1 + n_bs_core] if (cd_on and blend_bs) else []
                o_c = fn_c(
                    o_prev[0], o_prev[1 : 1 + n_bs_core], h_aux, bs_aux,
                    self._kv_ring, self._kv_full, self._kv_ring_st, self._kv_full_st,
                    meta_r, meta_f, _taus(cfvs), *conv_flat[4 * cv0 : 4 * cv1],
                )
                outs_all.append((o_c, cv1 - cv0, 0))
                kv_news, cws = [], []
                for out_i, nvi, nbsi in outs_all:
                    head = 1 + nbsi
                    kv_news.extend(out_i[head : head + nvi])
                    cws.extend(out_i[head + nvi : head + nvi + 4 * nvi])
                for v in range(self.nv):
                    self.conv[v] = [cws[4 * v], cws[4 * v + 1], cws[4 * v + 2], cws[4 * v + 3]]
                self._commit_b1_kv_stage(kv_news)
                deps = [self._kv_ring_st, self._kv_full_st]
                h_out = o_c[0]
                if self._stage_dep_mode == "hidden":
                    h_out = mx.depends(h_out, deps)
                    self._pending_stage_deps = ()
                elif self._stage_dep_mode == "step":
                    self._pending_stage_deps = tuple(deps)
                else:
                    self._pending_stage_deps = ()
                return h_out, None

        self._v = 0
        tau_cache = {}
        ref_state = None
        h = tm.embed(tokens.reshape(1, 1)).reshape(1, 2304)
        use_batch_msg = L._use_kernels() and getattr(tm, "message_input", None) is not None
        kv_news = [None] * self.nv if b1_staged else None
        b1_unp_news = {} if (self._b1_unpacked_full and not b1_staged) else None
        b1_stage_base = None
        if b1_staged:
            _, b1_tree_meta = RS._default_tree_meta(1)
            b1_stage_base = (meta_r, meta_f, s_bounds_r, s_bounds_f, b1_tree_meta, max_size_r, kv_news, None)

        def run_b1(ls, h, bs, bank=None, rec=True):
            msgs = None
            if bank is not None and use_batch_msg:
                msgs = IK.depth_attention_batch(bank[0], bank[1], tm.message_input.query, scale=2304 ** -0.5)
            for i, layer in enumerate(ls):
                if tap_buf is not None and rec and self._v in tap_set:
                    tap_buf[self._v] = h.reshape(1, 2304)
                msg = None if bank is None else (msgs[i] if msgs is not None else tm.message_input(bank, i))
                h, bs = self._layer_b1(
                    layer, self._v, h, bs, msg, tau_cache, b1_stage=b1_stage_base, b1_unp_news=b1_unp_news
                )
                self._v += 1
            return h, bs

        prelude = run_b1(layers[: core.start], h, None)
        h, bs = prelude
        pre_n = len(prelude[1])
        for p in range(num_passes):
            bank = None
            if p > 0:
                bank = (
                    mx.stack(bs[pre_n + 1 :], axis=0),
                    L.unweighted_rms_norm(h.astype(mx.float32)),
                )
                h, bs = prelude
            h, bs = run_b1(layers[core.start : core.stop], h, bs, bank)
            if p == ref_pass:
                keep_bs = cfg.blend_bs or cfg.mode == "logits"
                ref_state = (h, list(bs) if keep_bs else None)
        if use_cd and cfg.mode == "hidden":
            h, bs = loopcd.apply_hidden(
                h, bs, ref_state[0], ref_state[1], pre_n, cfg, skip_mask=self._loopcd_skip_mask(cfg)
            )
        coda_layers = layers[core.stop :]
        h, bs = run_b1(coda_layers, h, bs)
        assert self._v == self.nv
        if b1_staged:
            self._commit_b1_kv_stage(kv_news)
        elif b1_unp_news:
            self._commit_b1_full_unpacked(b1_unp_news, 1)
        if tap_buf is not None and "o" in tap_set:
            tap_buf["o"] = h.reshape(1, 2304)
        h_out = RL.k1_pre(h, bs, None, tm.final_input.query, [tm.norm.weight], sliding=True, eps_ln=tm.norm.eps)[
            "q_in"
        ].reshape(1, 1, 2304)
        if tap_buf is not None:
            tap_buf["F"] = h_out.reshape(1, 2304)
        h_ref_out = None
        if do_logits_cd:
            hr, bsr = ref_state
            for j, layer in enumerate(coda_layers):
                hr, bsr = self._layer_b1(
                    layer, j, hr, bsr, None, tau_cache, caches=self._ref_caches, conv=self._ref_conv
                )
            h_ref_out = RL.k1_pre(
                hr, bsr, None, tm.final_input.query, [tm.norm.weight], sliding=True, eps_ln=tm.norm.eps
            )["q_in"].reshape(1, 1, 2304)
        if b1_staged:
            deps = [self._kv_ring_st, self._kv_full_st]
            if self._stage_dep_mode == "hidden":
                h_out = mx.depends(h_out, deps)
                self._pending_stage_deps = ()
            elif self._stage_dep_mode == "step":
                self._pending_stage_deps = tuple(deps)
            else:
                self._pending_stage_deps = ()
        return h_out, h_ref_out

    def _get_batch_compiled(self, regime_sig, step_Ss, num_passes, cd_key, fuse_ck1, fuse_k3b, scap, sdpa_mid=True):
        fuse_mlp = (
            os.environ.get("LOOPLING_FUSE_K3B_MLP", "1") not in ("0", "")
            and self.C <= int(os.environ.get("LOOPLING_FUSE_K3B_MLP_MAX_B", "2"))
        )
        key = (self.C, regime_sig, num_passes, cd_key, fuse_ck1, fuse_k3b, fuse_mlp, scap, bool(sdpa_mid), _env_compile_sig())
        ent = self._batch_compiled.get(key)
        if ent is not None:
            return ent
        tm = self.tm
        batch_meta = self._batch_meta
        layers_seq = tm.layer_visits
        core = tm.args.loop_core
        cfg = tm.loopcd
        use_cd = cd_key is not None
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        skip_mask = mx.array(cd_key[4], dtype=mx.bool_) if (use_cd and len(cd_key) > 4) else None
        C = self.C
        nv = self.nv
        pre_n = sum(1 for v in range(0, core.start) if layers_seq[v].is_block_boundary)
        n_core = core.stop - core.start
        n_bs_core = pre_n + sum(1 for v in range(core.start, core.stop) if layers_seq[v].is_block_boundary)
        use_batch_msg = L._use_kernels() and getattr(tm, "message_input", None) is not None
        stage_scap = _ScapOnly(scap)

        def make_chunk(v0, v1, *, is_first=False, is_last=False, use_msg=False, apply_cd=False):
            n_v = v1 - v0
            f_vs = [v for v in range(v0, v1) if not batch_meta[id(layers_seq[v])][1]]
            f_pos = {v: i for i, v in enumerate(f_vs)}
            ss_slice = [step_Ss[v] for v in range(v0, v1)]

            @mx.compile
            def fn(h_in, bs_in, h_aux, bs_aux, stage_buf, metas_in, st_infos_in, taus_in, ks_flat, vs_flat, *convs):
                conv_local = [None] * nv
                for j, v in enumerate(range(v0, v1)):
                    conv_local[v] = [convs[4 * j], convs[4 * j + 1], convs[4 * j + 2], convs[4 * j + 3]]
                if is_first:
                    h = tm.embed(h_in.reshape(C, 1)).reshape(C, 2304)
                    bs = None
                    msgs = None
                    bank4 = None
                elif use_msg:
                    bank = (
                        mx.stack(list(bs_in[pre_n + 1 :]), axis=0),
                        L.unweighted_rms_norm(h_in.astype(mx.float32)),
                    )
                    if use_batch_msg:
                        msgs = IK.depth_attention_batch(bank[0], bank[1], tm.message_input.query, scale=2304 ** -0.5)
                        bank4 = None
                    else:
                        msgs = None
                        bank4 = (bank[0][:, :, None, :], bank[1][:, None, :])
                    h = h_aux
                    bs = list(bs_aux)
                else:
                    h = h_in
                    bs = list(bs_in) if bs_in else None
                    msgs = None
                    bank4 = None
                    if apply_cd:
                        h, bs = loopcd.apply_hidden(
                            h, bs, h_aux, (list(bs_aux) if cfg.blend_bs else None), pre_n, cfg, skip_mask=skip_mask
                        )
                kv_news = [None] * nv
                metas_map = [None] * nv
                ss_map = [None] * nv
                for j, v in enumerate(range(v0, v1)):
                    metas_map[v] = metas_in[j]
                    ss_map[v] = ss_slice[j]
                prev_moe = None
                last_j = n_v - 1
                for j, v in enumerate(range(v0, v1)):
                    layer = layers_seq[v]
                    if not use_msg:
                        msg_v = None
                    elif msgs is not None:
                        msg_v = msgs[j]
                    else:
                        msg_v = tm.message_input(bank4, j).reshape(C, 4, 2304)
                    next_gpar1 = j < last_j and (isinstance(layers_seq[v + 1], L.SlidingDecoderLayer) or use_msg)
                    tau_v = taus_in[f_pos[v]] if v in f_pos else None
                    ks_v = ks_flat[j * C : (j + 1) * C]
                    vs_v = vs_flat[j * C : (j + 1) * C]
                    h, bs, prev_moe = _layer_batch_body(
                        batch_meta, layer, v, h, bs, msg_v,
                        stage_scap, metas_map, ss_map, None, 0, kv_news,
                        {}, conv_local, prev_moe=prev_moe, skip_combine=(fuse_ck1 and next_gpar1),
                        tau_override=tau_v, st_info_arr=st_infos_in[j], stage_buf=stage_buf,
                        ks_v=ks_v, vs_v=vs_v,
                    )
                kvns = [kv_news[v] for v in range(v0, v1)]
                cws = [x for v in range(v0, v1) for x in conv_local[v]]
                if is_last:
                    h_out = RL.k1_pre(
                        h, bs, None, tm.final_input.query, [tm.norm.weight], sliding=True, eps_ln=tm.norm.eps
                    )["q_in"].reshape(C, 1, 2304)
                    return [h_out] + kvns + cws
                return [h] + (list(bs) if bs is not None else []) + kvns + cws

            n_bs_out = 0 if is_last else (pre_n if is_first else n_bs_core)
            return fn, v0, v1, n_bs_out, f_vs

        chunks = [make_chunk(0, core.start, is_first=True)]
        for p in range(num_passes):
            v0 = core.start + p * n_core
            chunks.append(make_chunk(v0, v0 + n_core, use_msg=(p > 0)))
        coda_v0 = core.start + num_passes * n_core
        chunks.append(make_chunk(coda_v0, nv, is_last=True, apply_cd=use_cd))
        if len(self._batch_compiled) >= 64:
            self._batch_compiled.clear()
        ent = self._batch_compiled[key] = (chunks, pre_n, n_bs_core, use_cd, ref_pass, bool(cfg.blend_bs))
        return ent

    def _get_batch_compiled_one(self, regime_sig, step_Ss, num_passes, cd_key, fuse_ck1, fuse_k3b, scap, sdpa_mid=True):
        fuse_mlp = (
            os.environ.get("LOOPLING_FUSE_K3B_MLP", "1") not in ("0", "")
            and self.C <= int(os.environ.get("LOOPLING_FUSE_K3B_MLP_MAX_B", "2"))
        )
        key = ("one", self.C, regime_sig, num_passes, cd_key, fuse_ck1, fuse_k3b, fuse_mlp, scap, bool(sdpa_mid), _env_compile_sig())
        ent = self._batch_compiled.get(key)
        if ent is not None:
            return ent
        tm = self.tm
        batch_meta = self._batch_meta
        layers_seq = tm.layer_visits
        core = tm.args.loop_core
        cfg = tm.loopcd
        use_cd = cd_key is not None
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        skip_mask = mx.array(cd_key[4], dtype=mx.bool_) if (use_cd and len(cd_key) > 4) else None
        blend_bs = bool(cfg.blend_bs)
        C = self.C
        nv = self.nv
        pre_n = sum(1 for v in range(0, core.start) if layers_seq[v].is_block_boundary)
        n_core = core.stop - core.start
        use_batch_msg = L._use_kernels() and getattr(tm, "message_input", None) is not None
        stage_scap = _ScapOnly(scap)
        f_vs = [v for v in range(nv) if not batch_meta[id(layers_seq[v])][1]]
        f_pos = {v: i for i, v in enumerate(f_vs)}
        ss_all = list(step_Ss)
        ranges = [(0, core.start, False, False)]
        for p in range(num_passes):
            v0 = core.start + p * n_core
            ranges.append((v0, v0 + n_core, p > 0, False))
        coda_v0 = core.start + num_passes * n_core
        ranges.append((coda_v0, nv, False, use_cd))

        @mx.compile
        def fn(tokens, stage_buf, metas_in, st_infos_in, taus_in, ks_flat, vs_flat, *convs):
            conv_local = [[convs[4 * v], convs[4 * v + 1], convs[4 * v + 2], convs[4 * v + 3]] for v in range(nv)]
            kv_news = [None] * nv
            h = tm.embed(tokens.reshape(C, 1)).reshape(C, 2304)
            bs = None
            h_pre = None
            bs_pre = None
            ref_h = None
            ref_bs = None
            for idx_r, (v0, v1, use_msg, apply_cd) in enumerate(ranges):
                if use_msg:
                    bank = (
                        mx.stack(list(bs[pre_n + 1 :]), axis=0),
                        L.unweighted_rms_norm(h.astype(mx.float32)),
                    )
                    if use_batch_msg:
                        msgs = IK.depth_attention_batch(bank[0], bank[1], tm.message_input.query, scale=2304 ** -0.5)
                        bank4 = None
                    else:
                        msgs = None
                        bank4 = (bank[0][:, :, None, :], bank[1][:, None, :])
                    h = h_pre
                    bs = list(bs_pre)
                else:
                    msgs = None
                    bank4 = None
                    if apply_cd:
                        h, bs = loopcd.apply_hidden(h, bs, ref_h, ref_bs, pre_n, cfg, skip_mask=skip_mask)
                prev_moe = None
                last_v = v1 - 1
                for j, v in enumerate(range(v0, v1)):
                    layer = layers_seq[v]
                    if not use_msg:
                        msg_v = None
                    elif msgs is not None:
                        msg_v = msgs[j]
                    else:
                        msg_v = tm.message_input(bank4, j).reshape(C, 4, 2304)
                    next_gpar1 = v < last_v and (isinstance(layers_seq[v + 1], L.SlidingDecoderLayer) or use_msg)
                    tau_v = taus_in[f_pos[v]] if v in f_pos else None
                    ks_v = ks_flat[v * C : (v + 1) * C]
                    vs_v = vs_flat[v * C : (v + 1) * C]
                    h, bs, prev_moe = _layer_batch_body(
                        batch_meta, layer, v, h, bs, msg_v,
                        stage_scap, metas_in, ss_all, None, 0, kv_news,
                        {}, conv_local, prev_moe=prev_moe, skip_combine=(fuse_ck1 and next_gpar1),
                        tau_override=tau_v, st_info_arr=st_infos_in[v], stage_buf=stage_buf,
                        ks_v=ks_v, vs_v=vs_v,
                    )
                if idx_r == 0:
                    h_pre = h
                    bs_pre = list(bs)
                elif use_cd and (idx_r - 1) == ref_pass:
                    ref_h = h
                    ref_bs = list(bs) if blend_bs else None
            h_out = RL.k1_pre(
                h, bs, None, tm.final_input.query, [tm.norm.weight], sliding=True, eps_ln=tm.norm.eps
            )["q_in"].reshape(C, 1, 2304)
            cws = [x for v in range(nv) for x in conv_local[v]]
            return [h_out] + kv_news + cws

        if len(self._batch_compiled) >= 64:
            self._batch_compiled.clear()
        ent = self._batch_compiled[key] = (fn, f_vs)
        return ent

    def _get_batch_cw_compiled(self, W, regime_sig, step_Ss, num_passes, cd_key, fuse_ck1, fuse_k3b, scap, tap_tuple, return_h_out):
        key = (self.C, int(W), regime_sig, num_passes, cd_key, fuse_ck1, fuse_k3b, scap, tap_tuple, bool(return_h_out))
        cw_cache = getattr(self, "_batch_cw_compiled", None)
        if cw_cache is None:
            cw_cache = self._batch_cw_compiled = {}
        ent = cw_cache.get(key)
        if ent is not None:
            return ent
        tm = self.tm
        m = self.model
        layers_seq = tm.layer_visits
        core = tm.args.loop_core
        cfg = tm.loopcd
        use_cd = cd_key is not None
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        skip_mask = mx.repeat(mx.array(cd_key[4], dtype=mx.bool_), int(W)) if (use_cd and len(cd_key) > 4) else None
        C = self.C
        B = C * int(W)
        nv = self.nv
        pre_n = sum(1 for v in range(0, core.start) if layers_seq[v].is_block_boundary)
        n_core = core.stop - core.start
        n_bs_core = pre_n + sum(1 for v in range(core.start, core.stop) if layers_seq[v].is_block_boundary)
        use_batch_msg = L._use_kernels() and getattr(tm, "message_input", None) is not None
        stage = self._stage_main
        tap_set = set(tap_tuple) if tap_tuple is not None else set()
        has_o = "o" in tap_set
        has_F = tap_tuple is not None

        def make_chunk(v0, v1, *, is_first=False, is_last=False, use_msg=False, apply_cd=False):
            n_v = v1 - v0
            t_in_chunk = frozenset(v for v in range(v0, v1) if v in tap_set)
            f_vs = [v for v in range(v0, v1) if not self._batch_meta[id(layers_seq[v])][1]]
            f_pos = {v: i for i, v in enumerate(f_vs)}
            ss_slice = [step_Ss[v] for v in range(v0, v1)]

            @mx.compile
            def fn(h_in, bs_in, h_aux, bs_aux, stage_buf, conv_anc, metas_in, taus_in, ks_flat, vs_flat, *convs):
                conv_local = [None] * nv
                for j, v in enumerate(range(v0, v1)):
                    conv_local[v] = [convs[4 * j], convs[4 * j + 1], convs[4 * j + 2], convs[4 * j + 3]]
                if is_first:
                    h = tm.embed(h_in.reshape(B, 1)).reshape(B, 2304)
                    bs = None
                    msgs = None
                    bank4 = None
                elif use_msg:
                    bank = (
                        mx.stack(list(bs_in[pre_n + 1 :]), axis=0),
                        L.unweighted_rms_norm(h_in.astype(mx.float32)),
                    )
                    if use_batch_msg:
                        msgs = IK.depth_attention_batch(bank[0], bank[1], tm.message_input.query, scale=2304 ** -0.5)
                        bank4 = None
                    else:
                        msgs = None
                        bank4 = (bank[0][:, :, None, :], bank[1][:, None, :])
                    h = h_aux
                    bs = list(bs_aux)
                else:
                    h = h_in
                    bs = list(bs_in) if bs_in else None
                    msgs = None
                    bank4 = None
                    if apply_cd:
                        h, bs = loopcd.apply_hidden(
                            h, bs, h_aux, (list(bs_aux) if cfg.blend_bs else None), pre_n, cfg, skip_mask=skip_mask
                        )
                kv_news = [None] * nv
                metas_map = [None] * nv
                ss_map = [None] * nv
                for j, v in enumerate(range(v0, v1)):
                    metas_map[v] = metas_in[j]
                    ss_map[v] = ss_slice[j]
                prev_moe = None
                last_j = n_v - 1
                taps_out = []
                for j, v in enumerate(range(v0, v1)):
                    layer = layers_seq[v]
                    if not use_msg:
                        msg_v = None
                    elif msgs is not None:
                        msg_v = msgs[j]
                    else:
                        msg_v = tm.message_input(bank4, j).reshape(B, 4, 2304)
                    next_gpar1 = j < last_j and (isinstance(layers_seq[v + 1], L.SlidingDecoderLayer) or use_msg)
                    tau_v = taus_in[f_pos[v]] if v in f_pos else None
                    ks_v = ks_flat[j * C : (j + 1) * C]
                    vs_v = vs_flat[j * C : (j + 1) * C]
                    h, bs, prev_moe, h_before_v = self._layer_batch_causal_w(
                        layer, v, h, bs, msg_v,
                        stage, metas_map, ss_map, None, kv_news,
                        {}, conv_local, conv_anc, C, W, prev_moe=prev_moe, skip_combine=(fuse_ck1 and next_gpar1),
                        tau_override=tau_v, stage_buf=stage_buf, ks_v=ks_v, vs_v=vs_v,
                    )
                    if v in t_in_chunk:
                        taps_out.append(h_before_v)
                kvns = [kv_news[v] for v in range(v0, v1)]
                cws = [x for v in range(v0, v1) for x in conv_local[v]]
                if is_last:
                    if has_o:
                        taps_out.append(h)
                    h_out = RL.k1_pre(
                        h, bs, None, tm.final_input.query, [tm.norm.weight], sliding=True, eps_ln=tm.norm.eps
                    )["q_in"].reshape(B, 2304)
                    if has_F:
                        taps_out.append(h_out)
                    if return_h_out:
                        return [h_out] + kvns + cws + taps_out
                    lm = m.lm_head
                    text = m.args.text
                    if "head" in _FUSED and "head" not in _PLAIN and FD._is_aff8(lm) and h_out.dtype == mx.bfloat16:
                        nt = RQ.qmv_softcap_argmax_rows(
                            h_out, lm["weight"], lm["scales"], lm["biases"],
                            text.final_logit_softcapping, text.unpadded_vocab_size,
                        )
                    else:
                        lg = RQ.qmv_rows(h_out, lm["weight"], lm["scales"], lm["biases"])
                        nt = RH.softcap_argmax_rows(lg, text.final_logit_softcapping, text.unpadded_vocab_size)
                    return [nt] + kvns + cws + taps_out
                return [h] + (list(bs) if bs is not None else []) + kvns + cws + taps_out

            n_bs_out = 0 if is_last else (pre_n if is_first else n_bs_core)
            t_keys = [v for v in range(v0, v1) if v in tap_set]
            if is_last:
                if has_o:
                    t_keys.append("o")
                if has_F:
                    t_keys.append("F")
            return fn, v0, v1, n_bs_out, f_vs, t_keys

        chunks = [make_chunk(0, core.start, is_first=True)]
        for p in range(num_passes):
            v0 = core.start + p * n_core
            chunks.append(make_chunk(v0, v0 + n_core, use_msg=(p > 0)))
        coda_v0 = core.start + num_passes * n_core
        chunks.append(make_chunk(coda_v0, nv, is_last=True, apply_cd=use_cd))
        if len(cw_cache) >= 64:
            cw_cache.clear()
        ent = cw_cache[key] = (chunks, pre_n, n_bs_core, use_cd, ref_pass, bool(cfg.blend_bs))
        return ent

    def verify_batch_w(self, tokens_cw, W: int, *, tap_visits=None, sampler=None, return_h_out: bool = False):
        """Non-mutating multi-stream causal W-token verify for C streams (`2<=C<=8, 2<=W<=8, C*W<=16`)."""
        assert self._batch_fast and 2 <= self.C <= 8 and 2 <= W <= 8 and self.C * W <= 16
        tm = self.tm
        num_passes = tm.args.loop_config["num_passes"]
        cfg = tm.loopcd
        assert not (cfg.enabled_for(num_passes) and cfg.mode != "hidden")
        assert self.nv == len(tm.layer_visits)
        B = self.C * int(W)
        tokens_cw = tokens_cw.reshape(B)
        stage = self._stage_main
        stage.maybe_flush_before_verify(W)
        step_metas, step_Ss, full_offs = stage.verify_view(W)

        is_greedy = sampler is None or getattr(sampler, "_sampler_key", None) == ("greedy",)
        want_h = bool(return_h_out) or (not is_greedy)
        tap_set = set(tap_visits) if tap_visits is not None else set()
        conv_anc = build_multi_chain_conv_anc(self.C, W)

        if stage.needs_long_or_cascade(W):
            # Direct uncompiled path for long-context Split-K / Shared-Prefix Cascade (0 JIT recompiles on split changes).
            core = tm.args.loop_core
            layers = tm.layers
            use_cd = cfg.enabled_for(num_passes)
            ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
            skip_mask = (
                mx.repeat(self._loopcd_skip_mask(cfg), int(W))
                if (use_cd and self._loopcd_skip_mask(cfg) is not None)
                else None
            )
            fuse_ck1 = os.environ.get("LOOPLING_FUSE_COMB_K1_BATCH", "0") not in ("0", "")
            use_batch_msg = L._use_kernels() and getattr(tm, "message_input", None) is not None
            conv_local = [list(cs) for cs in self.conv]
            kv_news = [None] * self.nv
            tap_buf = {} if tap_visits is not None else None
            tau_cache = {}
            v_idx = 0
            h = tm.embed(tokens_cw.reshape(B, 1)).reshape(B, 2304)

            def run_bcw(ls, h, bs, bank=None):
                nonlocal v_idx
                msgs = None
                bank4 = None
                if bank is not None:
                    if use_batch_msg:
                        msgs = IK.depth_attention_batch(bank[0], bank[1], tm.message_input.query, scale=2304 ** -0.5)
                    else:
                        bank4 = (bank[0][:, :, None, :], bank[1][:, None, :])
                prev_moe = None
                last_i = len(ls) - 1
                for i, layer in enumerate(ls):
                    if bank is None:
                        msg_v = None
                    elif msgs is not None:
                        msg_v = msgs[i]
                    else:
                        msg_v = tm.message_input(bank4, i).reshape(B, 4, 2304)
                    next_gpar1 = i < last_i and (isinstance(ls[i + 1], L.SlidingDecoderLayer) or bank is not None)
                    h, bs, prev_moe, h_before_v = self._layer_batch_causal_w(
                        layer, v_idx, h, bs, msg_v,
                        stage, step_metas, step_Ss, full_offs, kv_news,
                        tau_cache, conv_local, conv_anc, self.C, int(W),
                        prev_moe=prev_moe, skip_combine=(fuse_ck1 and next_gpar1),
                    )
                    if tap_buf is not None and v_idx in tap_set:
                        tap_buf[v_idx] = h_before_v
                    v_idx += 1
                return h, bs

            prelude = run_bcw(layers[: core.start], h, None)
            h, bs = prelude
            mx.async_eval(h)
            pre_n = len(prelude[1])
            ref_state = None
            for p in range(num_passes):
                bank = None
                if p > 0:
                    bank = (
                        mx.stack(bs[pre_n + 1 :], axis=0),
                        L.unweighted_rms_norm(h.astype(mx.float32)),
                    )
                    h, bs = prelude
                h, bs = run_bcw(layers[core.start : core.stop], h, bs, bank)
                mx.async_eval(h)
                if use_cd and p == ref_pass:
                    ref_state = (h, list(bs) if cfg.blend_bs else None)
            if use_cd:
                h, bs = loopcd.apply_hidden(
                    h, bs, ref_state[0], ref_state[1], pre_n, cfg, skip_mask=skip_mask
                )
            h, bs = run_bcw(layers[core.stop :], h, bs)
            assert v_idx == self.nv
            if tap_buf is not None and "o" in tap_set:
                tap_buf["o"] = h
            h_out = RL.k1_pre(
                h, bs, None, tm.final_input.query, [tm.norm.weight], sliding=True, eps_ln=tm.norm.eps
            )["q_in"].reshape(B, 2304)
            if tap_buf is not None:
                tap_buf["F"] = h_out
            conv_ws = [tuple(cs) for cs in conv_local]
            if return_h_out:
                return h_out, kv_news, conv_ws, tap_buf
            if not is_greedy:
                ret = self._sample_verify_tokens(h_out, sampler)
            else:
                lm = self.model.lm_head
                text = self.model.args.text
                if "head" in _FUSED and "head" not in _PLAIN and FD._is_aff8(lm) and h_out.dtype == mx.bfloat16:
                    ret = RQ.qmv_softcap_argmax_rows(
                        h_out, lm["weight"], lm["scales"], lm["biases"],
                        text.final_logit_softcapping, text.unpadded_vocab_size,
                    )
                else:
                    lg = RQ.qmv_rows(h_out, lm["weight"], lm["scales"], lm["biases"])
                    ret = RH.softcap_argmax_rows(lg, text.final_logit_softcapping, text.unpadded_vocab_size)
            return ret.reshape(self.C, int(W)), kv_news, conv_ws, tap_buf

        regime_sig = tuple(
            (max(Ss) >= RS._TWO_PASS_N, min(Ss) >= RS._TWO_PASS_N, sum(1 for s in Ss if s >= RS._TWO_PASS_N), max(Ss) >= 256)
            for Ss in step_Ss
        )
        use_cd = cfg.enabled_for(num_passes)
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        cd_key = self._loopcd_cd_key(cfg, ref_pass) if use_cd else None
        fuse_ck1 = os.environ.get("LOOPLING_FUSE_COMB_K1_BATCH", "0") not in ("0", "")
        fuse_k3b = (
            os.environ.get("LOOPLING_FUSE_K3B_QKV", "1") not in ("0", "")
            and B <= int(os.environ.get("LOOPLING_FUSE_K3B_QKV_MAX_B", "4"))
        )
        tap_tuple = (
            tuple(sorted(tap_set, key=lambda k: (0, k) if isinstance(k, int) else (1, str(k))))
            if tap_visits is not None
            else None
        )
        chunks, pre_n, n_bs_core, cd_on, r_pass, blend_bs = self._get_batch_cw_compiled(
            W, regime_sig, step_Ss, num_passes, cd_key, fuse_ck1, fuse_k3b, stage.scap, tap_tuple, want_h,
        )
        layers_seq = tm.layer_visits
        conv_flat = [x for cs in self.conv for x in cs]
        tau_cache = {}

        def _taus_b(fvs):
            out = []
            for v in fvs:
                layer = layers_seq[v]
                a = layer["self_attn"]
                oarr, max_p = full_offs if full_offs is not None else stage.full_off_for_verify_visit(v, W)
                tk = (-1 if full_offs is not None else v, a.log_scaling_alpha, a.log_scaling_n_floor)
                tv = tau_cache.get(tk)
                if tv is None:
                    tv = tau_cache[tk] = _get_tau_table(layer, max_p)[oarr]
                out.append(tv)
            return out

        def _kv_flat(v0, v1):
            ks = [x for v in range(v0, v1) for x in stage.ks_main[v]]
            vs = [x for v in range(v0, v1) for x in stage.vs_main[v]]
            return ks, vs

        outs_all = []
        fn0, v0, v1, nbs_pre, fvs0, tks0 = chunks[0]
        ks0, vs0 = _kv_flat(v0, v1)
        o_pre = fn0(
            tokens_cw, [], [], [], stage.buf, conv_anc, step_metas[v0:v1], _taus_b(fvs0),
            ks0, vs0, *conv_flat[4 * v0 : 4 * v1],
        )
        mx.async_eval(o_pre[0])
        outs_all.append((o_pre, v1 - v0, nbs_pre, tks0))
        h_pre = o_pre[0]
        bs_pre = o_pre[1 : 1 + nbs_pre]
        o_prev = o_pre
        ref_o = None
        for p in range(num_passes):
            fn_p, pv0, pv1, nbs_p, pfvs, ptks = chunks[1 + p]
            ksp, vsp = _kv_flat(pv0, pv1)
            if p == 0:
                o_p = fn_p(
                    h_pre, bs_pre, [], [], stage.buf, conv_anc, step_metas[pv0:pv1], _taus_b(pfvs),
                    ksp, vsp, *conv_flat[4 * pv0 : 4 * pv1],
                )
            else:
                o_p = fn_p(
                    o_prev[0], o_prev[1 : 1 + n_bs_core], h_pre, bs_pre,
                    stage.buf, conv_anc, step_metas[pv0:pv1], _taus_b(pfvs),
                    ksp, vsp, *conv_flat[4 * pv0 : 4 * pv1],
                )
            mx.async_eval(o_p[0])
            outs_all.append((o_p, pv1 - pv0, nbs_p, ptks))
            o_prev = o_p
            if cd_on and p == r_pass:
                ref_o = o_p
        fn_c, cv0, cv1, _, cfvs, ctks = chunks[-1]
        ksc, vsc = _kv_flat(cv0, cv1)
        h_aux = ref_o[0] if cd_on else []
        bs_aux = ref_o[1 : 1 + n_bs_core] if (cd_on and blend_bs) else []
        o_c = fn_c(
            o_prev[0], o_prev[1 : 1 + n_bs_core], h_aux, bs_aux,
            stage.buf, conv_anc, step_metas[cv0:cv1], _taus_b(cfvs),
            ksc, vsc, *conv_flat[4 * cv0 : 4 * cv1],
        )
        outs_all.append((o_c, cv1 - cv0, 0, ctks))

        kv_news, conv_ws = [], []
        tap_buf = {} if tap_visits is not None else None
        for out_i, nvi, nbsi, tksi in outs_all:
            head = 1 + nbsi
            ncw = 4 * nvi
            kv_news.extend(out_i[head : head + nvi])
            conv_ws.extend(out_i[head + nvi : head + nvi + ncw])
            if tap_buf is not None:
                for k, arr in zip(tksi, out_i[head + nvi + ncw :]):
                    tap_buf[k] = arr
        conv_ws = [tuple(conv_ws[4 * j : 4 * j + 4]) for j in range(self.nv)]
        ret = o_c[0]
        if return_h_out:
            return ret, kv_news, conv_ws, tap_buf
        if not is_greedy:
            ret = self._sample_verify_tokens(ret, sampler)
        return ret.reshape(self.C, int(W)), kv_news, conv_ws, tap_buf

    def commit_batch_w(self, kv_news, conv_ws, keeps, W: int):
        """Commit variable per-stream `keeps[c]` (`1..W`) verified tokens into `_BatchKVStage` + conv."""
        assert self._batch_fast and len(keeps) == self.C
        self._stage_main.commit_verify_w(kv_news, keeps, W)
        idx = mx.array([c * int(W) + (int(keeps[c]) - 1) for c in range(self.C)], mx.int32)
        for v in range(self.nv):
            self.conv[v] = [cw[idx] for cw in conv_ws[v]]

    def _batch_hidden_pair(self, tokens):
        tm = self.tm
        core = tm.args.loop_core
        layers = tm.layers
        num_passes = tm.args.loop_config["num_passes"]
        cfg = tm.loopcd
        use_cd = cfg.enabled_for(num_passes)
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        do_logits_cd = use_cd and cfg.mode == "logits"
        if do_logits_cd and self._stage_ref is None:
            raise RuntimeError("LoopCD-Logits shadow coda cache missing in BatchDecoder.")

        self._stage_main.maybe_flush_before_step()
        if do_logits_cd:
            self._stage_ref.maybe_flush_before_step()
        step_metas, step_Ss, full_offs, st_len = self._stage_main.step_view()
        fuse_ck1 = os.environ.get("LOOPLING_FUSE_COMB_K1_BATCH", "0") not in ("0", "")

        if (
            self._sdpa_staged_has_st_info
            and not do_logits_cd
            and not self._stage_main.needs_long_or_cascade(1)
            and os.environ.get("LOOPLING_COMPILE_BATCH", "1") not in ("0", "")
            and self.nv == len(tm.layer_visits)
        ):
            stage = self._stage_main
            sdpa_mid = os.environ.get("LOOPLING_SDPA_MID_BATCH", "1") not in ("0", "")
            regime_sig = tuple(
                (
                    max(Ss) >= RS._TWO_PASS_N,
                    min(Ss) >= RS._TWO_PASS_N,
                    sum(1 for s in Ss if s >= RS._TWO_PASS_N),
                    RS._blocks(max(Ss), 3) if max(Ss) >= RS._TWO_PASS_N else 0,
                )
                for Ss in step_Ss
            )
            cd_key = self._loopcd_cd_key(cfg, ref_pass, force_hidden=True) if (use_cd and cfg.mode == "hidden") else None
            fuse_k3b = (
                os.environ.get("LOOPLING_FUSE_K3B_QKV", "1") not in ("0", "")
                and self.C <= int(os.environ.get("LOOPLING_FUSE_K3B_QKV_MAX_B", "4"))
            )
            layers_seq = tm.layer_visits
            conv_flat = [x for cs in self.conv for x in cs]
            tau_cache = {}
            st_key = (self.nv, stage.scap, st_len)
            st_infos = _batch_st_infos_cache.get(st_key)
            if st_infos is None:
                if len(_batch_st_infos_cache) >= 512:
                    _batch_st_infos_cache.clear()
                st_infos = []
                for v in range(self.nv):
                    sk = (v, stage.scap, st_len)
                    si = RS._ST_INFO_CACHE.get(sk)
                    if si is None:
                        if len(RS._ST_INFO_CACHE) >= 4096:
                            RS._ST_INFO_CACHE.clear()
                        si = RS._ST_INFO_CACHE[sk] = mx.array(sk, mx.int32)
                    st_infos.append(si)
                _batch_st_infos_cache[st_key] = st_infos

            def _taus_b(fvs):
                out = []
                for v in fvs:
                    layer = layers_seq[v]
                    a = layer["self_attn"]
                    oarr, max_p = full_offs if full_offs is not None else stage.full_off_for_visit(v, st_len)
                    tk = (-1 if full_offs is not None else v, a.log_scaling_alpha, a.log_scaling_n_floor)
                    tv = tau_cache.get(tk)
                    if tv is None:
                        tv = tau_cache[tk] = _get_tau_table(layer, max_p)[oarr]
                    out.append(tv)
                return out

            mid_eval = os.environ.get("LOOPLING_BATCH_MID_EVAL", "none")
            if mid_eval == "none" and os.environ.get("LOOPLING_COMPILE_BATCH_ONE", "1") not in ("0", ""):
                fn_one, fvs_all = self._get_batch_compiled_one(
                    regime_sig, step_Ss, num_passes, cd_key, fuse_ck1, fuse_k3b, stage.scap, sdpa_mid=sdpa_mid,
                )
                o_all = fn_one(
                    tokens, stage.buf, step_metas, st_infos, _taus_b(fvs_all),
                    stage._ks_flat_all, stage._vs_flat_all, *conv_flat,
                )
                h_out = o_all[0]
                kv_news = o_all[1 : 1 + self.nv]
                cws = o_all[1 + self.nv :]
                for v in range(self.nv):
                    self.conv[v] = [cws[4 * v], cws[4 * v + 1], cws[4 * v + 2], cws[4 * v + 3]]
                stage.commit_step(kv_news)
                deps = [stage.buf]
                if self._stage_dep_mode == "hidden":
                    h_out = mx.depends(h_out, deps)
                    self._pending_stage_deps = ()
                elif self._stage_dep_mode == "step":
                    self._pending_stage_deps = tuple(deps)
                else:
                    self._pending_stage_deps = ()
                return h_out, None

            chunks, pre_n, n_bs_core, cd_on, r_pass, blend_bs = self._get_batch_compiled(
                regime_sig, step_Ss, num_passes, cd_key, fuse_ck1, fuse_k3b, stage.scap, sdpa_mid=sdpa_mid,
            )

            def _kv_flat(v0, v1):
                ks = [x for v in range(v0, v1) for x in stage.ks_main[v]]
                vs = [x for v in range(v0, v1) for x in stage.vs_main[v]]
                return ks, vs

            outs_all = []
            fn0, v0, v1, nbs_pre, fvs0 = chunks[0]
            ks0, vs0 = _kv_flat(v0, v1)
            o_pre = fn0(
                tokens, [], [], [], stage.buf, step_metas[v0:v1], st_infos[v0:v1], _taus_b(fvs0),
                ks0, vs0, *conv_flat[4 * v0 : 4 * v1],
            )
            if mid_eval in ("all", "pre"):
                mx.async_eval(o_pre[0])
            outs_all.append((o_pre, v1 - v0, nbs_pre))
            h_pre = o_pre[0]
            bs_pre = o_pre[1 : 1 + nbs_pre]
            o_prev = o_pre
            ref_o = None
            for p in range(num_passes):
                fn_p, pv0, pv1, nbs_p, pfvs = chunks[1 + p]
                ksp, vsp = _kv_flat(pv0, pv1)
                if p == 0:
                    o_p = fn_p(
                        h_pre, bs_pre, [], [], stage.buf, step_metas[pv0:pv1], st_infos[pv0:pv1], _taus_b(pfvs),
                        ksp, vsp, *conv_flat[4 * pv0 : 4 * pv1],
                    )
                else:
                    o_p = fn_p(
                        o_prev[0], o_prev[1 : 1 + n_bs_core], h_pre, bs_pre,
                        stage.buf, step_metas[pv0:pv1], st_infos[pv0:pv1], _taus_b(pfvs),
                        ksp, vsp, *conv_flat[4 * pv0 : 4 * pv1],
                    )
                if mid_eval == "all":
                    mx.async_eval(o_p[0])
                outs_all.append((o_p, pv1 - pv0, nbs_p))
                o_prev = o_p
                if cd_on and p == r_pass:
                    ref_o = o_p
            fn_c, cv0, cv1, _, cfvs = chunks[-1]
            ksc, vsc = _kv_flat(cv0, cv1)
            h_aux = ref_o[0] if cd_on else []
            bs_aux = ref_o[1 : 1 + n_bs_core] if (cd_on and blend_bs) else []
            o_c = fn_c(
                o_prev[0], o_prev[1 : 1 + n_bs_core], h_aux, bs_aux,
                stage.buf, step_metas[cv0:cv1], st_infos[cv0:cv1], _taus_b(cfvs),
                ksc, vsc, *conv_flat[4 * cv0 : 4 * cv1],
            )
            outs_all.append((o_c, cv1 - cv0, 0))
            kv_news, cws = [], []
            for out_i, nvi, nbsi in outs_all:
                head = 1 + nbsi
                kv_news.extend(out_i[head : head + nvi])
                cws.extend(out_i[head + nvi : head + nvi + 4 * nvi])
            for v in range(self.nv):
                self.conv[v] = [cws[4 * v], cws[4 * v + 1], cws[4 * v + 2], cws[4 * v + 3]]
            stage.commit_step(kv_news)
            deps = [stage.buf]
            h_out = o_c[0]
            if self._stage_dep_mode == "hidden":
                h_out = mx.depends(h_out, deps)
                self._pending_stage_deps = ()
            elif self._stage_dep_mode == "step":
                self._pending_stage_deps = tuple(deps)
            else:
                self._pending_stage_deps = ()
            return h_out, None

        kv_news = [None] * self.nv
        self._v = 0
        tau_cache = {}
        ref_state = None
        C = self.C
        h = tm.embed(tokens.reshape(C, 1)).reshape(C, 2304)
        use_batch_msg = L._use_kernels() and getattr(tm, "message_input", None) is not None
        split_batch = os.environ.get("LOOPLING_BATCH_SPLIT", "1") not in ("0", "")

        def run_batch(ls, h, bs, bank=None):
            msgs = None
            bank4 = None
            if bank is not None:
                if use_batch_msg:
                    msgs = IK.depth_attention_batch(bank[0], bank[1], tm.message_input.query, scale=2304 ** -0.5)
                else:
                    bank4 = (bank[0][:, :, None, :], bank[1][:, None, :])
            prev_moe = None
            last_i = len(ls) - 1
            for i, layer in enumerate(ls):
                if bank is None:
                    msg = None
                elif msgs is not None:
                    msg = msgs[i]
                else:
                    msg = tm.message_input(bank4, i).reshape(C, 4, 2304)
                next_gpar1 = i < last_i and (isinstance(ls[i + 1], L.SlidingDecoderLayer) or bank is not None)
                h, bs, prev_moe = self._layer_batch(
                    layer, self._v, h, bs, msg,
                    self._stage_main, step_metas, step_Ss, full_offs, st_len, kv_news,
                    tau_cache, self.conv, prev_moe=prev_moe, skip_combine=(fuse_ck1 and next_gpar1),
                )
                self._v += 1
            return h, bs

        prelude = run_batch(layers[: core.start], h, None)
        h, bs = prelude
        if split_batch:
            mx.async_eval(h)
        pre_n = len(prelude[1])
        for p in range(num_passes):
            bank = None
            if p > 0:
                bank = (
                    mx.stack(bs[pre_n + 1 :], axis=0),
                    L.unweighted_rms_norm(h.astype(mx.float32)),
                )
                h, bs = prelude
            h, bs = run_batch(layers[core.start : core.stop], h, bs, bank)
            if split_batch:
                mx.async_eval(h)
            if p == ref_pass:
                keep_bs = cfg.blend_bs or do_logits_cd
                ref_state = (h, list(bs) if keep_bs else None)
        if use_cd and cfg.mode == "hidden":
            h, bs = loopcd.apply_hidden(
                h, bs, ref_state[0], ref_state[1], pre_n, cfg, skip_mask=self._loopcd_skip_mask(cfg)
            )
        coda_layers = layers[core.stop :]
        h, bs = run_batch(coda_layers, h, bs)
        assert self._v == self.nv
        self._stage_main.commit_step(kv_news)
        deps = [self._stage_main.buf]
        h_out = RL.k1_pre(h, bs, None, tm.final_input.query, [tm.norm.weight], sliding=True, eps_ln=tm.norm.eps)[
            "q_in"
        ].reshape(C, 1, 2304)
        h_ref_out = None
        if do_logits_cd:
            ref_metas, ref_Ss, ref_full_offs, ref_st_len = self._stage_ref.step_view()
            ref_kv_news = [None] * self._n_coda
            hr, bsr = ref_state
            prev_moe = None
            last_j = len(coda_layers) - 1
            for j, layer in enumerate(coda_layers):
                hr, bsr, prev_moe = self._layer_batch(
                    layer, j, hr, bsr, None,
                    self._stage_ref, ref_metas, ref_Ss, ref_full_offs, ref_st_len, ref_kv_news,
                    tau_cache, self._ref_conv, prev_moe=prev_moe, skip_combine=(fuse_ck1 and j < last_j),
                )
            self._stage_ref.commit_step(ref_kv_news)
            deps.append(self._stage_ref.buf)
            h_ref_out = RL.k1_pre(
                hr, bsr, None, tm.final_input.query, [tm.norm.weight], sliding=True, eps_ln=tm.norm.eps
            )["q_in"].reshape(C, 1, 2304)
        if self._stage_dep_mode == "hidden":
            h_out = mx.depends(h_out, deps)
            self._pending_stage_deps = ()
        elif self._stage_dep_mode == "step":
            self._pending_stage_deps = tuple(deps)
        else:
            self._pending_stage_deps = ()
        return h_out, h_ref_out

    def step_with_taps(self, tokens, *, tap_visits=None, sampler=None):
        assert self._b1_fast
        tap_set = set(tap_visits) if tap_visits is not None else set()
        tap_buf = {}
        h, h_ref = self._b1_hidden_pair(tokens, tap_set=tap_set, tap_buf=tap_buf)
        self._flush_b1_pending()
        is_greedy = sampler is None or getattr(sampler, "_sampler_key", None) == ("greedy",)
        if h_ref is not None:
            z_R, z_ref = self._pair_logits_softcap(h, h_ref)
            logits = loopcd.combine_logits(z_R, z_ref, self.tm.loopcd, skip_mask=self._loopcd_skip_mask(self.tm.loopcd))
        elif not is_greedy:
            tok = self._sample_verify_tokens(h.reshape(1, 2304), sampler).reshape(1).astype(mx.uint32)
            return self._with_stage_deps(tok), tap_buf
        elif "head" in _FUSED:
            m = self.model
            lm = m.lm_head
            text = m.args.text
            if "head" not in _PLAIN and FD._is_aff8(lm) and h.dtype == mx.bfloat16:
                return (
                    self._with_stage_deps(
                        RQ.qmv_softcap_argmax_rows(
                            h.reshape(1, 2304), lm["weight"], lm["scales"], lm["biases"],
                            text.final_logit_softcapping, text.unpadded_vocab_size,
                        )
                    ),
                    tap_buf,
                )
            elif "head" in _PLAIN:
                logits = lm(h)
            else:
                with _RowsMode():
                    logits = lm(h)
            return self._with_stage_deps(RH.softcap_argmax_rows(logits, text.final_logit_softcapping, text.unpadded_vocab_size)), tap_buf
        else:
            logits = self._project_logits_softcap(h)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tok = mx.argmax(lp, axis=-1) if is_greedy else sampler(lp).reshape(1).astype(mx.uint32)
        return self._with_stage_deps(tok), tap_buf

    @staticmethod
    def _regime_key(s_min: int, s_max: int):
        if RS._LONG_SDPA and s_min < RS._LONG_SDPA_MIN_S <= s_max:
            return None
        if RS._LONG_SDPA and s_min >= RS._LONG_SDPA_MIN_S:
            return ("long", 256)
        if s_max < RS._TWO_PASS_N:
            return (False, 32)
        blk_min = RS._blocks(s_min, 3)
        blk_max = RS._blocks(s_max, 3)
        if s_min < RS._TWO_PASS_N or blk_min != blk_max:
            return None
        return (True, blk_max)

    def _cw_safe_bounds_range(self, end_lo: int, end_hi: int, max_depth: int):
        if not self._kv_stacked or not self._ring_visits or not self._full_visits:
            return False, None
        kv_r = self.caches[0][self._ring_visits[0]][0]
        max_size_r = int(kv_r.max_size)
        cap_r = int(self._kv_ring.shape[3])
        cap_f = int(self._kv_full.shape[3])
        s_min_r = min(int(end_lo) + 1, max_size_r)
        s_max_r = min(int(end_hi) + int(max_depth) + 1, max_size_r)
        s_min_f = int(end_lo) + 1
        s_max_f = int(end_hi) + int(max_depth) + 1
        r_key = self._regime_key(s_min_r, s_max_r)
        f_key = self._regime_key(s_min_f, s_max_f)
        if r_key is None or f_key is None:
            return False, None
        return True, (r_key, f_key, (s_min_r, s_max_r), (s_min_f, s_max_f), cap_r, cap_f, max_size_r)

    def _cw_safe_bounds(self, max_depth):
        if not self._kv_stacked or not self._ring_visits or not self._full_visits:
            return False, None
        kv_r = self.caches[0][self._ring_visits[0]][0]
        kv_f = self.caches[0][self._full_visits[0]][0]
        end0_r = int(kv_r.offset)
        max_size_r = int(kv_r.max_size)
        cap_r = int(self._kv_ring.shape[3])
        slot0_r = 0 if kv_r._idx == max_size_r else int(kv_r._idx)
        s0_r = min(end0_r, max_size_r)
        s_min_r = min(end0_r + 1, max_size_r)
        s_max_r = min(end0_r + max_depth + 1, max_size_r)

        end0_f = int(kv_f.offset)
        cap_f = int(self._kv_full.shape[3])
        s_min_f = end0_f + 1
        s_max_f = end0_f + max_depth + 1

        r_key = self._regime_key(s_min_r, s_max_r)
        f_key = self._regime_key(s_min_f, s_max_f)
        if r_key is None or f_key is None:
            return False, None

        meta_r = _get_cw_meta_arr((s0_r, end0_r, slot0_r, cap_r, max_size_r, 0))
        meta_f = _get_cw_meta_arr((end0_f, end0_f, 0, cap_f, 0, 0))
        return True, (r_key, f_key, meta_r, meta_f, (s_min_r, s_max_r), (s_min_f, s_max_f), end0_f, cap_r, cap_f)

    def _get_cw_compiled(
        self, W, max_depth, r_key, f_key, cap_r, cap_f, s_bounds_r, s_bounds_f, tap_tuple,
        use_tau_table=False, tau_size=0, use_walk=False, fuse_k1=None, use_stage=False, use_dyn=False,
        return_h_out=False,
    ):
        if fuse_k1 is None:
            fuse_k1 = (not use_dyn) and (os.environ.get("LOOPLING_FUSE_COMB_K1", "0") not in ("0", ""))
        md_key = None if use_tau_table else max_depth
        sdpa_mid_cw = os.environ.get("LOOPLING_SDPA_MID_CW", "1") not in ("0", "")
        key = (
            W, md_key, r_key, f_key, cap_r, cap_f, tap_tuple,
            use_tau_table, tau_size, use_walk, fuse_k1, use_stage, use_dyn, return_h_out, sdpa_mid_cw,
            _env_compile_sig(),
        )
        ent = self._cw_compiled.get(key)
        if ent is not None:
            return ent
        tm = self.tm
        m = self.model
        layers_seq = tm.layer_visits
        tap_set = set(tap_tuple) if tap_tuple is not None else set()
        nv = self.nv
        cw_meta = self._cw_meta
        kv_vmap = self._static_kv_vmap
        has_o = "o" in tap_set
        has_F = tap_tuple is not None

        def make_chunk(v0, v1, is_first=False, is_last=False):
            n_v = v1 - v0
            t_in_chunk = frozenset(v for v in range(v0, v1) if v in tap_set)
            f_vs = [v for v in range(v0, v1) if not cw_meta[v][1]]
            f_pos = {v: i for i, v in enumerate(f_vs)}
            n_bs_pre = sum(1 for v in range(0, v1) if layers_seq[v].is_block_boundary)

            @mx.compile
            def fn(h_or_toks, bs_in, kv_r, kv_f, kv_st_r, kv_st_f, ca_in, tm_in, dep_in, m_r, m_f, tau_luts, w_act_in, walk_in, *convs):
                conv_local = [None] * nv
                for j, v in enumerate(range(v0, v1)):
                    conv_local[v] = [convs[4 * j], convs[4 * j + 1], convs[4 * j + 2], convs[4 * j + 3]]
                kst_r_arg = kv_st_r if use_stage else None
                kst_f_arg = kv_st_f if use_stage else None
                wa = w_act_in[0] if use_dyn else None
                h = tm.embed(h_or_toks.reshape(W, 1)).reshape(W, 2304) if is_first else h_or_toks
                bs = None if is_first else (list(bs_in) if bs_in else None)
                prev_moe = None
                kvns, cws, taps_out = [], [], []
                for v in range(v0, v1):
                    tlut = tau_luts[f_pos[v]] if v in f_pos else None
                    is_ring = kv_vmap[v][0]
                    skip_v = fuse_k1 and (
                        (v + 1 < v1 and cw_meta[v + 1][1])
                        or (is_last and v == v1 - 1)
                    )
                    use_prev = prev_moe is not None
                    out_v, bs, kvn, cw, prev_w3, h_before_v = _layer_causal_w_body(
                        cw_meta, kv_vmap, kv_r, kv_f, kst_r_arg, kst_f_arg, conv_local,
                        layers_seq[v], v, h, bs, None, W,
                        depths_arr=dep_in,
                        tau_lut=(None if use_tau_table else tlut),
                        tau_table=(tlut if use_tau_table else None),
                        meta_arr=(m_r if is_ring else m_f),
                        s_bounds=(s_bounds_r if is_ring else s_bounds_f),
                        conv_anc=ca_in, tree_meta=tm_in,
                        prev_moe=prev_moe, skip_combine=skip_v, emit_h=use_prev,
                        use_stage=use_stage, w_act=wa,
                    )
                    if v in t_in_chunk:
                        taps_out.append(h_before_v)
                    kvns.append(kvn)
                    if prev_w3 is not None:
                        cws.append(prev_w3)
                    cws.extend(cw)
                    if skip_v:
                        prev_moe = out_v
                    else:
                        prev_moe = None
                        h = out_v
                if is_last:
                    if prev_moe is not None:
                        r_fn = RL.k1_from_down_causal(
                            prev_moe["down"], prev_moe["expert_w"], prev_moe["residual"], prev_moe["cs3"],
                            prev_moe["conv_w"], ca_in, bs, None, tm.final_input.query, [tm.norm.weight],
                            sliding=True, eps_ln=tm.norm.eps, emit_h=has_o,
                        )
                        cws.append(r_fn["w3"])
                        if has_o:
                            taps_out.append(r_fn["h"])
                        h_out = r_fn["q_in"]
                    else:
                        if has_o:
                            taps_out.append(h)
                        h_out = RL.k1_pre(
                            h, bs, None, tm.final_input.query, [tm.norm.weight],
                            sliding=True, eps_ln=tm.norm.eps, w_act=wa,
                        )["q_in"]
                    if has_F:
                        taps_out.append(h_out)
                    if return_h_out:
                        return [h_out] + kvns + cws + taps_out
                    lm = m.lm_head
                    text = m.args.text
                    if use_walk and FD._is_aff8(lm) and h_out.dtype == mx.bfloat16:
                        w_blk, w_par, w_dr_pos = walk_in
                        nt_or_walk = list(RQ.qmv_softcap_argmax_walk_rows(
                            h_out, lm["weight"], lm["scales"], lm["biases"],
                            w_blk, w_par, m_r, m_f, w_dr_pos,
                            text.final_logit_softcapping, text.unpadded_vocab_size, w_act=wa,
                        ))
                        return nt_or_walk + kvns + cws + taps_out
                    if "head" in _FUSED and "head" not in _PLAIN and FD._is_aff8(lm) and h_out.dtype == mx.bfloat16:
                        nt = RQ.qmv_softcap_argmax_rows(
                            h_out, lm["weight"], lm["scales"], lm["biases"],
                            text.final_logit_softcapping, text.unpadded_vocab_size,
                        )
                    else:
                        lg = RQ.qmv_rows(h_out, lm["weight"], lm["scales"], lm["biases"])
                        nt = RH.softcap_argmax_rows(lg, text.final_logit_softcapping, text.unpadded_vocab_size)
                    if use_walk:
                        from mlx_lm.models.inkling2.rows.qmv4 import tree_walk_commit_meta_gpu
                        w_blk, w_par, w_dr_pos = walk_in
                        return list(tree_walk_commit_meta_gpu(nt, w_blk, w_par, m_r, m_f, w_dr_pos, W, w_act=wa)) + kvns + cws + taps_out
                    return [nt] + kvns + cws + taps_out
                return [h] + (list(bs) if bs is not None else []) + kvns + cws + taps_out

            t_keys = [v for v in range(v0, v1) if v in tap_set]
            if is_last:
                if has_o:
                    t_keys.append("o")
                if has_F:
                    t_keys.append("F")
            return fn, n_v, (0 if is_last else n_bs_pre), f_vs, t_keys

        c0 = make_chunk(0, 6, is_first=True)
        c1 = make_chunk(6, 14)
        c2 = make_chunk(14, 24, is_last=True)
        if len(self._cw_compiled) >= 64:
            self._cw_compiled.clear()
        ent = self._cw_compiled[key] = (c0, c1, c2, use_walk, use_stage)
        return ent

    @staticmethod
    def _unpack_cw_outputs(o0, o1, o2, nv0, nv1, nv2, nbs0, nbs1, tks0, tks1, tks2, want_taps, use_walk=False, use_stage=False):
        head0 = 1 + nbs0
        head1 = 1 + nbs1
        head2 = (12 if use_stage else 11) if use_walk else 1
        ncw0 = 4 * nv0
        ncw1 = 4 * nv1
        ncw2 = 4 * nv2
        kv_news = o0[head0 : head0 + nv0] + o1[head1 : head1 + nv1] + o2[head2 : head2 + nv2]
        c0_end = head0 + nv0 + ncw0
        c1_end = head1 + nv1 + ncw1
        c2_end = head2 + nv2 + ncw2
        cw_f = o0[head0 + nv0 : c0_end] + o1[head1 + nv1 : c1_end] + o2[head2 + nv2 : c2_end]
        conv_ws = [tuple(cw_f[4 * j : 4 * j + 4]) for j in range(24)]
        tap_buf = None
        if want_taps:
            tap_buf = {}
            for k, arr in zip(tks0, o0[c0_end:]):
                tap_buf[k] = arr
            for k, arr in zip(tks1, o1[c1_end:]):
                tap_buf[k] = arr
            for k, arr in zip(tks2, o2[c2_end:]):
                tap_buf[k] = arr
        ret = tuple(o2[:head2]) if use_walk else o2[0]
        return ret, kv_news, conv_ws, tap_buf

    def _sample_verify_tokens(self, h_out, sampler, *, coupled=None):
        m = self.model
        lm = m.lm_head
        text = m.args.text
        std = _std_sampler_params(sampler)
        if coupled is not None and std is not None and FD._is_aff8(lm) and h_out.dtype == mx.bfloat16:
            q_ids, q_probs, child_toks = coupled
            lg = RQ.qmv_rows(h_out, lm["weight"], lm["scales"], lm["biases"])
            bonus, reject, child_pt, ovf = RH.softcap_coupled_verify_rows(
                lg, text.final_logit_softcapping, text.unpadded_vocab_size,
                q_ids, q_probs, child_toks, temp=std[0], top_p=std[1], min_p=std[2],
            )
            return bonus, reject, child_pt, ovf
        if "head" not in _PLAIN and FD._is_aff8(lm) and h_out.dtype == mx.bfloat16:
            lg = RQ.qmv_rows(h_out, lm["weight"], lm["scales"], lm["biases"])
            if std is not None and 0.0 < std[1] < 1.0:
                toks, ovf = RH.softcap_sample_rows(
                    lg, text.final_logit_softcapping, text.unpadded_vocab_size,
                    temp=std[0], top_p=std[1], min_p=std[2],
                )
                if int(mx.max(ovf).item()) == 0:
                    return toks
        elif "head" in _PLAIN:
            lg = lm(h_out)
        else:
            with _RowsMode():
                lg = lm(h_out)
        lg = lg[:, : text.unpadded_vocab_size].astype(mx.float32)
        softcap = float(text.final_logit_softcapping)
        if softcap > 0.0:
            lg = mx.tanh(lg / softcap) * softcap
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        return sampler(lp).astype(mx.uint32)

    def verify_w_gpu(
        self, tokens, *, meta_r, meta_f, depths_arr, conv_anc, tree_meta, max_depth, binfo, tau_cap,
        tap_visits=None, par_arr=None, dr_pos=None, use_stage=False, w_act=None, sampler=None, coupled=None,
    ):
        """Non-mutating causal W-token verify using GPU `meta_arr` and `tau_table`, without mid-async_eval."""
        assert self._b1_fast and self._kv_stacked
        tm = self.tm
        num_passes = tm.args.loop_config["num_passes"]
        assert not (tm.loopcd.enabled_for(num_passes) and tm.loopcd.mode != "hidden")
        assert self.nv == len(tm.layer_visits) and ((num_passes == 1 and self.nv == 24) or num_passes > 1)
        W = int(tokens.shape[0])
        assert 2 <= W <= 8
        use_walk = par_arr is not None and dr_pos is not None
        is_greedy = sampler is None or getattr(sampler, "_sampler_key", None) == ("greedy",)
        compile_walk = use_walk and is_greedy
        return_h_out = not is_greedy
        use_dyn = w_act is not None
        r_key, f_key, s_b_r, s_b_f, cap_r, cap_f, _ = binfo
        tap_set = set(tap_visits) if tap_visits is not None else set()
        tap_tuple = (
            tuple(sorted(tap_set, key=lambda k: (0, k) if isinstance(k, int) else (1, str(k))))
            if tap_visits is not None
            else None
        )
        layers_seq = tm.layer_visits
        tau_size = max(4096, ((int(tau_cap) + 4095) // 4096) * 4096)
        conv_flat = [x for cs in self.conv for x in cs]
        kst_r = self._kv_ring_st if use_stage else []
        kst_f = self._kv_full_st if use_stage else []
        wa_in = [w_act] if use_dyn else []

        if num_passes == 1 and self.nv == 24:
            (fn0, nv0, nbs0, fvs0, tks0), (fn1, nv1, nbs1, fvs1, tks1), (fn2, nv2, _, fvs2, tks2), _, _ = self._get_cw_compiled(
                W, max_depth, r_key, f_key, cap_r, cap_f, s_b_r, s_b_f, tap_tuple,
                use_tau_table=True, tau_size=tau_size, use_walk=compile_walk, use_stage=use_stage, use_dyn=use_dyn,
                return_h_out=return_h_out,
            )
            ttbl0 = [_get_tau_table(layers_seq[v], tau_size) for v in fvs0]
            ttbl1 = [_get_tau_table(layers_seq[v], tau_size) for v in fvs1]
            ttbl2 = [_get_tau_table(layers_seq[v], tau_size) for v in fvs2]

            o0 = fn0(tokens, [], self._kv_ring, self._kv_full, kst_r, kst_f, conv_anc, tree_meta, depths_arr, meta_r, meta_f, ttbl0, wa_in, [], *conv_flat[:24])
            o1 = fn1(o0[0], o0[1 : 1 + nbs0], self._kv_ring, self._kv_full, kst_r, kst_f, conv_anc, tree_meta, depths_arr, meta_r, meta_f, ttbl1, wa_in, [], *conv_flat[24:56])
            o2 = fn2(o1[0], o1[1 : 1 + nbs1], self._kv_ring, self._kv_full, kst_r, kst_f, conv_anc, tree_meta, depths_arr, meta_r, meta_f, ttbl2, wa_in, [tokens, par_arr, dr_pos] if compile_walk else [], *conv_flat[56:])
            ret, kv_news, conv_ws, tap_buf = self._unpack_cw_outputs(
                o0, o1, o2, nv0, nv1, nv2, nbs0, nbs1, tks0, tks1, tks2, tap_visits is not None,
                use_walk=compile_walk, use_stage=use_stage,
            )
            if return_h_out:
                if coupled is not None and use_walk:
                    q_ids, q_probs, q_child, u_acc = coupled
                    bonus, reject, child_pt, ovf = self._sample_verify_tokens(
                        ret, sampler, coupled=(q_ids, q_probs, tokens[1:])
                    )
                    from mlx_lm.models.inkling2.rows.qmv4 import coupled_walk_commit_meta_gpu
                    walk_t = coupled_walk_commit_meta_gpu(
                        tokens, bonus, reject, child_pt, q_child, u_acc, meta_r, meta_f, dr_pos, W,
                    )
                    ret = (walk_t, ovf)
                else:
                    nt = self._sample_verify_tokens(ret, sampler)
                    if use_walk:
                        from mlx_lm.models.inkling2.rows.qmv4 import tree_walk_commit_meta_gpu
                        ret = tuple(tree_walk_commit_meta_gpu(nt, tokens, par_arr, meta_r, meta_f, dr_pos, W, w_act=w_act))
                    else:
                        ret = nt
            return ret, kv_news, conv_ws, tap_buf

        chunks, pre_n, n_bs_core, cd_on, ref_pass, blend_bs = self._get_cw_compiled_loop(
            W, max_depth, r_key, f_key, cap_r, cap_f, s_b_r, s_b_f, tap_tuple,
            use_tau_table=True, tau_size=tau_size, use_walk=compile_walk, use_stage=use_stage, use_dyn=use_dyn,
            return_h_out=return_h_out,
        )
        outs_all = []
        fn0, v0, v1, nbs_pre, fvs0, tks0 = chunks[0]
        ttbl = [_get_tau_table(layers_seq[v], tau_size) for v in fvs0]
        o_pre = fn0(
            tokens, [], [], [], self._kv_ring, self._kv_full, kst_r, kst_f, conv_anc, tree_meta,
            depths_arr, meta_r, meta_f, ttbl, wa_in, [], *conv_flat[4 * v0 : 4 * v1],
        )
        outs_all.append((o_pre, v1 - v0, 1 + nbs_pre, tks0))
        h_pre = o_pre[0]
        bs_pre = o_pre[1 : 1 + nbs_pre]
        o_prev = o_pre
        ref_o = None
        for p in range(num_passes):
            fn_p, pv0, pv1, nbs_p, pfvs, ptks = chunks[1 + p]
            ttbl = [_get_tau_table(layers_seq[v], tau_size) for v in pfvs]
            if p == 0:
                o_p = fn_p(
                    h_pre, bs_pre, [], [], self._kv_ring, self._kv_full, kst_r, kst_f, conv_anc, tree_meta,
                    depths_arr, meta_r, meta_f, ttbl, wa_in, [], *conv_flat[4 * pv0 : 4 * pv1],
                )
            else:
                o_p = fn_p(
                    o_prev[0], o_prev[1 : 1 + n_bs_core], h_pre, bs_pre,
                    self._kv_ring, self._kv_full, kst_r, kst_f, conv_anc, tree_meta,
                    depths_arr, meta_r, meta_f, ttbl, wa_in, [], *conv_flat[4 * pv0 : 4 * pv1],
                )
            outs_all.append((o_p, pv1 - pv0, 1 + nbs_p, ptks))
            o_prev = o_p
            if cd_on and p == ref_pass:
                ref_o = o_p
        fn_c, cv0, cv1, _, cfvs, ctks = chunks[-1]
        ttbl = [_get_tau_table(layers_seq[v], tau_size) for v in cfvs]
        h_aux = ref_o[0] if cd_on else []
        bs_aux = ref_o[1 : 1 + n_bs_core] if (cd_on and blend_bs) else []
        head_c = (12 if use_stage else 11) if compile_walk else 1
        o_c = fn_c(
            o_prev[0], o_prev[1 : 1 + n_bs_core], h_aux, bs_aux,
            self._kv_ring, self._kv_full, kst_r, kst_f, conv_anc, tree_meta,
            depths_arr, meta_r, meta_f, ttbl, wa_in, [tokens, par_arr, dr_pos] if compile_walk else [],
            *conv_flat[4 * cv0 : 4 * cv1],
        )
        outs_all.append((o_c, cv1 - cv0, head_c, ctks))
        kv_news, conv_ws = [], []
        tap_buf = {} if tap_visits is not None else None
        for out_i, nvi, head_i, tksi in outs_all:
            ncw = 4 * nvi
            kv_news.extend(out_i[head_i : head_i + nvi])
            conv_ws.extend(out_i[head_i + nvi : head_i + nvi + ncw])
            if tap_buf is not None:
                for k, arr in zip(tksi, out_i[head_i + nvi + ncw :]):
                    tap_buf[k] = arr
        conv_ws = [tuple(conv_ws[4 * j : 4 * j + 4]) for j in range(self.nv)]
        ret = tuple(o_c[:head_c]) if compile_walk else o_c[0]
        if return_h_out:
            if coupled is not None and use_walk:
                q_ids, q_probs, q_child, u_acc = coupled
                bonus, reject, child_pt, ovf = self._sample_verify_tokens(
                    ret, sampler, coupled=(q_ids, q_probs, tokens[1:])
                )
                from mlx_lm.models.inkling2.rows.qmv4 import coupled_walk_commit_meta_gpu
                walk_t = coupled_walk_commit_meta_gpu(
                    tokens, bonus, reject, child_pt, q_child, u_acc, meta_r, meta_f, dr_pos, W,
                )
                ret = (walk_t, ovf)
            else:
                nt = self._sample_verify_tokens(ret, sampler)
                if use_walk:
                    from mlx_lm.models.inkling2.rows.qmv4 import tree_walk_commit_meta_gpu
                    ret = tuple(tree_walk_commit_meta_gpu(nt, tokens, par_arr, meta_r, meta_f, dr_pos, W, w_act=w_act))
                else:
                    ret = nt
        return ret, kv_news, conv_ws, tap_buf

    def _get_cw_compiled_loop(
        self, W, max_depth, r_key, f_key, cap_r, cap_f, s_bounds_r, s_bounds_f, tap_tuple,
        use_tau_table=False, tau_size=0, use_walk=False, fuse_k1=None, use_stage=False, use_dyn=False,
        return_h_out=False,
    ):
        if fuse_k1 is None:
            fuse_k1 = (not use_dyn) and (os.environ.get("LOOPLING_FUSE_COMB_K1", "0") not in ("0", ""))
        tm = self.tm
        m = self.model
        layers_seq = tm.layer_visits
        core = tm.args.loop_core
        num_passes = tm.args.loop_config["num_passes"]
        cfg = tm.loopcd
        use_cd = cfg.enabled_for(num_passes) and cfg.mode == "hidden"
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        cd_key = self._loopcd_cd_key(cfg, ref_pass, force_hidden=True) if use_cd else None
        md_key = None if use_tau_table else max_depth
        sdpa_mid_cw = os.environ.get("LOOPLING_SDPA_MID_CW", "1") not in ("0", "")
        key = (
            "loop", W, md_key, r_key, f_key, cap_r, cap_f, tap_tuple,
            use_tau_table, tau_size, use_walk, fuse_k1, use_stage, use_dyn, num_passes, cd_key, return_h_out, sdpa_mid_cw,
            _env_compile_sig(),
        )
        ent = self._cw_compiled.get(key)
        if ent is not None:
            return ent

        tap_set = set(tap_tuple) if tap_tuple is not None else set()
        nv = self.nv
        cw_meta = self._cw_meta
        kv_vmap = self._static_kv_vmap
        has_o = "o" in tap_set
        has_F = tap_tuple is not None
        pre_n = sum(1 for v in range(0, core.start) if layers_seq[v].is_block_boundary)
        n_core = core.stop - core.start
        n_bs_core = pre_n + sum(1 for v in range(core.start, core.stop) if layers_seq[v].is_block_boundary)
        use_batch_msg = L._use_kernels() and getattr(tm, "message_input", None) is not None

        def make_chunk(v0, v1, *, is_first=False, is_last=False, use_msg=False, apply_cd=False):
            n_v = v1 - v0
            t_in_chunk = frozenset(v for v in range(v0, v1) if v in tap_set)
            f_vs = [v for v in range(v0, v1) if not cw_meta[v][1]]
            f_pos = {v: i for i, v in enumerate(f_vs)}

            @mx.compile
            def fn(
                h_in, bs_in, h_aux, bs_aux,
                kv_r, kv_f, kv_st_r, kv_st_f,
                ca_in, tm_in, dep_in, m_r, m_f, tau_luts, w_act_in, walk_in, *convs
            ):
                conv_local = [None] * nv
                for j, v in enumerate(range(v0, v1)):
                    conv_local[v] = [convs[4 * j], convs[4 * j + 1], convs[4 * j + 2], convs[4 * j + 3]]
                kst_r_arg = kv_st_r if use_stage else None
                kst_f_arg = kv_st_f if use_stage else None
                wa = w_act_in[0] if use_dyn else None
                if is_first:
                    h = tm.embed(h_in.reshape(W, 1)).reshape(W, 2304)
                    bs = None
                    msgs = None
                elif use_msg:
                    bank = (
                        mx.stack(list(bs_in[pre_n + 1 :]), axis=0),
                        L.unweighted_rms_norm(h_in.astype(mx.float32)),
                    )
                    if use_batch_msg:
                        msgs = IK.depth_attention_batch(bank[0], bank[1], tm.message_input.query, scale=2304 ** -0.5)
                    else:
                        bank4 = (bank[0][:, :, None, :], bank[1][:, None, :])
                        msgs = [tm.message_input(bank4, i).reshape(W, 4, 2304) for i in range(n_v)]
                    h = h_aux
                    bs = list(bs_aux)
                else:
                    h = h_in
                    bs = list(bs_in) if bs_in else None
                    msgs = None
                    if apply_cd:
                        h, bs = loopcd.apply_hidden(
                            h, bs, h_aux, (list(bs_aux) if cfg.blend_bs else None), pre_n, cfg
                        )

                prev_moe = None
                kvns, cws, taps_out = [], [], []
                for j, v in enumerate(range(v0, v1)):
                    tlut = tau_luts[f_pos[v]] if v in f_pos else None
                    is_ring = kv_vmap[v][0]
                    skip_v = fuse_k1 and (
                        (v + 1 < v1 and cw_meta[v + 1][1])
                        or (is_last and v == v1 - 1)
                    )
                    use_prev = prev_moe is not None
                    msg_v = None if msgs is None else msgs[j]
                    out_v, bs, kvn, cw, prev_w3, h_before_v = _layer_causal_w_body(
                        cw_meta, kv_vmap, kv_r, kv_f, kst_r_arg, kst_f_arg, conv_local,
                        layers_seq[v], v, h, bs, msg_v, W,
                        depths_arr=dep_in,
                        tau_lut=(None if use_tau_table else tlut),
                        tau_table=(tlut if use_tau_table else None),
                        meta_arr=(m_r if is_ring else m_f),
                        s_bounds=(s_bounds_r if is_ring else s_bounds_f),
                        conv_anc=ca_in, tree_meta=tm_in,
                        prev_moe=prev_moe, skip_combine=skip_v, emit_h=use_prev,
                        use_stage=use_stage, w_act=wa,
                    )
                    if v in t_in_chunk:
                        taps_out.append(h_before_v)
                    kvns.append(kvn)
                    if prev_w3 is not None:
                        cws.append(prev_w3)
                    cws.extend(cw)
                    if skip_v:
                        prev_moe = out_v
                    else:
                        prev_moe = None
                        h = out_v
                if is_last:
                    if prev_moe is not None:
                        r_fn = RL.k1_from_down_causal(
                            prev_moe["down"], prev_moe["expert_w"], prev_moe["residual"], prev_moe["cs3"],
                            prev_moe["conv_w"], ca_in, bs, None, tm.final_input.query, [tm.norm.weight],
                            sliding=True, eps_ln=tm.norm.eps, emit_h=has_o,
                        )
                        cws.append(r_fn["w3"])
                        if has_o:
                            taps_out.append(r_fn["h"])
                        h_out = r_fn["q_in"]
                    else:
                        if has_o:
                            taps_out.append(h)
                        h_out = RL.k1_pre(
                            h, bs, None, tm.final_input.query, [tm.norm.weight],
                            sliding=True, eps_ln=tm.norm.eps, w_act=wa,
                        )["q_in"]
                    if has_F:
                        taps_out.append(h_out)
                    if return_h_out:
                        return [h_out] + kvns + cws + taps_out
                    lm = m.lm_head
                    text = m.args.text
                    if use_walk and FD._is_aff8(lm) and h_out.dtype == mx.bfloat16:
                        w_blk, w_par, w_dr_pos = walk_in
                        nt_or_walk = list(RQ.qmv_softcap_argmax_walk_rows(
                            h_out, lm["weight"], lm["scales"], lm["biases"],
                            w_blk, w_par, m_r, m_f, w_dr_pos,
                            text.final_logit_softcapping, text.unpadded_vocab_size, w_act=wa,
                        ))
                        return nt_or_walk + kvns + cws + taps_out
                    if "head" in _FUSED and "head" not in _PLAIN and FD._is_aff8(lm) and h_out.dtype == mx.bfloat16:
                        nt = RQ.qmv_softcap_argmax_rows(
                            h_out, lm["weight"], lm["scales"], lm["biases"],
                            text.final_logit_softcapping, text.unpadded_vocab_size,
                        )
                    else:
                        lg = RQ.qmv_rows(h_out, lm["weight"], lm["scales"], lm["biases"])
                        nt = RH.softcap_argmax_rows(lg, text.final_logit_softcapping, text.unpadded_vocab_size)
                    if use_walk:
                        from mlx_lm.models.inkling2.rows.qmv4 import tree_walk_commit_meta_gpu
                        w_blk, w_par, w_dr_pos = walk_in
                        return list(tree_walk_commit_meta_gpu(nt, w_blk, w_par, m_r, m_f, w_dr_pos, W, w_act=wa)) + kvns + cws + taps_out
                    return [nt] + kvns + cws + taps_out
                return [h] + (list(bs) if bs is not None else []) + kvns + cws + taps_out

            t_keys = [v for v in range(v0, v1) if v in tap_set]
            if is_last:
                if has_o:
                    t_keys.append("o")
                if has_F:
                    t_keys.append("F")
            n_bs_out = 0 if is_last else (pre_n if is_first else n_bs_core)
            return fn, v0, v1, n_bs_out, f_vs, t_keys

        chunks = [make_chunk(0, core.start, is_first=True)]
        for p in range(num_passes):
            v0 = core.start + p * n_core
            chunks.append(make_chunk(v0, v0 + n_core, use_msg=(p > 0)))
        coda_v0 = core.start + num_passes * n_core
        chunks.append(make_chunk(coda_v0, nv, is_last=True, apply_cd=use_cd))
        if len(self._cw_compiled) >= 64:
            self._cw_compiled.clear()
        ent = self._cw_compiled[key] = (chunks, pre_n, n_bs_core, use_cd, ref_pass, bool(cfg.blend_bs))
        return ent

    def verify_w(
        self, tokens, *, tap_visits=None, parents=None, depths=None, depths_arr=None, max_depth=None,
        conv_anc=None, tree_meta=None, sampler=None, return_h_out: bool = False,
    ):
        """Bit-exact causal W-token/tree verify (W in 2..16) without mutating caches or conv state."""
        assert self._b1_fast
        tm = self.tm
        num_passes = tm.args.loop_config["num_passes"]
        cfg = tm.loopcd
        use_cd = cfg.enabled_for(num_passes)
        assert not (use_cd and cfg.mode != "hidden")
        self._ensure_stacked_kv()
        if self._kv_st_len > 0:
            self.flush_kv_stage()
        W = int(tokens.shape[0])
        assert 2 <= W <= 16
        is_greedy = sampler is None or getattr(sampler, "_sampler_key", None) == ("greedy",)
        want_h = bool(return_h_out) or (not is_greedy)
        if conv_anc is None or tree_meta is None or (depths is None and depths_arr is None):
            if parents is None:
                parents = tuple(range(-1, W - 1))
                depths = tuple(range(W))
            depths, conv_anc, tree_meta = build_tree_tables(parents, depths)

        use_compile = (
            os.environ.get("LOOPLING_COMPILE_VERIFY", "1") not in ("0", "")
            and self.nv == len(tm.layer_visits)
            and ((num_passes == 1 and self.nv == 24) or num_passes > 1)
        )
        if use_compile:
            if depths_arr is None:
                darr = _cw_darr_cache.get(depths)
                if darr is None:
                    if len(_cw_darr_cache) >= 256:
                        _cw_darr_cache.clear()
                    darr = _cw_darr_cache[depths] = mx.array(depths, mx.int32)
                depths_arr = darr
                max_d = max(depths)
            else:
                max_d = (W - 1) if max_depth is None else int(max_depth)
            safe, binfo = self._cw_safe_bounds(max_d)
            if safe:
                r_key, f_key, m_r, m_f, s_b_r, s_b_f, end0_f, cap_r, cap_f = binfo
                tap_set = set(tap_visits) if tap_visits is not None else set()
                tap_tuple = tuple(sorted(tap_set, key=lambda k: (0, k) if isinstance(k, int) else (1, str(k)))) if tap_visits is not None else None
                layers_seq = tm.layer_visits
                conv_flat = [x for cs in self.conv for x in cs]
                if num_passes == 1 and self.nv == 24:
                    (fn0, nv0, nbs0, fvs0, tks0), (fn1, nv1, nbs1, fvs1, tks1), (fn2, nv2, _, fvs2, tks2), _, _ = self._get_cw_compiled(
                        W, max_d, r_key, f_key, cap_r, cap_f, s_b_r, s_b_f, tap_tuple,
                        return_h_out=want_h,
                    )
                    tluts0 = [_get_tau_lut(layers_seq[v], end0_f, max_d) for v in fvs0]
                    tluts1 = [_get_tau_lut(layers_seq[v], end0_f, max_d) for v in fvs1]
                    tluts2 = [_get_tau_lut(layers_seq[v], end0_f, max_d) for v in fvs2]

                    o0 = fn0(tokens, [], self._kv_ring, self._kv_full, [], [], conv_anc, tree_meta, depths_arr, m_r, m_f, tluts0, [], [], *conv_flat[:24])
                    mx.async_eval(o0[0])
                    o1 = fn1(o0[0], o0[1 : 1 + nbs0], self._kv_ring, self._kv_full, [], [], conv_anc, tree_meta, depths_arr, m_r, m_f, tluts1, [], [], *conv_flat[24:56])
                    mx.async_eval(o1[0])
                    o2 = fn2(o1[0], o1[1 : 1 + nbs1], self._kv_ring, self._kv_full, [], [], conv_anc, tree_meta, depths_arr, m_r, m_f, tluts2, [], [], *conv_flat[56:])
                    ret, kv_news, conv_ws, tap_buf = self._unpack_cw_outputs(o0, o1, o2, nv0, nv1, nv2, nbs0, nbs1, tks0, tks1, tks2, tap_visits is not None)
                    if not return_h_out and want_h:
                        ret = self._sample_verify_tokens(ret, sampler)
                    return ret, kv_news, conv_ws, tap_buf

                chunks, pre_n, n_bs_core, cd_on, ref_pass, blend_bs = self._get_cw_compiled_loop(
                    W, max_d, r_key, f_key, cap_r, cap_f, s_b_r, s_b_f, tap_tuple,
                    return_h_out=want_h,
                )
                outs_all = []
                # Prelude
                fn0, v0, v1, nbs_pre, fvs0, tks0 = chunks[0]
                tluts = [_get_tau_lut(layers_seq[v], end0_f, max_d) for v in fvs0]
                o_pre = fn0(
                    tokens, [], [], [], self._kv_ring, self._kv_full, [], [], conv_anc, tree_meta,
                    depths_arr, m_r, m_f, tluts, [], [], *conv_flat[4 * v0 : 4 * v1],
                )
                mx.async_eval(o_pre[0])
                outs_all.append((o_pre, v1 - v0, nbs_pre, tks0))
                h_pre = o_pre[0]
                bs_pre = o_pre[1 : 1 + nbs_pre]
                o_prev = o_pre
                ref_o = None
                for p in range(num_passes):
                    fn_p, pv0, pv1, nbs_p, pfvs, ptks = chunks[1 + p]
                    tluts = [_get_tau_lut(layers_seq[v], end0_f, max_d) for v in pfvs]
                    if p == 0:
                        o_p = fn_p(
                            h_pre, bs_pre, [], [], self._kv_ring, self._kv_full, [], [], conv_anc, tree_meta,
                            depths_arr, m_r, m_f, tluts, [], [], *conv_flat[4 * pv0 : 4 * pv1],
                        )
                    else:
                        o_p = fn_p(
                            o_prev[0], o_prev[1 : 1 + n_bs_core], h_pre, bs_pre,
                            self._kv_ring, self._kv_full, [], [], conv_anc, tree_meta,
                            depths_arr, m_r, m_f, tluts, [], [], *conv_flat[4 * pv0 : 4 * pv1],
                        )
                    mx.async_eval(o_p[0])
                    outs_all.append((o_p, pv1 - pv0, nbs_p, ptks))
                    o_prev = o_p
                    if cd_on and p == ref_pass:
                        ref_o = o_p
                # Coda
                fn_c, cv0, cv1, _, cfvs, ctks = chunks[-1]
                tluts = [_get_tau_lut(layers_seq[v], end0_f, max_d) for v in cfvs]
                h_aux = ref_o[0] if cd_on else []
                bs_aux = ref_o[1 : 1 + n_bs_core] if (cd_on and blend_bs) else []
                o_c = fn_c(
                    o_prev[0], o_prev[1 : 1 + n_bs_core], h_aux, bs_aux,
                    self._kv_ring, self._kv_full, [], [], conv_anc, tree_meta,
                    depths_arr, m_r, m_f, tluts, [], [], *conv_flat[4 * cv0 : 4 * cv1],
                )
                outs_all.append((o_c, cv1 - cv0, 0, ctks))
                kv_news, conv_ws = [], []
                tap_buf = {} if tap_visits is not None else None
                for out_i, nvi, nbsi, tksi in outs_all:
                    head = 1 + nbsi
                    ncw = 4 * nvi
                    kv_news.extend(out_i[head : head + nvi])
                    conv_ws.extend(out_i[head + nvi : head + nvi + ncw])
                    if tap_buf is not None:
                        for k, arr in zip(tksi, out_i[head + nvi + ncw :]):
                            tap_buf[k] = arr
                conv_ws = [tuple(conv_ws[4 * j : 4 * j + 4]) for j in range(self.nv)]
                ret = o_c[0]
                if not return_h_out and want_h:
                    ret = self._sample_verify_tokens(ret, sampler)
                return ret, kv_news, conv_ws, tap_buf

        if depths is None:
            depths = tuple(depths_arr.tolist())
        core = tm.args.loop_core
        layers = tm.layers
        v_idx = 0
        tau_cache = {}
        tap_set = set(tap_visits) if tap_visits is not None else set()
        tap_buf = {} if tap_visits is not None else None
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        ref_state = None
        h = tm.embed(tokens.reshape(W, 1)).reshape(W, 2304)
        use_batch_msg = L._use_kernels() and getattr(tm, "message_input", None) is not None
        kv_news = [None] * self.nv
        conv_ws = [None] * self.nv
        split = os.environ.get("LOOPLING_VERIFY_SPLIT", "1") not in ("0", "")
        split_pts = (6, 14) if num_passes == 1 else (6, 18, 30, 42)

        def run_cw(ls, h, bs, bank=None):
            nonlocal v_idx
            msgs = None
            if bank is not None and use_batch_msg:
                msgs = IK.depth_attention_batch(bank[0], bank[1], tm.message_input.query, scale=2304 ** -0.5)
            for i, layer in enumerate(ls):
                if tap_buf is not None and v_idx in tap_set:
                    tap_buf[v_idx] = h
                msg = None if bank is None else (msgs[i] if msgs is not None else tm.message_input(bank, i))
                h, bs, kvn, cw, _, _ = self._layer_causal_w(
                    layer, v_idx, h, bs, msg, W, tau_cache, depths=depths, conv_anc=conv_anc, tree_meta=tree_meta,
                )
                kv_news[v_idx] = kvn
                conv_ws[v_idx] = cw
                v_idx += 1
                if split and v_idx in split_pts:
                    mx.async_eval(h)
            return h, bs

        prelude = run_cw(layers[: core.start], h, None)
        h, bs = prelude
        pre_n = len(prelude[1])
        for p in range(num_passes):
            bank = None
            if p > 0:
                bank = (
                    mx.stack(bs[pre_n + 1 :], axis=0),
                    L.unweighted_rms_norm(h.astype(mx.float32)),
                )
                h, bs = prelude
            h, bs = run_cw(layers[core.start : core.stop], h, bs, bank)
            if use_cd and p == ref_pass:
                ref_state = (h, list(bs) if cfg.blend_bs else None)
        if use_cd and cfg.mode == "hidden":
            h, bs = loopcd.apply_hidden(
                h, bs, ref_state[0], ref_state[1], pre_n, cfg, skip_mask=self._loopcd_skip_mask(cfg)
            )
        h, bs = run_cw(layers[core.stop :], h, bs)
        assert v_idx == self.nv
        if tap_buf is not None and "o" in tap_set:
            tap_buf["o"] = h
        h_out = RL.k1_pre(h, bs, None, tm.final_input.query, [tm.norm.weight], sliding=True, eps_ln=tm.norm.eps)[
            "q_in"
        ]
        if tap_buf is not None:
            tap_buf["F"] = h_out
        m = self.model
        lm = m.lm_head
        text = m.args.text
        if return_h_out:
            return h_out, kv_news, conv_ws, tap_buf
        if not is_greedy:
            next_toks = self._sample_verify_tokens(h_out, sampler)
        elif "head" in _FUSED and "head" not in _PLAIN and FD._is_aff8(lm) and h_out.dtype == mx.bfloat16:
            next_toks = RQ.qmv_softcap_argmax_rows(
                h_out, lm["weight"], lm["scales"], lm["biases"],
                text.final_logit_softcapping, text.unpadded_vocab_size,
            )
        else:
            logits = RQ.qmv_rows(h_out, lm["weight"], lm["scales"], lm["biases"])
            next_toks = RH.softcap_argmax_rows(logits, text.final_logit_softcapping, text.unpadded_vocab_size)
        return next_toks, kv_news, conv_ws, tap_buf

    def commit_w(self, kv_news, conv_ws, keep, path=None):
        """Commit `keep` verified tokens (contiguous 0..keep-1 or explicit node `path`) from verify_w."""
        if path is not None:
            keep = len(path)
            last_node = path[-1]
            p_idx = None if list(path) == list(range(keep)) else mx.array(path, mx.int32)
        else:
            last_node = keep - 1
            p_idx = None
        if self._b1_unpacked_full:
            full_news = {}
            for v in range(self.nv):
                kvn = kv_news[v] if p_idx is None else kv_news[v][:, :, p_idx, :]
                if v in self._b1_full_unp_map:
                    full_news[v] = kvn
                else:
                    self.caches[0][v][0].commit_packed_w(kvn, keep, FD._wpos)
                self.conv[v] = [w[last_node : last_node + 1] for w in conv_ws[v]]
            self._commit_b1_full_unpacked(full_news, keep)
            return
        if self._ensure_stacked_kv():
            if self._kv_st_len > 0:
                self.flush_kv_stage()
            if self._ring_visits:
                buf = self._kv_ring
                self._kv_ring = None
                kv0 = self.caches[0][self._ring_visits[0]][0]
                prev = kv0.offset
                cap = buf.shape[3]
                max_size = kv0.max_size
                src = mx.stack(
                    [kv_news[v][:, :, :keep, :] for v in self._ring_visits]
                    if p_idx is None
                    else [kv_news[v][:, :, p_idx, :] for v in self._ring_visits],
                    axis=0,
                )
                idx = kv0._idx
                if cap < max_size and prev + keep > cap:
                    target_cap = min(max_size, ((prev + keep + kv0.step - 1) // kv0.step) * kv0.step)
                    ext = mx.zeros((len(self._ring_visits), 2, buf.shape[2], target_cap - cap, buf.shape[4]), buf.dtype)
                    buf = mx.concatenate([buf, ext], axis=3)
                    idx = prev
                slot0 = 0 if idx == max_size else idx
                if slot0 + keep <= max_size:
                    buf = mx.slice_update(buf, src, FD._wpos(slot0), axes=(3,))
                else:
                    n1 = max_size - slot0
                    buf = mx.slice_update(buf, src[:, :, :, :n1, :], FD._wpos(slot0), axes=(3,))
                    buf = mx.slice_update(buf, src[:, :, :, n1:, :], FD._wpos(0), axes=(3,))
                new_off = prev + keep
                new_idx = ((slot0 + keep - 1) % max_size) + 1
                for v in self._ring_visits:
                    kv = self.caches[0][v][0]
                    kv.offset = new_off
                    kv._idx = new_idx
                self._kv_ring = buf
                self._kv_main_end_r = new_off
                self._kv_main_slot_r = 0 if new_idx == max_size else new_idx
            if self._full_visits:
                buf = self._kv_full
                self._kv_full = None
                kv0 = self.caches[0][self._full_visits[0]][0]
                prev = kv0.offset
                cap = buf.shape[3]
                src = mx.stack(
                    [kv_news[v][:, :, :keep, :] for v in self._full_visits]
                    if p_idx is None
                    else [kv_news[v][:, :, p_idx, :] for v in self._full_visits],
                    axis=0,
                )
                if prev + keep > cap:
                    if prev % kv0.step != 0:
                        buf = buf[:, :, :, :prev, :]
                    need = prev + keep - buf.shape[3]
                    n_steps = (need + kv0.step - 1) // kv0.step
                    ext = mx.zeros((len(self._full_visits), 2, buf.shape[2], n_steps * kv0.step, buf.shape[4]), buf.dtype)
                    buf = mx.concatenate([buf, ext], axis=3)
                buf = mx.slice_update(buf, src, FD._wpos(prev), axes=(3,))
                new_off = prev + keep
                for v in self._full_visits:
                    self.caches[0][v][0].offset = new_off
                self._kv_full = buf
                self._kv_main_end_f = new_off
            for v in range(self.nv):
                self.conv[v] = [w[last_node : last_node + 1] for w in conv_ws[v]]
            return
        for v in range(self.nv):
            kvn = kv_news[v] if p_idx is None else kv_news[v][:, :, p_idx, :]
            self.caches[0][v][0].commit_packed_w(kvn, keep, FD._wpos)
            self.conv[v] = [w[last_node : last_node + 1] for w in conv_ws[v]]

    def commit_w_gpu(
        self, kv_news, conv_ws, path_pad, keep_arr, last_slot, slot_r_old, end_f_old, W, *, ring_may_wrap=True
    ):
        """Commit verified tokens on GPU without CPU sync; assumes `ensure_gpu_cap` already holds capacity."""
        assert self._kv_stacked and self._ring_visits and self._full_visits
        for v in range(self.nv):
            self.conv[v] = [w[last_slot] for w in conv_ws[v]]

        buf_f = self._kv_full
        self._kv_full = None
        src_f = _gather_stack_kv_gpu([kv_news[v] for v in self._full_visits], path_pad, W)
        self._kv_full = mx.slice_update(buf_f, src_f, end_f_old, axes=(3,))

        buf_r = self._kv_ring
        self._kv_ring = None
        if not ring_may_wrap:
            src_r = _gather_stack_kv_gpu([kv_news[v] for v in self._ring_visits], path_pad, W)
            self._kv_ring = mx.slice_update(buf_r, src_r, slot_r_old, axes=(3,))
        else:
            src_safe = _gather_stack_kv_gpu([kv_news[v] for v in self._ring_visits], path_pad, W, keep_arr=keep_arr)
            max_size = int(self.caches[0][self._ring_visits[0]][0].max_size)
            WW = int(W)
            ar = _cw_ar_cache.get(WW)
            if ar is None:
                ar = _cw_ar_cache[WW] = mx.arange(WW, dtype=mx.int32)
            cond = ar < keep_arr
            pos_r = mx.where(cond, (slot_r_old + ar) % max_size, slot_r_old)
            buf_r[:, :, :, pos_r, :] = src_safe
            self._kv_ring = buf_r

    def commit_w_stage_gpu(self, kv_news, conv_ws, path_pad, last_slot, st_pos_old, W):
        """Commit verified tokens into small (NL, 2, 8, KV_STAGE_CAP, 128) staging buffers on GPU."""
        assert self._kv_stacked and self._ring_visits and self._full_visits
        assert self._kv_ring_st is not None and self._kv_full_st is not None
        for v in range(self.nv):
            self.conv[v] = [w[last_slot] for w in conv_ws[v]]
        sr_st = self._kv_ring_st
        self._kv_ring_st = None
        self._kv_ring_st = update_kv_stage_gpu(
            sr_st, [kv_news[v] for v in self._ring_visits], path_pad, st_pos_old, W,
        )
        sf_st = self._kv_full_st
        self._kv_full_st = None
        self._kv_full_st = update_kv_stage_gpu(
            sf_st, [kv_news[v] for v in self._full_visits], path_pad, st_pos_old, W,
        )

    def _hidden_pair(self, tokens):
        tm = self.tm
        core = tm.args.loop_core
        layers = tm.layers
        self._v = 0
        tau_cache = {}
        num_passes = tm.args.loop_config["num_passes"]
        cfg = tm.loopcd
        use_cd = cfg.enabled_for(num_passes)
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        ref_state = None
        if self._b1_fast:
            return self._b1_hidden_pair(tokens)
        if self._batch_fast:
            return self._batch_hidden_pair(tokens)

        h = tm.embed(tokens.reshape(-1, 1))
        dtype = h.dtype
        use_batch_msg = L._use_kernels() and tm.message_input is not None

        def run(ls, h, bs, bank=None):
            msgs = None
            if bank is not None and use_batch_msg:
                msgs = IK.depth_attention_batch(bank[0], bank[1], tm.message_input.query, scale=bank[1].shape[-1] ** -0.5)
            for i, layer in enumerate(ls):
                msg = None if bank is None else (msgs[i] if msgs is not None else tm.message_input(bank, i))
                h, bs = self._layer(layer, self._v, h, bs, msg, tau_cache)
                self._v += 1
            return h, bs

        prelude = run(layers[: core.start], h, None)
        h, bs = prelude
        pre_n = prelude[1].shape[0]
        for p in range(num_passes):
            bank = None
            if p > 0:
                bank = (bs[pre_n + 1 :], L.unweighted_rms_norm(h.astype(mx.float32)))
                h, bs = prelude
            h, bs = run(layers[core.start : core.stop], h, bs, bank)
            if p == ref_pass:
                ref_state = (h, bs)
        if use_cd and cfg.mode == "hidden":
            h, bs = loopcd.apply_hidden(
                h, bs, ref_state[0], ref_state[1], pre_n, cfg, skip_mask=self._loopcd_skip_mask(cfg)
            )
        coda_layers = layers[core.stop :]
        h, bs = run(coda_layers, h, bs)
        assert self._v == self.nv
        h_out = tm.norm(tm.final_input(bs, L.unweighted_rms_norm(h.astype(mx.float32)))[..., 0, :]).astype(dtype)
        h_ref_out = None
        if use_cd and cfg.mode == "logits":
            if self._ref_caches is None:
                raise RuntimeError("LoopCD-Logits shadow coda cache missing in BatchDecoder.")
            hr, bsr = ref_state
            for j, layer in enumerate(coda_layers):
                hr, bsr = self._layer(
                    layer, j, hr, bsr, None, tau_cache, caches=self._ref_caches, conv=self._ref_conv, kv=self._ref_kv
                )
            h_ref_out = tm.norm(tm.final_input(bsr, L.unweighted_rms_norm(hr.astype(mx.float32)))[..., 0, :]).astype(
                dtype
            )
        return h_out, h_ref_out

    def _with_stage_deps(self, x):
        if self._pending_stage_deps:
            x = mx.depends(x, list(self._pending_stage_deps))
            self._pending_stage_deps = ()
        return x

    def hidden(self, tokens):
        return self._with_stage_deps(self._hidden_pair(tokens)[0])

    def _project_logits_softcap(self, h):
        m = self.model
        lm = m.lm_head
        if (self._b1_fast or self._batch_fast) and "head" not in _PLAIN and FD._is_aff8(lm) and h.dtype == mx.bfloat16:
            logits = RQ.qmv_rows(h.reshape(self.C, 2304), lm["weight"], lm["scales"], lm["biases"]).reshape(self.C, 1, -1)
        elif "head" in _PLAIN:
            logits = lm(h)
        else:
            with _RowsMode():
                logits = lm(h)
        text = m.args.text
        softcap = text.final_logit_softcapping
        logits = logits.astype(mx.float32)
        logits = mx.tanh(logits / softcap) * softcap
        return logits[..., : text.unpadded_vocab_size][:, -1, :]

    def _pair_logits_softcap(self, h_R, h_ref):
        m = self.model
        lm = m.lm_head
        text = m.args.text
        softcap = text.final_logit_softcapping
        if (
            "head" not in _PLAIN
            and "qmv" in _USE
            and 2 * self.C <= RQ.MAX_B
            and FD._is_aff8(lm)
            and h_R.dtype == mx.bfloat16
            and h_R.shape[-1] % 256 == 0
            and h_R.shape[-1] not in (64, 128)
        ):
            D = h_R.shape[-1]
            h_both = mx.concatenate([h_R.reshape(self.C, D), h_ref.reshape(self.C, D)], axis=0)
            raw = RQ.qmv_rows(h_both, lm["weight"], lm["scales"], lm["biases"]).astype(mx.float32)
            raw = mx.tanh(raw / softcap) * softcap
            raw = raw[:, : text.unpadded_vocab_size]
            return raw[: self.C], raw[self.C :]
        return self._project_logits_softcap(h_R), self._project_logits_softcap(h_ref)

    def _loopcd_skip_flags(self, cfg) -> tuple[bool, ...] | None:
        if not cfg.exclude_thinking:
            return None
        flags = tuple(loopcd.get_cache_skip(c) for c in self.caches)
        return flags if any(flags) else None

    def _loopcd_skip_mask(self, cfg):
        flags = self._loopcd_skip_flags(cfg)
        if flags is None:
            return None
        if all(flags):
            return True
        return mx.array(flags, dtype=mx.bool_)

    def _loopcd_cd_key(self, cfg, ref_pass: int, *, force_hidden: bool = False):
        flags = self._loopcd_skip_flags(cfg)
        if flags is not None and all(flags):
            return None
        mode = "hidden" if force_hidden else cfg.mode
        base = (mode, float(cfg.omega), int(ref_pass), bool(cfg.blend_bs))
        return base if flags is None else (*base, flags)

    def logits(self, h):
        if isinstance(h, tuple):
            h_R, h_ref = h
            if h_ref is None:
                return self._with_stage_deps(self._project_logits_softcap(h_R))
            z_R, z_ref = self._pair_logits_softcap(h_R, h_ref)
            return self._with_stage_deps(
                loopcd.combine_logits(z_R, z_ref, self.tm.loopcd, skip_mask=self._loopcd_skip_mask(self.tm.loopcd))
            )
        return self._with_stage_deps(self._project_logits_softcap(h))

    def logprobs(self, h, *, argmax: bool = False):
        """Return exact per-row logprobs (C, V) f32, or (toks (C,) u32, logprobs (C, V) f32) when argmax=True."""
        h_R, h_ref = h if isinstance(h, tuple) else (h, None)
        if h_ref is None:
            m = self.model
            lm = m.lm_head
            text = m.args.text
            softcap = text.final_logit_softcapping
            unpadded = text.unpadded_vocab_size
            if (
                "head" not in _PLAIN
                and (self._b1_fast or self._batch_fast or "qmv" in _USE)
                and 1 <= self.C <= RQ.MAX_B
                and FD._is_aff8(lm)
                and h_R.dtype == mx.bfloat16
            ):
                raw_bf16 = RQ.qmv_rows(h_R.reshape(self.C, 2304), lm["weight"], lm["scales"], lm["biases"])
                out = RH.softcap_logprobs_rows(raw_bf16, softcap, unpadded, argmax=argmax)
                if argmax:
                    toks, lp = out
                    return self._with_stage_deps(toks), lp
                return self._with_stage_deps(out)
        z = self.logits((h_R, h_ref))
        lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
        if argmax:
            return mx.argmax(lp, axis=-1), lp
        return lp

    def can_fast_sample(self, sampler) -> bool:
        """Return True if `sample_step` can run 0-sync fused GPU sampling for `sampler`."""
        if not _fast_sample_enabled() or _std_sampler_params(sampler) is None:
            return False
        if "head" in _PLAIN or not (1 <= self.C <= RQ.MAX_B):
            return False
        if not (self._b1_fast or self._batch_fast or "qmv" in _USE):
            return False
        if not FD._is_aff8(self.model.lm_head):
            return False
        lc = self.tm.args.loop_config
        cfg = self.tm.loopcd
        if lc is not None and cfg.mode == "logits" and cfg.enabled_for(lc["num_passes"]):
            return False
        return True

    def sample_step(self, tokens, sampler):
        """Zero-sync exact-distribution fused GPU sampling step returning `(C,) uint32`."""
        h, h_ref = self._hidden_pair(tokens)
        std = _std_sampler_params(sampler)
        m = self.model
        lm = m.lm_head
        if (
            h_ref is None
            and std is not None
            and "head" not in _PLAIN
            and (self._b1_fast or self._batch_fast or "qmv" in _USE)
            and 1 <= self.C <= RQ.MAX_B
            and FD._is_aff8(lm)
            and h.dtype == mx.bfloat16
        ):
            text = m.args.text
            raw_bf16 = RQ.qmv_rows(h.reshape(self.C, 2304), lm["weight"], lm["scales"], lm["biases"])
            toks = RH.softcap_sample_rows_nosync(
                raw_bf16,
                text.final_logit_softcapping,
                text.unpadded_vocab_size,
                temp=std[0],
                top_p=std[1],
                min_p=std[2],
            )
            return self._with_stage_deps(toks)
        lp = self.logprobs((h, h_ref))
        return self._with_stage_deps(sampler(lp).astype(mx.uint32))

    def step(self, tokens):
        h, h_ref = self._hidden_pair(tokens)
        if h_ref is not None:
            z_R, z_ref = self._pair_logits_softcap(h, h_ref)
            logits = loopcd.combine_logits(z_R, z_ref, self.tm.loopcd, skip_mask=self._loopcd_skip_mask(self.tm.loopcd))
        else:
            if "head" in _FUSED:
                m = self.model
                lm = m.lm_head
                text = m.args.text
                if (
                    "head" not in _PLAIN
                    and (self._b1_fast or self._batch_fast or "qmv" in _USE)
                    and 1 <= self.C <= RQ.MAX_B
                    and FD._is_aff8(lm)
                    and h.dtype == mx.bfloat16
                ):
                    return self._with_stage_deps(
                        RQ.qmv_softcap_argmax_rows(
                            h.reshape(self.C, 2304), lm["weight"], lm["scales"], lm["biases"],
                            text.final_logit_softcapping, text.unpadded_vocab_size,
                        )
                    )
                elif "head" in _PLAIN:
                    logits = lm(h)
                else:
                    with _RowsMode():
                        logits = lm(h)
                return self._with_stage_deps(
                    RH.softcap_argmax_rows(logits, text.final_logit_softcapping, text.unpadded_vocab_size)
                )
            logits = self._project_logits_softcap(h)
        out = []
        for i in range(self.C):
            row = logits[i : i + 1]
            lp = row - mx.logsumexp(row, keepdims=True)
            out.append(mx.argmax(lp, axis=-1))
        return self._with_stage_deps(mx.concatenate(out))
