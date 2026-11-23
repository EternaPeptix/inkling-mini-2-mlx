"""Fused MoE half of a layer visit (fused_ref.py stage E) for B rows in 3 launches, bitwise per row. See docs/FUSED_MOE.md.

    router_rows(x, w)                       K11  x (B,2304) bf16 -> router_logits (B,128) f32
    gateup_rows(logits, ...)                K12  routing + gate_up gather matvec + softcapped swiglu
    down_rows(swiglu, inds, wts, ...)       K13  down gather matvec + left-fold combine + post-mlp conv + residual
    moe_rows(layer, x, residual, cs3)       all three, modules taken from `layer`

Arithmetic is MLX's own: fp_quantized.h helpers (as rows/gather.py), GEMVKernel::run (rows/gemv.py), expert_routing
source, glue swiglu / moe_combine, fastdecode short_conv_decode source.
"""
import math
import os

import mlx.core as mx
import numpy as np

from mlx_lm.kernels.common import scalar
from mlx_lm.kernels.expert_routing import expert_routing

from . import gemv as _gemv
from .gather import mlx_header
from .glue import GLUE_HEADER, flit

__all__ = ["router_rows", "gateup_rows", "down_rows", "moe_rows"]

NSG_GU = int(os.environ.get("MOE_NSG_GU", "8"))   # simdgroups per K12 threadgroup (rows per tile = 4*NSG)
NSG_DN = int(os.environ.get("MOE_NSG_DN", "8"))   # simdgroups per K13 threadgroup (pairs processed in parallel)
TMAX_CAP = int(os.environ.get("MOE_TMAX_CAP", "2"))

# --------------------------------------------------------------------------------------------------------------------
# K11: router gemv (rows/gemv.py gemv_rows_run, with the bf16 -> f32 cast of x in the prologue load)
_ROUTER_HEADER = _gemv._HEADER.replace("const device T* in_vec,   // (B, K) row-major", "const device bfloat16_t* in_vec,")
assert _ROUTER_HEADER != _gemv._HEADER
_router_kernels = {}


def router_rows(x, w):
    """x (B,K) bf16, w (N,K) f32 as stored -> (B,N) f32; row b bitwise = f32(x[b]) @ w.T at M == 1."""
    B, K = x.shape
    N = w.shape[0]
    if x.dtype != mx.bfloat16 or w.dtype != mx.float32 or w.shape[1] != K:
        raise ValueError("router_rows: x bf16 (B,K), w f32 (N,K)")
    cfg = _gemv._cfg(N, K)
    bm, bn, sm, sn, tm, tn = cfg
    blockM = bm * sm * tm
    nt = (N + blockM - 1) // blockM
    key = (B, N, K)
    k = _router_kernels.get(key)
    if k is None:
        src = (
            "threadgroup float tgp_mem[%d];\n"
            "const uint t = threadgroup_position_in_grid.x;\n"
            "gemv_rows_run<float, %d, %d, %d, %d, %d, %d, %d>(w, x, o, %d, %d, tgp_mem, t, "
            "simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
        ) % (B * bn * (blockM + tm) if bn > 1 else 1, B, bm, bn, sm, sn, tm, tn, K, N)
        k = _router_kernels[key] = mx.fast.metal_kernel(
            name="moe_router_b%d_n%d_k%d" % key, input_names=["w", "x"], output_names=["o"],
            header=_ROUTER_HEADER, source=src)
    (o,) = k(inputs=[w, x], grid=(nt * 32, bn, bm), threadgroup=(32, bn, bm),
             output_shapes=[(B, N)], output_dtypes=[mx.float32])
    return o


# --------------------------------------------------------------------------------------------------------------------
_ROUTE_FN = r"""
// expert_routing (vendor/mlx_lm/kernels/expert_routing.py) for ONE row, run by one simdgroup; verbatim arithmetic.
inline void moe_route_row(const device float* row, const device float* bias, uint lane,
                          threadgroup uint* inds, threadgroup float* wts, float eps, float scale) {
    constexpr int E = 128;
    constexpr int K = 8;
    constexpr int R = (E + 31) / 32;
    float score[R];
    float biased[R];
    for (int i = 0; i < R; i++) {
        int e = lane + 32 * i;
        score[i] = 0.0f;
        biased[i] = -INFINITY;
        if (e < E) {
            float x = row[e] * 1.0f;
            float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
            score[i] = x < 0 ? y : 1.0f - y;
            biased[i] = score[i] + bias[e];
        }
    }
    float w[K];
    for (int k = 0; k < K; k++) {
        float best = -INFINITY;
        int best_e = E;
        for (int i = 0; i < R; i++) {
            int e = lane + 32 * i;
            if (biased[i] > best || (biased[i] == best && e < best_e)) {
                best = biased[i];
                best_e = e;
            }
        }
        float top = simd_max(best);
        int winner = simd_min(best == top ? best_e : E);
        float s = 0.0f;
        for (int i = 0; i < R; i++) {
            if (lane + 32 * i == winner) {
                s = score[i];
                biased[i] = -INFINITY;
            }
        }
        w[k] = simd_sum(s);
        if (lane == 0) {
            inds[k] = winner;
        }
    }
    float total = 0.0f;
    for (int k = 0; k < K; k++) {
        total = w[k] + total;
    }
    if (lane == 0) {
        for (int k = 0; k < K; k++) {
            wts[k] = w[k] / (total + eps) * scale;
        }
    }
}
"""

HEADER = mlx_header() + GLUE_HEADER + _ROUTE_FN

# Shared pair loop (rows/gather.py body). @XOFF@: pair pq -> x row index; @EPI@: epilogue after the K loop.
_PAIR_LOOP = r"""
    const int in_w = KK * (BITS) / 8;
    const int in_g = KK / GS;
    const device uint8_t* ws0 = (const device uint8_t*)w + (long)e * NN * in_w
        + (long)out_row * in_w + lane * PPT * 4;
    const device uint8_t* sc0 = scales + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
    float inv_scale_enc = 1.0f;
    if constexpr (HGS) {
        inv_scale_enc = gsc[e] / (F8E4M3_MAX * F4E2M1_MAX);
    }
    for (int tb = 0; tb < cnt; tb += TMAX) {
        int xo[TMAX];
        int pqs[TMAX];
        for (int t = 0; t < TMAX; t++) {
            int pq = (tb + t < cnt) ? (int)PL[tb + t] : (int)PQ0;
            pqs[t] = pq;
            xo[t] = (@XOFF@) * KK + (int)lane * VPT;
        }
        thread U result[TMAX][ROWS];
        for (int t = 0; t < TMAX; t++)
            for (int r = 0; r < ROWS; r++) result[t][r] = 0;
        const device uint8_t* ws = ws0;
        const device uint8_t* sc = sc0;
        int k = 0;
        constexpr int KEND_SUB = FAST ? 0 : BLOCK;
        for (; k < KK - KEND_SUB; k += BLOCK) {
            thread U wv[ROWS][VPT];
            U sv[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                dequant_vals<BITS, VPT>(ws + r * in_w, wv[r]);
                sv[r] = dequantize_scale<U, GS>((sc + r * in_g)[0]);
            }
#pragma unroll
            for (int t = 0; t < TMAX; t++) {
                if (tb + t < cnt) {
                    thread U xt[VPT];
                    load_vector<T, U, VPT>(x + xo[t] + k, xt);
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        result[t][r] += dot_vals<BITS, VPT>(xt, wv[r], sv[r]);
                    }
                }
            }
            ws += BLOCK * 4 / PF;
            sc += BLOCK / GS;
        }
        if constexpr (!FAST) {
            const int rem = clamp((int)(KK - k - (int)lane * VPT), 0, VPT);
            if (rem > 0) {
                for (int t = 0; t < TMAX; t++) {
                    if (tb + t < cnt) {
                        thread U xt[VPT];
                        load_vector_safe<T, U, VPT>(x + xo[t] + k, xt, rem);
                        for (int r = 0; r < ROWS; r++) {
                            const device uint8_t* wl = ws + r * in_w;
                            U s = dequantize_scale<U, GS>((sc + r * in_g)[0]);
                            result[t][r] += qdot_safe<U, VPT, BITS>(wl, xt, s, rem);
                        }
                    }
                }
            }
        }
        @EPI@
    }
"""

_CONSTS = r"""
    constexpr int NPAIR = BB * 8;
    constexpr int PF = 32 / BITS;
    constexpr int PPT = FAST ? 2 : 1;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    constexpr int ROWS = 4;
    typedef float U;
    typedef bfloat16_t T;
    uint3 tid = threadgroup_position_in_grid;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint ttid = thread_index_in_threadgroup;
"""

# The scan, run by one simdgroup for pair P0 (expert e): `dup` = an earlier pair picked e; PL = pairs with expert e
# (all >= P0). Every lane of the simdgroup ends with the same cnt / dup.
_SCAN = r"""
        int cnt = 0;
        bool dup = false;
        for (int base = 0; base < NPAIR; base += 32) {
            int p = base + (int)lane;
            bool match = p < NPAIR && ind_s[p] == e;
            if (match && p < P0) dup = true;
            int take = (match && p >= P0) ? 1 : 0;
            int pre = simd_prefix_exclusive_sum(take);
            if (take) PL[cnt + pre] = (uint)p;
            cnt += simd_sum(take);
        }
        dup = simd_any(dup);
        simdgroup_barrier(mem_flags::mem_threadgroup);
"""


def _gateup_src(route, dbg, gc, uc):
    epi = r"""
#pragma unroll
        for (int t = 0; t < TMAX; t++) {
            float gv[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                U v = simd_sum(result[t][r]);
                if constexpr (HGS) { gv[r] = float(static_cast<T>(v * inv_scale_enc)); }
                else { gv[r] = float(static_cast<T>(v)); }
            }
            if (lane == 0 && tb + t < cnt) {
                const int pq = pqs[t];
                sw[pq * (NN / 2) + out_row / 2] = bfloat16_t(glue_swiglu(gv[0], gv[1], %s, %s));
                sw[pq * (NN / 2) + out_row / 2 + 1] = bfloat16_t(glue_swiglu(gv[2], gv[3], %s, %s));
                %s
            }
        }
""" % (gc, uc, gc, uc, "for (int r = 0; r < ROWS; r++) gu[pq * NN + out_row + r] = bfloat16_t(gv[r]);" if dbg else "")
    loop = _PAIR_LOOP.replace("@XOFF@", "pq / 8").replace("@EPI@", epi).replace("PL[", "pl[").replace("PQ0", "q")
    s = _CONSTS + r"""
    const int q = tid.z;
    threadgroup uint ind_s[NPAIR];
    threadgroup uint pl[NPAIR];
"""
    if route:
        s += r"""
    threadgroup float wt_s[NPAIR];
    for (int b = sg; b < BB; b += NSG)
        moe_route_row(logits + b * 128, bias, lane, ind_s + b * 8, wt_s + b * 8, eps, rscale);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid.y == 0 && q == 0) {
        for (int i = ttid; i < NPAIR; i += 32 * NSG) { inds_o[i] = ind_s[i]; wts_o[i] = wt_s[i]; }
    }
"""
    else:
        s += r"""
    for (int i = ttid; i < NPAIR; i += 32 * NSG) ind_s[i] = inds_in[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
    s += r"""
    const uint e = ind_s[q];
    const int P0 = q;
    threadgroup int dupc[2];
    if (sg == 0) {
""" + _SCAN.replace("PL[", "pl[") + r"""
        if (lane == 0) { dupc[0] = cnt; dupc[1] = dup ? 1 : 0; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dupc[1]) return;
    const int cnt = dupc[0];
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
""" + loop
    return s


def _down_src(dbg, scale_lit):
    epi = r"""
#pragma unroll
        for (int t = 0; t < TMAX; t++) {
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                U v = simd_sum(result[t][r]);
                if (lane == 0 && tb + t < cnt) {
                    if constexpr (HGS) { dn_s[pqs[t] * 4 + r] = float(static_cast<T>(v * inv_scale_enc)); }
                    else { dn_s[pqs[t] * 4 + r] = float(static_cast<T>(v)); }
                }
            }
        }
"""
    loop = _PAIR_LOOP.replace("@XOFF@", "pq").replace("@EPI@", epi).replace("PL[", "plg[").replace("PQ0", "P0")
    s = _CONSTS + r"""
    threadgroup uint ind_s[NPAIR];
    threadgroup uint pl[NSG * NPAIR];
    threadgroup float dn_s[NPAIR * 4];
    for (int i = ttid; i < NPAIR; i += 32 * NSG) ind_s[i] = inds[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint* plg = pl + sg * NPAIR;
    const int out_row = tid.y * ROWS;
    for (int P0 = sg; P0 < NPAIR; P0 += NSG) {
        const uint e = ind_s[P0];
""" + _SCAN.replace("PL[", "plg[") + r"""
        if (!dup) {
""" + loop + r"""
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ttid < BB * 4) {
        const int b = ttid / 4, r = ttid % 4;
        const int c = tid.y * 4 + r;
        float ev[8], ww[8];
        for (int j = 0; j < 8; j++) { ev[j] = dn_s[(b * 8 + j) * 4 + r]; ww[j] = wts[b * 8 + j]; }
        bfloat16_t moe = bfloat16_t(glue_moe_combine8(ev, ww));
        // short_conv_decode (fastdecode._CONV_SRC) arithmetic
        float x0 = float(moe);
        const device float* st = cs3 + b * 3 * CC;
        float r1 = st[CC + c], r2 = st[2 * CC + c];
        float s0 = float(T(st[c])), s1 = float(T(r1)), s2 = float(T(r2));
        float acc = 0.0f;
        acc = fma(float(cw[c * 4 + 0]), s0, acc);
        acc = fma(float(cw[c * 4 + 1]), s1, acc);
        acc = fma(float(cw[c * 4 + 2]), s2, acc);
        acc = fma(float(cw[c * 4 + 3]), x0, acc);
        float conv = float(T(acc));
        T pmc_v = T((conv + x0) / @SCALE@);
        w3[b * 3 * CC + c] = r1;
        w3[b * 3 * CC + CC + c] = r2;
        w3[b * 3 * CC + 2 * CC + c] = x0;
        out[b * CC + c] = bfloat16_t(resid[b * CC + c] + float(pmc_v));
""".replace("@SCALE@", scale_lit)
    if dbg:
        s += r"""
        moe_out[b * CC + c] = moe;
        pmc[b * CC + c] = pmc_v;
    }
    for (int i = ttid; i < NPAIR * 4; i += 32 * NSG) down[(i / 4) * CC + tid.y * 4 + (i % 4)] = bfloat16_t(dn_s[i]);
"""
    else:
        s += "    }\n"
    return s


_kernels = {}


def _geom(w, scales, K, mode, group_size, bits):
    N = w.shape[1]
    if mode not in ("nvfp4", "mxfp8"):
        raise ValueError("moe: mode must be nvfp4 or mxfp8, got %r" % (mode,))
    if K % 32 != 0 or K != scales.shape[-1] * group_size or w.shape[2] * 32 // bits != K or N % 8 != 0:
        raise ValueError("moe: unsupported shape")
    fast = K % ((32 // bits) * 2 * 32) == 0
    return N, fast


def _tmax(B):
    for t in (1, 2, 4, 8, 16):
        if B <= t or t >= TMAX_CAP:
            return t
    return 16


def gateup_rows(logits, bias, x, w, scales, global_scale, *, mode, group_size, bits, route_scale, gate_softcap,
                up_softcap, score_eps=1e-20, route="kernel", inds=None, debug=False, nsg=None):
    """K12. logits (B,128) f32, bias (128,) f32, x (B,K) bf16, expert weights (X,N,K*bits/32) u32 / scales u8 /
    global_scale (X,) f32 or None. Returns [expert_inds (B,8) u32, expert_w (B,8) f32, swiglu (B,8,N/2) bf16
    (, gate_up (B,8,N) bf16 if debug)]. route="kernel": routing recomputed in every threadgroup;
    route="launch": routing is a separate `expert_routing` launch before the gather kernel."""
    B, K = x.shape
    if B > 8 or x.dtype != mx.bfloat16:
        raise ValueError("gateup_rows: B <= 8, x bf16")
    N, fast = _geom(w, scales, K, mode, group_size, bits)
    nsg = nsg or NSG_GU
    if N % (4 * nsg) != 0:
        raise ValueError("N not divisible by the tile")
    in_kernel = route == "kernel"
    if not in_kernel:
        if route != "launch":
            raise ValueError("route must be 'kernel' or 'launch'")
        inds_arr, wts_arr = expert_routing(logits, bias, top_k=8, multiplier=1.0, score_eps=score_eps,
                                           route_scale=route_scale)
    hgs = global_scale is not None
    gsc = global_scale.astype(mx.float32) if hgs else mx.zeros((1,), mx.float32)
    gc, uc = flit(gate_softcap), flit(up_softcap)
    key = ("gu", in_kernel, debug, gc, uc)
    k = _kernels.get(key)
    if k is None:
        ins = ["w", "scales", "gsc", "x"] + (["logits", "bias", "eps", "rscale"] if in_kernel else ["inds_in"])
        outs = (["inds_o", "wts_o"] if in_kernel else []) + ["sw"] + (["gu"] if debug else [])
        k = _kernels[key] = mx.fast.metal_kernel(
            name="moe_gateup_%d%d_%s_%s" % (in_kernel, debug, gc.replace(".", "p"), uc.replace(".", "p")),
            input_names=ins, output_names=outs, source=_gateup_src(in_kernel, debug, gc, uc), header=HEADER,
            ensure_row_contiguous=True)
    ins = [w, scales, gsc, x] + ([logits, bias, scalar(score_eps), scalar(route_scale)] if in_kernel
                                 else [inds_arr.astype(mx.uint32)])
    shapes = ([(B, 8), (B, 8)] if in_kernel else []) + [(B, 8, N // 2)] + ([(B, 8, N)] if debug else [])
    dts = ([mx.uint32, mx.float32] if in_kernel else []) + [mx.bfloat16] + ([mx.bfloat16] if debug else [])
    outs = list(k(inputs=ins,
                  template=[("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast), ("TMAX", _tmax(B)),
                            ("KK", K), ("NN", N), ("BB", B), ("NSG", nsg)],
                  grid=(32 * nsg, N // (4 * nsg), B * 8), threadgroup=(32 * nsg, 1, 1),
                  output_shapes=shapes, output_dtypes=dts))
    return outs if in_kernel else [inds_arr, wts_arr] + outs


def down_rows(sw, inds, wts, residual, cs3, conv_w, w, scales, global_scale, *, mode, group_size, bits, debug=False,
              nsg=None):
    """K13. sw (B,8,K) bf16, inds (B,8) u32, wts (B,8) f32, residual (B,C) f32, cs3 (B,3,C) f32, conv_w post-mlp conv
    weight (C,*,4) bf16, expert weights (X,C,K*bits/32). Returns [out (B,C) bf16, w3 (B,3,C) f32
    (, moe_out (B,C) bf16, post_mlp_conv (B,C) bf16, down (B,8,C) bf16 if debug)]."""
    B, E, K = sw.shape
    C = w.shape[1]
    if B > 8 or E != 8 or sw.dtype != mx.bfloat16 or residual.shape != (B, C) or cs3.shape != (B, 3, C):
        raise ValueError("down_rows: shapes / dtypes")
    if residual.dtype != mx.float32 or cs3.dtype != mx.float32 or conv_w.dtype != mx.bfloat16 or conv_w.size != C * 4:
        raise ValueError("down_rows: residual/cs3 f32, conv weight bf16 (C,4)")
    N, fast = _geom(w, scales, K, mode, group_size, bits)
    nsg = nsg or NSG_DN
    if C % 4 != 0:
        raise ValueError("C % 4")
    hgs = global_scale is not None
    gsc = global_scale.astype(mx.float32) if hgs else mx.zeros((1,), mx.float32)
    scale_lit = repr(float(np.float32(math.sqrt(4 + 1)))) + "f"
    key = ("dn", debug, scale_lit)
    k = _kernels.get(key)
    if k is None:
        ins = ["w", "scales", "gsc", "x", "inds", "wts", "resid", "cs3", "cw"]
        outs = ["out", "w3"] + (["moe_out", "pmc", "down"] if debug else [])
        k = _kernels[key] = mx.fast.metal_kernel(
            name="moe_down_%d" % debug, input_names=ins, output_names=outs, source=_down_src(debug, scale_lit),
            header=HEADER, ensure_row_contiguous=True)
    shapes = [(B, C), (B, 3, C)] + ([(B, C), (B, C), (B, 8, C)] if debug else [])
    dts = [mx.bfloat16, mx.float32] + ([mx.bfloat16] * 3 if debug else [])
    return list(k(inputs=[w, scales, gsc, sw, inds.astype(mx.uint32), wts, residual, cs3, conv_w],
                  template=[("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast), ("TMAX", _tmax(B)),
                            ("KK", K), ("NN", N), ("BB", B), ("NSG", nsg), ("CC", C)],
                  grid=(32 * nsg, C // 4, 1), threadgroup=(32 * nsg, 1, 1), output_shapes=shapes, output_dtypes=dts))


def moe_rows(layer, x, residual, cs3, *, route="kernel", debug=False):
    """The MoE half of a layer visit for B rows. x = mlp_x (B,D) bf16, residual (B,D) f32, cs3 (B,3,D) f32 (post-mlp conv
    state). Returns dict: out (B,D) bf16, w3, expert_inds, expert_w, swiglu, router_logits (+ gate_up, down, moe_out,
    post_mlp_conv if debug)."""
    moe = layer.mlp
    ex = moe.experts
    gu, dn = ex.gate_up_proj, ex.down_proj
    logits = router_rows(x, moe.gate.weight)
    act = ex.act_fn
    r = gateup_rows(logits, moe.gate.e_score_correction_bias, x, gu["weight"], gu["scales"], gu.get("global_scale"),
                    mode=gu.mode, group_size=gu.group_size, bits=gu.bits, route_scale=moe.gate.routed_scaling_factor,
                    gate_softcap=act.gate_softcap, up_softcap=act.up_softcap, route=route, debug=debug)
    inds, wts, sw = r[0], r[1], r[2]
    o = down_rows(sw, inds, wts, residual, cs3, layer.post_mlp_convolution.conv1d.weight, dn["weight"], dn["scales"],
                  dn.get("global_scale"), mode=dn.mode, group_size=dn.group_size, bits=dn.bits, debug=debug)
    res = dict(router_logits=logits, expert_inds=inds, expert_w=wts, swiglu=sw, out=o[0], w3=o[1])
    if debug:
        res.update(gate_up=r[3], moe_out=o[2], post_mlp_conv=o[3], down=o[4])
    return res
