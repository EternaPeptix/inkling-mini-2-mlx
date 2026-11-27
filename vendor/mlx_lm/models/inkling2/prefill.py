"""Fused bit-exact Metal kernels for L > 1 prefill (LOOPLING_FAST_PREFILL=1 default)."""
import math
import os

import mlx.core as mx
import numpy as np

from mlx_lm.models.inkling2.rows.gather import mlx_header as _fpq_mlx_header
from mlx_lm.models.inkling2.rows.glue import GLUE_HEADER
from mlx_lm.models.inkling2.rows.moe import _DOT_BF16, _FP_LUT_HEADER
from mlx_lm.models.inkling2.rows.proj import _maybe_bf16_w

_FAST = os.environ.get("LOOPLING_FAST_PREFILL", "1") not in ("0", "")
_PF_MOE_GMM = os.environ.get("LOOPLING_PF_MOE_GMM", "1") not in ("0", "")
_PF_MOE_SHORT = os.environ.get("LOOPLING_PF_MOE_SHORT", "1") not in ("0", "")
_PF_MOE_SHORT_TILED = os.environ.get("LOOPLING_PF_MOE_SHORT_TILED", "1") not in ("0", "")
_PF_MOE_SHORT_NSG = int(os.environ.get("LOOPLING_PF_MOE_SHORT_NSG", "2"))
_PF_MOE_SHORT_TMAX = int(os.environ.get("LOOPLING_PF_MOE_SHORT_TMAX", "4"))
_PF_MOE_BM_ENV = int(os.environ.get("LOOPLING_PF_MOE_BM", "0"))
_PF_MOE_CFG_ENV = os.environ.get("LOOPLING_PF_MOE_CFG", "")
_PF_MOE_CFG = tuple(int(x) for x in _PF_MOE_CFG_ENV.split(",")) if _PF_MOE_CFG_ENV else None
_PF_MOE_MIN_L = int(os.environ.get("LOOPLING_PF_MOE_MIN_L", "64"))
_PF_AFF8_QMM = os.environ.get("LOOPLING_PF_AFF8_QMM", "1") not in ("0", "")
_PF_AFF8_WK = os.environ.get("LOOPLING_PF_AFF8_WK", "1") not in ("0", "")
_PF_AFF8_CFG_ENV = os.environ.get("LOOPLING_PF_AFF8_CFG", "64,32,32,2,1")
_PF_AFF8_CFG = tuple(int(x) for x in _PF_AFF8_CFG_ENV.split(","))
_IS_G15D = (
    mx.metal.is_available()
    and getattr(mx, "device_info", None) is not None
    and mx.device_info().get("architecture", "") == "applegpu_g15d"
)
_SPLASH_MSGGATE = os.environ.get("LOOPLING_SPLASH_MSGGATE", "0") not in ("0", "", "false", "False")
_D = 2304
_TG_RMS = 576
_DUMMY_F32 = mx.zeros((256,), mx.float32)  # >= 512 B ensures Metal device address space
_DUMMY_BF16 = mx.zeros((256,), mx.bfloat16)
_DUMMY_U8 = mx.zeros((512,), mx.uint8)
_DUMMY_I32 = mx.zeros((128,), mx.int32)
_S5_LIT = repr(float(np.float32(math.sqrt(5.0)))) + "f"
_S9_LIT = repr(float(np.float32(math.sqrt(9.0)))) + "f"


def active(dtype: mx.Dtype, x: mx.array) -> bool:
    return (
        _FAST
        and dtype == mx.bfloat16
        and x.ndim >= 3
        and x.shape[1] > 1
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


def enabled(x: mx.array) -> bool:
    return active(x.dtype, x)


def msg_gate_ok(mod, h: mx.array, msg: mx.array, ln) -> bool:
    return (
        h.dtype == mx.float32
        and msg.dtype == mx.float32
        and h.shape[-1] == _D
        and getattr(mod.down_proj, "weight", None) is not None
        and mod.down_proj.weight.dtype == mx.bfloat16
        and mod.forget_gate_proj.weight.dtype == mx.float32
        and mod.message_gate_proj.weight.dtype == mx.float32
        and ln.weight.dtype == mx.bfloat16
    )


_kernels: dict = {}


def _k(name, ins, outs, src):
    k = _kernels.get(name)
    if k is None:
        k = _kernels[name] = mx.fast.metal_kernel(
            name=name,
            input_names=list(ins),
            output_names=list(outs),
            source=src,
            header=GLUE_HEADER,
            ensure_row_contiguous=True,
        )
    return k


def _k_mma(name, ins, outs, src, extra_hdr=""):
    k = _kernels.get(name)
    if k is None:
        k = _kernels[name] = mx.fast.metal_kernel(
            name=name,
            input_names=list(ins),
            output_names=list(outs),
            source=src,
            header="#include <metal_simdgroup_matrix>\n" + GLUE_HEADER + extra_hdr,
            ensure_row_contiguous=True,
        )
    return k


# ---------------------------------------------------------------------------------------------------- Fused bf16 -> f32 Router GEMM
_ROUTER_BF16_SRC = r"""
    constexpr int BM = 32, BN = 64, BK = 32, LDX = 40, TG = 128, TM = 2, TN = 4;
    const uint tgn = threadgroup_position_in_grid.x;
    const uint tgm = threadgroup_position_in_grid.y;
    const uint m0 = tgm * BM, n0 = tgn * BN;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg_m = (sg / 2) * 16, sg_n = (sg % 2) * 32;
    const ushort fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const ushort fn = ((lane >> 2) & 2) * 2 | ((lane & 1) * 2);

    threadgroup float Xs[BM * LDX];
    threadgroup float Ws[BN * LDX];
    const uint x_mr = tid / 4, x_mk = (tid % 4) * 8;
    const bool x_ok = (m0 + x_mr) < uint(MM);
    const device bfloat16_t* x_ptr = x + (size_t)(m0 + x_mr) * KK + x_mk;
    const uint w_nr = tid / 2, w_koff = (tid % 2) * 16;
    const device float* w_ptr = w + (size_t)(n0 + w_nr) * KK + w_koff;

    simdgroup_float8x8 C[TM][TN];
    for (int tm = 0; tm < TM; ++tm)
        for (int tn = 0; tn < TN; ++tn)
            C[tm][tn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k = 0; k < KK; k += BK) {
        if (x_ok) {
            uint4 raw = ((const device uint4*)(x_ptr + k))[0];
            ((threadgroup float4*)(Xs + x_mr * LDX + x_mk))[0] = float4(as_type<bfloat4>(raw.xy));
            ((threadgroup float4*)(Xs + x_mr * LDX + x_mk))[1] = float4(as_type<bfloat4>(raw.zw));
        } else {
            ((threadgroup float4*)(Xs + x_mr * LDX + x_mk))[0] = float4(0);
            ((threadgroup float4*)(Xs + x_mr * LDX + x_mk))[1] = float4(0);
        }
        const device float4* wp4 = (const device float4*)(w_ptr + k);
        threadgroup float4* wdp = (threadgroup float4*)(Ws + w_nr * LDX + w_koff);
        wdp[0] = wp4[0]; wdp[1] = wp4[1]; wdp[2] = wp4[2]; wdp[3] = wp4[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int ks = 0; ks < BK; ks += 8) {
            simdgroup_float8x8 A[TM], B[TN];
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tm = 0; tm < TM; ++tm) {
                const float2 av = *((const threadgroup float2*)(Xs + (sg_m + tm * 8 + fm) * LDX + ks + fn));
                A[tm].thread_elements()[0] = av.x; A[tm].thread_elements()[1] = av.y;
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < TN; ++tn) {
                const threadgroup float* bp = Ws + (sg_n + tn * 8 + fn) * LDX + ks + fm;
                B[tn].thread_elements()[0] = bp[0]; B[tn].thread_elements()[1] = bp[LDX];
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tm = 0; tm < TM; ++tm)
                for (int tn = 0; tn < TN; ++tn)
                    simdgroup_multiply_accumulate(C[tm][tn], A[tm], B[tn], C[tm][tn]);
        }
        if (k + BK < KK) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int tm = 0; tm < TM; ++tm) {
        const uint mr_o = m0 + sg_m + uint(tm) * 8u + fm;
        if (mr_o < uint(MM)) {
            device float* dst = out + (size_t)mr_o * 128;
            for (int tn = 0; tn < TN; ++tn) {
                const uint nc_o = n0 + sg_n + uint(tn) * 8u + fn;
                ((device float2*)(dst + nc_o))[0] = float2(C[tm][tn].thread_elements()[0], C[tm][tn].thread_elements()[1]);
            }
        }
    }
"""


def router_logits_bf16(x: mx.array, w: mx.array) -> mx.array:
    """Bit-exact fused bf16->f32 router GEMM: (B, L, 2304) bf16 @ (128, 2304).T f32 -> (B, L, 128) f32."""
    M = x.size // 2304
    k = _k_mma("pf_router_bf16", ("x", "w"), ("out",), _ROUTER_BF16_SRC)
    return k(
        inputs=[x, w],
        template=[("MM", M), ("KK", 2304)],
        grid=(128 * 2, (M + 31) // 32, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[x.shape[:-1] + (128,)],
        output_dtypes=[mx.float32],
    )[0]


# ---------------------------------------------------------------------------------------------------- Wave 6B: Custom BMx64x32 Grouped-GEMM MoE (NVFP4 & MXFP8)
_MOE_CSORT_TILES_SRC = r"""
    constexpr uint NSG = 32;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    threadgroup atomic_uint hist[NSG][EE];
    threadgroup uint exp_tot[EE];
    threadgroup uint tile_cnt[EE];
    threadgroup uint sg_sums[8];
    threadgroup uint sg_tsums[8];

    for (uint i = tid; i < NSG * uint(EE); i += 1024) {
        atomic_store_explicit(&hist[i / uint(EE)][i % uint(EE)], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk = (uint(PP) + NSG - 1) / NSG;
    uint s0 = sg * chunk;
    uint s1 = min(s0 + chunk, uint(PP));
    for (uint p = s0 + lane; p < s1; p += 32) {
        uint e = inds[p];
        atomic_fetch_add_explicit(&hist[sg][e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(EE)) {
        uint acc = 0;
        for (uint s = 0; s < NSG; s++) {
            uint c = atomic_load_explicit(&hist[s][tid], memory_order_relaxed);
            atomic_store_explicit(&hist[s][tid], acc, memory_order_relaxed);
            acc += c;
        }
        exp_tot[tid] = acc;
        tile_cnt[tid] = (acc + uint(BM) - 1) / uint(BM);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint NESG = uint(EE) / 32;
    if (tid < uint(EE)) {
        uint v = exp_tot[tid];
        uint ex = simd_prefix_exclusive_sum(v);
        if (lane == 31) sg_sums[sg] = ex + v;
        exp_tot[tid] = ex;

        uint tv = tile_cnt[tid];
        uint tex = simd_prefix_exclusive_sum(tv);
        if (lane == 31) sg_tsums[sg] = tex + tv;
        tile_cnt[tid] = tex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        uint sv = (tid < NESG) ? sg_sums[tid] : 0u;
        uint sp = simd_prefix_exclusive_sum(sv);
        if (tid < NESG) sg_sums[tid] = sp;

        uint stv = (tid < NESG) ? sg_tsums[tid] : 0u;
        uint stp = simd_prefix_exclusive_sum(stv);
        if (tid < NESG) sg_tsums[tid] = stp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(EE)) {
        uint base = exp_tot[tid] + sg_sums[sg];
        exp_off[tid] = base;
        for (uint s = 0; s < NSG; s++) {
            uint off = atomic_load_explicit(&hist[s][tid], memory_order_relaxed);
            atomic_store_explicit(&hist[s][tid], base + off, memory_order_relaxed);
        }
        uint tbase = tile_cnt[tid] + sg_tsums[sg];
        uint nte = ((tid + 1 == uint(EE) ? uint(PP) : (exp_tot[tid + 1] + sg_sums[(tid + 1) >> 5])) - base + uint(BM) - 1) / uint(BM);
        for (uint mt = 0; mt < nte; mt++) {
            tile_meta[tbase + mt] = (tid << 20) | mt;
        }
        if (tid == uint(EE) - 1) {
            exp_off[EE] = uint(PP);
            uint tot_t = tbase + nte;
            for (uint t = tot_t; t < uint(MAX_TILES); t++) {
                tile_meta[t] = 0xFFFFFFFFu;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = s0 + lane; p < s1; p += 32) {
        uint e = inds[p];
        uint pos = atomic_fetch_add_explicit(&hist[sg][e], 1u, memory_order_relaxed);
        lhs_idx[pos] = p >> 3;
        inv_order[p] = pos;
    }
"""

_MOE_GMM_SRC = r"""
    constexpr int LDX = BK + 8;
    constexpr int TG = WM * WN * 32;
    constexpr int TM = (BM / WM) / 8;
    constexpr int TN = (BN / WN) / 8;
    constexpr int XPT = (BM * BK) / TG;
    constexpr int WPT = (BN * BK) / TG;
    constexpr int W_ROW_BYTES = KK * BITS / 8;
    constexpr int S_ROW_BYTES = KK / GS;

    const uint tgm = threadgroup_position_in_grid.x;
    const uint tgn = threadgroup_position_in_grid.y;
    const uint meta = tile_meta[tgm];
    if (meta == 0xFFFFFFFFu) return;

    const uint e = meta >> 20;
    const uint mt = meta & 0xFFFFFu;
    const uint m_base = exp_off[e] + mt * uint(BM);
    const uint m_end = exp_off[e + 1];
    const uint m_cnt = min(uint(BM), m_end - m_base);
    const uint n0 = tgn * uint(BN);

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg_m = (sg / uint(WN)) * uint(TM * 8);
    const uint sg_n = (sg % uint(WN)) * uint(TN * 8);
    const ushort fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const ushort fn = ((lane >> 2) & 2) * 2 | ((lane & 1) * 2);

    threadgroup bfloat16_t Xs[BM * LDX];
    threadgroup bfloat16_t Ws[BN * LDX];

    constexpr uint XTPR = uint(BK / XPT);
    const uint x_mr = tid / XTPR;
    const uint x_mk = (tid % XTPR) * uint(XPT);
    const bool x_ok = x_mr < m_cnt;
    const uint x_row = x_ok ? (USE_LHS ? lhs_idx[m_base + x_mr] : (m_base + x_mr)) : 0u;
    const device bfloat16_t* x_ptr = x + (size_t)x_row * KK + x_mk;

    constexpr uint WTPR = uint(BK / WPT);
    const uint w_nr = tid / WTPR;
    const uint w_koff = (tid % WTPR) * uint(WPT);
    const device uint8_t* w_ptr = (const device uint8_t*)w
        + ((size_t)e * NN + (n0 + w_nr)) * W_ROW_BYTES + (w_koff * BITS / 8);
    const device uint8_t* s_ptr = scales
        + ((size_t)e * NN + (n0 + w_nr)) * S_ROW_BYTES + (w_koff / GS);

    constexpr bool USE_FP4_LUT = (BM >= 32);
    float inv_scale_enc = 1.0f;
    if constexpr (HGS) {
        inv_scale_enc = gsc[e] / (F8E4M3_MAX * F4E2M1_MAX);
    }
    const float inv_sc_fold = (BITS == 4)
        ? (USE_FP4_LUT ? inv_scale_enc : (inv_scale_enc * 16384.0f))
        : (inv_scale_enc * 256.0f);

    simdgroup_float8x8 C[TM][TN];
    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            C[tm][tn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    threadgroup uint4* xd = (threadgroup uint4*)(Xs + x_mr * LDX + x_mk);
    threadgroup uint2* xd2 = (threadgroup uint2*)(Xs + x_mr * LDX + x_mk);
    threadgroup uint4* wd4 = (threadgroup uint4*)(Ws + w_nr * LDX + w_koff);
    const threadgroup bfloat16_t* as_base = Xs + (sg_m + fm) * LDX + fn;
    const threadgroup bfloat16_t* bs_base = Ws + (sg_n + fn) * LDX + fm;

    for (int k = 0; k < KK; k += BK) {
        // 1. Load X tile (coalesced)
        if constexpr (XPT == 32) {
            if (x_ok) {
                const device uint4* xp = (const device uint4*)(x_ptr + k);
                xd[0] = xp[0]; xd[1] = xp[1]; xd[2] = xp[2]; xd[3] = xp[3];
            } else {
                xd[0] = uint4(0); xd[1] = uint4(0); xd[2] = uint4(0); xd[3] = uint4(0);
            }
        } else if constexpr (XPT == 16) {
            if (x_ok) {
                const device uint4* xp = (const device uint4*)(x_ptr + k);
                xd[0] = xp[0];
                xd[1] = xp[1];
            } else {
                xd[0] = uint4(0);
                xd[1] = uint4(0);
            }
        } else if constexpr (XPT == 8) {
            xd[0] = x_ok ? ((const device uint4*)(x_ptr + k))[0] : uint4(0);
        } else {
            xd2[0] = x_ok ? ((const device uint2*)(x_ptr + k))[0] : uint2(0);
        }

        // 2. Load + dequant W tile (exact fp_quantized.h arithmetic, folded power-of-2 scale + 128-bit TGSM stores)
        const device uint8_t* wp = w_ptr + (k * BITS / 8);
        const device uint8_t* sp = s_ptr + (k / GS);
        if constexpr (BITS == 4) {
            if constexpr (WPT == 8) {
                const float sc = gmm_dequant_scale<GS>(sp[0]) * inv_sc_fold;
                const uint v0 = ((const device uint*)wp)[0];
                if constexpr (USE_FP4_LUT) {
                    const uchar4 bs = as_type<uchar4>(v0);
                    const float4 fa = float4(float2(FP4_H2_LUT[bs.x]), float2(FP4_H2_LUT[bs.y]));
                    const uint2 a2 = as_type<uint2>(bfloat4(fa * sc));
                    const float4 fb = float4(float2(FP4_H2_LUT[bs.z]), float2(FP4_H2_LUT[bs.w]));
                    const uint2 b2 = as_type<uint2>(bfloat4(fb * sc));
                    wd4[0] = uint4(a2.x, a2.y, b2.x, b2.y);
                } else {
                    ushort4 u0 = ushort4(v0, v0 >> 4, v0 >> 8, v0 >> 12);
                    half4 h0 = as_type<half4>((u0 & ushort4(7)) << 9);
                    h0 = select(h0, -h0, (u0 & ushort4(8)) != ushort4(0));
                    uint2 a2 = as_type<uint2>(bfloat4(float4(h0) * sc));
                    ushort4 u1 = ushort4(v0 >> 16, v0 >> 20, v0 >> 24, v0 >> 28);
                    half4 h1 = as_type<half4>((u1 & ushort4(7)) << 9);
                    h1 = select(h1, -h1, (u1 & ushort4(8)) != ushort4(0));
                    uint2 b2 = as_type<uint2>(bfloat4(float4(h1) * sc));
                    wd4[0] = uint4(a2.x, a2.y, b2.x, b2.y);
                }
            } else if constexpr (WPT == 16) {
                const float sc = gmm_dequant_scale<GS>(sp[0]) * inv_sc_fold;
                const uint2 raw = ((const device uint2*)wp)[0];
                const uint vs[2] = {raw.x, raw.y};
                #pragma clang loop unroll(full)
                for (int gi = 0; gi < 2; ++gi) {
                    const uint vv = vs[gi];
                    if constexpr (USE_FP4_LUT) {
                        const uchar4 bs = as_type<uchar4>(vv);
                        const float4 fa = float4(float2(FP4_H2_LUT[bs.x]), float2(FP4_H2_LUT[bs.y]));
                        const uint2 a2 = as_type<uint2>(bfloat4(fa * sc));
                        const float4 fb = float4(float2(FP4_H2_LUT[bs.z]), float2(FP4_H2_LUT[bs.w]));
                        const uint2 b2 = as_type<uint2>(bfloat4(fb * sc));
                        wd4[gi] = uint4(a2.x, a2.y, b2.x, b2.y);
                    } else {
                        ushort4 ua = ushort4(vv, vv >> 4, vv >> 8, vv >> 12);
                        half4 ha = as_type<half4>((ua & ushort4(7)) << 9);
                        ha = select(ha, -ha, (ua & ushort4(8)) != ushort4(0));
                        uint2 a2 = as_type<uint2>(bfloat4(float4(ha) * sc));
                        ushort4 ub = ushort4(vv >> 16, vv >> 20, vv >> 24, vv >> 28);
                        half4 hb = as_type<half4>((ub & ushort4(7)) << 9);
                        hb = select(hb, -hb, (ub & ushort4(8)) != ushort4(0));
                        uint2 b2 = as_type<uint2>(bfloat4(float4(hb) * sc));
                        wd4[gi] = uint4(a2.x, a2.y, b2.x, b2.y);
                    }
                }
            } else {
                const float sc0 = gmm_dequant_scale<GS>(sp[0]) * inv_sc_fold;
                const float sc1 = gmm_dequant_scale<GS>(sp[1]) * inv_sc_fold;
                const uint4 raw = ((const device uint4*)wp)[0];
                const uint vs[4] = {raw.x, raw.y, raw.z, raw.w};
                #pragma clang loop unroll(full)
                for (int gi = 0; gi < 4; ++gi) {
                    const float sc = (gi < 2) ? sc0 : sc1;
                    const uint vv = vs[gi];
                    if constexpr (USE_FP4_LUT) {
                        const uchar4 bs = as_type<uchar4>(vv);
                        const float4 fa = float4(float2(FP4_H2_LUT[bs.x]), float2(FP4_H2_LUT[bs.y]));
                        const uint2 a2 = as_type<uint2>(bfloat4(fa * sc));
                        const float4 fb = float4(float2(FP4_H2_LUT[bs.z]), float2(FP4_H2_LUT[bs.w]));
                        const uint2 b2 = as_type<uint2>(bfloat4(fb * sc));
                        wd4[gi] = uint4(a2.x, a2.y, b2.x, b2.y);
                    } else {
                        ushort4 ua = ushort4(vv, vv >> 4, vv >> 8, vv >> 12);
                        half4 ha = as_type<half4>((ua & ushort4(7)) << 9);
                        ha = select(ha, -ha, (ua & ushort4(8)) != ushort4(0));
                        uint2 a2 = as_type<uint2>(bfloat4(float4(ha) * sc));
                        ushort4 ub = ushort4(vv >> 16, vv >> 20, vv >> 24, vv >> 28);
                        half4 hb = as_type<half4>((ub & ushort4(7)) << 9);
                        hb = select(hb, -hb, (ub & ushort4(8)) != ushort4(0));
                        uint2 b2 = as_type<uint2>(bfloat4(float4(hb) * sc));
                        wd4[gi] = uint4(a2.x, a2.y, b2.x, b2.y);
                    }
                }
            }
        } else {
            const float sc = float(dequantize_scale<bfloat16_t, GS>(sp[0])) * inv_sc_fold;
            if constexpr (WPT == 8) {
                const uint2 raw = ((const device uint2*)wp)[0];
                const uchar4 b0 = as_type<uchar4>(raw.x);
                const uchar4 b1 = as_type<uchar4>(raw.y);
                ushort4 u0 = ushort4(b0.x, b0.y, b0.z, b0.w);
                half4 h0 = as_type<half4>((u0 & ushort4(127)) << 7);
                h0 = select(h0, -h0, (u0 & ushort4(128)) != ushort4(0));
                uint2 a2 = as_type<uint2>(bfloat4(float4(h0) * sc));
                ushort4 u1 = ushort4(b1.x, b1.y, b1.z, b1.w);
                half4 h1 = as_type<half4>((u1 & ushort4(127)) << 7);
                h1 = select(h1, -h1, (u1 & ushort4(128)) != ushort4(0));
                uint2 b2 = as_type<uint2>(bfloat4(float4(h1) * sc));
                wd4[0] = uint4(a2.x, a2.y, b2.x, b2.y);
            } else if constexpr (WPT == 16) {
                const uint4 raw = ((const device uint4*)wp)[0];
                const uint ws[4] = {raw.x, raw.y, raw.z, raw.w};
                #pragma clang loop unroll(full)
                for (int gi = 0; gi < 2; ++gi) {
                    const uchar4 b0 = as_type<uchar4>(ws[gi * 2]);
                    const uchar4 b1 = as_type<uchar4>(ws[gi * 2 + 1]);
                    ushort4 u0 = ushort4(b0.x, b0.y, b0.z, b0.w);
                    half4 h0 = as_type<half4>((u0 & ushort4(127)) << 7);
                    h0 = select(h0, -h0, (u0 & ushort4(128)) != ushort4(0));
                    uint2 a2 = as_type<uint2>(bfloat4(float4(h0) * sc));
                    ushort4 u1 = ushort4(b1.x, b1.y, b1.z, b1.w);
                    half4 h1 = as_type<half4>((u1 & ushort4(127)) << 7);
                    h1 = select(h1, -h1, (u1 & ushort4(128)) != ushort4(0));
                    uint2 b2 = as_type<uint2>(bfloat4(float4(h1) * sc));
                    wd4[gi] = uint4(a2.x, a2.y, b2.x, b2.y);
                }
            } else {
                const device uint4* wp4 = (const device uint4*)wp;
                const uint4 r0 = wp4[0], r1 = wp4[1];
                const uint ws[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
                #pragma clang loop unroll(full)
                for (int gi = 0; gi < 4; ++gi) {
                    const uchar4 b0 = as_type<uchar4>(ws[gi * 2]);
                    const uchar4 b1 = as_type<uchar4>(ws[gi * 2 + 1]);
                    ushort4 u0 = ushort4(b0.x, b0.y, b0.z, b0.w);
                    half4 h0 = as_type<half4>((u0 & ushort4(127)) << 7);
                    h0 = select(h0, -h0, (u0 & ushort4(128)) != ushort4(0));
                    uint2 a2 = as_type<uint2>(bfloat4(float4(h0) * sc));
                    ushort4 u1 = ushort4(b1.x, b1.y, b1.z, b1.w);
                    half4 h1 = as_type<half4>((u1 & ushort4(127)) << 7);
                    h1 = select(h1, -h1, (u1 & ushort4(128)) != ushort4(0));
                    uint2 b2 = as_type<uint2>(bfloat4(float4(h1) * sc));
                    wd4[gi] = uint4(a2.x, a2.y, b2.x, b2.y);
                }
            }
        }

        if constexpr (TG == 32) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // 3. Steel-scheduled MMA: explicit simdgroup_barrier(mem_none) + hoisted 32-bit TG loads
        #pragma clang loop unroll(full)
        for (int ks = 0; ks < BK; ks += 8) {
            simdgroup_bfloat8x8 A[TM];
            simdgroup_bfloat8x8 B[TN];
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tm = 0; tm < TM; ++tm) {
                const bfloat2 av = *((const threadgroup bfloat2*)(as_base + (tm * 8) * LDX + ks));
                A[tm].thread_elements()[0] = av.x;
                A[tm].thread_elements()[1] = av.y;
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < TN; ++tn) {
                const threadgroup bfloat16_t* bp = bs_base + (tn * 8) * LDX + ks;
                B[tn].thread_elements()[0] = bp[0];
                B[tn].thread_elements()[1] = bp[LDX];
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tm = 0; tm < TM; ++tm) {
                #pragma clang loop unroll(full)
                for (int tn = 0; tn < TN; ++tn) {
                    simdgroup_multiply_accumulate(C[tm][tn], A[tm], B[tn], C[tm][tn]);
                }
            }
        }

        if (k + BK < KK) {
            if constexpr (TG == 32) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // Store epilogue (fn is even -> te[0]=even/gate, te[1]=odd/up)
    if constexpr (FUSED_SWIGLU) {
        const float gc = caps[0], uc = caps[1];
        for (int tm = 0; tm < TM; ++tm) {
            const uint mr_o = sg_m + uint(tm) * 8u + fm;
            if (mr_o < m_cnt) {
                device bfloat16_t* dst = out + (size_t)(m_base + mr_o) * (NN / 2);
                for (int tn = 0; tn < TN; ++tn) {
                    const uint nc_o = n0 + sg_n + uint(tn) * 8u + fn;
                    const float g = float(bfloat16_t(C[tm][tn].thread_elements()[0]));
                    const float u = float(bfloat16_t(C[tm][tn].thread_elements()[1]));
                    dst[nc_o >> 1] = bfloat16_t(glue_swiglu(g, u, gc, uc));
                }
            }
        }
    } else {
        for (int tm = 0; tm < TM; ++tm) {
            const uint mr_o = sg_m + uint(tm) * 8u + fm;
            if (mr_o < m_cnt) {
                device bfloat16_t* dst = out + (size_t)(m_base + mr_o) * NN;
                for (int tn = 0; tn < TN; ++tn) {
                    const uint nc_o = n0 + sg_n + uint(tn) * 8u + fn;
                    const bfloat2 ov = bfloat2(
                        bfloat16_t(C[tm][tn].thread_elements()[0]),
                        bfloat16_t(C[tm][tn].thread_elements()[1]));
                    ((device uint*)(dst + nc_o))[0] = as_type<uint>(ov);
                }
            }
        }
    }
"""


_FP4_V16 = (
    "0.0h", "0.5h", "1.0h", "1.5h", "2.0h", "3.0h", "4.0h", "6.0h",
    "-0.0h", "-0.5h", "-1.0h", "-1.5h", "-2.0h", "-3.0h", "-4.0h", "-6.0h",
)
_FP4_H2_LUT_HEADER = (
    "constant constexpr half2 FP4_H2_LUT[256] = {"
    + ",".join(f"{{{_FP4_V16[b & 15]},{_FP4_V16[b >> 4]}}}" for b in range(256))
    + "};\n"
)


def _k_moe_gmm():
    k = _kernels.get("pf_moe_gmm")
    if k is None:
        sc_hdr = (
            _FP_LUT_HEADER
            + "\ntemplate <int GS> inline float gmm_dequant_scale(uint8_t s) {\n"
            + "  if constexpr (GS == 16) return FP8_E4M3_LUT[s];\n"
            + "  else return float(dequantize_scale<bfloat16_t, GS>(s));\n}\n"
            if _FP_LUT_HEADER
            else "template <int GS> inline float gmm_dequant_scale(uint8_t s) { return float(dequantize_scale<bfloat16_t, GS>(s)); }\n"
        )
        hdr = (
            "#include <metal_simdgroup_matrix>\n"
            + _fpq_mlx_header()
            + "\n"
            + _FP4_H2_LUT_HEADER
            + sc_hdr
            + "\n"
            + GLUE_HEADER
        )
        k = _kernels["pf_moe_gmm"] = mx.fast.metal_kernel(
            name="pf_moe_gmm",
            input_names=["x", "lhs_idx", "exp_off", "tile_meta", "w", "scales", "gsc", "caps"],
            output_names=["out"],
            source=_MOE_GMM_SRC,
            header=hdr,
            ensure_row_contiguous=True,
        )
    return k


# ---------------------------------------------------------------------------------------------------- Wave 6D/8B: Steel-scheduled BMxBNxBK Affine-8 Dense qmm_t (gs=64)
_AFF8_QMM_SRC = r"""
    constexpr int GS = 64;
    constexpr int LDX = BK + 8;
    constexpr int TG = WM * WN * 32;
    constexpr int TM = (BM / WM) / 8;
    constexpr int TN = (BN / WN) / 8;
    constexpr int XPT = (BM * BK) / TG;
    constexpr int WPT = (BN * BK) / TG;

    const uint tgn = threadgroup_position_in_grid.x;
    const uint tgm = threadgroup_position_in_grid.y;
    const uint m0 = tgm * uint(BM);
    const uint n0 = tgn * uint(BN);

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg_m = (sg / uint(WN)) * uint(TM * 8);
    const uint sg_n = (sg % uint(WN)) * uint(TN * 8);
    const ushort fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const ushort fn = ((lane >> 2) & 2) * 2 | ((lane & 1) * 2);

    threadgroup bfloat16_t Xs[BM * LDX];
    threadgroup bfloat16_t Ws[BN * LDX];

    constexpr uint XTPR = uint(BK / XPT);
    const uint x_mr = tid / XTPR;
    const uint x_mk = (tid % XTPR) * uint(XPT);
    const bool x_ok = (m0 + x_mr) < uint(MM);
    const device bfloat16_t* x_ptr = x + (size_t)(m0 + x_mr) * KK + x_mk;

    constexpr uint WTPR = uint(BK / WPT);
    const uint w_nr = tid / WTPR;
    const uint w_koff = (tid % WTPR) * uint(WPT);
    const device uint8_t* w_ptr = (const device uint8_t*)w + (size_t)(n0 + w_nr) * KK + w_koff;
    const device bfloat16_t* s_ptr = scales + (size_t)(n0 + w_nr) * (KK / GS);
    const device bfloat16_t* b_ptr = biases + (size_t)(n0 + w_nr) * (KK / GS);

    simdgroup_float8x8 C[TM][TN];
    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            C[tm][tn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    threadgroup uint4* xd = (threadgroup uint4*)(Xs + x_mr * LDX + x_mk);
    threadgroup uint2* xd2 = (threadgroup uint2*)(Xs + x_mr * LDX + x_mk);
    threadgroup uint4* wd4 = (threadgroup uint4*)(Ws + w_nr * LDX + w_koff);
    const threadgroup bfloat16_t* as_base = Xs + (sg_m + fm) * LDX + fn;
    const threadgroup bfloat16_t* bs_base = Ws + (sg_n + fn) * LDX + fm;

    float sc = 0.0f, bi = 0.0f;
    for (int k = 0; k < KK; k += BK) {
        if constexpr (XPT == 32) {
            if (x_ok) {
                const device uint4* xp = (const device uint4*)(x_ptr + k);
                xd[0] = xp[0]; xd[1] = xp[1]; xd[2] = xp[2]; xd[3] = xp[3];
            } else {
                xd[0] = uint4(0); xd[1] = uint4(0); xd[2] = uint4(0); xd[3] = uint4(0);
            }
        } else if constexpr (XPT == 16) {
            if (x_ok) {
                const device uint4* xp = (const device uint4*)(x_ptr + k);
                xd[0] = xp[0];
                xd[1] = xp[1];
            } else {
                xd[0] = uint4(0);
                xd[1] = uint4(0);
            }
        } else if constexpr (XPT == 8) {
            xd[0] = x_ok ? ((const device uint4*)(x_ptr + k))[0] : uint4(0);
        } else {
            xd2[0] = x_ok ? ((const device uint2*)(x_ptr + k))[0] : uint2(0);
        }

        if constexpr (BK == 64) {
            sc = float(s_ptr[0]); bi = float(b_ptr[0]);
            s_ptr += 1; b_ptr += 1;
        } else if ((k & 63) == 0) {
            sc = float(s_ptr[0]); bi = float(b_ptr[0]);
            s_ptr += 1; b_ptr += 1;
        }
        const device uint8_t* wp = w_ptr + k;
        if constexpr (WPT == 8) {
            const uint2 raw = ((const device uint2*)wp)[0];
            const uchar4 c0 = as_type<uchar4>(raw.x);
            const uchar4 c1 = as_type<uchar4>(raw.y);
            uint2 a2 = as_type<uint2>(bfloat4(sc * float4(c0.x, c0.y, c0.z, c0.w) + bi));
            uint2 b2 = as_type<uint2>(bfloat4(sc * float4(c1.x, c1.y, c1.z, c1.w) + bi));
            wd4[0] = uint4(a2.x, a2.y, b2.x, b2.y);
        } else if constexpr (WPT == 16) {
            const uint4 raw = ((const device uint4*)wp)[0];
            const uchar4 c0 = as_type<uchar4>(raw.x);
            const uchar4 c1 = as_type<uchar4>(raw.y);
            const uchar4 c2 = as_type<uchar4>(raw.z);
            const uchar4 c3 = as_type<uchar4>(raw.w);
            uint2 a2 = as_type<uint2>(bfloat4(sc * float4(c0.x, c0.y, c0.z, c0.w) + bi));
            uint2 b2 = as_type<uint2>(bfloat4(sc * float4(c1.x, c1.y, c1.z, c1.w) + bi));
            wd4[0] = uint4(a2.x, a2.y, b2.x, b2.y);
            uint2 c2u = as_type<uint2>(bfloat4(sc * float4(c2.x, c2.y, c2.z, c2.w) + bi));
            uint2 d2u = as_type<uint2>(bfloat4(sc * float4(c3.x, c3.y, c3.z, c3.w) + bi));
            wd4[1] = uint4(c2u.x, c2u.y, d2u.x, d2u.y);
        } else {
            const device uint4* wp4 = (const device uint4*)wp;
            const uint4 r0 = wp4[0], r1 = wp4[1];
            const uint ws[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
            #pragma clang loop unroll(full)
            for (int gi = 0; gi < 4; ++gi) {
                const uchar4 c0 = as_type<uchar4>(ws[gi * 2]);
                const uchar4 c1 = as_type<uchar4>(ws[gi * 2 + 1]);
                uint2 a2 = as_type<uint2>(bfloat4(sc * float4(c0.x, c0.y, c0.z, c0.w) + bi));
                uint2 b2 = as_type<uint2>(bfloat4(sc * float4(c1.x, c1.y, c1.z, c1.w) + bi));
                wd4[gi] = uint4(a2.x, a2.y, b2.x, b2.y);
            }
        }

        if constexpr (TG == 32) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma clang loop unroll(full)
        for (int ks = 0; ks < BK; ks += 8) {
            simdgroup_bfloat8x8 A[TM];
            simdgroup_bfloat8x8 B[TN];
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tm = 0; tm < TM; ++tm) {
                const bfloat2 av = *((const threadgroup bfloat2*)(as_base + (tm * 8) * LDX + ks));
                A[tm].thread_elements()[0] = av.x;
                A[tm].thread_elements()[1] = av.y;
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < TN; ++tn) {
                const threadgroup bfloat16_t* bp = bs_base + (tn * 8) * LDX + ks;
                B[tn].thread_elements()[0] = bp[0];
                B[tn].thread_elements()[1] = bp[LDX];
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tm = 0; tm < TM; ++tm) {
                #pragma clang loop unroll(full)
                for (int tn = 0; tn < TN; ++tn) {
                    simdgroup_multiply_accumulate(C[tm][tn], A[tm], B[tn], C[tm][tn]);
                }
            }
        }

        if (k + BK < KK) {
            if constexpr (TG == 32) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    for (int tm = 0; tm < TM; ++tm) {
        const uint mr_o = m0 + sg_m + uint(tm) * 8u + fm;
        if (mr_o < uint(MM)) {
            device bfloat16_t* dst = out + (size_t)mr_o * NN;
            for (int tn = 0; tn < TN; ++tn) {
                const uint nc_o = n0 + sg_n + uint(tn) * 8u + fn;
                const bfloat2 ov = bfloat2(
                    bfloat16_t(C[tm][tn].thread_elements()[0]),
                    bfloat16_t(C[tm][tn].thread_elements()[1]));
                ((device uint*)(dst + nc_o))[0] = as_type<uint>(ov);
            }
        }
    }
"""


_AFF8_WK_SRC = r"""
    constexpr int GS = 64;
    constexpr int LDX = BK + 8;
    constexpr int NSG_MN = WM * WN;
    constexpr int TM = (BM / WM) / 8;
    constexpr int TN = (BN / WN) / 8;
    constexpr int XPT = (BM * BK) / (NSG_MN * 32);
    constexpr int WPT = (BN * BK) / (NSG_MN * 32);

    const uint tgn = threadgroup_position_in_grid.x;
    const uint tgm = threadgroup_position_in_grid.y;
    const uint m0 = tgm * uint(BM), n0 = tgn * uint(BN);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint wk_id = sg / uint(NSG_MN);
    const uint mn_sg = sg % uint(NSG_MN);
    const uint mn_tid = mn_sg * 32u + lane;
    const uint sg_m = (mn_sg / uint(WN)) * uint(TM * 8);
    const uint sg_n = (mn_sg % uint(WN)) * uint(TN * 8);
    const ushort fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const ushort fn = ((lane >> 2) & 2) * 2 | ((lane & 1) * 2);

    const int k_per = ((KK / BK + WK - 1) / WK) * BK;
    const int k0 = int(wk_id) * k_per;
    const int k1 = min(KK, k0 + k_per);

    constexpr int BUF_BF = WK * (BM + BN) * LDX;
    constexpr int BUF_F32 = WK * BM * BN;
    constexpr int BUF_U32 = (BUF_BF / 2 > BUF_F32) ? (BUF_BF / 2) : BUF_F32;
    threadgroup uint smem[BUF_U32];
    threadgroup bfloat16_t* Xs = ((threadgroup bfloat16_t*)smem) + wk_id * (BM * LDX);
    threadgroup bfloat16_t* Ws = ((threadgroup bfloat16_t*)smem) + WK * BM * LDX + wk_id * (BN * LDX);

    constexpr uint XTPR = uint(BK / XPT);
    const uint x_mr = mn_tid / XTPR;
    const uint x_mk = (mn_tid % XTPR) * uint(XPT);
    const bool x_ok = (m0 + x_mr) < uint(MM);
    const device bfloat16_t* x_ptr = x + (size_t)(m0 + x_mr) * KK + x_mk;

    constexpr uint WTPR = uint(BK / WPT);
    const uint w_nr = mn_tid / WTPR;
    const uint w_koff = (mn_tid % WTPR) * uint(WPT);
    const device uint8_t* w_ptr = (const device uint8_t*)w + (size_t)(n0 + w_nr) * KK + w_koff;
    const device bfloat16_t* s_ptr = scales + (size_t)(n0 + w_nr) * (KK / GS);
    const device bfloat16_t* b_ptr = biases + (size_t)(n0 + w_nr) * (KK / GS);

    simdgroup_float8x8 C[TM][TN];
    for (int tm = 0; tm < TM; ++tm)
        for (int tn = 0; tn < TN; ++tn)
            C[tm][tn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup uint4* xd = (threadgroup uint4*)(Xs + x_mr * LDX + x_mk);
    threadgroup uint4* wd4 = (threadgroup uint4*)(Ws + w_nr * LDX + w_koff);
    const threadgroup bfloat16_t* as_base = Xs + (sg_m + fm) * LDX + fn;
    const threadgroup bfloat16_t* bs_base = Ws + (sg_n + fn) * LDX + fm;

    for (int k = k0; k < k1; k += BK) {
        if constexpr (XPT == 16) {
            if (x_ok) { const device uint4* xp = (const device uint4*)(x_ptr + k); xd[0] = xp[0]; xd[1] = xp[1]; }
            else { xd[0] = uint4(0); xd[1] = uint4(0); }
        } else if constexpr (XPT == 8) {
            xd[0] = x_ok ? ((const device uint4*)(x_ptr + k))[0] : uint4(0);
        }
        const float sc = float(s_ptr[k / GS]), bi = float(b_ptr[k / GS]);
        const device uint8_t* wp = w_ptr + k;
        const device uint4* wp4 = (const device uint4*)wp;
        const uint4 r0 = wp4[0], r1 = wp4[1];
        const uint ws[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
        #pragma clang loop unroll(full)
        for (int gi = 0; gi < 4; ++gi) {
            const uchar4 c0 = as_type<uchar4>(ws[gi * 2]);
            const uchar4 c1 = as_type<uchar4>(ws[gi * 2 + 1]);
            uint2 a2 = as_type<uint2>(bfloat4(sc * float4(c0.x, c0.y, c0.z, c0.w) + bi));
            uint2 b2 = as_type<uint2>(bfloat4(sc * float4(c1.x, c1.y, c1.z, c1.w) + bi));
            wd4[gi] = uint4(a2.x, a2.y, b2.x, b2.y);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int ks = 0; ks < BK; ks += 8) {
            simdgroup_bfloat8x8 A[TM], B[TN];
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tm = 0; tm < TM; ++tm) {
                const bfloat2 av = *((const threadgroup bfloat2*)(as_base + (tm * 8) * LDX + ks));
                A[tm].thread_elements()[0] = av.x; A[tm].thread_elements()[1] = av.y;
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < TN; ++tn) {
                const threadgroup bfloat16_t* bp = bs_base + (tn * 8) * LDX + ks;
                B[tn].thread_elements()[0] = bp[0]; B[tn].thread_elements()[1] = bp[LDX];
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int tm = 0; tm < TM; ++tm)
                for (int tn = 0; tn < TN; ++tn)
                    simdgroup_multiply_accumulate(C[tm][tn], A[tm], B[tn], C[tm][tn]);
        }
        if (k + BK < k1) simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* Cs = (threadgroup float*)smem;
    for (int tm = 0; tm < TM; ++tm) {
        const uint rm = sg_m + uint(tm) * 8u + fm;
        for (int tn = 0; tn < TN; ++tn) {
            const uint cn = sg_n + uint(tn) * 8u + fn;
            ((threadgroup float2*)(Cs + (wk_id * BM + rm) * BN + cn))[0] =
                float2(C[tm][tn].thread_elements()[0], C[tm][tn].thread_elements()[1]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wk_id == 0) {
        for (int tm = 0; tm < TM; ++tm) {
            const uint rm = sg_m + uint(tm) * 8u + fm;
            const uint mr_o = m0 + rm;
            if (mr_o < uint(MM)) {
                device bfloat16_t* dst = out + (size_t)mr_o * NN;
                for (int tn = 0; tn < TN; ++tn) {
                    const uint cn = sg_n + uint(tn) * 8u + fn;
                    float2 s = ((const threadgroup float2*)(Cs + rm * BN + cn))[0];
                    #pragma clang loop unroll(full)
                    for (int w = 1; w < WK; ++w) {
                        s += ((const threadgroup float2*)(Cs + (w * BM + rm) * BN + cn))[0];
                    }
                    ((device uint*)(dst + n0 + cn))[0] = as_type<uint>(bfloat2(bfloat16_t(s.x), bfloat16_t(s.y)));
                }
            }
        }
    }
"""


def _k_aff8_qmm():
    k = _kernels.get("pf_aff8_qmm")
    if k is None:
        k = _kernels["pf_aff8_qmm"] = mx.fast.metal_kernel(
            name="pf_aff8_qmm",
            input_names=["x", "w", "scales", "biases"],
            output_names=["out"],
            source=_AFF8_QMM_SRC,
            header="#include <metal_simdgroup_matrix>\n",
            ensure_row_contiguous=True,
        )
    return k


def _k_aff8_wk():
    k = _kernels.get("pf_aff8_wk")
    if k is None:
        k = _kernels["pf_aff8_wk"] = mx.fast.metal_kernel(
            name="pf_aff8_wk",
            input_names=["x", "w", "scales", "biases"],
            output_names=["out"],
            source=_AFF8_WK_SRC,
            header="#include <metal_simdgroup_matrix>\n",
            ensure_row_contiguous=True,
        )
    return k


def _aff8_spk_g15d(M: int, N: int, K: int) -> int:
    """Exact MLX affine_qmm split-K factor on g15d for proven-faster M in [33, 128] (0 = fallback)."""
    if not (33 <= M <= 128):
        return 0
    if N == 3072 and K == 2304:
        return 1 if M >= 65 else 2
    if N == 2304 and K == 3072:
        if M >= 128:
            return 1
        if M >= 65:
            return 2 if M <= 96 else 0
        return 3
    if N == 1024 and K == 2304:
        return 4 if M >= 65 else 6
    return 0


def aff8_linear(mod, x: mx.array) -> mx.array:
    """Bit-exact Steel-scheduled aff8 (gs=64) dense qmm_t for prefill when LOOPLING_PF_AFF8_QMM=1."""
    if (
        not _PF_AFF8_QMM
        or x.dtype != mx.bfloat16
        or getattr(mod, "bits", None) != 8
        or getattr(mod, "group_size", None) != 64
        or getattr(mod, "mode", None) != "affine"
        or "bias" in mod
    ):
        return mod(x)
    K = x.shape[-1]
    M = x.size // K
    N = mod["weight"].shape[0]
    if K % 64 != 0 or N % 32 != 0 or mod["scales"].dtype != mx.bfloat16:
        return mod(x)
    if _IS_G15D and _PF_AFF8_WK and 33 <= M <= 128:
        spk = _aff8_spk_g15d(M, N, K)
        if spk > 1:
            bm, bn, bk = 16, 32, 32
            tg = spk * 32
            ntm = (M + bm - 1) // bm
            return _k_aff8_wk()(
                inputs=[x, mod["weight"], mod["scales"], mod["biases"]],
                template=[
                    ("BM", bm), ("BN", bn), ("BK", bk), ("WM", 1), ("WN", 1), ("WK", spk),
                    ("MM", M), ("NN", N), ("KK", K),
                ],
                grid=(tg * (N // bn), ntm, 1),
                threadgroup=(tg, 1, 1),
                output_shapes=[x.shape[:-1] + (N,)],
                output_dtypes=[mx.bfloat16],
            )[0]
        if spk == 1:
            bm, bn, bk, wm, wn = (16, 32, 32, 1, 1)
            tg = 32
            ntm = (M + bm - 1) // bm
            return _k_aff8_qmm()(
                inputs=[x, mod["weight"], mod["scales"], mod["biases"]],
                template=[("BM", bm), ("BN", bn), ("BK", bk), ("WM", wm), ("WN", wn), ("MM", M), ("NN", N), ("KK", K)],
                grid=(tg * (N // bn), ntm, 1),
                threadgroup=(tg, 1, 1),
                output_shapes=[x.shape[:-1] + (N,)],
                output_dtypes=[mx.bfloat16],
            )[0]
        return mod(x)
    if len(_PF_AFF8_CFG) == 5:
        bm, bn, bk, wm, wn = _PF_AFF8_CFG
    else:
        bm, bn, wm, wn = _PF_AFF8_CFG
        bk = 32
    if M < 1024 or N < 1024 or M % bm != 0 or N % bn != 0:
        return mod(x)
    tg = wm * wn * 32
    ntm = (M + bm - 1) // bm
    k = _k_aff8_qmm()
    return k(
        inputs=[x, mod["weight"], mod["scales"], mod["biases"]],
        template=[("BM", bm), ("BN", bn), ("BK", bk), ("WM", wm), ("WN", wn), ("MM", M), ("NN", N), ("KK", K)],
        grid=(tg * (N // bn), ntm, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[x.shape[:-1] + (N,)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _moe_fpq_ok(ex, hidden_states: mx.array, selected_experts: mx.array) -> bool:
    gu, dn = ex.gate_up_proj, ex.down_proj
    return (
        hidden_states.dtype == mx.bfloat16
        and selected_experts.shape[-1] == 8
        and getattr(gu, "mode", None) in ("nvfp4", "mxfp8")
        and getattr(dn, "mode", None) in ("nvfp4", "mxfp8")
        and gu.get("biases") is None
        and dn.get("biases") is None
        and "bias" not in gu
        and "bias" not in dn
        and gu.num_experts % 32 == 0
        and gu.num_experts <= 256
        and gu.input_dims % 32 == 0
        and gu.output_dims % 64 == 0
        and dn.input_dims % 32 == 0
        and dn.output_dims % 64 == 0
    )


def moe_gmm_ok(ex, hidden_states: mx.array, selected_experts: mx.array) -> bool:
    return (
        _PF_MOE_GMM
        and hidden_states.shape[0] * hidden_states.shape[1] >= _PF_MOE_MIN_L
        and _moe_fpq_ok(ex, hidden_states, selected_experts)
    )


def moe_short_ok(ex, hidden_states: mx.array, selected_experts: mx.array) -> bool:
    m = hidden_states.shape[0] * hidden_states.shape[1]
    return (
        _PF_MOE_SHORT
        and 2 <= m < _PF_MOE_MIN_L
        and _moe_fpq_ok(ex, hidden_states, selected_experts)
    )


_MOE_CSORT_SHORT_SRC = r"""
    constexpr uint NSG = 8;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    threadgroup atomic_uint hist[NSG][EE];
    threadgroup uint exp_tot[EE];
    threadgroup uint act_flg[EE];
    threadgroup uint sg_sums[8];
    threadgroup uint sg_asums[8];

    for (uint i = tid; i < NSG * uint(EE); i += 256) {
        atomic_store_explicit(&hist[i / uint(EE)][i % uint(EE)], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk = (uint(PP) + NSG - 1) / NSG;
    uint s0 = sg * chunk;
    uint s1 = min(s0 + chunk, uint(PP));
    for (uint p = s0 + lane; p < s1; p += 32) {
        uint e = inds[p];
        atomic_fetch_add_explicit(&hist[sg][e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(EE)) {
        uint acc = 0;
        for (uint s = 0; s < NSG; s++) {
            uint c = atomic_load_explicit(&hist[s][tid], memory_order_relaxed);
            atomic_store_explicit(&hist[s][tid], acc, memory_order_relaxed);
            acc += c;
        }
        exp_tot[tid] = acc;
        act_flg[tid] = (acc > 0u) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint NESG = uint(EE) / 32;
    if (tid < uint(EE)) {
        uint v = exp_tot[tid];
        uint ex = simd_prefix_exclusive_sum(v);
        if (lane == 31) sg_sums[sg] = ex + v;
        exp_tot[tid] = ex;

        uint av = act_flg[tid];
        uint aex = simd_prefix_exclusive_sum(av);
        if (lane == 31) sg_asums[sg] = aex + av;
        act_flg[tid] = aex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        uint sv = (tid < NESG) ? sg_sums[tid] : 0u;
        uint sp = simd_prefix_exclusive_sum(sv);
        if (tid < NESG) sg_sums[tid] = sp;

        uint sav = (tid < NESG) ? sg_asums[tid] : 0u;
        uint sap = simd_prefix_exclusive_sum(sav);
        if (tid < NESG) sg_asums[tid] = sap;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(EE)) {
        uint base = exp_tot[tid] + sg_sums[sg];
        exp_off[tid] = base;
        for (uint s = 0; s < NSG; s++) {
            uint off = atomic_load_explicit(&hist[s][tid], memory_order_relaxed);
            atomic_store_explicit(&hist[s][tid], base + off, memory_order_relaxed);
        }
        uint nxt = (tid + 1u == uint(EE)) ? uint(PP) : (exp_tot[tid + 1u] + sg_sums[(tid + 1u) >> 5]);
        if (nxt > base) {
            act_e[act_flg[tid] + sg_asums[sg]] = tid;
        }
        if (tid == uint(EE) - 1u) {
            exp_off[EE] = uint(PP);
            act_e[EE] = act_flg[tid] + sg_asums[sg] + ((nxt > base) ? 1u : 0u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = s0 + lane; p < s1; p += 32) {
        uint e = inds[p];
        uint pos = atomic_fetch_add_explicit(&hist[sg][e], 1u, memory_order_relaxed);
        lhs_idx[pos] = p >> 3;
        inv_order[p] = pos;
    }
"""

_MOE_SHORT_QMV_SRC = r"""
    constexpr int ROWS = 4;
    constexpr int PF = 32 / BITS;
    constexpr int PPT = ((BITS == 8) && (KK % 256 == 0)) ? 2 : 1;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    constexpr int NFULL = KK / BLOCK;
    constexpr int TAILK = KK - NFULL * BLOCK;
    constexpr bool HAS_TAIL = (TAILK > 0);
    typedef float U;
    typedef bfloat16_t T;

    const uint u = threadgroup_position_in_grid.z;
    const uint nact = act_e[EE];
    if (u >= nact) return;
    const uint e = act_e[u];
    const uint m_base = exp_off[e];
    const int cnt = (int)(exp_off[e + 1] - m_base);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int out_row = (int)threadgroup_position_in_grid.y * (ROWS * NSG) + (int)sg * ROWS;

    constexpr int in_w = KK * BITS / 8;
    constexpr int in_g = KK / GS;
    const device uint8_t* ws0 = (const device uint8_t*)w
        + (size_t)e * NN * in_w + (size_t)out_row * in_w + lane * PPT * 4;
    const device uint8_t* sc0 = scales
        + (size_t)e * NN * in_g + (size_t)out_row * in_g + lane / SSTEP;

    float inv_scale_enc = 1.0f;
    if constexpr (HGS) {
        inv_scale_enc = gsc[e] / (F8E4M3_MAX * F4E2M1_MAX);
    }
    const bool tail_on = HAS_TAIL && ((int)lane * VPT < TAILK);

    auto emit_one = [&](uint pos, const thread float* result) {
        if constexpr (FUSED_SWIGLU) {
            float gv[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float v = simd_sum(result[r]);
                gv[r] = HGS ? float(static_cast<T>(v * inv_scale_enc)) : float(static_cast<T>(v));
            }
            if (lane == 0) {
                const float gc = caps[0], uc = caps[1];
                bfloat2 ov = bfloat2(
                    static_cast<T>(glue_swiglu(gv[0], gv[1], gc, uc)),
                    static_cast<T>(glue_swiglu(gv[2], gv[3], gc, uc)));
                ((device uint*)(out + (size_t)pos * (NN / 2) + (size_t)(out_row / 2)))[0] = as_type<uint>(ov);
            }
        } else {
            T ov[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float v = simd_sum(result[r]);
                ov[r] = HGS ? static_cast<T>(v * inv_scale_enc) : static_cast<T>(v);
            }
            if (lane == 0) {
                ((device uint2*)(out + (size_t)pos * NN + (size_t)out_row))[0] =
                    as_type<uint2>(bfloat4(ov[0], ov[1], ov[2], ov[3]));
            }
        }
    };

    if (cnt == 1) {
        const uint pos = m_base;
        const uint tok = USE_LHS ? lhs_idx[pos] : pos;
        const size_t xo = (size_t)tok * KK + (size_t)lane * VPT;
        thread U result[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) result[r] = 0.0f;
        const device uint8_t* ws = ws0;
        const device uint8_t* sc = sc0;
        for (int it = 0; it < NFULL; it++) {
            const uint k_off = (uint)it * (uint)BLOCK;
            if constexpr (VPT == 8) {
                uint4 x4 = ((const device uint4*)(x + xo + k_off))[0];
                float4 xf0 = float4(as_type<bfloat4>(x4.xy));
                float4 xf1 = float4(as_type<bfloat4>(x4.zw));
                if constexpr (BITS == 4) {
                    thread uint32_t wr[ROWS];
                    thread uint8_t sr[ROWS];
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        wr[r] = ((const device uint32_t*)(ws + r * in_w))[0];
                        sr[r] = (sc + r * in_g)[0];
                    }
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        thread U wv[8];
                        dequant_u32_fp4_8(wr[r], wv);
                        U sv = moe_dequant_scale<GS>(sr[r]);
                        result[r] += dot_vals_f4_8<4>(xf0, xf1, wv, sv);
                    }
                } else {
                    thread uint2 wr[ROWS];
                    thread uint8_t sr[ROWS];
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        wr[r] = ((const device uint2*)(ws + r * in_w))[0];
                        sr[r] = (sc + r * in_g)[0];
                    }
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        thread U wv[8];
                        dequant_u2_fp8_8(wr[r], wv);
                        U sv = moe_dequant_scale<GS>(sr[r]);
                        result[r] += dot_vals_f4_8<8>(xf0, xf1, wv, sv);
                    }
                }
            } else {
                uint2 x2 = ((const device uint2*)(x + xo + k_off))[0];
                float4 xf0 = float4(as_type<bfloat4>(x2));
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    thread U wv[VPT];
                    dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    result[r] += dot_vals_f4_4<BITS>(xf0, wv, sv);
                }
            }
            ws += BLOCK * BITS / 8;
            sc += BLOCK / GS;
        }
        if constexpr (HAS_TAIL) {
            if (tail_on) {
                const uint k_off = (uint)NFULL * (uint)BLOCK;
                if constexpr (VPT == 8) {
                    uint4 x4 = ((const device uint4*)(x + xo + k_off))[0];
                    float4 xf0 = float4(as_type<bfloat4>(x4.xy));
                    float4 xf1 = float4(as_type<bfloat4>(x4.zw));
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        thread U wv[8];
                        dequant_vals<BITS, 8>(ws + r * in_w, wv);
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        result[r] += dot_vals_f4_8<BITS>(xf0, xf1, wv, sv);
                    }
                } else {
                    uint2 x2 = ((const device uint2*)(x + xo + k_off))[0];
                    float4 xf0 = float4(as_type<bfloat4>(x2));
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        thread U wv[VPT];
                        dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        result[r] += dot_vals_f4_4<BITS>(xf0, wv, sv);
                    }
                }
            }
        }
        emit_one(pos, result);
    } else {
        for (int tb = 0; tb < cnt; tb += TMAX) {
            const int nb = min(TMAX, cnt - tb);
            uint xo[TMAX];
            #pragma unroll
            for (int t = 0; t < TMAX; t++) {
                if (t < nb) {
                    uint pos = m_base + uint(tb + t);
                    uint tok = USE_LHS ? lhs_idx[pos] : pos;
                    xo[t] = tok * uint(KK) + lane * uint(VPT);
                } else {
                    xo[t] = 0u;
                }
            }
            thread U result[TMAX][ROWS];
            #pragma unroll
            for (int t = 0; t < TMAX; t++) {
                #pragma unroll
                for (int r = 0; r < ROWS; r++) result[t][r] = 0.0f;
            }
            const device uint8_t* ws = ws0;
            const device uint8_t* sc = sc0;
            for (int it = 0; it < NFULL; it++) {
                const uint k_off = (uint)it * (uint)BLOCK;
                if constexpr (VPT == 8) {
                    thread uint4 x4[TMAX];
                    #pragma unroll
                    for (int t = 0; t < TMAX; t++) {
                        if (t < nb) x4[t] = ((const device uint4*)(x + xo[t] + k_off))[0];
                    }
                    if constexpr (BITS == 4) {
                        thread uint32_t wr[ROWS];
                        thread uint8_t sr[ROWS];
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            wr[r] = ((const device uint32_t*)(ws + r * in_w))[0];
                            sr[r] = (sc + r * in_g)[0];
                        }
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            thread U wv[8];
                            dequant_u32_fp4_8(wr[r], wv);
                            U sv = moe_dequant_scale<GS>(sr[r]);
                            #pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < nb) result[t][r] += dot_vals_bf16_u4val<4>(x4[t], wv, sv);
                            }
                        }
                    } else {
                        thread uint2 wr[ROWS];
                        thread uint8_t sr[ROWS];
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            wr[r] = ((const device uint2*)(ws + r * in_w))[0];
                            sr[r] = (sc + r * in_g)[0];
                        }
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            thread U wv[8];
                            dequant_u2_fp8_8(wr[r], wv);
                            U sv = moe_dequant_scale<GS>(sr[r]);
                            #pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < nb) result[t][r] += dot_vals_bf16_u4val<8>(x4[t], wv, sv);
                            }
                        }
                    }
                } else {
                    thread uint2 x2[TMAX];
                    #pragma unroll
                    for (int t = 0; t < TMAX; t++) {
                        if (t < nb) x2[t] = ((const device uint2*)(x + xo[t] + k_off))[0];
                    }
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        thread U wv[VPT];
                        dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        #pragma unroll
                        for (int t = 0; t < TMAX; t++) {
                            if (t < nb) result[t][r] += dot_vals_bf16_u2val<BITS>(x2[t], wv, sv);
                        }
                    }
                }
                ws += BLOCK * BITS / 8;
                sc += BLOCK / GS;
            }
            if constexpr (HAS_TAIL) {
                if (tail_on) {
                    const uint k_off = (uint)NFULL * (uint)BLOCK;
                    if constexpr (VPT == 8) {
                        thread uint4 x4[TMAX];
                        #pragma unroll
                        for (int t = 0; t < TMAX; t++) {
                            if (t < nb) x4[t] = ((const device uint4*)(x + xo[t] + k_off))[0];
                        }
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            thread U wv[8];
                            dequant_vals<BITS, 8>(ws + r * in_w, wv);
                            U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                            #pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < nb) result[t][r] += dot_vals_bf16_u4val<BITS>(x4[t], wv, sv);
                            }
                        }
                    } else {
                        thread uint2 x2[TMAX];
                        #pragma unroll
                        for (int t = 0; t < TMAX; t++) {
                            if (t < nb) x2[t] = ((const device uint2*)(x + xo[t] + k_off))[0];
                        }
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            thread U wv[VPT];
                            dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                            U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                            #pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < nb) result[t][r] += dot_vals_bf16_u2val<BITS>(x2[t], wv, sv);
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < TMAX; t++) {
                if (t < nb) emit_one(m_base + uint(tb + t), result[t]);
            }
        }
    }
"""


_MOE_CSORT_SHORT_TILES_SRC = r"""
    constexpr uint NSG = 8;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    threadgroup atomic_uint hist[NSG][EE];
    threadgroup uint exp_tot[EE];
    threadgroup uint tile_cnt[EE];
    threadgroup uint sg_sums[8];
    threadgroup uint sg_tsums[8];

    for (uint i = tid; i < NSG * uint(EE); i += 256) {
        atomic_store_explicit(&hist[i / uint(EE)][i % uint(EE)], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk = (uint(PP) + NSG - 1) / NSG;
    uint s0 = sg * chunk;
    uint s1 = min(s0 + chunk, uint(PP));
    for (uint p = s0 + lane; p < s1; p += 32) {
        uint e = inds[p];
        atomic_fetch_add_explicit(&hist[sg][e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(EE)) {
        uint acc = 0;
        for (uint s = 0; s < NSG; s++) {
            uint c = atomic_load_explicit(&hist[s][tid], memory_order_relaxed);
            atomic_store_explicit(&hist[s][tid], acc, memory_order_relaxed);
            acc += c;
        }
        exp_tot[tid] = acc;
        tile_cnt[tid] = (acc + uint(BM) - 1) / uint(BM);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint NESG = uint(EE) / 32;
    if (tid < uint(EE)) {
        uint v = exp_tot[tid];
        uint ex = simd_prefix_exclusive_sum(v);
        if (lane == 31) sg_sums[sg] = ex + v;
        exp_tot[tid] = ex;

        uint tv = tile_cnt[tid];
        uint tex = simd_prefix_exclusive_sum(tv);
        if (lane == 31) sg_tsums[sg] = tex + tv;
        tile_cnt[tid] = tex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        uint sv = (tid < NESG) ? sg_sums[tid] : 0u;
        uint sp = simd_prefix_exclusive_sum(sv);
        if (tid < NESG) sg_sums[tid] = sp;

        uint stv = (tid < NESG) ? sg_tsums[tid] : 0u;
        uint stp = simd_prefix_exclusive_sum(stv);
        if (tid < NESG) sg_tsums[tid] = stp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(EE)) {
        uint base = exp_tot[tid] + sg_sums[sg];
        exp_off[tid] = base;
        for (uint s = 0; s < NSG; s++) {
            uint off = atomic_load_explicit(&hist[s][tid], memory_order_relaxed);
            atomic_store_explicit(&hist[s][tid], base + off, memory_order_relaxed);
        }
        uint tbase = tile_cnt[tid] + sg_tsums[sg];
        uint nxt = (tid + 1u == uint(EE)) ? uint(PP) : (exp_tot[tid + 1u] + sg_sums[(tid + 1u) >> 5]);
        uint nte = (nxt - base + uint(BM) - 1) / uint(BM);
        for (uint mt = 0; mt < nte; mt++) {
            tile_meta[tbase + mt] = (tid << 20) | mt;
        }
        if (tid == uint(EE) - 1u) {
            exp_off[EE] = uint(PP);
            uint tot_t = tbase + nte;
            for (uint t = tot_t; t < uint(MAX_TILES); t++) {
                tile_meta[t] = 0xFFFFFFFFu;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = s0 + lane; p < s1; p += 32) {
        uint e = inds[p];
        uint pos = atomic_fetch_add_explicit(&hist[sg][e], 1u, memory_order_relaxed);
        lhs_idx[pos] = p >> 3;
        inv_order[p] = pos;
    }
"""

def _moe_short_tiled_branch(nb: int) -> str:
    return f"""
        uint xo[{nb}];
        #pragma unroll
        for (int t = 0; t < {nb}; t++) {{
            uint pos = m_base + uint(t);
            uint tok = USE_LHS ? lhs_idx[pos] : pos;
            xo[t] = tok * uint(KK) + lane * uint(VPT);
        }}
        thread U result[{nb}][ROWS];
        #pragma unroll
        for (int t = 0; t < {nb}; t++) {{
            #pragma unroll
            for (int r = 0; r < ROWS; r++) result[t][r] = 0.0f;
        }}
        const device uint8_t* ws = ws0;
        const device uint8_t* sc = sc0;
        for (int it = 0; it < NFULL; it++) {{
            const uint k_off = (uint)it * (uint)BLOCK;
            if constexpr (VPT == 8) {{
                thread uint4 x4[{nb}];
                #pragma unroll
                for (int t = 0; t < {nb}; t++) x4[t] = ((const device uint4*)(x + xo[t] + k_off))[0];
                if constexpr (BITS == 4) {{
                    thread uint32_t wr[ROWS];
                    thread uint8_t sr[ROWS];
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {{
                        wr[r] = ((const device uint32_t*)(ws + r * in_w))[0];
                        sr[r] = (sc + r * in_g)[0];
                    }}
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {{
                        thread U wv[8];
                        dequant_u32_fp4_8(wr[r], wv);
                        U sv = moe_dequant_scale<GS>(sr[r]);
                        #pragma unroll
                        for (int t = 0; t < {nb}; t++) result[t][r] += dot_vals_bf16_u4val<4>(x4[t], wv, sv);
                    }}
                }} else {{
                    thread uint2 wr[ROWS];
                    thread uint8_t sr[ROWS];
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {{
                        wr[r] = ((const device uint2*)(ws + r * in_w))[0];
                        sr[r] = (sc + r * in_g)[0];
                    }}
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {{
                        thread U wv[8];
                        dequant_u2_fp8_8(wr[r], wv);
                        U sv = moe_dequant_scale<GS>(sr[r]);
                        #pragma unroll
                        for (int t = 0; t < {nb}; t++) result[t][r] += dot_vals_bf16_u4val<8>(x4[t], wv, sv);
                    }}
                }}
            }} else {{
                thread uint2 x2[{nb}];
                #pragma unroll
                for (int t = 0; t < {nb}; t++) x2[t] = ((const device uint2*)(x + xo[t] + k_off))[0];
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {{
                    thread U wv[VPT];
                    dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    #pragma unroll
                    for (int t = 0; t < {nb}; t++) result[t][r] += dot_vals_bf16_u2val<BITS>(x2[t], wv, sv);
                }}
            }}
            ws += BLOCK * BITS / 8;
            sc += BLOCK / GS;
        }}
        if constexpr (HAS_TAIL) {{
            if (tail_on) {{
                const uint k_off = (uint)NFULL * (uint)BLOCK;
                if constexpr (VPT == 8) {{
                    thread uint4 x4[{nb}];
                    #pragma unroll
                    for (int t = 0; t < {nb}; t++) x4[t] = ((const device uint4*)(x + xo[t] + k_off))[0];
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {{
                        thread U wv[8];
                        dequant_vals<BITS, 8>(ws + r * in_w, wv);
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        #pragma unroll
                        for (int t = 0; t < {nb}; t++) result[t][r] += dot_vals_bf16_u4val<BITS>(x4[t], wv, sv);
                    }}
                }} else {{
                    thread uint2 x2[{nb}];
                    #pragma unroll
                    for (int t = 0; t < {nb}; t++) x2[t] = ((const device uint2*)(x + xo[t] + k_off))[0];
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {{
                        thread U wv[VPT];
                        dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        #pragma unroll
                        for (int t = 0; t < {nb}; t++) result[t][r] += dot_vals_bf16_u2val<BITS>(x2[t], wv, sv);
                    }}
                }}
            }}
        }}
        #pragma unroll
        for (int t = 0; t < {nb}; t++) emit_one(m_base + uint(t), result[t]);
"""


_MOE_SHORT_TILED_SRC = (
    r"""
    constexpr int ROWS = 4;
    constexpr int PF = 32 / BITS;
    constexpr int PPT = ((BITS == 8) && (KK % 256 == 0)) ? 2 : 1;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    constexpr int NFULL = KK / BLOCK;
    constexpr int TAILK = KK - NFULL * BLOCK;
    constexpr bool HAS_TAIL = (TAILK > 0);
    constexpr int in_w = KK * BITS / 8;
    constexpr int in_g = KK / GS;
    typedef float U;
    typedef bfloat16_t T;

    const uint tile_id = threadgroup_position_in_grid.y;
    const uint n_blk = threadgroup_position_in_grid.z;
    const uint meta = tile_meta[tile_id];
    if (meta == 0xFFFFFFFFu) return;
    const uint e = meta >> 20;
    const uint mt = meta & 0xFFFFFu;
    const uint m_base = exp_off[e] + mt * uint(TMAX);
    const uint m_end = exp_off[e + 1];
    const int nb = (int)min(uint(TMAX), m_end - m_base);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int out_row = (int)n_blk * (ROWS * NSG) + (int)sg * ROWS;

    const device uint8_t* ws0 = (const device uint8_t*)w
        + (size_t)e * NN * in_w + (size_t)out_row * in_w + lane * PPT * 4;
    const device uint8_t* sc0 = scales
        + (size_t)e * NN * in_g + (size_t)out_row * in_g + lane / SSTEP;

    float inv_scale_enc = 1.0f;
    if constexpr (HGS) {
        inv_scale_enc = gsc[e] / (F8E4M3_MAX * F4E2M1_MAX);
    }
    const bool tail_on = HAS_TAIL && ((int)lane * VPT < TAILK);

    auto emit_one = [&](uint pos, const thread float* result) {
        if constexpr (FUSED_SWIGLU) {
            float gv[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float v = simd_sum(result[r]);
                gv[r] = HGS ? float(static_cast<T>(v * inv_scale_enc)) : float(static_cast<T>(v));
            }
            if (lane == 0) {
                const float gc = caps[0], uc = caps[1];
                bfloat2 ov = bfloat2(
                    static_cast<T>(glue_swiglu(gv[0], gv[1], gc, uc)),
                    static_cast<T>(glue_swiglu(gv[2], gv[3], gc, uc)));
                ((device uint*)(out + (size_t)pos * (NN / 2) + (size_t)(out_row / 2)))[0] = as_type<uint>(ov);
            }
        } else {
            T ov[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float v = simd_sum(result[r]);
                ov[r] = HGS ? static_cast<T>(v * inv_scale_enc) : static_cast<T>(v);
            }
            if (lane == 0) {
                ((device uint2*)(out + (size_t)pos * NN + (size_t)out_row))[0] =
                    as_type<uint2>(bfloat4(ov[0], ov[1], ov[2], ov[3]));
            }
        }
    };

    if (nb == 1) {
"""
    + _moe_short_tiled_branch(1)
    + "    } else if (nb == 2) {\n"
    + _moe_short_tiled_branch(2)
    + "    } else if (TMAX >= 3 && nb == 3) {\n"
    + _moe_short_tiled_branch(3)
    + "    } else if constexpr (TMAX >= 4) {\n"
    + _moe_short_tiled_branch(4)
    + "    }\n"
)


def _k_moe_short():
    k = _kernels.get("pf_moe_short_qmv")
    if k is None:
        hdr = _fpq_mlx_header() + "\n" + _DOT_BF16 + "\n" + GLUE_HEADER
        k = _kernels["pf_moe_short_qmv"] = mx.fast.metal_kernel(
            name="pf_moe_short_qmv",
            input_names=["x", "lhs_idx", "exp_off", "act_e", "w", "scales", "gsc", "caps"],
            output_names=["out"],
            source=_MOE_SHORT_QMV_SRC,
            header=hdr,
            ensure_row_contiguous=True,
        )
    return k


def _k_moe_short_tiled():
    k = _kernels.get("pf_moe_short_tiled")
    if k is None:
        hdr = _fpq_mlx_header() + "\n" + _DOT_BF16 + "\n" + GLUE_HEADER
        k = _kernels["pf_moe_short_tiled"] = mx.fast.metal_kernel(
            name="pf_moe_short_tiled",
            input_names=["x", "lhs_idx", "exp_off", "tile_meta", "w", "scales", "gsc", "caps"],
            output_names=["out"],
            source=_MOE_SHORT_TILED_SRC,
            header=hdr,
            ensure_row_contiguous=True,
        )
    return k


def moe_short_forward(
    ex,
    hidden_states: mx.array,
    selected_experts: mx.array,
    routing_weights: mx.array,
) -> mx.array:
    """Bit-exact expert-order fp_qmv MoE for short suffixes (2 <= B*L < _PF_MOE_MIN_L)."""
    B, L, D = hidden_states.shape
    gu, dn = ex.gate_up_proj, ex.down_proj
    E = gu.num_experts
    P = B * L * 8
    ckey = ("swiglu", ex.act_fn.gate_softcap, ex.act_fn.up_softcap)
    caps = _PARAMS_CACHE.get(ckey)
    if caps is None:
        caps = _PARAMS_CACHE[ckey] = mx.array([ex.act_fn.gate_softcap, ex.act_fn.up_softcap], dtype=mx.float32)

    hgs_gu = int(gu.global_scale is not None)
    gsc_gu = gu.global_scale.astype(mx.float32) if hgs_gu else _DUMMY_F32
    N_gu, K_gu = gu.output_dims, gu.input_dims
    hgs_dn = int(dn.global_scale is not None)
    gsc_dn = dn.global_scale.astype(mx.float32) if hgs_dn else _DUMMY_F32
    N_dn, K_dn = dn.output_dims, dn.input_dims

    if _PF_MOE_SHORT_TILED and B * L >= 8:
        nsg = 8 if (N_gu % 32 == 0 and N_dn % 32 == 0) else 4
        tmax = 3 if B * L <= 44 else 4
        max_tiles = min(P, P // tmax + E)
        ks = _k(
            "pf_moe_csort_short_tiles",
            ("inds",),
            ("lhs_idx", "inv_order", "exp_off", "tile_meta"),
            _MOE_CSORT_SHORT_TILES_SRC,
        )
        lhs_idx, inv_order, exp_off, tile_meta = ks(
            inputs=[selected_experts.reshape(-1).astype(mx.uint32)],
            template=[("PP", P), ("EE", E), ("BM", tmax), ("MAX_TILES", max_tiles)],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(P,), (P,), (E + 1,), (max_tiles,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
        )
        kt = _k_moe_short_tiled()
        act = kt(
            inputs=[hidden_states, lhs_idx, exp_off, tile_meta, gu["weight"], gu["scales"], gsc_gu, caps],
            template=[
                ("BITS", gu.bits), ("GS", gu.group_size), ("HGS", hgs_gu),
                ("KK", K_gu), ("NN", N_gu), ("NSG", nsg), ("TMAX", tmax),
                ("USE_LHS", 1), ("FUSED_SWIGLU", 1),
            ],
            grid=(32, nsg * max_tiles, N_gu // (4 * nsg)),
            threadgroup=(32, nsg, 1),
            output_shapes=[(P, N_gu // 2)],
            output_dtypes=[mx.bfloat16],
        )[0]
        dn_s = kt(
            inputs=[act, lhs_idx, exp_off, tile_meta, dn["weight"], dn["scales"], gsc_dn, caps],
            template=[
                ("BITS", dn.bits), ("GS", dn.group_size), ("HGS", hgs_dn),
                ("KK", K_dn), ("NN", N_dn), ("NSG", nsg), ("TMAX", tmax),
                ("USE_LHS", 0), ("FUSED_SWIGLU", 0),
            ],
            grid=(32, nsg * max_tiles, N_dn // (4 * nsg)),
            threadgroup=(32, nsg, 1),
            output_shapes=[(P, N_dn)],
            output_dtypes=[mx.bfloat16],
        )[0]
    else:
        nsg = _PF_MOE_SHORT_NSG
        tmax = _PF_MOE_SHORT_TMAX
        ks = _k(
            "pf_moe_csort_short",
            ("inds",),
            ("lhs_idx", "inv_order", "exp_off", "act_e"),
            _MOE_CSORT_SHORT_SRC,
        )
        lhs_idx, inv_order, exp_off, act_e = ks(
            inputs=[selected_experts.reshape(-1).astype(mx.uint32)],
            template=[("PP", P), ("EE", E)],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(P,), (P,), (E + 1,), (E + 1,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
        )
        kq = _k_moe_short()
        z_act = min(P, E)
        act = kq(
            inputs=[hidden_states, lhs_idx, exp_off, act_e, gu["weight"], gu["scales"], gsc_gu, caps],
            template=[
                ("BITS", gu.bits), ("GS", gu.group_size), ("HGS", hgs_gu),
                ("KK", K_gu), ("NN", N_gu), ("EE", E), ("NSG", nsg), ("TMAX", tmax),
                ("USE_LHS", 1), ("FUSED_SWIGLU", 1),
            ],
            grid=(32, nsg * (N_gu // (4 * nsg)), z_act),
            threadgroup=(32, nsg, 1),
            output_shapes=[(P, N_gu // 2)],
            output_dtypes=[mx.bfloat16],
        )[0]
        dn_s = kq(
            inputs=[act, lhs_idx, exp_off, act_e, dn["weight"], dn["scales"], gsc_dn, caps],
            template=[
                ("BITS", dn.bits), ("GS", dn.group_size), ("HGS", hgs_dn),
                ("KK", K_dn), ("NN", N_dn), ("EE", E), ("NSG", nsg), ("TMAX", tmax),
                ("USE_LHS", 0), ("FUSED_SWIGLU", 0),
            ],
            grid=(32, nsg * (N_dn // (4 * nsg)), z_act),
            threadgroup=(32, nsg, 1),
            output_shapes=[(P, N_dn)],
            output_dtypes=[mx.bfloat16],
        )[0]

    return unsort_combine(dn_s, inv_order, routing_weights)


def moe_gmm_forward(
    ex,
    hidden_states: mx.array,
    selected_experts: mx.array,
    routing_weights: mx.array,
) -> mx.array:
    """Wave 6B/8 fused MoE prefill: 1-launch counting sort+tile_meta -> TG=32/64 gate_up+SwiGLU -> down -> unsort_combine."""
    B, L, D = hidden_states.shape
    gu, dn = ex.gate_up_proj, ex.down_proj
    E = gu.num_experts
    P = B * L * 8
    if _PF_MOE_CFG is not None:
        if len(_PF_MOE_CFG) == 5:
            bm, bn, bk, wm, wn = _PF_MOE_CFG
        else:
            bm, bn, wm, wn = _PF_MOE_CFG
            bk = 32
    elif _PF_MOE_BM_ENV in (8, 16, 32, 64):
        bm, bn, bk, wm, wn = (_PF_MOE_BM_ENV, 32, 32, 1, 1)
    elif B * L < 256:
        bm, bn, bk, wm, wn = (8, 32, 32, 1, 1)
    elif B * L < 1024:
        bm, bn, bk, wm, wn = (16, 32, 32, 1, 1)
    else:
        bm, bn, bk, wm, wn = (32, 64, 32, 1, 2)
    tg = wm * wn * 32
    max_tiles = P // bm + E

    ks = _k(
        "pf_moe_csort_tiles",
        ("inds",),
        ("lhs_idx", "inv_order", "exp_off", "tile_meta"),
        _MOE_CSORT_TILES_SRC,
    )
    lhs_idx, inv_order, exp_off, tile_meta = ks(
        inputs=[selected_experts.reshape(-1).astype(mx.uint32)],
        template=[("PP", P), ("EE", E), ("BM", bm), ("MAX_TILES", max_tiles)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(P,), (P,), (E + 1,), (max_tiles,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
    )

    ckey = ("swiglu", ex.act_fn.gate_softcap, ex.act_fn.up_softcap)
    caps = _PARAMS_CACHE.get(ckey)
    if caps is None:
        caps = _PARAMS_CACHE[ckey] = mx.array([ex.act_fn.gate_softcap, ex.act_fn.up_softcap], dtype=mx.float32)

    kg = _k_moe_gmm()

    # 1. Fused gate_up + SwiGLU (direct lhs_idx gather from hidden_states, zero x_sorted & zero gu DRAM!)
    hgs_gu = int(gu.global_scale is not None)
    gsc_gu = gu.global_scale.astype(mx.float32) if hgs_gu else _DUMMY_F32
    N_gu, K_gu = gu.output_dims, gu.input_dims
    act = kg(
        inputs=[hidden_states, lhs_idx, exp_off, tile_meta, gu["weight"], gu["scales"], gsc_gu, caps],
        template=[
            ("BM", bm), ("BN", bn), ("BK", bk), ("WM", wm), ("WN", wn),
            ("BITS", gu.bits), ("GS", gu.group_size), ("HGS", hgs_gu),
            ("KK", K_gu), ("NN", N_gu), ("USE_LHS", 1), ("FUSED_SWIGLU", 1),
        ],
        grid=(tg * max_tiles, N_gu // bn, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(P, N_gu // 2)],
        output_dtypes=[mx.bfloat16],
    )[0]

    # 2. Grouped down_proj (reads contiguous act directly)
    hgs_dn = int(dn.global_scale is not None)
    gsc_dn = dn.global_scale.astype(mx.float32) if hgs_dn else _DUMMY_F32
    N_dn, K_dn = dn.output_dims, dn.input_dims
    dn_s = kg(
        inputs=[act, lhs_idx, exp_off, tile_meta, dn["weight"], dn["scales"], gsc_dn, caps],
        template=[
            ("BM", bm), ("BN", bn), ("BK", bk), ("WM", wm), ("WN", wn),
            ("BITS", dn.bits), ("GS", dn.group_size), ("HGS", hgs_dn),
            ("KK", K_dn), ("NN", N_dn), ("USE_LHS", 0), ("FUSED_SWIGLU", 0),
        ],
        grid=(tg * max_tiles, N_dn // bn, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(P, N_dn)],
        output_dtypes=[mx.bfloat16],
    )[0]

    return unsort_combine(dn_s, inv_order, routing_weights)


# ---------------------------------------------------------------------------------------------------- MoE counting sort + gather
_MOE_CSORT_SRC = r"""
    constexpr uint NSG = 32;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    threadgroup atomic_uint hist[NSG][EE];
    threadgroup uint exp_tot[EE];
    threadgroup uint sg_sums[8];

    for (uint i = tid; i < NSG * EE; i += 1024) {
        atomic_store_explicit(&hist[i / EE][i % EE], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk = (uint(PP) + NSG - 1) / NSG;
    uint s0 = sg * chunk;
    uint s1 = min(s0 + chunk, uint(PP));
    for (uint p = s0 + lane; p < s1; p += 32) {
        uint e = inds[p];
        atomic_fetch_add_explicit(&hist[sg][e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < EE) {
        uint acc = 0;
        for (uint s = 0; s < NSG; s++) {
            uint c = atomic_load_explicit(&hist[s][tid], memory_order_relaxed);
            atomic_store_explicit(&hist[s][tid], acc, memory_order_relaxed);
            acc += c;
        }
        exp_tot[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint NESG = EE / 32;
    if (tid < EE) {
        uint v = exp_tot[tid];
        uint ex = simd_prefix_exclusive_sum(v);
        if (lane == 31) sg_sums[sg] = ex + v;
        exp_tot[tid] = ex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        uint sv = (tid < NESG) ? sg_sums[tid] : 0u;
        uint sp = simd_prefix_exclusive_sum(sv);
        if (tid < NESG) sg_sums[tid] = sp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < EE) {
        uint base = exp_tot[tid] + sg_sums[sg];
        for (uint s = 0; s < NSG; s++) {
            uint off = atomic_load_explicit(&hist[s][tid], memory_order_relaxed);
            atomic_store_explicit(&hist[s][tid], base + off, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = s0 + lane; p < s1; p += 32) {
        uint e = inds[p];
        uint pos = atomic_fetch_add_explicit(&hist[sg][e], 1u, memory_order_relaxed);
        lhs_idx[pos] = p >> 3;
        idx_s[pos] = e;
        inv_order[p] = pos;
    }
"""

_MOE_GATHER8_SRC = r"""
    uint d8 = thread_position_in_grid.x;
    uint pos = thread_position_in_grid.y;
    if (d8 >= DD8) return;
    uint tok = lhs_idx[pos];
    ((device uint4*)(out + (size_t)pos * DD8 * 8))[d8] =
        ((const device uint4*)(x + (size_t)tok * DD8 * 8))[d8];
"""


def moe_gather_sort(hidden_states: mx.array, selected_experts: mx.array, num_experts: int = 128):
    """1-launch E=128 counting sort + 128-bit vectorized x gather -> (x_sorted (P,1,1,D), idx_s (P,), inv_order (P,))."""
    B, L, D = hidden_states.shape
    K = selected_experts.shape[-1]
    P = B * L * K
    assert K == 8 and num_experts % 32 == 0 and num_experts <= 256 and D % 8 == 0
    ks = _k("pf_moe_csort", ("inds",), ("lhs_idx", "idx_s", "inv_order"), _MOE_CSORT_SRC)
    lhs_idx, idx_s, inv_order = ks(
        inputs=[selected_experts.reshape(-1).astype(mx.uint32)],
        template=[("PP", P), ("EE", num_experts)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(P,), (P,), (P,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
    )
    dd8 = D // 8
    tg = dd8 if dd8 <= 1024 and dd8 % 32 == 0 else 256
    gx = ((dd8 + tg - 1) // tg) * tg
    kg = _k("pf_moe_gather8", ("x", "lhs_idx"), ("out",), _MOE_GATHER8_SRC)
    x_s = kg(
        inputs=[hidden_states, lhs_idx],
        template=[("DD8", dd8)],
        grid=(gx, P, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(P, 1, D)],
        output_dtypes=[hidden_states.dtype],
    )[0]
    return x_s, idx_s, inv_order


# ---------------------------------------------------------------------------------------------------- Fused conv + residual / RMSNorm
_SC4_RES_SRC = f"""
    GL_NOFMA
    uint c4 = thread_position_in_grid.x;
    uint c = c4 * 4;
    if (c >= DD) return;
    uint tile = thread_position_in_grid.y;
    uint ntiles = (LL + TT - 1) / TT;
    uint b = tile / ntiles;
    uint t0 = (tile % ntiles) * TT;
    float4 w0, w1, w2, w3;
    for (uint k = 0; k < 4; k++) {{
        const device bfloat16_t* cp = cw + (c + k) * 4;
        w0[k] = float(cp[0]);
        w1[k] = float(cp[1]);
        w2[k] = float(cp[2]);
        w3[k] = float(cp[3]);
    }}
    auto tap4 = [&](int p) -> float4 {{
        if (p >= 0) {{
            if constexpr (HAS_MASK) {{
                if (!cmask[(size_t)b * LL + p]) return float4(0.0f);
            }}
            return float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0]));
        }}
        int sidx = 3 + p;
        if (!HAS_CS) return float4(0.0f);
        float4 csf = ((const device float4*)(cs + ((size_t)b * 3 + sidx) * DD + c))[0];
        return float4(bfloat4(csf));
    }};
    float4 s0 = tap4(int(t0) - 3);
    float4 s1 = tap4(int(t0) - 2);
    float4 s2 = tap4(int(t0) - 1);
    for (uint tk = 0; tk < TT; tk++) {{
        uint t = t0 + tk;
        if (t >= LL) break;
        size_t off = ((size_t)b * LL + t) * DD + c;
        float4 x3 = float4(as_type<bfloat4>(((const device uint2*)(x + off))[0]));
        float4 xw = x3;
        if constexpr (HAS_MASK) {{
            if (!cmask[(size_t)b * LL + t]) xw = float4(0.0f);
        }}
        float4 acc = float4(0.0f);
        acc = fma(w0, s0, acc);
        acc = fma(w1, s1, acc);
        acc = fma(w2, s2, acc);
        acc = fma(w3, xw, acc);
        float4 conv = float4(bfloat4(acc));
        float4 y = float4(bfloat4((conv + x3) / {_S5_LIT}));
        float4 rv = ((const device float4*)(res + off))[0];
        ((device uint2*)(out + off))[0] = as_type<uint2>(bfloat4(rv + y));
        s0 = s1;
        s1 = s2;
        s2 = xw;
    }}
    if (t0 == 0) {{
        int end = int(LL);
        if constexpr (HAS_LENS) {{
            end = clamp(clens[b], 0, int(LL));
        }}
        for (int r = 0; r < 3; r++) {{
            int p = end - 3 + r;
            float4 v;
            if (p >= 0) {{
                bool ok = true;
                if constexpr (HAS_MASK) ok = (cmask[(size_t)b * LL + p] != 0);
                v = ok ? float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0])) : float4(0.0f);
            }} else {{
                v = HAS_CS ? ((const device float4*)(cs + ((size_t)b * 3 + (3 + p)) * DD + c))[0] : float4(0.0f);
            }}
            ((device float4*)(ncs + ((size_t)b * 3 + r) * DD + c))[0] = v;
        }}
    }}
"""

def short_conv_residual(
    x: mx.array,
    conv_state: mx.array | None,
    weight: mx.array,
    residual: mx.array,
    *,
    conv_mask: mx.array | None = None,
    conv_lengths: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Fused post_mlp_convolution + (residual + conv.f32).bf16 -> (out bf16, new_cs (B,3,C) f32)."""
    B, L, C = x.shape
    has_cs = int(conv_state is not None)
    has_mask = int(conv_mask is not None)
    has_lens = int(conv_lengths is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    cm_in = conv_mask.astype(mx.uint8) if has_mask else _DUMMY_U8
    cl_in = conv_lengths.astype(mx.int32) if has_lens else _DUMMY_I32
    tt = 16
    ntiles = (L + tt - 1) // tt
    k = _k("pf_short_conv_res4", ("x", "cs", "cw", "res", "cmask", "clens"), ("out", "ncs"), _SC4_RES_SRC)
    return k(
        inputs=[x, cs_in, weight, residual, cm_in, cl_in],
        template=[
            ("DD", C), ("LL", L), ("HAS_CS", has_cs), ("TT", tt),
            ("HAS_MASK", has_mask), ("HAS_LENS", has_lens),
        ],
        grid=(((C // 4 + 255) // 256) * 256, B * ntiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, C), (B, 3, C)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


_SC_ATTN_POST_SRC = f"""
    GL_NOFMA
    uint tok = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint b = tok / LL;
    uint t = tok % LL;
    uint c = lid * 4;
    threadgroup float part[32];

    float4 w0, w1, w2, w3;
    for (uint k = 0; k < 4; k++) {{
        const device bfloat16_t* cp = cw + (c + k) * 4;
        w0[k] = float(cp[0]);
        w1[k] = float(cp[1]);
        w2[k] = float(cp[2]);
        w3[k] = float(cp[3]);
    }}
    auto tap4 = [&](int p) -> float4 {{
        if (p >= 0) {{
            if constexpr (HAS_MASK) {{
                if (!cmask[(size_t)b * LL + p]) return float4(0.0f);
            }}
            return float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0]));
        }}
        int sidx = 3 + p;
        if (!HAS_CS) return float4(0.0f);
        float4 csf = ((const device float4*)(cs + ((size_t)b * 3 + sidx) * DD + c))[0];
        return float4(bfloat4(csf));
    }};
    float4 s0 = tap4(int(t) - 3);
    float4 s1 = tap4(int(t) - 2);
    float4 s2 = tap4(int(t) - 1);
    size_t off = (size_t)tok * DD + c;
    float4 x3 = float4(as_type<bfloat4>(((const device uint2*)(x + off))[0]));
    float4 xw = x3;
    if constexpr (HAS_MASK) {{
        if (!cmask[(size_t)b * LL + t]) xw = float4(0.0f);
    }}
    float4 acc = float4(0.0f);
    acc = fma(w0, s0, acc);
    acc = fma(w1, s1, acc);
    acc = fma(w2, s2, acc);
    acc = fma(w3, xw, acc);
    float4 conv = float4(bfloat4(acc));
    float4 y = float4(bfloat4((conv + x3) / {_S5_LIT}));
    float4 rv = ADD_RES ? (float4(as_type<bfloat4>(((const device uint2*)(bres + off))[0])) + y) : y;
    ((device float4*)(res_out + off))[0] = rv;

    // Exact glue_rms_inv_tg<2304, 576, true> directly from registers (NSG = 18 = TG / 32)
    float racc = 0.0f;
    racc = fma(rv.x, rv.x, racc);
    racc = fma(rv.y, rv.y, racc);
    racc = fma(rv.z, rv.z, racc);
    racc = fma(rv.w, rv.w, racc);
    racc = simd_sum(racc);
    if (lane == 0) part[sg] = racc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = simd_sum(lane < 18u ? part[lane] : 0.0f);
    float inv = metal::precise::rsqrt(tot / float(DD) + 1e-6f);
    ((device float4*)(norm_out + off))[0] = rv * inv;

    if (t == 0) {{
        int end = int(LL);
        if constexpr (HAS_LENS) {{
            end = clamp(clens[b], 0, int(LL));
        }}
        for (int r = 0; r < 3; r++) {{
            int p = end - 3 + r;
            float4 v;
            if (p >= 0) {{
                bool ok = true;
                if constexpr (HAS_MASK) ok = (cmask[(size_t)b * LL + p] != 0);
                v = ok ? float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0])) : float4(0.0f);
            }} else {{
                v = HAS_CS ? ((const device float4*)(cs + ((size_t)b * 3 + (3 + p)) * DD + c))[0] : float4(0.0f);
            }}
            ((device float4*)(ncs + ((size_t)b * 3 + r) * DD + c))[0] = v;
        }}
    }}
"""


def short_conv_attn_post(
    x: mx.array,
    conv_state: mx.array | None,
    weight: mx.array,
    block_residual: mx.array | None,
    *,
    conv_mask: mx.array | None = None,
    conv_lengths: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused post_attention_convolution + residual (if block_residual is not None) + unweighted_rms_norm -> (residual f32, normed_states f32, new_cs f32)."""
    B, L, C = x.shape
    assert C == _D
    has_cs = int(conv_state is not None)
    add_res = int(block_residual is not None)
    has_mask = int(conv_mask is not None)
    has_lens = int(conv_lengths is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    br_in = block_residual if block_residual is not None else _DUMMY_BF16
    cm_in = conv_mask.astype(mx.uint8) if has_mask else _DUMMY_U8
    cl_in = conv_lengths.astype(mx.int32) if has_lens else _DUMMY_I32
    k = _k(
        "pf_sc_attn_post",
        ("x", "cs", "cw", "bres", "cmask", "clens"),
        ("res_out", "norm_out", "ncs"),
        _SC_ATTN_POST_SRC,
    )
    return k(
        inputs=[x, cs_in, weight, br_in, cm_in, cl_in],
        template=[
            ("DD", C), ("LL", L), ("HAS_CS", has_cs), ("ADD_RES", add_res),
            ("HAS_MASK", has_mask), ("HAS_LENS", has_lens),
        ],
        grid=(B * L * _TG_RMS, 1, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, C), (B, L, C), (B, 3, C)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


# ---------------------------------------------------------------------------------------------------- Fused _attend_pre (contiguous clips + v_rms + tau, zero TG mem)
_PARAMS_CACHE: dict = {}
_EPS_CACHE: dict = {}


def _get_eps(eps: float) -> mx.array:
    a = _EPS_CACHE.get(eps)
    if a is None:
        a = _EPS_CACHE[eps] = mx.array([eps], mx.float32)
    return a


_PF_ATTEND_PRE = os.environ.get("LOOPLING_PF_ATTEND_PRE", "1") not in ("0", "")
_PF_OTF_REL = os.environ.get("LOOPLING_PF_OTF_REL", "1") not in ("0", "")
_PF_FG_MIX = os.environ.get("LOOPLING_PF_FG_MIX", "1") not in ("0", "")
_PF_MGIN_DOWN = os.environ.get("LOOPLING_PF_MGIN_DOWN", "1") not in ("0", "")
_MGIN_DOWN_SPK = {1024: 4, 2048: 2, 4096: 2, 8192: 1, 16384: 1}
_PF_ROUTER = os.environ.get("LOOPLING_PF_ROUTER", "1") not in ("0", "")
_PF_DEPTH_RMS = os.environ.get("LOOPLING_PF_DEPTH_RMS", "1") not in ("0", "")

_QKV_CLIP_RMS_SRC = r"""
    GL_NOFMA
    uint tgx = threadgroup_position_in_grid.x;
    uint tok = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint s = tgx * 8 + sg;

    if (s < HQ) {
        uint h = s;
        size_t qoff = ((size_t)tok * HQ + h) * 128 + lane * 4;
        bfloat4 qv = as_type<bfloat4>(((const device uint2*)(q + qoff))[0]);
        bfloat4 sc = as_type<bfloat4>(((const device uint2*)(q_scale + h * 128 + lane * 4))[0]);
        float sv[4] = {
            float(qv.x) * float(sc.x),
            float(qv.y) * float(sc.y),
            float(qv.z) * float(sc.z),
            float(qv.w) * float(sc.w),
        };
        glue_l2_clip_lane128(sv, params[0]);
        bfloat4 ov = bfloat4(bfloat16_t(sv[0]), bfloat16_t(sv[1]), bfloat16_t(sv[2]), bfloat16_t(sv[3]));
        if (HAS_TAU) {
            float tau_v = tau[tok];
            ov = bfloat4(
                bfloat16_t(glue_tau(float(ov.x), tau_v)),
                bfloat16_t(glue_tau(float(ov.y), tau_v)),
                bfloat16_t(glue_tau(float(ov.z), tau_v)),
                bfloat16_t(glue_tau(float(ov.w), tau_v)));
        }
        ((device uint2*)(q_out + qoff))[0] = as_type<uint2>(ov);
    } else if (s < HQ + HKV) {
        uint h = s - HQ;
        size_t koff = ((size_t)tok * HKV + h) * 128 + lane * 4;
        bfloat4 kv = as_type<bfloat4>(((const device uint2*)(k + koff))[0]);
        float sv[4] = {float(kv.x), float(kv.y), float(kv.z), float(kv.w)};
        glue_l2_clip_lane128(sv, params[1]);
        bfloat4 ov = bfloat4(bfloat16_t(sv[0]), bfloat16_t(sv[1]), bfloat16_t(sv[2]), bfloat16_t(sv[3]));
        ((device uint2*)(k_out + koff))[0] = as_type<uint2>(ov);
    } else if (s < HQ + 2 * HKV) {
        uint h = s - HQ - HKV;
        size_t voff = ((size_t)tok * HKV + h) * 128 + lane * 4;
        bfloat4 vv = as_type<bfloat4>(((const device uint2*)(v + voff))[0]);
        float sv[4] = {float(vv.x), float(vv.y), float(vv.z), float(vv.w)};
        float racc = 0.0f;
        racc = fma(sv[0], sv[0], racc);
        racc = fma(sv[1], sv[1], racc);
        racc = fma(sv[2], sv[2], racc);
        racc = fma(sv[3], sv[3], racc);
        racc = simd_sum(racc);
        float inv = metal::precise::rsqrt(racc / 128.0f + params[2]);
        bfloat4 ov = bfloat4(
            glue_rms_apply_nw<bfloat16_t>(sv[0], inv),
            glue_rms_apply_nw<bfloat16_t>(sv[1], inv),
            glue_rms_apply_nw<bfloat16_t>(sv[2], inv),
            glue_rms_apply_nw<bfloat16_t>(sv[3], inv));
        ((device uint2*)(v_out + voff))[0] = as_type<uint2>(ov);
    } else if (HAS_REL && s == HQ + 2 * HKV) {
        if (lane < HQ) {
            uint h = lane;
            size_t roff = (size_t)tok * HQ * 16 + h * 16;
            const device uint2* rp = (const device uint2*)(r + roff);
            const device uint2* rsp = (const device uint2*)(r_scale + h * 16);
            float sv[16];
            #pragma clang loop unroll(full)
            for (int c = 0; c < 4; ++c) {
                bfloat4 rv = as_type<bfloat4>(rp[c]);
                bfloat4 rsc = as_type<bfloat4>(rsp[c]);
                sv[c * 4 + 0] = float(rv.x) * float(rsc.x);
                sv[c * 4 + 1] = float(rv.y) * float(rsc.y);
                sv[c * 4 + 2] = float(rv.z) * float(rsc.z);
                sv[c * 4 + 3] = float(rv.w) * float(rsc.w);
            }
            glue_l2_clip_row16(sv, params[3]);
            device uint2* rop = (device uint2*)(r_out + roff);
            float tau_v = HAS_TAU ? tau[tok] : 1.0f;
            #pragma clang loop unroll(full)
            for (int c = 0; c < 4; ++c) {
                bfloat4 ov = bfloat4(
                    bfloat16_t(sv[c * 4 + 0]),
                    bfloat16_t(sv[c * 4 + 1]),
                    bfloat16_t(sv[c * 4 + 2]),
                    bfloat16_t(sv[c * 4 + 3]));
                if (HAS_TAU) {
                    ov = bfloat4(
                        bfloat16_t(glue_tau(float(ov.x), tau_v)),
                        bfloat16_t(glue_tau(float(ov.y), tau_v)),
                        bfloat16_t(glue_tau(float(ov.z), tau_v)),
                        bfloat16_t(glue_tau(float(ov.w), tau_v)));
                }
                rop[c] = as_type<uint2>(ov);
            }
        }
    }
"""


def qkv_clip_rms(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_scale: mx.array,
    *,
    num_heads: int,
    num_kv_heads: int,
    thr_q: float,
    thr_k: float,
    eps: float,
    tau: mx.array | None = None,
    r: mx.array | None = None,
    r_scale: mx.array | None = None,
    thr_r: float = 8.0,
):
    """Lean zero-spill fused q_l2_clip (+tau) + k_l2_clip + v_rms_norm [+ rel_l2_clip (+tau)] -> transposed views."""
    B, L, _ = q.shape
    has_tau = int(tau is not None)
    has_rel = int(r is not None)
    if tau is not None:
        tau_in = mx.broadcast_to(tau.reshape(-1, L), (B, L)).astype(mx.float32)
    else:
        tau_in = _DUMMY_F32
    r_in = r if has_rel else _DUMMY_BF16
    rs_in = r_scale if has_rel else _DUMMY_BF16
    pkey = ("qkv_cr4", thr_q, thr_k, eps, thr_r)
    params = _PARAMS_CACHE.get(pkey)
    if params is None:
        params = _PARAMS_CACHE[pkey] = mx.array([thr_q, thr_k, eps, thr_r], dtype=mx.float32)
    ntg_x = (num_heads + 2 * num_kv_heads + (1 if has_rel else 0) + 7) // 8
    k_fn = _k(
        "pf_qkv_clip_rms_r",
        ("q", "k", "v", "q_scale", "tau", "params", "r", "r_scale"),
        ("q_out", "k_out", "v_out", "r_out"),
        _QKV_CLIP_RMS_SRC,
    )
    qo, ko, vo, ro = k_fn(
        inputs=[q, k, v, q_scale, tau_in, params, r_in, rs_in],
        template=[("HQ", num_heads), ("HKV", num_kv_heads), ("HAS_TAU", has_tau), ("HAS_REL", has_rel)],
        grid=(ntg_x * 256, B * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (B, L, num_heads, 128),
            (B, L, num_kv_heads, 128),
            (B, L, num_kv_heads, 128),
            (B, L, num_heads, 16) if has_rel else (1,),
        ],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )
    if has_rel:
        return qo.transpose(0, 2, 1, 3), ko.transpose(0, 2, 1, 3), vo.transpose(0, 2, 1, 3), ro
    return qo.transpose(0, 2, 1, 3), ko.transpose(0, 2, 1, 3), vo.transpose(0, 2, 1, 3)


# ---------------------------------------------------------------------------------------------------- Fused SlidingAttention (k_sconv+k_clip) + (v_sconv+v_rms) + q_clip
_SWA_QKV_CONV_CLIP_SRC = f"""
    GL_NOFMA
    uint tgx = threadgroup_position_in_grid.x;
    uint tile = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint s = tgx * 8 + sg;

    uint ntiles = (uint(LL) + TT - 1) / TT;
    uint b = tile / ntiles;
    uint t0 = (tile % ntiles) * TT;

    if (s < HQ) {{
        uint h = s;
        bfloat4 sc = as_type<bfloat4>(((const device uint2*)(q_scale + h * 128 + lane * 4))[0]);
        float4 scf = float4(sc);
        float thr_q = params[0];
        for (uint tk = 0; tk < TT; tk++) {{
            uint t = t0 + tk;
            if (t >= uint(LL)) break;
            size_t qoff = (((size_t)b * LL + t) * HQ + h) * 128 + lane * 4;
            bfloat4 qv = as_type<bfloat4>(((const device uint2*)(q + qoff))[0]);
            float sv[4] = {{
                float(qv.x) * scf.x,
                float(qv.y) * scf.y,
                float(qv.z) * scf.z,
                float(qv.w) * scf.w,
            }};
            glue_l2_clip_lane128(sv, thr_q);
            bfloat4 ov = bfloat4(bfloat16_t(sv[0]), bfloat16_t(sv[1]), bfloat16_t(sv[2]), bfloat16_t(sv[3]));
            ((device uint2*)(q_out + qoff))[0] = as_type<uint2>(ov);
        }}
    }} else if (s < HQ + HKV) {{
        uint h = s - HQ;
        uint c = h * 128 + lane * 4;
        uint DD = HKV * 128;
        float4 w0, w1, w2, w3;
        for (uint j = 0; j < 4; j++) {{
            const device bfloat16_t* cp = kcw + (c + j) * 4;
            w0[j] = float(cp[0]); w1[j] = float(cp[1]); w2[j] = float(cp[2]); w3[j] = float(cp[3]);
        }}
        auto tap4 = [&](int p) -> float4 {{
            if (p >= 0) {{
                if constexpr (HAS_MASK) {{
                    if (!cmask[(size_t)b * LL + p]) return float4(0.0f);
                }}
                return float4(as_type<bfloat4>(((const device uint2*)(k + ((size_t)b * LL + p) * DD + c))[0]));
            }}
            if (!HAS_KCS) return float4(0.0f);
            float4 csf = ((const device float4*)(kcs + ((size_t)b * 3 + (3 + p)) * DD + c))[0];
            return float4(bfloat4(csf));
        }};
        float4 s0 = tap4(int(t0) - 3);
        float4 s1 = tap4(int(t0) - 2);
        float4 s2 = tap4(int(t0) - 1);
        float thr_k = params[1];
        for (uint tk = 0; tk < TT; tk++) {{
            uint t = t0 + tk;
            if (t >= uint(LL)) break;
            size_t koff = ((size_t)b * LL + t) * DD + c;
            float4 x3 = float4(as_type<bfloat4>(((const device uint2*)(k + koff))[0]));
            float4 xw = x3;
            if constexpr (HAS_MASK) {{
                if (!cmask[(size_t)b * LL + t]) xw = float4(0.0f);
            }}
            float4 acc = float4(0.0f);
            acc = fma(w0, s0, acc);
            acc = fma(w1, s1, acc);
            acc = fma(w2, s2, acc);
            acc = fma(w3, xw, acc);
            float4 conv = float4(bfloat4(acc));
            bfloat4 ybf = bfloat4((conv + x3) / {_S5_LIT});
            float sv[4] = {{float(ybf.x), float(ybf.y), float(ybf.z), float(ybf.w)}};
            glue_l2_clip_lane128(sv, thr_k);
            bfloat4 ov = bfloat4(bfloat16_t(sv[0]), bfloat16_t(sv[1]), bfloat16_t(sv[2]), bfloat16_t(sv[3]));
            ((device uint2*)(k_out + koff))[0] = as_type<uint2>(ov);
            s0 = s1; s1 = s2; s2 = xw;
        }}
        if (t0 == 0) {{
            int end = int(LL);
            if constexpr (HAS_LENS) {{
                end = clamp(clens[b], 0, int(LL));
            }}
            for (int r = 0; r < 3; r++) {{
                int p = end - 3 + r;
                float4 fv;
                if (p >= 0) {{
                    bool ok = true;
                    if constexpr (HAS_MASK) ok = (cmask[(size_t)b * LL + p] != 0);
                    fv = ok ? float4(as_type<bfloat4>(((const device uint2*)(k + ((size_t)b * LL + p) * DD + c))[0])) : float4(0.0f);
                }} else {{
                    fv = HAS_KCS ? ((const device float4*)(kcs + ((size_t)b * 3 + (3 + p)) * DD + c))[0] : float4(0.0f);
                }}
                ((device float4*)(nkcs + ((size_t)b * 3 + r) * DD + c))[0] = fv;
            }}
        }}
    }} else if (s < HQ + 2 * HKV) {{
        uint h = s - HQ - HKV;
        uint c = h * 128 + lane * 4;
        uint DD = HKV * 128;
        float4 w0, w1, w2, w3;
        for (uint j = 0; j < 4; j++) {{
            const device bfloat16_t* cp = vcw + (c + j) * 4;
            w0[j] = float(cp[0]); w1[j] = float(cp[1]); w2[j] = float(cp[2]); w3[j] = float(cp[3]);
        }}
        auto tap4 = [&](int p) -> float4 {{
            if (p >= 0) {{
                if constexpr (HAS_MASK) {{
                    if (!cmask[(size_t)b * LL + p]) return float4(0.0f);
                }}
                return float4(as_type<bfloat4>(((const device uint2*)(v + ((size_t)b * LL + p) * DD + c))[0]));
            }}
            if (!HAS_VCS) return float4(0.0f);
            float4 csf = ((const device float4*)(vcs + ((size_t)b * 3 + (3 + p)) * DD + c))[0];
            return float4(bfloat4(csf));
        }};
        float4 s0 = tap4(int(t0) - 3);
        float4 s1 = tap4(int(t0) - 2);
        float4 s2 = tap4(int(t0) - 1);
        float eps = params[2];
        for (uint tk = 0; tk < TT; tk++) {{
            uint t = t0 + tk;
            if (t >= uint(LL)) break;
            size_t voff = ((size_t)b * LL + t) * DD + c;
            float4 x3 = float4(as_type<bfloat4>(((const device uint2*)(v + voff))[0]));
            float4 xw = x3;
            if constexpr (HAS_MASK) {{
                if (!cmask[(size_t)b * LL + t]) xw = float4(0.0f);
            }}
            float4 acc = float4(0.0f);
            acc = fma(w0, s0, acc);
            acc = fma(w1, s1, acc);
            acc = fma(w2, s2, acc);
            acc = fma(w3, xw, acc);
            float4 conv = float4(bfloat4(acc));
            bfloat4 ybf = bfloat4((conv + x3) / {_S5_LIT});
            float sv[4] = {{float(ybf.x), float(ybf.y), float(ybf.z), float(ybf.w)}};
            float racc = 0.0f;
            racc = fma(sv[0], sv[0], racc);
            racc = fma(sv[1], sv[1], racc);
            racc = fma(sv[2], sv[2], racc);
            racc = fma(sv[3], sv[3], racc);
            racc = simd_sum(racc);
            float inv = metal::precise::rsqrt(racc / 128.0f + eps);
            bfloat4 ov = bfloat4(
                glue_rms_apply_nw<bfloat16_t>(sv[0], inv),
                glue_rms_apply_nw<bfloat16_t>(sv[1], inv),
                glue_rms_apply_nw<bfloat16_t>(sv[2], inv),
                glue_rms_apply_nw<bfloat16_t>(sv[3], inv));
            ((device uint2*)(v_out + voff))[0] = as_type<uint2>(ov);
            s0 = s1; s1 = s2; s2 = xw;
        }}
        if (t0 == 0) {{
            int end = int(LL);
            if constexpr (HAS_LENS) {{
                end = clamp(clens[b], 0, int(LL));
            }}
            for (int r = 0; r < 3; r++) {{
                int p = end - 3 + r;
                float4 fv;
                if (p >= 0) {{
                    bool ok = true;
                    if constexpr (HAS_MASK) ok = (cmask[(size_t)b * LL + p] != 0);
                    fv = ok ? float4(as_type<bfloat4>(((const device uint2*)(v + ((size_t)b * LL + p) * DD + c))[0])) : float4(0.0f);
                }} else {{
                    fv = HAS_VCS ? ((const device float4*)(vcs + ((size_t)b * 3 + (3 + p)) * DD + c))[0] : float4(0.0f);
                }}
                ((device float4*)(nvcs + ((size_t)b * 3 + r) * DD + c))[0] = fv;
            }}
        }}
    }} else if (HAS_REL && s == HQ + 2 * HKV) {{
        if (lane < HQ) {{
            uint h = lane;
            const device uint2* rsp = (const device uint2*)(r_scale + h * 16);
            float4 scf[4];
            #pragma clang loop unroll(full)
            for (int c = 0; c < 4; ++c) {{
                scf[c] = float4(as_type<bfloat4>(rsp[c]));
            }}
            float thr_r = params[3];
            for (uint tk = 0; tk < TT; tk++) {{
                uint t = t0 + tk;
                if (t >= uint(LL)) break;
                size_t roff = ((size_t)b * LL + t) * HQ * 16 + h * 16;
                const device uint2* rp = (const device uint2*)(r + roff);
                float sv[16];
                #pragma clang loop unroll(full)
                for (int c = 0; c < 4; ++c) {{
                    bfloat4 rv = as_type<bfloat4>(rp[c]);
                    sv[c * 4 + 0] = float(rv.x) * scf[c].x;
                    sv[c * 4 + 1] = float(rv.y) * scf[c].y;
                    sv[c * 4 + 2] = float(rv.z) * scf[c].z;
                    sv[c * 4 + 3] = float(rv.w) * scf[c].w;
                }}
                glue_l2_clip_row16(sv, thr_r);
                device uint2* rop = (device uint2*)(r_out + roff);
                #pragma clang loop unroll(full)
                for (int c = 0; c < 4; ++c) {{
                    bfloat4 ov = bfloat4(
                        bfloat16_t(sv[c * 4 + 0]),
                        bfloat16_t(sv[c * 4 + 1]),
                        bfloat16_t(sv[c * 4 + 2]),
                        bfloat16_t(sv[c * 4 + 3]));
                    rop[c] = as_type<uint2>(ov);
                }}
            }}
        }}
    }}
"""


def swa_qkv_conv_clip(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_scale: mx.array,
    k_cw: mx.array,
    v_cw: mx.array,
    k_cs: mx.array | None,
    v_cs: mx.array | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    thr_q: float,
    thr_k: float,
    eps: float,
    r: mx.array | None = None,
    r_scale: mx.array | None = None,
    thr_r: float = 8.0,
    conv_mask: mx.array | None = None,
    conv_lengths: mx.array | None = None,
):
    """1-launch SlidingAttention fusion: q_l2_clip + (k_sconv->k_l2_clip) + (v_sconv->v_rms_norm) [+ rel_l2_clip]."""
    B, L, _ = q.shape
    Dkv = num_kv_heads * 128
    has_kcs = int(k_cs is not None)
    has_vcs = int(v_cs is not None)
    has_rel = int(r is not None)
    has_mask = int(conv_mask is not None)
    has_lens = int(conv_lengths is not None)
    kcs_in = k_cs if k_cs is not None else _DUMMY_F32
    vcs_in = v_cs if v_cs is not None else _DUMMY_F32
    r_in = r if has_rel else _DUMMY_BF16
    rs_in = r_scale if has_rel else _DUMMY_BF16
    cm_in = conv_mask.astype(mx.uint8) if has_mask else _DUMMY_U8
    cl_in = conv_lengths.astype(mx.int32) if has_lens else _DUMMY_I32
    pkey = ("qkv_cr4", thr_q, thr_k, eps, thr_r)
    params = _PARAMS_CACHE.get(pkey)
    if params is None:
        params = _PARAMS_CACHE[pkey] = mx.array([thr_q, thr_k, eps, thr_r], dtype=mx.float32)
    tt = 16
    ntiles = (L + tt - 1) // tt
    ntg_x = (num_heads + 2 * num_kv_heads + (1 if has_rel else 0) + 7) // 8
    k_fn = _k(
        "pf_swa_qkv_cc_r",
        ("q", "k", "v", "q_scale", "kcw", "vcw", "kcs", "vcs", "params", "r", "r_scale", "cmask", "clens"),
        ("q_out", "k_out", "v_out", "nkcs", "nvcs", "r_out"),
        _SWA_QKV_CONV_CLIP_SRC,
    )
    qo, ko, vo, nkcs, nvcs, ro = k_fn(
        inputs=[q, k, v, q_scale, k_cw, v_cw, kcs_in, vcs_in, params, r_in, rs_in, cm_in, cl_in],
        template=[
            ("HQ", num_heads), ("HKV", num_kv_heads), ("LL", L),
            ("TT", tt), ("HAS_KCS", has_kcs), ("HAS_VCS", has_vcs), ("HAS_REL", has_rel),
            ("HAS_MASK", has_mask), ("HAS_LENS", has_lens),
        ],
        grid=(ntg_x * 256, B * ntiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (B, L, num_heads, 128), (B, L, num_kv_heads, 128), (B, L, num_kv_heads, 128),
            (B, 3, Dkv), (B, 3, Dkv), (B, L, num_heads, 16) if has_rel else (1,),
        ],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32, mx.float32, mx.bfloat16],
    )
    if has_rel:
        return qo.transpose(0, 2, 1, 3), ko.transpose(0, 2, 1, 3), vo.transpose(0, 2, 1, 3), nkcs, nvcs, ro
    return qo.transpose(0, 2, 1, 3), ko.transpose(0, 2, 1, 3), vo.transpose(0, 2, 1, 3), nkcs, nvcs


# ---------------------------------------------------------------------------------------------------- Fused interleaved SoftcappedSwiGLU
_SWIGLU_INT_SRC = r"""
    uint d4 = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (d4 >= HH4) return;
    uint4 raw = ((const device uint4*)(gu + (size_t)row * HH4 * 8))[d4];
    bfloat4 p0 = as_type<bfloat4>(raw.xy);
    bfloat4 p1 = as_type<bfloat4>(raw.zw);
    float gc = caps[0], uc = caps[1];
    bfloat4 ov = bfloat4(
        bfloat16_t(glue_swiglu(float(p0.x), float(p0.y), gc, uc)),
        bfloat16_t(glue_swiglu(float(p0.z), float(p0.w), gc, uc)),
        bfloat16_t(glue_swiglu(float(p1.x), float(p1.y), gc, uc)),
        bfloat16_t(glue_swiglu(float(p1.z), float(p1.w), gc, uc)));
    ((device uint2*)(out + (size_t)row * HH4 * 4))[d4] = as_type<uint2>(ov);
"""


def swiglu_interleaved(gu: mx.array, gate_softcap: float, up_softcap: float) -> mx.array:
    """128-bit coalesced SoftcappedSwiGLU on alternating (gate, up) rows -> (..., H) bf16."""
    H2 = gu.shape[-1]
    H = H2 // 2
    assert H % 4 == 0 and gu.dtype == mx.bfloat16
    nrows = gu.size // H2
    hh4 = H // 4
    ckey = ("swiglu", gate_softcap, up_softcap)
    caps = _PARAMS_CACHE.get(ckey)
    if caps is None:
        caps = _PARAMS_CACHE[ckey] = mx.array([gate_softcap, up_softcap], dtype=mx.float32)
    tg = hh4 if hh4 <= 1024 and hh4 % 32 == 0 else 256
    gx = ((hh4 + tg - 1) // tg) * tg
    k = _k("pf_swiglu_int4", ("gu", "caps"), ("out",), _SWIGLU_INT_SRC)
    return k(
        inputs=[gu, caps],
        template=[("HH4", hh4)],
        grid=(gx, nrows, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[gu.shape[:-1] + (H,)],
        output_dtypes=[mx.bfloat16],
    )[0]


# ---------------------------------------------------------------------------------------------------- Direct f32 -> bf16 RMSNorm
_RMS_F32_BF16_SRC = r"""
    GL_NOFMA
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint c = lid * 4;
    threadgroup float part[32];
    size_t off = (size_t)row * DD + c;
    float4 xv = ((const device float4*)(x + off))[0];
    float racc = 0.0f;
    racc = fma(xv.x, xv.x, racc);
    racc = fma(xv.y, xv.y, racc);
    racc = fma(xv.z, xv.z, racc);
    racc = fma(xv.w, xv.w, racc);
    racc = simd_sum(racc);
    if (lane == 0) part[sg] = racc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = simd_sum(lane < 18u ? part[lane] : 0.0f);
    float inv = metal::precise::rsqrt(tot / float(DD) + eps[0]);
    bfloat4 wv = as_type<bfloat4>(((const device uint2*)(nw + c))[0]);
    bfloat4 ov = bfloat4(
        bfloat16_t(glue_rms_apply<float>(xv.x, inv, wv.x)),
        bfloat16_t(glue_rms_apply<float>(xv.y, inv, wv.y)),
        bfloat16_t(glue_rms_apply<float>(xv.z, inv, wv.z)),
        bfloat16_t(glue_rms_apply<float>(xv.w, inv, wv.w)));
    ((device uint2*)(out + off))[0] = as_type<uint2>(ov);
"""

_RMS_F32_BF16_QKV_SRC = r"""
    GL_NOFMA
    uint row = threadgroup_position_in_grid.x;
    uint g = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint c = lid * 4;
    threadgroup float part[32];
    size_t xoff = ((size_t)row * 3 + g) * DD + c;
    float4 xv = ((const device float4*)(x + xoff))[0];
    float racc = 0.0f;
    racc = fma(xv.x, xv.x, racc);
    racc = fma(xv.y, xv.y, racc);
    racc = fma(xv.z, xv.z, racc);
    racc = fma(xv.w, xv.w, racc);
    racc = simd_sum(racc);
    if (lane == 0) part[sg] = racc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = simd_sum(lane < 18u ? part[lane] : 0.0f);
    float inv = metal::precise::rsqrt(tot / float(DD) + eps[0]);
    const device bfloat16_t* nwp = (g == 0 ? nw0 : (g == 1 ? nw1 : nw2));
    bfloat4 wv = as_type<bfloat4>(((const device uint2*)(nwp + c))[0]);
    bfloat4 ov = bfloat4(
        bfloat16_t(glue_rms_apply<float>(xv.x, inv, wv.x)),
        bfloat16_t(glue_rms_apply<float>(xv.y, inv, wv.y)),
        bfloat16_t(glue_rms_apply<float>(xv.z, inv, wv.z)),
        bfloat16_t(glue_rms_apply<float>(xv.w, inv, wv.w)));
    device bfloat16_t* dst = (g == 0 ? o0 : (g == 1 ? o1 : o2)) + (size_t)row * DD + c;
    ((device uint2*)dst)[0] = as_type<uint2>(ov);
"""


_DEPTH_ATTN_RMS_SRC = r"""
    constexpr int N = K + 1;
    uint row = threadgroup_position_in_grid.x;
    uint rows = threadgroups_per_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    threadgroup float buf[N][16];
    threadgroup float nr_s[D];
    if (PRE_UNW_RMS) {
        GL_NOFMA
        uint c0 = (sg * 2 * 32 + lane) * 4;
        uint c1 = ((sg * 2 + 1) * 32 + lane) * 4;
        float4 p0 = float4(as_type<bfloat4>(((const device uint2*)(h + (size_t)row * D + c0))[0]));
        float4 p1 = float4(as_type<bfloat4>(((const device uint2*)(h + (size_t)row * D + c1))[0]));
        float r0 = 0.0f, r1 = 0.0f;
        r0 = fma(p0.x, p0.x, r0); r0 = fma(p0.y, p0.y, r0); r0 = fma(p0.z, p0.z, r0); r0 = fma(p0.w, p0.w, r0);
        r1 = fma(p1.x, p1.x, r1); r1 = fma(p1.y, p1.y, r1); r1 = fma(p1.z, p1.z, r1); r1 = fma(p1.w, p1.w, r1);
        r0 = simd_sum(r0); r1 = simd_sum(r1);
        if (lane == 0) { nr_s[sg * 2] = r0; nr_s[sg * 2 + 1] = r1; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv0 = metal::precise::rsqrt(simd_sum(lane < 18u ? nr_s[lane] : 0.0f) / float(D) + 1e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        p0 *= inv0; p1 *= inv0;
        ((threadgroup float4*)(nr_s + c0))[0] = p0;
        ((threadgroup float4*)(nr_s + c1))[0] = p1;
        if (OUT_NR) {
            ((device float4*)(nr_out + (size_t)row * D + c0))[0] = p0;
            ((device float4*)(nr_out + (size_t)row * D + c1))[0] = p1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float acc[N];
    for (int i = 0; i < N; i++) acc[i] = 0.0f;
    float xc[8][N];
    int it_idx = 0;
    for (int d = lid; d < D; d += 288, it_idx++) {
        for (int i = 0; i < K; i++) xc[it_idx][i] = blocks[(size_t(i) * rows + row) * D + d];
        xc[it_idx][K] = PRE_UNW_RMS ? nr_s[d] : part_in[size_t(row) * D + d];
        float qv = queries[d];
        for (int i = 0; i < N; i++) acc[i] = fma(xc[it_idx][i], qv, acc[i]);
    }
    for (int i = 0; i < N; i++) {
        float v = simd_sum(acc[i]);
        if (lane == 0) buf[i][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[N];
    for (int i = 0; i < N; i++) {
        float tv = 0.0f;
        for (int s = 0; s < 9; s++) tv += buf[i][s];
        w[i] = tv * scale[0];
    }
    float m = -INFINITY;
    for (int i = 0; i < N; i++) m = max(m, w[i]);
    float z = 0.0f;
    for (int i = 0; i < N; i++) { w[i] = metal::exp(w[i] - m); z += w[i]; }
    for (int i = 0; i < N; i++) w[i] /= z;
    it_idx = 0;
    for (int d = lid; d < D; d += 288, it_idx++) {
        float o = 0.0f;
        for (int i = 0; i < N; i++) o = fma(w[i], xc[it_idx][i], o);
        nr_s[d] = o;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        GL_NOFMA
        uint c0 = (sg * 2 * 32 + lane) * 4;
        uint c1 = ((sg * 2 + 1) * 32 + lane) * 4;
        float4 xv0 = ((threadgroup float4*)(nr_s + c0))[0];
        float4 xv1 = ((threadgroup float4*)(nr_s + c1))[0];
        float r0 = 0.0f, r1 = 0.0f;
        r0 = fma(xv0.x, xv0.x, r0); r0 = fma(xv0.y, xv0.y, r0); r0 = fma(xv0.z, xv0.z, r0); r0 = fma(xv0.w, xv0.w, r0);
        r1 = fma(xv1.x, xv1.x, r1); r1 = fma(xv1.y, xv1.y, r1); r1 = fma(xv1.z, xv1.z, r1); r1 = fma(xv1.w, xv1.w, r1);
        r0 = simd_sum(r0); r1 = simd_sum(r1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) { nr_s[sg * 2] = r0; nr_s[sg * 2 + 1] = r1; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv2 = metal::precise::rsqrt(simd_sum(lane < 18u ? nr_s[lane] : 0.0f) / float(D) + eps[0]);
        bfloat4 wv0 = as_type<bfloat4>(((const device uint2*)(nw + c0))[0]);
        bfloat4 wv1 = as_type<bfloat4>(((const device uint2*)(nw + c1))[0]);
        bfloat4 ov0(
            bfloat16_t(glue_rms_apply<float>(xv0.x, inv2, wv0.x)),
            bfloat16_t(glue_rms_apply<float>(xv0.y, inv2, wv0.y)),
            bfloat16_t(glue_rms_apply<float>(xv0.z, inv2, wv0.z)),
            bfloat16_t(glue_rms_apply<float>(xv0.w, inv2, wv0.w)));
        bfloat4 ov1(
            bfloat16_t(glue_rms_apply<float>(xv1.x, inv2, wv1.x)),
            bfloat16_t(glue_rms_apply<float>(xv1.y, inv2, wv1.y)),
            bfloat16_t(glue_rms_apply<float>(xv1.z, inv2, wv1.z)),
            bfloat16_t(glue_rms_apply<float>(xv1.w, inv2, wv1.w)));
        ((device uint2*)(out + (size_t)row * D + c0))[0] = as_type<uint2>(ov0);
        ((device uint2*)(out + (size_t)row * D + c1))[0] = as_type<uint2>(ov1);
    }
"""

_URMS_DA_F32_SRC = r"""
    constexpr int N = K + 1;
    uint row = threadgroup_position_in_grid.x;
    uint rows = threadgroups_per_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    threadgroup float buf[N * Q][16];
    threadgroup float nr_s[D];
    {
        GL_NOFMA
        uint c0 = (sg * 2 * 32 + lane) * 4;
        uint c1 = ((sg * 2 + 1) * 32 + lane) * 4;
        float4 p0 = float4(as_type<bfloat4>(((const device uint2*)(h + (size_t)row * D + c0))[0]));
        float4 p1 = float4(as_type<bfloat4>(((const device uint2*)(h + (size_t)row * D + c1))[0]));
        float r0 = 0.0f, r1 = 0.0f;
        r0 = fma(p0.x, p0.x, r0); r0 = fma(p0.y, p0.y, r0); r0 = fma(p0.z, p0.z, r0); r0 = fma(p0.w, p0.w, r0);
        r1 = fma(p1.x, p1.x, r1); r1 = fma(p1.y, p1.y, r1); r1 = fma(p1.z, p1.z, r1); r1 = fma(p1.w, p1.w, r1);
        r0 = simd_sum(r0); r1 = simd_sum(r1);
        if (lane == 0) { nr_s[sg * 2] = r0; nr_s[sg * 2 + 1] = r1; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv0 = metal::precise::rsqrt(simd_sum(lane < 18u ? nr_s[lane] : 0.0f) / float(D) + 1e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        p0 *= inv0; p1 *= inv0;
        ((threadgroup float4*)(nr_s + c0))[0] = p0;
        ((threadgroup float4*)(nr_s + c1))[0] = p1;
        if (OUT_NR) {
            ((device float4*)(nr_out + (size_t)row * D + c0))[0] = p0;
            ((device float4*)(nr_out + (size_t)row * D + c1))[0] = p1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float acc[N * Q];
    for (int i = 0; i < N * Q; i++) acc[i] = 0.0f;
    float xc[8][N];
    int it_idx = 0;
    for (int d = lid; d < D; d += 288, it_idx++) {
        for (int i = 0; i < K; i++) xc[it_idx][i] = blocks[(size_t(i) * rows + row) * D + d];
        xc[it_idx][K] = nr_s[d];
        for (int j = 0; j < Q; j++) {
            float qv = queries[j * D + d];
            for (int i = 0; i < N; i++) acc[i * Q + j] += xc[it_idx][i] * qv;
        }
    }
    for (int i = 0; i < N * Q; i++) {
        float v = simd_sum(acc[i]);
        if (lane == 0) buf[i][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[N * Q];
    for (int i = 0; i < N * Q; i++) {
        float tv = 0.0f;
        for (int s = 0; s < 9; s++) tv += buf[i][s];
        w[i] = tv * scale[0];
    }
    for (int j = 0; j < Q; j++) {
        float m = -INFINITY;
        for (int i = 0; i < N; i++) m = max(m, w[i * Q + j]);
        float z = 0.0f;
        for (int i = 0; i < N; i++) { w[i * Q + j] = metal::exp(w[i * Q + j] - m); z += w[i * Q + j]; }
        for (int i = 0; i < N; i++) w[i * Q + j] /= z;
    }
    it_idx = 0;
    for (int d = lid; d < D; d += 288, it_idx++) {
        for (int j = 0; j < Q; j++) {
            float o = 0.0f;
            for (int i = 0; i < N; i++) o += w[i * Q + j] * xc[it_idx][i];
            out[(size_t(row) * Q + j) * D + d] = o;
        }
    }
"""

_SC_ATTN_DA_SRC = f"""
    constexpr int N = K + 1;
    uint row = threadgroup_position_in_grid.x;
    uint rows = threadgroups_per_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint b = row / LL;
    uint t = row % LL;
    threadgroup float buf[N][16];
    threadgroup float nr_s[D];
    {{
        GL_NOFMA
        auto conv_chunk = [&](uint c) -> float4 {{
            float4 w0, w1, w2, w3;
            for (uint k = 0; k < 4; k++) {{
                const device bfloat16_t* cp = cw + (c + k) * 4;
                w0[k] = float(cp[0]); w1[k] = float(cp[1]); w2[k] = float(cp[2]); w3[k] = float(cp[3]);
            }}
            auto tap4 = [&](int p) -> float4 {{
                if (p >= 0) {{
                    if constexpr (HAS_MASK) {{
                        if (!cmask[(size_t)b * LL + p]) return float4(0.0f);
                    }}
                    return float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * D + c))[0]));
                }}
                int sidx = 3 + p;
                if (!HAS_CS) return float4(0.0f);
                return float4(bfloat4(((const device float4*)(cs + ((size_t)b * 3 + sidx) * D + c))[0]));
            }};
            float4 s0 = tap4(int(t) - 3), s1 = tap4(int(t) - 2), s2 = tap4(int(t) - 1);
            size_t off = (size_t)row * D + c;
            float4 x3 = float4(as_type<bfloat4>(((const device uint2*)(x + off))[0]));
            float4 xw = x3;
            if constexpr (HAS_MASK) {{
                if (!cmask[(size_t)b * LL + t]) xw = float4(0.0f);
            }}
            float4 acc = float4(0.0f);
            acc = fma(w0, s0, acc); acc = fma(w1, s1, acc); acc = fma(w2, s2, acc); acc = fma(w3, xw, acc);
            float4 conv = float4(bfloat4(acc));
            float4 y = float4(bfloat4((conv + x3) / {_S5_LIT}));
            float4 rv = ADD_RES ? (float4(as_type<bfloat4>(((const device uint2*)(bres + off))[0])) + y) : y;
            ((device float4*)(res_out + off))[0] = rv;
            if (t == 0) {{
                int end = int(LL);
                if constexpr (HAS_LENS) {{
                    end = clamp(clens[b], 0, int(LL));
                }}
                for (int r = 0; r < 3; r++) {{
                    int p = end - 3 + r;
                    float4 v;
                    if (p >= 0) {{
                        bool ok = true;
                        if constexpr (HAS_MASK) ok = (cmask[(size_t)b * LL + p] != 0);
                        v = ok ? float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * D + c))[0])) : float4(0.0f);
                    }} else {{
                        v = HAS_CS ? ((const device float4*)(cs + ((size_t)b * 3 + (3 + p)) * D + c))[0] : float4(0.0f);
                    }}
                    ((device float4*)(ncs + ((size_t)b * 3 + r) * D + c))[0] = v;
                }}
            }}
            return rv;
        }};
        uint c0 = (sg * 2 * 32 + lane) * 4;
        uint c1 = ((sg * 2 + 1) * 32 + lane) * 4;
        float4 rv0 = conv_chunk(c0);
        float4 rv1 = conv_chunk(c1);
        float r0 = 0.0f, r1 = 0.0f;
        r0 = fma(rv0.x, rv0.x, r0); r0 = fma(rv0.y, rv0.y, r0); r0 = fma(rv0.z, rv0.z, r0); r0 = fma(rv0.w, rv0.w, r0);
        r1 = fma(rv1.x, rv1.x, r1); r1 = fma(rv1.y, rv1.y, r1); r1 = fma(rv1.z, rv1.z, r1); r1 = fma(rv1.w, rv1.w, r1);
        r0 = simd_sum(r0); r1 = simd_sum(r1);
        if (lane == 0) {{ nr_s[sg * 2] = r0; nr_s[sg * 2 + 1] = r1; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv0 = metal::precise::rsqrt(simd_sum(lane < 18u ? nr_s[lane] : 0.0f) / float(D) + 1e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ((threadgroup float4*)(nr_s + c0))[0] = rv0 * inv0;
        ((threadgroup float4*)(nr_s + c1))[0] = rv1 * inv0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float acc[N];
    for (int i = 0; i < N; i++) acc[i] = 0.0f;
    float xc[8][N];
    int it_idx = 0;
    for (int d = lid; d < D; d += 288, it_idx++) {{
        for (int i = 0; i < K; i++) xc[it_idx][i] = blocks[(size_t(i) * rows + row) * D + d];
        xc[it_idx][K] = nr_s[d];
        float qv = queries[d];
        for (int i = 0; i < N; i++) acc[i] = fma(xc[it_idx][i], qv, acc[i]);
    }}
    for (int i = 0; i < N; i++) {{
        float v = simd_sum(acc[i]);
        if (lane == 0) buf[i][sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[N];
    for (int i = 0; i < N; i++) {{
        float tv = 0.0f;
        for (int s = 0; s < 9; s++) tv += buf[i][s];
        w[i] = tv * scale[0];
    }}
    float m = -INFINITY;
    for (int i = 0; i < N; i++) m = max(m, w[i]);
    float z = 0.0f;
    for (int i = 0; i < N; i++) {{ w[i] = metal::exp(w[i] - m); z += w[i]; }}
    for (int i = 0; i < N; i++) w[i] /= z;
    it_idx = 0;
    for (int d = lid; d < D; d += 288, it_idx++) {{
        float o = 0.0f;
        for (int i = 0; i < N; i++) o = fma(w[i], xc[it_idx][i], o);
        if constexpr (DO_RMS) nr_s[d] = o;
        else out_f32[(size_t)row * D + d] = o;
    }}
    if constexpr (DO_RMS) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {{
            GL_NOFMA
            uint c0 = (sg * 2 * 32 + lane) * 4;
            uint c1 = ((sg * 2 + 1) * 32 + lane) * 4;
            float4 xv0 = ((threadgroup float4*)(nr_s + c0))[0];
            float4 xv1 = ((threadgroup float4*)(nr_s + c1))[0];
            float r0 = 0.0f, r1 = 0.0f;
            r0 = fma(xv0.x, xv0.x, r0); r0 = fma(xv0.y, xv0.y, r0); r0 = fma(xv0.z, xv0.z, r0); r0 = fma(xv0.w, xv0.w, r0);
            r1 = fma(xv1.x, xv1.x, r1); r1 = fma(xv1.y, xv1.y, r1); r1 = fma(xv1.z, xv1.z, r1); r1 = fma(xv1.w, xv1.w, r1);
            r0 = simd_sum(r0); r1 = simd_sum(r1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {{ nr_s[sg * 2] = r0; nr_s[sg * 2 + 1] = r1; }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv2 = metal::precise::rsqrt(simd_sum(lane < 18u ? nr_s[lane] : 0.0f) / float(D) + eps[0]);
            bfloat4 wv0 = as_type<bfloat4>(((const device uint2*)(nw + c0))[0]);
            bfloat4 wv1 = as_type<bfloat4>(((const device uint2*)(nw + c1))[0]);
            bfloat4 ov0(
                bfloat16_t(glue_rms_apply<float>(xv0.x, inv2, wv0.x)),
                bfloat16_t(glue_rms_apply<float>(xv0.y, inv2, wv0.y)),
                bfloat16_t(glue_rms_apply<float>(xv0.z, inv2, wv0.z)),
                bfloat16_t(glue_rms_apply<float>(xv0.w, inv2, wv0.w)));
            bfloat4 ov1(
                bfloat16_t(glue_rms_apply<float>(xv1.x, inv2, wv1.x)),
                bfloat16_t(glue_rms_apply<float>(xv1.y, inv2, wv1.y)),
                bfloat16_t(glue_rms_apply<float>(xv1.z, inv2, wv1.z)),
                bfloat16_t(glue_rms_apply<float>(xv1.w, inv2, wv1.w)));
            ((device uint2*)(out_bf16 + (size_t)row * D + c0))[0] = as_type<uint2>(ov0);
            ((device uint2*)(out_bf16 + (size_t)row * D + c1))[0] = as_type<uint2>(ov1);
        }}
    }}
"""


def unw_rms_depth_attn_rms_bf16(h: mx.array, blocks: mx.array, query: mx.array, ln, *, need_nr: bool):
    """Fused unweighted_rms_norm(h.f32) + attention_input(blocks, nr) + rms_norm_f32_bf16 (TG=288)."""
    B, L, D = h.shape
    assert D == _D
    sc = _PARAMS_CACHE.get(("da_sc", D))
    if sc is None:
        sc = _PARAMS_CACHE[("da_sc", D)] = mx.array([D ** -0.5], dtype=mx.float32)
    k = _k(
        "pf_urms_da_rms",
        ("h", "part_in", "blocks", "queries", "scale", "nw", "eps"),
        ("out", "nr_out"),
        _DEPTH_ATTN_RMS_SRC,
    )
    out, nr_out = k(
        inputs=[h, _DUMMY_F32, blocks, query.reshape(-1), sc, ln.weight, _get_eps(ln.eps)],
        template=[("K", blocks.shape[0]), ("D", D), ("PRE_UNW_RMS", 1), ("OUT_NR", int(need_nr))],
        grid=(B * L * 288, 1, 1),
        threadgroup=(288, 1, 1),
        output_shapes=[(B, L, D), (B, L, D) if need_nr else (1,)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    return out, (nr_out if need_nr else None)


def unw_rms_depth_attn_f32(h: mx.array, blocks: mx.array, query: mx.array, *, need_nr: bool):
    """Fused unweighted_rms_norm(h.f32) + attention_input(blocks, nr) -> ((B, L, Q, D) f32, nr_out)."""
    B, L, D = h.shape
    assert D == _D
    q2 = query.reshape(-1, D)
    Q = q2.shape[0]
    sc = _PARAMS_CACHE.get(("da_sc", D))
    if sc is None:
        sc = _PARAMS_CACHE[("da_sc", D)] = mx.array([D ** -0.5], dtype=mx.float32)
    k = _k(
        "pf_urms_da_f32",
        ("h", "blocks", "queries", "scale"),
        ("out", "nr_out"),
        _URMS_DA_F32_SRC,
    )
    out, nr_out = k(
        inputs=[h, blocks, q2, sc],
        template=[("K", blocks.shape[0]), ("D", D), ("Q", Q), ("OUT_NR", int(need_nr))],
        grid=(B * L * 288, 1, 1),
        threadgroup=(288, 1, 1),
        output_shapes=[(B, L, Q, D), (B, L, D) if need_nr else (1,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return out, (nr_out if need_nr else None)


def short_conv_attn_depth_attn(
    x: mx.array,
    conv_state: mx.array | None,
    weight: mx.array,
    block_residual: mx.array | None,
    blocks: mx.array,
    query: mx.array,
    ln=None,
    *,
    conv_mask: mx.array | None = None,
    conv_lengths: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused post_attn_conv + residual + unw_rms + mlp_input [+ post_attn_ln when ln is not None] (TG=288)."""
    B, L, D = x.shape
    assert D == _D
    has_cs = int(conv_state is not None)
    add_res = int(block_residual is not None)
    do_rms = int(ln is not None)
    has_mask = int(conv_mask is not None)
    has_lens = int(conv_lengths is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    br_in = block_residual if block_residual is not None else _DUMMY_BF16
    nw_in = ln.weight if ln is not None else _DUMMY_BF16
    eps_in = _get_eps(ln.eps) if ln is not None else _DUMMY_F32
    cm_in = conv_mask.astype(mx.uint8) if has_mask else _DUMMY_U8
    cl_in = conv_lengths.astype(mx.int32) if has_lens else _DUMMY_I32
    sc = _PARAMS_CACHE.get(("da_sc", D))
    if sc is None:
        sc = _PARAMS_CACHE[("da_sc", D)] = mx.array([D ** -0.5], dtype=mx.float32)
    k = _k(
        "pf_sc_attn_da",
        ("x", "cs", "cw", "bres", "blocks", "queries", "scale", "nw", "eps", "cmask", "clens"),
        ("res_out", "ncs", "out_bf16", "out_f32"),
        _SC_ATTN_DA_SRC,
    )
    res_out, ncs, out_bf16, out_f32 = k(
        inputs=[x, cs_in, weight, br_in, blocks, query.reshape(-1), sc, nw_in, eps_in, cm_in, cl_in],
        template=[
            ("K", blocks.shape[0]), ("D", D), ("LL", L),
            ("HAS_CS", has_cs), ("ADD_RES", add_res), ("DO_RMS", do_rms),
            ("HAS_MASK", has_mask), ("HAS_LENS", has_lens),
        ],
        grid=(B * L * 288, 1, 1),
        threadgroup=(288, 1, 1),
        output_shapes=[
            (B, L, D),
            (B, 3, D),
            (B, L, D) if do_rms else (1,),
            (1,) if do_rms else (B, L, D),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.bfloat16, mx.float32],
    )
    return res_out, (out_bf16 if do_rms else out_f32), ncs


def depth_attn_rms_bf16(blocks: mx.array, partial: mx.array, query: mx.array, ln) -> mx.array:
    """Fused mlp_input(blocks, partial) + rms_norm_f32_bf16 -> (B, L, D) bf16 (TG=288)."""
    B, L, D = partial.shape
    assert D == _D
    sc = _PARAMS_CACHE.get(("da_sc", D))
    if sc is None:
        sc = _PARAMS_CACHE[("da_sc", D)] = mx.array([D ** -0.5], dtype=mx.float32)
    k = _k(
        "pf_urms_da_rms",
        ("h", "part_in", "blocks", "queries", "scale", "nw", "eps"),
        ("out", "nr_out"),
        _DEPTH_ATTN_RMS_SRC,
    )
    return k(
        inputs=[_DUMMY_BF16, partial, blocks, query.reshape(-1), sc, ln.weight, _get_eps(ln.eps)],
        template=[("K", blocks.shape[0]), ("D", D), ("PRE_UNW_RMS", 0), ("OUT_NR", 0)],
        grid=(B * L * 288, 1, 1),
        threadgroup=(288, 1, 1),
        output_shapes=[(B, L, D), (1,)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )[0]


def rms_norm_f32_bf16(x: mx.array, ln) -> mx.array:
    """1-pass exact f32 -> bf16 nn.RMSNorm for D=2304 (avoids f32 DRAM write + cast)."""
    B, L, D = x.shape
    assert D == _D
    k = _k("pf_rms_f32_bf16", ("x", "nw", "eps"), ("out",), _RMS_F32_BF16_SRC)
    return k(
        inputs=[x, ln.weight, _get_eps(ln.eps)],
        template=[("DD", D)],
        grid=(B * L * _TG_RMS, 1, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, D)],
        output_dtypes=[mx.bfloat16],
    )[0]


def rms_norm_f32_bf16_qkv(ai: mx.array, lns) -> tuple[mx.array, mx.array, mx.array]:
    """1-launch batched f32 -> bf16 Q/K/V RMSNorm for FullDecoderLayer (ai shape (B, L, 3, D))."""
    B, L, _, D = ai.shape
    assert D == _D
    k = _k("pf_rms_f32_bf16_qkv", ("x", "nw0", "nw1", "nw2", "eps"), ("o0", "o1", "o2"), _RMS_F32_BF16_QKV_SRC)
    return k(
        inputs=[ai, lns[0].weight, lns[1].weight, lns[2].weight, _get_eps(lns[0].eps)],
        template=[("DD", D)],
        grid=(B * L * _TG_RMS, 3, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, D), (B, L, D), (B, L, D)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )


_RMS_F32_BF16_QKV_SEP_SRC = r"""
    GL_NOFMA
    uint row = threadgroup_position_in_grid.x;
    uint g = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint c = lid * 4;
    threadgroup float part[32];
    size_t xoff = (size_t)row * DD + c;
    const device float* xp = (g == 0 ? x0 : (g == 1 ? x1 : x2)) + xoff;
    float4 xv = ((const device float4*)xp)[0];
    float racc = 0.0f;
    racc = fma(xv.x, xv.x, racc);
    racc = fma(xv.y, xv.y, racc);
    racc = fma(xv.z, xv.z, racc);
    racc = fma(xv.w, xv.w, racc);
    racc = simd_sum(racc);
    if (lane == 0) part[sg] = racc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = simd_sum(lane < 18u ? part[lane] : 0.0f);
    float inv = metal::precise::rsqrt(tot / float(DD) + eps[0]);
    const device bfloat16_t* nwp = SHARED_LN ? nw0 : (g == 0 ? nw0 : (g == 1 ? nw1 : nw2));
    bfloat4 wv = as_type<bfloat4>(((const device uint2*)(nwp + c))[0]);
    bfloat4 ov = bfloat4(
        bfloat16_t(glue_rms_apply<float>(xv.x, inv, wv.x)),
        bfloat16_t(glue_rms_apply<float>(xv.y, inv, wv.y)),
        bfloat16_t(glue_rms_apply<float>(xv.z, inv, wv.z)),
        bfloat16_t(glue_rms_apply<float>(xv.w, inv, wv.w)));
    device bfloat16_t* dst = (g == 0 ? o0 : (g == 1 ? o1 : o2)) + xoff;
    ((device uint2*)dst)[0] = as_type<uint2>(ov);
"""


def rms_norm_f32_bf16_qkv_sep(x0: mx.array, x1: mx.array, x2: mx.array, lns, *, shared_ln: bool):
    """1-launch batched f32 -> bf16 RMSNorm over 3 separate (B, L, D) arrays."""
    B, L, D = x0.shape
    assert D == _D
    k = _k(
        "pf_rms_f32_bf16_qkv_sep",
        ("x0", "x1", "x2", "nw0", "nw1", "nw2", "eps"),
        ("o0", "o1", "o2"),
        _RMS_F32_BF16_QKV_SEP_SRC,
    )
    return k(
        inputs=[x0, x1, x2, lns[0].weight, lns[1].weight, lns[2].weight, _get_eps(lns[0].eps)],
        template=[("DD", D), ("SHARED_LN", int(shared_ln))],
        grid=(B * L * _TG_RMS, 3, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, D), (B, L, D), (B, L, D)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )


# ---------------------------------------------------------------------------------------------------- MoE combine
_UC_SRC = r"""
    GL_NOFMA
    uint d = thread_position_in_grid.x;
    uint tok = thread_position_in_grid.y;
    if (d >= DD) return;
    float ev[8], ww[8];
    for (uint e = 0; e < 8; e++) {
        uint p = tok * 8 + e;
        ev[e] = float(dn[(size_t)inv[p] * DD + d]);
        ww[e] = wts[p];
    }
    out[(size_t)tok * DD + d] = bfloat16_t(glue_moe_combine8(ev, ww));
"""


_UC4_SRC = r"""
    GL_NOFMA
    uint d4 = thread_position_in_grid.x;
    uint tok = thread_position_in_grid.y;
    uint d = d4 * 4;
    if (d >= DD) return;
    float es0[8], es1[8], es2[8], es3[8];
    float ww[8];
    for (uint e = 0; e < 8; e++) {
        uint p = tok * 8 + e;
        bfloat4 dv = as_type<bfloat4>(((const device uint2*)(dn + (size_t)inv[p] * DD + d))[0]);
        es0[e] = float(dv.x);
        es1[e] = float(dv.y);
        es2[e] = float(dv.z);
        es3[e] = float(dv.w);
        ww[e] = wts[p];
    }
    bfloat4 res(
        bfloat16_t(glue_moe_combine8(es0, ww)),
        bfloat16_t(glue_moe_combine8(es1, ww)),
        bfloat16_t(glue_moe_combine8(es2, ww)),
        bfloat16_t(glue_moe_combine8(es3, ww)));
    ((device uint2*)(out + (size_t)tok * DD + d))[0] = as_type<uint2>(res);
"""


_UC8_SRC = r"""
    GL_NOFMA
    uint d8 = thread_position_in_threadgroup.x;
    uint tok = threadgroup_position_in_grid.y;
    threadgroup uint inv_s[8];
    threadgroup float wts_s[8];
    if (d8 < 8u) {
        inv_s[d8] = inv[tok * 8u + d8];
        wts_s[d8] = wts[tok * 8u + d8];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint d = d8 * 8u;
    if (d >= DD) return;
    float e0[8], e1[8], e2[8], e3[8], e4[8], e5[8], e6[8], e7[8];
    float ww[8];
    #pragma unroll
    for (uint e = 0; e < 8u; e++) {
        uint4 raw = ((const device uint4*)(dn + (size_t)inv_s[e] * DD + d))[0];
        bfloat4 d0 = as_type<bfloat4>(raw.xy);
        bfloat4 d1 = as_type<bfloat4>(raw.zw);
        e0[e] = float(d0.x); e1[e] = float(d0.y); e2[e] = float(d0.z); e3[e] = float(d0.w);
        e4[e] = float(d1.x); e5[e] = float(d1.y); e6[e] = float(d1.z); e7[e] = float(d1.w);
        ww[e] = wts_s[e];
    }
    bfloat4 r0(
        bfloat16_t(glue_moe_combine8(e0, ww)),
        bfloat16_t(glue_moe_combine8(e1, ww)),
        bfloat16_t(glue_moe_combine8(e2, ww)),
        bfloat16_t(glue_moe_combine8(e3, ww)));
    bfloat4 r1(
        bfloat16_t(glue_moe_combine8(e4, ww)),
        bfloat16_t(glue_moe_combine8(e5, ww)),
        bfloat16_t(glue_moe_combine8(e6, ww)),
        bfloat16_t(glue_moe_combine8(e7, ww)));
    uint2 u0 = as_type<uint2>(r0), u1 = as_type<uint2>(r1);
    ((device uint4*)(out + (size_t)tok * DD + d))[0] = uint4(u0.x, u0.y, u1.x, u1.y);
"""


def unsort_combine(dn_sorted: mx.array, inv_order: mx.array, routing_weights: mx.array) -> mx.array:
    """Fuse _scatter_unsort + (es.f32 * wts[...,None]).sum(-2).astype(bf16) for K=8."""
    B, L, E = routing_weights.shape
    D = dn_sorted.shape[-1]
    assert E == 8 and dn_sorted.dtype == mx.bfloat16
    if D % 8 == 0 and (D // 8) % 32 == 0 and D // 8 <= 1024:
        tg = D // 8
        k = _k("pf_unsort_combine8", ("dn", "inv", "wts"), ("out",), _UC8_SRC)
        return k(
            inputs=[dn_sorted, inv_order.astype(mx.uint32), routing_weights.astype(mx.float32)],
            template=[("DD", D)],
            grid=(tg, B * L, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(B, L, D)],
            output_dtypes=[mx.bfloat16],
        )[0]
    if D % 4 == 0:
        k = _k("pf_unsort_combine4", ("dn", "inv", "wts"), ("out",), _UC4_SRC)
        return k(
            inputs=[dn_sorted, inv_order.astype(mx.uint32), routing_weights.astype(mx.float32)],
            template=[("DD", D)],
            grid=(((D // 4 + 255) // 256) * 256, B * L, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D)],
            output_dtypes=[mx.bfloat16],
        )[0]
    k = _k("pf_unsort_combine", ("dn", "inv", "wts"), ("out",), _UC_SRC)
    return k(
        inputs=[dn_sorted, inv_order.astype(mx.uint32), routing_weights.astype(mx.float32)],
        template=[("DD", D)],
        grid=(((D + 255) // 256) * 256, B * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, D)],
        output_dtypes=[mx.bfloat16],
    )[0]


# ---------------------------------------------------------------------------------------------------- ShortConvolution
_SC_SRC = f"""
    uint c = thread_position_in_grid.x;
    uint tok = thread_position_in_grid.y;
    uint b = tok / LL;
    uint t = tok % LL;
    if (c >= DD) return;
    float x3 = float(x[(size_t)tok * DD + c]);
    auto tap = [&](int idx) -> float {{
        int p = int(t) - idx;
        if (p >= 0) {{
            if constexpr (HAS_MASK) {{
                if (!cmask[(size_t)b * LL + p]) return 0.0f;
            }}
            return float(x[(size_t)(b * LL + p) * DD + c]);
        }}
        int sidx = 3 + p;
        return HAS_CS ? float(bfloat16_t(cs[((size_t)b * 3 + sidx) * DD + c])) : 0.0f;
    }};
    float acc = 0.0f;
    acc = fma(float(cw[c * 4 + 0]), tap(3), acc);
    acc = fma(float(cw[c * 4 + 1]), tap(2), acc);
    acc = fma(float(cw[c * 4 + 2]), tap(1), acc);
    acc = fma(float(cw[c * 4 + 3]), tap(0), acc);
    float conv = float(bfloat16_t(acc));
    out[(size_t)tok * DD + c] = bfloat16_t((conv + x3) / {_S5_LIT});
    if (t == 0) {{
        int end = int(LL);
        if constexpr (HAS_LENS) {{
            end = clamp(clens[b], 0, int(LL));
        }}
        for (int r = 0; r < 3; r++) {{
            int p = end - 3 + r;
            float v;
            if (p >= 0) {{
                bool ok = true;
                if constexpr (HAS_MASK) ok = (cmask[(size_t)b * LL + p] != 0);
                v = ok ? float(x[(size_t)(b * LL + p) * DD + c]) : 0.0f;
            }} else {{
                v = HAS_CS ? cs[((size_t)b * 3 + (3 + p)) * DD + c] : 0.0f;
            }}
            ncs[((size_t)b * 3 + r) * DD + c] = v;
        }}
    }}
"""

_SC4_SRC = f"""
    uint c4 = thread_position_in_grid.x;
    uint c = c4 * 4;
    if (c >= DD) return;
    uint tile = thread_position_in_grid.y;
    uint ntiles = (LL + TT - 1) / TT;
    uint b = tile / ntiles;
    uint t0 = (tile % ntiles) * TT;
    float4 w0, w1, w2, w3;
    for (uint k = 0; k < 4; k++) {{
        const device bfloat16_t* cp = cw + (c + k) * 4;
        w0[k] = float(cp[0]);
        w1[k] = float(cp[1]);
        w2[k] = float(cp[2]);
        w3[k] = float(cp[3]);
    }}
    auto tap4 = [&](int p) -> float4 {{
        if (p >= 0) {{
            if constexpr (HAS_MASK) {{
                if (!cmask[(size_t)b * LL + p]) return float4(0.0f);
            }}
            return float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0]));
        }}
        int sidx = 3 + p;
        if (!HAS_CS) return float4(0.0f);
        float4 csf = ((const device float4*)(cs + ((size_t)b * 3 + sidx) * DD + c))[0];
        return float4(bfloat4(csf));
    }};
    float4 s0 = tap4(int(t0) - 3);
    float4 s1 = tap4(int(t0) - 2);
    float4 s2 = tap4(int(t0) - 1);
    for (uint tk = 0; tk < TT; tk++) {{
        uint t = t0 + tk;
        if (t >= LL) break;
        float4 x3 = float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + t) * DD + c))[0]));
        float4 xw = x3;
        if constexpr (HAS_MASK) {{
            if (!cmask[(size_t)b * LL + t]) xw = float4(0.0f);
        }}
        float4 acc = float4(0.0f);
        acc = fma(w0, s0, acc);
        acc = fma(w1, s1, acc);
        acc = fma(w2, s2, acc);
        acc = fma(w3, xw, acc);
        float4 conv = float4(bfloat4(acc));
        ((device uint2*)(out + ((size_t)b * LL + t) * DD + c))[0] = as_type<uint2>(bfloat4((conv + x3) / {_S5_LIT}));
        s0 = s1;
        s1 = s2;
        s2 = xw;
    }}
    if (t0 == 0) {{
        int end = int(LL);
        if constexpr (HAS_LENS) {{
            end = clamp(clens[b], 0, int(LL));
        }}
        for (int r = 0; r < 3; r++) {{
            int p = end - 3 + r;
            float4 v;
            if (p >= 0) {{
                bool ok = true;
                if constexpr (HAS_MASK) ok = (cmask[(size_t)b * LL + p] != 0);
                v = ok ? float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0])) : float4(0.0f);
            }} else {{
                v = HAS_CS ? ((const device float4*)(cs + ((size_t)b * 3 + (3 + p)) * DD + c))[0] : float4(0.0f);
            }}
            ((device float4*)(ncs + ((size_t)b * 3 + r) * DD + c))[0] = v;
        }}
    }}
"""


def short_conv(
    x: mx.array,
    conv_state: mx.array | None,
    weight: mx.array,
    *,
    conv_mask: mx.array | None = None,
    conv_lengths: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """K=4 causal depthwise conv + residual/sqrt(5) -> (out bf16, new_cs (B,3,C) f32)."""
    B, L, C = x.shape
    has_cs = int(conv_state is not None)
    has_mask = int(conv_mask is not None)
    has_lens = int(conv_lengths is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    cm_in = conv_mask.astype(mx.uint8) if has_mask else _DUMMY_U8
    cl_in = conv_lengths.astype(mx.int32) if has_lens else _DUMMY_I32
    if C % 4 == 0 and L >= 16:
        tt = 16
        ntiles = (L + tt - 1) // tt
        k = _k("pf_short_conv4", ("x", "cs", "cw", "cmask", "clens"), ("out", "ncs"), _SC4_SRC)
        return k(
            inputs=[x, cs_in, weight, cm_in, cl_in],
            template=[
                ("DD", C), ("LL", L), ("HAS_CS", has_cs), ("TT", tt),
                ("HAS_MASK", has_mask), ("HAS_LENS", has_lens),
            ],
            grid=(((C // 4 + 255) // 256) * 256, B * ntiles, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, C), (B, 3, C)],
            output_dtypes=[mx.bfloat16, mx.float32],
        )
    k = _k("pf_short_conv", ("x", "cs", "cw", "cmask", "clens"), ("out", "ncs"), _SC_SRC)
    return k(
        inputs=[x, cs_in, weight, cm_in, cl_in],
        template=[
            ("DD", C), ("LL", L), ("HAS_CS", has_cs),
            ("HAS_MASK", has_mask), ("HAS_LENS", has_lens),
        ],
        grid=(((C + 255) // 256) * 256, B * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, C), (B, 3, C)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


# ---------------------------------------------------------------------------------------------------- DynamicShortConv
_DSC_SRC = f"""
    uint c = thread_position_in_grid.x;
    uint tok = thread_position_in_grid.y;
    uint b = tok / LL;
    uint t = tok % LL;
    if (c >= CC) return;
    uint h = c / HD;
    float x0 = float(x[(size_t)tok * CC + c]);
    auto tap = [&](int lag) -> float {{
        int p = int(t) - lag;
        if (p >= 0) {{
            if constexpr (HAS_MASK) {{
                if (!cmask[(size_t)b * LL + p]) return 0.0f;
            }}
            return float(x[(size_t)(b * LL + p) * CC + c]);
        }}
        int sidx = 3 + p;
        return HAS_CS ? cs[((size_t)b * 3 + sidx) * CC + c] : 0.0f;
    }};
    float p0 = tap(0) * (float(sw[0 * CC + c]) + float(dw[((size_t)tok * HH + h) * 4 + 0]));
    float p1 = tap(1) * (float(sw[1 * CC + c]) + float(dw[((size_t)tok * HH + h) * 4 + 1]));
    float p2 = tap(2) * (float(sw[2 * CC + c]) + float(dw[((size_t)tok * HH + h) * 4 + 2]));
    float p3 = tap(3) * (float(sw[3 * CC + c]) + float(dw[((size_t)tok * HH + h) * 4 + 3]));
    float s = ((p0 + p1) + p2) + p3;
    out[(size_t)tok * CC + c] = bfloat16_t((s + x0) / {_S9_LIT});
    if (t == 0) {{
        int end = int(LL);
        if constexpr (HAS_LENS) {{
            end = clamp(clens[b], 0, int(LL));
        }}
        for (int r = 0; r < 3; r++) {{
            int p = end - 3 + r;
            float v;
            if (p >= 0) {{
                bool ok = true;
                if constexpr (HAS_MASK) ok = (cmask[(size_t)b * LL + p] != 0);
                v = ok ? float(x[(size_t)(b * LL + p) * CC + c]) : 0.0f;
            }} else {{
                v = HAS_CS ? cs[((size_t)b * 3 + (3 + p)) * CC + c] : 0.0f;
            }}
            ncs[((size_t)b * 3 + r) * CC + c] = v;
        }}
    }}
"""

_DSC4_SRC = f"""
    uint c4 = thread_position_in_grid.x;
    uint c = c4 * 4;
    if (c >= CC) return;
    uint tile = thread_position_in_grid.y;
    uint ntiles = (LL + TT - 1) / TT;
    uint b = tile / ntiles;
    uint t0 = (tile % ntiles) * TT;
    uint h = c / HD;
    float4 sw0 = float4(as_type<bfloat4>(((const device uint2*)(sw + 0 * CC + c))[0]));
    float4 sw1 = float4(as_type<bfloat4>(((const device uint2*)(sw + 1 * CC + c))[0]));
    float4 sw2 = float4(as_type<bfloat4>(((const device uint2*)(sw + 2 * CC + c))[0]));
    float4 sw3 = float4(as_type<bfloat4>(((const device uint2*)(sw + 3 * CC + c))[0]));
    auto tap4 = [&](int p) -> float4 {{
        if (p >= 0) {{
            if constexpr (HAS_MASK) {{
                if (!cmask[(size_t)b * LL + p]) return float4(0.0f);
            }}
            return float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * CC + c))[0]));
        }}
        int sidx = 3 + p;
        return HAS_CS ? ((const device float4*)(cs + ((size_t)b * 3 + sidx) * CC + c))[0] : float4(0.0f);
    }};
    float4 s3 = tap4(int(t0) - 3);
    float4 s2 = tap4(int(t0) - 2);
    float4 s1 = tap4(int(t0) - 1);
    for (uint tk = 0; tk < TT; tk++) {{
        uint t = t0 + tk;
        if (t >= LL) break;
        size_t tok = (size_t)b * LL + t;
        float4 x0 = float4(as_type<bfloat4>(((const device uint2*)(x + tok * CC + c))[0]));
        float4 xw = x0;
        if constexpr (HAS_MASK) {{
            if (!cmask[(size_t)b * LL + t]) xw = float4(0.0f);
        }}
        bfloat4 dw4 = as_type<bfloat4>(((const device uint2*)(dw + (tok * HH + h) * 4))[0]);
        float4 p0 = xw * (sw0 + float(dw4.x));
        float4 p1 = s1 * (sw1 + float(dw4.y));
        float4 p2 = s2 * (sw2 + float(dw4.z));
        float4 p3 = s3 * (sw3 + float(dw4.w));
        float4 s = ((p0 + p1) + p2) + p3;
        ((device uint2*)(out + tok * CC + c))[0] = as_type<uint2>(bfloat4((s + x0) / {_S9_LIT}));
        s3 = s2;
        s2 = s1;
        s1 = xw;
    }}
    if (t0 == 0) {{
        int end = int(LL);
        if constexpr (HAS_LENS) {{
            end = clamp(clens[b], 0, int(LL));
        }}
        for (int r = 0; r < 3; r++) {{
            int p = end - 3 + r;
            float4 v;
            if (p >= 0) {{
                bool ok = true;
                if constexpr (HAS_MASK) ok = (cmask[(size_t)b * LL + p] != 0);
                v = ok ? float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * CC + c))[0])) : float4(0.0f);
            }} else {{
                v = HAS_CS ? ((const device float4*)(cs + ((size_t)b * 3 + (3 + p)) * CC + c))[0] : float4(0.0f);
            }}
            ((device float4*)(ncs + ((size_t)b * 3 + r) * CC + c))[0] = v;
        }}
    }}
"""


def dyn_short_conv(
    x: mx.array,
    dyn_weights: mx.array,
    static_weight: mx.array,
    conv_state: mx.array | None,
    num_heads: int,
    *,
    conv_mask: mx.array | None = None,
    conv_lengths: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """K=4 dynamic short conv + residual/sqrt(9) -> (out bf16, new_cs (B,3,C) f32)."""
    B, L, C = x.shape
    hd = C // num_heads
    has_cs = int(conv_state is not None)
    has_mask = int(conv_mask is not None)
    has_lens = int(conv_lengths is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    cm_in = conv_mask.astype(mx.uint8) if has_mask else _DUMMY_U8
    cl_in = conv_lengths.astype(mx.int32) if has_lens else _DUMMY_I32
    if C % 4 == 0 and hd % 4 == 0 and L >= 16:
        tt = 16
        ntiles = (L + tt - 1) // tt
        k = _k("pf_dyn_short_conv4", ("x", "dw", "sw", "cs", "cmask", "clens"), ("out", "ncs"), _DSC4_SRC)
        return k(
            inputs=[x, dyn_weights, static_weight, cs_in, cm_in, cl_in],
            template=[
                ("CC", C), ("LL", L), ("HH", num_heads), ("HD", hd), ("HAS_CS", has_cs), ("TT", tt),
                ("HAS_MASK", has_mask), ("HAS_LENS", has_lens),
            ],
            grid=(((C // 4 + 255) // 256) * 256, B * ntiles, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, C), (B, 3, C)],
            output_dtypes=[mx.bfloat16, mx.float32],
        )
    k = _k("pf_dyn_short_conv", ("x", "dw", "sw", "cs", "cmask", "clens"), ("out", "ncs"), _DSC_SRC)
    return k(
        inputs=[x, dyn_weights, static_weight, cs_in, cm_in, cl_in],
        template=[
            ("CC", C), ("LL", L), ("HH", num_heads), ("HD", hd), ("HAS_CS", has_cs),
            ("HAS_MASK", has_mask), ("HAS_LENS", has_lens),
        ],
        grid=(((C + 255) // 256) * 256, B * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, C), (B, 3, C)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


# ---------------------------------------------------------------------------------------------------- MessageGate fused (1-pass float4 register-RMS)
_MGIN_SINGLE_SRC = r"""
    GL_NOFMA
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint c = lid * 4;
    threadgroup float ph[32], pm[32];
    float4 hv = ((const device float4*)(h + size_t(row) * DD + c))[0];
    float4 mv = ((const device float4*)(msg + (size_t(row) * MSTRIDE + SLOT) * DD + c))[0];
    float ah = 0.0f, am = 0.0f;
    ah = fma(hv.x, hv.x, ah); ah = fma(hv.y, hv.y, ah); ah = fma(hv.z, hv.z, ah); ah = fma(hv.w, hv.w, ah);
    am = fma(mv.x, mv.x, am); am = fma(mv.y, mv.y, am); am = fma(mv.z, mv.z, am); am = fma(mv.w, mv.w, am);
    ah = simd_sum(ah); am = simd_sum(am);
    if (lane == 0) { ph[sg] = ah; pm[sg] = am; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invh = metal::precise::rsqrt(simd_sum(lane < 18u ? ph[lane] : 0.0f) / float(DD) + 1e-6f);
    float invm = metal::precise::rsqrt(simd_sum(lane < 18u ? pm[lane] : 0.0f) / float(DD) + 1e-6f);
    device bfloat16_t* dst = out + size_t(row) * 2 * DD;
    ((device uint2*)(dst + c))[0] = as_type<uint2>(bfloat4(hv * invh));
    ((device uint2*)(dst + DD + c))[0] = as_type<uint2>(bfloat4(mv * invm));
"""

_MGIN_QKV_SRC = r"""
    GL_NOFMA
    uint row = threadgroup_position_in_grid.x;
    uint g = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint c = lid * 4;
    threadgroup float ph[32], pm[32];
    float4 hv = ((const device float4*)(h + (SHARED_H ? size_t(row) : size_t(row) * 3 + g) * DD + c))[0];
    float4 mv = ((const device float4*)(msg + (size_t(row) * 4 + g) * DD + c))[0];
    float ah = 0.0f, am = 0.0f;
    ah = fma(hv.x, hv.x, ah); ah = fma(hv.y, hv.y, ah); ah = fma(hv.z, hv.z, ah); ah = fma(hv.w, hv.w, ah);
    am = fma(mv.x, mv.x, am); am = fma(mv.y, mv.y, am); am = fma(mv.z, mv.z, am); am = fma(mv.w, mv.w, am);
    ah = simd_sum(ah); am = simd_sum(am);
    if (lane == 0) { ph[sg] = ah; pm[sg] = am; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invh = metal::precise::rsqrt(simd_sum(lane < 18u ? ph[lane] : 0.0f) / float(DD) + 1e-6f);
    float invm = metal::precise::rsqrt(simd_sum(lane < 18u ? pm[lane] : 0.0f) / float(DD) + 1e-6f);
    device bfloat16_t* dst = (g == 0 ? o0 : (g == 1 ? o1 : o2)) + size_t(row) * 2 * DD;
    ((device uint2*)(dst + c))[0] = as_type<uint2>(bfloat4(hv * invh));
    ((device uint2*)(dst + DD + c))[0] = as_type<uint2>(bfloat4(mv * invm));
"""

_MIXNORM_SINGLE_SRC = r"""
    GL_NOFMA
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint c = lid * 4;
    threadgroup float part[32];
    size_t base = size_t(row) * DD + c;
    size_t mbase = (size_t(row) * 4 + SLOT) * DD + c;
    float4 flv = ((const device float4*)(fl + base))[0];
    float4 mlv = ((const device float4*)(ml + base))[0];
    float4 hv = ((const device float4*)(h + base))[0];
    float4 mv = ((const device float4*)(msg + mbase))[0];
    float4 cb = float4(
        glue_msg_combine<0>(flv.x, mlv.x, hv.x, mv.x),
        glue_msg_combine<0>(flv.y, mlv.y, hv.y, mv.y),
        glue_msg_combine<0>(flv.z, mlv.z, hv.z, mv.z),
        glue_msg_combine<0>(flv.w, mlv.w, hv.w, mv.w));
    float acc = 0.0f;
    acc = fma(cb.x, cb.x, acc); acc = fma(cb.y, cb.y, acc); acc = fma(cb.z, cb.z, acc); acc = fma(cb.w, cb.w, acc);
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = metal::precise::rsqrt(simd_sum(lane < 18u ? part[lane] : 0.0f) / float(DD) + eps[0]);
    bfloat4 wv = as_type<bfloat4>(((const device uint2*)(nw + c))[0]);
    bfloat4 ov = bfloat4(
        bfloat16_t(glue_rms_apply<float>(cb.x, inv, wv.x)),
        bfloat16_t(glue_rms_apply<float>(cb.y, inv, wv.y)),
        bfloat16_t(glue_rms_apply<float>(cb.z, inv, wv.z)),
        bfloat16_t(glue_rms_apply<float>(cb.w, inv, wv.w)));
    ((device uint2*)(out + base))[0] = as_type<uint2>(ov);
"""

_MIXNORM_QKV_SRC = r"""
    GL_NOFMA
    uint row = threadgroup_position_in_grid.x;
    uint g = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31;
    uint sg = lid >> 5;
    uint c = lid * 4;
    threadgroup float part[32];
    size_t base = size_t(row) * DD + c;
    size_t hbase = (SHARED_H ? size_t(row) : size_t(row) * 3 + g) * DD + c;
    size_t mbase = (size_t(row) * 4 + g) * DD + c;
    const device float* flp = (g == 0 ? fl0 : (g == 1 ? fl1 : fl2)) + base;
    const device float* mlp = (g == 0 ? ml0 : (g == 1 ? ml1 : ml2)) + base;
    const device bfloat16_t* nwp = SHARED_LN ? nw0 : (g == 0 ? nw0 : (g == 1 ? nw1 : nw2));
    device bfloat16_t* dst = (g == 0 ? o0 : (g == 1 ? o1 : o2)) + base;
    float4 flv = ((const device float4*)flp)[0];
    float4 mlv = ((const device float4*)mlp)[0];
    float4 hv = ((const device float4*)(h + hbase))[0];
    float4 mv = ((const device float4*)(msg + mbase))[0];
    float4 cb = float4(
        glue_msg_combine<0>(flv.x, mlv.x, hv.x, mv.x),
        glue_msg_combine<0>(flv.y, mlv.y, hv.y, mv.y),
        glue_msg_combine<0>(flv.z, mlv.z, hv.z, mv.z),
        glue_msg_combine<0>(flv.w, mlv.w, hv.w, mv.w));
    float acc = 0.0f;
    acc = fma(cb.x, cb.x, acc); acc = fma(cb.y, cb.y, acc); acc = fma(cb.z, cb.z, acc); acc = fma(cb.w, cb.w, acc);
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = metal::precise::rsqrt(simd_sum(lane < 18u ? part[lane] : 0.0f) / float(DD) + eps[0]);
    bfloat4 wv = as_type<bfloat4>(((const device uint2*)(nwp + c))[0]);
    bfloat4 ov = bfloat4(
        bfloat16_t(glue_rms_apply<float>(cb.x, inv, wv.x)),
        bfloat16_t(glue_rms_apply<float>(cb.y, inv, wv.y)),
        bfloat16_t(glue_rms_apply<float>(cb.z, inv, wv.z)),
        bfloat16_t(glue_rms_apply<float>(cb.w, inv, wv.w)));
    ((device uint2*)dst)[0] = as_type<uint2>(ov);
"""


# ---------------------------------------------------------------------------------------------------- Splash tiled MessageGate
_SPLASH_MG_HEADER = (
    "#pragma clang fp reassociate(off)\n"
    "#include <metal_simdgroup_matrix>\n"
    + GLUE_HEADER
    + r"""
namespace mg_sg {
struct Lane { ushort fm; ushort fn; };
inline Lane lane_map(uint lane) {
  const uint qid = lane >> 2;
  return { ushort((qid & 4) | ((lane >> 1) & 3)), ushort(((qid & 2) << 1) | ((lane & 1) << 1)) };
}
template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2>& te(thread metal::simdgroup_matrix<T, 8, 8>& m) {
  return reinterpret_cast<thread vec<T, 2>&>(m.thread_elements());
}
__attribute__((always_inline)) inline void mma_acc(thread float2& c, float2 a, float2 b) {
  metal::simdgroup_matrix<float, 8, 8> A, B, C, D;
  te(A) = a; te(B) = b; te(C) = c;
  metal::simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}
}  // namespace mg_sg

constant constexpr uint kMgHidden = 2304;
constant constexpr uint kMgRank = 64;
constant constexpr uint kMgTileRows = 32;
constant constexpr uint kMgTileSg = 8;
constant constexpr uint kMgTileThreads = 256;
constant constexpr uint kMgTileStep = 64;
constant constexpr uint kMgTileStride = 72;
constant constexpr uint kMgFrags = 4;

struct MgEight {
  float4 low, high;
  float operator[](uint i) const { return i < 4 ? low[i] : high[i - 4]; }
};
inline MgEight mg_eight_bf16(const device bfloat16_t* row, uint k) {
  const device bfloat4* q = reinterpret_cast<const device bfloat4*>(row + k);
  return {float4(q[0]), float4(q[1])};
}
inline MgEight mg_eight_f32(const device float* row, uint k) {
  const device float4* q = reinterpret_cast<const device float4*>(row + k);
  return {q[0], q[1]};
}
struct MgShared {
  float scales[2 * kMgTileRows];
  union {
    bfloat16_t stages[2][kMgTileRows * kMgTileStride];
    struct {
      bfloat16_t lows[kMgTileRows * kMgTileStride];
      float4 partials[kMgTileSg * 32];
    } gated;
  };
  float inverses[kMgTileRows];
};
struct MgRun { float4 low, high; };
struct MgChains { float2 even0, odd0, even1, odd1, even2, odd2, even3, odd3; };

inline const device float* mg_row(const device float* x, const device float* msg, uint x_stride, uint x_col, uint m_col,
                                  uint first, uint count, uint r, bool own) {
  const ulong row = first + min(r, count - 1);
  return own ? (x + row * x_stride + x_col) : (msg + row * (4u * kMgHidden) + m_col);
}

inline void mg_read(threadgroup const bfloat16_t* row, thread MgEight& low, thread MgEight& high) {
  threadgroup const bfloat4* a = reinterpret_cast<threadgroup const bfloat4*>(row);
  low = {float4(a[0]), float4(a[1])};
  high = {float4(a[2]), float4(a[3])};
}

inline void mg_mma(threadgroup const bfloat16_t* row, MgEight b0, MgEight b1, thread float2& even, thread float2& odd) {
  MgEight low, high;
  mg_read(row, low, high);
  for (uint j = 0; j < 8; j += 2) {
    mg_sg::mma_acc(even, float2(low[j], high[j]), float2(b0[j], b1[j]));
    mg_sg::mma_acc(odd, float2(low[j + 1], high[j + 1]), float2(b0[j + 1], b1[j + 1]));
  }
}

METAL_FUNC void mg_tile_run(
    const device float* x, const device float* msg,
    const device bfloat16_t* down, const device float* forget, const device float* gain,
    const device bfloat16_t* ln_w, device float* gated, device bfloat16_t* out,
    uint x_stride, uint x_col, uint m_col, uint first, uint count, float eps,
    uint tid, uint lane, uint sg, threadgroup MgShared& shared) {
  // Phase 1: row & message inv-RMS
  const uint quads = kMgHidden / 4;
  for (uint i = sg; i < 2 * kMgTileRows; i += kMgTileSg) {
    const device float4* row = reinterpret_cast<const device float4*>(
        mg_row(x, msg, x_stride, x_col, m_col, first, count, i / 2, (i & 1) == 0));
    float squares = 0.0f;
    if (i / 2 < count) {
      for (uint d = lane; d < quads; d += 32) {
        const float4 v = row[d];
        squares += v.x * v.x;
        squares += v.y * v.y;
        squares += v.z * v.z;
        squares += v.w * v.w;
      }
    }
    squares = metal::simd_sum(squares);
    if (lane == 0) shared.scales[i] = (i / 2 < count) ? metal::precise::rsqrt(squares / kMgHidden + 1e-6f) : 0.0f;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // Phase 2: down_proj (4608 -> 64) via double-buffered 32x64 stages
  const mg_sg::Lane l = mg_sg::lane_map(lane);
  const uint steps = kMgHidden / kMgTileStep;  // 36
  const uint column = 8 * sg + l.fn;
  const device bfloat16_t* w0 = down + ulong(column) * (2 * kMgHidden);
  const device bfloat16_t* w1 = down + ulong(column + 1) * (2 * kMgHidden);

  auto load_run = [&](uint step) -> MgRun {
    const bool own = step < steps;
    const device float4* run = reinterpret_cast<const device float4*>(
        mg_row(x, msg, x_stride, x_col, m_col, first, count, tid / 8, own)
        + (own ? step : step - steps) * kMgTileStep + 8 * (tid & 7));
    return {run[0], run[1]};
  };
  auto stage_run = [&](MgRun run, bool own, threadgroup bfloat16_t* st) {
    const uint r = tid / 8;
    const float sc = shared.scales[2 * r + (own ? 0u : 1u)];
    threadgroup bfloat4* dst = reinterpret_cast<threadgroup bfloat4*>(st + r * kMgTileStride + 8 * (tid & 7));
    dst[0] = bfloat4(run.low * sc);
    dst[1] = bfloat4(run.high * sc);
  };

  MgChains own{}, other{};
  stage_run(load_run(0), true, shared.stages[0]);
  MgEight next0 = mg_eight_bf16(w0, 8 * l.fm);
  MgEight next1 = mg_eight_bf16(w1, 8 * l.fm);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  const auto run_step = [&](uint step, thread MgChains& chains) __attribute__((always_inline)) {
    const bool has_next = (step + 1) < (2 * steps);
    const MgRun ahead = has_next ? load_run(step + 1) : MgRun{};
    const MgEight b0 = next0, b1 = next1;
    if (has_next) {
      next0 = mg_eight_bf16(w0, (step + 1) * kMgTileStep + 8 * l.fm);
      next1 = mg_eight_bf16(w1, (step + 1) * kMgTileStep + 8 * l.fm);
    }
    threadgroup const bfloat16_t* row = shared.stages[step & 1] + l.fm * kMgTileStride + 8 * l.fn;
    mg_mma(row, b0, b1, chains.even0, chains.odd0);
    mg_mma(row + 8 * kMgTileStride, b0, b1, chains.even1, chains.odd1);
    mg_mma(row + 16 * kMgTileStride, b0, b1, chains.even2, chains.odd2);
    mg_mma(row + 24 * kMgTileStride, b0, b1, chains.even3, chains.odd3);
    if (has_next) stage_run(ahead, (step + 1) < steps, shared.stages[(step + 1) & 1]);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  };
  for (uint step = 0; step < steps; ++step) run_step(step, own);
  for (uint step = steps; step < 2 * steps; ++step) run_step(step, other);

  auto store_low = [&](uint f, float2 sum) {
    const uint r = 8 * f + l.fm;
    shared.gated.lows[r * kMgTileStride + column] = bfloat16_t(sum.x);
    shared.gated.lows[r * kMgTileStride + column + 1] = bfloat16_t(sum.y);
  };
  store_low(0, (own.even0 + own.odd0) + (other.even0 + other.odd0));
  store_low(1, (own.even1 + own.odd1) + (other.even1 + other.odd1));
  store_low(2, (own.even2 + own.odd2) + (other.even2 + other.odd2));
  store_low(3, (own.even3 + own.odd3) + (other.even3 + other.odd3));
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // Phase 3: forget + gain projections + gated combine
  const uint blocks = kMgHidden / kMgTileStep;  // 36
  const uint k8 = 8 * l.fm;
  float squares[kMgFrags] = {0.0f, 0.0f, 0.0f, 0.0f};
  auto load_fg = [&](uint block, thread MgEight (&w)[4]) {
    const uint col = block * kMgTileStep + 8 * sg + l.fn;
    w[0] = mg_eight_f32(forget + ulong(col) * kMgRank, k8);
    w[1] = mg_eight_f32(forget + ulong(col + 1) * kMgRank, k8);
    w[2] = mg_eight_f32(gain + ulong(col) * kMgRank, k8);
    w[3] = mg_eight_f32(gain + ulong(col + 1) * kMgRank, k8);
  };
  MgEight next_fg[4];
  load_fg(0, next_fg);
  for (uint block = 0; block < blocks; ++block) {
    const uint col = block * kMgTileStep + 8 * sg + l.fn;
    const MgEight f0 = next_fg[0], f1 = next_fg[1], e0 = next_fg[2], e1 = next_fg[3];
    if (block + 1 < blocks) load_fg(block + 1, next_fg);
    float2 xs[kMgFrags], ms[kMgFrags];
    for (uint f = 0; f < kMgFrags; ++f) {
      const ulong row = first + min(8 * f + l.fm, count - 1);
      xs[f] = *reinterpret_cast<const device float2*>(x + row * x_stride + x_col + col);
      ms[f] = *reinterpret_cast<const device float2*>(msg + row * (4u * kMgHidden) + m_col + col);
    }
    float2 forgets[kMgFrags], gains[kMgFrags];
    for (uint f = 0; f < kMgFrags; ++f) {
      MgEight low, high;
      mg_read(shared.gated.lows + (8 * f + l.fm) * kMgTileStride + 8 * l.fn, low, high);
      forgets[f] = float2(0.0f);
      gains[f] = float2(0.0f);
      for (uint j = 0; j < 8; ++j) {
        mg_sg::mma_acc(forgets[f], float2(low[j], high[j]), float2(f0[j], f1[j]));
        mg_sg::mma_acc(gains[f], float2(low[j], high[j]), float2(e0[j], e1[j]));
      }
    }
    for (uint f = 0; f < kMgFrags; ++f) {
      const uint r = 8 * f + l.fm;
      if (r < count) {
        const ulong row = first + r;
        const float2 xv = xs[f], mv = ms[f];
        float2 v;
        v.x = (1.0f - metal::precise::tanh(forgets[f].x)) * xv.x + metal::precise::tanh(gains[f].x) * mv.x;
        v.y = (1.0f - metal::precise::tanh(forgets[f].y)) * xv.y + metal::precise::tanh(gains[f].y) * mv.y;
        *reinterpret_cast<device float2*>(gated + row * kMgHidden + col) = v;
        squares[f] += v.x * v.x;
        squares[f] += v.y * v.y;
      }
    }
  }
  shared.gated.partials[sg * 32 + lane] = float4(squares[0], squares[1], squares[2], squares[3]);

  // Phase 4: RMSNorm into bf16
  threadgroup_barrier(metal::mem_flags::mem_device | metal::mem_flags::mem_threadgroup);
  if (tid < count) {
    const uint f = tid / 8, fm = tid & 7;
    float total = 0.0f;
    for (uint s = 0; s < kMgTileSg; ++s) {
      for (uint ln = 0; ln < 32; ++ln) {
        if (mg_sg::lane_map(ln).fm == fm) total += shared.gated.partials[s * 32 + ln][f];
      }
    }
    shared.inverses[tid] = metal::precise::rsqrt(total / kMgHidden + eps);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const device float4* rows = reinterpret_cast<const device float4*>(gated + ulong(first) * kMgHidden);
  device bfloat4* dst = reinterpret_cast<device bfloat4*>(out + ulong(first) * kMgHidden);
  for (uint q = tid; q < count * quads; q += kMgTileThreads) {
    const uint r = q / quads, e = 4 * (q % quads);
    const bfloat4 wbf = *reinterpret_cast<const device bfloat4*>(ln_w + e);
    const float4 sc = rows[q] * shared.inverses[r];
    dst[q] = bfloat4(float4(wbf) * sc);
  }
}
"""
)

_SPLASH_SINGLE_SRC = r"""
    threadgroup MgShared shared;
    const uint first = threadgroup_position_in_grid.x * kMgTileRows;
    if (first >= NN) return;
    const uint count = min(kMgTileRows, NN - first);
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    mg_tile_run(
        h, msg, dn, wf, wg, nw, gated, out,
        kMgHidden, 0u, SLOT * kMgHidden, first, count, eps[0],
        tid, lane, sg, shared);
"""

_SPLASH_QKV_SRC = r"""
    threadgroup MgShared shared;
    const uint first = threadgroup_position_in_grid.x * kMgTileRows;
    const uint g = threadgroup_position_in_grid.y;
    if (first >= NN || g >= 3) return;
    const uint count = min(kMgTileRows, NN - first);
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const device bfloat16_t* dn = (g == 0 ? dn0 : (g == 1 ? dn1 : dn2));
    const device float* wf = (g == 0 ? wf0 : (g == 1 ? wf1 : wf2));
    const device float* wg = (g == 0 ? wg0 : (g == 1 ? wg1 : wg2));
    const device bfloat16_t* nw = SHARED_LN ? nw0 : (g == 0 ? nw0 : (g == 1 ? nw1 : nw2));
    device bfloat16_t* dst = (g == 0 ? o0 : (g == 1 ? o1 : o2));
    device float* gt = gated + ulong(g) * NN * kMgHidden;
    mg_tile_run(
        h, msg, dn, wf, wg, nw, gt, dst,
        SHARED_H ? kMgHidden : 3u * kMgHidden,
        SHARED_H ? 0u : g * kMgHidden,
        g * kMgHidden,
        first, count, eps[0],
        tid, lane, sg, shared);
"""


def _k_splash(name, ins, outs, src):
    k = _kernels.get(name)
    if k is None:
        k = _kernels[name] = mx.fast.metal_kernel(
            name=name,
            input_names=list(ins),
            output_names=list(outs),
            source=src,
            header=_SPLASH_MG_HEADER,
            ensure_row_contiguous=True,
        )
    return k


def _splash_ok(gates, N: int) -> bool:
    return (
        _SPLASH_MSGGATE
        and len(gates) in (1, 3)
        and N >= 32
        and all(g.down_proj.weight.shape == (64, 4608) for g in gates)
    )


def _gate_logits(gate_mod, x_bf16):
    st = gate_mod.down_proj(x_bf16).astype(mx.float32)
    return gate_mod.forget_gate_proj(st), gate_mod.message_gate_proj(st)


_MGIN_DOWN_QKV_SRC = r"""
    GL_NOFMA
    const uint g = threadgroup_position_in_grid.y;
    const device bfloat16_t* w = (g == 0 ? w0 : (g == 1 ? w1 : w2));
    device float* out_p = (g == 0 ? o0 : (g == 1 ? o1 : o2));
    auto h_row = [&](uint mr) { return (SHARED_H ? size_t(mr) : size_t(mr) * 3 + g) * DD; };
    auto msg_row = [&](uint mr) { return (size_t(mr) * 4 + g) * DD; };
    constexpr int BM = 32, BN = 64, BK = 32, LDX = 40, TG = 128, TM = 2, TN = 4;
    const uint tgm = threadgroup_position_in_grid.x;
    const uint m0 = tgm * BM;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg_m = (sg / 2) * 16, sg_n = (sg % 2) * 32;
    const ushort fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const ushort fn = ((lane >> 2) & 2) * 2 | ((lane & 1) * 2);

    threadgroup float invh_s[BM], invm_s[BM];
    for (uint r = sg; r < BM; r += 4) {
        uint mr = m0 + r;
        if (mr < uint(NN)) {
            size_t hb = h_row(mr);
            size_t mb = msg_row(mr);
            float bh = 0.0f, bm_ = 0.0f;
            for (uint s = 0; s < 18; ++s) {
                uint c = (s * 32 + lane) * 4;
                float4 hv = ((const device float4*)(h + hb + c))[0];
                float4 mv = ((const device float4*)(msg + mb + c))[0];
                float sh = 0.0f, sm = 0.0f;
                sh = fma(hv.x, hv.x, sh); sh = fma(hv.y, hv.y, sh); sh = fma(hv.z, hv.z, sh); sh = fma(hv.w, hv.w, sh);
                sm = fma(mv.x, mv.x, sm); sm = fma(mv.y, mv.y, sm); sm = fma(mv.z, mv.z, sm); sm = fma(mv.w, mv.w, sm);
                sh = simd_sum(sh); sm = simd_sum(sm);
                if (lane == s) { bh = sh; bm_ = sm; }
            }
            float ah = simd_sum(bh), am = simd_sum(bm_);
            if (lane == 0) {
                invh_s[r] = metal::precise::rsqrt(ah / float(DD) + 1e-6f);
                invm_s[r] = metal::precise::rsqrt(am / float(DD) + 1e-6f);
            }
        } else if (lane == 0) {
            invh_s[r] = 0.0f; invm_s[r] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup bfloat16_t Xs[BM * LDX];
    threadgroup bfloat16_t Ws[BN * LDX];

    const uint x_mr = tid / 4, x_mk = (tid % 4) * 8;
    const bool x_ok = (m0 + x_mr) < uint(NN);
    const float my_invh = invh_s[x_mr], my_invm = invm_s[x_mr];
    const device float* hx_ptr = h + h_row(m0 + x_mr) + x_mk;
    const device float* mx_ptr = msg + msg_row(m0 + x_mr) + x_mk;

    const uint w_nr = tid / 2, w_koff = (tid % 2) * 16;
    const device bfloat16_t* w_ptr = w + (size_t)w_nr * (2 * DD) + w_koff;

    simdgroup_float8x8 Ctot[TM][TN];
    for (int tm = 0; tm < TM; ++tm)
        for (int tn = 0; tn < TN; ++tn)
            Ctot[tm][tn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const int part_k = (2 * DD) / SPK;
    threadgroup uint2* xd2 = (threadgroup uint2*)(Xs + x_mr * LDX + x_mk);
    threadgroup uint4* wd4 = (threadgroup uint4*)(Ws + w_nr * LDX + w_koff);
    const threadgroup bfloat16_t* as_base = Xs + (sg_m + fm) * LDX + fn;
    const threadgroup bfloat16_t* bs_base = Ws + (sg_n + fn) * LDX + fm;

    for (int p = 0; p < SPK; ++p) {
        simdgroup_float8x8 C[TM][TN];
        for (int tm = 0; tm < TM; ++tm)
            for (int tn = 0; tn < TN; ++tn)
                C[tm][tn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        const int k0 = p * part_k, k1 = k0 + part_k;
        for (int k = k0; k < k1; k += BK) {
            if (x_ok) {
                const bool is_h = (k < DD);
                const device float4* xp4 = (const device float4*)(is_h ? (hx_ptr + k) : (mx_ptr + (k - DD)));
                const float sc = is_h ? my_invh : my_invm;
                xd2[0] = as_type<uint2>(bfloat4(xp4[0] * sc));
                xd2[1] = as_type<uint2>(bfloat4(xp4[1] * sc));
            } else {
                xd2[0] = uint2(0); xd2[1] = uint2(0);
            }
            const device uint4* wp4 = (const device uint4*)(w_ptr + k);
            wd4[0] = wp4[0]; wd4[1] = wp4[1];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma clang loop unroll(full)
            for (int ks = 0; ks < BK; ks += 8) {
                simdgroup_bfloat8x8 A[TM], B[TN];
                simdgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (int tm = 0; tm < TM; ++tm) {
                    const bfloat2 av = *((const threadgroup bfloat2*)(as_base + (tm * 8) * LDX + ks));
                    A[tm].thread_elements()[0] = av.x; A[tm].thread_elements()[1] = av.y;
                }
                simdgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (int tn = 0; tn < TN; ++tn) {
                    const threadgroup bfloat16_t* bp = bs_base + (tn * 8) * LDX + ks;
                    B[tn].thread_elements()[0] = bp[0]; B[tn].thread_elements()[1] = bp[LDX];
                }
                simdgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (int tm = 0; tm < TM; ++tm)
                    for (int tn = 0; tn < TN; ++tn)
                        simdgroup_multiply_accumulate(C[tm][tn], A[tm], B[tn], C[tm][tn]);
            }
            if (p + 1 < SPK || k + BK < k1) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (int tm = 0; tm < TM; ++tm)
            for (int tn = 0; tn < TN; ++tn) {
                Ctot[tm][tn].thread_elements()[0] += C[tm][tn].thread_elements()[0];
                Ctot[tm][tn].thread_elements()[1] += C[tm][tn].thread_elements()[1];
            }
    }

    for (int tm = 0; tm < TM; ++tm) {
        const uint mr_o = m0 + sg_m + uint(tm) * 8u + fm;
        if (mr_o < uint(NN)) {
            device float* dst = out_p + (size_t)mr_o * 64;
            for (int tn = 0; tn < TN; ++tn) {
                const uint nc_o = sg_n + uint(tn) * 8u + fn;
                dst[nc_o] = float(bfloat16_t(Ctot[tm][tn].thread_elements()[0]));
                dst[nc_o + 1] = float(bfloat16_t(Ctot[tm][tn].thread_elements()[1]));
            }
        }
    }
"""


_MGIN_DOWN_S_SRC = _MGIN_DOWN_QKV_SRC.replace(
    "    const uint g = threadgroup_position_in_grid.y;\n"
    "    const device bfloat16_t* w = (g == 0 ? w0 : (g == 1 ? w1 : w2));\n"
    "    device float* out_p = (g == 0 ? o0 : (g == 1 ? o1 : o2));\n"
    "    auto h_row = [&](uint mr) { return (SHARED_H ? size_t(mr) : size_t(mr) * 3 + g) * DD; };\n"
    "    auto msg_row = [&](uint mr) { return (size_t(mr) * 4 + g) * DD; };",
    "    const device bfloat16_t* w = w0;\n"
    "    device float* out_p = o0;\n"
    "    auto h_row = [&](uint mr) { return size_t(mr) * DD; };\n"
    "    auto msg_row = [&](uint mr) { return (size_t(mr) * 4 + SLOT) * DD; };",
)
assert _MGIN_DOWN_S_SRC != _MGIN_DOWN_QKV_SRC


def _fg_mix_src(is_qkv: bool, w_bf16: bool = False) -> str:
    wt = "bfloat16_t" if w_bf16 else "float"
    if is_qkv:
        setup = (
            "    const uint tgc = SWAP_GRID ? threadgroup_position_in_grid.y : threadgroup_position_in_grid.x;\n"
            "    const uint c0 = (tgc % (DD / 64)) * 64;\n"
            "    const uint g = tgc / (DD / 64);\n"
            "    const device float* stp = (g == 0 ? st0 : (g == 1 ? st1 : st2));\n"
            f"    const device {wt}* wfp_base = (g == 0 ? wf0 : (g == 1 ? wf1 : wf2));\n"
            f"    const device {wt}* wmp_base = (g == 0 ? wm0 : (g == 1 ? wm1 : wm2));\n"
            "    device float* out_p = (g == 0 ? o0 : (g == 1 ? o1 : o2));\n"
            "    auto hoff_fn = [&](uint mr) { return (SHARED_H ? size_t(mr) : (size_t(mr) * 3 + g)) * DD; };\n"
            "    auto moff_fn = [&](uint mr) { return (size_t(mr) * 4 + g) * DD; };\n"
        )
    else:
        setup = (
            "    const uint c0 = (SWAP_GRID ? threadgroup_position_in_grid.y : threadgroup_position_in_grid.x) * 64;\n"
            "    const device float* stp = st;\n"
            f"    const device {wt}* wfp_base = wf;\n"
            f"    const device {wt}* wmp_base = wm;\n"
            "    device float* out_p = mixed;\n"
            "    auto hoff_fn = [&](uint mr) { return size_t(mr) * DD; };\n"
            "    auto moff_fn = [&](uint mr) { return (size_t(mr) * 4 + SLOT) * DD; };\n"
        )
    return (
        "    GL_NOFMA\n"
        "    constexpr int BM = 32, TM = 4;\n"
        "    const uint m0 = (SWAP_GRID ? threadgroup_position_in_grid.x : threadgroup_position_in_grid.y) * BM;\n"
        + setup
        + f"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint col = c0 + sg * 8 + fn;

    threadgroup float Ss[BM * 72];
    const uint tid = thread_position_in_threadgroup.x;
    for (uint i = tid; i < BM * 32; i += 256) {{
        uint r = i >> 5;
        uint k2 = (i & 31) * 2;
        uint rr = min(m0 + r, uint(NN - 1));
        ((threadgroup float2*)(Ss + r * 72 + k2))[0] = ((const device float2*)(stp + (size_t)rr * 64 + k2))[0];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 cf[TM], cm[TM];
    #pragma clang loop unroll(full)
    for (int tm = 0; tm < TM; ++tm) {{
        cf[tm] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        cm[tm] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }}
    const device {wt}* wfp0 = wfp_base + (size_t)col * 64;
    const device {wt}* wfp1 = wfp_base + (size_t)(col + 1) * 64;
    const device {wt}* wmp0 = wmp_base + (size_t)col * 64;
    const device {wt}* wmp1 = wmp_base + (size_t)(col + 1) * 64;
    #pragma clang loop unroll(full)
    for (int k = 0; k < 64; k += 8) {{
        simdgroup_float8x8 Bf, Bm;
        Bf.thread_elements()[0] = float(wfp0[k + fm]); Bf.thread_elements()[1] = float(wfp1[k + fm]);
        Bm.thread_elements()[0] = float(wmp0[k + fm]); Bm.thread_elements()[1] = float(wmp1[k + fm]);
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < TM; ++tm) {{
            simdgroup_float8x8 A;
            const float2 av = *((const threadgroup float2*)(Ss + (tm * 8 + fm) * 72 + k + fn));
            A.thread_elements()[0] = av.x; A.thread_elements()[1] = av.y;
            simdgroup_multiply_accumulate(cf[tm], A, Bf, cf[tm]);
            simdgroup_multiply_accumulate(cm[tm], A, Bm, cm[tm]);
        }}
    }}
    #pragma clang loop unroll(full)
    for (int tm = 0; tm < TM; ++tm) {{
        const uint mr = m0 + uint(tm) * 8u + fm;
        if (mr < uint(NN)) {{
            size_t off = (size_t)mr * DD + col;
            size_t hoff = hoff_fn(mr) + col;
            size_t moff = moff_fn(mr) + col;
            float2 hv = ((const device float2*)(h + hoff))[0];
            float2 mv = ((const device float2*)(msg + moff))[0];
            float v0 = glue_msg_combine<0>(cf[tm].thread_elements()[0], cm[tm].thread_elements()[0], hv.x, mv.x);
            float v1 = glue_msg_combine<0>(cf[tm].thread_elements()[1], cm[tm].thread_elements()[1], hv.y, mv.y);
            ((device float2*)(out_p + off))[0] = float2(v0, v1);
        }}
    }}
"""
    )


_FG_MIX_SINGLE_SRC = _fg_mix_src(False, False)
_FG_MIX_QKV_SRC = _fg_mix_src(True, False)
_FG_MIX_SINGLE_BF16_SRC = _fg_mix_src(False, True)
_FG_MIX_QKV_BF16_SRC = _fg_mix_src(True, True)


_PF_FG_SWAP = os.environ.get("LOOPLING_PF_FG_SWAP", "1") not in ("0", "")


def msg_gate_norm_single(gate_mod, h: mx.array, msg: mx.array, slot: int, ln) -> mx.array:
    """Single MessageGate + downstream RMSNorm -> (B, L, D) bf16."""
    B, L, D = h.shape
    N = B * L
    if _splash_ok((gate_mod,), N):
        nt = (N + 31) // 32
        k = _k_splash(
            "pf_splash_mg_s",
            ("h", "msg", "dn", "wf", "wg", "nw", "eps"),
            ("out", "gated"),
            _SPLASH_SINGLE_SRC,
        )
        out, _ = k(
            inputs=[
                h, msg, gate_mod.down_proj.weight, gate_mod.forget_gate_proj.weight,
                gate_mod.message_gate_proj.weight, ln.weight, mx.array([ln.eps], mx.float32),
            ],
            template=[("NN", N), ("SLOT", slot)],
            grid=(nt * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D), (N, D)],
            output_dtypes=[mx.bfloat16, mx.float32],
        )
        return out
    if _PF_FG_MIX and N >= 256:
        spk = (
            _MGIN_DOWN_SPK.get(N)
            if (_PF_MGIN_DOWN and N >= 8192 and gate_mod.down_proj.weight.shape == (64, 2 * D))
            else None
        )
        if spk is not None:
            k_md = _k_mma("pf_mgin_down_s", ("h", "msg", "w0"), ("o0",), _MGIN_DOWN_S_SRC)
            st = k_md(
                inputs=[h, msg, gate_mod.down_proj.weight],
                template=[("DD", D), ("NN", N), ("SPK", spk), ("SLOT", slot)],
                grid=(128 * ((N + 31) // 32), 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(B, L, 64)],
                output_dtypes=[mx.float32],
            )[0]
        else:
            k_in = _k("pf_mgin_s", ("h", "msg"), ("out",), _MGIN_SINGLE_SRC)
            g_in = k_in(
                inputs=[h, msg],
                template=[("DD", D), ("TG", _TG_RMS), ("MSTRIDE", 4), ("SLOT", slot)],
                grid=(N * _TG_RMS, 1, 1),
                threadgroup=(_TG_RMS, 1, 1),
                output_shapes=[(B, L, 2 * D)],
                output_dtypes=[mx.bfloat16],
            )[0]
            st = gate_mod.down_proj(g_in).astype(mx.float32)
        wf = _maybe_bf16_w(gate_mod.forget_gate_proj.weight)
        wm = _maybe_bf16_w(gate_mod.message_gate_proj.weight)
        use_bf16 = wf is not None and wm is not None
        swap_fg = int(_PF_FG_SWAP and N >= 4096)
        nt_m = (N + 31) // 32
        k_fg = _k_mma(
            "pf_fg_mix_s_bf16" if use_bf16 else "pf_fg_mix_s",
            ("st", "wf", "wm", "h", "msg"),
            ("mixed",),
            _FG_MIX_SINGLE_BF16_SRC if use_bf16 else _FG_MIX_SINGLE_SRC,
        )
        mixed = k_fg(
            inputs=[
                st,
                wf if use_bf16 else gate_mod.forget_gate_proj.weight,
                wm if use_bf16 else gate_mod.message_gate_proj.weight,
                h,
                msg,
            ],
            template=[("DD", D), ("NN", N), ("SLOT", slot), ("SWAP_GRID", swap_fg)],
            grid=(nt_m * 256, D // 64, 1) if swap_fg else ((D // 64) * 256, nt_m, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D)],
            output_dtypes=[mx.float32],
        )[0]
        return rms_norm_f32_bf16(mixed, ln)
    k_in = _k("pf_mgin_s", ("h", "msg"), ("out",), _MGIN_SINGLE_SRC)
    g_in = k_in(
        inputs=[h, msg],
        template=[("DD", D), ("TG", _TG_RMS), ("MSTRIDE", 4), ("SLOT", slot)],
        grid=(N * _TG_RMS, 1, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, 2 * D)],
        output_dtypes=[mx.bfloat16],
    )[0]
    fl, ml = _gate_logits(gate_mod, g_in)
    k_mn = _k("pf_mn_s", ("fl", "ml", "h", "msg", "nw", "eps"), ("out",), _MIXNORM_SINGLE_SRC)
    return k_mn(
        inputs=[fl, ml, h, msg, ln.weight, _get_eps(ln.eps)],
        template=[("DD", D), ("TG", _TG_RMS), ("SLOT", slot)],
        grid=(N * _TG_RMS, 1, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, D)],
        output_dtypes=[mx.bfloat16],
    )[0]


def msg_gate_norm_qkv(gates, h_or_ai: mx.array, msg: mx.array, lns, *, shared_h: bool, shared_ln: bool):
    """Batched Q/K/V MessageGate + input RMSNorm -> (q_in, k_in, v_in) bf16 in 2 fused launches + projections."""
    B, L = msg.shape[0], msg.shape[1]
    D = msg.shape[-1]
    N = B * L
    if _splash_ok(gates, N):
        nt = (N + 31) // 32
        k = _k_splash(
            "pf_splash_mg_qkv",
            ("h", "msg", "dn0", "dn1", "dn2", "wf0", "wf1", "wf2", "wg0", "wg1", "wg2", "nw0", "nw1", "nw2", "eps"),
            ("o0", "o1", "o2", "gated"),
            _SPLASH_QKV_SRC,
        )
        o0, o1, o2, _ = k(
            inputs=[
                h_or_ai, msg,
                gates[0].down_proj.weight, gates[1].down_proj.weight, gates[2].down_proj.weight,
                gates[0].forget_gate_proj.weight, gates[1].forget_gate_proj.weight, gates[2].forget_gate_proj.weight,
                gates[0].message_gate_proj.weight, gates[1].message_gate_proj.weight, gates[2].message_gate_proj.weight,
                lns[0].weight, lns[1].weight, lns[2].weight,
                mx.array([lns[0].eps], mx.float32),
            ],
            template=[("NN", N), ("SHARED_H", int(shared_h)), ("SHARED_LN", int(shared_ln))],
            grid=(nt * 256, 3, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D), (B, L, D), (B, L, D), (3, N, D)],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32],
        )
        return (o0, o1, o2)
    if _PF_FG_MIX and N >= 256:
        spk = (
            _MGIN_DOWN_SPK.get(N)
            if (_PF_MGIN_DOWN and all(g.down_proj.weight.shape == (64, 2 * D) for g in gates))
            else None
        )
        if spk is not None:
            k_md = _k_mma(
                "pf_mgin_down_qkv",
                ("h", "msg", "w0", "w1", "w2"),
                ("o0", "o1", "o2"),
                _MGIN_DOWN_QKV_SRC,
            )
            st0, st1, st2 = k_md(
                inputs=[h_or_ai, msg, gates[0].down_proj.weight, gates[1].down_proj.weight, gates[2].down_proj.weight],
                template=[("DD", D), ("NN", N), ("SPK", spk), ("SHARED_H", int(shared_h))],
                grid=(128 * ((N + 31) // 32), 3, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(B, L, 64)] * 3,
                output_dtypes=[mx.float32] * 3,
            )
        else:
            k_in = _k("pf_mgin_qkv", ("h", "msg"), ("o0", "o1", "o2"), _MGIN_QKV_SRC)
            in0, in1, in2 = k_in(
                inputs=[h_or_ai, msg],
                template=[("DD", D), ("TG", _TG_RMS), ("SHARED_H", int(shared_h))],
                grid=(N * _TG_RMS, 3, 1),
                threadgroup=(_TG_RMS, 1, 1),
                output_shapes=[(B, L, 2 * D)] * 3,
                output_dtypes=[mx.bfloat16] * 3,
            )
            st0 = gates[0].down_proj(in0).astype(mx.float32)
            st1 = gates[1].down_proj(in1).astype(mx.float32)
            st2 = gates[2].down_proj(in2).astype(mx.float32)
        wf_b = [_maybe_bf16_w(g.forget_gate_proj.weight) for g in gates]
        wm_b = [_maybe_bf16_w(g.message_gate_proj.weight) for g in gates]
        use_bf16 = all(x is not None for x in wf_b) and all(x is not None for x in wm_b)
        swap_fg = int(_PF_FG_SWAP and N >= 4096)
        nt_m = (N + 31) // 32
        k_fg = _k_mma(
            "pf_fg_mix_qkv_bf16" if use_bf16 else "pf_fg_mix_qkv",
            ("st0", "st1", "st2", "wf0", "wf1", "wf2", "wm0", "wm1", "wm2", "h", "msg"),
            ("o0", "o1", "o2"),
            _FG_MIX_QKV_BF16_SRC if use_bf16 else _FG_MIX_QKV_SRC,
        )
        mx0, mx1, mx2 = k_fg(
            inputs=[
                st0, st1, st2,
                *(wf_b if use_bf16 else [g.forget_gate_proj.weight for g in gates]),
                *(wm_b if use_bf16 else [g.message_gate_proj.weight for g in gates]),
                h_or_ai, msg,
            ],
            template=[("DD", D), ("NN", N), ("SHARED_H", int(shared_h)), ("SWAP_GRID", swap_fg)],
            grid=(nt_m * 256, 3 * (D // 64), 1) if swap_fg else (3 * (D // 64) * 256, nt_m, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D)] * 3,
            output_dtypes=[mx.float32] * 3,
        )
        return rms_norm_f32_bf16_qkv_sep(mx0, mx1, mx2, lns, shared_ln=shared_ln)
    k_in = _k("pf_mgin_qkv", ("h", "msg"), ("o0", "o1", "o2"), _MGIN_QKV_SRC)
    in0, in1, in2 = k_in(
        inputs=[h_or_ai, msg],
        template=[("DD", D), ("TG", _TG_RMS), ("SHARED_H", int(shared_h))],
        grid=(N * _TG_RMS, 3, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, 2 * D)] * 3,
        output_dtypes=[mx.bfloat16] * 3,
    )
    fl0, ml0 = _gate_logits(gates[0], in0)
    fl1, ml1 = _gate_logits(gates[1], in1)
    fl2, ml2 = _gate_logits(gates[2], in2)
    k_mn = _k(
        "pf_mn_qkv",
        ("fl0", "ml0", "fl1", "ml1", "fl2", "ml2", "h", "msg", "nw0", "nw1", "nw2", "eps"),
        ("o0", "o1", "o2"),
        _MIXNORM_QKV_SRC,
    )
    return k_mn(
        inputs=[
            fl0, ml0, fl1, ml1, fl2, ml2,
            h_or_ai, msg,
            lns[0].weight, lns[1].weight, lns[2].weight,
            _get_eps(lns[0].eps),
        ],
        template=[("DD", D), ("TG", _TG_RMS), ("SHARED_H", int(shared_h)), ("SHARED_LN", int(shared_ln))],
        grid=(N * _TG_RMS, 3, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, D)] * 3,
        output_dtypes=[mx.bfloat16] * 3,
    )
