"""Fused bit-exact Metal kernels for L > 1 prefill (LOOPLING_FAST_PREFILL=1 default)."""
import math
import os

import mlx.core as mx
import numpy as np

from mlx_lm.models.inkling2.rows.gather import mlx_header as _fpq_mlx_header
from mlx_lm.models.inkling2.rows.glue import GLUE_HEADER

_FAST = os.environ.get("LOOPLING_FAST_PREFILL", "1") not in ("0", "")
_PF_MOE_GMM = os.environ.get("LOOPLING_PF_MOE_GMM", "1") not in ("0", "")
_PF_MOE_BM_ENV = int(os.environ.get("LOOPLING_PF_MOE_BM", "0"))
_PF_MOE_CFG_ENV = os.environ.get("LOOPLING_PF_MOE_CFG", "")
_PF_MOE_CFG = tuple(int(x) for x in _PF_MOE_CFG_ENV.split(",")) if _PF_MOE_CFG_ENV else None
_PF_MOE_MIN_L = int(os.environ.get("LOOPLING_PF_MOE_MIN_L", "256"))
_PF_AFF8_QMM = os.environ.get("LOOPLING_PF_AFF8_QMM", "1") not in ("0", "")
_PF_AFF8_CFG_ENV = os.environ.get("LOOPLING_PF_AFF8_CFG", "64,32,32,2,1")
_PF_AFF8_CFG = tuple(int(x) for x in _PF_AFF8_CFG_ENV.split(","))
_SPLASH_MSGGATE = os.environ.get("LOOPLING_SPLASH_MSGGATE", "0") not in ("0", "", "false", "False")
_D = 2304
_TG_RMS = 576
_DUMMY_F32 = mx.zeros((256,), mx.float32)  # >= 512 B ensures Metal device address space
_DUMMY_BF16 = mx.zeros((256,), mx.bfloat16)
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

    float inv_scale_enc = 1.0f;
    if constexpr (HGS) {
        inv_scale_enc = gsc[e] / (F8E4M3_MAX * F4E2M1_MAX);
    }
    const float inv_sc_fold = inv_scale_enc * ((BITS == 4) ? 16384.0f : 256.0f);

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
                const float sc = float(dequantize_scale<bfloat16_t, GS>(sp[0])) * inv_sc_fold;
                const uint v0 = ((const device uint*)wp)[0];
                ushort4 u0 = ushort4(v0, v0 >> 4, v0 >> 8, v0 >> 12);
                half4 h0 = as_type<half4>((u0 & ushort4(7)) << 9);
                h0 = select(h0, -h0, (u0 & ushort4(8)) != ushort4(0));
                uint2 a2 = as_type<uint2>(bfloat4(float4(h0) * sc));
                ushort4 u1 = ushort4(v0 >> 16, v0 >> 20, v0 >> 24, v0 >> 28);
                half4 h1 = as_type<half4>((u1 & ushort4(7)) << 9);
                h1 = select(h1, -h1, (u1 & ushort4(8)) != ushort4(0));
                uint2 b2 = as_type<uint2>(bfloat4(float4(h1) * sc));
                wd4[0] = uint4(a2.x, a2.y, b2.x, b2.y);
            } else if constexpr (WPT == 16) {
                const float sc = float(dequantize_scale<bfloat16_t, GS>(sp[0])) * inv_sc_fold;
                const uint2 raw = ((const device uint2*)wp)[0];
                const uint vs[2] = {raw.x, raw.y};
                #pragma clang loop unroll(full)
                for (int gi = 0; gi < 2; ++gi) {
                    const uint vv = vs[gi];
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
            } else {
                const float sc0 = float(dequantize_scale<bfloat16_t, GS>(sp[0])) * inv_sc_fold;
                const float sc1 = float(dequantize_scale<bfloat16_t, GS>(sp[1])) * inv_sc_fold;
                const uint4 raw = ((const device uint4*)wp)[0];
                const uint vs[4] = {raw.x, raw.y, raw.z, raw.w};
                #pragma clang loop unroll(full)
                for (int gi = 0; gi < 4; ++gi) {
                    const float sc = (gi < 2) ? sc0 : sc1;
                    const uint vv = vs[gi];
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


def _k_moe_gmm():
    k = _kernels.get("pf_moe_gmm")
    if k is None:
        hdr = "#include <metal_simdgroup_matrix>\n" + _fpq_mlx_header() + "\n" + GLUE_HEADER
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
    if len(_PF_AFF8_CFG) == 5:
        bm, bn, bk, wm, wn = _PF_AFF8_CFG
    else:
        bm, bn, wm, wn = _PF_AFF8_CFG
        bk = 32
    if M < 1024 or N < 1024 or M % bm != 0 or K % 64 != 0 or N % bn != 0 or mod["scales"].dtype != mx.bfloat16:
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


def moe_gmm_ok(ex, hidden_states: mx.array, selected_experts: mx.array) -> bool:
    gu, dn = ex.gate_up_proj, ex.down_proj
    return (
        _PF_MOE_GMM
        and hidden_states.shape[0] * hidden_states.shape[1] >= _PF_MOE_MIN_L
        and hidden_states.dtype == mx.bfloat16
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
    elif _PF_MOE_BM_ENV in (16, 32, 64):
        bm, bn, bk, wm, wn = (_PF_MOE_BM_ENV, 32, 32, 1, 1)
    elif B * L < 1024:
        bm, bn, bk, wm, wn = (16, 32, 32, 1, 1)
    elif B * L < 3072:
        bm, bn, bk, wm, wn = (32, 64, 32, 1, 2)
    else:
        bm, bn, bk, wm, wn = (32, 32, 32, 1, 1)
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
        if (p >= 0) return float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0]));
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
        float4 acc = float4(0.0f);
        acc = fma(w0, s0, acc);
        acc = fma(w1, s1, acc);
        acc = fma(w2, s2, acc);
        acc = fma(w3, x3, acc);
        float4 conv = float4(bfloat4(acc));
        float4 y = float4(bfloat4((conv + x3) / {_S5_LIT}));
        float4 rv = ((const device float4*)(res + off))[0];
        ((device uint2*)(out + off))[0] = as_type<uint2>(bfloat4(rv + y));
        s0 = s1;
        s1 = s2;
        s2 = x3;
    }}
    if (t0 == 0) {{
        for (int r = 0; r < 3; r++) {{
            int p = int(LL) - 3 + r;
            float4 v = (p >= 0)
                ? float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0]))
                : (HAS_CS ? ((const device float4*)(cs + ((size_t)b * 3 + (3 + p)) * DD + c))[0] : float4(0.0f));
            ((device float4*)(ncs + ((size_t)b * 3 + r) * DD + c))[0] = v;
        }}
    }}
"""


def short_conv_residual(
    x: mx.array, conv_state: mx.array | None, weight: mx.array, residual: mx.array
) -> tuple[mx.array, mx.array]:
    """Fused post_mlp_convolution + (residual + conv.f32).bf16 -> (out bf16, new_cs (B,3,C) f32)."""
    B, L, C = x.shape
    has_cs = int(conv_state is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    tt = 16
    ntiles = (L + tt - 1) // tt
    k = _k("pf_short_conv_res4", ("x", "cs", "cw", "res"), ("out", "ncs"), _SC4_RES_SRC)
    return k(
        inputs=[x, cs_in, weight, residual],
        template=[("DD", C), ("LL", L), ("HAS_CS", has_cs), ("TT", tt)],
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
        if (p >= 0) return float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0]));
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
    float4 acc = float4(0.0f);
    acc = fma(w0, s0, acc);
    acc = fma(w1, s1, acc);
    acc = fma(w2, s2, acc);
    acc = fma(w3, x3, acc);
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
        for (int r = 0; r < 3; r++) {{
            int p = int(LL) - 3 + r;
            float4 v = (p >= 0)
                ? float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0]))
                : (HAS_CS ? ((const device float4*)(cs + ((size_t)b * 3 + (3 + p)) * DD + c))[0] : float4(0.0f));
            ((device float4*)(ncs + ((size_t)b * 3 + r) * DD + c))[0] = v;
        }}
    }}
"""


def short_conv_attn_post(
    x: mx.array,
    conv_state: mx.array | None,
    weight: mx.array,
    block_residual: mx.array | None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused post_attention_convolution + residual (if block_residual is not None) + unweighted_rms_norm -> (residual f32, normed_states f32, new_cs f32)."""
    B, L, C = x.shape
    assert C == _D
    has_cs = int(conv_state is not None)
    add_res = int(block_residual is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    br_in = block_residual if block_residual is not None else _DUMMY_BF16
    k = _k("pf_sc_attn_post", ("x", "cs", "cw", "bres"), ("res_out", "norm_out", "ncs"), _SC_ATTN_POST_SRC)
    return k(
        inputs=[x, cs_in, weight, br_in],
        template=[("DD", C), ("LL", L), ("HAS_CS", has_cs), ("ADD_RES", add_res)],
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
            if (p >= 0) return float4(as_type<bfloat4>(((const device uint2*)(k + ((size_t)b * LL + p) * DD + c))[0]));
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
            float4 acc = float4(0.0f);
            acc = fma(w0, s0, acc);
            acc = fma(w1, s1, acc);
            acc = fma(w2, s2, acc);
            acc = fma(w3, x3, acc);
            float4 conv = float4(bfloat4(acc));
            bfloat4 ybf = bfloat4((conv + x3) / {_S5_LIT});
            float sv[4] = {{float(ybf.x), float(ybf.y), float(ybf.z), float(ybf.w)}};
            glue_l2_clip_lane128(sv, thr_k);
            bfloat4 ov = bfloat4(bfloat16_t(sv[0]), bfloat16_t(sv[1]), bfloat16_t(sv[2]), bfloat16_t(sv[3]));
            ((device uint2*)(k_out + koff))[0] = as_type<uint2>(ov);
            s0 = s1; s1 = s2; s2 = x3;
        }}
        if (t0 == 0) {{
            for (int r = 0; r < 3; r++) {{
                int p = int(LL) - 3 + r;
                float4 fv = (p >= 0)
                    ? float4(as_type<bfloat4>(((const device uint2*)(k + ((size_t)b * LL + p) * DD + c))[0]))
                    : (HAS_KCS ? ((const device float4*)(kcs + ((size_t)b * 3 + (3 + p)) * DD + c))[0] : float4(0.0f));
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
            if (p >= 0) return float4(as_type<bfloat4>(((const device uint2*)(v + ((size_t)b * LL + p) * DD + c))[0]));
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
            float4 acc = float4(0.0f);
            acc = fma(w0, s0, acc);
            acc = fma(w1, s1, acc);
            acc = fma(w2, s2, acc);
            acc = fma(w3, x3, acc);
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
            s0 = s1; s1 = s2; s2 = x3;
        }}
        if (t0 == 0) {{
            for (int r = 0; r < 3; r++) {{
                int p = int(LL) - 3 + r;
                float4 fv = (p >= 0)
                    ? float4(as_type<bfloat4>(((const device uint2*)(v + ((size_t)b * LL + p) * DD + c))[0]))
                    : (HAS_VCS ? ((const device float4*)(vcs + ((size_t)b * 3 + (3 + p)) * DD + c))[0] : float4(0.0f));
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
):
    """1-launch SlidingAttention fusion: q_l2_clip + (k_sconv->k_l2_clip) + (v_sconv->v_rms_norm) [+ rel_l2_clip]."""
    B, L, _ = q.shape
    Dkv = num_kv_heads * 128
    has_kcs = int(k_cs is not None)
    has_vcs = int(v_cs is not None)
    has_rel = int(r is not None)
    kcs_in = k_cs if k_cs is not None else _DUMMY_F32
    vcs_in = v_cs if v_cs is not None else _DUMMY_F32
    r_in = r if has_rel else _DUMMY_BF16
    rs_in = r_scale if has_rel else _DUMMY_BF16
    pkey = ("qkv_cr4", thr_q, thr_k, eps, thr_r)
    params = _PARAMS_CACHE.get(pkey)
    if params is None:
        params = _PARAMS_CACHE[pkey] = mx.array([thr_q, thr_k, eps, thr_r], dtype=mx.float32)
    tt = 16
    ntiles = (L + tt - 1) // tt
    ntg_x = (num_heads + 2 * num_kv_heads + (1 if has_rel else 0) + 7) // 8
    k_fn = _k(
        "pf_swa_qkv_cc_r",
        ("q", "k", "v", "q_scale", "kcw", "vcw", "kcs", "vcs", "params", "r", "r_scale"),
        ("q_out", "k_out", "v_out", "nkcs", "nvcs", "r_out"),
        _SWA_QKV_CONV_CLIP_SRC,
    )
    qo, ko, vo, nkcs, nvcs, ro = k_fn(
        inputs=[q, k, v, q_scale, k_cw, v_cw, kcs_in, vcs_in, params, r_in, rs_in],
        template=[
            ("HQ", num_heads), ("HKV", num_kv_heads), ("LL", L),
            ("TT", tt), ("HAS_KCS", has_kcs), ("HAS_VCS", has_vcs), ("HAS_REL", has_rel),
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
    uint c = lid * 4;
    threadgroup float buf[N][32];
    threadgroup float nr_s[D];
    if (PRE_UNW_RMS) {
        GL_NOFMA
        bfloat4 hv = as_type<bfloat4>(((const device uint2*)(h + (size_t)row * D + c))[0]);
        float h0 = float(hv.x), h1 = float(hv.y), h2 = float(hv.z), h3 = float(hv.w);
        float racc = 0.0f;
        racc = fma(h0, h0, racc); racc = fma(h1, h1, racc);
        racc = fma(h2, h2, racc); racc = fma(h3, h3, racc);
        racc = simd_sum(racc);
        if (lane == 0) buf[0][sg] = racc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv0 = metal::precise::rsqrt(simd_sum(lane < 18u ? buf[0][lane] : 0.0f) / float(D) + 1e-6f);
        float4 p(h0 * inv0, h1 * inv0, h2 * inv0, h3 * inv0);
        ((threadgroup float4*)(nr_s + c))[0] = p;
        if (OUT_NR) ((device float4*)(nr_out + (size_t)row * D + c))[0] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float xc[8][N];
    if (lid < 288) {
        float acc[N];
        for (int i = 0; i < N; i++) acc[i] = 0.0f;
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
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 288) {
        float w[N];
        for (int i = 0; i < N; i++) {
            float t = 0.0f;
            for (int s = 0; s < 9; s++) t += buf[i][s];
            w[i] = t * scale[0];
        }
        float m = -INFINITY;
        for (int i = 0; i < N; i++) m = max(m, w[i]);
        float z = 0.0f;
        for (int i = 0; i < N; i++) { w[i] = metal::exp(w[i] - m); z += w[i]; }
        for (int i = 0; i < N; i++) w[i] /= z;
        int it_idx = 0;
        for (int d = lid; d < D; d += 288, it_idx++) {
            float o = 0.0f;
            for (int i = 0; i < N; i++) o = fma(w[i], xc[it_idx][i], o);
            nr_s[d] = o;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        GL_NOFMA
        float4 xv = ((threadgroup float4*)(nr_s + c))[0];
        float acc2 = 0.0f;
        acc2 = fma(xv.x, xv.x, acc2); acc2 = fma(xv.y, xv.y, acc2);
        acc2 = fma(xv.z, xv.z, acc2); acc2 = fma(xv.w, xv.w, acc2);
        acc2 = simd_sum(acc2);
        if (lane == 0) buf[0][sg] = acc2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv2 = metal::precise::rsqrt(simd_sum(lane < 18u ? buf[0][lane] : 0.0f) / float(D) + eps[0]);
        bfloat4 wv = as_type<bfloat4>(((const device uint2*)(nw + c))[0]);
        bfloat4 ov = bfloat4(
            bfloat16_t(glue_rms_apply<float>(xv.x, inv2, wv.x)),
            bfloat16_t(glue_rms_apply<float>(xv.y, inv2, wv.y)),
            bfloat16_t(glue_rms_apply<float>(xv.z, inv2, wv.z)),
            bfloat16_t(glue_rms_apply<float>(xv.w, inv2, wv.w)));
        ((device uint2*)(out + (size_t)row * D + c))[0] = as_type<uint2>(ov);
    }
"""


def unw_rms_depth_attn_rms_bf16(h: mx.array, blocks: mx.array, query: mx.array, ln, *, need_nr: bool):
    """Fused unweighted_rms_norm(h.f32) + attention_input(blocks, nr) + rms_norm_f32_bf16."""
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
        grid=(B * L * _TG_RMS, 1, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, D), (B, L, D) if need_nr else (1,)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    return out, (nr_out if need_nr else None)


def depth_attn_rms_bf16(blocks: mx.array, partial: mx.array, query: mx.array, ln) -> mx.array:
    """Fused mlp_input(blocks, partial) + rms_norm_f32_bf16 -> (B, L, D) bf16."""
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
        grid=(B * L * _TG_RMS, 1, 1),
        threadgroup=(_TG_RMS, 1, 1),
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


def unsort_combine(dn_sorted: mx.array, inv_order: mx.array, routing_weights: mx.array) -> mx.array:
    """Fuse _scatter_unsort + (es.f32 * wts[...,None]).sum(-2).astype(bf16) for K=8."""
    B, L, E = routing_weights.shape
    D = dn_sorted.shape[-1]
    assert E == 8 and dn_sorted.dtype == mx.bfloat16
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
        if (p >= 0) return float(x[(size_t)(b * LL + p) * DD + c]);
        int sidx = 3 + p;
        return HAS_CS ? float(bfloat16_t(cs[((size_t)b * 3 + sidx) * DD + c])) : 0.0f;
    }};
    float acc = 0.0f;
    acc = fma(float(cw[c * 4 + 0]), tap(3), acc);
    acc = fma(float(cw[c * 4 + 1]), tap(2), acc);
    acc = fma(float(cw[c * 4 + 2]), tap(1), acc);
    acc = fma(float(cw[c * 4 + 3]), x3, acc);
    float conv = float(bfloat16_t(acc));
    out[(size_t)tok * DD + c] = bfloat16_t((conv + x3) / {_S5_LIT});
    if (t == 0) {{
        for (int r = 0; r < 3; r++) {{
            int p = int(LL) - 3 + r;
            ncs[((size_t)b * 3 + r) * DD + c] = (p >= 0)
                ? float(x[(size_t)(b * LL + p) * DD + c])
                : (HAS_CS ? cs[((size_t)b * 3 + (3 + p)) * DD + c] : 0.0f);
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
        if (p >= 0) return float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0]));
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
        float4 acc = float4(0.0f);
        acc = fma(w0, s0, acc);
        acc = fma(w1, s1, acc);
        acc = fma(w2, s2, acc);
        acc = fma(w3, x3, acc);
        float4 conv = float4(bfloat4(acc));
        ((device uint2*)(out + ((size_t)b * LL + t) * DD + c))[0] = as_type<uint2>(bfloat4((conv + x3) / {_S5_LIT}));
        s0 = s1;
        s1 = s2;
        s2 = x3;
    }}
    if (t0 == 0) {{
        for (int r = 0; r < 3; r++) {{
            int p = int(LL) - 3 + r;
            float4 v = (p >= 0)
                ? float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * DD + c))[0]))
                : (HAS_CS ? ((const device float4*)(cs + ((size_t)b * 3 + (3 + p)) * DD + c))[0] : float4(0.0f));
            ((device float4*)(ncs + ((size_t)b * 3 + r) * DD + c))[0] = v;
        }}
    }}
"""


def short_conv(x: mx.array, conv_state: mx.array | None, weight: mx.array) -> tuple[mx.array, mx.array]:
    """K=4 causal depthwise conv + residual/sqrt(5) -> (out bf16, new_cs (B,3,C) f32)."""
    B, L, C = x.shape
    has_cs = int(conv_state is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    if C % 4 == 0 and L >= 16:
        tt = 16
        ntiles = (L + tt - 1) // tt
        k = _k("pf_short_conv4", ("x", "cs", "cw"), ("out", "ncs"), _SC4_SRC)
        return k(
            inputs=[x, cs_in, weight],
            template=[("DD", C), ("LL", L), ("HAS_CS", has_cs), ("TT", tt)],
            grid=(((C // 4 + 255) // 256) * 256, B * ntiles, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, C), (B, 3, C)],
            output_dtypes=[mx.bfloat16, mx.float32],
        )
    k = _k("pf_short_conv", ("x", "cs", "cw"), ("out", "ncs"), _SC_SRC)
    return k(
        inputs=[x, cs_in, weight],
        template=[("DD", C), ("LL", L), ("HAS_CS", has_cs)],
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
        if (p >= 0) return float(x[(size_t)(b * LL + p) * CC + c]);
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
        for (int r = 0; r < 3; r++) {{
            int p = int(LL) - 3 + r;
            ncs[((size_t)b * 3 + r) * CC + c] = (p >= 0)
                ? float(x[(size_t)(b * LL + p) * CC + c])
                : (HAS_CS ? cs[((size_t)b * 3 + (3 + p)) * CC + c] : 0.0f);
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
        if (p >= 0) return float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * CC + c))[0]));
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
        bfloat4 dw4 = as_type<bfloat4>(((const device uint2*)(dw + (tok * HH + h) * 4))[0]);
        float4 p0 = x0 * (sw0 + float(dw4.x));
        float4 p1 = s1 * (sw1 + float(dw4.y));
        float4 p2 = s2 * (sw2 + float(dw4.z));
        float4 p3 = s3 * (sw3 + float(dw4.w));
        float4 s = ((p0 + p1) + p2) + p3;
        ((device uint2*)(out + tok * CC + c))[0] = as_type<uint2>(bfloat4((s + x0) / {_S9_LIT}));
        s3 = s2;
        s2 = s1;
        s1 = x0;
    }}
    if (t0 == 0) {{
        for (int r = 0; r < 3; r++) {{
            int p = int(LL) - 3 + r;
            float4 v = (p >= 0)
                ? float4(as_type<bfloat4>(((const device uint2*)(x + ((size_t)b * LL + p) * CC + c))[0]))
                : (HAS_CS ? ((const device float4*)(cs + ((size_t)b * 3 + (3 + p)) * CC + c))[0] : float4(0.0f));
            ((device float4*)(ncs + ((size_t)b * 3 + r) * CC + c))[0] = v;
        }}
    }}
"""


def dyn_short_conv(
    x: mx.array, dyn_weights: mx.array, static_weight: mx.array, conv_state: mx.array | None, num_heads: int
) -> tuple[mx.array, mx.array]:
    """K=4 dynamic short conv + residual/sqrt(9) -> (out bf16, new_cs (B,3,C) f32)."""
    B, L, C = x.shape
    hd = C // num_heads
    has_cs = int(conv_state is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    if C % 4 == 0 and hd % 4 == 0 and L >= 16:
        tt = 16
        ntiles = (L + tt - 1) // tt
        k = _k("pf_dyn_short_conv4", ("x", "dw", "sw", "cs"), ("out", "ncs"), _DSC4_SRC)
        return k(
            inputs=[x, dyn_weights, static_weight, cs_in],
            template=[("CC", C), ("LL", L), ("HH", num_heads), ("HD", hd), ("HAS_CS", has_cs), ("TT", tt)],
            grid=(((C // 4 + 255) // 256) * 256, B * ntiles, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, C), (B, 3, C)],
            output_dtypes=[mx.bfloat16, mx.float32],
        )
    k = _k("pf_dyn_short_conv", ("x", "dw", "sw", "cs"), ("out", "ncs"), _DSC_SRC)
    return k(
        inputs=[x, dyn_weights, static_weight, cs_in],
        template=[("CC", C), ("LL", L), ("HH", num_heads), ("HD", hd), ("HAS_CS", has_cs)],
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


def _fg_mix_src(is_qkv: bool) -> str:
    if is_qkv:
        setup = (
            "    const uint c0 = (threadgroup_position_in_grid.x % (DD / 64)) * 64;\n"
            "    const uint g = threadgroup_position_in_grid.x / (DD / 64);\n"
            "    const device float* stp = (g == 0 ? st0 : (g == 1 ? st1 : st2));\n"
            "    const device float* wfp_base = (g == 0 ? wf0 : (g == 1 ? wf1 : wf2));\n"
            "    const device float* wmp_base = (g == 0 ? wm0 : (g == 1 ? wm1 : wm2));\n"
            "    device float* out_p = (g == 0 ? o0 : (g == 1 ? o1 : o2));\n"
            "    auto hoff_fn = [&](uint mr) { return (SHARED_H ? size_t(mr) : (size_t(mr) * 3 + g)) * DD; };\n"
            "    auto moff_fn = [&](uint mr) { return (size_t(mr) * 4 + g) * DD; };\n"
        )
    else:
        setup = (
            "    const uint c0 = threadgroup_position_in_grid.x * 64;\n"
            "    const device float* stp = st;\n"
            "    const device float* wfp_base = wf;\n"
            "    const device float* wmp_base = wm;\n"
            "    device float* out_p = mixed;\n"
            "    auto hoff_fn = [&](uint mr) { return size_t(mr) * DD; };\n"
            "    auto moff_fn = [&](uint mr) { return (size_t(mr) * 4 + SLOT) * DD; };\n"
        )
    return (
        "    GL_NOFMA\n"
        "    constexpr int BM = 32, TM = 4;\n"
        "    const uint m0 = threadgroup_position_in_grid.y * BM;\n"
        + setup
        + r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint col = c0 + sg * 8 + fn;

    threadgroup float Ss[BM * 72];
    const uint tid = thread_position_in_threadgroup.x;
    for (uint i = tid; i < BM * 32; i += 256) {
        uint r = i >> 5;
        uint k2 = (i & 31) * 2;
        uint rr = min(m0 + r, uint(NN - 1));
        ((threadgroup float2*)(Ss + r * 72 + k2))[0] = ((const device float2*)(stp + (size_t)rr * 64 + k2))[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 cf[TM], cm[TM];
    #pragma clang loop unroll(full)
    for (int tm = 0; tm < TM; ++tm) {
        cf[tm] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        cm[tm] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const device float* wfp0 = wfp_base + (size_t)col * 64;
    const device float* wfp1 = wfp_base + (size_t)(col + 1) * 64;
    const device float* wmp0 = wmp_base + (size_t)col * 64;
    const device float* wmp1 = wmp_base + (size_t)(col + 1) * 64;
    #pragma clang loop unroll(full)
    for (int k = 0; k < 64; k += 8) {
        simdgroup_float8x8 Bf, Bm;
        Bf.thread_elements()[0] = wfp0[k + fm]; Bf.thread_elements()[1] = wfp1[k + fm];
        Bm.thread_elements()[0] = wmp0[k + fm]; Bm.thread_elements()[1] = wmp1[k + fm];
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < TM; ++tm) {
            simdgroup_float8x8 A;
            const float2 av = *((const threadgroup float2*)(Ss + (tm * 8 + fm) * 72 + k + fn));
            A.thread_elements()[0] = av.x; A.thread_elements()[1] = av.y;
            simdgroup_multiply_accumulate(cf[tm], A, Bf, cf[tm]);
            simdgroup_multiply_accumulate(cm[tm], A, Bm, cm[tm]);
        }
    }
    #pragma clang loop unroll(full)
    for (int tm = 0; tm < TM; ++tm) {
        const uint mr = m0 + uint(tm) * 8u + fm;
        if (mr < uint(NN)) {
            size_t off = (size_t)mr * DD + col;
            size_t hoff = hoff_fn(mr) + col;
            size_t moff = moff_fn(mr) + col;
            float2 hv = ((const device float2*)(h + hoff))[0];
            float2 mv = ((const device float2*)(msg + moff))[0];
            float v0 = glue_msg_combine<0>(cf[tm].thread_elements()[0], cm[tm].thread_elements()[0], hv.x, mv.x);
            float v1 = glue_msg_combine<0>(cf[tm].thread_elements()[1], cm[tm].thread_elements()[1], hv.y, mv.y);
            ((device float2*)(out_p + off))[0] = float2(v0, v1);
        }
    }
"""
    )


_FG_MIX_SINGLE_SRC = _fg_mix_src(False)
_FG_MIX_QKV_SRC = _fg_mix_src(True)


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
    k_in = _k("pf_mgin_s", ("h", "msg"), ("out",), _MGIN_SINGLE_SRC)
    g_in = k_in(
        inputs=[h, msg],
        template=[("DD", D), ("TG", _TG_RMS), ("MSTRIDE", 4), ("SLOT", slot)],
        grid=(N * _TG_RMS, 1, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, 2 * D)],
        output_dtypes=[mx.bfloat16],
    )[0]
    if _PF_FG_MIX and N >= 256:
        st = gate_mod.down_proj(g_in).astype(mx.float32)
        k_fg = _k_mma("pf_fg_mix_s", ("st", "wf", "wm", "h", "msg"), ("mixed",), _FG_MIX_SINGLE_SRC)
        mixed = k_fg(
            inputs=[st, gate_mod.forget_gate_proj.weight, gate_mod.message_gate_proj.weight, h, msg],
            template=[("DD", D), ("NN", N), ("SLOT", slot)],
            grid=((D // 64) * 256, (N + 31) // 32, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D)],
            output_dtypes=[mx.float32],
        )[0]
        return rms_norm_f32_bf16(mixed, ln)
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
        k_fg = _k_mma(
            "pf_fg_mix_qkv",
            ("st0", "st1", "st2", "wf0", "wf1", "wf2", "wm0", "wm1", "wm2", "h", "msg"),
            ("o0", "o1", "o2"),
            _FG_MIX_QKV_SRC,
        )
        mx0, mx1, mx2 = k_fg(
            inputs=[
                st0, st1, st2,
                gates[0].forget_gate_proj.weight, gates[1].forget_gate_proj.weight, gates[2].forget_gate_proj.weight,
                gates[0].message_gate_proj.weight, gates[1].message_gate_proj.weight, gates[2].message_gate_proj.weight,
                h_or_ai, msg,
            ],
            template=[("DD", D), ("NN", N), ("SHARED_H", int(shared_h))],
            grid=(3 * (D // 64) * 256, (N + 31) // 32, 1),
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
