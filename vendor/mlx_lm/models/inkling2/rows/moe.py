"""Fused MoE half of a layer visit (fused_ref.py stage E) for B rows, bitwise exact per row. See docs/FUSED_MOE.md.

    router_rows(x, w)                       K11  x (B,2304) bf16 -> router_logits (B,128) f32
    route_group_rows(logits, bias, ...)     K11b routing + compact unique-expert worklist (sorted by count desc)
    gateup_rows(logits, ...)                K12  gate_up gather matvec + softcapped swiglu
    down_rows(swiglu, inds, wts, ...)       K13  down gather + left-fold combine + post-mlp conv + residual
    moe_rows(layer, x, residual, cs3)       full stage E, modules taken from `layer`

Arithmetic matches MLX's fp_qmv_impl / fp_qmv_fast_impl (fp_quantized.h), GEMVKernel::run (rows/gemv.py),
expert_routing source, glue swiglu / moe_combine, and fastdecode short_conv_decode source bit for bit.
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

__all__ = ["router_rows", "router_and_route_rows", "route_group_rows", "gateup_rows", "down_rows", "moe_rows"]

NSG_GU = int(os.environ.get("MOE_NSG_GU", "4"))
NSG_DN = int(os.environ.get("MOE_NSG_DN", "4"))
NSG_DN_FUSED = int(os.environ.get("MOE_NSG_DN_FUSED", "4"))
ROWS_GU = int(os.environ.get("MOE_ROWS_GU", "4"))
ROWS_DN = int(os.environ.get("MOE_ROWS_DN", "4"))
TMAX_CAP = int(os.environ.get("MOE_TMAX_CAP", "2"))
TMAX_CAP_GU = int(os.environ.get("MOE_TMAX_CAP_GU", os.environ.get("MOE_TMAX_CAP", "3")))
TMAX_CAP_DN = int(os.environ.get("MOE_TMAX_CAP_DN", os.environ.get("MOE_TMAX_CAP", "3")))
DOWN_MODE = os.environ.get("MOE_DOWN_MODE", "")  # "" -> fused for B==1, split for B>1
AUTO_ROUTE = os.environ.get("MOE_AUTO_ROUTE", "launch")  # B>1 auto route: "launch" or "group"
MOE_B1_FAST = os.environ.get("MOE_B1_FAST", "1") not in ("0", "")
MOE_B1_NSG_GU = int(os.environ.get("MOE_B1_NSG_GU", "1"))
MOE_B1_ROWS_GU = int(os.environ.get("MOE_B1_ROWS_GU", "4"))
MOE_B1_NSG_DN = int(os.environ.get("MOE_B1_NSG_DN", "8"))
MOE_B1_ROWS_DN = int(os.environ.get("MOE_B1_ROWS_DN", "2"))
MOE_FP8_LUT = os.environ.get("MOE_FP8_LUT", "1") not in ("0", "")
_ZERO_F32 = mx.zeros((1,), mx.float32)
_SCALE_CONV_LIT = repr(float(np.float32(math.sqrt(4 + 1)))) + "f"

# --------------------------------------------------------------------------------------------------------------------
# K11: router gemv (rows/gemv.py gemv_rows_run, with the bf16 -> f32 cast of x in the prologue load)
_ROUTER_HEADER = _gemv._HEADER.replace("const device T* in_vec,   // (B, K) row-major", "const device bfloat16_t* in_vec,")
assert _ROUTER_HEADER != _gemv._HEADER
_router_kernels = {}


def _router_tm1_src(B):
    return f"""
    const int row = (int)threadgroup_position_in_grid.x;
    const int sgN = (int)simdgroup_index_in_threadgroup;
    const int thrN = (int)thread_index_in_simdgroup;
    const int bn0 = (sgN * 32 + thrN) * 4;
    const device float* wr = w + (size_t)row * 2304;
    float result[{B}];
    for (int b = 0; b < {B}; b++) result[b] = 0.0f;
    float4 wv0 = ((const device float4*)(wr + bn0))[0];
    float4 wv1 = ((const device float4*)(wr + 1024 + bn0))[0];
    for (int b = 0; b < {B}; b++) {{
      bfloat4 vv0 = as_type<bfloat4>(((const device uint2*)(x + (size_t)b * 2304 + bn0))[0]);
      result[b] += wv0.x * static_cast<float>(vv0.x);
      result[b] += wv0.y * static_cast<float>(vv0.y);
      result[b] += wv0.z * static_cast<float>(vv0.z);
      result[b] += wv0.w * static_cast<float>(vv0.w);
      bfloat4 vv1 = as_type<bfloat4>(((const device uint2*)(x + (size_t)b * 2304 + 1024 + bn0))[0]);
      result[b] += wv1.x * static_cast<float>(vv1.x);
      result[b] += wv1.y * static_cast<float>(vv1.y);
      result[b] += wv1.z * static_cast<float>(vv1.z);
      result[b] += wv1.w * static_cast<float>(vv1.w);
    }}
    if (bn0 + 2048 < 2304) {{
      const int off = 2048 + bn0;
      float4 wv = ((const device float4*)(wr + off))[0];
      for (int b = 0; b < {B}; b++) {{
        bfloat4 vv = as_type<bfloat4>(((const device uint2*)(x + (size_t)b * 2304 + off))[0]);
        result[b] += wv.x * static_cast<float>(vv.x);
        result[b] += wv.y * static_cast<float>(vv.y);
        result[b] += wv.z * static_cast<float>(vv.z);
        result[b] += wv.w * static_cast<float>(vv.w);
      }}
    }}
    for (int b = 0; b < {B}; b++) {{
      for (ushort sn = 16; sn >= 1; sn >>= 1) result[b] += simd_shuffle_down(result[b], sn);
    }}
    threadgroup float tgp[{B} * 8];
    if (thrN == 0) {{
      for (int b = 0; b < {B}; b++) tgp[b * 8 + sgN] = result[b];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgN == 0 && thrN == 0) {{
      for (int b = 0; b < {B}; b++) {{
        float s = result[b];
        for (int k = 1; k < 8; k++) s += tgp[b * 8 + k];
        o[(size_t)b * 128 + row] = s;
      }}
    }}
"""


_router_mixed_kernels = {}


def _router_from_mixed_tm1_src(B, eps_lit):
    return f"""
    const int row = (int)threadgroup_position_in_grid.x;
    const int sgN = (int)simdgroup_index_in_threadgroup;
    const int thrN = (int)thread_index_in_simdgroup;
    const int bn0 = (sgN * 32 + thrN) * 4;
    float inv[{B}];
    for (int b = 0; b < {B}; b++) {{
      GL_NOFMA
      const device float* pp = part + (size_t)b * 36u;
      float pv = (thrN < 18) ? (pp[thrN * 2] + pp[thrN * 2 + 1]) : 0.0f;
      float tot = simd_sum(pv);
      inv[b] = metal::precise::rsqrt(tot / 2304.0f + {eps_lit});
    }}
    const device float* wr = w + (size_t)row * 2304;
    float result[{B}];
    for (int b = 0; b < {B}; b++) result[b] = 0.0f;
    for (int it = 0; it < 2; it++) {{
      const int off = it * 1024 + bn0;
      float4 wv = ((const device float4*)(wr + off))[0];
      bfloat4 lv = as_type<bfloat4>(((const device uint2*)(ln_w + off))[0]);
      for (int b = 0; b < {B}; b++) {{
        float4 mv = ((const device float4*)(mixed + (size_t)b * 2304 + off))[0];
        bfloat4 vv = bfloat4(
            bfloat16_t(glue_rms_apply<float>(mv.x, inv[b], lv.x)),
            bfloat16_t(glue_rms_apply<float>(mv.y, inv[b], lv.y)),
            bfloat16_t(glue_rms_apply<float>(mv.z, inv[b], lv.z)),
            bfloat16_t(glue_rms_apply<float>(mv.w, inv[b], lv.w)));
        if (row == 0) ((device uint2*)(x_out + (size_t)b * 2304 + off))[0] = as_type<uint2>(vv);
        result[b] += wv.x * static_cast<float>(vv.x);
        result[b] += wv.y * static_cast<float>(vv.y);
        result[b] += wv.z * static_cast<float>(vv.z);
        result[b] += wv.w * static_cast<float>(vv.w);
      }}
    }}
    if (bn0 + 2048 < 2304) {{
      const int off = 2048 + bn0;
      float4 wv = ((const device float4*)(wr + off))[0];
      bfloat4 lv = as_type<bfloat4>(((const device uint2*)(ln_w + off))[0]);
      for (int b = 0; b < {B}; b++) {{
        float4 mv = ((const device float4*)(mixed + (size_t)b * 2304 + off))[0];
        bfloat4 vv = bfloat4(
            bfloat16_t(glue_rms_apply<float>(mv.x, inv[b], lv.x)),
            bfloat16_t(glue_rms_apply<float>(mv.y, inv[b], lv.y)),
            bfloat16_t(glue_rms_apply<float>(mv.z, inv[b], lv.z)),
            bfloat16_t(glue_rms_apply<float>(mv.w, inv[b], lv.w)));
        if (row == 0) ((device uint2*)(x_out + (size_t)b * 2304 + off))[0] = as_type<uint2>(vv);
        result[b] += wv.x * static_cast<float>(vv.x);
        result[b] += wv.y * static_cast<float>(vv.y);
        result[b] += wv.z * static_cast<float>(vv.z);
        result[b] += wv.w * static_cast<float>(vv.w);
      }}
    }}
    for (int b = 0; b < {B}; b++) {{
      for (ushort sn = 16; sn >= 1; sn >>= 1) result[b] += simd_shuffle_down(result[b], sn);
    }}
    threadgroup float tgp[{B} * 8];
    if (thrN == 0) {{
      for (int b = 0; b < {B}; b++) tgp[b * 8 + sgN] = result[b];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgN == 0 && thrN == 0) {{
      for (int b = 0; b < {B}; b++) {{
        float s = result[b];
        for (int k = 1; k < 8; k++) s += tgp[b * 8 + k];
        o[(size_t)b * 128 + row] = s;
      }}
    }}
"""


def _router_tm1_chunk_src(RB=8):
    return f"""
    const int row = (int)threadgroup_position_in_grid.x;
    const int b0 = (int)threadgroup_position_in_grid.z * {RB};
    const int nb = min({RB}, BB - b0);
    const int sgN = (int)simdgroup_index_in_threadgroup;
    const int thrN = (int)thread_index_in_simdgroup;
    const int bn0 = (sgN * 32 + thrN) * 4;
    const device float* wr = w + (size_t)row * 2304;
    float result[{RB}];
    for (int b = 0; b < {RB}; b++) result[b] = 0.0f;
    float4 wv0 = ((const device float4*)(wr + bn0))[0];
    float4 wv1 = ((const device float4*)(wr + 1024 + bn0))[0];
    for (int b = 0; b < {RB}; b++) {{
      if (b < nb) {{
        bfloat4 vv0 = as_type<bfloat4>(((const device uint2*)(x + (size_t)(b0 + b) * 2304 + bn0))[0]);
        result[b] += wv0.x * static_cast<float>(vv0.x);
        result[b] += wv0.y * static_cast<float>(vv0.y);
        result[b] += wv0.z * static_cast<float>(vv0.z);
        result[b] += wv0.w * static_cast<float>(vv0.w);
        bfloat4 vv1 = as_type<bfloat4>(((const device uint2*)(x + (size_t)(b0 + b) * 2304 + 1024 + bn0))[0]);
        result[b] += wv1.x * static_cast<float>(vv1.x);
        result[b] += wv1.y * static_cast<float>(vv1.y);
        result[b] += wv1.z * static_cast<float>(vv1.z);
        result[b] += wv1.w * static_cast<float>(vv1.w);
      }}
    }}
    if (bn0 + 2048 < 2304) {{
      const int off = 2048 + bn0;
      float4 wv = ((const device float4*)(wr + off))[0];
      for (int b = 0; b < {RB}; b++) {{
        if (b < nb) {{
          bfloat4 vv = as_type<bfloat4>(((const device uint2*)(x + (size_t)(b0 + b) * 2304 + off))[0]);
          result[b] += wv.x * static_cast<float>(vv.x);
          result[b] += wv.y * static_cast<float>(vv.y);
          result[b] += wv.z * static_cast<float>(vv.z);
          result[b] += wv.w * static_cast<float>(vv.w);
        }}
      }}
    }}
    for (int b = 0; b < {RB}; b++) {{
      for (ushort sn = 16; sn >= 1; sn >>= 1) result[b] += simd_shuffle_down(result[b], sn);
    }}
    threadgroup float tgp[{RB} * 8];
    if (thrN == 0) {{
      for (int b = 0; b < {RB}; b++) tgp[b * 8 + sgN] = result[b];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgN == 0 && thrN == 0) {{
      for (int b = 0; b < {RB}; b++) {{
        if (b < nb) {{
          float s = result[b];
          for (int k = 1; k < 8; k++) s += tgp[b * 8 + k];
          o[(size_t)(b0 + b) * 128 + row] = s;
        }}
      }}
    }}
"""


def router_from_mixed_rows(mixed, part, ln_w, w, *, eps_ln=1e-6):
    """Fused K3b(mlp) + K11 router: mixed (B,1,2304) f32, part (B,1,36) f32, ln_w (2304,) bf16, w (128,2304) f32
    -> (logits (B,128) f32, x_mlp (B,2304) bf16); bitwise identical to k3b_norm_layer + router_rows."""
    B = int(mixed.shape[0])
    N, K = w.shape
    if B < 1 or B > 32 or N != 128 or K != 2304 or mixed.dtype != mx.float32 or part.dtype != mx.float32 or w.dtype != mx.float32:
        raise ValueError("router_from_mixed_rows: bad shapes/dtypes")
    eps_lit = flit(eps_ln)
    key = (B, eps_lit)
    k = _router_mixed_kernels.get(key)
    if k is None:
        k = _router_mixed_kernels[key] = mx.fast.metal_kernel(
            name="moe_router_mixed_b%d_%s" % (B, eps_lit.replace(".", "p").replace("-", "m")),
            input_names=["w", "mixed", "part", "ln_w"],
            output_names=["o", "x_out"],
            source=_router_from_mixed_tm1_src(B, eps_lit),
            header=GLUE_HEADER,
            ensure_row_contiguous=True,
        )
    return k(
        inputs=[w, mixed.reshape(B, 2304), part.reshape(B, 36), ln_w],
        grid=(128 * 32, 8, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(B, N), (B, K)],
        output_dtypes=[mx.float32, mx.bfloat16],
    )


def router_rows(x, w):
    """x (B,K) bf16, w (N,K) f32 as stored -> (B,N) f32; row b bitwise = f32(x[b]) @ w.T at M == 1."""
    B, K = x.shape
    N = w.shape[0]
    if x.dtype != mx.bfloat16 or w.dtype != mx.float32 or w.shape[1] != K:
        raise ValueError("router_rows: x bf16 (B,K), w f32 (N,K)")
    tm1 = (os.environ.get("MOE_ROUTER_TM1", "1") not in ("0", "")) and N == 128 and K == 2304 and B <= 32
    if tm1:
        if B <= 16:
            key = (B, N, K, True)
            k = _router_kernels.get(key)
            if k is None:
                k = _router_kernels[key] = mx.fast.metal_kernel(
                    name="moe_router_tm1_b%d" % B, input_names=["w", "x"], output_names=["o"],
                    source=_router_tm1_src(B), ensure_row_contiguous=True)
            (o,) = k(inputs=[w, x], grid=(128 * 32, 8, 1), threadgroup=(32, 8, 1),
                     output_shapes=[(B, N)], output_dtypes=[mx.float32])
            return o
        key = ("chunk8", N, K)
        k = _router_kernels.get(key)
        if k is None:
            k = _router_kernels[key] = mx.fast.metal_kernel(
                name="moe_router_tm1_c8", input_names=["w", "x"], output_names=["o"],
                source=_router_tm1_chunk_src(8), ensure_row_contiguous=True)
        (o,) = k(inputs=[w, x], template=[("BB", B)],
                 grid=(128 * 32, 8, (B + 7) // 8), threadgroup=(32, 8, 1),
                 output_shapes=[(B, N)], output_dtypes=[mx.float32])
        return o
    cfg = _gemv._cfg(N, K)
    bm, bn, sm, sn, tm, tn = cfg
    blockM = bm * sm * tm
    nt = (N + blockM - 1) // blockM
    key = (B, N, K, False)
    k = _router_kernels.get(key)
    if k is None:
        src = (
            "threadgroup float tgp_mem[%d];\n"
            "const uint t = threadgroup_position_in_grid.x;\n"
            "gemv_rows_run<float, %d, %d, %d, %d, %d, %d, %d>(w, x, o, %d, %d, tgp_mem, t, "
            "simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
        ) % (B * bn * (blockM + tm) if bn > 1 else 1, B, bm, bn, sm, sn, tm, tn, K, N)
        k = _router_kernels[key] = mx.fast.metal_kernel(
            name="moe_router_b%d_n%d_k%d" % (B, N, K), input_names=["w", "x"], output_names=["o"],
            header=_ROUTER_HEADER, source=src)
    (o,) = k(inputs=[w, x], grid=(nt * 32, bn, bm), threadgroup=(32, bn, bm),
             output_shapes=[(B, N)], output_dtypes=[mx.float32])
    return o


# --------------------------------------------------------------------------------------------------------------------
_ROUTE_FN = r"""
// expert_routing (vendor/mlx_lm/kernels/expert_routing.py) for ONE row, run by one simdgroup; verbatim arithmetic.
template <typename RPtr, typename IPtr, typename WPtr>
inline void moe_route_row(RPtr row, const device float* bias, uint lane,
                          IPtr inds, WPtr wts, float eps, float scale) {
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

// Partial top-k routing returning only the target_q-th winner (0 <= target_q < 8); bitwise identical to moe_route_row.
template <typename RPtr>
inline uint moe_route_row_q(RPtr row, const device float* bias, uint lane, int target_q) {
    constexpr int E = 128;
    constexpr int R = (E + 31) / 32;
    float biased[R];
    for (int i = 0; i < R; i++) {
        int e = lane + 32 * i;
        biased[i] = -INFINITY;
        if (e < E) {
            float x = row[e] * 1.0f;
            float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
            float sc = x < 0 ? y : 1.0f - y;
            biased[i] = sc + bias[e];
        }
    }
    int winner = 0;
    for (int k = 0; k <= target_q; k++) {
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
        winner = simd_min(best == top ? best_e : E);
        if (k == target_q) break;
        for (int i = 0; i < R; i++) {
            if (lane + 32 * i == winner) {
                biased[i] = -INFINITY;
            }
        }
    }
    return uint(winner);
}
"""

def _gen_fp_luts():
    e4m3 = []
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 15 and man == 7:
            e4m3.append("NAN")
        elif exp == 0:
            v = s * (man / 8.0) * (2.0 ** -6)
            e4m3.append(("-0.0f" if (b & 0x80) else "0.0f") if man == 0 else f"{v!r}f")
        else:
            v = s * (1.0 + man / 8.0) * (2.0 ** (exp - 7))
            e4m3.append(f"{v!r}f")
    fp4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
    fp4_s = ", ".join(("-0.0f" if i == 8 else f"{v!r}f") for i, v in enumerate(fp4))
    return (
        "constant float FP8_E4M3_LUT[256] = {" + ", ".join(e4m3) + "};\n"
        + "constant float FP4_E2M1_LUT[16] = {" + fp4_s + "};\n"
    )


_FP_LUT_HEADER = _gen_fp_luts() if MOE_FP8_LUT else ""

_DOT_BF16 = _FP_LUT_HEADER + (r"""
template <int GS>
inline float moe_dequant_scale(uint8_t s) {
  if constexpr (GS == 16) return FP8_E4M3_LUT[s];
  else return dequantize_scale<float, GS>(s);
}
""" if MOE_FP8_LUT else r"""
template <int GS>
inline float moe_dequant_scale(uint8_t s) {
  return dequantize_scale<float, GS>(s);
}
""") + r"""
template <int bits, int values_per_thread>
inline float dot_vals_bf16(const thread bfloat16_t* xt, const thread float* d, float scale) {
  float accum = 0;
  if constexpr (bits == 4) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (float(xt[4 * i]) * d[4 * i] +
           float(xt[4 * i + 1]) * d[4 * i + 1] +
           float(xt[4 * i + 2]) * d[4 * i + 2] +
           float(xt[4 * i + 3]) * d[4 * i + 3]);
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      accum += float(xt[i]) * d[i];
    }
  }
  return scale * accum;
}

template <int bits>
inline float dot_vals_bf16_u4val(uint4 xpk, const thread float* d, float scale) {
  bfloat4 v0 = as_type<bfloat4>(xpk.xy);
  bfloat4 v1 = as_type<bfloat4>(xpk.zw);
  float accum = 0;
  if constexpr (bits == 4) {
    accum += (float(v0.x) * d[0] + float(v0.y) * d[1] + float(v0.z) * d[2] + float(v0.w) * d[3]);
    accum += (float(v1.x) * d[4] + float(v1.y) * d[5] + float(v1.z) * d[6] + float(v1.w) * d[7]);
  } else {
    accum += float(v0.x) * d[0];
    accum += float(v0.y) * d[1];
    accum += float(v0.z) * d[2];
    accum += float(v0.w) * d[3];
    accum += float(v1.x) * d[4];
    accum += float(v1.y) * d[5];
    accum += float(v1.z) * d[6];
    accum += float(v1.w) * d[7];
  }
  return scale * accum;
}

template <int bits>
inline float dot_vals_bf16_u2val(uint2 xpk, const thread float* d, float scale) {
  bfloat4 v0 = as_type<bfloat4>(xpk);
  float accum = 0;
  if constexpr (bits == 4) {
    accum += (float(v0.x) * d[0] + float(v0.y) * d[1] + float(v0.z) * d[2] + float(v0.w) * d[3]);
  } else {
    accum += float(v0.x) * d[0];
    accum += float(v0.y) * d[1];
    accum += float(v0.z) * d[2];
    accum += float(v0.w) * d[3];
  }
  return scale * accum;
}

template <int bits>
inline float dot_vals_f4_8(float4 v0, float4 v1, const thread float* d, float scale) {
  float accum = 0;
  if constexpr (bits == 4) {
    accum += (v0.x * d[0] + v0.y * d[1] + v0.z * d[2] + v0.w * d[3]);
    accum += (v1.x * d[4] + v1.y * d[5] + v1.z * d[6] + v1.w * d[7]);
  } else {
    accum += v0.x * d[0];
    accum += v0.y * d[1];
    accum += v0.z * d[2];
    accum += v0.w * d[3];
    accum += v1.x * d[4];
    accum += v1.y * d[5];
    accum += v1.z * d[6];
    accum += v1.w * d[7];
  }
  return scale * accum;
}

template <int bits>
inline float dot_vals_f4_4(float4 v0, const thread float* d, float scale) {
  float accum = 0;
  if constexpr (bits == 4) {
    accum += (v0.x * d[0] + v0.y * d[1] + v0.z * d[2] + v0.w * d[3]);
  } else {
    accum += v0.x * d[0];
    accum += v0.y * d[1];
    accum += v0.z * d[2];
    accum += v0.w * d[3];
  }
  return scale * accum;
}

inline void dequant_u32_fp4_8(uint32_t v, thread float* d) {
  d[0] = Dequantize<4>{}(uint16_t(v));
  d[1] = Dequantize<4>{}(uint16_t(v >> 4));
  d[2] = Dequantize<4>{}(uint16_t(v >> 8));
  d[3] = Dequantize<4>{}(uint16_t(v >> 12));
  d[4] = Dequantize<4>{}(uint16_t(v >> 16));
  d[5] = Dequantize<4>{}(uint16_t(v >> 20));
  d[6] = Dequantize<4>{}(uint16_t(v >> 24));
  d[7] = Dequantize<4>{}(uint16_t(v >> 28));
}
""" + (r"""
inline void dequant_u32_fp4_8_b1(uint32_t v, thread float* d) {
  d[0] = FP4_E2M1_LUT[v & 15u];
  d[1] = FP4_E2M1_LUT[(v >> 4) & 15u];
  d[2] = FP4_E2M1_LUT[(v >> 8) & 15u];
  d[3] = FP4_E2M1_LUT[(v >> 12) & 15u];
  d[4] = FP4_E2M1_LUT[(v >> 16) & 15u];
  d[5] = FP4_E2M1_LUT[(v >> 20) & 15u];
  d[6] = FP4_E2M1_LUT[(v >> 24) & 15u];
  d[7] = FP4_E2M1_LUT[(v >> 28) & 15u];
}

inline void dequant_u2_fp8_8(uint2 v, thread float* d) {
  uchar4 c0 = as_type<uchar4>(v.x);
  uchar4 c1 = as_type<uchar4>(v.y);
  d[0] = FP8_E4M3_LUT[c0.x];
  d[1] = FP8_E4M3_LUT[c0.y];
  d[2] = FP8_E4M3_LUT[c0.z];
  d[3] = FP8_E4M3_LUT[c0.w];
  d[4] = FP8_E4M3_LUT[c1.x];
  d[5] = FP8_E4M3_LUT[c1.y];
  d[6] = FP8_E4M3_LUT[c1.z];
  d[7] = FP8_E4M3_LUT[c1.w];
}
""" if MOE_FP8_LUT else r"""
inline void dequant_u32_fp4_8_b1(uint32_t v, thread float* d) {
  dequant_u32_fp4_8(v, d);
}

inline void dequant_u2_fp8_8(uint2 v, thread float* d) {
  uchar4 c0 = as_type<uchar4>(v.x);
  uchar4 c1 = as_type<uchar4>(v.y);
  d[0] = Dequantize<8>{}(c0.x);
  d[1] = Dequantize<8>{}(c0.y);
  d[2] = Dequantize<8>{}(c0.z);
  d[3] = Dequantize<8>{}(c0.w);
  d[4] = Dequantize<8>{}(c1.x);
  d[5] = Dequantize<8>{}(c1.y);
  d[6] = Dequantize<8>{}(c1.z);
  d[7] = Dequantize<8>{}(c1.w);
}
""")

HEADER = mlx_header() + _DOT_BF16 + GLUE_HEADER + _ROUTE_FN

# --------------------------------------------------------------------------------------------------------------------
# K11b: route + compact unique-expert table (1 parallel 128-thread threadgroup launch)
_ROUTE_GROUP_SRC = r"""
    constexpr int NPAIR = BB * 8;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint ttid = thread_index_in_threadgroup;
    threadgroup uint ind_s[64];
    threadgroup float wt_s[64];
    threadgroup uint cnt_e[128];
    threadgroup uint64_t mask_e[128];

    for (int row = (int)sg; row < BB; row += 4) {
        moe_route_row(logits + row * 128, bias, lane, ind_s + row * 8, wt_s + row * 8, eps, rscale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = (int)ttid; i < NPAIR; i += 128) {
        inds_o[i] = ind_s[i];
        wts_o[i] = wt_s[i];
    }
    uint64_t my_mask = 0;
    for (int p = 0; p < NPAIR; p++) {
        if (ind_s[p] == ttid) my_mask |= (1ULL << p);
    }
    cnt_e[ttid] = (uint)metal::popcount(my_mask);
    mask_e[ttid] = my_mask;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint out_u = 0;
        uint off_acc = 0;
        for (int tc = BB; tc >= 1; tc--) {
            for (int base = 0; base < 128; base += 32) {
                uint e = uint(base) + lane;
                uint c = cnt_e[e];
                bool take = (int(c) == tc);
                int pre = simd_prefix_exclusive_sum(take ? 1 : 0);
                uint slot = out_u + uint(pre);
                uint my_off = off_acc + uint(pre) * uint(tc);
                if (take) {
                    uniq_e[slot] = e;
                    u_cnt[slot] = c;
                    u_off[slot] = my_off;
                    uint64_t m = mask_e[e];
                    for (int j = 0; j < tc; j++) {
                        uint p = uint(metal::ctz(m));
                        pair_list[my_off + uint(j)] = p;
                        m &= (m - 1ULL);
                    }
                }
                int taken = simd_sum(take ? 1 : 0);
                out_u += uint(taken);
                off_acc += uint(taken * tc);
            }
        }
        if (lane == 0) num_uniq[0] = out_u;
    }
"""

_rg_kernel = None
_rf_kernels = {}
_rr_kernels = {}


def _router_route_tm1_src(ec, rc):
    return f"""
    const uint b = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint lid = thread_position_in_threadgroup.x;
    const uint grp = sg >> 3;
    const uint sgN = sg & 7;
    const uint thrN = lane;
    const int bn0 = (int)(sgN * 32 + thrN) * 4;

    threadgroup float tg_x[2304];
    threadgroup float tgp[128 * 8];
    threadgroup float tg_logits[128];

    const device bfloat16_t* xb = x + (size_t)b * 2304;
    for (int i = (int)lid; i < 2304; i += 1024) {{
        tg_x[i] = static_cast<float>(xb[i]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int row = (int)grp; row < 128; row += 4) {{
        const device float* wr = w + (size_t)row * 2304;
        float result = 0.0f;
        for (int it = 0; it < 2; it++) {{
            const int off = it * 1024 + bn0;
            float4 wv = ((const device float4*)(wr + off))[0];
            result += wv.x * tg_x[off + 0];
            result += wv.y * tg_x[off + 1];
            result += wv.z * tg_x[off + 2];
            result += wv.w * tg_x[off + 3];
        }}
        if (bn0 + 2048 < 2304) {{
            const int off = 2048 + bn0;
            float4 wv = ((const device float4*)(wr + off))[0];
            result += wv.x * tg_x[off + 0];
            result += wv.y * tg_x[off + 1];
            result += wv.z * tg_x[off + 2];
            result += wv.w * tg_x[off + 3];
        }}
        for (ushort sn = 16; sn >= 1; sn >>= 1) result += simd_shuffle_down(result, sn);
        if (thrN == 0) tgp[row * 8 + sgN] = result;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 128) {{
        float s = tgp[lid * 8 + 0];
        for (int k = 1; k < 8; k++) s += tgp[lid * 8 + k];
        tg_logits[lid] = s;
        logits_o[(size_t)b * 128 + lid] = s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
        moe_route_row(tg_logits, bias, lane, inds_o + b * 8, wts_o + b * 8, float({ec}), float({rc}));
    }}
"""


def router_and_route_rows(x, w, bias, *, score_eps=1e-20, route_scale=1.0):
    """Fused K11+K11a: x (B,2304) bf16, w (128,2304) f32, bias (128,) f32 -> (logits, inds, wts), bitwise identical."""
    B, K = x.shape
    N = w.shape[0]
    if B < 2 or B > 8 or N != 128 or K != 2304 or x.dtype != mx.bfloat16 or w.dtype != mx.float32:
        logits = router_rows(x, w)
        inds, wts = route_fast_rows(logits, bias, score_eps=score_eps, route_scale=route_scale)
        return logits, inds, wts
    if bias.dtype != mx.float32:
        bias = bias.astype(mx.float32)
    ec, rc = flit(score_eps), flit(route_scale)
    key = (ec, rc)
    k = _rr_kernels.get(key)
    if k is None:
        k = _rr_kernels[key] = mx.fast.metal_kernel(
            name=f"moe_router_route_{ec}_{rc}".replace(".", "p").replace("-", "m").replace("+", "P"),
            input_names=["x", "w", "bias"],
            output_names=["logits_o", "inds_o", "wts_o"],
            source=_router_route_tm1_src(ec, rc),
            header=_ROUTE_FN,
            ensure_row_contiguous=True,
        )
    return list(k(
        inputs=[x, w, bias],
        grid=(B * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B, N), (B, 8), (B, 8)],
        output_dtypes=[mx.float32, mx.uint32, mx.float32],
    ))


def route_fast_rows(logits, bias, *, score_eps=1e-20, route_scale=1.0, w_act=None):
    """K11a: B independent 1-simdgroup threadgroups using moe_route_row for E=128."""
    B = int(logits.shape[0])
    if bias.dtype != mx.float32:
        bias = bias.astype(mx.float32)
    ec, rc = flit(score_eps), flit(route_scale)
    has_wa = w_act is not None
    key = (ec, rc, has_wa)
    k = _rf_kernels.get(key)
    if k is None:
        src = (
            "uint b = threadgroup_position_in_grid.x;\n"
            + ("if ((int)b >= w_act[0]) return;\n" if has_wa else "")
            + "uint lane = thread_index_in_simdgroup;\n"
            + f"moe_route_row(logits + b * 128, bias, lane, inds + b * 8, wts + b * 8, float({ec}), float({rc}));"
        )
        ins = ["logits", "bias", "w_act"] if has_wa else ["logits", "bias"]
        k = _rf_kernels[key] = mx.fast.metal_kernel(
            name=f"moe_route_fast_{ec}_{rc}_{int(has_wa)}".replace(".", "p").replace("-", "m").replace("+", "P"),
            input_names=ins,
            output_names=["inds", "wts"],
            source=src,
            header=_ROUTE_FN,
            ensure_row_contiguous=True,
        )
    args = [logits, bias, w_act.reshape(-1)] if has_wa else [logits, bias]
    return k(
        inputs=args,
        grid=(B * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, 8), (B, 8)],
        output_dtypes=[mx.uint32, mx.float32],
    )


def route_group_rows(logits, bias, *, score_eps=1e-20, route_scale=1.0):
    """K11b: logits (B,128) f32, bias (128,) f32 -> (inds (B,8) u32, wts (B,8) f32,
    num_uniq (1,) u32, uniq_e (B*8,) u32, u_off (B*8,) u32, u_cnt (B*8,) u32, pair_list (B*8,) u32)."""
    global _rg_kernel
    B = logits.shape[0]
    if B < 1 or B > 8 or logits.shape != (B, 128) or logits.dtype != mx.float32:
        raise ValueError("route_group_rows: logits must be (B,128) f32, B in 1..8")
    if bias.shape != (128,) or bias.dtype != mx.float32:
        raise ValueError("route_group_rows: bias must be (128,) f32")
    if _rg_kernel is None:
        _rg_kernel = mx.fast.metal_kernel(
            name="moe_route_group",
            input_names=["logits", "bias", "eps", "rscale"],
            output_names=["inds_o", "wts_o", "num_uniq", "uniq_e", "u_off", "u_cnt", "pair_list"],
            source=_ROUTE_GROUP_SRC,
            header=HEADER,
            ensure_row_contiguous=True,
        )
    npair = B * 8
    return list(_rg_kernel(
        inputs=[logits, bias, scalar(score_eps), scalar(route_scale)],
        template=[("BB", B)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(B, 8), (B, 8), (1,), (npair,), (npair,), (npair,), (npair,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.uint32, mx.uint32, mx.uint32, mx.uint32],
    ))


# --------------------------------------------------------------------------------------------------------------------
# Zero-sx, zero-barrier register-lean K-loop for gate_up and down_gather: cnt==1 fast path, or r-outside-t over bf16 x cache.
def _k_body_lean(xoff_expr, epi_one_src, epi_multi_src, *, use_mask=False, has_stg=False, pq_one_q=False):
    pq_one_decl = "const int pq = q;" if (use_mask or pq_one_q) else "const int pq = (int)pl[0];"
    tgsm_decl = (
        "        threadgroup uint4 x4_s[(STG_X && VPT == 8) ? (TMAX * 32) : 1];\n"
        "        threadgroup uint2 x2_s[(STG_X && VPT == 4) ? (TMAX * 32) : 1];\n"
        if has_stg
        else ""
    )
    rem_init = "        uint64_t rem = mask;\n" if use_mask else ""
    tb_setup = (
        r"""            uint xo[TMAX];
            uint pqs_tb[TMAX];
            for (int t = 0; t < TMAX; t++) {
                if (t < nb) {
                    int pq = int(metal::ctz(rem));
                    rem &= (rem - 1ULL);
                    pqs_tb[t] = (uint)pq;
                    xo[t] = (uint)(@XOFF@) * (uint)KK + lane * (uint)VPT;
                } else {
                    pqs_tb[t] = 0u;
                    xo[t] = 0u;
                }
            }"""
        if use_mask
        else r"""            uint xo[TMAX];
            for (int t = 0; t < TMAX; t++) {
                if (t < nb) {
                    int pq = (int)pl[tb + t];
                    xo[t] = (uint)(@XOFF@) * (uint)KK + lane * (uint)VPT;
                } else {
                    xo[t] = 0;
                }
            }"""
    )
    epi_m = epi_multi_src.replace("pl[tb + t]", "pqs_tb[t]") if use_mask else epi_multi_src
    stg_v8_body = (
        r"""                    if constexpr (STG_X) {
                        for (int t = (int)sg; t < nb; t += NSG) {
                            x4_s[t * 32 + lane] = ((const device uint4*)(x + xo[t] + k_off))[0];
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            thread U wv[VPT];
                            dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                            U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                            #pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < nb) {
                                    result[t][r] += dot_vals_bf16_u4val<BITS>(x4_s[t * 32 + lane], wv, sv);
                                }
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    } else """
        if has_stg
        else ""
    )
    stg_v4_body = (
        r"""                    if constexpr (STG_X) {
                        for (int t = (int)sg; t < nb; t += NSG) {
                            x2_s[t * 32 + lane] = ((const device uint2*)(x + xo[t] + k_off))[0];
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            thread U wv[VPT];
                            dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                            U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                            #pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < nb) {
                                    result[t][r] += dot_vals_bf16_u2val<BITS>(x2_s[t * 32 + lane], wv, sv);
                                }
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    } else """
        if has_stg
        else ""
    )
    stg_tail_body = (
        r"""                if constexpr (STG_X && (VPT == 4 || VPT == 8)) {
                    if constexpr (VPT == 8) {
                        for (int t = (int)sg; t < nb; t += NSG) {
                            if ((int)lane * VPT < TAILK) {
                                x4_s[t * 32 + lane] = ((const device uint4*)(x + xo[t] + k_off))[0];
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (tail_on) {
                            #pragma unroll
                            for (int r = 0; r < ROWS; r++) {
                                thread U wv[VPT];
                                dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                                U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                                #pragma unroll
                                for (int t = 0; t < TMAX; t++) {
                                    if (t < nb) result[t][r] += dot_vals_bf16_u4val<BITS>(x4_s[t * 32 + lane], wv, sv);
                                }
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    } else {
                        for (int t = (int)sg; t < nb; t += NSG) {
                            if ((int)lane * VPT < TAILK) {
                                x2_s[t * 32 + lane] = ((const device uint2*)(x + xo[t] + k_off))[0];
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (tail_on) {
                            #pragma unroll
                            for (int r = 0; r < ROWS; r++) {
                                thread U wv[VPT];
                                dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                                U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                                #pragma unroll
                                for (int t = 0; t < TMAX; t++) {
                                    if (t < nb) result[t][r] += dot_vals_bf16_u2val<BITS>(x2_s[t * 32 + lane], wv, sv);
                                }
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                } else """
        if has_stg
        else ""
    )
    return (
        r"""
    constexpr int NFULL = KK / BLOCK;
    constexpr int TAILK = KK - NFULL * BLOCK;
    constexpr bool HAS_TAIL = (TAILK > 0);
    const bool tail_on = HAS_TAIL && ((int)lane * VPT < TAILK);
    const int in_w = KK * (BITS) / 8;
    const int in_g = KK / GS;
    const device uint8_t* ws0 = (const device uint8_t*)w + (long)e * NN * in_w
        + (long)out_row * in_w + lane * PPT * 4;
    const device uint8_t* sc0 = scales + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
    float inv_scale_enc = 1.0f;
    if constexpr (HGS) {
        inv_scale_enc = gsc[e] / (F8E4M3_MAX * F4E2M1_MAX);
    }

    if (cnt == 1) {
        @PQ_ONE@
        const long xo = (long)(@XOFF@) * KK + (long)lane * VPT;
        thread U result[ROWS];
        for (int r = 0; r < ROWS; r++) result[r] = 0;
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
            } else if constexpr (VPT == 4) {
                uint2 x2 = ((const device uint2*)(x + xo + k_off))[0];
                float4 xf0 = float4(as_type<bfloat4>(x2));
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    thread U wv[VPT];
                    dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    result[r] += dot_vals_f4_4<BITS>(xf0, wv, sv);
                }
            } else {
                thread U xt[VPT];
                load_vector<T, U, VPT>(x + xo + k_off, xt);
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    thread U wv[VPT];
                    dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    result[r] += dot_vals<BITS, VPT>(xt, wv, sv);
                }
            }
            ws += BLOCK * 4 / PF;
            sc += BLOCK / GS;
        }
        if constexpr (HAS_TAIL) {
            if (tail_on) {
                const uint k_off = (uint)NFULL * (uint)BLOCK;
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
                } else if constexpr (VPT == 4) {
                    uint2 x2 = ((const device uint2*)(x + xo + k_off))[0];
                    float4 xf0 = float4(as_type<bfloat4>(x2));
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        thread U wv[VPT];
                        dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        result[r] += dot_vals_f4_4<BITS>(xf0, wv, sv);
                    }
                } else {
                    thread U xt[VPT];
                    load_vector<T, U, VPT>(x + xo + k_off, xt);
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        thread U wv[VPT];
                        dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        result[r] += dot_vals<BITS, VPT>(xt, wv, sv);
                    }
                }
            }
        }
        @EPI_ONE@
    } else {
@TGSM_DECL@@REM_INIT@        for (int tb = 0; tb < cnt; tb += TMAX) {
            const int nb = min(TMAX, cnt - tb);
@TB_SETUP@
            thread U result[TMAX][ROWS];
            for (int t = 0; t < TMAX; t++)
                for (int r = 0; r < ROWS; r++) result[t][r] = 0;

            const device uint8_t* ws = ws0;
            const device uint8_t* sc = sc0;
            for (int it = 0; it < NFULL; it++) {
                const uint k_off = (uint)it * (uint)BLOCK;
                if constexpr (VPT == 8) {
@STG_V8@                    if constexpr (STREAM_X) {
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            thread U wv[VPT];
                            dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                            U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                            #pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < nb) {
                                    uint4 xpk = ((const device uint4*)(x + xo[t] + k_off))[0];
                                    result[t][r] += dot_vals_bf16_u4val<BITS>(xpk, wv, sv);
                                }
                            }
                        }
                    } else {
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
                    }
                } else if constexpr (VPT == 4) {
@STG_V4@                    if constexpr (STREAM_X) {
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            thread U wv[VPT];
                            dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                            U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                            #pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < nb) {
                                    uint2 xpk = ((const device uint2*)(x + xo[t] + k_off))[0];
                                    result[t][r] += dot_vals_bf16_u2val<BITS>(xpk, wv, sv);
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
                } else {
                    thread T xb[TMAX][VPT];
                    #pragma unroll
                    for (int t = 0; t < TMAX; t++) {
                        if (t < nb) {
                            const device T* xp = x + xo[t] + k_off;
                            for (int i = 0; i < VPT; i++) xb[t][i] = xp[i];
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        thread U wv[VPT];
                        dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        #pragma unroll
                        for (int t = 0; t < TMAX; t++) {
                            if (t < nb) result[t][r] += dot_vals_bf16<BITS, VPT>(xb[t], wv, sv);
                        }
                    }
                }
                ws += BLOCK * 4 / PF;
                sc += BLOCK / GS;
            }
            if constexpr (HAS_TAIL) {
                const uint k_off = (uint)NFULL * (uint)BLOCK;
@STG_TAIL@                if (tail_on) {
                    if constexpr (VPT == 8) {
                        if constexpr (STREAM_X) {
                            #pragma unroll
                            for (int r = 0; r < ROWS; r++) {
                                thread U wv[VPT];
                                dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                                U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                                #pragma unroll
                                for (int t = 0; t < TMAX; t++) {
                                    if (t < nb) {
                                        uint4 xpk = ((const device uint4*)(x + xo[t] + k_off))[0];
                                        result[t][r] += dot_vals_bf16_u4val<BITS>(xpk, wv, sv);
                                    }
                                }
                            }
                        } else {
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
                        }
                    } else if constexpr (VPT == 4) {
                        if constexpr (STREAM_X) {
                            #pragma unroll
                            for (int r = 0; r < ROWS; r++) {
                                thread U wv[VPT];
                                dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                                U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                                #pragma unroll
                                for (int t = 0; t < TMAX; t++) {
                                    if (t < nb) {
                                        uint2 xpk = ((const device uint2*)(x + xo[t] + k_off))[0];
                                        result[t][r] += dot_vals_bf16_u2val<BITS>(xpk, wv, sv);
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
                    } else {
                        thread T xb[TMAX][VPT];
                        #pragma unroll
                        for (int t = 0; t < TMAX; t++) {
                            if (t < nb) {
                                const device T* xp = x + xo[t] + k_off;
                                for (int i = 0; i < VPT; i++) xb[t][i] = xp[i];
                            }
                        }
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            thread U wv[VPT];
                            dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                            U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                            #pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < nb) result[t][r] += dot_vals_bf16<BITS, VPT>(xb[t], wv, sv);
                            }
                        }
                    }
                }
            }
            @EPI_MULTI@
        }
    }
    """
        .replace("@PQ_ONE@", pq_one_decl)
        .replace("@TGSM_DECL@", tgsm_decl)
        .replace("@REM_INIT@", rem_init)
        .replace("@TB_SETUP@", tb_setup)
        .replace("@STG_V8@", stg_v8_body)
        .replace("@STG_V4@", stg_v4_body)
        .replace("@STG_TAIL@", stg_tail_body)
        .replace("@EPI_ONE@", epi_one_src)
        .replace("@EPI_MULTI@", epi_m)
        .replace("@XOFF@", xoff_expr)
    )


def _k_body(xoff_expr, epi_src):
    return r"""
    constexpr int NFULL = KK / BLOCK;
    constexpr int TAILK = KK - NFULL * BLOCK;
    constexpr bool HAS_TAIL = (TAILK > 0);
    const bool tail_on = HAS_TAIL && ((int)lane * VPT < TAILK);
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
        for (int it = 0; it < NFULL; it++) {
            thread U wv[ROWS][VPT];
            U sv[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                dequant_vals<BITS, VPT>(ws + r * in_w, wv[r]);
                sv[r] = moe_dequant_scale<GS>((sc + r * in_g)[0]);
            }
            #pragma unroll
            for (int t = 0; t < TMAX; t++) {
                if (tb + t < cnt) {
                    thread U xt[VPT];
                    load_vector<T, U, VPT>(x + xo[t] + it * BLOCK, xt);
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        result[t][r] += dot_vals<BITS, VPT>(xt, wv[r], sv[r]);
                    }
                }
            }
            ws += BLOCK * 4 / PF;
            sc += BLOCK / GS;
        }
        if constexpr (HAS_TAIL) {
            if (tail_on) {
                thread U wv[ROWS][VPT];
                U sv[ROWS];
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    dequant_vals<BITS, VPT>(ws + r * in_w, wv[r]);
                    sv[r] = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                }
                #pragma unroll
                for (int t = 0; t < TMAX; t++) {
                    if (tb + t < cnt) {
                        thread U xt[VPT];
                        load_vector<T, U, VPT>(x + xo[t] + NFULL * BLOCK, xt);
                        #pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            result[t][r] += dot_vals<BITS, VPT>(xt, wv[r], sv[r]);
                        }
                    }
                }
            }
        }
        @EPI@
    }
    """.replace("@EPI@", epi_src).replace("@XOFF@", xoff_expr)


_CONSTS_STAGED = r"""
    constexpr int NPAIR = BB * 8;
    constexpr int PF = 32 / BITS;
    constexpr int PPT = FAST ? 2 : 1;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    typedef float U;
    typedef bfloat16_t T;
    uint3 tid = threadgroup_position_in_grid;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint ttid = sg * 32 + lane;
"""

_CONSTS = _CONSTS_STAGED + "    constexpr int ROWS = 4;\n"

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


def _early_dup_scan(ind_name):
    return r"""
    const int q = tid.z;
    const uint e = @IND@[q];
    bool dup = false;
    for (int p = (int)lane; p < q; p += 32) {
        if (@IND@[p] == e) dup = true;
    }
    if (simd_any(dup)) return;
    threadgroup uint pl[NPAIR];
    threadgroup int cnt_s[1];
    if (sg == 0) {
        int cnt = 0;
        for (int base = q; base < NPAIR; base += 32) {
            int p = base + (int)lane;
            int take = (p < NPAIR && @IND@[p] == e) ? 1 : 0;
            int pre = simd_prefix_exclusive_sum(take);
            if (take) pl[cnt + pre] = (uint)p;
            cnt += simd_sum(take);
        }
        if (lane == 0) cnt_s[0] = cnt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int cnt = cnt_s[0];
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
""".replace("@IND@", ind_name)


def _early_dup_scan_dyn(ind_name):
    return r"""
    const int q = tid.z;
    const int act_pairs = w_act[0] * 8;
    if (q >= act_pairs) return;
    const uint e = @IND@[q];
    bool dup = false;
    for (int p = (int)lane; p < q; p += 32) {
        if (@IND@[p] == e) dup = true;
    }
    if (simd_any(dup)) return;
    threadgroup uint pl[NPAIR];
    threadgroup int cnt_s[1];
    if (sg == 0) {
        int cnt = 0;
        for (int base = q; base < act_pairs; base += 32) {
            int p = base + (int)lane;
            int take = (p < act_pairs && @IND@[p] == e) ? 1 : 0;
            int pre = simd_prefix_exclusive_sum(take);
            if (take) pl[cnt + pre] = (uint)p;
            cnt += simd_sum(take);
        }
        if (lane == 0) cnt_s[0] = cnt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int cnt = cnt_s[0];
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
""".replace("@IND@", ind_name)


def _early_dup_scan_ballot(ind_name):
    return r"""
    const int q = (int)tid.z;
    const uint e = @IND@[q];
    int cnt;
    threadgroup uint pl[NPAIR];
    if constexpr (NPAIR <= 32) {
        uint32_t mask = (uint32_t)((uint64_t)simd_ballot((int)lane < NPAIR && @IND@[lane] == e));
        if (mask & ((1u << q) - 1u)) return;
        cnt = (int)metal::popcount(mask);
        if (cnt > 1) {
            if (sg == 0 && lane == 0) {
                uint32_t m = mask;
                int i = 0;
                while (m) {
                    pl[i++] = (uint)metal::ctz(m);
                    m &= (m - 1u);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        uint64_t mask = (uint64_t)simd_ballot((int)lane < NPAIR && @IND@[lane] == e) & 0xFFFFFFFFULL;
        uint64_t m1 = (uint64_t)simd_ballot((int)lane + 32 < NPAIR && @IND@[lane + 32] == e) & 0xFFFFFFFFULL;
        mask |= (m1 << 32);
        if (mask & ((1ULL << q) - 1ULL)) return;
        cnt = (int)metal::popcount(mask);
        if (cnt > 1) {
            if (sg == 0 && lane == 0) {
                uint64_t m = mask;
                int i = 0;
                while (m) {
                    pl[i++] = (uint)metal::ctz(m);
                    m &= (m - 1ULL);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
""".replace("@IND@", ind_name)


def _early_dup_scan_ballot_dyn(ind_name):
    return r"""
    const int q = (int)tid.z;
    const int act_pairs = w_act[0] * 8;
    if (q >= act_pairs) return;
    const uint e = @IND@[q];
    int cnt;
    threadgroup uint pl[NPAIR];
    if constexpr (NPAIR <= 32) {
        uint32_t mask = (uint32_t)((uint64_t)simd_ballot((int)lane < act_pairs && @IND@[lane] == e));
        if (mask & ((1u << q) - 1u)) return;
        cnt = (int)metal::popcount(mask);
        if (cnt > 1) {
            if (sg == 0 && lane == 0) {
                uint32_t m = mask;
                int i = 0;
                while (m) {
                    pl[i++] = (uint)metal::ctz(m);
                    m &= (m - 1u);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        uint64_t mask = (uint64_t)simd_ballot((int)lane < act_pairs && @IND@[lane] == e) & 0xFFFFFFFFULL;
        uint64_t m1 = (uint64_t)simd_ballot((int)lane + 32 < act_pairs && @IND@[lane + 32] == e) & 0xFFFFFFFFULL;
        mask |= (m1 << 32);
        if (mask & ((1ULL << q) - 1ULL)) return;
        cnt = (int)metal::popcount(mask);
        if (cnt > 1) {
            if (sg == 0 && lane == 0) {
                uint64_t m = mask;
                int i = 0;
                while (m) {
                    pl[i++] = (uint)metal::ctz(m);
                    m &= (m - 1ULL);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
""".replace("@IND@", ind_name)


def _gateup_src(mode, dbg, gc, uc, has_w_act=False, has_stg=False, use_ballot=False):
    epi_one = r"""
        float gv[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            U v = simd_sum(result[r]);
            if constexpr (HGS) { gv[r] = float(static_cast<T>(v * inv_scale_enc)); }
            else { gv[r] = float(static_cast<T>(v)); }
        }
        if (lane == 0) {
            #pragma unroll
            for (int pair = 0; pair < ROWS / 2; pair++) {
                sw[pq * (NN / 2) + out_row / 2 + pair] = bfloat16_t(glue_swiglu(gv[2 * pair], gv[2 * pair + 1], %s, %s));
            }
            %s
        }
""" % (gc, uc, "for (int r = 0; r < ROWS; r++) gu[pq * NN + out_row + r] = bfloat16_t(gv[r]);" if dbg else "")

    epi_multi = r"""
        #pragma unroll
        for (int t = 0; t < TMAX; t++) {
            if (t < nb) {
                float gv[ROWS];
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    U v = simd_sum(result[t][r]);
                    if constexpr (HGS) { gv[r] = float(static_cast<T>(v * inv_scale_enc)); }
                    else { gv[r] = float(static_cast<T>(v)); }
                }
                if (lane == 0) {
                    const int pq = (int)pl[tb + t];
                    #pragma unroll
                    for (int pair = 0; pair < ROWS / 2; pair++) {
                        sw[pq * (NN / 2) + out_row / 2 + pair] = bfloat16_t(glue_swiglu(gv[2 * pair], gv[2 * pair + 1], %s, %s));
                    }
                    %s
                }
            }
        }
""" % (gc, uc, "for (int r = 0; r < ROWS; r++) gu[pq * NN + out_row + r] = bfloat16_t(gv[r]);" if dbg else "")

    use_bal = bool(use_ballot and mode == "launch")
    loop = _k_body_lean("pq / 8", epi_one, epi_multi, use_mask=False, has_stg=has_stg, pq_one_q=use_bal)
    if mode == "group":
        return _CONSTS_STAGED + r"""
    const uint u = tid.z;
    if (u >= num_uniq[0]) return;
    const uint e = uniq_e[u];
    const int cnt = (int)u_cnt[u];
    const device uint* pl = pair_list + u_off[u];
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
""" + loop
    if mode == "launch":
        if use_bal:
            scan = _early_dup_scan_ballot_dyn("inds_in") if has_w_act else _early_dup_scan_ballot("inds_in")
        else:
            scan = _early_dup_scan_dyn("inds_in") if has_w_act else _early_dup_scan("inds_in")
        return _CONSTS_STAGED + scan + loop

    return _CONSTS_STAGED + r"""
    const int q = tid.z;
    threadgroup uint ind_s[NPAIR];
    threadgroup uint pl[NPAIR];
    threadgroup float wt_s[NPAIR];
    for (int b = sg; b < BB; b += NSG)
        moe_route_row(logits + b * 128, bias, lane, ind_s + b * 8, wt_s + b * 8, eps, rscale);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid.y == 0 && q == 0) {
        for (int i = ttid; i < NPAIR; i += 32 * NSG) { inds_o[i] = ind_s[i]; wts_o[i] = wt_s[i]; }
    }
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


def _down_gather_src(has_grp, has_w_act=False, has_stg=False, use_ballot=False):
    epi_one = r"""
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            U v = simd_sum(result[r]);
            if (lane == 0) {
                if constexpr (HGS) { down[pq * NN + out_row + r] = static_cast<T>(v * inv_scale_enc); }
                else { down[pq * NN + out_row + r] = static_cast<T>(v); }
            }
        }
"""
    epi_multi = r"""
        #pragma unroll
        for (int t = 0; t < TMAX; t++) {
            if (t < nb) {
                const int pq = (int)pl[tb + t];
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    U v = simd_sum(result[t][r]);
                    if (lane == 0) {
                        if constexpr (HGS) { down[pq * NN + out_row + r] = static_cast<T>(v * inv_scale_enc); }
                        else { down[pq * NN + out_row + r] = static_cast<T>(v); }
                    }
                }
            }
        }
"""
    use_bal = bool(use_ballot and not has_grp)
    loop = _k_body_lean("pq", epi_one, epi_multi, use_mask=False, has_stg=has_stg, pq_one_q=use_bal)
    if has_grp:
        return _CONSTS_STAGED + r"""
    const uint u = tid.z;
    if (u >= num_uniq[0]) return;
    const uint e = uniq_e[u];
    const int cnt = (int)u_cnt[u];
    const device uint* pl = pair_list + u_off[u];
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
""" + loop
    if use_bal:
        scan = _early_dup_scan_ballot_dyn("inds") if has_w_act else _early_dup_scan_ballot("inds")
    else:
        scan = _early_dup_scan_dyn("inds") if has_w_act else _early_dup_scan("inds")
    return _CONSTS_STAGED + scan + loop


def _combine_conv_src(dbg, scale_lit):
    s = r"""
    uint b = threadgroup_position_in_grid.y;
    uint c = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    threadgroup float ww[8];
    if (lid < 8) ww[lid] = wts[b * 8 + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c >= CC) return;
    float ev[8], wv[8];
    for (int j = 0; j < 8; j++) {
        ev[j] = float(down[(b * 8 + j) * CC + c]);
        wv[j] = ww[j];
    }
    bfloat16_t moe = bfloat16_t(glue_moe_combine8(ev, wv));
    float x0 = float(moe);
    const device float* st = cs3 + b * 3 * CC;
    float r1 = st[CC + c], r2 = st[2 * CC + c];
    float s0 = float(bfloat16_t(st[c])), s1 = float(bfloat16_t(r1)), s2 = float(bfloat16_t(r2));
    float acc = 0.0f;
    acc = fma(float(cw[c * 4 + 0]), s0, acc);
    acc = fma(float(cw[c * 4 + 1]), s1, acc);
    acc = fma(float(cw[c * 4 + 2]), s2, acc);
    acc = fma(float(cw[c * 4 + 3]), x0, acc);
    float conv = float(bfloat16_t(acc));
    bfloat16_t pmc_v = bfloat16_t((conv + x0) / @SCALE@);
    w3[b * 3 * CC + c] = r1;
    w3[b * 3 * CC + CC + c] = r2;
    w3[b * 3 * CC + 2 * CC + c] = x0;
    out[b * CC + c] = bfloat16_t(resid[b * CC + c] + float(pmc_v));
"""
    if dbg:
        s += r"""
    moe_out[b * CC + c] = moe;
    pmc[b * CC + c] = pmc_v;
"""
    return s.replace("@SCALE@", scale_lit)


def _combine_conv_causal_src(dbg, scale_lit, has_w_act=False):
    w_lim = "w_a" if has_w_act else "BB"
    w_decl = "\n    const int w_a = w_act[0];" if has_w_act else ""
    s = r"""
    uint c = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    threadgroup float ww[128];""" + w_decl + r"""
    for (int i = (int)lid; i < @WLIM@ * 8; i += 64) ww[i] = wts[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)c >= CC) return;
    float w0c = float(cw[c * 4 + 0]);
    float w1c = float(cw[c * 4 + 1]);
    float w2c = float(cw[c * 4 + 2]);
    float w3c = float(cw[c * 4 + 3]);
    float moe_x[BB];
    for (int b = 0; b < @WLIM@; b++) {
        float ev[8], wv[8];
        for (int j = 0; j < 8; j++) {
            ev[j] = float(down[(b * 8 + j) * CC + c]);
            wv[j] = ww[b * 8 + j];
        }
        moe_x[b] = float(bfloat16_t(glue_moe_combine8(ev, wv)));
    }
    for (int b = 0; b < @WLIM@; b++) {
        int a0 = conv_anc[b * 3 + 0];
        int a1 = conv_anc[b * 3 + 1];
        int a2 = conv_anc[b * 3 + 2];
        float r0 = (a0 >= 0) ? moe_x[a0] : cs3[(-1 - a0) * CC + c];
        float r1 = (a1 >= 0) ? moe_x[a1] : cs3[(-1 - a1) * CC + c];
        float r2 = (a2 >= 0) ? moe_x[a2] : cs3[(-1 - a2) * CC + c];
        float x0 = moe_x[b];
        float s0 = float(bfloat16_t(r0)), s1 = float(bfloat16_t(r1)), s2 = float(bfloat16_t(r2));
        float acc = 0.0f;
        acc = fma(w0c, s0, acc);
        acc = fma(w1c, s1, acc);
        acc = fma(w2c, s2, acc);
        acc = fma(w3c, x0, acc);
        float conv = float(bfloat16_t(acc));
        bfloat16_t pmc_v = bfloat16_t((conv + x0) / @SCALE@);
        w3[(b * 3 + 0) * CC + c] = r1;
        w3[(b * 3 + 1) * CC + c] = r2;
        w3[(b * 3 + 2) * CC + c] = x0;
        out[b * CC + c] = bfloat16_t(resid[b * CC + c] + float(pmc_v));
"""
    if dbg:
        s += r"""
        moe_out[b * CC + c] = bfloat16_t(x0);
        pmc[b * CC + c] = pmc_v;
"""
    s += r"""
    }
"""
    return s.replace("@SCALE@", scale_lit).replace("@WLIM@", w_lim)


def _down_fused_src(has_group, dbg, scale_lit):
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
    if has_group:
        loop = _k_body("pq", epi).replace("PL[", "pl[").replace("PQ0", "pl[0]")
        s = _CONSTS + r"""
    threadgroup float dn_s[NPAIR * 4];
    const int nu = (int)num_uniq[0];
    const int out_row = tid.y * ROWS;
    for (int u = (int)sg; u < nu; u += NSG) {
        const uint e = uniq_e[u];
        const int cnt = (int)u_cnt[u];
        const device uint* pl = pair_list + u_off[u];
""" + loop + r"""
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
    else:
        loop = _k_body("pq", epi).replace("PL[", "plg[").replace("PQ0", "P0")
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
"""
    s += r"""
    if (ttid < BB * 4) {
        const int b = ttid / 4, r = ttid % 4;
        const int c = tid.y * 4 + r;
        float ev[8], ww[8];
        for (int j = 0; j < 8; j++) { ev[j] = dn_s[(b * 8 + j) * 4 + r]; ww[j] = wts[b * 8 + j]; }
        bfloat16_t moe = bfloat16_t(glue_moe_combine8(ev, ww));
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


def _gateup_b1_src(nsg_b1, dbg, gc, uc, ec, rc):
    if nsg_b1 == 1:
        route_blk = f"""
    uint e;
    if (tid.y == 0 && q == 0) {{
        uint ind_s[8];
        float wt_s[8];
        moe_route_row(logits, bias, lane, ind_s, wt_s, {ec}, {rc});
        if (lane == 0) {{
            for (int j = 0; j < 8; j++) {{ inds_o[j] = ind_s[j]; wts_o[j] = wt_s[j]; }}
        }}
        e = simd_broadcast(ind_s[0], 0);
    }} else {{
        e = moe_route_row_q(logits, bias, lane, q);
    }}
"""
    else:
        route_blk = f"""
    threadgroup uint ind_s[8];
    threadgroup float wt_s[8];
    threadgroup uint e_s[1];
    if (sg == 0) {{
        if (tid.y == 0 && q == 0) {{
            moe_route_row(logits, bias, lane, ind_s, wt_s, {ec}, {rc});
            if (lane < 8) {{ inds_o[lane] = ind_s[lane]; wts_o[lane] = wt_s[lane]; }}
            if (lane == 0) e_s[0] = ind_s[0];
        }} else {{
            uint w_e = moe_route_row_q(logits, bias, lane, q);
            if (lane == 0) e_s[0] = w_e;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint e = e_s[0];
"""
    dbg_store = "for (int r = 0; r < ROWS; r++) gu[q * NN + out_row + r] = bfloat16_t(gv[r]);" if dbg else ""
    return f"""
    constexpr int PF = 32 / BITS;
    constexpr int PPT = FAST ? 2 : 1;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    constexpr int NFULL = KK / BLOCK;
    constexpr int TAILK = KK - NFULL * BLOCK;
    constexpr bool HAS_TAIL = (TAILK > 0);
    typedef float U;
    typedef bfloat16_t T;
    uint3 tid = threadgroup_position_in_grid;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    const int q = tid.z;
{route_blk}
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
    const int in_w = KK * BITS / 8;
    const int in_g = KK / GS;
    const device uint8_t* ws = (const device uint8_t*)w + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
    const device uint8_t* sc = scales + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
    const device T* xp = x + lane * VPT;
    float inv_scale_enc = 1.0f;
    if constexpr (HGS) inv_scale_enc = float(gsc[e]) / (F8E4M3_MAX * F4E2M1_MAX);
    thread U result[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) result[r] = 0;
    for (int it = 0; it < NFULL; it++) {{
        const uint k_off = (uint)it * (uint)BLOCK;
        if constexpr (VPT == 8) {{
            uint4 x4 = ((const device uint4*)(xp + k_off))[0];
            float4 xf0 = float4(as_type<bfloat4>(x4.xy));
            float4 xf1 = float4(as_type<bfloat4>(x4.zw));
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
                    dequant_u32_fp4_8_b1(wr[r], wv);
                    U sv = moe_dequant_scale<GS>(sr[r]);
                    result[r] += dot_vals_f4_8<4>(xf0, xf1, wv, sv);
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
                    result[r] += dot_vals_f4_8<8>(xf0, xf1, wv, sv);
                }}
            }}
        }} else if constexpr (VPT == 4) {{
            uint2 x2 = ((const device uint2*)(xp + k_off))[0];
            float4 xf0 = float4(as_type<bfloat4>(x2));
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {{
                thread U wv[VPT];
                dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                result[r] += dot_vals_f4_4<BITS>(xf0, wv, sv);
            }}
        }} else {{
            thread U xt[VPT];
            load_vector<T, U, VPT>(xp + k_off, xt);
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {{
                U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                result[r] += qdot<U, VPT, BITS>(ws + r * in_w, xt, sv);
            }}
        }}
        ws += BLOCK * 4 / PF;
        sc += BLOCK / GS;
    }}
    if constexpr (HAS_TAIL) {{
        if ((int)lane * VPT < TAILK) {{
            const uint k_off = (uint)NFULL * (uint)BLOCK;
            if constexpr (VPT == 8) {{
                uint4 x4 = ((const device uint4*)(xp + k_off))[0];
                float4 xf0 = float4(as_type<bfloat4>(x4.xy));
                float4 xf1 = float4(as_type<bfloat4>(x4.zw));
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
                        dequant_u32_fp4_8_b1(wr[r], wv);
                        U sv = moe_dequant_scale<GS>(sr[r]);
                        result[r] += dot_vals_f4_8<4>(xf0, xf1, wv, sv);
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
                        result[r] += dot_vals_f4_8<8>(xf0, xf1, wv, sv);
                    }}
                }}
            }} else if constexpr (VPT == 4) {{
                uint2 x2 = ((const device uint2*)(xp + k_off))[0];
                float4 xf0 = float4(as_type<bfloat4>(x2));
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {{
                    thread U wv[VPT];
                    dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    result[r] += dot_vals_f4_4<BITS>(xf0, wv, sv);
                }}
            }} else {{
                thread U xt[VPT];
                load_vector<T, U, VPT>(xp + k_off, xt);
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {{
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    result[r] += qdot<U, VPT, BITS>(ws + r * in_w, xt, sv);
                }}
            }}
        }}
    }}
    float gv[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {{
        U v = simd_sum(result[r]);
        if constexpr (HGS) gv[r] = float(static_cast<T>(v * inv_scale_enc));
        else gv[r] = float(static_cast<T>(v));
    }}
    if (lane == 0) {{
        #pragma unroll
        for (int pair = 0; pair < ROWS / 2; pair++) {{
            sw[q * (NN / 2) + out_row / 2 + pair] = bfloat16_t(glue_swiglu(gv[2 * pair], gv[2 * pair + 1], {gc}, {uc}));
        }}
        {dbg_store}
    }}
"""


def _down_b1_src(dbg, scale_lit):
    dbg_comb = "moe_out[c] = moe; pmc[c] = pmc_v;" if dbg else ""
    dbg_dn = (
        "for (int i = int(ttid); i < 8 * ROWS; i += 32 * NSG) "
        "down[(i / ROWS) * CC + out_row + (i % ROWS)] = bfloat16_t(dn_s[i]);"
        if dbg else ""
    )
    return f"""
    constexpr int PF = 32 / BITS;
    constexpr int PPT = FAST ? 2 : 1;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    constexpr int NFULL = KK / BLOCK;
    constexpr int TAILK = KK - NFULL * BLOCK;
    constexpr bool HAS_TAIL = (TAILK > 0);
    typedef float U;
    typedef bfloat16_t T;
    uint3 tid = threadgroup_position_in_grid;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint ttid = sg * 32 + lane;
    threadgroup float dn_s[8 * ROWS];
    const int out_row = tid.y * ROWS;
    const int in_w = KK * BITS / 8;
    const int in_g = KK / GS;
    const bool tail_on = HAS_TAIL && ((int)lane * VPT < TAILK);
    for (int P0 = sg; P0 < 8; P0 += NSG) {{
        const uint e = inds[P0];
        const device uint8_t* ws = (const device uint8_t*)w + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
        const device uint8_t* sc = scales + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
        const device T* xp = x + (long)P0 * KK + lane * VPT;
        float inv_scale_enc = 1.0f;
        if constexpr (HGS) inv_scale_enc = float(gsc[e]) / (F8E4M3_MAX * F4E2M1_MAX);
        thread U result[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) result[r] = 0;
        for (int it = 0; it < NFULL; it++) {{
            const uint k_off = (uint)it * (uint)BLOCK;
            if constexpr (VPT == 8) {{
                uint4 x4 = ((const device uint4*)(xp + k_off))[0];
                float4 xf0 = float4(as_type<bfloat4>(x4.xy));
                float4 xf1 = float4(as_type<bfloat4>(x4.zw));
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
                        dequant_u32_fp4_8_b1(wr[r], wv);
                        U sv = moe_dequant_scale<GS>(sr[r]);
                        result[r] += dot_vals_f4_8<4>(xf0, xf1, wv, sv);
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
                        result[r] += dot_vals_f4_8<8>(xf0, xf1, wv, sv);
                    }}
                }}
            }} else if constexpr (VPT == 4) {{
                uint2 x2 = ((const device uint2*)(xp + k_off))[0];
                float4 xf0 = float4(as_type<bfloat4>(x2));
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {{
                    thread U wv[VPT];
                    dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    result[r] += dot_vals_f4_4<BITS>(xf0, wv, sv);
                }}
            }} else {{
                thread U xt[VPT];
                load_vector<T, U, VPT>(xp + k_off, xt);
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {{
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    result[r] += qdot<U, VPT, BITS>(ws + r * in_w, xt, sv);
                }}
            }}
            ws += BLOCK * 4 / PF;
            sc += BLOCK / GS;
        }}
        if constexpr (HAS_TAIL) {{
            if (tail_on) {{
                const uint k_off = (uint)NFULL * (uint)BLOCK;
                if constexpr (VPT == 8) {{
                    uint4 x4 = ((const device uint4*)(xp + k_off))[0];
                    float4 xf0 = float4(as_type<bfloat4>(x4.xy));
                    float4 xf1 = float4(as_type<bfloat4>(x4.zw));
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {{
                        thread U wv[VPT];
                        dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        result[r] += dot_vals_f4_8<BITS>(xf0, xf1, wv, sv);
                    }}
                }} else if constexpr (VPT == 4) {{
                    uint2 x2 = ((const device uint2*)(xp + k_off))[0];
                    float4 xf0 = float4(as_type<bfloat4>(x2));
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {{
                        thread U wv[VPT];
                        dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        result[r] += dot_vals_f4_4<BITS>(xf0, wv, sv);
                    }}
                }} else {{
                    thread U xt[VPT];
                    load_vector<T, U, VPT>(xp + k_off, xt);
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {{
                        U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                        result[r] += qdot<U, VPT, BITS>(ws + r * in_w, xt, sv);
                    }}
                }}
            }}
        }}
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {{
            U v = simd_sum(result[r]);
            if (lane == 0) {{
                if constexpr (HGS) dn_s[P0 * ROWS + r] = float(static_cast<T>(v * inv_scale_enc));
                else dn_s[P0 * ROWS + r] = float(static_cast<T>(v));
            }}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ttid < ROWS) {{
        const int r = ttid;
        const int c = out_row + r;
        float ev[8], ww[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {{ ev[j] = dn_s[j * ROWS + r]; ww[j] = wts[j]; }}
        bfloat16_t moe = bfloat16_t(glue_moe_combine8(ev, ww));
        float x0 = float(moe);
        float r1 = cs3[CC + c], r2 = cs3[2 * CC + c];
        float s0 = float(T(cs3[c])), s1 = float(T(r1)), s2 = float(T(r2));
        float acc = 0.0f;
        acc = fma(float(cw[c * 4 + 0]), s0, acc);
        acc = fma(float(cw[c * 4 + 1]), s1, acc);
        acc = fma(float(cw[c * 4 + 2]), s2, acc);
        acc = fma(float(cw[c * 4 + 3]), x0, acc);
        float conv = float(T(acc));
        T pmc_v = T((conv + x0) / {scale_lit});
        w3[c] = r1; w3[CC + c] = r2; w3[2 * CC + c] = x0;
        out[c] = bfloat16_t(resid[c] + float(pmc_v));
        {dbg_comb}
    }}
    {dbg_dn}
"""


def _geom(w, scales, K, mode, group_size, bits):
    N = w.shape[1]
    if mode not in ("nvfp4", "mxfp8"):
        raise ValueError("moe: mode must be nvfp4 or mxfp8, got %r" % (mode,))
    if K % 32 != 0 or K != scales.shape[-1] * group_size or w.shape[2] * 32 // bits != K or N % 8 != 0:
        raise ValueError("moe: unsupported shape")
    fast = K % ((32 // bits) * 2 * 32) == 0
    return N, fast


def _tmax(B, cap=None):
    c = TMAX_CAP if cap is None else cap
    for t in (1, 2, 3, 4, 6, 8, 16):
        if B <= t or t >= c:
            return t
    return 16


def _stg_x(tmax):
    ov = os.environ.get("MOE_STG_X")
    if ov is not None:
        return ov not in ("0", "", "false", "False")
    return False


def _stream_x(tmax):
    if _stg_x(tmax):
        return False
    ov = os.environ.get("MOE_STREAM_X")
    if ov is not None:
        return ov not in ("0", "", "false", "False")
    return False


_ROUTE_TILE_SRC = r"""
    constexpr uint NSG = 8;
    constexpr uint EE = 128;
    constexpr uint PP = uint(BB) * 8u;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    threadgroup uint ind_s[PP];
    threadgroup float wt_s[PP];
    threadgroup atomic_uint hist[NSG][EE];
    threadgroup uint exp_tot[EE];
    threadgroup uint tile_cnt[EE];
    threadgroup uint sg_sums[4];
    threadgroup uint sg_tsums[4];

    for (int row = (int)sg; row < BB; row += 8) {
        moe_route_row(logits + row * 128, bias, lane, ind_s + row * 8, wt_s + row * 8, eps, rscale);
    }
    for (uint i = tid; i < NSG * EE; i += 256) {
        atomic_store_explicit(&hist[i / EE][i % EE], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < PP; i += 256) {
        inds_o[i] = ind_s[i];
        wts_o[i] = wt_s[i];
    }

    uint chunk = (PP + NSG - 1u) / NSG;
    uint s0 = sg * chunk;
    uint s1 = min(s0 + chunk, PP);
    for (uint p = s0 + lane; p < s1; p += 32) {
        uint e = ind_s[p];
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
        tile_cnt[tid] = (acc + uint(BM) - 1u) / uint(BM);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < EE) {
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
        uint sv = (tid < 4) ? sg_sums[tid] : 0u;
        uint sp = simd_prefix_exclusive_sum(sv);
        if (tid < 4) sg_sums[tid] = sp;

        uint stv = (tid < 4) ? sg_tsums[tid] : 0u;
        uint stp = simd_prefix_exclusive_sum(stv);
        if (tid < 4) sg_tsums[tid] = stp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < EE) {
        uint base = exp_tot[tid] + sg_sums[sg];
        exp_off[tid] = base;
        for (uint s = 0; s < NSG; s++) {
            uint off = atomic_load_explicit(&hist[s][tid], memory_order_relaxed);
            atomic_store_explicit(&hist[s][tid], base + off, memory_order_relaxed);
        }
        uint tbase = tile_cnt[tid] + sg_tsums[sg];
        uint nxt = (tid + 1u == EE) ? PP : (exp_tot[tid + 1u] + sg_sums[(tid + 1u) >> 5]);
        uint nte = (nxt - base + uint(BM) - 1u) / uint(BM);
        for (uint mt = 0; mt < nte; mt++) {
            tile_meta[tbase + mt] = (tid << 20) | mt;
        }
        if (tid == EE - 1u) {
            exp_off[EE] = PP;
            uint tot_t = tbase + nte;
            for (uint t = tot_t; t < uint(MAX_TILES); t++) {
                tile_meta[t] = 0xFFFFFFFFu;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = s0 + lane; p < s1; p += 32) {
        uint e = ind_s[p];
        uint pos = atomic_fetch_add_explicit(&hist[sg][e], 1u, memory_order_relaxed);
        pair_list[pos] = p;
    }
"""

_rt_kernel = None


def route_tile_rows(logits, bias, *, score_eps=1e-20, route_scale=1.0, tmax=4):
    """K11t: fused route + counting-sort + tile_meta worklist for 1 <= B <= 32, E == 128."""
    global _rt_kernel
    B = int(logits.shape[0])
    if B < 1 or B > 32 or logits.shape != (B, 128) or logits.dtype != mx.float32:
        raise ValueError("route_tile_rows: logits must be (B,128) f32, B in 1..32")
    if bias.dtype != mx.float32:
        bias = bias.astype(mx.float32)
    if _rt_kernel is None:
        _rt_kernel = mx.fast.metal_kernel(
            name="moe_route_tile",
            input_names=["logits", "bias", "eps", "rscale"],
            output_names=["inds_o", "wts_o", "pair_list", "exp_off", "tile_meta"],
            source=_ROUTE_TILE_SRC,
            header=HEADER,
            ensure_row_contiguous=True,
        )
    P = B * 8
    max_tiles = min(P, (P + (tmax - 1) * 128) // tmax)
    return list(_rt_kernel(
        inputs=[logits, bias, scalar(score_eps), scalar(route_scale)],
        template=[("BB", B), ("BM", tmax), ("MAX_TILES", max_tiles)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, 8), (B, 8), (P,), (129,), (max_tiles,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.uint32, mx.uint32],
    ))


def _tiled_moe_src(is_gu, dbg, gc="", uc=""):
    xoff_expr = "(pq >> 3)" if is_gu else "pq"
    if is_gu:
        dbg_s = (
            "if constexpr (ROWS == 4) { ((device uint2*)(gu + (size_t)pq * NN + (size_t)out_row))[0] = "
            "as_type<uint2>(bfloat4(bfloat16_t(gv[0]), bfloat16_t(gv[1]), bfloat16_t(gv[2]), bfloat16_t(gv[3]))); } "
            "else { for (int r = 0; r < ROWS; r++) gu[(size_t)pq * NN + out_row + r] = bfloat16_t(gv[r]); }"
            if dbg else ""
        )
        emit_code = f"""
            float gv[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {{
                U v = simd_sum(res[r]);
                if constexpr (HGS) gv[r] = float(static_cast<T>(v * inv_scale_enc));
                else gv[r] = float(static_cast<T>(v));
            }}
            if (lane == 0) {{
                if constexpr (ROWS == 4) {{
                    bfloat2 ov = bfloat2(
                        bfloat16_t(glue_swiglu(gv[0], gv[1], {gc}, {uc})),
                        bfloat16_t(glue_swiglu(gv[2], gv[3], {gc}, {uc})));
                    ((device uint*)(sw + (size_t)pq * (NN / 2) + (size_t)(out_row / 2)))[0] = as_type<uint>(ov);
                }} else {{
                    #pragma unroll
                    for (int pair = 0; pair < ROWS / 2; pair++) {{
                        sw[(size_t)pq * (NN / 2) + out_row / 2 + pair] = bfloat16_t(glue_swiglu(gv[2 * pair], gv[2 * pair + 1], {gc}, {uc}));
                    }}
                }}
                {dbg_s}
            }}"""
    else:
        emit_code = """
            T ov[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                U v = simd_sum(res[r]);
                if constexpr (HGS) ov[r] = static_cast<T>(v * inv_scale_enc);
                else ov[r] = static_cast<T>(v);
            }
            if (lane == 0) {
                if constexpr (ROWS == 4) {
                    ((device uint2*)(down + (size_t)pq * NN + (size_t)out_row))[0] = as_type<uint2>(bfloat4(ov[0], ov[1], ov[2], ov[3]));
                } else {
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) down[(size_t)pq * NN + out_row + r] = ov[r];
                }
            }"""

    def k_step_one():
        return """
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
                        res[r] += dot_vals_f4_8<4>(xf0, xf1, wv, sv);
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
                        res[r] += dot_vals_f4_8<8>(xf0, xf1, wv, sv);
                    }
                }
            } else if constexpr (VPT == 4) {
                uint2 x2 = ((const device uint2*)(x + xo + k_off))[0];
                float4 xf0 = float4(as_type<bfloat4>(x2));
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    thread U wv[VPT];
                    dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    res[r] += dot_vals_f4_4<BITS>(xf0, wv, sv);
                }
            } else {
                thread U xt[VPT];
                load_vector<T, U, VPT>(x + xo + k_off, xt);
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    thread U wv[VPT];
                    dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    res[r] += dot_vals<BITS, VPT>(xt, wv, sv);
                }
            }"""

    def k_step_multi():
        return """
            if constexpr (VPT == 8) {
                thread uint4 x4[TMAX];
                #pragma unroll
                for (int t = 0; t < TMAX; t++) {
                    if (TMAX == 2 || t < nb) x4[t] = ((const device uint4*)(x + xo[t] + k_off))[0];
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
                            if (TMAX == 2 || t < nb) result[t][r] += dot_vals_bf16_u4val<4>(x4[t], wv, sv);
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
                            if (TMAX == 2 || t < nb) result[t][r] += dot_vals_bf16_u4val<8>(x4[t], wv, sv);
                        }
                    }
                }
            } else if constexpr (VPT == 4) {
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
            } else {
                thread T xb[TMAX][VPT];
                #pragma unroll
                for (int t = 0; t < TMAX; t++) {
                    if (t < nb) {
                        const device T* xp = x + xo[t] + k_off;
                        for (int i = 0; i < VPT; i++) xb[t][i] = xp[i];
                    }
                }
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    thread U wv[VPT];
                    dequant_vals<BITS, VPT>(ws + r * in_w, wv);
                    U sv = moe_dequant_scale<GS>((sc + r * in_g)[0]);
                    #pragma unroll
                    for (int t = 0; t < TMAX; t++) {
                        if (t < nb) result[t][r] += dot_vals_bf16<BITS, VPT>(xb[t], wv, sv);
                    }
                }
            }"""

    s_one = k_step_one()
    s_mul = k_step_multi()
    return f"""
    constexpr int PF = 32 / BITS;
    constexpr int PPT = FAST ? 2 : 1;
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
    uint3 tid = threadgroup_position_in_grid;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    const uint tile_id = tid.y;
    const uint n_blk = tid.z;
    const uint meta = tile_meta[tile_id];
    if (meta == 0xFFFFFFFFu) return;
    const uint e = meta >> 20;
    const uint mt = meta & 0xFFFFFu;
    const uint m_base = exp_off[e] + mt * uint(TMAX);
    const uint m_end = exp_off[e + 1];
    const int nb = (int)min(uint(TMAX), m_end - m_base);
    const int out_row = (int)n_blk * (ROWS * NSG) + (int)sg * ROWS;
    const bool tail_on = HAS_TAIL && ((int)lane * VPT < TAILK);

    const device uint8_t* ws = (const device uint8_t*)w + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
    const device uint8_t* sc = scales + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
    float inv_scale_enc = 1.0f;
    if constexpr (HGS) inv_scale_enc = gsc[e] / (F8E4M3_MAX * F4E2M1_MAX);

    if (nb == 1) {{
        const uint pq = pair_list[m_base];
        const long xo = (long)({xoff_expr}) * KK + (long)lane * VPT;
        thread U res[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) res[r] = 0.0f;
        for (int it = 0; it < NFULL; it++) {{
            const uint k_off = (uint)it * (uint)BLOCK;
{s_one}
            ws += BLOCK * 4 / PF;
            sc += BLOCK / GS;
        }}
        if constexpr (HAS_TAIL) {{
            if (tail_on) {{
                const uint k_off = (uint)NFULL * (uint)BLOCK;
{s_one}
            }}
        }}
{emit_code}
    }} else {{
        uint pqs[TMAX];
        uint xo[TMAX];
        #pragma unroll
        for (int t = 0; t < TMAX; t++) {{
            if (t < nb) {{
                uint pq = pair_list[m_base + uint(t)];
                pqs[t] = pq;
                xo[t] = (uint)({xoff_expr}) * (uint)KK + lane * (uint)VPT;
            }} else {{
                pqs[t] = 0u;
                xo[t] = 0u;
            }}
        }}
        thread U result[TMAX][ROWS];
        #pragma unroll
        for (int t = 0; t < TMAX; t++)
            #pragma unroll
            for (int r = 0; r < ROWS; r++) result[t][r] = 0.0f;
        for (int it = 0; it < NFULL; it++) {{
            const uint k_off = (uint)it * (uint)BLOCK;
{s_mul}
            ws += BLOCK * 4 / PF;
            sc += BLOCK / GS;
        }}
        if constexpr (HAS_TAIL) {{
            if (tail_on) {{
                const uint k_off = (uint)NFULL * (uint)BLOCK;
{s_mul}
            }}
        }}
        #pragma unroll
        for (int t = 0; t < TMAX; t++) {{
            if (TMAX == 2 || t < nb) {{
                const uint pq = pqs[t];
                thread U res[ROWS];
                #pragma unroll
                for (int r = 0; r < ROWS; r++) res[r] = result[t][r];
{emit_code}
            }}
        }}
    }}
"""


def _tile_cfg(N, B):
    tmax_env = os.environ.get("MOE_TILE_TMAX")
    tmax = int(tmax_env) if tmax_env else (2 if B <= 20 else 3)
    rows = int(os.environ.get("MOE_TILE_ROWS", "4"))
    nsg_env = os.environ.get("MOE_TILE_NSG")
    if nsg_env:
        nsg = int(nsg_env)
    else:
        nsg = (8 if (N % (rows * 8) == 0) else 4) if B <= 20 else 4
    if rows % 2 != 0 or N % (rows * nsg) != 0:
        rows = 2
        nsg = 4
    return tmax, rows, nsg


def gateup_rows(logits, bias, x, w, scales, global_scale, *, mode, group_size, bits, route_scale, gate_softcap,
                up_softcap, score_eps=1e-20, route="kernel", inds=None, wts=None, grp=None, tile=None, debug=False, nsg=None, rows=None, w_act=None):
    """K12. logits (B,128) f32, bias (128,) f32, x (B,K) bf16, expert weights (X,N,K*bits/32) u32 / scales u8 /
    global_scale (X,) or None.
    route: 'kernel' (routing inside TG), 'launch' (expert_routing first), 'group' (route_group_rows), or 'tile'
    (fused route_tile_rows + tile_meta-sharded grouped QMV for B=1..32).
    Returns [expert_inds (B,8) u32, expert_w (B,8) f32, swiglu (B,8,N/2) bf16 (, gate_up (B,8,N) bf16 if debug)]."""
    B, K = x.shape
    if B > 32 or x.dtype != mx.bfloat16:
        raise ValueError("gateup_rows: B <= 32, x bf16")
    N, fast = _geom(w, scales, K, mode, group_size, bits)
    if route not in ("kernel", "launch", "group", "tile"):
        raise ValueError("route must be 'kernel', 'launch', 'group', or 'tile'")

    hgs = global_scale is not None
    if route == "tile":
        tmax_t, rows_t, nsg_t = _tile_cfg(N, B)
        if rows is not None and rows % 2 == 0 and N % (rows * nsg_t) == 0:
            rows_t = rows
        if nsg is not None and N % (rows_t * nsg) == 0:
            nsg_t = nsg
        wts_arr, inds_arr = wts, inds
        if tile is None:
            rt = route_tile_rows(logits, bias, score_eps=score_eps, route_scale=route_scale, tmax=tmax_t)
            inds_arr, wts_arr = rt[0], rt[1]
            tile = (rt[2], rt[3], rt[4], tmax_t)
        pair_list, exp_off, tile_meta, tmax_t = tile
        max_tiles = int(tile_meta.shape[0])
        gsc = (global_scale if global_scale.dtype == mx.float32 else global_scale.astype(mx.float32)) if hgs else _ZERO_F32
        key = ("gu_tile", debug, gate_softcap, up_softcap)
        k = _kernels.get(key)
        if k is None:
            gc, uc = flit(gate_softcap), flit(up_softcap)
            outs = ["sw"] + (["gu"] if debug else [])
            k = _kernels[key] = mx.fast.metal_kernel(
                name="moe_gu_tile_%d_%s_%s" % (debug, gc.replace(".", "p"), uc.replace(".", "p")),
                input_names=["w", "scales", "gsc", "x", "pair_list", "exp_off", "tile_meta"],
                output_names=outs,
                source=_tiled_moe_src(True, debug, gc, uc),
                header=HEADER,
                ensure_row_contiguous=True,
            )
        shapes = [(B, 8, N // 2)] + ([(B, 8, N)] if debug else [])
        dts = [mx.bfloat16] + ([mx.bfloat16] if debug else [])
        outs = list(k(
            inputs=[w, scales, gsc, x, pair_list, exp_off, tile_meta],
            template=[("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast),
                      ("TMAX", tmax_t), ("KK", K), ("NN", N), ("NSG", nsg_t), ("ROWS", rows_t)],
            grid=(32, nsg_t * max_tiles, N // (rows_t * nsg_t)),
            threadgroup=(32, nsg_t, 1),
            output_shapes=shapes,
            output_dtypes=dts,
        ))
        return [inds_arr, wts_arr] + outs
    if MOE_B1_FAST and B == 1 and route == "kernel" and logits.shape[-1] == 128:
        nsg_b1 = nsg or MOE_B1_NSG_GU
        rows_b1 = rows or MOE_B1_ROWS_GU
        if rows_b1 % 2 == 0 and N % (rows_b1 * nsg_b1) == 0:
            gsc = global_scale if hgs else _ZERO_F32
            if bias.dtype != mx.float32:
                bias = bias.astype(mx.float32)
            key = ("gub1", nsg_b1, debug, gate_softcap, up_softcap, score_eps, route_scale)
            k = _kernels.get(key)
            if k is None:
                gc, uc, ec, rc = flit(gate_softcap), flit(up_softcap), flit(score_eps), flit(route_scale)
                outs = ["inds_o", "wts_o", "sw"] + (["gu"] if debug else [])
                k = _kernels[key] = mx.fast.metal_kernel(
                    name="moe_gu_b1_%d_%d_%s_%s" % (nsg_b1, debug, gc.replace(".", "p"), uc.replace(".", "p")),
                    input_names=["w", "scales", "gsc", "x", "logits", "bias"],
                    output_names=outs,
                    source=_gateup_b1_src(nsg_b1, debug, gc, uc, ec, rc),
                    header=HEADER,
                    ensure_row_contiguous=True,
                )
            shapes = [(1, 8), (1, 8), (1, 8, N // 2)] + ([(1, 8, N)] if debug else [])
            dts = [mx.uint32, mx.float32, mx.bfloat16] + ([mx.bfloat16] if debug else [])
            return list(k(
                inputs=[w, scales, gsc, x, logits, bias],
                template=[("GT", gsc.dtype), ("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast),
                          ("KK", K), ("NN", N), ("NSG", nsg_b1), ("ROWS", rows_b1)],
                grid=(32, nsg_b1 * (N // (rows_b1 * nsg_b1)), 8),
                threadgroup=(32, nsg_b1, 1),
                output_shapes=shapes,
                output_dtypes=dts,
            ))

    nsg = nsg or NSG_GU
    rows = rows or ROWS_GU
    if rows % 2 != 0 or N % (rows * nsg) != 0:
        raise ValueError("N not divisible by the tile or rows odd")

    wts_arr = wts
    inds_arr = inds
    if route == "group":
        if grp is None:
            r = route_group_rows(logits, bias, score_eps=score_eps, route_scale=route_scale)
            inds_arr, wts_arr = r[0], r[1]
            grp = r[2:]
        num_uniq, uniq_e, u_off, u_cnt, pair_list = grp
    elif route == "launch" and inds_arr is None:
        if logits.shape[-1] == 128:
            inds_arr, wts_arr = route_fast_rows(logits, bias, score_eps=score_eps, route_scale=route_scale, w_act=w_act)
        else:
            inds_arr, wts_arr = expert_routing(logits, bias, top_k=8, multiplier=1.0, score_eps=score_eps,
                                               route_scale=route_scale)
            inds_arr = inds_arr.reshape(B, 8)
            wts_arr = wts_arr.reshape(B, 8)

    gsc = (global_scale if global_scale.dtype == mx.float32 else global_scale.astype(mx.float32)) if hgs else _ZERO_F32
    tmax_gu = _tmax(B, TMAX_CAP_GU)
    stream_x = _stream_x(tmax_gu)
    stg_x = _stg_x(tmax_gu)
    has_wa = (w_act is not None and route == "launch")
    use_ballot = (B <= 8 and route == "launch" and os.environ.get("MOE_BALLOT_SCAN", "1") not in ("0", ""))
    key = ("gu6", route, debug, gate_softcap, up_softcap, has_wa, stg_x, use_ballot)
    k = _kernels.get(key)
    if k is None:
        gc, uc = flit(gate_softcap), flit(up_softcap)
        if route == "group":
            ins = ["w", "scales", "gsc", "x", "num_uniq", "uniq_e", "u_off", "u_cnt", "pair_list"]
            outs = ["sw"] + (["gu"] if debug else [])
        else:
            ins = ["w", "scales", "gsc", "x"] + (["logits", "bias", "eps", "rscale"] if route == "kernel" else ["inds_in"])
            if has_wa:
                ins.append("w_act")
            outs = (["inds_o", "wts_o"] if route == "kernel" else []) + ["sw"] + (["gu"] if debug else [])
        k = _kernels[key] = mx.fast.metal_kernel(
            name="moe6_gateup_%s_%d_%d_%d_%d_%s_%s" % (route, debug, int(has_wa), int(stg_x), int(use_ballot), gc.replace(".", "p"), uc.replace(".", "p")),
            input_names=ins, output_names=outs, source=_gateup_src(route, debug, gc, uc, has_w_act=has_wa, has_stg=stg_x, use_ballot=use_ballot), header=HEADER,
            ensure_row_contiguous=True)

    if route == "group":
        ins = [w, scales, gsc, x, num_uniq, uniq_e, u_off, u_cnt, pair_list]
        shapes = [(B, 8, N // 2)] + ([(B, 8, N)] if debug else [])
        dts = [mx.bfloat16] + ([mx.bfloat16] if debug else [])
    elif route == "kernel":
        ins = [w, scales, gsc, x, logits, bias, scalar(score_eps), scalar(route_scale)]
        shapes = [(B, 8), (B, 8), (B, 8, N // 2)] + ([(B, 8, N)] if debug else [])
        dts = [mx.uint32, mx.float32, mx.bfloat16] + ([mx.bfloat16] if debug else [])
    else:
        ins = [w, scales, gsc, x, inds_arr if inds_arr.dtype == mx.uint32 else inds_arr.astype(mx.uint32)]
        if has_wa:
            ins.append(w_act.reshape(-1))
        shapes = [(B, 8, N // 2)] + ([(B, 8, N)] if debug else [])
        dts = [mx.bfloat16] + ([mx.bfloat16] if debug else [])

    outs = list(k(inputs=ins,
                  template=[("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast),
                            ("TMAX", tmax_gu), ("STREAM_X", stream_x), ("STG_X", stg_x),
                            ("KK", K), ("NN", N), ("BB", B), ("NSG", nsg), ("ROWS", rows)],
                  grid=(32, nsg * (N // (rows * nsg)), B * 8), threadgroup=(32, nsg, 1),
                  output_shapes=shapes, output_dtypes=dts))
    if route == "kernel":
        return outs
    return [inds_arr, wts_arr] + outs


_CHAIN_CONV_ANC = {}


def _default_conv_anc(W):
    a = _CHAIN_CONV_ANC.get(W)
    if a is None:
        rows = [[b - 3 if b >= 3 else -(b + 1), b - 2 if b >= 2 else -(b + 2), b - 1 if b >= 1 else -3] for b in range(W)]
        a = _CHAIN_CONV_ANC[W] = mx.array(rows, mx.int32)
    return a


def down_rows(sw, inds, wts, residual, cs3, conv_w, w, scales, global_scale, *, mode, group_size, bits, debug=False,
              nsg=None, grp=None, tile=None, down_mode=None, rows=None, causal=False, conv_anc=None, skip_combine=False, w_act=None):
    """K13. sw (B,8,K) bf16, inds (B,8) u32, wts (B,8) f32, residual (B,C) f32, cs3 (B,3,C) [or (1,3,C) when causal=True] f32,
    conv_w post-mlp conv weight (C,*,4) bf16, expert weights (X,C,K*bits/32). Returns [out (B,C) bf16, w3 (B,3,C) f32
    (, moe_out (B,C) bf16, post_mlp_conv (B,C) bf16, down (B,8,C) bf16 if debug)]."""
    B, E, K = sw.shape
    C = w.shape[1]
    cs3_ok = (cs3.ndim == 3 and cs3.shape[1:] == (3, C)) if causal else (cs3.shape == (B, 3, C))
    if B > 32 or E != 8 or sw.dtype != mx.bfloat16 or residual.shape != (B, C) or not cs3_ok:
        raise ValueError("down_rows: shapes / dtypes")
    if residual.dtype != mx.float32 or cs3.dtype != mx.float32 or conv_w.dtype != mx.bfloat16 or conv_w.size != C * 4:
        raise ValueError("down_rows: residual/cs3 f32, conv weight bf16 (C,4)")
    N, fast = _geom(w, scales, K, mode, group_size, bits)
    dm = "split" if (causal or tile is not None or (skip_combine and not debug)) else (down_mode or DOWN_MODE or ("fused" if B == 1 else "split"))
    hgs = global_scale is not None
    scale_lit = _SCALE_CONV_LIT
    if C % 4 != 0:
        raise ValueError("C % 4")

    if MOE_B1_FAST and not causal and B == 1 and dm == "fused" and grp is None and tile is None:
        nsg_b1 = nsg or MOE_B1_NSG_DN
        rows_b1 = rows or MOE_B1_ROWS_DN
        if C % rows_b1 != 0 and C % 4 == 0:
            rows_b1 = 4
        if C % rows_b1 == 0 and 32 * nsg_b1 >= rows_b1:
            gsc = global_scale if hgs else _ZERO_F32
            key = ("dnb1", debug, scale_lit)
            k = _kernels.get(key)
            if k is None:
                outs = ["out", "w3"] + (["moe_out", "pmc", "down"] if debug else [])
                k = _kernels[key] = mx.fast.metal_kernel(
                    name="moe_dn_b1_%d" % debug,
                    input_names=["w", "scales", "gsc", "x", "inds", "wts", "resid", "cs3", "cw"],
                    output_names=outs,
                    source=_down_b1_src(debug, scale_lit),
                    header=HEADER,
                    ensure_row_contiguous=True,
                )
            shapes = [(1, C), (1, 3, C)] + ([(1, C), (1, C), (1, 8, C)] if debug else [])
            dts = [mx.bfloat16, mx.float32] + ([mx.bfloat16] * 3 if debug else [])
            inds_u = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
            return list(k(
                inputs=[w, scales, gsc, sw, inds_u, wts, residual, cs3, conv_w],
                template=[("GT", gsc.dtype), ("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast),
                          ("KK", K), ("NN", C), ("NSG", nsg_b1), ("ROWS", rows_b1), ("CC", C)],
                grid=(32, nsg_b1 * (C // rows_b1), 1),
                threadgroup=(32, nsg_b1, 1),
                output_shapes=shapes,
                output_dtypes=dts,
            ))

    nsg = nsg or (NSG_DN if dm == "split" else NSG_DN_FUSED)
    rows = rows or ROWS_DN
    if dm == "split" and C % (rows * nsg) != 0:
        raise ValueError("C not divisible by rows * nsg")
    gsc = (global_scale if global_scale.dtype == mx.float32 else global_scale.astype(mx.float32)) if hgs else _ZERO_F32

    if dm == "split":
        if tile is not None:
            pair_list, exp_off, tile_meta, tmax_t = tile
            _, rows_t, nsg_t = _tile_cfg(C, B)
            if rows is not None and C % (rows * nsg_t) == 0:
                rows_t = rows
            if nsg is not None and C % (rows_t * nsg) == 0:
                nsg_t = nsg
            max_tiles = int(tile_meta.shape[0])
            kt = _kernels.get("dn_tile")
            if kt is None:
                kt = _kernels["dn_tile"] = mx.fast.metal_kernel(
                    name="moe_dn_tile",
                    input_names=["w", "scales", "gsc", "x", "pair_list", "exp_off", "tile_meta"],
                    output_names=["down"],
                    source=_tiled_moe_src(False, False),
                    header=HEADER,
                    ensure_row_contiguous=True,
                )
            (down,) = kt(
                inputs=[w, scales, gsc, sw, pair_list, exp_off, tile_meta],
                template=[("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast),
                          ("TMAX", tmax_t), ("KK", K), ("NN", C), ("NSG", nsg_t), ("ROWS", rows_t)],
                grid=(32, nsg_t * max_tiles, C // (rows_t * nsg_t)),
                threadgroup=(32, nsg_t, 1),
                output_shapes=[(B, 8, C)],
                output_dtypes=[mx.bfloat16],
            )
        else:
            has_grp = grp is not None
            has_wa = (w_act is not None and not has_grp)
            use_ballot = (B <= 8 and not has_grp and os.environ.get("MOE_BALLOT_SCAN", "1") not in ("0", ""))
            tmax_dn = _tmax(B, TMAX_CAP_DN)
            stream_x = _stream_x(tmax_dn)
            stg_x = _stg_x(tmax_dn)
            kg = _kernels.get(("dn_gather6", has_grp, has_wa, stg_x, use_ballot))
            if kg is None:
                ins_g = ["w", "scales", "gsc", "x"] + (
                    ["num_uniq", "uniq_e", "u_off", "u_cnt", "pair_list"] if has_grp else ["inds"]
                ) + (["w_act"] if has_wa else [])
                kg = _kernels[("dn_gather6", has_grp, has_wa, stg_x, use_ballot)] = mx.fast.metal_kernel(
                    name="moe6_down_gather_%d_%d_%d_%d" % (has_grp, int(has_wa), int(stg_x), int(use_ballot)),
                    input_names=ins_g,
                    output_names=["down"],
                    source=_down_gather_src(has_grp, has_w_act=has_wa, has_stg=stg_x, use_ballot=use_ballot),
                    header=HEADER,
                    ensure_row_contiguous=True,
                )
            if has_grp:
                num_uniq, uniq_e, u_off, u_cnt, pair_list = grp
                args_g = [w, scales, gsc, sw, num_uniq, uniq_e, u_off, u_cnt, pair_list]
            else:
                args_g = [w, scales, gsc, sw, inds.astype(mx.uint32)] + ([w_act.reshape(-1)] if has_wa else [])
            (down,) = kg(
                inputs=args_g,
                template=[("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast), ("TMAX", tmax_dn),
                          ("STREAM_X", stream_x), ("STG_X", stg_x),
                          ("KK", K), ("NN", C), ("BB", B), ("NSG", nsg), ("ROWS", rows)],
                grid=(32, nsg * (C // (rows * nsg)), B * 8),
                threadgroup=(32, nsg, 1),
                output_shapes=[(B, 8, C)],
                output_dtypes=[mx.bfloat16],
            )
        if causal:
            if skip_combine and not debug:
                return [down]
            if conv_anc is None:
                conv_anc = _default_conv_anc(B)
            else:
                assert conv_anc.shape == (B, 3) and conv_anc.dtype == mx.int32
            has_wa_c = w_act is not None
            kc = _kernels.get(("dn_comb_causal2", debug, scale_lit, has_wa_c))
            if kc is None:
                ins_c = ["down", "wts", "resid", "cs3", "cw", "conv_anc"] + (["w_act"] if has_wa_c else [])
                outs_c = ["out", "w3"] + (["moe_out", "pmc"] if debug else [])
                kc = _kernels[("dn_comb_causal2", debug, scale_lit, has_wa_c)] = mx.fast.metal_kernel(
                    name="moe3_comb_causal_%d_%d" % (debug, int(has_wa_c)),
                    input_names=ins_c,
                    output_names=outs_c,
                    source=_combine_conv_causal_src(debug, scale_lit, has_w_act=has_wa_c),
                    header=HEADER,
                    ensure_row_contiguous=True,
                )
            shapes_c = [(B, C), (B, 3, C)] + ([(B, C), (B, C)] if debug else [])
            dts_c = [mx.bfloat16, mx.float32] + ([mx.bfloat16] * 2 if debug else [])
            args_c = [down, wts, residual, cs3, conv_w, conv_anc] + ([w_act.reshape(-1)] if has_wa_c else [])
            rc = list(kc(
                inputs=args_c,
                template=[("BB", B), ("CC", C)],
                grid=(((C + 63) // 64) * 64, 1, 1),
                threadgroup=(64, 1, 1),
                output_shapes=shapes_c,
                output_dtypes=dts_c,
            ))
            return rc + ([down] if debug else [])
        if skip_combine and not debug:
            return [down]
        kc = _kernels.get(("dn_comb3", debug, scale_lit))
        if kc is None:
            outs_c = ["out", "w3"] + (["moe_out", "pmc"] if debug else [])
            kc = _kernels[("dn_comb3", debug, scale_lit)] = mx.fast.metal_kernel(
                name="moe3_comb_conv_%d" % debug,
                input_names=["down", "wts", "resid", "cs3", "cw"],
                output_names=outs_c,
                source=_combine_conv_src(debug, scale_lit),
                header=HEADER,
                ensure_row_contiguous=True,
            )
        shapes_c = [(B, C), (B, 3, C)] + ([(B, C), (B, C)] if debug else [])
        dts_c = [mx.bfloat16, mx.float32] + ([mx.bfloat16] * 2 if debug else [])
        rc = list(kc(
            inputs=[down, wts, residual, cs3, conv_w],
            template=[("CC", C)],
            grid=(((C + 255) // 256) * 256, B, 1),
            threadgroup=(256, 1, 1),
            output_shapes=shapes_c,
            output_dtypes=dts_c,
        ))
        return rc + ([down] if debug else [])

    has_grp = grp is not None
    key = ("dn3", has_grp, debug, scale_lit)
    k = _kernels.get(key)
    if k is None:
        if has_grp:
            ins = ["w", "scales", "gsc", "x", "num_uniq", "uniq_e", "u_off", "u_cnt", "pair_list", "wts", "resid", "cs3", "cw"]
        else:
            ins = ["w", "scales", "gsc", "x", "inds", "wts", "resid", "cs3", "cw"]
        outs = ["out", "w3"] + (["moe_out", "pmc", "down"] if debug else [])
        k = _kernels[key] = mx.fast.metal_kernel(
            name="moe3_down_%d_%d" % (has_grp, debug), input_names=ins, output_names=outs,
            source=_down_fused_src(has_grp, debug, scale_lit), header=HEADER, ensure_row_contiguous=True)
    shapes = [(B, C), (B, 3, C)] + ([(B, C), (B, C), (B, 8, C)] if debug else [])
    dts = [mx.bfloat16, mx.float32] + ([mx.bfloat16] * 3 if debug else [])
    if has_grp:
        num_uniq, uniq_e, u_off, u_cnt, pair_list = grp
        args = [w, scales, gsc, sw, num_uniq, uniq_e, u_off, u_cnt, pair_list, wts, residual, cs3, conv_w]
    else:
        args = [w, scales, gsc, sw, inds.astype(mx.uint32), wts, residual, cs3, conv_w]
    tmax_dnf = _tmax(B, TMAX_CAP_DN)
    return list(k(inputs=args,
                  template=[("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast), ("TMAX", tmax_dnf),
                            ("STREAM_X", _stream_x(tmax_dnf)),
                            ("KK", K), ("NN", C), ("BB", B), ("NSG", nsg), ("CC", C)],
                  grid=(32, nsg * (C // 4), 1), threadgroup=(32, nsg, 1), output_shapes=shapes, output_dtypes=dts))


def moe_rows(layer, x, residual, cs3, *, route="auto", debug=False, causal=False, conv_anc=None, skip_combine=False, w_act=None, mlp_mixed=None):
    """The MoE half of a layer visit for B rows. x = mlp_x (B,D) bf16 (or None when mlp_mixed=(mixed,part,ln_w,eps) is given),
    residual (B,D) f32, cs3 (B,3,D) [or (1,3,D) when causal=True] f32 (post-mlp conv state).
    Returns dict: out (B,D) bf16, w3, expert_inds, expert_w, swiglu, router_logits (+ gate_up, down, moe_out, post_mlp_conv if debug)."""
    if isinstance(layer, dict):
        moe = layer["mlp"]
        gate = moe["gate"]
        ex = moe["experts"]
        gu, dn, act = ex["gate_up_proj"], ex["down_proj"], ex["act_fn"]
        cw = layer["post_mlp_convolution"]["conv1d"]["weight"]
    else:
        moe = layer.mlp
        gate = moe.gate
        ex = moe.experts
        gu, dn, act = ex.gate_up_proj, ex.down_proj, ex.act_fn
        cw = layer.post_mlp_convolution.conv1d.weight
    B = int(mlp_mixed[0].shape[0]) if mlp_mixed is not None else x.shape[0]
    w_gate = gate["weight"]
    bias = gate["e_score_correction_bias"]
    rscale = gate.routed_scaling_factor
    tile_min_b = int(os.environ.get("MOE_TILE_MIN_B", "9"))
    if w_act is not None:
        assert causal and B > 1
        route = "launch"
    elif route == "auto":
        if "MOE_AUTO_ROUTE" in os.environ:
            route = AUTO_ROUTE if (B > 1 or causal) else "kernel"
        elif B >= tile_min_b and w_gate.shape[0] == 128:
            route = "tile"
        else:
            route = "kernel" if (B <= 2 and not causal) else AUTO_ROUTE
    if B > 8 and route == "group":
        route = "tile" if w_gate.shape[0] == 128 else "launch"
    rows_gu = 2 if (route != "tile" and 2 <= B <= 32 and "MOE_ROWS_GU" not in os.environ) else None
    rows_dn = 2 if (route != "tile" and 2 <= B <= 32 and "MOE_ROWS_DN" not in os.environ) else None

    grp = None
    tile = None
    if mlp_mixed is not None:
        m_mix, m_part, m_lw, m_eps = mlp_mixed
        logits, x = router_from_mixed_rows(m_mix, m_part, m_lw, w_gate, eps_ln=m_eps)
        if route == "tile":
            tmax_t, _, _ = _tile_cfg(gu["weight"].shape[1], B)
            rt = route_tile_rows(logits, bias, route_scale=rscale, tmax=tmax_t)
            inds_arr, wts_arr = rt[0], rt[1]
            tile = (rt[2], rt[3], rt[4], tmax_t)
            r = gateup_rows(logits, bias, x, gu["weight"], gu["scales"], gu.get("global_scale"),
                            mode=gu.mode, group_size=gu.group_size, bits=gu.bits, route_scale=rscale,
                            gate_softcap=act.gate_softcap, up_softcap=act.up_softcap, route="tile", inds=inds_arr, wts=wts_arr,
                            tile=tile, debug=debug, rows=rows_gu)
            inds, wts, sw = inds_arr, wts_arr, r[2]
        elif route == "group":
            rg = route_group_rows(logits, bias, route_scale=rscale)
            inds_arr, wts_arr = rg[0], rg[1]
            grp = rg[2:]
            r = gateup_rows(logits, bias, x, gu["weight"], gu["scales"], gu.get("global_scale"),
                            mode=gu.mode, group_size=gu.group_size, bits=gu.bits, route_scale=rscale,
                            gate_softcap=act.gate_softcap, up_softcap=act.up_softcap, route="group", inds=inds_arr, grp=grp,
                            debug=debug, rows=rows_gu)
            inds, wts, sw = inds_arr, wts_arr, r[2]
        else:
            r = gateup_rows(logits, bias, x, gu["weight"], gu["scales"], gu.get("global_scale"),
                            mode=gu.mode, group_size=gu.group_size, bits=gu.bits, route_scale=rscale,
                            gate_softcap=act.gate_softcap, up_softcap=act.up_softcap, route=route, debug=debug,
                            rows=rows_gu, w_act=w_act)
            inds, wts, sw = r[0], r[1], r[2]
    elif route == "tile":
        logits = router_rows(x, w_gate)
        tmax_t, _, _ = _tile_cfg(gu["weight"].shape[1], B)
        rt = route_tile_rows(logits, bias, route_scale=rscale, tmax=tmax_t)
        inds_arr, wts_arr = rt[0], rt[1]
        tile = (rt[2], rt[3], rt[4], tmax_t)
        r = gateup_rows(logits, bias, x, gu["weight"], gu["scales"], gu.get("global_scale"),
                        mode=gu.mode, group_size=gu.group_size, bits=gu.bits, route_scale=rscale,
                        gate_softcap=act.gate_softcap, up_softcap=act.up_softcap, route="tile", inds=inds_arr, wts=wts_arr,
                        tile=tile, debug=debug, rows=rows_gu)
        inds, wts, sw = inds_arr, wts_arr, r[2]
    elif route == "group":
        logits = router_rows(x, w_gate)
        rg = route_group_rows(logits, bias, route_scale=rscale)
        inds_arr, wts_arr = rg[0], rg[1]
        grp = rg[2:]
        r = gateup_rows(logits, bias, x, gu["weight"], gu["scales"], gu.get("global_scale"),
                        mode=gu.mode, group_size=gu.group_size, bits=gu.bits, route_scale=rscale,
                        gate_softcap=act.gate_softcap, up_softcap=act.up_softcap, route="group", inds=inds_arr, grp=grp,
                        debug=debug, rows=rows_gu)
        inds, wts, sw = inds_arr, wts_arr, r[2]
    elif (
        route == "launch"
        and os.environ.get("MOE_ROUTER_FUSE", "0") not in ("0", "")
        and 2 <= B <= 8
        and w_gate.shape == (128, 2304)
        and w_gate.dtype == mx.float32
        and x.dtype == mx.bfloat16
    ):
        logits, inds_arr, wts_arr = router_and_route_rows(x, w_gate, bias, route_scale=rscale)
        r = gateup_rows(logits, bias, x, gu["weight"], gu["scales"], gu.get("global_scale"),
                        mode=gu.mode, group_size=gu.group_size, bits=gu.bits, route_scale=rscale,
                        gate_softcap=act.gate_softcap, up_softcap=act.up_softcap, route="launch", inds=inds_arr,
                        wts=wts_arr, debug=debug, rows=rows_gu, w_act=w_act)
        inds, wts, sw = r[0], r[1], r[2]
    else:
        logits = router_rows(x, w_gate)
        r = gateup_rows(logits, bias, x, gu["weight"], gu["scales"], gu.get("global_scale"),
                        mode=gu.mode, group_size=gu.group_size, bits=gu.bits, route_scale=rscale,
                        gate_softcap=act.gate_softcap, up_softcap=act.up_softcap, route=route, debug=debug,
                        rows=rows_gu, w_act=w_act)
        inds, wts, sw = r[0], r[1], r[2]

    o = down_rows(sw, inds, wts, residual, cs3, cw, dn["weight"], dn["scales"],
                  dn.get("global_scale"), mode=dn.mode, group_size=dn.group_size, bits=dn.bits, debug=debug, grp=grp,
                  tile=tile, causal=causal, conv_anc=conv_anc, skip_combine=skip_combine, rows=rows_dn, w_act=w_act)
    if skip_combine and not debug and len(o) == 1:
        return dict(router_logits=logits, expert_inds=inds, expert_w=wts, swiglu=sw, down=o[0], residual=residual, cs3=cs3, conv_w=cw)
    res = dict(router_logits=logits, expert_inds=inds, expert_w=wts, swiglu=sw, out=o[0], w3=o[1])
    if debug:
        res.update(gate_up=r[3], moe_out=o[2], post_mlp_conv=o[3], down=o[4])
    return res
