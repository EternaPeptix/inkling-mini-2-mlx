"""Exact multi-row affine 8-bit quantized matvec (weights read once for B = 1..16 rows).

Every output row is bitwise identical to ``mx.quantized_matmul(x[b].reshape(1, 1, K), w, scales, biases,
transpose=True, group_size=64, bits=8, mode="affine")`` (the B=1 decode call), which MLX 0.32.4 runs as
``affine_qmv_fast`` (K % 256 == 0, K not in {64, 128}). The per-output arithmetic is MLX's own: this module inlines
MLX's ``quantized.h`` as the kernel header and calls ``load_vector`` unchanged; ``qdot`` is its 8-bit body verbatim,
only reading the weight bytes from registers (so one load serves all B rows) instead of device memory.
See docs/ROWS_QMV.md.
"""
import os
import re

import mlx.core as mx

from .glue import GLUE_HEADER, flit

MAX_B = 16
_GS = 64
_BITS = 8
_RESULTS_PER_SG = 4  # results_per_simdgroup of qmv_fast_impl
_ROWS_PER_TG = 8  # 2 simdgroups x 4 rows
_K_ALIGN = 256  # qmv_fast K step for 8 bits: pack_factor 4 * 2 packs * 32 lanes


def _expand_includes(path, root, seen):
    out = []
    with open(path) as f:
        for line in f:
            m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
            if m:
                inc = os.path.join(root, m.group(1))
                if inc not in seen:
                    seen.add(inc)
                    out.append(_expand_includes(inc, root, seen))
            else:
                out.append(line)
    return "".join(out)


_TILE_SRC = r"""
// One tile (NSG simdgroups x RPS rows) for RB input rows starting at row b0; mirrors qmv_fast_impl<bfloat16_t, 64, 8, PR>
// (per output row: same per-lane K partition, same accumulation order, same simd_sum), with RPS and the row group RB free.
// STG (prologue kernels only): the x block is computed once per threadgroup into threadgroup memory sx (double buffered).
template <int B, int K, bool PR, int RPS, int NSG, int RB, int STG, bool PACKW = @PACKW_DEF@>
METAL_FUNC void rows_qmv_tile(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device bfloat16_t* x,
    const device bfloat16_t* gt,
    device bfloat16_t* y,
    const int N,
    const uint ty,
    const uint b0,
    const uint simd_gid,
    const uint simd_lid,
    threadgroup bfloat16_t* sx) {
  constexpr int group_size = 64;
  constexpr int bits = 8;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  static_assert(K % block_size == 0, "rows qmv needs K % 256 == 0");
  constexpr bool FULL = (B % RB) == 0;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  const int nb = FULL ? RB : min(RB, B - int(b0));
  thread U result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0;

  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  const int out_row = ty * (NSG * RPS) + simd_gid * RPS;

  int last_row = RPS - 1;
  bool active = true;
  if (PR) {
    if (out_row >= N) {
      active = false;
      if (!STG) return;
    } else {
      last_row = min(last_row, N - 1 - out_row);
    }
  }

  const device bfloat16_t* xb = x + b0 * K;  // block-origin pointers (staging)
  const device bfloat16_t* gb = gt + b0 * K;
  if (active) {
    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  }
  x += b0 * K + simd_lid * values_per_thread;
  gt += b0 * K + simd_lid * values_per_thread;
  y += b0 * N + out_row;
  const uint tid = simd_gid * 32 + simd_lid;

  static_assert(values_per_thread == 8, "wreg uint2 expects values_per_thread == 8");
  for (int k = 0; k < K; k += block_size) {
    if constexpr (PACKW) {
      thread uint2 wr_r[RPS];
      thread bfloat16_t sc_bf[RPS];
      thread bfloat16_t bi_bf[RPS];
      if (active) {
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          int src = row;
          if (PR) src = min(row, last_row);
          wr_r[row] = ((const device uint2*)(ws + src * in_vec_size_w))[0];
          sc_bf[row] = scales[src * in_vec_size_g];
          bi_bf[row] = biases[src * in_vec_size_g];
        }
      }
      //@STG_FILL@
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < RB; b++) {
        if (FULL || b < nb) {
          thread U xtb[values_per_thread];
          U smb;
          //@XLOAD@
          if (active) {
            _Pragma("clang loop unroll(full)")
            for (int row = 0; row < RPS; row++) {
              float4 w0 = float4(as_type<uchar4>(wr_r[row].x));
              float4 w1 = float4(as_type<uchar4>(wr_r[row].y));
              U accum = 0;
              accum += xtb[0] * w0.x;
              accum += xtb[1] * w0.y;
              accum += xtb[2] * w0.z;
              accum += xtb[3] * w0.w;
              accum += xtb[4] * w1.x;
              accum += xtb[5] * w1.y;
              accum += xtb[6] * w1.z;
              accum += xtb[7] * w1.w;
              result[b][row] += static_cast<U>(sc_bf[row]) * accum + smb * static_cast<U>(bi_bf[row]);
            }
          }
        }
      }
    } else {
      thread float4 w0[RPS];
      thread float4 w1[RPS];
      thread U sc[RPS];
      thread U bi[RPS];
      if (active) {
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          int src = row;
          if (PR) src = min(row, last_row);
          uint2 wr = ((const device uint2*)(ws + src * in_vec_size_w))[0];
          w0[row] = float4(as_type<uchar4>(wr.x));
          w1[row] = float4(as_type<uchar4>(wr.y));
          sc[row] = scales[src * in_vec_size_g];
          bi[row] = biases[src * in_vec_size_g];
        }
      }
      //@STG_FILL@
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < RB; b++) {
        if (FULL || b < nb) {
          thread U xtb[values_per_thread];
          U smb;
          //@XLOAD@
          if (active) {
            _Pragma("clang loop unroll(full)")
            for (int row = 0; row < RPS; row++) {
              U accum = 0;
              accum += xtb[0] * w0[row].x;
              accum += xtb[1] * w0[row].y;
              accum += xtb[2] * w0[row].z;
              accum += xtb[3] * w0[row].w;
              accum += xtb[4] * w1[row].x;
              accum += xtb[5] * w1[row].y;
              accum += xtb[6] * w1[row].z;
              accum += xtb[7] * w1[row].w;
              result[b][row] += sc[row] * accum + smb * bi[row];
            }
          }
        }
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
    gt += block_size;
  }

  if (!active) return;
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      U r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row && (FULL || b < nb)) {
        y[b * N + row] = static_cast<bfloat16_t>(r);
      }
    }
  }
}
"""

_ROUT_TILE_SRC = r"""
template <int B, int K, bool PR, int RPS, int NSG, int RB, bool STG_X = false, bool PACKX = @PACKX_DEF@>
METAL_FUNC void rows_qmv_rout(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device bfloat16_t* x,
    const device bfloat16_t* gt,
    device bfloat16_t* y,
    const int N,
    const uint ty,
    const uint b0,
    const uint simd_gid,
    const uint simd_lid,
    threadgroup bfloat16_t* sx) {
  (void)gt;
  if (!STG_X) (void)sx;
  constexpr int group_size = 64;
  constexpr int bits = 8;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  static_assert(K % block_size == 0, "rows qmv needs K % 256 == 0");
  static_assert(values_per_thread == 8, "rows_qmv_rout expects values_per_thread == 8");
  constexpr bool FULL = (B % RB) == 0;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  const int nb = FULL ? RB : min(RB, B - int(b0));
  thread U result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0;

  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;

  bool active = true;
  int last_row = RPS - 1;
  if (PR) {
    if (out_row >= N) {
      active = false;
      if (!STG_X) return;
    } else {
      last_row = min(last_row, N - 1 - out_row);
    }
  }

  if (active) {
    ws += size_t(out_row) * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += size_t(out_row) * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += size_t(out_row) * in_vec_size_g + simd_lid / scale_step_per_thread;
  }
  x += size_t(b0) * K + simd_lid * values_per_thread;
  y += size_t(b0) * N + out_row;
  threadgroup float* smb_s = (threadgroup float*)(sx + RB * block_size);

  for (int k = 0; k < K; k += block_size) {
    if constexpr (STG_X) {
      for (int b = (int)simd_gid; b < nb; b += NSG) {
        threadgroup uint4* dst_s = (threadgroup uint4*)(sx + b * block_size) + simd_lid;
        dst_s[0] = ((const device uint4*)(x + size_t(b) * K))[0];
        thread U xtmp[8];
        smb_s[b * 32 + simd_lid] = load_vector<bfloat16_t, U, values_per_thread, bits>(x + size_t(b) * K, xtmp);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active) {
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          int src = row;
          if (PR) src = min(row, last_row);
          uint2 wr = ((const device uint2*)(ws + size_t(src) * in_vec_size_w))[0];
          U sc = scales[size_t(src) * in_vec_size_g];
          U bi = biases[size_t(src) * in_vec_size_g];
          float4 w0 = float4(as_type<uchar4>(wr.x));
          float4 w1 = float4(as_type<uchar4>(wr.y));
          _Pragma("clang loop unroll(full)")
          for (int b = 0; b < RB; b++) {
            if (FULL || b < nb) {
              uint4 xpk = ((const threadgroup uint4*)(sx + b * block_size) + simd_lid)[0];
              bfloat4 v0 = as_type<bfloat4>(xpk.xy);
              bfloat4 v1 = as_type<bfloat4>(xpk.zw);
              U smb = smb_s[b * 32 + simd_lid];
              U accum = 0;
              accum += static_cast<U>(v0.x) * w0.x;
              accum += static_cast<U>(v0.y) * w0.y;
              accum += static_cast<U>(v0.z) * w0.z;
              accum += static_cast<U>(v0.w) * w0.w;
              accum += static_cast<U>(v1.x) * w1.x;
              accum += static_cast<U>(v1.y) * w1.y;
              accum += static_cast<U>(v1.z) * w1.z;
              accum += static_cast<U>(v1.w) * w1.w;
              result[b][row] += sc * accum + smb * bi;
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else if constexpr (PACKX) {
      thread uint4 xpk[RB];
      thread U smb[RB];
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < RB; b++) {
        if (FULL || b < nb) {
          uint4 xp = ((const device uint4*)(x + size_t(b) * K))[0];
          xpk[b] = xp;
          bfloat4 v0 = as_type<bfloat4>(xp.xy);
          bfloat4 v1 = as_type<bfloat4>(xp.zw);
          U s = 0;
          s += static_cast<U>(v0.x); s += static_cast<U>(v0.y); s += static_cast<U>(v0.z); s += static_cast<U>(v0.w);
          s += static_cast<U>(v1.x); s += static_cast<U>(v1.y); s += static_cast<U>(v1.z); s += static_cast<U>(v1.w);
          smb[b] = s;
        }
      }
      _Pragma("clang loop unroll(full)")
      for (int row = 0; row < RPS; row++) {
        int src = row;
        if (PR) src = min(row, last_row);
        uint2 wr = ((const device uint2*)(ws + size_t(src) * in_vec_size_w))[0];
        U sc = scales[size_t(src) * in_vec_size_g];
        U bi = biases[size_t(src) * in_vec_size_g];
        float4 w0 = float4(as_type<uchar4>(wr.x));
        float4 w1 = float4(as_type<uchar4>(wr.y));
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < RB; b++) {
          if (FULL || b < nb) {
            bfloat4 v0 = as_type<bfloat4>(xpk[b].xy);
            bfloat4 v1 = as_type<bfloat4>(xpk[b].zw);
            U accum = 0;
            accum += static_cast<U>(v0.x) * w0.x;
            accum += static_cast<U>(v0.y) * w0.y;
            accum += static_cast<U>(v0.z) * w0.z;
            accum += static_cast<U>(v0.w) * w0.w;
            accum += static_cast<U>(v1.x) * w1.x;
            accum += static_cast<U>(v1.y) * w1.y;
            accum += static_cast<U>(v1.z) * w1.z;
            accum += static_cast<U>(v1.w) * w1.w;
            result[b][row] += sc * accum + smb[b] * bi;
          }
        }
      }
    } else {
      thread U xtb[RB][8];
      thread U smb[RB];
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < RB; b++) {
        if (FULL || b < nb) {
          smb[b] = load_vector<bfloat16_t, U, values_per_thread, bits>(x + size_t(b) * K, xtb[b]);
        }
      }
      _Pragma("clang loop unroll(full)")
      for (int row = 0; row < RPS; row++) {
        int src = row;
        if (PR) src = min(row, last_row);
        uint2 wr = ((const device uint2*)(ws + size_t(src) * in_vec_size_w))[0];
        U sc = scales[size_t(src) * in_vec_size_g];
        U bi = biases[size_t(src) * in_vec_size_g];
        float4 w0 = float4(as_type<uchar4>(wr.x));
        float4 w1 = float4(as_type<uchar4>(wr.y));
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < RB; b++) {
          if (FULL || b < nb) {
            U accum = 0;
            accum += xtb[b][0] * w0.x;
            accum += xtb[b][1] * w0.y;
            accum += xtb[b][2] * w0.z;
            accum += xtb[b][3] * w0.w;
            accum += xtb[b][4] * w1.x;
            accum += xtb[b][5] * w1.y;
            accum += xtb[b][6] * w1.z;
            accum += xtb[b][7] * w1.w;
            result[b][row] += sc * accum + smb[b] * bi;
          }
        }
      }
    }
    if (active) {
      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
    }
    x += block_size;
  }

  if (STG_X && !active) return;
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      U r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row && (FULL || b < nb)) {
        y[size_t(b) * N + row] = static_cast<bfloat16_t>(r);
      }
    }
  }
}
"""

_RO_TILE_SRC = r"""
template <int B, int K, bool PR, int RPS, int NSG, int RB>
METAL_FUNC void rows_qmv_ro(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device bfloat16_t* x,
    const device bfloat16_t* gt,
    device bfloat16_t* y,
    const int N,
    const uint ty,
    const uint b0,
    const uint simd_gid,
    const uint simd_lid,
    threadgroup bfloat16_t* sx) {
  (void)gt;
  (void)sx;
  constexpr int group_size = 64;
  constexpr int bits = 8;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  static_assert(K % block_size == 0, "rows qmv needs K % 256 == 0");
  static_assert(values_per_thread == 8, "rows_qmv_ro expects values_per_thread == 8");
  constexpr bool FULL = (B % RB) == 0;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  const int nb = FULL ? RB : min(RB, B - int(b0));
  thread U result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0;

  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;

  int last_row = RPS - 1;
  if (PR) {
    if (out_row >= N) return;
    last_row = min(last_row, N - 1 - out_row);
  }

  ws += size_t(out_row) * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += size_t(out_row) * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += size_t(out_row) * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += size_t(b0) * K + simd_lid * values_per_thread;
  y += size_t(b0) * N + out_row;

  for (int k = 0; k < K; k += block_size) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      int src = PR ? min(row, last_row) : row;
      uint2 wr = ((const device uint2*)(ws + size_t(src) * in_vec_size_w))[0];
      float4 w0 = float4(as_type<uchar4>(wr.x));
      float4 w1 = float4(as_type<uchar4>(wr.y));
      U sc = scales[size_t(src) * in_vec_size_g];
      U bi = biases[size_t(src) * in_vec_size_g];
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < RB; b++) {
        if (FULL || b < nb) {
          uint4 xpk = ((const device uint4*)(x + size_t(b) * K))[0];
          bfloat4 v0 = as_type<bfloat4>(xpk.xy);
          bfloat4 v1 = as_type<bfloat4>(xpk.zw);
          U s = 0;
          s += static_cast<U>(v0.x); s += static_cast<U>(v0.y); s += static_cast<U>(v0.z); s += static_cast<U>(v0.w);
          s += static_cast<U>(v1.x); s += static_cast<U>(v1.y); s += static_cast<U>(v1.z); s += static_cast<U>(v1.w);
          U accum = 0;
          accum += static_cast<U>(v0.x) * w0.x;
          accum += static_cast<U>(v0.y) * w0.y;
          accum += static_cast<U>(v0.z) * w0.z;
          accum += static_cast<U>(v0.w) * w0.w;
          accum += static_cast<U>(v1.x) * w1.x;
          accum += static_cast<U>(v1.y) * w1.y;
          accum += static_cast<U>(v1.z) * w1.z;
          accum += static_cast<U>(v1.w) * w1.w;
          result[b][row] += sc * accum + s * bi;
        }
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      U r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row && (FULL || b < nb)) {
        y[size_t(b) * N + row] = static_cast<bfloat16_t>(r);
      }
    }
  }
}
"""

_XLOAD_PLAIN = (
    "uint4 xpk = ((const device uint4*)(x + size_t(b) * K))[0]; "
    "bfloat4 xv0 = as_type<bfloat4>(xpk.xy); bfloat4 xv1 = as_type<bfloat4>(xpk.zw); "
    "xtb[0] = static_cast<U>(xv0.x); xtb[1] = static_cast<U>(xv0.y); xtb[2] = static_cast<U>(xv0.z); xtb[3] = static_cast<U>(xv0.w); "
    "xtb[4] = static_cast<U>(xv1.x); xtb[5] = static_cast<U>(xv1.y); xtb[6] = static_cast<U>(xv1.z); xtb[7] = static_cast<U>(xv1.w); "
    "smb = 0; smb += xtb[0]; smb += xtb[1]; smb += xtb[2]; smb += xtb[3]; smb += xtb[4]; smb += xtb[5]; smb += xtb[6]; smb += xtb[7];"
)


def tile_source(xload=_XLOAD_PLAIN, stg_fill="", include_rout=False, include_ro=False):
    """The tile text; `xload` computes smb and xtb[*] for row b of the current block (x, gt point at this lane's slice of
    block k, row 0), `stg_fill` runs once per block before the x loads (cooperative staging)."""
    assert _TILE_SRC.count("//@XLOAD@") == 2 and _TILE_SRC.count("//@STG_FILL@") == 2
    packw = "true" if os.environ.get("LOOPLING_QMV_PACKW", "1") not in ("0", "") else "false"
    packx = "true" if os.environ.get("LOOPLING_QMV_PACKX", "1") not in ("0", "") else "false"
    src = (
        _TILE_SRC.replace("@PACKW_DEF@", packw)
        .replace("//@XLOAD@", xload)
        .replace("//@STG_FILL@", stg_fill)
    )
    if include_rout:
        src += _ROUT_TILE_SRC.replace("@PACKX_DEF@", packx)
    if include_ro:
        src += _RO_TILE_SRC
    return src


_TILE = tile_source(include_rout=True, include_ro=True)

# (RPS, NSG, RB, STG) per (B, pro): tuned table (see docs/ROWS_QMV_B.md).  LOOPLING_QMV_CFG="rps,nsg,rb[,stg]" overrides
# it for tuning runs only (STG is honoured by the prologue kernels in proj.py; qmv_rows ignores it).
_CFG_DEFAULT = (4, 2, 4, 0)


# Tuned per-(bucket, B) table for N <= 4096 (0), 4096 < N < 16384 (1), N >= 16384 (2):
# entries are (mode, rps, nsg, rb) with mode 0=tile, 1=rout, 2=ro.
_CFG_TABLE = {
    0: {  # N <= 4096 (e.g. o_proj N=2304)
        1: (1, 1, 2, 1), 2: (0, 1, 2, 2), 3: (1, 2, 4, 3), 4: (1, 4, 4, 4),
        5: (0, 2, 4, 5), 6: (1, 2, 4, 6), 7: (1, 2, 4, 7), 8: (1, 4, 2, 4),
    },
    1: {  # 4096 < N < 16384 (e.g. QKV concat)
        1: (1, 4, 2, 1), 2: (2, 2, 4, 2), 3: (2, 2, 4, 3), 4: (2, 4, 2, 4),
        5: (2, 2, 4, 5), 6: (1, 2, 2, 6), 7: (1, 2, 2, 7), 8: (1, 4, 2, 4),
    },
    2: {  # N >= 16384 (e.g. lm_head)
        1: (1, 1, 2, 1), 2: (0, 1, 4, 2), 3: (0, 4, 2, 3), 4: (1, 2, 4, 4),
        5: (1, 4, 4, 5), 6: (1, 2, 2, 6), 7: (0, 4, 1, 7), 8: (0, 4, 4, 8),
    },
}


def _cfg_bucket(B, tot_n, nw, pro):
    if pro or B > 8 or (tot_n >= 16384 and nw != 1):
        return None
    bkt = 2 if tot_n >= 16384 else (1 if tot_n > 4096 else 0)
    ov = os.environ.get(f"LOOPLING_QMV_BKT{bkt}")
    if ov:
        v = [int(t) for t in ov.split(",")]
        return (v[0], v[1], v[2], min(v[3], B))
    return _CFG_TABLE[bkt].get(B)


def _use_rout(B, tot_n, nw, pro=False):
    if pro:
        return False
    ov = os.environ.get("LOOPLING_QMV_ROUT")
    if ov:
        if ov == "all":
            return True
        if ov in ("0", ""):
            return False
        return tot_n < 16384
    ent = _cfg_bucket(B, tot_n, nw, pro)
    if ent is not None:
        return ent[0] == 1
    return B == 1 or (tot_n >= 16384 and nw == 1 and B in (4, 5, 7, 8))


def _use_ro(B, tot_n, pro=False, rout=False):
    if pro or rout or not (2 <= B <= 8):
        return False
    ov = os.environ.get("LOOPLING_QMV_RO")
    if ov:
        if ov == "all":
            return True
        if ov in ("0", ""):
            return False
        return B <= 6 and tot_n < 16384
    ent = _cfg_bucket(B, tot_n, 1, pro)
    if ent is not None and not os.environ.get("LOOPLING_QMV_ROUT"):
        return ent[0] == 2
    return B <= 6 and tot_n < 16384


def _use_stg_x(B, tot_n, rout, pro=False):
    if pro or not rout:
        return False
    ov = os.environ.get("LOOPLING_QMV_STG_X")
    if ov is not None:
        return ov not in ("0", "", "false", "False") and B >= 2
    return False


def _cfg(B, pro=False, N=0, rout=False, ro=False):
    ov = os.environ.get("LOOPLING_QMV_CFG")
    if ov:
        v = [int(t) for t in ov.split(",")]
        rps, nsg, rb = v[:3]
        stg = v[3] if len(v) > 3 else 0
        return rps, nsg, min(rb, B), stg if pro else 0
    if not pro and not os.environ.get("LOOPLING_QMV_ROUT") and not os.environ.get("LOOPLING_QMV_RO"):
        ent = _cfg_bucket(B, N, 1, pro)
        if ent is not None and ((ent[0] == 1) == bool(rout)) and ((ent[0] == 2) == bool(ro)):
            return ent[1], ent[2], min(ent[3], B), 0
    if not pro and B == 1:
        return 1, 2, 1, 0
    if ro:
        return 2, 4, min(8, B), 0
    if not pro and N >= 16384:
        if rout and B in (4, 5):
            return 4, 4, B, 0
        if rout and B >= 7:
            return 4, 2, 4, 0
        if not rout and not ro and B == 6:
            return 2, 4, 6, 0
    if not pro and N >= 16384 and B >= 8:
        return 4, 4, min(8, B), 0
    if rout:
        return 4, 2, min(4, B), 0
    if not pro and B >= 5:
        return 2, 2, min(8, B), 0
    rps, nsg, rb, stg = _CFG_DEFAULT
    return rps, nsg, B if rb is None else min(rb, B), stg if pro else 0


_header_cache = None


def base_header():
    """MLX's quantized.h scalar helpers (no tile)."""
    global _header_cache
    if _header_cache is None:
        root = os.path.join(os.path.dirname(mx.__file__), "include")
        qh = os.path.join(root, "mlx/backend/metal/kernels/quantized.h")
        src = _expand_includes(qh, root, {qh})
        # keep only the scalar helpers (pack factors, load_vector, qdot, ...): everything from qouter on needs
        # steel/ headers that the custom-kernel preamble does not provide
        cut = src.index("inline void\nqouter")
        cut = src.rindex("template <", 0, cut)
        _header_cache = src[:cut]
    return _header_cache


def _header():
    return base_header() + _TILE


_kernels = {}


def _kernel(nw, rout=False, ro=False, stg_x=False):
    stg_x = bool(stg_x and rout and not ro)
    key = (nw, rout, ro, stg_x)
    k = _kernels.get(key)
    if k is not None:
        return k
    ins = ["x"]
    for i in range(nw):
        ins += [f"w{i}", f"s{i}", f"b{i}"]
    outs = [f"y{i}" for i in range(nw)]
    lines = []
    if stg_x:
        lines.append("threadgroup uint32_t sx_raw[RB * 160]; threadgroup bfloat16_t* sx = (threadgroup bfloat16_t*)sx_raw;")
    else:
        lines.append("threadgroup bfloat16_t* sx = (threadgroup bfloat16_t*)0;")
    lines += [
        "uint ty = threadgroup_position_in_grid.y;",
        "uint rb0 = threadgroup_position_in_grid.z * RB;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint sl = thread_index_in_simdgroup;",
    ]
    fn = "rows_qmv_ro" if ro else ("rows_qmv_rout" if rout else "rows_qmv_tile")
    tail_t = (", true" if stg_x else "") if rout else ("" if ro else ", 0")
    prev = "0"
    for i in range(nw):
        end = f"{prev} + ((N{i} + RPS * NSG - 1) / (RPS * NSG))"
        cond = f"ty < {end}"
        pre = "if" if i == 0 else "else if"
        if i == nw - 1:
            cond = "true"
            pre = "else" if i else "if"
        head = f"{pre} ({cond})" if cond != "true" or i == 0 else "else"
        lines.append(
            f"{head} {{ {fn}<B, K, (N{i} % (RPS * NSG)) != 0, RPS, NSG, RB{tail_t}>"
            f"(w{i}, s{i}, b{i}, x, x, y{i}, N{i}, ty - ({prev}), rb0, sg, sl, sx); }}"
        )
        prev = end
    suf = ("_rout_sx" if stg_x else "_rout") if rout else ("_ro" if ro else "")
    k = mx.fast.metal_kernel(
        name=f"rows_qmv_aff8_gs64_n{nw}{suf}",
        input_names=ins,
        output_names=outs,
        source="\n".join(lines),
        header=_header(),
        ensure_row_contiguous=True,
    )
    _kernels[key] = k
    return k


def qmv_rows_multi(x, weights, group_size=64, bits=8):
    """x (B, K) bf16, weights [(w, scales, biases), ...] sharing the input rows -> [(B, N_i) bf16, ...], one launch."""
    if group_size != _GS or bits != _BITS:
        raise ValueError("only affine 8-bit group_size 64 is implemented")
    if x.ndim != 2 or x.dtype != mx.bfloat16:
        raise ValueError("x must be (B, K) bfloat16")
    B, K = x.shape
    if not 1 <= B <= MAX_B:
        raise ValueError(f"B must be 1..{MAX_B}")
    if K % _K_ALIGN:
        raise ValueError(f"K must be a multiple of {_K_ALIGN} (MLX would pick the non-fast qmv/qmv_quad kernel)")
    if K in (64, 128):
        raise ValueError("K in {64,128} routes to qmv_quad in MLX")
    nw = len(weights)
    args = [x]
    tot_n = sum(w.shape[0] for w, _, _ in weights)
    rout = _use_rout(B, tot_n, nw, pro=False)
    ro = _use_ro(B, tot_n, pro=False, rout=rout)
    stg_x = _use_stg_x(B, tot_n, rout, pro=False)
    rps, nsg, rb, _ = _cfg(B, N=tot_n, rout=rout, ro=ro)
    tmpl = [("B", B), ("K", K), ("RPS", rps), ("NSG", nsg), ("RB", rb)]
    shapes = []
    tiles = 0
    for i, (w, s, b) in enumerate(weights):
        N = w.shape[0]
        if w.shape[1] * 4 != K or s.shape != (N, K // _GS) or b.shape != (N, K // _GS) or s.dtype != mx.bfloat16:
            raise ValueError(f"weight {i}: bad shapes/dtype w{w.shape} s{s.shape}")
        args += [w, s, b]
        tmpl.append((f"N{i}", N))
        shapes.append((B, N))
        tiles += (N + rps * nsg - 1) // (rps * nsg)
    return _kernel(nw, rout=rout, ro=ro, stg_x=stg_x)(
        inputs=args,
        template=tmpl,
        grid=(32, nsg * tiles, (B + rb - 1) // rb),
        threadgroup=(32, nsg, 1),
        output_shapes=shapes,
        output_dtypes=[mx.bfloat16] * nw,
    )


def qmv_rows(x, w, scales, biases, group_size=64, bits=8):
    """(B, K) bf16 -> (B, N) bf16; row b bitwise == quantized_matmul(x[b].reshape(1,1,K), ...)."""
    return qmv_rows_multi(x, [(w, scales, biases)], group_size, bits)[0]


def _arg_sc_header(has_sc, sc_lit):
    body = f"(metal::precise::tanh(x / {sc_lit}) * {sc_lit})" if has_sc else "x"
    sat = f"(5.5f * {sc_lit})" if has_sc else "INFINITY"
    return f"""
constant constexpr bool ARG_HAS_SC = {'true' if has_sc else 'false'};
constant constexpr float ARG_SAT_THR = {sat};
inline float qmv_head_sc(bfloat16_t v) {{
  GL_NOFMA
  float x = float(v);
  return {body};
}}
"""


_ARG_P1_SRC = r"""
  constexpr int group_size = 64;
  constexpr int bits = 8;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  static_assert(K % block_size == 0, "qmv_arg needs K % 256 == 0");
  static_assert(values_per_thread == 8, "qmv_arg expects values_per_thread == 8");
  constexpr bool FULL = (B % RB) == 0;

  uint sp = threadgroup_position_in_grid.y;
  uint b0 = threadgroup_position_in_grid.z * RB;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  typedef float U;
  const int nb = FULL ? RB : min(RB, B - int(b0));
  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;

  const int chunk = (TOT_TILES + SP - 1) / SP;
  const int t0 = int(sp) * chunk;
  const int t1 = min(TOT_TILES, t0 + chunk);

  threadgroup float tg_bv[RB][NSG];
  threadgroup uint tg_bi[RB][NSG];
  thread float best_v[RB];
  thread uint best_i[RB];
  thread bool best_sat[RB];
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    best_v[b] = -INFINITY;
    best_i[b] = 0xFFFFFFFFu;
    best_sat[b] = false;
  }

  for (int ty = t0; ty < t1; ty++) {
    const int out_row = ty * (NSG * RPS) + int(simd_gid) * RPS;
    if (out_row >= VV) continue;
    const int last_row = min(RPS - 1, VV - 1 - out_row);

    thread U result[RB][RPS];
    for (int b = 0; b < RB; b++)
      for (int r = 0; r < RPS; r++) result[b][r] = 0;

    const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    const device bfloat16_t* sc_p = scales + size_t(out_row) * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device bfloat16_t* bi_p = biases + size_t(out_row) * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device bfloat16_t* xp = x + size_t(b0) * K + simd_lid * values_per_thread;

    for (int k = 0; k < K; k += block_size) {
      if (MODE == 2) {
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          int src = min(row, last_row);
          uint2 wr = ((const device uint2*)(ws + size_t(src) * in_vec_size_w))[0];
          float4 w0 = float4(as_type<uchar4>(wr.x));
          float4 w1 = float4(as_type<uchar4>(wr.y));
          U sc = sc_p[size_t(src) * in_vec_size_g];
          U bi = bi_p[size_t(src) * in_vec_size_g];
          _Pragma("clang loop unroll(full)")
          for (int b = 0; b < RB; b++) {
            if (FULL || b < nb) {
              uint4 xpk = ((const device uint4*)(xp + size_t(b) * K))[0];
              bfloat4 v0 = as_type<bfloat4>(xpk.xy);
              bfloat4 v1 = as_type<bfloat4>(xpk.zw);
              U s = 0;
              s += static_cast<U>(v0.x); s += static_cast<U>(v0.y); s += static_cast<U>(v0.z); s += static_cast<U>(v0.w);
              s += static_cast<U>(v1.x); s += static_cast<U>(v1.y); s += static_cast<U>(v1.z); s += static_cast<U>(v1.w);
              U accum = 0;
              accum += static_cast<U>(v0.x) * w0.x; accum += static_cast<U>(v0.y) * w0.y;
              accum += static_cast<U>(v0.z) * w0.z; accum += static_cast<U>(v0.w) * w0.w;
              accum += static_cast<U>(v1.x) * w1.x; accum += static_cast<U>(v1.y) * w1.y;
              accum += static_cast<U>(v1.z) * w1.z; accum += static_cast<U>(v1.w) * w1.w;
              result[b][row] += sc * accum + s * bi;
            }
          }
        }
      } else if (MODE == 1) {
        thread uint4 xpk[RB];
        thread U smb[RB];
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < RB; b++) {
          if (FULL || b < nb) {
            uint4 x4 = ((const device uint4*)(xp + size_t(b) * K))[0];
            xpk[b] = x4;
            bfloat4 v0 = as_type<bfloat4>(x4.xy);
            bfloat4 v1 = as_type<bfloat4>(x4.zw);
            U s = 0;
            s += static_cast<U>(v0.x); s += static_cast<U>(v0.y); s += static_cast<U>(v0.z); s += static_cast<U>(v0.w);
            s += static_cast<U>(v1.x); s += static_cast<U>(v1.y); s += static_cast<U>(v1.z); s += static_cast<U>(v1.w);
            smb[b] = s;
          }
        }
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          int src = min(row, last_row);
          uint2 wr = ((const device uint2*)(ws + size_t(src) * in_vec_size_w))[0];
          U sc = sc_p[size_t(src) * in_vec_size_g];
          U bi = bi_p[size_t(src) * in_vec_size_g];
          float4 w0 = float4(as_type<uchar4>(wr.x));
          float4 w1 = float4(as_type<uchar4>(wr.y));
          _Pragma("clang loop unroll(full)")
          for (int b = 0; b < RB; b++) {
            if (FULL || b < nb) {
              bfloat4 v0 = as_type<bfloat4>(xpk[b].xy);
              bfloat4 v1 = as_type<bfloat4>(xpk[b].zw);
              U accum = 0;
              accum += static_cast<U>(v0.x) * w0.x;
              accum += static_cast<U>(v0.y) * w0.y;
              accum += static_cast<U>(v0.z) * w0.z;
              accum += static_cast<U>(v0.w) * w0.w;
              accum += static_cast<U>(v1.x) * w1.x;
              accum += static_cast<U>(v1.y) * w1.y;
              accum += static_cast<U>(v1.z) * w1.z;
              accum += static_cast<U>(v1.w) * w1.w;
              result[b][row] += sc * accum + smb[b] * bi;
            }
          }
        }
      } else {
        thread uint2 wr_r[RPS];
        thread bfloat16_t sc_bf[RPS];
        thread bfloat16_t bi_bf[RPS];
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          int src = min(row, last_row);
          wr_r[row] = ((const device uint2*)(ws + size_t(src) * in_vec_size_w))[0];
          sc_bf[row] = sc_p[size_t(src) * in_vec_size_g];
          bi_bf[row] = bi_p[size_t(src) * in_vec_size_g];
        }
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < RB; b++) {
          if (FULL || b < nb) {
            uint4 xpk = ((const device uint4*)(xp + size_t(b) * K))[0];
            bfloat4 xv0 = as_type<bfloat4>(xpk.xy);
            bfloat4 xv1 = as_type<bfloat4>(xpk.zw);
            U xtb0 = static_cast<U>(xv0.x), xtb1 = static_cast<U>(xv0.y), xtb2 = static_cast<U>(xv0.z), xtb3 = static_cast<U>(xv0.w);
            U xtb4 = static_cast<U>(xv1.x), xtb5 = static_cast<U>(xv1.y), xtb6 = static_cast<U>(xv1.z), xtb7 = static_cast<U>(xv1.w);
            U smb = 0;
            smb += xtb0; smb += xtb1; smb += xtb2; smb += xtb3;
            smb += xtb4; smb += xtb5; smb += xtb6; smb += xtb7;
            _Pragma("clang loop unroll(full)")
            for (int row = 0; row < RPS; row++) {
              float4 w0 = float4(as_type<uchar4>(wr_r[row].x));
              float4 w1 = float4(as_type<uchar4>(wr_r[row].y));
              U accum = 0;
              accum += xtb0 * w0.x;
              accum += xtb1 * w0.y;
              accum += xtb2 * w0.z;
              accum += xtb3 * w0.w;
              accum += xtb4 * w1.x;
              accum += xtb5 * w1.y;
              accum += xtb6 * w1.z;
              accum += xtb7 * w1.w;
              result[b][row] += static_cast<U>(sc_bf[row]) * accum + smb * static_cast<U>(bi_bf[row]);
            }
          }
        }
      }
      ws += block_size * bytes_per_pack / pack_factor;
      sc_p += block_size / group_size;
      bi_p += block_size / group_size;
      xp += block_size;
    }

    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < RB; b++) {
      if (FULL || b < nb) {
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          U r = simd_sum(result[b][row]);
          if (simd_lid == 0 && row <= last_row) {
            uint idx = uint(out_row + row);
            if constexpr (ARG_HAS_SC) {
              float rv = float(static_cast<bfloat16_t>(r));
              if (!best_sat[b] && !(metal::abs(rv) >= ARG_SAT_THR || metal::isnan(rv))) {
                if (rv > best_v[b] || (rv == best_v[b] && idx < best_i[b])) {
                  best_v[b] = rv;
                  best_i[b] = idx;
                }
              } else {
                GlueIV cur_best;
                if (!best_sat[b]) {
                  cur_best.val = (best_i[b] == 0xFFFFFFFFu) ? -INFINITY : qmv_head_sc(bfloat16_t(best_v[b]));
                  cur_best.idx = best_i[b];
                  best_sat[b] = true;
                } else {
                  cur_best.val = best_v[b];
                  cur_best.idx = best_i[b];
                }
                GlueIV cur;
                cur.val = qmv_head_sc(static_cast<bfloat16_t>(r));
                cur.idx = idx;
                cur_best = glue_iv_reduce(cur_best, cur);
                best_v[b] = cur_best.val;
                best_i[b] = cur_best.idx;
              }
            } else {
              GlueIV cur_best;
              cur_best.val = best_v[b];
              cur_best.idx = best_i[b];
              GlueIV cur;
              cur.val = float(static_cast<bfloat16_t>(r));
              cur.idx = idx;
              cur_best = glue_iv_reduce(cur_best, cur);
              best_v[b] = cur_best.val;
              best_i[b] = cur_best.idx;
            }
          }
        }
      } else {
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          (void)simd_sum(result[b][row]);
        }
      }
    }
  }

  if (simd_lid == 0) {
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < RB; b++) {
      if (ARG_HAS_SC && !best_sat[b] && best_i[b] != 0xFFFFFFFFu) {
        best_v[b] = qmv_head_sc(bfloat16_t(best_v[b]));
      }
      tg_bv[b][simd_gid] = best_v[b];
      tg_bi[b][simd_gid] = best_i[b];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0 && simd_lid == 0) {
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < RB; b++) {
      if (FULL || b < nb) {
        GlueIV g;
        g.val = tg_bv[b][0];
        g.idx = tg_bi[b][0];
        for (int s = 1; s < NSG; s++) {
          GlueIV o;
          o.val = tg_bv[b][s];
          o.idx = tg_bi[b][s];
          g = glue_iv_reduce(g, o);
        }
        size_t o = size_t(b0 + b) * SP + sp;
        pval[o] = g.val;
        pidx[o] = g.idx;
      }
    }
  }
"""

_ARG_P2_SRC = r"""
  uint b = threadgroup_position_in_grid.x;
  uint lane = thread_index_in_simdgroup;
  GlueIV best;
  best.val = -INFINITY;
  best.idx = 0xFFFFFFFFu;
  for (int i = int(lane); i < SP; i += 32) {
    size_t o = size_t(b) * SP + uint(i);
    GlueIV c;
    c.val = pval[o];
    c.idx = pidx[o];
    best = glue_iv_reduce(best, c);
  }
  for (uint d = 16; d > 0; d /= 2) {
    GlueIV o;
    o.val = simd_shuffle_down(best.val, d);
    o.idx = simd_shuffle_down(best.idx, d);
    best = glue_iv_reduce(best, o);
  }
  if (lane == 0) out[b] = best.idx;
"""

_ARG_P2_WALK_SRC = r"""
  uint b = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  threadgroup uint t_row[8];
  GlueIV best;
  best.val = -INFINITY;
  best.idx = 0xFFFFFFFFu;
  for (int i = int(lane); i < SP; i += 32) {
    size_t o = size_t(b) * SP + uint(i);
    GlueIV c;
    c.val = pval[o];
    c.idx = pidx[o];
    best = glue_iv_reduce(best, c);
  }
  for (uint d = 16; d > 0; d /= 2) {
    GlueIV o;
    o.val = simd_shuffle_down(best.val, d);
    o.idx = simd_shuffle_down(best.idx, d);
    best = glue_iv_reduce(best, o);
  }
  if (lane == 0) t_row[b] = best.idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (b != 0 || lane != 0) return;

  int path[8];
  int keep = 1;
  path[0] = 0;
  int cur_slot = 0;
  while (true) {
    uint target_tok = t_row[cur_slot];
    int matched = -1;
    for (int s = 1; s < WW; s++) {
      if (par[s] == cur_slot && blk[s] == target_tok) {
        matched = s;
        break;
      }
    }
    if (matched < 0) break;
    path[keep++] = matched;
    cur_slot = matched;
  }
  uint bonus_tok = t_row[cur_slot];
  for (int i = 0; i < WW; i++) {
    path_pad[i] = (i < keep) ? path[i] : 0;
    if (i < keep - 1) {
      emitted[i] = blk[path[i + 1]];
    } else if (i == keep - 1) {
      emitted[i] = bonus_tok;
    } else {
      emitted[i] = 0u;
    }
  }
  keep_arr[0] = keep;
  last_slot[0] = cur_slot;
  anchor_next[0] = bonus_tok;

  const int end0_r = meta_r[1];
  const int slot0_r = meta_r[2];
  const int cap_r = meta_r[3];
  const int max_size_r = meta_r[4];
  const int end1_r = end0_r + keep;
  const int s1_r = min(end1_r, max_size_r);
  const int next_slot0_r = (slot0_r + keep) % max_size_r;
  slot_r_old[0] = slot0_r;
  next_meta_r[0] = s1_r;
  next_meta_r[1] = end1_r;
  next_meta_r[2] = next_slot0_r;
  next_meta_r[3] = cap_r;
  next_meta_r[4] = max_size_r;
  next_meta_r[5] = 0;

  const int end0_f = meta_f[1];
  const int cap_f = meta_f[3];
  const int end1_f = end0_f + keep;
  end_f_old[0] = end0_f;
  next_meta_f[0] = end1_f;
  next_meta_f[1] = end1_f;
  next_meta_f[2] = 0;
  next_meta_f[3] = cap_f;
  next_meta_f[4] = 0;
  next_meta_f[5] = 0;

  if (META_LEN == 8) {
    const int st_old = meta_r[7];
    st_pos_old[0] = st_old;
    next_meta_r[6] = meta_r[6];
    next_meta_r[7] = st_old + keep;
    next_meta_f[6] = meta_f[6];
    next_meta_f[7] = st_old + keep;
  } else {
    st_pos_old[0] = 0;
  }

  const int P = dr_pos[0];
  dr_pos_old[0] = P;
  next_dr_pos[0] = P + keep;
"""

_arg_p1_k = {}
_arg_p2_k = None
_arg_p2_walk_k = {}


def _arg_p1_src(has_wa=False):
    if not has_wa:
        return _ARG_P1_SRC
    s = _ARG_P1_SRC.replace(
        "  const int nb = FULL ? RB : min(RB, B - int(b0));",
        "  (void)FULL;\n  const int w_a = min(B, max(1, w_act[0]));\n  if (int(b0) >= w_a) return;\n  const int nb = min(RB, w_a - int(b0));",
    )
    assert s != _ARG_P1_SRC
    s2 = s.replace("if (FULL || b < nb)", "if (b < nb)")
    assert s2.count("if (b < nb)") == 6
    return s2


def _arg_p2_walk_src(has_wa=False):
    if not has_wa:
        return _ARG_P2_WALK_SRC
    s = _ARG_P2_WALK_SRC.replace(
        "  threadgroup uint t_row[8];\n  GlueIV best;",
        "  threadgroup uint t_row[8];\n  const int w_a = min(WW, max(1, w_act[0]));\n  if (int(b) < w_a) {\n  GlueIV best;",
    )
    s = s.replace(
        "  if (lane == 0) t_row[b] = best.idx;\n  threadgroup_barrier(mem_flags::mem_threadgroup);",
        "  if (lane == 0) t_row[b] = best.idx;\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);",
    )
    s = s.replace("for (int s = 1; s < WW; s++)", "for (int s = 1; s < w_a; s++)")
    assert s != _ARG_P2_WALK_SRC
    return s


def _get_arg_walk_p2(has_wa=False):
    k = _arg_p2_walk_k.get(has_wa)
    if k is None:
        ins = ["pval", "pidx", "blk", "par", "meta_r", "meta_f", "dr_pos"] + (["w_act"] if has_wa else [])
        k = _arg_p2_walk_k[has_wa] = mx.fast.metal_kernel(
            name="rows_qmv_arg_p2_walk_wa" if has_wa else "rows_qmv_arg_p2_walk",
            input_names=ins,
            output_names=[
                "path_pad", "emitted", "keep_arr", "last_slot", "anchor_next",
                "slot_r_old", "end_f_old", "dr_pos_old", "next_meta_r", "next_meta_f", "next_dr_pos",
                "st_pos_old",
            ],
            source=_arg_p2_walk_src(has_wa),
            header=GLUE_HEADER,
            ensure_row_contiguous=True,
        )
    return k


def _get_arg_kernels(softcap, has_wa=False):
    global _arg_p2_k
    has_sc = softcap is not None and float(softcap) > 0.0
    sc_lit = flit(float(softcap)) if has_sc else "1.0f"
    key = (has_sc, sc_lit, has_wa)
    p1 = _arg_p1_k.get(key)
    if p1 is None:
        sfx = sc_lit.replace(".", "p").replace("-", "m").replace("+", "P") if has_sc else "nosc"
        ins = ["x", "w", "scales", "biases"] + (["w_act"] if has_wa else [])
        p1 = mx.fast.metal_kernel(
            name=f"rows_qmv_arg_p1_{sfx}" + ("_wa" if has_wa else ""),
            input_names=ins,
            output_names=["pval", "pidx"],
            source=_arg_p1_src(has_wa),
            header=base_header() + GLUE_HEADER + _arg_sc_header(has_sc, sc_lit),
            ensure_row_contiguous=True,
        )
        _arg_p1_k[key] = p1
    if _arg_p2_k is None:
        _arg_p2_k = mx.fast.metal_kernel(
            name="rows_qmv_arg_p2",
            input_names=["pval", "pidx"],
            output_names=["out"],
            source=_ARG_P2_SRC,
            header=GLUE_HEADER,
            ensure_row_contiguous=True,
        )
    return p1, _arg_p2_k


_ARG_CFG_TABLE = {
    1: (2, 2, 2, 1, 1024),
    2: (0, 2, 2, 2, 1024),
    3: (1, 2, 2, 3, 1024),
    4: (1, 2, 2, 4, 2048),
    5: (1, 2, 2, 5, 1024),
    6: (2, 2, 1, 6, 2048),
    7: (1, 2, 4, 7, 4096),
    8: (1, 2, 4, 8, 4096),
}


def _arg_cfg(B, N, VV, walk=False):
    """Return (mode, rps, nsg, rb, sp, tot_tiles) for fused lm_head softcap+argmax (0=tile, 1=rout, 2=ro)."""
    ov = os.environ.get("LOOPLING_LM_CFG")
    if ov:
        vals = [int(t) for t in ov.split(",")]
        mode, rps, nsg, rb, sp_req = vals
        rb = min(rb, B)
    elif os.environ.get("LOOPLING_LM_SP") or os.environ.get("LOOPLING_QMV_ROUT") or os.environ.get("LOOPLING_QMV_CFG"):
        rout = _use_rout(B, N, 1, pro=False)
        ro = _use_ro(B, N, pro=False, rout=rout)
        mode = 2 if ro else (1 if rout else 0)
        rps, nsg, rb, _ = _cfg(B, N=N, rout=rout, ro=ro)
        sp_env = os.environ.get("LOOPLING_LM_SP")
        sp_req = int(sp_env) if sp_env else (1024 if B <= 3 else (512 if B == 8 else 2048))
    elif walk and B == 4:
        mode, rps, nsg, rb, sp_req = 1, 4, 4, 4, 2048
    else:
        mode, rps, nsg, rb, sp_req = _ARG_CFG_TABLE.get(B, (1, 2, 4, B, 4096))
    tile_w = rps * nsg
    tot_tiles = (VV + tile_w - 1) // tile_w
    sp = min(sp_req, tot_tiles)
    return mode, rps, nsg, rb, sp, tot_tiles


def qmv_softcap_argmax_rows(x, w, scales, biases, softcap=None, unpadded_vocab=None):
    """Fused 8-bit qmv_rows -> softcap -> argmax returning (B,) uint32 (bit-exact with qmv_rows + softcap_argmax_rows)."""
    use_two_k = (
        x.ndim == 2
        and x.shape[0] in (5, 7, 8)
        and w.shape[0] >= 16384
        and not os.environ.get("LOOPLING_LM_CFG")
        and not os.environ.get("LOOPLING_LM_SP")
        and os.environ.get("LOOPLING_LM_TWO_K", "1") not in ("0", "")
    )
    if (
        use_two_k
        or os.environ.get("LOOPLING_FUSED_LM_ARGMAX", "1") in ("0", "")
        or x.ndim != 2
        or x.dtype != mx.bfloat16
        or x.shape[0] not in (1, 2, 3, 4, 5, 6, 7, 8)
        or x.shape[1] % _K_ALIGN != 0
        or x.shape[1] in (64, 128)
        or scales.dtype != mx.bfloat16
    ):
        from .head import softcap_argmax_rows

        return softcap_argmax_rows(qmv_rows(x, w, scales, biases), softcap, unpadded_vocab)

    B, K = x.shape
    N = w.shape[0]
    VV = int(unpadded_vocab) if unpadded_vocab is not None else int(N)
    assert 1 <= VV <= N and w.shape[1] * 4 == K
    mode, rps, nsg, rb, sp, tot_tiles = _arg_cfg(B, N, VV, walk=False)
    p1, p2 = _get_arg_kernels(softcap)
    pval, pidx = p1(
        inputs=[x, w, scales, biases],
        template=[
            ("B", B), ("K", K), ("VV", VV), ("SP", sp), ("TOT_TILES", tot_tiles),
            ("MODE", mode), ("RPS", rps), ("NSG", nsg), ("RB", rb),
        ],
        grid=(32, nsg * sp, (B + rb - 1) // rb),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, sp), (B, sp)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return p2(
        inputs=[pval, pidx],
        template=[("SP", sp)],
        grid=(B * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B,)],
        output_dtypes=[mx.uint32],
    )[0]


def qmv_softcap_argmax_walk_rows(
    x, w, scales, biases, blk, par, meta_r, meta_f, dr_pos, softcap=None, unpadded_vocab=None, w_act=None
):
    """Fused 8-bit qmv_rows -> softcap -> argmax p2 + on-GPU tree walk & metadata update (11-tuple)."""
    B, K = x.shape
    N = w.shape[0]
    if (
        os.environ.get("LOOPLING_FUSED_LM_ARGMAX", "1") in ("0", "")
        or x.ndim != 2
        or x.dtype != mx.bfloat16
        or B not in (2, 3, 4, 5, 6, 7, 8)
        or K % _K_ALIGN != 0
        or K in (64, 128)
        or scales.dtype != mx.bfloat16
    ):
        from .qmv4 import tree_walk_commit_meta_gpu

        t = qmv_softcap_argmax_rows(x, w, scales, biases, softcap, unpadded_vocab)
        return tree_walk_commit_meta_gpu(t, blk, par, meta_r, meta_f, dr_pos, B, w_act=w_act)

    VV = int(unpadded_vocab) if unpadded_vocab is not None else int(N)
    assert 1 <= VV <= N and w.shape[1] * 4 == K
    has_wa = w_act is not None
    mode, rps, nsg, rb, sp, tot_tiles = _arg_cfg(B, N, VV, walk=True)
    p1, _ = _get_arg_kernels(softcap, has_wa=False)
    wa_arr = w_act.reshape(-1) if has_wa else None
    pval, pidx = p1(
        inputs=[x, w, scales, biases],
        template=[
            ("B", B), ("K", K), ("VV", VV), ("SP", sp), ("TOT_TILES", tot_tiles),
            ("MODE", mode), ("RPS", rps), ("NSG", nsg), ("RB", rb),
        ],
        grid=(32, nsg * sp, (B + rb - 1) // rb),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, sp), (B, sp)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    bu = blk.reshape(-1)
    if bu.dtype != mx.uint32:
        bu = bu.astype(mx.uint32)
    pi = par.reshape(-1)
    if pi.dtype != mx.int32:
        pi = pi.astype(mx.int32)
    mr = meta_r.reshape(-1)
    mf = meta_f.reshape(-1)
    mlen = int(mr.shape[0])
    assert mlen in (6, 8) and int(mf.shape[0]) == mlen
    p2_ins = [pval, pidx, bu, pi, mr, mf, dr_pos.reshape(-1)] + ([wa_arr] if has_wa else [])
    res = _get_arg_walk_p2(has_wa=has_wa)(
        inputs=p2_ins,
        template=[("SP", sp), ("WW", B), ("META_LEN", mlen)],
        grid=(32, B, 1),
        threadgroup=(32, B, 1),
        output_shapes=[(B,), (B,), (1,), (1,), (1,), (1,), (1,), (1,), (mlen,), (mlen,), (1,), (1,)],
        output_dtypes=[
            mx.int32, mx.uint32, mx.int32, mx.int32, mx.uint32,
            mx.int32, mx.int32, mx.int32, mx.int32, mx.int32, mx.int32, mx.int32,
        ],
    )
    return res if mlen == 8 else res[:11]
