"""Exact, weights-read-once MoE expert matvecs for B tokens x E experts (fp-quantized: nvfp4 / mxfp8 / mxfp4).

`gather_rows(x, w, scales, biases, idx, mode=..., group_size=..., bits=..., global_scale=...)` returns (B, E, N) bf16 where
every (token, expert) output is bitwise identical to `mx.gather_qmm(...)` called for that token alone at M=1
(MLX picks `fp_gather_qmv` / `fp_gather_qmv_fast` there). The kernel inlines MLX's own `fp_quantized.h` (expanded from the
installed wheel) and calls its `qdot`, `qdot_safe`, `load_vector`, `dequantize_scale` so the per-output arithmetic is the
same code: same per-lane K partition, same accumulation order, same `simd_sum`, same epilogue (`result * inv_scale_enc`
then bf16 cast). See docs/ROWS_GATHER.md.

Grouping without a host sync: one threadgroup (32x2 threads) per (token, expert) pair and 8-row output tile. A pair
threadgroup first scans the (B*E) index vector (simdgroup 0, prefix sums); if an earlier pair already picked the same
expert it exits, otherwise it becomes the owner of the expert and handles every pair that picked it, streaming the weight
tile once and running the token loop inside the K loop (per-pair accumulators stay in registers until the final store).
"""

import os
import re
from functools import lru_cache

import mlx.core as mx

__all__ = ["gather_rows", "mlx_header"]

_INC_ROOT = os.path.join(os.path.dirname(mx.__file__), "include")
_INC_RE = re.compile(r'^\s*#\s*include\s+"(mlx/[^"]+)"\s*$')


@lru_cache(maxsize=None)
def mlx_header(rel="mlx/backend/metal/kernels/fp_quantized.h"):
    """Recursively expand `#include "mlx/..."` of an MLX kernel header from the wheel (each file once)."""
    seen = set()

    def expand(path):
        if path in seen:
            return ""
        seen.add(path)
        out = []
        with open(os.path.join(_INC_ROOT, path)) as f:
            for line in f:
                m = _INC_RE.match(line)
                if m:
                    out.append(expand(m.group(1)))
                elif line.strip() == "#pragma once":
                    continue
                else:
                    out.append(line)
        return "".join(out)

    text = expand(rel)
    # Only the scalar helpers (dequantize_scale, Dequantize, load_vector*, qdot*) are needed; the rest of the file (block
    # loaders, steel GEMM kernels) depends on includes that the .metal wrappers provide, so cut right before `qouter`.
    cut = text.index("template <typename U, int values_per_thread, int bits>\ninline void qouter")
    return text[:cut] + _HELPERS


_HELPERS = r"""
// Same arithmetic as MLX's qdot (fp_quantized.h), split into "dequantize once" and "dot with x" so that several tokens
// can share the dequantized weights. The expression grouping is copied verbatim from qdot.
template <int bits, int values_per_thread>
inline void dequant_vals(const device uint8_t* w, thread float* d) {
  if constexpr (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      d[4 * i] = Dequantize<4>{}(ws[i]);
      d[4 * i + 1] = Dequantize<4>{}(ws[i] >> 4);
      d[4 * i + 2] = Dequantize<4>{}(ws[i] >> 8);
      d[4 * i + 3] = Dequantize<4>{}(ws[i] >> 12);
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      d[i] = Dequantize<8>{}(w[i]);
    }
  }
}

template <int bits, int values_per_thread>
inline float dot_vals(const thread float* x_thread, const thread float* d, float scale) {
  float accum = 0;
  if constexpr (bits == 4) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * d[4 * i] +
           x_thread[4 * i + 1] * d[4 * i + 1] +
           x_thread[4 * i + 2] * d[4 * i + 2] +
           x_thread[4 * i + 3] * d[4 * i + 3]);
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * d[i];
    }
  }
  return scale * accum;
}
"""

_SRC = r"""
    const int E_ = idx_shape[1];
    const int NPAIR = idx_shape[0] * E_;
    constexpr int PF = 32 / BITS;                      // values per uint32 pack
    constexpr int PPT = FAST ? 2 : 1;                  // packs per thread
    constexpr int VPT = PF * PPT;                      // values per thread per step
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    constexpr int ROWS = 4;                            // results per simdgroup
    typedef float U;
    typedef bfloat16_t T;

    uint3 tid = threadgroup_position_in_grid;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const int q = tid.z;
    const uint e = idx[q];

    // ---- group pairs by expert (no host sync): owner = first pair that picked this expert ----
    threadgroup uint pl[256];
    threadgroup int cnt_s;
    threadgroup int dup_s;
    if (simd_gid == 0) {
        int running = 0;
        bool dup = false;
        for (int base = 0; base < NPAIR; base += 32) {
            int p = base + (int)simd_lid;
            bool match = p < NPAIR && idx[p] == e;
            if (match && p < q) dup = true;
            int take = (match && p >= q) ? 1 : 0;
            int pre = simd_prefix_exclusive_sum(take);
            if (take) pl[running + pre] = (uint)p;
            running += simd_sum(take);
        }
        dup = simd_any(dup);
        if (simd_lid == 0) { cnt_s = running; dup_s = dup ? 1 : 0; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dup_s) return;
    const int cnt = cnt_s;

    const int in_w = KK * (BITS) / 8;                  // bytes per weight row
    const int in_g = KK / GS;                          // scales per row
    const int out_row = tid.y * (2 * ROWS) + simd_gid * ROWS;

    const device uint8_t* ws0 = (const device uint8_t*)w + (long)e * NN * in_w
        + (long)out_row * in_w + simd_lid * PPT * 4;
    const device uint8_t* sc0 = scales + (long)e * NN * in_g + (long)out_row * in_g + simd_lid / SSTEP;

    float inv_scale_enc = 1.0f;
    if constexpr (HGS) {
        inv_scale_enc = gsc[e] / (F8E4M3_MAX * F4E2M1_MAX);
    }

    for (int tb = 0; tb < cnt; tb += TMAX) {
        int xo[TMAX];
        int yo[TMAX];
        for (int t = 0; t < TMAX; t++) {
            int pq = (tb + t < cnt) ? (int)pl[tb + t] : q;
            xo[t] = (XPP ? pq : pq / E_) * KK + (int)simd_lid * VPT;
            yo[t] = pq * NN + out_row;
        }
        thread U result[TMAX][ROWS];
        for (int t = 0; t < TMAX; t++)
            for (int r = 0; r < ROWS; r++) result[t][r] = 0;

        const device uint8_t* ws = ws0;
        const device uint8_t* sc = sc0;
        int k = 0;
        constexpr int KEND_SUB = FAST ? 0 : BLOCK;
        for (; k < KK - KEND_SUB; k += BLOCK) {
            // dequantize this thread's weights of the 4 rows once, reuse for every token (same values as qdot's inline dequant)
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
            const int rem = clamp((int)(KK - k - (int)simd_lid * VPT), 0, VPT);
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
#pragma unroll
        for (int t = 0; t < TMAX; t++) {
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                U v = simd_sum(result[t][r]);
                if (simd_lid == 0 && tb + t < cnt) {
                    if constexpr (HGS) {
                        y[yo[t] + r] = static_cast<T>(v * inv_scale_enc);
                    } else {
                        y[yo[t] + r] = static_cast<T>(v);
                    }
                }
            }
        }
    }
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="rows_gather_fpq",
            input_names=["w", "scales", "gsc", "x", "idx"],
            output_names=["y"],
            source=_SRC,
            header=mlx_header(),
            ensure_row_contiguous=True,
        )
    return _kernel


_TMAX_CAP = int(os.environ.get("ROWS_GATHER_TMAX_CAP", "2"))


def _tmax(B):
    """Tokens per expert handled in one weight pass (register budget); a pair-count above it only costs a weight re-read."""
    for t in (1, 2, 4, 8, 16):
        if B <= t or t >= _TMAX_CAP:
            return t
    return 16


def gather_rows(x, w, scales, biases, idx, *, mode, group_size, bits, global_scale=None):
    """x: (B, K) shared per token or (B, E, K) per pair; idx (B, E) uint32; w (X, N, K*bits/32) uint32;
    scales (X, N, K/group_size) uint8. Returns (B, E, N) bf16 (bitwise = gather_qmm at M=1, token by token)."""
    if mode not in ("nvfp4", "mxfp8", "mxfp4"):
        raise ValueError(f"gather_rows supports fp-quantized modes only, got {mode!r}")
    if biases is not None:
        raise ValueError("fp-quantized modes have no biases")
    if x.dtype != mx.bfloat16:
        raise ValueError("x must be bfloat16")
    B, E = idx.shape
    if B * E > 256:
        raise ValueError("B*E must be <= 256")
    xpp = x.ndim == 3
    K = x.shape[-1]
    N = w.shape[1]
    if N % 8 != 0 or K % 32 != 0 or K != scales.shape[-1] * group_size or w.shape[2] * 32 // bits != K:
        raise ValueError("unsupported shape")
    if xpp and x.shape[:2] != (B, E) or (not xpp and x.shape[0] != B):
        raise ValueError("x / idx shape mismatch")
    fast = (N % 8 == 0) and (K % ((32 // bits) * 2 * 32) == 0)
    hgs = global_scale is not None
    gsc = global_scale.astype(mx.float32) if hgs else mx.zeros((1,), mx.float32)
    tm = _tmax(B)
    (y,) = _get_kernel()(
        inputs=[w, scales, gsc, x, idx.astype(mx.uint32)],
        template=[
            ("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast), ("TMAX", tm),
            ("XPP", xpp), ("KK", K), ("NN", N),
        ],
        grid=(32, 2 * (N // 8), B * E),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, E, N)],
        output_dtypes=[mx.bfloat16],
    )
    return y
