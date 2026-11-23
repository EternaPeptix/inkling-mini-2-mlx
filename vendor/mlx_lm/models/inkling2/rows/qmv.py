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


_TILE = r"""
// One 8-output tile (2 simdgroups x 4 rows) for B input rows; mirrors qmv_fast_impl<bfloat16_t, 64, 8, PR>.
template <int B, int K, bool PR>
METAL_FUNC void rows_qmv_tile(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device bfloat16_t* x,
    device bfloat16_t* y,
    const int N,
    const uint ty,
    const uint simd_gid,
    const uint simd_lid) {
  constexpr int group_size = 64;
  constexpr int bits = 8;
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  static_assert(K % block_size == 0, "rows qmv needs K % 256 == 0");

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[B][results_per_simdgroup];
  for (int b = 0; b < B; b++)
    for (int r = 0; r < results_per_simdgroup; r++) result[b][r] = 0;

  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  const int out_row = ty * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;

  int last_row = results_per_simdgroup - 1;
  if (PR) {
    if (out_row >= N) return;
    last_row = min(last_row, N - 1 - out_row);
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += out_row;

  for (int k = 0; k < K; k += block_size) {
    // weights, scales, biases of the 4 rows: loaded once, applied to all B rows
    thread uint8_t wreg[results_per_simdgroup][values_per_thread];
    thread U sc[results_per_simdgroup];
    thread U bi[results_per_simdgroup];
    for (int row = 0; row < results_per_simdgroup; row++) {
      int src = row;
      if (PR) src = min(row, last_row);
      auto wl = (const device uint8_t*)(ws + src * in_vec_size_w);
      const device bfloat16_t* sl = scales + src * in_vec_size_g;
      const device bfloat16_t* bl = biases + src * in_vec_size_g;
      for (int i = 0; i < values_per_thread; i++) wreg[row][i] = wl[i];
      sc[row] = sl[0];
      bi[row] = bl[0];
    }
    for (int b = 0; b < B; b++) {
      U sum = load_vector<bfloat16_t, U, values_per_thread, bits>(x + b * K, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        // qdot<U, values_per_thread, 8> verbatim, weights from registers
        U accum = 0;
        for (int i = 0; i < values_per_thread; i++) {
          accum += x_thread[i] * wreg[row][i];
        }
        result[b][row] += sc[row] * accum + sum * bi[row];
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int b = 0; b < B; b++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      U r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row) {
        y[b * N + row] = static_cast<bfloat16_t>(r);
      }
    }
  }
}
"""

_header_cache = None


def _header():
    global _header_cache
    if _header_cache is None:
        root = os.path.join(os.path.dirname(mx.__file__), "include")
        qh = os.path.join(root, "mlx/backend/metal/kernels/quantized.h")
        src = _expand_includes(qh, root, {qh})
        # keep only the scalar helpers (pack factors, load_vector, qdot, ...): everything from qouter on needs
        # steel/ headers that the custom-kernel preamble does not provide
        cut = src.index("inline void\nqouter")
        cut = src.rindex("template <", 0, cut)
        _header_cache = src[:cut] + _TILE
    return _header_cache


_kernels = {}


def _kernel(nw):
    k = _kernels.get(nw)
    if k is not None:
        return k
    ins = ["x"]
    for i in range(nw):
        ins += [f"w{i}", f"s{i}", f"b{i}"]
    outs = [f"y{i}" for i in range(nw)]
    lines = [
        "uint ty = threadgroup_position_in_grid.y;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint sl = thread_index_in_simdgroup;",
    ]
    prev = "0"
    for i in range(nw):
        end = f"{prev} + ((N{i} + 7) / 8)"
        cond = f"ty < {end}"
        pre = "if" if i == 0 else "else if"
        if i == nw - 1:
            cond = "true"
            pre = "else" if i else "if"
        head = f"{pre} ({cond})" if cond != "true" or i == 0 else "else"
        lines.append(
            f"{head} {{ rows_qmv_tile<B, K, (N{i} % 8) != 0>(w{i}, s{i}, b{i}, x, y{i}, N{i}, ty - ({prev}), sg, sl); }}"
        )
        prev = end
    k = mx.fast.metal_kernel(
        name=f"rows_qmv_aff8_gs64_n{nw}",
        input_names=ins,
        output_names=outs,
        source="\n".join(lines),
        header=_header(),
        ensure_row_contiguous=True,
    )
    _kernels[nw] = k
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
    tmpl = [("B", B), ("K", K)]
    shapes = []
    tiles = 0
    for i, (w, s, b) in enumerate(weights):
        N = w.shape[0]
        if w.shape[1] * 4 != K or s.shape != (N, K // _GS) or b.shape != (N, K // _GS) or s.dtype != mx.bfloat16:
            raise ValueError(f"weight {i}: bad shapes/dtype w{w.shape} s{s.shape}")
        args += [w, s, b]
        tmpl.append((f"N{i}", N))
        shapes.append((B, N))
        tiles += (N + _ROWS_PER_TG - 1) // _ROWS_PER_TG
    return _kernel(nw)(
        inputs=args,
        template=tmpl,
        grid=(32, 2 * tiles, 1),
        threadgroup=(32, 2, 1),
        output_shapes=shapes,
        output_dtypes=[mx.bfloat16] * nw,
    )


def qmv_rows(x, w, scales, biases, group_size=64, bits=8):
    """(B, K) bf16 -> (B, N) bf16; row b bitwise == quantized_matmul(x[b].reshape(1,1,K), ...)."""
    return qmv_rows_multi(x, [(w, scales, biases)], group_size, bits)[0]
