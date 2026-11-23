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


_TILE_SRC = r"""
// One tile (NSG simdgroups x RPS rows) for RB input rows starting at row b0; mirrors qmv_fast_impl<bfloat16_t, 64, 8, PR>
// (per output row: same per-lane K partition, same accumulation order, same simd_sum), with RPS and the row group RB free.
// STG (prologue kernels only): the x block is computed once per threadgroup into threadgroup memory sx (double buffered).
template <int B, int K, bool PR, int RPS, int NSG, int RB, int STG>
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

  for (int k = 0; k < K; k += block_size) {
    thread U wreg[RPS][values_per_thread];
    thread U sc[RPS];
    thread U bi[RPS];
    if (active) {
      for (int row = 0; row < RPS; row++) {
        int src = row;
        if (PR) src = min(row, last_row);
        auto wl = (const device uint8_t*)(ws + src * in_vec_size_w);
        const device bfloat16_t* sl = scales + src * in_vec_size_g;
        const device bfloat16_t* bl = biases + src * in_vec_size_g;
        for (int i = 0; i < values_per_thread; i++) wreg[row][i] = static_cast<U>(wl[i]);
        sc[row] = sl[0];
        bi[row] = bl[0];
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
            // qdot<U, values_per_thread, 8> verbatim, weights from registers
            U accum = 0;
            for (int i = 0; i < values_per_thread; i++) {
              accum += xtb[i] * wreg[row][i];
            }
            result[b][row] += sc[row] * accum + smb * bi[row];
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

_XLOAD_PLAIN = "smb = load_vector<bfloat16_t, U, values_per_thread, bits>(x + b * K, xtb);"


def tile_source(xload=_XLOAD_PLAIN, stg_fill=""):
    """The tile text; `xload` computes smb and xtb[*] for row b of the current block (x, gt point at this lane's slice of
    block k, row 0), `stg_fill` runs once per block before the x loads (cooperative staging)."""
    assert _TILE_SRC.count("//@XLOAD@") == 1 and _TILE_SRC.count("//@STG_FILL@") == 1
    return _TILE_SRC.replace("//@XLOAD@", xload).replace("//@STG_FILL@", stg_fill)


_TILE = tile_source()

# (RPS, NSG, RB, STG) per (B, pro): tuned table (see docs/ROWS_QMV_B.md).  LOOPLING_QMV_CFG="rps,nsg,rb[,stg]" overrides
# it for tuning runs only (STG is honoured by the prologue kernels in proj.py; qmv_rows ignores it).
_CFG_DEFAULT = (4, 2, 4, 0)


def _cfg(B, pro=False):
    ov = os.environ.get("LOOPLING_QMV_CFG")
    if ov:
        v = [int(t) for t in ov.split(",")]
        rps, nsg, rb = v[:3]
        stg = v[3] if len(v) > 3 else 0
        return rps, nsg, min(rb, B), stg if pro else 0
    rps, nsg, rb, stg = _CFG_DEFAULT
    return rps, nsg, B if rb is None else rb, stg if pro else 0


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
        "uint rb0 = threadgroup_position_in_grid.z * RB;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint sl = thread_index_in_simdgroup;",
    ]
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
            f"{head} {{ rows_qmv_tile<B, K, (N{i} % (RPS * NSG)) != 0, RPS, NSG, RB, 0>"
            f"(w{i}, s{i}, b{i}, x, x, y{i}, N{i}, ty - ({prev}), rb0, sg, sl, (threadgroup bfloat16_t*)0); }}"
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
    rps, nsg, rb, _ = _cfg(B)
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
    return _kernel(nw)(
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
