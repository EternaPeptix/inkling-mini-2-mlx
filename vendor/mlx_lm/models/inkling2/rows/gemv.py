"""Row kernels (lane E4): exact multi-row versions of the small dense decode matmuls.

Row b of a B-row launch is bitwise identical to the reference op (`x[b:b+1] @ w.T`, M == 1) evaluated alone, and every
weight is read once per launch.

* gemv_rows(x, w)            x (B,K), w (N,K) as stored          -> (B,N)   [message-gate down/forget/message, router]
* gemv_rows_multi(x, [w..])  one shared input, several weights   -> [(B,N_i)]
* gemv_rows_group(xs, ws)    independent (x_i, w_i) problems     -> [(B,N_i)]  (e.g. the 4 gates of a layer in one launch)
* rel_logits_rows(r, proj)   r (B,24,16) -> (B,24,R)             [RelativeLogits]

Design: for M == 1, MLX (0.32.4, mlx/backend/metal/matmul.cpp `gemv_axbpy`) runs the non-transposed `gemv` kernel
(`GEMVKernel::run` in kernels/gemv.h) with a block config (bm,bn,sm,sn,tm,tn) picked from (N, K). We replicate that
config choice (`_cfg`) and the per-output arithmetic of `GEMVKernel::run` line by line (same per-thread K partition
`bn = thread*TN + i*blockN`, same row-major accumulation `result[tm] += inter[tn]*v[tn]`, same `simd_shuffle_down`
tree, same threadgroup reduction order over simdgroups), but each thread keeps B independent accumulator sets and
streams each weight element once for all B vectors. Because the order of operations per output element does not depend
on B, the result is bitwise equal to the M == 1 call.
"""

import mlx.core as mx

_KERNELS = {}

_HEADER = r"""
template <typename T, int B, int BM, int BN, int SM, int SN, int TM, int TN>
METAL_FUNC void gemv_rows_run(
    const device T* mat,
    const device T* in_vec,   // (B, K) row-major
    device T* out_vec,        // (B, N) row-major
    const int K,
    const int N,
    threadgroup float* tgp_memory,
    uint tgx,
    uint simd_gid,
    uint simd_lid) {
  constexpr int threadsM = BM * SM;
  constexpr int threadsN = BN * SN;
  constexpr int blockM = threadsM * TM;
  constexpr int blockN = threadsN * TN;

  float result[B][TM];
  for (int b = 0; b < B; b++)
    for (int tm = 0; tm < TM; tm++) result[b][tm] = 0.0f;
  T inter[TN];
  float v_coeff[B][TN];

  const int thrM = SN != 32 ? int(simd_lid) / SN : 0;
  const int thrN = SN != 32 ? int(simd_lid) % SN : int(simd_lid);
  const int sgN = BN != 1 ? int(simd_gid % BN) : 0;
  const int simdM = BN != 1 ? SM * int(simd_gid / BN) : int(SM * simd_gid);
  const int simdN = BN != 1 ? SN * int(simd_gid % BN) : 0;
  int bm = (simdM + thrM) * TM;
  int bn = (simdN + thrN) * TN;

  int out_row = int(tgx) * blockM + bm;
  if (out_row >= N) return;
  out_row = out_row + TM <= N ? out_row : N - TM;
  mat += size_t(out_row) * K;

  const int n_iter = K / blockN;
  const int last_iter = blockN * n_iter;
  const int leftover = K - last_iter;

  for (int i = 0; i < n_iter; ++i) {
    for (int b = 0; b < B; b++)
      for (int tn = 0; tn < TN; tn++)
        v_coeff[b][tn] = static_cast<float>(in_vec[b * K + bn + tn]);
    int mat_offset = 0;
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++) inter[tn] = mat[mat_offset + bn + tn];
      for (int b = 0; b < B; b++)
        for (int tn = 0; tn < TN; tn++)
          result[b][tm] += inter[tn] * v_coeff[b][tn];
      mat_offset += K;
    }
    bn += blockN;
  }

  if (leftover > 0) {
    for (int b = 0; b < B; b++)
      for (int tn = 0; tn < TN; tn++)
        v_coeff[b][tn] = (bn + tn < K) ? static_cast<float>(in_vec[b * K + bn + tn]) : 0.0f;
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++)
        inter[tn] = (bn + tn < K) ? mat[tm * K + bn + tn] : T(0);
      for (int b = 0; b < B; b++)
        for (int tn = 0; tn < TN; tn++)
          result[b][tm] += inter[tn] * v_coeff[b][tn];
    }
  }

  for (int b = 0; b < B; b++)
    for (int tm = 0; tm < TM; tm++)
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1)
        result[b][tm] += simd_shuffle_down(result[b][tm], sn);

  if (BN > 1) {
    // per-b slab: BN * (blockM + TM) floats, same indexing as MLX
    threadgroup float* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
    if (thrN == 0) {
      for (int b = 0; b < B; b++)
        for (int tm = 0; tm < TM; tm++)
          tgp_results[b * (BN * (blockM + TM)) + tm] = result[b][tm];
      threadgroup_barrier(mem_flags::mem_none);
      if (sgN == 0) {
        for (int sgn = 1; sgn < BN; sgn++)
          for (int b = 0; b < B; b++)
            for (int tm = 0; tm < TM; tm++)
              result[b][tm] += tgp_results[b * (BN * (blockM + TM)) + sgn * (blockM + TM) + tm];
      }
    }
  }

  if (simdN == 0 && thrN == 0) {
    for (int b = 0; b < B; b++)
      for (int tm = 0; tm < TM; tm++)
        out_vec[b * N + out_row + tm] = static_cast<T>(result[b][tm]);
  }
}
"""


def _cfg(N, K):
    """Block config MLX 0.32.4 picks for M == 1, weight stored (N, K) (non-transposed gemv kernel)."""
    tm, tn, sm, sn, bm, bn = 4, 4, 1, 32, 1, 1
    bm = 8 if N >= 4096 else 4
    sn = 32
    if K <= 64:
        bm, sm, sn = 1, 8, 4
    elif K >= 16 * N:
        bm, bn = 1, 8
    tm = 1 if N < tm else tm
    return bm, bn, sm, sn, tm, tn


def _build(dtype, B, G, Ns, K, cfg):
    key = (dtype, B, G, tuple(Ns), K, cfg)
    if key in _KERNELS:
        return _KERNELS[key]
    bm, bn, sm, sn, tm, tn = cfg
    blockM = bm * sm * tm
    rb = min(B, 8)
    nslab = (B + rb - 1) // rb
    tg_floats = rb * bn * (blockM + tm) if bn > 1 else 1
    ntg_per_slab = sum((N + blockM - 1) // blockM for N in Ns)
    if nslab == 1:
        lines = [
            "threadgroup float tgp_mem[%d];" % tg_floats,
            "const uint t = threadgroup_position_in_grid.x;",
        ]
        start = 0
        for g, N in enumerate(Ns):
            nt = (N + blockM - 1) // blockM
            lines.append(
                "if (t < %d) { gemv_rows_run<T, %d, %d, %d, %d, %d, %d, %d>(w%d, x%d, o%d, %d, %d, tgp_mem, t - %d, "
                "simdgroup_index_in_threadgroup, thread_index_in_simdgroup); return; }"
                % (start + nt, B, bm, bn, sm, sn, tm, tn, g, g, g, K, N, start)
            )
            start += nt
    else:
        q, rem = divmod(B, rb)
        lines = [
            "threadgroup float tgp_mem[%d];" % tg_floats,
            "const uint t_raw = threadgroup_position_in_grid.x;",
            "const uint slab = t_raw / %du;" % ntg_per_slab,
            "const uint t = t_raw %% %du;" % ntg_per_slab,
            "const size_t b0 = size_t(slab) * %du;" % rb,
        ]
        start = 0
        for g, N in enumerate(Ns):
            nt = (N + blockM - 1) // blockM
            if rem == 0:
                lines.append(
                    "if (t < %d) { gemv_rows_run<T, %d, %d, %d, %d, %d, %d, %d>(w%d, x%d + b0 * %d, o%d + b0 * %d, %d, %d, tgp_mem, t - %d, "
                    "simdgroup_index_in_threadgroup, thread_index_in_simdgroup); return; }"
                    % (start + nt, rb, bm, bn, sm, sn, tm, tn, g, g, K, g, N, K, N, start)
                )
            else:
                lines.append(
                    "if (t < %d) { if (slab < %du) gemv_rows_run<T, %d, %d, %d, %d, %d, %d, %d>(w%d, x%d + b0 * %d, o%d + b0 * %d, %d, %d, tgp_mem, t - %d, "
                    "simdgroup_index_in_threadgroup, thread_index_in_simdgroup); else gemv_rows_run<T, %d, %d, %d, %d, %d, %d, %d>(w%d, x%d + b0 * %d, o%d + b0 * %d, %d, %d, tgp_mem, t - %d, "
                    "simdgroup_index_in_threadgroup, thread_index_in_simdgroup); return; }"
                    % (
                        start + nt, q, rb, bm, bn, sm, sn, tm, tn, g, g, K, g, N, K, N, start,
                        rem, bm, bn, sm, sn, tm, tn, g, g, K, g, N, K, N, start,
                    )
                )
            start += nt
    kern = mx.fast.metal_kernel(
        name="e4_gemv_rows_%s_b%d_g%d_k%d_%s" % (str(dtype).split(".")[-1], B, G, K, "_".join(map(str, Ns + list(cfg)))),
        input_names=["w%d" % g for g in range(G)] + ["x%d" % g for g in range(G)],
        output_names=["o%d" % g for g in range(G)],
        header=_HEADER,
        source="\n".join(lines),
    )
    _KERNELS[key] = (kern, nslab * ntg_per_slab)
    return _KERNELS[key]


def gemv_rows_group(xs, ws):
    """Independent problems out_i = xs[i] @ ws[i].T with xs[i] (B,K), ws[i] (N_i,K), all one dtype and K.
    One launch when all problems share a block config, otherwise one launch per config class."""
    xs, ws = list(xs), list(ws)
    B, K = xs[0].shape
    dt = ws[0].dtype
    assert B >= 1 and all(x.shape == (B, K) and x.dtype == dt for x in xs), "x must be (B,K) in the weight dtype"
    assert all(w.ndim == 2 and w.shape[1] == K and w.dtype == dt for w in ws)
    cfgs = [_cfg(w.shape[0], K) for w in ws]
    if len(set(cfgs)) > 1:
        out = [None] * len(ws)
        for c in dict.fromkeys(cfgs):
            idx = [i for i, cc in enumerate(cfgs) if cc == c]
            for i, o in zip(idx, gemv_rows_group([xs[i] for i in idx], [ws[i] for i in idx])):
                out[i] = o
        return out
    cfg = cfgs[0]
    Ns = [w.shape[0] for w in ws]
    G = len(ws)
    kern, ntg = _build(dt, B, G, Ns, K, cfg)
    bm, bn, sm, sn, tm, tn = cfg
    outs = kern(
        inputs=list(ws) + list(xs),
        template=[("T", dt)],
        grid=(ntg * 32, bn, bm),
        threadgroup=(32, bn, bm),
        output_shapes=[(B, n) for n in Ns],
        output_dtypes=[dt] * G,
    )
    return list(outs)


def gemv_rows(x, w):
    """x (B,K), w (N,K) as stored -> (B,N), row b bitwise = (x[b:b+1] @ w.T) at M == 1."""
    return gemv_rows_group([x], [w])[0]


def gemv_rows_multi(x, ws):
    """Shared input x (B,K) against several weights (e.g. forget + message gate) -> list of (B,N_i)."""
    return gemv_rows_group([x] * len(ws), list(ws))


def rel_logits_rows(r, proj):
    """r (B,24,16) bf16, proj (16,R) bf16 -> (B,24,R), row b bitwise = RelativeLogits on r[b:b+1].

    MLX picks `steel_gemm_fused_nn_bf16_bf16_bm64_bn64_bk16_wm1_wn2` for the single-row call (M=24, N=R, K=16) and the
    same tile config for every M*N < 2**20 (bf16, 2*max(M,N) > K, nn): per-element arithmetic (two 8x8x8 MMA k-steps
    of one 16-wide K tile) does not depend on M, so stacking the B rows into one M=24*B GEMM reads proj once and is
    bitwise equal per row. Guarded so the tile config cannot silently change."""
    B, M, K = r.shape
    N = proj.shape[1]
    assert r.dtype == proj.dtype == mx.bfloat16 and proj.shape[0] == K
    assert B * M * N < (1 << 20) and 2 * max(B * M, N) > K, "outside the config range verified identical to M=24"
    return (r.reshape(B * M, K) @ proj).reshape(B, M, N)
