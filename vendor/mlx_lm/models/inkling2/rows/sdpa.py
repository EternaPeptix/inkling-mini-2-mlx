"""Row kernel: decode attention (one query token) for B rows, each with its OWN KV cache, in one launch, bitwise
identical per row to the B=1 path (`fastdecode._core`: mask from the relative position bias + `mx.fast.scaled_dot_product_attention`).

What is replicated (MLX 0.32.4, arch 'd' = applegpu_g15d, bf16 q/k/v, D=128, gqa 3, float mask, sinks):
  * S < 1024  -> `sdpa_vector`: one 1024-thread threadgroup per q head, 32 simdgroups, key i handled by simdgroup i % 32,
                 online softmax per simdgroup, then the max/sum/rescale combine through threadgroup memory.
  * S >= 1024 -> `sdpa_vector_2pass_1` + `_2`: 128 blocks (gqa 3 => n_simds 3 => blocks = 128 on arch d), key i handled by
                 block i % 128, 4-way unrolled prefetch loop + tail loop, bf16 partials, f32 sums/maxs, second pass reduces.
  The arithmetic of every (row, head, simdgroup) is copied statement by statement from MLX's sdpa_vector.h; the only
  difference is the work mapping: stage 1 threadgroups have 32 simdgroups when short rows are present, else gqa simdgroups
  (MLX's own 2-pass geometry); the 128*gqa (block, head) simdgroups of a kv head are spread over several groups (each
  simdgroup is independent there).
  So rows in different regimes share one launch (stage 1) and, if any row is 2-pass, one more launch (stage 2, which also
  passes the 1-pass rows through).
The mask is built inside the kernel from (end, rotation index, window): key slot i has position
((i - idx) mod S if rotating else i) + end - S, distance = end - 1 - position (same formulas as `_key_positions`).

Layout contract (the caches' own layout, nothing is copied): each ks[b] / vs[b] is (1, Hkv, S_b, D) with row stride D and
head stride cap*D where cap is the keys/values buffer capacity (`kv.keys.shape[2]`); pass cap in the row meta. Metal allows
31 buffer bindings, so at most MAX_ROWS (=11) rows go into one launch; larger B is split in chunks.
"""
import numpy as np

import mlx.core as mx

MAX_ROWS = 11
_TWO_PASS_N = 1024  # arch 'd': sdpa_vector_2pass when the key length >= 1024

_HEADER = r"""
template <typename T, typename PBT>
struct RowMask {
  const device PBT* pb;
  int S, end, idx, rot, R, window;
  // returns use_key; v = mask value (as the bf16 float mask would hold it) when use_key
  METAL_FUNC bool at(int i, thread float& v) const {
    int kp;
    if (rot) {
      int m = (i - idx) % S;
      if (m < 0) m += S;
      kp = m + (end - S);
    } else {
      kp = i + (end - S);
    }
    int dist = (end - 1) - kp;
    bool visible = (dist >= 0) && (kp >= 0);
    if (window > 0) visible = visible && (dist < window);
    if (!visible) return false;
    T mv = (dist < R) ? static_cast<T>(pb[dist]) : static_cast<T>(0.0f);
    v = static_cast<float>(mv);
    return mv >= Limits<T>::finite_min;
  }
};

// == sdpa_vector (single pass), one threadgroup = one (row, q head); verbatim arithmetic ==
template <typename T, typename PBT, int D>
METAL_FUNC void sdpa1_head(
    const device T* queries, const device T* keys, const device T* values, device T* out, const int N,
    const float scale, const float sink, const RowMask<T, PBT> msk, threadgroup float* outputs,
    threadgroup float* max_scores, threadgroup float* sum_exp_scores, uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int k_seq_stride = D;
  int inner_k_stride = BN * k_seq_stride;
  int inner_v_stride = BN * k_seq_stride;
  typedef float U;
  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];
  queries += simd_lid * qk_per_thread;
  keys += simd_gid * k_seq_stride + simd_lid * qk_per_thread;
  values += simd_gid * k_seq_stride + simd_lid * v_per_thread;
  out += simd_gid * v_per_thread;
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (simd_gid == 0) {
    max_score = sink;
    sum_exp_score = 1;
  }
  for (int i = simd_gid; i < N; i += BN) {
    float fm = 0;
    bool use_key = msk.at(i, fm);
    if (use_key) {
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      score += static_cast<U>(fm);
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }
    keys += inner_k_stride;
    values += inner_v_stride;
  }
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// == sdpa_vector_2pass_1 for ONE simdgroup = one (row, q head, block); verbatim arithmetic (TK = 4) ==
template <typename T, typename PBT, int D, int BLOCKS>
METAL_FUNC void sdpa2_block(
    const device T* queries, const device T* keys, const device T* values, device T* out, device float* sums,
    device float* maxs, const int N, const float scale, const float sink, const RowMask<T, PBT> msk,
    const int block_idx, uint simd_lid) {
  constexpr int TK = 4;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int k_seq_stride = D;
  constexpr int blocks = BLOCKS;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};
  queries += simd_lid * qk_per_thread;
  keys += block_idx * k_seq_stride + simd_lid * qk_per_thread;
  values += block_idx * k_seq_stride + simd_lid * v_per_thread;
  out += simd_lid * v_per_thread;
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (block_idx == 0) {
    max_score = sink;
    sum_exp_score = 1;
  }
  const int stride_k_tok = blocks * k_seq_stride;
  const int stride_v_tok = blocks * k_seq_stride;
  int i = block_idx;
  for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
    T k_reg[TK][qk_per_thread];
    T v_reg[TK][v_per_thread];
    for (int t = 0; t < TK; t++) {
      const device T* kp = keys + t * stride_k_tok;
      for (int j = 0; j < qk_per_thread; j++) {
        k_reg[t][j] = kp[j];
      }
    }
    for (int t = 0; t < TK; t++) {
      const device T* vp = values + t * stride_v_tok;
      for (int j = 0; j < v_per_thread; j++) {
        v_reg[t][j] = vp[j];
      }
    }
    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      float fm = 0;
      bool use_key = msk.at(ii, fm);
      if (use_key) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * static_cast<U>(k_reg[t][j]);
        }
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
        }
      }
    }
    keys += TK * stride_k_tok;
    values += TK * stride_v_tok;
  }
  for (; i < N; i += blocks) {
    float fm = 0;
    bool use_key = msk.at(i, fm);
    if (use_key) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * keys[j];
      }
      score = simd_sum(score);
      score += fm;
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }
    keys += stride_k_tok;
    values += stride_v_tok;
  }
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i2 = 0; i2 < v_per_thread; i2++) {
    out[i2] = static_cast<T>(o[i2]);
  }
}

// == sdpa_vector_2pass_2, one threadgroup = one (row, q head); verbatim arithmetic ==
template <typename T, int D>
METAL_FUNC void sdpa2_reduce(
    const device T* partials, const device float* sums, const device float* maxs, device T* out, const int blocks,
    threadgroup float* outputs, uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  typedef float U;
  thread U o[elem_per_thread] = {0};
  partials += simd_gid * D + simd_lid * elem_per_thread;
  out += simd_gid * elem_per_thread;
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
"""

# meta columns (int32, one row per batch row)
_MW = 6  # S, end, idx, rot, kcap, vcap

_STAGE1 = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;  // threadgroups per kv head in the 2-pass stage 1
    constexpr int NTG2 = HKV * TGPK;
    threadgroup float outputs[1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = 0, acc = 0;
    for (; b < NROWS; ++b) {
        int n = meta[b * MW] >= 1024 ? NTG2 : HQ;
        if ((int)t < acc + n) break;
        acc += n;
    }
    int tl = (int)t - acc;
    const int S = meta[b * MW + 0];
    const device T* K; const device T* V;
    SELECT_KV
    const int kcap = meta[b * MW + 4];
    const int vcap = meta[b * MW + 5];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[b * MW + 1]; msk.idx = meta[b * MW + 2]; msk.rot = meta[b * MW + 3];
    msk.R = R; msk.window = WIN;
    const float sc = scale[0];
    if (S < 1024) {
        int h = tl;
        int kvh = h / GQA;
        int bh = b * HQ + h;
        msk.pb = pb + (size_t)bh * R;
        sdpa1_head<T, PBT, D>(
            q + (size_t)bh * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D, o1 + (size_t)bh * D, S, sc,
            float(static_cast<T>(sinks[bh])), msk, outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
    } else {
        int kvh = tl / TGPK;
        int f = (tl % TGPK) * NS + (int)simd_gid;
        if (f < BLOCKS * GQA) {
            int block = f / GQA;
            int h = kvh * GQA + f % GQA;
            int bh = b * HQ + h;
            msk.pb = pb + (size_t)bh * R;
            sdpa2_block<T, PBT, D, BLOCKS>(
                q + (size_t)bh * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
                partials + ((size_t)bh * BLOCKS + block) * D, sums + (size_t)bh * BLOCKS + block,
                maxs + (size_t)bh * BLOCKS + block, S, sc, float(static_cast<T>(sinks[bh])), msk, block, simd_lid);
        }
    }
"""

_STAGE2 = r"""
    constexpr int D = 128;
    threadgroup float outputs[1024];
    uint bh = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint tix = thread_position_in_threadgroup.x;
    int b = bh / HQ;
    if (meta[b * MW] >= 1024) {
        sdpa2_reduce<T, D>(partials + (size_t)bh * BLOCKS * D, sums + (size_t)bh * BLOCKS, maxs + (size_t)bh * BLOCKS,
                           out + (size_t)bh * D, BLOCKS, outputs, simd_gid, simd_lid);
    } else if (tix < D) {
        out[(size_t)bh * D + tix] = o1[(size_t)bh * D + tix];
    }
"""

_kernels: dict = {}


def _blocks(N, n_simds):
    """sdpa_vector_2pass block count on arch 'd' (applegpu_g15d), copied from scaled_dot_product_attention.cpp."""
    blocks = 128
    if n_simds <= 2 and N > 8192:
        blocks = 256
    elif n_simds >= 6:
        if 16384 <= N < 65536:
            blocks = 512
        elif N >= 65536:
            blocks = 1024
    return blocks


def row_meta(kv):
    """(end, idx, rot, kcap, vcap) of a KVCache / RotatingKVCache after its update_and_fetch."""
    rot = hasattr(kv, "_idx")
    return (int(kv.offset), int(kv._idx) if rot else 0, 1 if rot else 0, int(kv.keys.shape[2]), int(kv.values.shape[2]))


def _get_kernels(B):
    ks = _kernels.get(B)
    if ks is None:
        sel = "switch (b) {\n" + "".join(
            f"        case {i}: K = k{i}; V = v{i}; break;\n" for i in range(B)
        ) + "        default: K = k0; V = v0; break;\n    }"
        names = ["q", "pb", "sinks", "meta", "scale"] + [f"k{i}" for i in range(B)] + [f"v{i}" for i in range(B)]
        s1 = mx.fast.metal_kernel(
            name=f"rows_sdpa_s1_b{B}", input_names=names, output_names=["o1", "partials", "sums", "maxs"],
            source=_STAGE1.replace("SELECT_KV", sel), header=_HEADER, ensure_row_contiguous=False,
        )
        s2 = mx.fast.metal_kernel(
            name="rows_sdpa_s2", input_names=["partials", "sums", "maxs", "o1", "meta"], output_names=["out"],
            source=_STAGE2, header=_HEADER, ensure_row_contiguous=False,
        )
        ks = _kernels[B] = (s1, s2)
    return ks


def sdpa_rows(q, ks, vs, pb, sinks, meta, *, scale, window):
    """q (B, Hq, 1, D) bf16; ks / vs: B arrays (1, Hkv, S_b, D) as returned by each row's update_and_fetch;
    pb (B, 1, Hq, R) or (B, Hq, R) (bf16 or f32), sinks (B, 1, Hq) or (B, Hq); meta: B tuples (end, idx, rot, kcap, vcap)
    (see row_meta); scale: python float; window: 0 (full attention) or the sliding window.
    Returns (B, 1, Hq, D) bf16, row b bitwise == fastdecode._core on that row alone."""
    B = q.shape[0]
    if B > MAX_ROWS:
        outs = []
        for s in range(0, B, MAX_ROWS):
            e = min(B, s + MAX_ROWS)
            outs.append(sdpa_rows(q[s:e], ks[s:e], vs[s:e], pb[s:e], sinks[s:e], meta[s:e], scale=scale, window=window))
        return mx.concatenate(outs, axis=0)
    assert q.dtype == mx.bfloat16 and q.shape[2] == 1 and q.shape[3] == 128
    Hq = q.shape[1]
    Hkv = ks[0].shape[1]
    gqa = Hq // Hkv
    assert Hq == gqa * Hkv and 3 <= gqa <= 5, "block count rule implemented for n_simds in 3..5 (blocks=128)"
    blocks = _blocks(max(k.shape[2] for k in ks), gqa)
    assert blocks == 128
    R = pb.shape[-1]
    Ss = [k.shape[2] for k in ks]
    mt = np.empty((B, _MW), np.int32)
    for i, (end, idx, rot, kcap, vcap) in enumerate(meta):
        mt[i] = (Ss[i], end, idx, rot, kcap, vcap)
    has2 = any(s >= _TWO_PASS_N for s in Ss)
    # threadgroup width: 32 simdgroups when short (single-pass) rows are present; otherwise GQA simdgroups per group
    # (MLX's own sdpa_vector_2pass_1 geometry; 96-thread groups balance better over the GPU cores than 1024-thread ones)
    ns = gqa if all(s >= _TWO_PASS_N for s in Ss) else 32
    tgpk = (blocks * gqa + ns - 1) // ns
    ntg = sum(Hkv * tgpk if s >= _TWO_PASS_N else Hq for s in Ss)
    s1, s2 = _get_kernels(B)
    qc = mx.contiguous(q)
    pbc = mx.contiguous(pb)
    sk = mx.contiguous(sinks)
    mta = mx.array(mt)
    sc = mx.array([scale], mx.float32)
    tmpl = [("T", mx.bfloat16), ("PBT", pb.dtype), ("NROWS", B), ("MW", _MW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
            ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns)]
    n2 = B * Hq * blocks if has2 else 1
    o1, partials, sums, maxs = s1(
        inputs=[qc, pbc, sk, mta, sc] + list(ks) + list(vs), template=tmpl, grid=(ntg * 32 * ns, 1, 1),
        threadgroup=(32 * ns, 1, 1), output_shapes=[(B, 1, Hq, 128), (n2 * 128 if has2 else 1,), (n2,), (n2,)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
    )
    if not has2:
        return o1
    (out,) = s2(
        inputs=[partials, sums, maxs, o1, mta], template=[("T", mx.bfloat16), ("MW", _MW), ("HQ", Hq), ("BLOCKS", blocks)],
        grid=(B * Hq * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(B, 1, Hq, 128)],
        output_dtypes=[mx.bfloat16],
    )
    return out
