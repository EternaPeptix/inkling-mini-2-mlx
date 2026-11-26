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
import math
import os
import weakref

import numpy as np

import mlx.core as mx

from .glue import GLUE_HEADER, flit
from .mid import _MID_HEADER, _default_conv_anc, mid_causal_w_flat

MAX_ROWS = 10  # 31-buffer Metal limit: 6 fixed inputs + 2*10 KV + 4 outputs = 30 <= 31
_TWO_PASS_N = 1024  # arch 'd': sdpa_vector_2pass when the key length >= 1024

_HEADER = GLUE_HEADER + r"""
template <typename T, typename PBT>
struct RowMask {
  const device PBT* pb;
  int S, end, idx, rot, R, window;
  // returns use_key; v = mask value (as the bf16 float mask would hold it) when use_key
  METAL_FUNC bool at(int i, thread float& v) const {
    int kp;
    if (rot) {
      int m = i - idx;
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
template <typename T, typename PBT, int D, bool HAS_GATE, typename GT>
METAL_FUNC void sdpa1_head(
    const device T* queries, const device T* keys, const device T* values, GT gate, device T* out,
    const int N, const float scale, const float sink, const RowMask<T, PBT> msk, threadgroup float* outputs,
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
  gate += simd_gid * v_per_thread;
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
    outputs[(i * BD + simd_lid) * BN + simd_gid] = o[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = simd_sum(outputs[(i * BD + simd_gid) * BN + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
  }
  if (simd_lid == 0) {
    if constexpr (HAS_GATE) {
      for (int i = 0; i < v_per_thread; i++) {
        T ov = static_cast<T>(o[i]);
        out[i] = static_cast<T>(glue_gate(static_cast<float>(ov), static_cast<float>(gate[i])));
      }
    } else {
      for (int i = 0; i < v_per_thread; i++) {
        out[i] = static_cast<T>(o[i]);
      }
    }
  }
}

// == sdpa_vector (single pass, NS physical simdgroups simulating BN=32 virtual simdgroups); verbatim arithmetic ==
template <typename T, typename PBT, int D, int NS, bool HAS_GATE, typename GT>
METAL_FUNC void sdpa1_head_v32(
    const device T* queries, const device T* keys, const device T* values, GT gate, device T* out,
    const int N, const float scale, const float sink, const RowMask<T, PBT> msk, threadgroup float* outputs,
    threadgroup float* max_scores, threadgroup float* sum_exp_scores, uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int NV = BN / NS;
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
  if constexpr (D == 128) {
    bfloat4 qb = as_type<bfloat4>(((const device uint2*)queries)[0]);
    q[0] = static_cast<U>(scale) * qb.x;
    q[1] = static_cast<U>(scale) * qb.y;
    q[2] = static_cast<U>(scale) * qb.z;
    q[3] = static_cast<U>(scale) * qb.w;
  } else {
    for (int i = 0; i < qk_per_thread; i++) {
      q[i] = static_cast<U>(scale) * queries[i];
    }
  }
  for (int v = 0; v < NV; v++) {
    int vg = (int)simd_gid + v * NS;
    const device T* kp = keys + vg * k_seq_stride + simd_lid * qk_per_thread;
    const device T* vp = values + vg * k_seq_stride + simd_lid * v_per_thread;
    for (int i = 0; i < v_per_thread; i++) {
      o[i] = 0;
    }
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    if (vg == 0) {
      max_score = sink;
      sum_exp_score = 1;
    }
    for (int i = vg; i < N; i += BN) {
      float fm = 0;
      bool use_key = msk.at(i, fm);
      if (use_key) {
        U score = 0;
        if constexpr (D == 128) {
          bfloat4 kb = as_type<bfloat4>(((const device uint2*)kp)[0]);
          score += q[0] * kb.x;
          score += q[1] * kb.y;
          score += q[2] * kb.z;
          score += q[3] * kb.w;
        } else {
          for (int j = 0; j < qk_per_thread; j++) {
            k[j] = kp[j];
          }
          for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
          }
        }
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        if constexpr (D == 128) {
          bfloat4 vb = as_type<bfloat4>(((const device uint2*)vp)[0]);
          o[0] = o[0] * factor + exp_score * vb.x;
          o[1] = o[1] * factor + exp_score * vb.y;
          o[2] = o[2] * factor + exp_score * vb.z;
          o[3] = o[3] * factor + exp_score * vb.w;
        } else {
          for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vp[j];
          }
        }
      }
      kp += inner_k_stride;
      vp += inner_v_stride;
    }
    if (simd_lid == 0) {
      max_scores[vg] = max_score;
      sum_exp_scores[vg] = sum_exp_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
      outputs[(i * BD + simd_lid) * BN + vg] = o[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int f = (int)simd_gid; f < BD; f += NS) {
    for (int i = 0; i < v_per_thread; i++) {
      o[i] = simd_sum(outputs[(i * BD + f) * BN + simd_lid] * factor);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    }
    if (simd_lid == 0) {
      int base = f * v_per_thread;
      if constexpr (D == 128) {
        if constexpr (HAS_GATE) {
          bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
          bfloat4 ob = bfloat4(
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[0])), static_cast<float>(gb.x))),
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[1])), static_cast<float>(gb.y))),
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[2])), static_cast<float>(gb.z))),
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[3])), static_cast<float>(gb.w))));
          ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
        } else {
          bfloat4 ob = bfloat4(static_cast<T>(o[0]), static_cast<T>(o[1]), static_cast<T>(o[2]), static_cast<T>(o[3]));
          ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
        }
      } else if constexpr (HAS_GATE) {
        for (int i = 0; i < v_per_thread; i++) {
          T ov = static_cast<T>(o[i]);
          out[base + i] = static_cast<T>(glue_gate(static_cast<float>(ov), static_cast<float>(gate[base + i])));
        }
      } else {
        for (int i = 0; i < v_per_thread; i++) {
          out[base + i] = static_cast<T>(o[i]);
        }
      }
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
  if constexpr (D == 128) {
    bfloat4 qb = as_type<bfloat4>(((const device uint2*)queries)[0]);
    q[0] = static_cast<U>(scale) * qb.x;
    q[1] = static_cast<U>(scale) * qb.y;
    q[2] = static_cast<U>(scale) * qb.z;
    q[3] = static_cast<U>(scale) * qb.w;
  } else {
    for (int i = 0; i < qk_per_thread; i++) {
      q[i] = static_cast<U>(scale) * queries[i];
    }
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
    if constexpr (D == 128) {
      bfloat4 k_reg[TK];
      bfloat4 v_reg[TK];
      for (int t = 0; t < TK; t++) {
        k_reg[t] = as_type<bfloat4>(((const device uint2*)(keys + t * stride_k_tok))[0]);
      }
      for (int t = 0; t < TK; t++) {
        v_reg[t] = as_type<bfloat4>(((const device uint2*)(values + t * stride_v_tok))[0]);
      }
      for (int t = 0; t < TK; t++) {
        int ii = i + t * blocks;
        float fm = 0;
        bool use_key = msk.at(ii, fm);
        if (use_key) {
          U score = 0;
          score += q[0] * static_cast<U>(k_reg[t].x);
          score += q[1] * static_cast<U>(k_reg[t].y);
          score += q[2] * static_cast<U>(k_reg[t].z);
          score += q[3] * static_cast<U>(k_reg[t].w);
          score = simd_sum(score);
          score += static_cast<U>(fm);
          U new_max = max(max_score, score);
          U factor = fast::exp(max_score - new_max);
          U exp_score = fast::exp(score - new_max);
          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + exp_score;
          o[0] = o[0] * factor + exp_score * static_cast<U>(v_reg[t].x);
          o[1] = o[1] * factor + exp_score * static_cast<U>(v_reg[t].y);
          o[2] = o[2] * factor + exp_score * static_cast<U>(v_reg[t].z);
          o[3] = o[3] * factor + exp_score * static_cast<U>(v_reg[t].w);
        }
      }
    } else {
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
    }
    keys += TK * stride_k_tok;
    values += TK * stride_v_tok;
  }
  for (; i < N; i += blocks) {
    float fm = 0;
    bool use_key = msk.at(i, fm);
    if (use_key) {
      if constexpr (D == 128) {
        bfloat4 kb = as_type<bfloat4>(((const device uint2*)keys)[0]);
        U score = 0;
        score += q[0] * kb.x;
        score += q[1] * kb.y;
        score += q[2] * kb.z;
        score += q[3] * kb.w;
        score = simd_sum(score);
        score += fm;
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        bfloat4 vb = as_type<bfloat4>(((const device uint2*)values)[0]);
        o[0] = o[0] * factor + exp_score * vb.x;
        o[1] = o[1] * factor + exp_score * vb.y;
        o[2] = o[2] * factor + exp_score * vb.z;
        o[3] = o[3] * factor + exp_score * vb.w;
      } else {
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
    }
    keys += stride_k_tok;
    values += stride_v_tok;
  }
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  if constexpr (D == 128) {
    bfloat4 ob = bfloat4(static_cast<T>(o[0]), static_cast<T>(o[1]), static_cast<T>(o[2]), static_cast<T>(o[3]));
    ((device uint2*)out)[0] = as_type<uint2>(ob);
  } else {
    for (int i2 = 0; i2 < v_per_thread; i2++) {
      out[i2] = static_cast<T>(o[i2]);
    }
  }
}

// == sdpa_vector_2pass_2, one threadgroup = one (row, q head); verbatim arithmetic ==
template <typename T, int D, bool HAS_GATE, typename GT>
METAL_FUNC void sdpa2_reduce(
    const device T* partials, const device float* sums, const device float* maxs, GT gate, device T* out,
    const int blocks, threadgroup float* outputs, uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  typedef float U;
  thread U o[elem_per_thread] = {0};
  partials += simd_gid * D + simd_lid * elem_per_thread;
  gate += simd_gid * elem_per_thread;
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
    if constexpr (D == 128) {
      bfloat4 pb = as_type<bfloat4>(((const device uint2*)partials)[0]);
      o[0] += factor * static_cast<U>(pb.x);
      o[1] += factor * static_cast<U>(pb.y);
      o[2] += factor * static_cast<U>(pb.z);
      o[3] += factor * static_cast<U>(pb.w);
    } else {
      for (int i = 0; i < elem_per_thread; i++) {
        o[i] += factor * static_cast<U>(partials[i]);
      }
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[(i * BD + simd_lid) * BN + simd_gid] = o[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = simd_sum(outputs[(i * BD + simd_gid) * BN + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
  }
  if (simd_lid == 0) {
    if constexpr (D == 128) {
      if constexpr (HAS_GATE) {
        bfloat4 gb = as_type<bfloat4>(((const device uint2*)gate)[0]);
        bfloat4 ob = bfloat4(
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[0])), static_cast<float>(gb.x))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[1])), static_cast<float>(gb.y))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[2])), static_cast<float>(gb.z))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[3])), static_cast<float>(gb.w))));
        ((device uint2*)out)[0] = as_type<uint2>(ob);
      } else {
        bfloat4 ob = bfloat4(static_cast<T>(o[0]), static_cast<T>(o[1]), static_cast<T>(o[2]), static_cast<T>(o[3]));
        ((device uint2*)out)[0] = as_type<uint2>(ob);
      }
    } else if constexpr (HAS_GATE) {
      for (int i = 0; i < elem_per_thread; i++) {
        T ov = static_cast<T>(o[i]);
        out[i] = static_cast<T>(glue_gate(static_cast<float>(ov), static_cast<float>(gate[i])));
      }
    } else {
      for (int i = 0; i < elem_per_thread; i++) {
        out[i] = static_cast<T>(o[i]);
      }
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
    threadgroup float outputs[4 * 1024];
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
        sdpa1_head<T, PBT, D, HAS_GATE>(
            q + (size_t)bh * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
            gate + (size_t)bh * D, o1 + (size_t)bh * D, S, sc,
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
    threadgroup float outputs[4 * 1024];
    uint bh = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint tix = thread_position_in_threadgroup.x;
    int b = bh / HQ;
    if (meta[b * MW] >= 1024) {
        sdpa2_reduce<T, D, HAS_GATE>(partials + (size_t)bh * BLOCKS * D, sums + (size_t)bh * BLOCKS,
                                     maxs + (size_t)bh * BLOCKS, gate + (size_t)bh * D, out + (size_t)bh * D,
                                     BLOCKS, outputs, simd_gid, simd_lid);
    } else if (tix < D) {
        out[(size_t)bh * D + tix] = o1[(size_t)bh * D + tix];
    }
"""

_STAGE1_SHORT = r"""
    constexpr int D = 128;
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = (int)t / HQ;
    int h = (int)t % HQ;
    int kvh = h / GQA;
    int bh = (int)t;
    const int S = meta[b * MW + 0];
    const device T* K; const device T* V;
    SELECT_KV
    const int kcap = meta[b * MW + 4];
    const int vcap = meta[b * MW + 5];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[b * MW + 1]; msk.idx = meta[b * MW + 2]; msk.rot = meta[b * MW + 3];
    msk.R = R; msk.window = WIN;
    msk.pb = pb + (size_t)bh * R;
    const float sc = scale[0];
    sdpa1_head_v32<T, PBT, D, NS, HAS_GATE>(
        q + (size_t)bh * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
        gate + (size_t)bh * D, o1 + (size_t)bh * D, S, sc,
        float(static_cast<T>(sinks[bh])), msk, outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""

_STAGE1_2PASS = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;  // threadgroups per kv head in the 2-pass stage 1
    constexpr int NTG2 = HKV * TGPK;
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = (int)t / NTG2;
    int tl = (int)t % NTG2;
    const int S = meta[b * MW + 0];
    const device T* K; const device T* V;
    SELECT_KV
    const int kcap = meta[b * MW + 4];
    const int vcap = meta[b * MW + 5];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[b * MW + 1]; msk.idx = meta[b * MW + 2]; msk.rot = meta[b * MW + 3];
    msk.R = R; msk.window = WIN;
    const float sc = scale[0];
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
"""

_kernels: dict = {}
_DUMMY_GATE = mx.zeros((256,), mx.bfloat16)


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
        kv_names = [f"k{i}" for i in range(B)] + [f"v{i}" for i in range(B)]
        names = ["q", "pb", "sinks", "meta", "scale", "gate"] + kv_names
        names2 = ["q", "pb", "sinks", "meta", "scale"] + kv_names
        s1 = mx.fast.metal_kernel(
            name=f"rows_sdpa_s1_b{B}", input_names=names, output_names=["o1", "partials", "sums", "maxs"],
            source=_STAGE1.replace("SELECT_KV", sel), header=_HEADER, ensure_row_contiguous=False,
        )
        s2 = mx.fast.metal_kernel(
            name="rows_sdpa_s2", input_names=["partials", "sums", "maxs", "o1", "meta", "gate"], output_names=["out"],
            source=_STAGE2, header=_HEADER, ensure_row_contiguous=False,
        )
        s1_short = mx.fast.metal_kernel(
            name=f"rows_sdpa_s1_short_b{B}", input_names=names, output_names=["o1"],
            source=_STAGE1_SHORT.replace("SELECT_KV", sel), header=_HEADER, ensure_row_contiguous=False,
        )
        s1_2pass = mx.fast.metal_kernel(
            name=f"rows_sdpa_s1_2pass_b{B}", input_names=names2, output_names=["partials", "sums", "maxs"],
            source=_STAGE1_2PASS.replace("SELECT_KV", sel), header=_HEADER, ensure_row_contiguous=False,
        )
        ks = _kernels[B] = (s1, s2, s1_short, s1_2pass)
    return ks


_SCALE_CACHE = {}
_META_CACHE = {}


def sdpa_rows(q, ks, vs, pb, sinks, meta, *, scale, window, gate=None):
    """q (B, Hq, 1, D) bf16; ks / vs: B arrays (1, Hkv, S_b, D) as returned by each row's update_and_fetch;
    pb (B, 1, Hq, R) or (B, Hq, R) (bf16 or f32), sinks (B, 1, Hq) or (B, Hq); meta: B tuples (end, idx, rot, kcap, vcap)
    (see row_meta); scale: python float; window: 0 (full attention) or the sliding window;
    gate: optional (B, 1, Hq * D) or (B, Hq * D) bf16 to fuse `bf16(out) * sigmoid(gate)` on store.
    Returns (B, 1, Hq, D) bf16, row b bitwise == fastdecode._core (and _attend_post gate if provided)."""
    B = q.shape[0]
    if B > MAX_ROWS:
        outs = []
        for s in range(0, B, MAX_ROWS):
            e = min(B, s + MAX_ROWS)
            outs.append(sdpa_rows(q[s:e], ks[s:e], vs[s:e], pb[s:e], sinks[s:e], meta[s:e], scale=scale, window=window,
                                  gate=None if gate is None else gate[s:e]))
        return mx.concatenate(outs, axis=0)
    assert q.dtype == mx.bfloat16 and q.shape[2] == 1 and q.shape[3] == 128
    Hq = q.shape[1]
    Hkv = ks[0].shape[1]
    gqa = Hq // Hkv
    assert Hq == gqa * Hkv and 3 <= gqa <= 5, "block count rule implemented for n_simds in 3..5 (blocks=128)"
    blocks = _blocks(max(k.shape[2] for k in ks), gqa)
    assert blocks == 128
    R = pb.shape[-1]
    Ss = tuple(k.shape[2] for k in ks)
    mkey = (Ss, tuple(meta))
    mta = _META_CACHE.get(mkey)
    if mta is None:
        if len(_META_CACHE) >= 64:
            _META_CACHE.clear()
        mt = np.empty((B, _MW), np.int32)
        for i, (end, idx, rot, kcap, vcap) in enumerate(meta):
            mt[i] = (Ss[i], end, idx, rot, kcap, vcap)
        mta = _META_CACHE[mkey] = mx.array(mt)
    has2 = any(s >= _TWO_PASS_N for s in Ss)
    has_gate = gate is not None
    s1, s2, s1_short, s1_2pass = _get_kernels(B)
    qc = mx.contiguous(q)
    pbc = mx.contiguous(pb)
    sk = mx.contiguous(sinks)
    gt = mx.contiguous(gate) if has_gate else _DUMMY_GATE
    sck = float(scale)
    sc = _SCALE_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_CACHE[sck] = mx.array([sck], mx.float32)
    if not has2:
        ns = int(os.environ.get("LOOPLING_SDPA1_NS", "0")) or (16 if max(Ss) < 256 else 8)
        tmpl = [("T", mx.bfloat16), ("PBT", pb.dtype), ("NROWS", B), ("MW", _MW), ("HQ", Hq), ("GQA", gqa),
                ("R", R), ("WIN", int(window)), ("NS", ns), ("HAS_GATE", has_gate)]
        (o1,) = s1_short(
            inputs=[qc, pbc, sk, mta, sc, gt] + list(ks) + list(vs), template=tmpl,
            grid=(B * Hq * 32 * ns, 1, 1), threadgroup=(32 * ns, 1, 1),
            output_shapes=[(B, 1, Hq, 128)], output_dtypes=[mx.bfloat16],
        )
        return o1
    # threadgroup width: 32 simdgroups when short (single-pass) rows are present; otherwise GQA simdgroups per group
    # (MLX's own sdpa_vector_2pass_1 geometry; 96-thread groups balance better over the GPU cores than 1024-thread ones)
    all2 = all(s >= _TWO_PASS_N for s in Ss)
    ns = gqa if all2 else 32
    tgpk = (blocks * gqa + ns - 1) // ns
    n2 = B * Hq * blocks
    if all2:
        ntg = B * Hkv * tgpk
        tmpl = [("T", mx.bfloat16), ("PBT", pb.dtype), ("MW", _MW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns)]
        partials, sums, maxs = s1_2pass(
            inputs=[qc, pbc, sk, mta, sc] + list(ks) + list(vs), template=tmpl,
            grid=(ntg * 32 * ns, 1, 1), threadgroup=(32 * ns, 1, 1),
            output_shapes=[(n2 * 128,), (n2,), (n2,)], output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
        )
        o1 = _DUMMY_GATE
    else:
        ntg = sum(Hkv * tgpk if s >= _TWO_PASS_N else Hq for s in Ss)
        tmpl = [("T", mx.bfloat16), ("PBT", pb.dtype), ("NROWS", B), ("MW", _MW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns), ("HAS_GATE", has_gate)]
        o1, partials, sums, maxs = s1(
            inputs=[qc, pbc, sk, mta, sc, gt] + list(ks) + list(vs), template=tmpl, grid=(ntg * 32 * ns, 1, 1),
            threadgroup=(32 * ns, 1, 1), output_shapes=[(B, 1, Hq, 128), (n2 * 128,), (n2,), (n2,)],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
        )
    (out,) = s2(
        inputs=[partials, sums, maxs, o1, mta, gt],
        template=[("T", mx.bfloat16), ("MW", _MW), ("HQ", Hq), ("BLOCKS", blocks), ("HAS_GATE", has_gate)],
        grid=(B * Hq * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(B, 1, Hq, 128)],
        output_dtypes=[mx.bfloat16],
    )
    return out


_B1_PACKED_SHORT = r"""
    constexpr int D = 128;
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    int h = (int)threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int kvh = h / GQA;
    const int S = meta[0];
    const int cap = meta[4];
    const int lidx = meta[5];
    const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * cap * D;
    const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * cap * D;
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[1]; msk.idx = meta[2]; msk.rot = meta[3];
    msk.R = R; msk.window = WIN;
    msk.pb = pb + (size_t)h * R;
    const float sc = scale[0];
    float sink = float(static_cast<T>(qkv[SINK_OFF + h]));
    sdpa1_head_v32<T, PBT, D, NS, true>(
        q + (size_t)h * D, K, V, qkv + GATE_OFF + (size_t)h * D, out + (size_t)h * D,
        S, sc, sink, msk, outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""

_B1_PACKED_2P1 = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    uint tl = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const int S = meta[0];
    const int cap = meta[4];
    const int lidx = meta[5];
    int kvh = (int)tl / TGPK;
    int f = ((int)tl % TGPK) * NS + (int)simd_gid;
    if (f < BLOCKS * GQA) {
        int block = f / GQA;
        int h = kvh * GQA + f % GQA;
        const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * cap * D;
        const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * cap * D;
        RowMask<T, PBT> msk;
        msk.S = S; msk.end = meta[1]; msk.idx = meta[2]; msk.rot = meta[3];
        msk.R = R; msk.window = WIN;
        msk.pb = pb + (size_t)h * R;
        sdpa2_block<T, PBT, D, BLOCKS>(
            q + (size_t)h * D, K, V,
            partials + ((size_t)h * BLOCKS + block) * D, sums + (size_t)h * BLOCKS + block,
            maxs + (size_t)h * BLOCKS + block, S, scale[0],
            float(static_cast<T>(qkv[SINK_OFF + h])), msk, block, simd_lid);
    }
"""

_B1_PACKED_2P2 = r"""
    constexpr int D = 128;
    threadgroup float outputs[4 * 1024];
    uint h = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    sdpa2_reduce<T, D, true>(
        partials + (size_t)h * BLOCKS * D, sums + (size_t)h * BLOCKS,
        maxs + (size_t)h * BLOCKS, qkv + GATE_OFF + (size_t)h * D, out + (size_t)h * D,
        BLOCKS, outputs, simd_gid, simd_lid);
"""

_b1_packed_kernels = None
_b1_int8_kernels = None
_B1_META_CACHE: dict = {}


_KV_Q8_B1_SRC = r"""
    constexpr int D = 128;
    int t = (int)threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    int is_v = t / HKV;
    int h = t % HKV;
    const device T* src = (is_v ? v_in : k_in) + (size_t)h * D + lane * 4;
    bfloat4 xb = as_type<bfloat4>(((const device uint2*)src)[0]);
    float x0 = float(xb.x), x1 = float(xb.y), x2 = float(xb.z), x3 = float(xb.w);
    float amax = metal::max(metal::max(metal::abs(x0), metal::abs(x1)), metal::max(metal::abs(x2), metal::abs(x3)));
    amax = metal::simd_max(amax);
    float sc = metal::max(amax / 127.0f, 1e-12f);
    float inv = 1.0f / sc;
    char4 q = char4(
        int8_t(metal::rint(x0 * inv)),
        int8_t(metal::rint(x1 * inv)),
        int8_t(metal::rint(x2 * inv)),
        int8_t(metal::rint(x3 * inv)));
    device char4* qout = reinterpret_cast<device char4*>((is_v ? vq : kq) + (size_t)h * D + lane * 4);
    qout[0] = q;
    if (lane == 0) (is_v ? vs : ks)[h] = sc;
"""

_INT8_HEADER = _HEADER + r"""
template <typename T, typename PBT, int D, int NS, bool HAS_GATE>
METAL_FUNC void sdpa1_head_v32_int8(
    const device T* queries,
    const device int8_t* keys_q, const device int8_t* values_q,
    const device float* keys_s, const device float* values_s,
    const device T* gate, device T* out,
    const int N, const float scale, const float sink, const RowMask<T, PBT> msk,
    threadgroup float* outputs, threadgroup float* max_scores, threadgroup float* sum_exp_scores,
    uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int NV = BN / NS;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread];
  queries += simd_lid * qk_per_thread;
  bfloat4 qb = as_type<bfloat4>(((const device uint2*)queries)[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  for (int v = 0; v < NV; v++) {
    int vg = (int)simd_gid + v * NS;
    const device int8_t* kp = keys_q + (size_t)vg * D + simd_lid * qk_per_thread;
    const device int8_t* vp = values_q + (size_t)vg * D + simd_lid * v_per_thread;
    for (int i = 0; i < v_per_thread; i++) o[i] = 0;
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    if (vg == 0) {
      max_score = sink;
      sum_exp_score = 1;
    }
    for (int i = vg; i < N; i += BN) {
      float fm = 0;
      if (msk.at(i, fm)) {
        char4 kb = ((const device char4*)kp)[0];
        U score = q[0] * kb.x + q[1] * kb.y + q[2] * kb.z + q[3] * kb.w;
        score = simd_sum(score) * keys_s[i] + static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        char4 vb = ((const device char4*)vp)[0];
        U ev = exp_score * values_s[i];
        o[0] = o[0] * factor + ev * vb.x;
        o[1] = o[1] * factor + ev * vb.y;
        o[2] = o[2] * factor + ev * vb.z;
        o[3] = o[3] * factor + ev * vb.w;
      }
      kp += BN * D;
      vp += BN * D;
    }
    if (simd_lid == 0) {
      max_scores[vg] = max_score;
      sum_exp_scores[vg] = sum_exp_score;
    }
    for (int i = 0; i < v_per_thread; i++) outputs[(i * BD + simd_lid) * BN + vg] = o[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int f = (int)simd_gid; f < BD; f += NS) {
    for (int i = 0; i < v_per_thread; i++) {
      o[i] = simd_sum(outputs[(i * BD + f) * BN + simd_lid] * factor);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    }
    if (simd_lid == 0) {
      int base = f * v_per_thread;
      if constexpr (HAS_GATE) {
        bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
        bfloat4 ob = bfloat4(
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[0])), static_cast<float>(gb.x))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[1])), static_cast<float>(gb.y))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[2])), static_cast<float>(gb.z))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[3])), static_cast<float>(gb.w))));
        ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
      } else {
        bfloat4 ob = bfloat4(static_cast<T>(o[0]), static_cast<T>(o[1]), static_cast<T>(o[2]), static_cast<T>(o[3]));
        ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
      }
    }
  }
}

template <typename T, typename PBT, int D, int BLOCKS>
METAL_FUNC void sdpa2_block_int8(
    const device T* queries,
    const device int8_t* keys_q, const device int8_t* values_q,
    const device float* keys_s, const device float* values_s,
    device T* out, device float* sums, device float* maxs,
    const int N, const float scale, const float sink, const RowMask<T, PBT> msk,
    const int block_idx, uint simd_lid) {
  constexpr int TK = 4;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int blocks = BLOCKS;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};
  queries += simd_lid * qk_per_thread;
  out += simd_lid * v_per_thread;
  bfloat4 qb = as_type<bfloat4>(((const device uint2*)queries)[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (block_idx == 0) {
    max_score = sink;
    sum_exp_score = 1;
  }
  const int loff = (int)simd_lid * qk_per_thread;
  const device int8_t* kp = keys_q + (size_t)block_idx * D + loff;
  const device int8_t* vp = values_q + (size_t)block_idx * D + loff;
  const int stride_tok = blocks * D;
  int i = block_idx;
  for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
    char4 k_reg[TK], v_reg[TK];
    float ks_reg[TK], vs_reg[TK];
    for (int t = 0; t < TK; t++) {
      k_reg[t] = ((const device char4*)(kp + t * stride_tok))[0];
      ks_reg[t] = keys_s[i + t * blocks];
    }
    for (int t = 0; t < TK; t++) {
      v_reg[t] = ((const device char4*)(vp + t * stride_tok))[0];
      vs_reg[t] = values_s[i + t * blocks];
    }
    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      float fm = 0;
      if (msk.at(ii, fm)) {
        U score = q[0] * k_reg[t].x + q[1] * k_reg[t].y + q[2] * k_reg[t].z + q[3] * k_reg[t].w;
        score = simd_sum(score) * ks_reg[t] + static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        U ev = exp_score * vs_reg[t];
        o[0] = o[0] * factor + ev * v_reg[t].x;
        o[1] = o[1] * factor + ev * v_reg[t].y;
        o[2] = o[2] * factor + ev * v_reg[t].z;
        o[3] = o[3] * factor + ev * v_reg[t].w;
      }
    }
    kp += TK * stride_tok;
    vp += TK * stride_tok;
  }
  for (; i < N; i += blocks) {
    float fm = 0;
    if (msk.at(i, fm)) {
      char4 kb = ((const device char4*)kp)[0];
      U score = q[0] * kb.x + q[1] * kb.y + q[2] * kb.z + q[3] * kb.w;
      score = simd_sum(score) * keys_s[i] + static_cast<U>(fm);
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      char4 vb = ((const device char4*)vp)[0];
      U ev = exp_score * values_s[i];
      o[0] = o[0] * factor + ev * vb.x;
      o[1] = o[1] * factor + ev * vb.y;
      o[2] = o[2] * factor + ev * vb.z;
      o[3] = o[3] * factor + ev * vb.w;
    }
    kp += stride_tok;
    vp += stride_tok;
  }
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i2 = 0; i2 < v_per_thread; i2++) out[i2] = static_cast<T>(o[i2]);
}
"""

_B1_INT8_SHORT = r"""
    constexpr int D = 128;
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    int h = (int)threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int kvh = h / GQA;
    const int S = meta[0];
    const int cap = meta[2];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[1]; msk.idx = 0; msk.rot = 0;
    msk.R = R; msk.window = WIN;
    msk.pb = pb + (size_t)h * R;
    sdpa1_head_v32_int8<T, PBT, D, NS, true>(
        q + (size_t)h * D,
        kq + (size_t)kvh * cap * D, vq + (size_t)kvh * cap * D,
        ks + (size_t)kvh * cap, vs + (size_t)kvh * cap,
        qkv + GATE_OFF + (size_t)h * D, out + (size_t)h * D,
        S, scale[0], float(static_cast<T>(qkv[SINK_OFF + h])), msk,
        outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""

_B1_INT8_2P1 = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    uint tl = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const int S = meta[0];
    const int cap = meta[2];
    int kvh = (int)tl / TGPK;
    int f = ((int)tl % TGPK) * NS + (int)simd_gid;
    if (f < BLOCKS * GQA) {
        int block = f / GQA;
        int h = kvh * GQA + f % GQA;
        RowMask<T, PBT> msk;
        msk.S = S; msk.end = meta[1]; msk.idx = 0; msk.rot = 0;
        msk.R = R; msk.window = WIN;
        msk.pb = pb + (size_t)h * R;
        sdpa2_block_int8<T, PBT, D, BLOCKS>(
            q + (size_t)h * D,
            kq + (size_t)kvh * cap * D, vq + (size_t)kvh * cap * D,
            ks + (size_t)kvh * cap, vs + (size_t)kvh * cap,
            partials + ((size_t)h * BLOCKS + block) * D,
            sums + (size_t)h * BLOCKS + block,
            maxs + (size_t)h * BLOCKS + block,
            S, scale[0], float(static_cast<T>(qkv[SINK_OFF + h])), msk, block, simd_lid);
    }
"""


def _get_b1_int8_kernels():
    global _b1_int8_kernels
    if _b1_int8_kernels is None:
        q_k = mx.fast.metal_kernel(
            name="quantize_kv_b1_int8",
            input_names=["k_in", "v_in"],
            output_names=["kq", "vq", "ks", "vs"],
            source=_KV_Q8_B1_SRC,
            header=_HEADER,
            ensure_row_contiguous=True,
        )
        s1_short = mx.fast.metal_kernel(
            name="b1_int8_sdpa_short",
            input_names=["q", "kq", "vq", "ks", "vs", "pb", "qkv", "meta", "scale"],
            output_names=["out"],
            source=_B1_INT8_SHORT,
            header=_INT8_HEADER,
            ensure_row_contiguous=True,
        )
        s1_2p = mx.fast.metal_kernel(
            name="b1_int8_sdpa_2p1",
            input_names=["q", "kq", "vq", "ks", "vs", "pb", "qkv", "meta", "scale"],
            output_names=["partials", "sums", "maxs"],
            source=_B1_INT8_2P1,
            header=_INT8_HEADER,
            ensure_row_contiguous=True,
        )
        _get_b1_packed_kernels()
        s2_2p = _b1_packed_kernels[2]
        _b1_int8_kernels = (q_k, s1_short, s1_2p, s2_2p)
    return _b1_int8_kernels


def quantize_kv_b1_int8(k, v):
    """Quantize B=1, L=1 (1, Hkv, 128) or (1, Hkv, 1, 128) bf16 K/V -> (kq, vq int8, ks, vs f32)."""
    Hkv = k.shape[1]
    assert k.dtype == mx.bfloat16 and v.dtype == mx.bfloat16 and k.shape[-1] == 128
    q_k = _get_b1_int8_kernels()[0]
    kq, vq, ks, vs = q_k(
        inputs=[k.reshape(1, Hkv, 1, 128), v.reshape(1, Hkv, 1, 128)],
        template=[("T", mx.bfloat16), ("HKV", Hkv)],
        grid=(2 * Hkv * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, Hkv, 1, 128), (1, Hkv, 1, 128), (1, Hkv, 1), (1, Hkv, 1)],
        output_dtypes=[mx.int8, mx.int8, mx.float32, mx.float32],
    )
    return kq, vq, ks, vs


def sdpa_b1_int8(
    q, k_q, v_q, k_s, v_s, pb, qkv_flat, gate_off, sink_off, S, end, cap, *, scale, window=0, Hq=24, Hkv=8
):
    """B=1 decode attention directly over INT8 K/V buffers + per-head-token float32 scales, fused with gate."""
    assert q.dtype == mx.bfloat16 and k_q.dtype == mx.int8 and v_q.dtype == mx.int8 and qkv_flat.dtype == mx.bfloat16
    gqa = Hq // Hkv
    R = pb.shape[-1]
    sck = float(scale)
    sc = _SCALE_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_CACHE[sck] = mx.array([sck], mx.float32)
    mkey = ("q8", int(S), int(end), int(cap))
    mta = _B1_META_CACHE.get(mkey)
    if mta is None:
        if len(_B1_META_CACHE) >= 256:
            _B1_META_CACHE.clear()
        mta = _B1_META_CACHE[mkey] = mx.array((int(S), int(end), int(cap)), mx.int32)
    _, s1_short, s1_2p, s2_2p = _get_b1_int8_kernels()
    if S < _TWO_PASS_N:
        ns_env = os.environ.get("LOOPLING_SDPA1_NS", "")
        ns = int(ns_env) if ns_env and ns_env != "0" else 32
        (out,) = s1_short(
            inputs=[q, k_q, v_q, k_s, v_s, pb, qkv_flat, mta, sc],
            template=[
                ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                ("R", R), ("WIN", int(window)), ("NS", ns), ("GATE_OFF", int(gate_off)), ("SINK_OFF", int(sink_off)),
            ],
            grid=(Hq * 32 * ns, 1, 1),
            threadgroup=(32 * ns, 1, 1),
            output_shapes=[(1, Hq * 128)],
            output_dtypes=[mx.bfloat16],
        )
        return out
    blocks = 128 if S < 8192 else (256 if S < 32768 else 512)
    ns = gqa
    tgpk = (blocks * gqa + ns - 1) // ns
    ntg = Hkv * tgpk
    partials, sums, maxs = s1_2p(
        inputs=[q, k_q, v_q, k_s, v_s, pb, qkv_flat, mta, sc],
        template=[
            ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
            ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns), ("SINK_OFF", int(sink_off)),
        ],
        grid=(ntg * 32 * ns, 1, 1),
        threadgroup=(32 * ns, 1, 1),
        output_shapes=[(Hq * blocks * 128,), (Hq * blocks,), (Hq * blocks,)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
    )
    (out,) = s2_2p(
        inputs=[partials, sums, maxs, qkv_flat],
        template=[("T", mx.bfloat16), ("BLOCKS", blocks), ("GATE_OFF", int(gate_off))],
        grid=(Hq * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, Hq * 128)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def _get_b1_packed_kernels():
    global _b1_packed_kernels
    if _b1_packed_kernels is None:
        s1_short = mx.fast.metal_kernel(
            name="b1_packed_sdpa_short",
            input_names=["q", "kv", "pb", "qkv", "meta", "scale"],
            output_names=["out"],
            source=_B1_PACKED_SHORT,
            header=_HEADER,
            ensure_row_contiguous=True,
        )
        s1_2p = mx.fast.metal_kernel(
            name="b1_packed_sdpa_2p1",
            input_names=["q", "kv", "pb", "qkv", "meta", "scale"],
            output_names=["partials", "sums", "maxs"],
            source=_B1_PACKED_2P1,
            header=_HEADER,
            ensure_row_contiguous=True,
        )
        s2_2p = mx.fast.metal_kernel(
            name="b1_packed_sdpa_2p2",
            input_names=["partials", "sums", "maxs", "qkv"],
            output_names=["out"],
            source=_B1_PACKED_2P2,
            header=_HEADER,
            ensure_row_contiguous=True,
        )
        _b1_packed_kernels = (s1_short, s1_2p, s2_2p)
    return _b1_packed_kernels


def sdpa_b1_packed(
    q, kv_buf, pb, qkv_flat, gate_off, sink_off, S, end, idx, rot, cap, *, scale, window, Hq=24, Hkv=8, kv_layer_idx=0,
    gate_stride=None, sink_stride=None,
):
    """Dedicated B=1 decode attention directly over packed head-major KV (2, Hkv, cap, 128) or stacked (NL, 2, Hkv, cap, 128)."""
    assert q.dtype == mx.bfloat16 and kv_buf.dtype == mx.bfloat16 and qkv_flat.dtype == mx.bfloat16
    gqa = Hq // Hkv
    assert Hq == gqa * Hkv and (
        (kv_buf.ndim == 4 and kv_buf.shape == (2, Hkv, cap, 128) and kv_layer_idx == 0)
        or (kv_buf.ndim == 5 and kv_buf.shape[1:] == (2, Hkv, cap, 128) and 0 <= kv_layer_idx < kv_buf.shape[0])
    )
    if _LONG_SDPA and int(window) == 0 and int(rot) == 0 and int(S) >= _LONG_SDPA_MIN_S:
        return sdpa_long_full_packed(
            q, kv_buf, None, pb, qkv_flat, gate_off, sink_off, S, S, cap, 1,
            scale=scale, Hq=Hq, Hkv=Hkv, kv_layer_idx=kv_layer_idx,
            gate_stride=gate_stride, sink_stride=sink_stride,
        )
    R = pb.shape[-1]
    sck = float(scale)
    sc = _SCALE_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_CACHE[sck] = mx.array([sck], mx.float32)
    mkey = (int(S), int(end), int(idx), int(rot), int(cap), int(kv_layer_idx))
    mta = _B1_META_CACHE.get(mkey)
    if mta is None:
        if len(_B1_META_CACHE) >= 256:
            _B1_META_CACHE.clear()
        mta = _B1_META_CACHE[mkey] = mx.array(mkey, mx.int32)
    s1_short, s1_2p, s2_2p = _get_b1_packed_kernels()
    if S < _TWO_PASS_N:
        ns_env = os.environ.get("LOOPLING_SDPA1_NS", "")
        ns = int(ns_env) if ns_env and ns_env != "0" else 32
        (out,) = s1_short(
            inputs=[q, kv_buf, pb, qkv_flat, mta, sc],
            template=[("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                      ("R", R), ("WIN", int(window)), ("NS", ns), ("GATE_OFF", int(gate_off)),
                      ("SINK_OFF", int(sink_off))],
            grid=(Hq * 32 * ns, 1, 1),
            threadgroup=(32 * ns, 1, 1),
            output_shapes=[(1, Hq * 128)],
            output_dtypes=[mx.bfloat16],
        )
        return out
    blocks = _blocks(S, gqa)
    ns = gqa
    tgpk = (blocks * gqa + ns - 1) // ns
    ntg = Hkv * tgpk
    partials, sums, maxs = s1_2p(
        inputs=[q, kv_buf, pb, qkv_flat, mta, sc],
        template=[("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                  ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns),
                  ("SINK_OFF", int(sink_off))],
        grid=(ntg * 32 * ns, 1, 1),
        threadgroup=(32 * ns, 1, 1),
        output_shapes=[(Hq * blocks * 128,), (Hq * blocks,), (Hq * blocks,)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
    )
    (out,) = s2_2p(
        inputs=[partials, sums, maxs, qkv_flat],
        template=[("T", mx.bfloat16), ("BLOCKS", blocks), ("GATE_OFF", int(gate_off))],
        grid=(Hq * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, Hq * 128)],
        output_dtypes=[mx.bfloat16],
    )
    return out


_CAUSAL_HEADER = _HEADER + r"""
template <typename T, typename PBT, int D, int NS, bool HAS_GATE, bool HAS_STAGE = false, int OPASS = 2>
METAL_FUNC void sdpa1_head_v32_causal(
    const device T* queries, const device T* K, const device T* V,
    const device T* Knew, const device T* Vnew, const device T* gate, device T* out,
    const int N, const float scale, const float sink, const RowMask<T, PBT> msk,
    const int db, const device int* anc_ptr, const int S0, const int slot0, const int max_size, const int rot,
    const device T* Kst, const device T* Vst, const int slot_main, const int st_len,
    threadgroup float* outputs, threadgroup float* max_scores, threadgroup float* sum_exp_scores,
    uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int NV = BN / NS;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int EP = v_per_thread / OPASS;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o_reg[NV][v_per_thread];
  const int loff = (int)simd_lid * qk_per_thread;
  queries += loff;
  bfloat4 qb = as_type<bfloat4>(((const device uint2*)queries)[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  for (int v = 0; v < NV; v++) {
    int vg = (int)simd_gid + v * NS;
    U o0 = 0, o1 = 0, o2 = 0, o3 = 0;
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    if (vg == 0) {
      max_score = sink;
      sum_exp_score = 1;
    }
    for (int i = vg; i < N; i += BN) {
      float fm = 0;
      if (msk.at(i, fm)) {
        int d = rot ? (i - slot0 + (i < slot0 ? max_size : 0)) : (i - S0);
        bool is_new = (d >= 0 && d <= db);
        const device T* kp;
        const device T* vp;
        if (is_new) {
          int ancd = anc_ptr[d];
          kp = Knew + (size_t)ancd * D + loff;
          vp = Vnew + (size_t)ancd * D + loff;
        } else if constexpr (HAS_STAGE) {
          int ds = rot ? (i - slot_main + (i < slot_main ? max_size : 0)) : (i - slot_main);
          if (ds >= 0 && ds < st_len) {
            kp = Kst + (size_t)ds * D + loff;
            vp = Vst + (size_t)ds * D + loff;
          } else {
            kp = K + (size_t)i * D + loff;
            vp = V + (size_t)i * D + loff;
          }
        } else {
          kp = K + (size_t)i * D + loff;
          vp = V + (size_t)i * D + loff;
        }
        bfloat4 kb = as_type<bfloat4>(((const device uint2*)kp)[0]);
        U score = 0;
        score += q[0] * kb.x;
        score += q[1] * kb.y;
        score += q[2] * kb.z;
        score += q[3] * kb.w;
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        bfloat4 vb = as_type<bfloat4>(((const device uint2*)vp)[0]);
        o0 = o0 * factor + exp_score * vb.x;
        o1 = o1 * factor + exp_score * vb.y;
        o2 = o2 * factor + exp_score * vb.z;
        o3 = o3 * factor + exp_score * vb.w;
      }
    }
    if (simd_lid == 0) {
      max_scores[vg] = max_score;
      sum_exp_scores[vg] = sum_exp_score;
    }
    if constexpr (OPASS == 1) {
      outputs[(0 * BD + simd_lid) * BN + vg] = o0;
      outputs[(1 * BD + simd_lid) * BN + vg] = o1;
      outputs[(2 * BD + simd_lid) * BN + vg] = o2;
      outputs[(3 * BD + simd_lid) * BN + vg] = o3;
    } else {
      o_reg[v][0] = o0;
      o_reg[v][1] = o1;
      o_reg[v][2] = o2;
      o_reg[v][3] = o3;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  if constexpr (OPASS == 1) {
    for (int f = (int)simd_gid; f < BD; f += NS) {
      U o[4];
      for (int i = 0; i < 4; i++) {
        o[i] = simd_sum(outputs[(i * BD + f) * BN + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      }
      if (simd_lid == 0) {
        int base = f * v_per_thread;
        if constexpr (HAS_GATE) {
          bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
          bfloat4 ob = bfloat4(
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[0])), static_cast<float>(gb.x))),
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[1])), static_cast<float>(gb.y))),
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[2])), static_cast<float>(gb.z))),
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[3])), static_cast<float>(gb.w))));
          ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
        } else {
          bfloat4 ob = bfloat4(static_cast<T>(o[0]), static_cast<T>(o[1]), static_cast<T>(o[2]), static_cast<T>(o[3]));
          ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
        }
      }
    }
  } else {
    for (int p = 0; p < OPASS; p++) {
      for (int e = 0; e < EP; e++) {
        int i = p * EP + e;
        for (int v = 0; v < NV; v++) {
          int vg = (int)simd_gid + v * NS;
          outputs[(e * BD + simd_lid) * BN + vg] = o_reg[v][i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int e = 0; e < EP; e++) {
        int i = p * EP + e;
        for (int fi = 0; fi < NV; fi++) {
          int f = (int)simd_gid + fi * NS;
          U r = simd_sum(outputs[(e * BD + f) * BN + simd_lid] * factor);
          o_reg[fi][i] = sum_exp_score == 0 ? r : (r / sum_exp_score);
        }
      }
      if (p + 1 < OPASS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    if (simd_lid == 0) {
      for (int fi = 0; fi < NV; fi++) {
        int base = ((int)simd_gid + fi * NS) * v_per_thread;
        if constexpr (HAS_GATE) {
          bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
          bfloat4 ob = bfloat4(
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o_reg[fi][0])), static_cast<float>(gb.x))),
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o_reg[fi][1])), static_cast<float>(gb.y))),
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o_reg[fi][2])), static_cast<float>(gb.z))),
              static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o_reg[fi][3])), static_cast<float>(gb.w))));
          ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
        } else {
          bfloat4 ob = bfloat4(static_cast<T>(o_reg[fi][0]), static_cast<T>(o_reg[fi][1]), static_cast<T>(o_reg[fi][2]), static_cast<T>(o_reg[fi][3]));
          ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
        }
      }
    }
  }
}

template <typename T, typename PBT, int D, int BLOCKS, bool HAS_STAGE = false>
METAL_FUNC void sdpa2_block_causal(
    const device T* queries, const device T* K, const device T* V,
    const device T* Knew, const device T* Vnew, device T* out, device float* sums,
    device float* maxs, const int N, const float scale, const float sink, const RowMask<T, PBT> msk,
    const int db, const device int* anc_ptr, const int S0, const int slot0, const int max_size, const int rot,
    const device T* Kst, const device T* Vst, const int slot_main, const int st_len,
    const int block_idx, uint simd_lid) {
  constexpr int TK = 4;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int blocks = BLOCKS;
  typedef float U;
  int anc[16];
  for (int t = 0; t <= db; t++) anc[t] = anc_ptr[t];
  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};
  const int loff = (int)simd_lid * qk_per_thread;
  queries += loff;
  out += loff;
  bfloat4 qb = as_type<bfloat4>(((const device uint2*)queries)[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (block_idx == 0) {
    max_score = sink;
    sum_exp_score = 1;
  }
  int i = block_idx;
  for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
    bfloat4 k_reg[TK];
    bfloat4 v_reg[TK];
    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      int d = rot ? (ii - slot0 + (ii < slot0 ? max_size : 0)) : (ii - S0);
      bool is_new = (d >= 0 && d <= db);
      const device T* kp;
      if (is_new) {
        kp = Knew + (size_t)anc[d] * D + loff;
      } else if constexpr (HAS_STAGE) {
        int ds = rot ? (ii - slot_main + (ii < slot_main ? max_size : 0)) : (ii - slot_main);
        kp = ((ds >= 0 && ds < st_len) ? (Kst + (size_t)ds * D) : (K + (size_t)ii * D)) + loff;
      } else {
        kp = K + (size_t)ii * D + loff;
      }
      k_reg[t] = as_type<bfloat4>(((const device uint2*)kp)[0]);
    }
    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      int d = rot ? (ii - slot0 + (ii < slot0 ? max_size : 0)) : (ii - S0);
      bool is_new = (d >= 0 && d <= db);
      const device T* vp;
      if (is_new) {
        vp = Vnew + (size_t)anc[d] * D + loff;
      } else if constexpr (HAS_STAGE) {
        int ds = rot ? (ii - slot_main + (ii < slot_main ? max_size : 0)) : (ii - slot_main);
        vp = ((ds >= 0 && ds < st_len) ? (Vst + (size_t)ds * D) : (V + (size_t)ii * D)) + loff;
      } else {
        vp = V + (size_t)ii * D + loff;
      }
      v_reg[t] = as_type<bfloat4>(((const device uint2*)vp)[0]);
    }
    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      float fm = 0;
      if (msk.at(ii, fm)) {
        U score = 0;
        score += q[0] * static_cast<U>(k_reg[t].x);
        score += q[1] * static_cast<U>(k_reg[t].y);
        score += q[2] * static_cast<U>(k_reg[t].z);
        score += q[3] * static_cast<U>(k_reg[t].w);
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        o[0] = o[0] * factor + exp_score * static_cast<U>(v_reg[t].x);
        o[1] = o[1] * factor + exp_score * static_cast<U>(v_reg[t].y);
        o[2] = o[2] * factor + exp_score * static_cast<U>(v_reg[t].z);
        o[3] = o[3] * factor + exp_score * static_cast<U>(v_reg[t].w);
      }
    }
  }
  for (; i < N; i += blocks) {
    float fm = 0;
    if (msk.at(i, fm)) {
      int d = rot ? (i - slot0 + (i < slot0 ? max_size : 0)) : (i - S0);
      bool is_new = (d >= 0 && d <= db);
      const device T* kp;
      const device T* vp;
      if (is_new) {
        kp = Knew + (size_t)anc[d] * D + loff;
        vp = Vnew + (size_t)anc[d] * D + loff;
      } else if constexpr (HAS_STAGE) {
        int ds = rot ? (i - slot_main + (i < slot_main ? max_size : 0)) : (i - slot_main);
        if (ds >= 0 && ds < st_len) {
          kp = Kst + (size_t)ds * D + loff;
          vp = Vst + (size_t)ds * D + loff;
        } else {
          kp = K + (size_t)i * D + loff;
          vp = V + (size_t)i * D + loff;
        }
      } else {
        kp = K + (size_t)i * D + loff;
        vp = V + (size_t)i * D + loff;
      }
      bfloat4 kb = as_type<bfloat4>(((const device uint2*)kp)[0]);
      U score = 0;
      score += q[0] * kb.x;
      score += q[1] * kb.y;
      score += q[2] * kb.z;
      score += q[3] * kb.w;
      score = simd_sum(score);
      score += fm;
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      bfloat4 vb = as_type<bfloat4>(((const device uint2*)vp)[0]);
      o[0] = o[0] * factor + exp_score * vb.x;
      o[1] = o[1] * factor + exp_score * vb.y;
      o[2] = o[2] * factor + exp_score * vb.z;
      o[3] = o[3] * factor + exp_score * vb.w;
    }
  }
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  bfloat4 ob = bfloat4(static_cast<T>(o[0]), static_cast<T>(o[1]), static_cast<T>(o[2]), static_cast<T>(o[3]));
  ((device uint2*)out)[0] = as_type<uint2>(ob);
}
"""

_CAUSAL_SHORT_SRC = r"""
    constexpr int D = 128;
    threadgroup float outputs[(4 / OPASS) * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    int tg = (int)threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b_idx, h, kvh;
    if constexpr (KVH_MAJ) {
        int gpk = B_CNT * GQA;
        kvh = tg / gpk;
        int rem = tg % gpk;
        b_idx = rem / GQA;
        h = kvh * GQA + (rem % GQA);
    } else {
        b_idx = tg / HQ;
        h = tg % HQ;
        kvh = h / GQA;
    }
    int bh_l = b_idx * HQ + h;
    int b = BOFF + b_idx;
    int bh = b * HQ + h;
    const int S0 = meta[0], end0 = meta[1], slot0 = meta[2], cap = meta[3], max_size = meta[4];
    const int lidx = (KV_LIDX >= 0) ? KV_LIDX : meta[5];
    constexpr int TMW = (WW > 8) ? 17 : 9;
    const int db = tree_meta[b * TMW];
    const int S_b = ROT ? min(end0 + db + 1, max_size) : (S0 + db + 1);
    const int end_b = end0 + db + 1;
    const int idx_b = ROT ? (((slot0 + db) % max_size) + 1) : 0;
    const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * cap * D;
    const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * cap * D;
    const device T* Knew = kv_new + (size_t)kvh * WW * D;
    const device T* Vnew = kv_new + (size_t)(HKV + kvh) * WW * D;
    RowMask<T, PBT> msk;
    msk.S = S_b; msk.end = end_b; msk.idx = idx_b; msk.rot = ROT;
    msk.R = R; msk.window = WIN; msk.pb = pb + (size_t)bh * R;
    float sink = float(static_cast<T>(qkv[SINK_OFF + (size_t)b * SINK_STRIDE + h]));
    sdpa1_head_v32_causal<T, PBT, D, NS, true, false, OPASS>(
        q + (size_t)bh * D, K, V, Knew, Vnew, qkv + GATE_OFF + (size_t)b * GATE_STRIDE + (size_t)h * D, out + (size_t)bh_l * D,
        S_b, scale[0], sink, msk, db, tree_meta + b * TMW + 1, S0, slot0, max_size, ROT,
        nullptr, nullptr, 0, 0,
        outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""

_CAUSAL_2P1_SRC = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    constexpr int NTG2 = HKV * TGPK;
    int t_all = (int)threadgroup_position_in_grid.x;
    int b_idx = t_all / NTG2;
    int tl = t_all % NTG2;
    int b = BOFF + b_idx;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const int S0 = meta[0], end0 = meta[1], slot0 = meta[2], cap = meta[3], max_size = meta[4];
    const int lidx = (KV_LIDX >= 0) ? KV_LIDX : meta[5];
    constexpr int TMW = (WW > 8) ? 17 : 9;
    const int db = tree_meta[b * TMW];
    const int S_b = ROT ? min(end0 + db + 1, max_size) : (S0 + db + 1);
    const int end_b = end0 + db + 1;
    const int idx_b = ROT ? (((slot0 + db) % max_size) + 1) : 0;
    int kvh = tl / TGPK;
    int f = (tl % TGPK) * NS + (int)simd_gid;
    if (f < BLOCKS * GQA) {
        int block = f / GQA;
        int h = kvh * GQA + f % GQA;
        int bh = b * HQ + h;
        int bh_l = b_idx * HQ + h;
        const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * cap * D;
        const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * cap * D;
        const device T* Knew = kv_new + (size_t)kvh * WW * D;
        const device T* Vnew = kv_new + (size_t)(HKV + kvh) * WW * D;
        RowMask<T, PBT> msk;
        msk.S = S_b; msk.end = end_b; msk.idx = idx_b; msk.rot = ROT;
        msk.R = R; msk.window = WIN; msk.pb = pb + (size_t)bh * R;
        sdpa2_block_causal<T, PBT, D, BLOCKS>(
            q + (size_t)bh * D, K, V, Knew, Vnew,
            partials + ((size_t)bh_l * BLOCKS + block) * D, sums + (size_t)bh_l * BLOCKS + block,
            maxs + (size_t)bh_l * BLOCKS + block, S_b, scale[0],
            float(static_cast<T>(qkv[SINK_OFF + (size_t)b * SINK_STRIDE + h])), msk,
            db, tree_meta + b * TMW + 1, S0, slot0, max_size, ROT, nullptr, nullptr, 0, 0, block, simd_lid);
    }
"""

_CAUSAL_2P1_GQA3_SRC = r"""
    constexpr int D = 128;
    constexpr int TK = 4;
    constexpr int blocks = BLOCKS;
    int tg = (int)threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int kvh = tg / BLOCKS;
    int block_idx = tg % BLOCKS;
    int b_idx = (int)simd_gid;
    if (b_idx >= B_CNT) return;
    int b = BOFF + b_idx;
    int h0 = kvh * 3;
    const int S0 = meta[0], end0 = meta[1], slot0 = meta[2], cap = meta[3], max_size = meta[4];
    const int lidx = (KV_LIDX >= 0) ? KV_LIDX : meta[5];
    constexpr int TMW = (WW > 8) ? 17 : 9;
    const int db = tree_meta[b * TMW];
    const int N = ROT ? min(end0 + db + 1, max_size) : (S0 + db + 1);
    const int end_b = end0 + db + 1;
    const int idx_b = ROT ? (((slot0 + db) % max_size) + 1) : 0;
    const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * cap * D;
    const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * cap * D;
    const device T* Knew = kv_new + (size_t)kvh * WW * D;
    const device T* Vnew = kv_new + (size_t)(HKV + kvh) * WW * D;
    const device int* anc_ptr = tree_meta + b * TMW + 1;
    const int loff = (int)simd_lid * 4;
    const float sc = scale[0];
    typedef float U;
    thread U q0[4], q1[4], q2[4];
    {
        int bh0 = b * HQ + h0;
        bfloat4 qb0 = as_type<bfloat4>(((const device uint2*)(q + (size_t)(bh0 + 0) * D + loff))[0]);
        bfloat4 qb1 = as_type<bfloat4>(((const device uint2*)(q + (size_t)(bh0 + 1) * D + loff))[0]);
        bfloat4 qb2 = as_type<bfloat4>(((const device uint2*)(q + (size_t)(bh0 + 2) * D + loff))[0]);
        q0[0] = sc * qb0.x; q0[1] = sc * qb0.y; q0[2] = sc * qb0.z; q0[3] = sc * qb0.w;
        q1[0] = sc * qb1.x; q1[1] = sc * qb1.y; q1[2] = sc * qb1.z; q1[3] = sc * qb1.w;
        q2[0] = sc * qb2.x; q2[1] = sc * qb2.y; q2[2] = sc * qb2.z; q2[3] = sc * qb2.w;
    }
    RowMask<T, PBT> msk0;
    msk0.S = N; msk0.end = end_b; msk0.idx = idx_b; msk0.rot = ROT; msk0.R = R; msk0.window = WIN;
    RowMask<T, PBT> msk1 = msk0, msk2 = msk0;
    msk0.pb = pb + (size_t)(b * HQ + h0 + 0) * R;
    msk1.pb = pb + (size_t)(b * HQ + h0 + 1) * R;
    msk2.pb = pb + (size_t)(b * HQ + h0 + 2) * R;
    thread U o0[4] = {0, 0, 0, 0}, o1[4] = {0, 0, 0, 0}, o2[4] = {0, 0, 0, 0};
    U m0 = Limits<U>::finite_min, m1 = m0, m2 = m0;
    U s0 = 0, s1 = 0, s2 = 0;
    if (block_idx == 0) {
        m0 = float(static_cast<T>(qkv[SINK_OFF + (size_t)b * SINK_STRIDE + h0 + 0])); s0 = 1;
        m1 = float(static_cast<T>(qkv[SINK_OFF + (size_t)b * SINK_STRIDE + h0 + 1])); s1 = 1;
        m2 = float(static_cast<T>(qkv[SINK_OFF + (size_t)b * SINK_STRIDE + h0 + 2])); s2 = 1;
    }
    auto do_key = [&](int ii, bfloat4 kb, bfloat4 vb) __attribute__((always_inline)) {
        float fm0 = 0, fm1 = 0, fm2 = 0;
        bool u0 = msk0.at(ii, fm0);
        bool u1 = msk1.at(ii, fm1);
        bool u2 = msk2.at(ii, fm2);
        if (u0) {
            U score = 0;
            score += q0[0] * static_cast<U>(kb.x); score += q0[1] * static_cast<U>(kb.y);
            score += q0[2] * static_cast<U>(kb.z); score += q0[3] * static_cast<U>(kb.w);
            score = simd_sum(score) + static_cast<U>(fm0);
            U nm = max(m0, score); U f = fast::exp(m0 - nm); U e = fast::exp(score - nm);
            m0 = nm; s0 = s0 * f + e;
            o0[0] = o0[0] * f + e * static_cast<U>(vb.x); o0[1] = o0[1] * f + e * static_cast<U>(vb.y);
            o0[2] = o0[2] * f + e * static_cast<U>(vb.z); o0[3] = o0[3] * f + e * static_cast<U>(vb.w);
        }
        if (u1) {
            U score = 0;
            score += q1[0] * static_cast<U>(kb.x); score += q1[1] * static_cast<U>(kb.y);
            score += q1[2] * static_cast<U>(kb.z); score += q1[3] * static_cast<U>(kb.w);
            score = simd_sum(score) + static_cast<U>(fm1);
            U nm = max(m1, score); U f = fast::exp(m1 - nm); U e = fast::exp(score - nm);
            m1 = nm; s1 = s1 * f + e;
            o1[0] = o1[0] * f + e * static_cast<U>(vb.x); o1[1] = o1[1] * f + e * static_cast<U>(vb.y);
            o1[2] = o1[2] * f + e * static_cast<U>(vb.z); o1[3] = o1[3] * f + e * static_cast<U>(vb.w);
        }
        if (u2) {
            U score = 0;
            score += q2[0] * static_cast<U>(kb.x); score += q2[1] * static_cast<U>(kb.y);
            score += q2[2] * static_cast<U>(kb.z); score += q2[3] * static_cast<U>(kb.w);
            score = simd_sum(score) + static_cast<U>(fm2);
            U nm = max(m2, score); U f = fast::exp(m2 - nm); U e = fast::exp(score - nm);
            m2 = nm; s2 = s2 * f + e;
            o2[0] = o2[0] * f + e * static_cast<U>(vb.x); o2[1] = o2[1] * f + e * static_cast<U>(vb.y);
            o2[2] = o2[2] * f + e * static_cast<U>(vb.z); o2[3] = o2[3] * f + e * static_cast<U>(vb.w);
        }
    };
    int i = block_idx;
    for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
        bfloat4 kr[TK], vr[TK];
        for (int t = 0; t < TK; t++) {
            int ii = i + t * blocks;
            int d = ROT ? (ii - slot0 + (ii < slot0 ? max_size : 0)) : (ii - S0);
            const device T* kp = (d >= 0 && d <= db) ? (Knew + (size_t)anc_ptr[d] * D + loff) : (K + (size_t)ii * D + loff);
            kr[t] = as_type<bfloat4>(((const device uint2*)kp)[0]);
        }
        for (int t = 0; t < TK; t++) {
            int ii = i + t * blocks;
            int d = ROT ? (ii - slot0 + (ii < slot0 ? max_size : 0)) : (ii - S0);
            const device T* vp = (d >= 0 && d <= db) ? (Vnew + (size_t)anc_ptr[d] * D + loff) : (V + (size_t)ii * D + loff);
            vr[t] = as_type<bfloat4>(((const device uint2*)vp)[0]);
        }
        for (int t = 0; t < TK; t++) do_key(i + t * blocks, kr[t], vr[t]);
    }
    for (; i < N; i += blocks) {
        int d = ROT ? (i - slot0 + (i < slot0 ? max_size : 0)) : (i - S0);
        const device T* kp = (d >= 0 && d <= db) ? (Knew + (size_t)anc_ptr[d] * D + loff) : (K + (size_t)i * D + loff);
        const device T* vp = (d >= 0 && d <= db) ? (Vnew + (size_t)anc_ptr[d] * D + loff) : (V + (size_t)i * D + loff);
        do_key(i, as_type<bfloat4>(((const device uint2*)kp)[0]), as_type<bfloat4>(((const device uint2*)vp)[0]));
    }
    size_t p0 = (size_t)(b_idx * HQ + h0 + 0) * BLOCKS + block_idx;
    size_t p1 = (size_t)(b_idx * HQ + h0 + 1) * BLOCKS + block_idx;
    size_t p2 = (size_t)(b_idx * HQ + h0 + 2) * BLOCKS + block_idx;
    if (simd_lid == 0) {
        sums[p0] = s0; maxs[p0] = m0;
        sums[p1] = s1; maxs[p1] = m1;
        sums[p2] = s2; maxs[p2] = m2;
    }
    ((device uint2*)(partials + p0 * D + loff))[0] =
        as_type<uint2>(bfloat4(static_cast<T>(o0[0]), static_cast<T>(o0[1]), static_cast<T>(o0[2]), static_cast<T>(o0[3])));
    ((device uint2*)(partials + p1 * D + loff))[0] =
        as_type<uint2>(bfloat4(static_cast<T>(o1[0]), static_cast<T>(o1[1]), static_cast<T>(o1[2]), static_cast<T>(o1[3])));
    ((device uint2*)(partials + p2 * D + loff))[0] =
        as_type<uint2>(bfloat4(static_cast<T>(o2[0]), static_cast<T>(o2[1]), static_cast<T>(o2[2]), static_cast<T>(o2[3])));
"""

_CAUSAL_2PFUSED_SRC = r"""
    constexpr int D = 128;
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int TK = 4;
    constexpr int blocks = BLOCKS;
    constexpr int NB = BLOCKS / BN;
    constexpr int NV = BN / NS;
    constexpr int EP = 4 / OPASS;
    threadgroup float outputs[EP * 1024];
    threadgroup float sh_max[BLOCKS];
    threadgroup float sh_sum[BLOCKS];
    int tg = (int)threadgroup_position_in_grid.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int gpk = B_CNT * GQA;
    int kvh = tg / gpk;
    int rem = tg % gpk;
    int b_idx = rem / GQA;
    int h = kvh * GQA + (rem % GQA);
    int b = BOFF + b_idx;
    int bh = b * HQ + h;
    int bh_l = b_idx * HQ + h;
    const int S0 = meta[0], end0 = meta[1], slot0 = meta[2], cap = meta[3], max_size = meta[4];
    const int lidx = (KV_LIDX >= 0) ? KV_LIDX : meta[5];
    constexpr int TMW = (WW > 8) ? 17 : 9;
    const int db = tree_meta[b * TMW];
    const int N = ROT ? min(end0 + db + 1, max_size) : (S0 + db + 1);
    const int end_b = end0 + db + 1;
    const int idx_b = ROT ? (((slot0 + db) % max_size) + 1) : 0;
    const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * cap * D;
    const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * cap * D;
    const device T* Knew = kv_new + (size_t)kvh * WW * D;
    const device T* Vnew = kv_new + (size_t)(HKV + kvh) * WW * D;
    const device int* anc_ptr = tree_meta + b * TMW + 1;
    const int loff = (int)lane * 4;
    const float scl = scale[0];
    bfloat4 qb = as_type<bfloat4>(((const device uint2*)(q + (size_t)bh * D + loff))[0]);
    float q0 = scl * qb.x, q1 = scl * qb.y, q2 = scl * qb.z, q3 = scl * qb.w;
    RowMask<T, PBT> msk;
    msk.S = N; msk.end = end_b; msk.idx = idx_b; msk.rot = ROT;
    msk.R = R; msk.window = WIN; msk.pb = pb + (size_t)bh * R;
    float sink = float(static_cast<T>(qkv[SINK_OFF + (size_t)b * SINK_STRIDE + h]));
    bfloat4 part[NV][NB];
    for (int v = 0; v < NV; v++) {
        int vg = (int)sg + v * NS;
        for (int bi = 0; bi < NB; bi++) {
            int block_idx = vg + BN * bi;
            float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
            float max_score = Limits<float>::finite_min;
            float sum_exp_score = 0;
            if (block_idx == 0) { max_score = sink; sum_exp_score = 1; }
            int i = block_idx;
            for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
                bfloat4 kr[TK], vr[TK];
                for (int t = 0; t < TK; t++) {
                    int ii = i + t * blocks;
                    int d = ROT ? (ii - slot0 + (ii < slot0 ? max_size : 0)) : (ii - S0);
                    const device T* kp = (d >= 0 && d <= db) ? (Knew + (size_t)anc_ptr[d] * D + loff) : (K + (size_t)ii * D + loff);
                    kr[t] = as_type<bfloat4>(((const device uint2*)kp)[0]);
                }
                for (int t = 0; t < TK; t++) {
                    int ii = i + t * blocks;
                    int d = ROT ? (ii - slot0 + (ii < slot0 ? max_size : 0)) : (ii - S0);
                    const device T* vp = (d >= 0 && d <= db) ? (Vnew + (size_t)anc_ptr[d] * D + loff) : (V + (size_t)ii * D + loff);
                    vr[t] = as_type<bfloat4>(((const device uint2*)vp)[0]);
                }
                for (int t = 0; t < TK; t++) {
                    float fm = 0;
                    if (msk.at(i + t * blocks, fm)) {
                        float score = 0;
                        score += q0 * float(kr[t].x); score += q1 * float(kr[t].y);
                        score += q2 * float(kr[t].z); score += q3 * float(kr[t].w);
                        score = simd_sum(score) + fm;
                        float nm = max(max_score, score);
                        float fac = fast::exp(max_score - nm);
                        float e = fast::exp(score - nm);
                        max_score = nm;
                        sum_exp_score = sum_exp_score * fac + e;
                        o0 = o0 * fac + e * float(vr[t].x); o1 = o1 * fac + e * float(vr[t].y);
                        o2 = o2 * fac + e * float(vr[t].z); o3 = o3 * fac + e * float(vr[t].w);
                    }
                }
            }
            for (; i < N; i += blocks) {
                float fm = 0;
                if (msk.at(i, fm)) {
                    int d = ROT ? (i - slot0 + (i < slot0 ? max_size : 0)) : (i - S0);
                    const device T* kp = (d >= 0 && d <= db) ? (Knew + (size_t)anc_ptr[d] * D + loff) : (K + (size_t)i * D + loff);
                    const device T* vp = (d >= 0 && d <= db) ? (Vnew + (size_t)anc_ptr[d] * D + loff) : (V + (size_t)i * D + loff);
                    bfloat4 kb = as_type<bfloat4>(((const device uint2*)kp)[0]);
                    float score = 0;
                    score += q0 * kb.x; score += q1 * kb.y; score += q2 * kb.z; score += q3 * kb.w;
                    score = simd_sum(score) + fm;
                    float nm = max(max_score, score);
                    float fac = fast::exp(max_score - nm);
                    float e = fast::exp(score - nm);
                    max_score = nm;
                    sum_exp_score = sum_exp_score * fac + e;
                    bfloat4 vb = as_type<bfloat4>(((const device uint2*)vp)[0]);
                    o0 = o0 * fac + e * vb.x; o1 = o1 * fac + e * vb.y;
                    o2 = o2 * fac + e * vb.z; o3 = o3 * fac + e * vb.w;
                }
            }
            if (lane == 0) { sh_max[block_idx] = max_score; sh_sum[block_idx] = sum_exp_score; }
            part[v][bi] = bfloat4(static_cast<T>(o0), static_cast<T>(o1), static_cast<T>(o2), static_cast<T>(o3));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_score = Limits<float>::finite_min;
    for (int bi = 0; bi < NB; ++bi) max_score = max(max_score, sh_max[lane + BN * bi]);
    max_score = simd_max(max_score);
    float sum_exp_score = 0.0f;
    for (int bi = 0; bi < NB; ++bi) sum_exp_score += fast::exp(sh_max[lane + BN * bi] - max_score) * sh_sum[lane + BN * bi];
    sum_exp_score = simd_sum(sum_exp_score);
    float o_reg[NV][4];
    for (int v = 0; v < NV; v++) {
        int vg = (int)sg + v * NS;
        float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
        for (int bi = 0; bi < NB; ++bi) {
            float fac = fast::exp(sh_max[vg + BN * bi] - max_score);
            bfloat4 pbv = part[v][bi];
            o0 += fac * float(pbv.x); o1 += fac * float(pbv.y);
            o2 += fac * float(pbv.z); o3 += fac * float(pbv.w);
        }
        o_reg[v][0] = o0; o_reg[v][1] = o1; o_reg[v][2] = o2; o_reg[v][3] = o3;
    }
    float out_r[NV][4];
    for (int pass = 0; pass < OPASS; pass++) {
        int i0 = pass * EP;
        for (int v = 0; v < NV; v++) {
            int vg = (int)sg + v * NS;
            for (int ii = 0; ii < EP; ii++) outputs[(ii * BD + lane) * BN + vg] = o_reg[v][i0 + ii];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int fi = 0; fi < NV; fi++) {
            int f = (int)sg + fi * NS;
            for (int ii = 0; ii < EP; ii++) {
                float r = simd_sum(outputs[(ii * BD + f) * BN + lane]);
                out_r[fi][i0 + ii] = (sum_exp_score == 0.0f) ? r : (r / sum_exp_score);
            }
        }
        if (pass + 1 < OPASS) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const device T* gp = qkv + GATE_OFF + (size_t)b * GATE_STRIDE + (size_t)h * D;
        device T* op = out + (size_t)bh_l * D;
        for (int fi = 0; fi < NV; fi++) {
            int base = ((int)sg + fi * NS) * 4;
            bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gp + base))[0]);
            bfloat4 ob = bfloat4(
                static_cast<T>(glue_gate(float(static_cast<T>(out_r[fi][0])), float(gb.x))),
                static_cast<T>(glue_gate(float(static_cast<T>(out_r[fi][1])), float(gb.y))),
                static_cast<T>(glue_gate(float(static_cast<T>(out_r[fi][2])), float(gb.z))),
                static_cast<T>(glue_gate(float(static_cast<T>(out_r[fi][3])), float(gb.w))));
            ((device uint2*)(op + base))[0] = as_type<uint2>(ob);
        }
    }
"""

_CAUSAL_2P2_SRC = r"""
    constexpr int D = 128;
    threadgroup float outputs[4 * 1024];
    int bh_l = (int)threadgroup_position_in_grid.x;
    int b_idx = bh_l / HQ;
    int h = bh_l % HQ;
    int b = BOFF + b_idx;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    sdpa2_reduce<T, D, true>(
        partials + (size_t)bh_l * BLOCKS * D, sums + (size_t)bh_l * BLOCKS,
        maxs + (size_t)bh_l * BLOCKS, qkv + GATE_OFF + (size_t)b * GATE_STRIDE + (size_t)h * D, out + (size_t)bh_l * D,
        BLOCKS, outputs, simd_gid, simd_lid);
"""

_CAUSAL_SHORT_ST_SRC = r"""
    constexpr int D = 128;
    threadgroup float outputs[2 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    int bh_l = (int)threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b_idx = bh_l / HQ;
    int h = bh_l % HQ;
    int b = BOFF + b_idx;
    int bh = b * HQ + h;
    int kvh = h / GQA;
    const int S0 = meta[0], end0 = meta[1], slot0 = meta[2], cap = meta[3], max_size = meta[4];
    const int lidx = (KV_LIDX >= 0) ? KV_LIDX : meta[5];
    const int slot_main = meta[6], st_len = meta[7];
    constexpr int TMW = (WW > 8) ? 17 : 9;
    const int db = tree_meta[b * TMW];
    const int S_b = ROT ? min(end0 + db + 1, max_size) : (S0 + db + 1);
    const int end_b = end0 + db + 1;
    const int idx_b = ROT ? (((slot0 + db) % max_size) + 1) : 0;
    const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * cap * D;
    const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * cap * D;
    const device T* Kst = kv_st + ((size_t)lidx * 2 * HKV + kvh) * SCAP * D;
    const device T* Vst = kv_st + ((size_t)lidx * 2 * HKV + HKV + kvh) * SCAP * D;
    const device T* Knew = kv_new + (size_t)kvh * WW * D;
    const device T* Vnew = kv_new + (size_t)(HKV + kvh) * WW * D;
    RowMask<T, PBT> msk;
    msk.S = S_b; msk.end = end_b; msk.idx = idx_b; msk.rot = ROT;
    msk.R = R; msk.window = WIN; msk.pb = pb + (size_t)bh * R;
    float sink = float(static_cast<T>(qkv[SINK_OFF + (size_t)b * SINK_STRIDE + h]));
    sdpa1_head_v32_causal<T, PBT, D, NS, true, true>(
        q + (size_t)bh * D, K, V, Knew, Vnew, qkv + GATE_OFF + (size_t)b * GATE_STRIDE + (size_t)h * D, out + (size_t)bh_l * D,
        S_b, scale[0], sink, msk, db, tree_meta + b * TMW + 1, S0, slot0, max_size, ROT,
        Kst, Vst, slot_main, st_len,
        outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""

_CAUSAL_2P1_ST_SRC = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    constexpr int NTG2 = HKV * TGPK;
    int t_all = (int)threadgroup_position_in_grid.x;
    int b_idx = t_all / NTG2;
    int tl = t_all % NTG2;
    int b = BOFF + b_idx;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const int S0 = meta[0], end0 = meta[1], slot0 = meta[2], cap = meta[3], max_size = meta[4];
    const int lidx = (KV_LIDX >= 0) ? KV_LIDX : meta[5];
    const int slot_main = meta[6], st_len = meta[7];
    constexpr int TMW = (WW > 8) ? 17 : 9;
    const int db = tree_meta[b * TMW];
    const int S_b = ROT ? min(end0 + db + 1, max_size) : (S0 + db + 1);
    const int end_b = end0 + db + 1;
    const int idx_b = ROT ? (((slot0 + db) % max_size) + 1) : 0;
    int kvh = tl / TGPK;
    int f = (tl % TGPK) * NS + (int)simd_gid;
    if (f < BLOCKS * GQA) {
        int block = f / GQA;
        int h = kvh * GQA + f % GQA;
        int bh = b * HQ + h;
        int bh_l = b_idx * HQ + h;
        const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * cap * D;
        const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * cap * D;
        const device T* Kst = kv_st + ((size_t)lidx * 2 * HKV + kvh) * SCAP * D;
        const device T* Vst = kv_st + ((size_t)lidx * 2 * HKV + HKV + kvh) * SCAP * D;
        const device T* Knew = kv_new + (size_t)kvh * WW * D;
        const device T* Vnew = kv_new + (size_t)(HKV + kvh) * WW * D;
        RowMask<T, PBT> msk;
        msk.S = S_b; msk.end = end_b; msk.idx = idx_b; msk.rot = ROT;
        msk.R = R; msk.window = WIN; msk.pb = pb + (size_t)bh * R;
        sdpa2_block_causal<T, PBT, D, BLOCKS, true>(
            q + (size_t)bh * D, K, V, Knew, Vnew,
            partials + ((size_t)bh_l * BLOCKS + block) * D, sums + (size_t)bh_l * BLOCKS + block,
            maxs + (size_t)bh_l * BLOCKS + block, S_b, scale[0],
            float(static_cast<T>(qkv[SINK_OFF + (size_t)b * SINK_STRIDE + h])), msk,
            db, tree_meta + b * TMW + 1, S0, slot0, max_size, ROT,
            Kst, Vst, slot_main, st_len, block, simd_lid);
    }
"""

_causal_w_kernels = None
_causal_w_dyn_kernels = None
_causal_w_stage_kernels = None
_CW_META_CACHE: dict = {}
_CHAIN_TREE_META: dict = {}

_LONG_SDPA = os.environ.get("LOOPLING_LONG_SDPA", "1") not in ("0", "", "false", "False")
_LONG_SDPA_MIN_S = int(os.environ.get("LOOPLING_LONG_SDPA_MIN_S", "32768"))
_LONG_SPLIT_K = 256
_long_full_kernels = None
_DUMMY_LONG_BF = mx.zeros((512,), mx.bfloat16)
_DUMMY_LONG_I32 = mx.zeros((256,), mx.int32)

_LONG_FULL_P1_SRC = r"""
    constexpr int D = 128;
    constexpr int TK = 4;
    constexpr int TMW = (WW > 8) ? 17 : 9;
    int tg_id = (int)threadgroup_position_in_grid.x;
    int sp = tg_id % N_SPLITS;
    int tmp = tg_id / N_SPLITS;
    int kvh = tmp % HKV;
    int rtile = tmp / HKV;
    int r0 = rtile * BW;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int b_loc = (int)sg / GQA;
    int g = (int)sg % GQA;
    int b = r0 + b_loc;
    if (b >= WW) return;
    if constexpr (HAS_WA) {
        if (b >= w_act[0]) return;
    }
    int h = kvh * GQA + g;
    const int S0 = meta[0];
    const int cap = HAS_NEW ? meta[3] : meta[4];
    const int lidx = (KV_LIDX >= 0) ? KV_LIDX : meta[5];
    const int slot_main = HAS_STAGE ? meta[6] : 0;
    const int st_len = HAS_STAGE ? meta[7] : 0;
    const int sp_start = sp * SPLIT_K;

    int db = 0;
    int anc[16];
    if constexpr (HAS_NEW && WW > 1) {
        db = tree_meta[b * TMW];
        const device int* ap = tree_meta + b * TMW + 1;
        for (int d = 0; d <= db; d++) anc[d] = ap[d];
    }
    const int S_b = S0 + (HAS_NEW ? (db + 1) : 0);
    const int sp_end_b = min(sp_start + SPLIT_K, S_b);
    if (sp_start >= sp_end_b) return;

    const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * cap * D;
    const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * cap * D;
    const device T* Kst = HAS_STAGE ? (kv_st + ((size_t)lidx * 2 * HKV + kvh) * SCAP * D) : K;
    const device T* Vst = HAS_STAGE ? (kv_st + ((size_t)lidx * 2 * HKV + HKV + kvh) * SCAP * D) : V;
    const device T* Knew = HAS_NEW ? (kv_new + (size_t)kvh * WW * D) : K;
    const device T* Vnew = HAS_NEW ? (kv_new + (size_t)(HKV + kvh) * WW * D) : V;

    int bh = b * HQ + h;
    const device PBT* pb_bh = pb + (size_t)bh * R;
    float sc = scale[0];
    bfloat4 qb = as_type<bfloat4>(((const device uint2*)(q + (size_t)bh * D))[lane]);
    float q0 = sc * float(qb.x), q1 = sc * float(qb.y), q2 = sc * float(qb.z), q3 = sc * float(qb.w);
    float max_score = Limits<float>::finite_min;
    float sum_exp_score = 0.0f;
    if (sp == 0) {
        max_score = float(static_cast<T>(qkv[SINK_OFF + (size_t)b * SINK_STRIDE + h]));
        sum_exp_score = 1.0f;
    }
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    int t0 = sp_start;
    const int fast_end = min(sp_end_b, min(S0, S_b - R));
    for (; t0 + TK <= fast_end; t0 += TK) {
        bfloat4 k_reg[TK];
        bfloat4 v_reg[TK];
        for (int t = 0; t < TK; t++) {
            int ii = t0 + t;
            const device T* kp;
            if constexpr (HAS_STAGE) {
                int ds = ii - slot_main;
                kp = (ds >= 0 && ds < st_len) ? (Kst + (size_t)ds * D) : (K + (size_t)ii * D);
            } else {
                kp = K + (size_t)ii * D;
            }
            k_reg[t] = as_type<bfloat4>(((const device uint2*)kp)[lane]);
        }
        for (int t = 0; t < TK; t++) {
            int ii = t0 + t;
            const device T* vp;
            if constexpr (HAS_STAGE) {
                int ds = ii - slot_main;
                vp = (ds >= 0 && ds < st_len) ? (Vst + (size_t)ds * D) : (V + (size_t)ii * D);
            } else {
                vp = V + (size_t)ii * D;
            }
            v_reg[t] = as_type<bfloat4>(((const device uint2*)vp)[lane]);
        }
        for (int t = 0; t < TK; t++) {
            float score = q0 * float(k_reg[t].x) + q1 * float(k_reg[t].y) + q2 * float(k_reg[t].z) + q3 * float(k_reg[t].w);
            score = simd_sum(score);
            float new_max = max(max_score, score);
            float factor = fast::exp(max_score - new_max);
            float exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            o0 = o0 * factor + exp_score * float(v_reg[t].x);
            o1 = o1 * factor + exp_score * float(v_reg[t].y);
            o2 = o2 * factor + exp_score * float(v_reg[t].z);
            o3 = o3 * factor + exp_score * float(v_reg[t].w);
        }
    }
    for (int ii = t0; ii < sp_end_b; ii++) {
        int dist = (S_b - 1) - ii;
        T mv = (dist < R) ? static_cast<T>(pb_bh[dist]) : static_cast<T>(0.0f);
        if (mv >= Limits<T>::finite_min) {
            const device T* kp;
            const device T* vp;
            if (ii < S0) {
                if constexpr (HAS_STAGE) {
                    int ds = ii - slot_main;
                    if (ds >= 0 && ds < st_len) {
                        kp = Kst + (size_t)ds * D; vp = Vst + (size_t)ds * D;
                    } else {
                        kp = K + (size_t)ii * D; vp = V + (size_t)ii * D;
                    }
                } else {
                    kp = K + (size_t)ii * D; vp = V + (size_t)ii * D;
                }
            } else {
                int nd = HAS_NEW ? anc[ii - S0] : 0;
                kp = Knew + (size_t)nd * D; vp = Vnew + (size_t)nd * D;
            }
            bfloat4 kb = as_type<bfloat4>(((const device uint2*)kp)[lane]);
            float score = q0 * float(kb.x) + q1 * float(kb.y) + q2 * float(kb.z) + q3 * float(kb.w);
            score = simd_sum(score) + float(mv);
            float new_max = max(max_score, score);
            float factor = fast::exp(max_score - new_max);
            float exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            bfloat4 vb = as_type<bfloat4>(((const device uint2*)vp)[lane]);
            o0 = o0 * factor + exp_score * float(vb.x);
            o1 = o1 * factor + exp_score * float(vb.y);
            o2 = o2 * factor + exp_score * float(vb.z);
            o3 = o3 * factor + exp_score * float(vb.w);
        }
    }
    size_t bh_sp = ((size_t)b * HQ + h) * N_SPLITS + sp;
    if (lane == 0) {
        sums[bh_sp] = sum_exp_score;
        maxs[bh_sp] = max_score;
    }
    bfloat4 ob = bfloat4(static_cast<T>(o0), static_cast<T>(o1), static_cast<T>(o2), static_cast<T>(o3));
    ((device uint2*)(partials + bh_sp * D))[lane] = as_type<uint2>(ob);
"""

_LONG_FULL_P2_SRC = r"""
    constexpr int D = 128;
    constexpr int TMW = (WW > 8) ? 17 : 9;
    threadgroup float outputs[4 * 1024];
    int bh = (int)threadgroup_position_in_grid.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int b = bh / HQ;
    int h = bh % HQ;
    if constexpr (HAS_WA) {
        if (b >= w_act[0]) return;
    }
    const int S0 = meta[0];
    const int db = (HAS_NEW && WW > 1) ? tree_meta[b * TMW] : 0;
    const int S_b = S0 + (HAS_NEW ? (db + 1) : 0);
    const int n_act = min(N_SPLITS, (S_b + SPLIT_K - 1) / SPLIT_K);
    size_t base = (size_t)bh * N_SPLITS;

    float gmax = Limits<float>::finite_min;
    for (int s0 = 0; s0 < n_act; s0 += 32) {
        int sp = s0 + (int)lane;
        float m = (sp < n_act) ? maxs[base + sp] : Limits<float>::finite_min;
        gmax = max(gmax, simd_max(m));
    }

    float sum_exp = 0.0f;
    for (int s0 = 0; s0 < n_act; s0 += 32) {
        int sp = s0 + (int)lane;
        float term = (sp < n_act) ? (fast::exp(maxs[base + sp] - gmax) * sums[base + sp]) : 0.0f;
        sum_exp += simd_sum(term);
    }

    float4 o = float4(0.0f);
    for (int s0 = 0; s0 < n_act; s0 += 32) {
        int sp = s0 + (int)sg;
        if (sp < n_act) {
            float f = fast::exp(maxs[base + sp] - gmax);
            bfloat4 pbv = as_type<bfloat4>(((const device uint2*)(partials + (base + sp) * D))[lane]);
            o += f * float4(float(pbv.x), float(pbv.y), float(pbv.z), float(pbv.w));
        }
    }
    outputs[(0 * 32 + (int)lane) * 32 + (int)sg] = o.x;
    outputs[(1 * 32 + (int)lane) * 32 + (int)sg] = o.y;
    outputs[(2 * 32 + (int)lane) * 32 + (int)sg] = o.z;
    outputs[(3 * 32 + (int)lane) * 32 + (int)sg] = o.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 red;
    red.x = simd_sum(outputs[(0 * 32 + (int)sg) * 32 + (int)lane]);
    red.y = simd_sum(outputs[(1 * 32 + (int)sg) * 32 + (int)lane]);
    red.z = simd_sum(outputs[(2 * 32 + (int)sg) * 32 + (int)lane]);
    red.w = simd_sum(outputs[(3 * 32 + (int)sg) * 32 + (int)lane]);
    if (lane == 0) {
        if (sum_exp != 0.0f) red /= sum_exp;
        int base_d = (int)sg * 4;
        bfloat4 gb = as_type<bfloat4>(((const device uint2*)(qkv + GATE_OFF + (size_t)b * GATE_STRIDE + (size_t)h * D + base_d))[0]);
        bfloat4 ob = bfloat4(
            static_cast<T>(glue_gate(float(static_cast<T>(red.x)), float(gb.x))),
            static_cast<T>(glue_gate(float(static_cast<T>(red.y)), float(gb.y))),
            static_cast<T>(glue_gate(float(static_cast<T>(red.z)), float(gb.z))),
            static_cast<T>(glue_gate(float(static_cast<T>(red.w)), float(gb.w))));
        ((device uint2*)(out + (size_t)bh * D + base_d))[0] = as_type<uint2>(ob);
    }
"""


def _get_long_full_kernels():
    global _long_full_kernels
    if _long_full_kernels is None:
        p1 = mx.fast.metal_kernel(
            name="sdpa_long_full_p1",
            input_names=["q", "kv", "kv_st", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta", "w_act"],
            output_names=["partials", "sums", "maxs"],
            source=_LONG_FULL_P1_SRC,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        p2 = mx.fast.metal_kernel(
            name="sdpa_long_full_p2",
            input_names=["partials", "sums", "maxs", "qkv", "meta", "tree_meta", "w_act"],
            output_names=["out"],
            source=_LONG_FULL_P2_SRC,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        _long_full_kernels = (p1, p2)
    return _long_full_kernels


def sdpa_long_full_packed(
    q, kv_buf, kv_new, pb, qkv_flat, gate_off, sink_off, S_prefix, S_alloc, cap, W, *,
    scale, Hq=24, Hkv=8, kv_layer_idx=0, tree_meta=None, meta_arr=None, kv_stage=None,
    w_act=None, gate_stride=None, sink_stride=None,
):
    """Split-K full-attention (window==0, rot==0) sharing K/V loads across W rows and GQA heads."""
    gqa = Hq // Hkv
    R = int(pb.shape[-1])
    g_str = int(gate_stride or (Hq * 128))
    s_str = int(sink_stride or Hq)
    sck = float(scale)
    sc = _SCALE_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_CACHE[sck] = mx.array([sck], mx.float32)

    has_new = kv_new is not None
    has_stage = kv_stage is not None
    has_wa = w_act is not None
    if meta_arr is not None:
        mta = meta_arr
        kv_lidx_t = int(kv_layer_idx)
    else:
        if has_new:
            mkey = (int(S_prefix), int(S_prefix), 0, int(cap), 0, int(kv_layer_idx))
            mta = _CW_META_CACHE.get(mkey)
            if mta is None:
                if len(_CW_META_CACHE) >= 256:
                    _CW_META_CACHE.clear()
                mta = _CW_META_CACHE[mkey] = mx.array(mkey, mx.int32)
        else:
            mkey = (int(S_prefix), int(S_prefix), 0, 0, int(cap), int(kv_layer_idx))
            mta = _B1_META_CACHE.get(mkey)
            if mta is None:
                if len(_B1_META_CACHE) >= 256:
                    _B1_META_CACHE.clear()
                mta = _B1_META_CACHE[mkey] = mx.array(mkey, mx.int32)
        kv_lidx_t = -1

    if tree_meta is None:
        _, tm_arg = _default_tree_meta(W)
    else:
        tm_arg = tree_meta

    kv_st_arg = kv_stage if has_stage else _DUMMY_LONG_BF
    scap = int(kv_stage.shape[3]) if has_stage else 0
    kv_new_arg = kv_new if has_new else _DUMMY_LONG_BF
    wa_arg = w_act.reshape(-1) if has_wa else _DUMMY_LONG_I32

    n_splits = max(1, (int(S_alloc) + _LONG_SPLIT_K - 1) // _LONG_SPLIT_K)
    bw = min(W, 8)
    ns = gqa * bw
    n_rtiles = (W + bw - 1) // bw
    ntg = n_splits * Hkv * n_rtiles

    p1, p2 = _get_long_full_kernels()
    partials, sums, maxs = p1(
        inputs=[q, kv_buf, kv_st_arg, kv_new_arg, pb, qkv_flat, mta, sc, tm_arg, wa_arg],
        template=[
            ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
            ("R", R), ("WW", int(W)), ("BW", int(bw)), ("NS", int(ns)),
            ("SPLIT_K", _LONG_SPLIT_K), ("N_SPLITS", int(n_splits)),
            ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", s_str),
            ("KV_LIDX", kv_lidx_t), ("SCAP", scap),
            ("HAS_NEW", has_new), ("HAS_STAGE", has_stage), ("HAS_WA", has_wa),
        ],
        grid=(ntg * 32 * ns, 1, 1),
        threadgroup=(32 * ns, 1, 1),
        output_shapes=[(W * Hq * n_splits * 128,), (W * Hq * n_splits,), (W * Hq * n_splits,)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
    )
    (out,) = p2(
        inputs=[partials, sums, maxs, qkv_flat, mta, tm_arg, wa_arg],
        template=[
            ("T", mx.bfloat16), ("HQ", Hq), ("WW", int(W)),
            ("SPLIT_K", _LONG_SPLIT_K), ("N_SPLITS", int(n_splits)),
            ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", g_str),
            ("HAS_NEW", has_new), ("HAS_WA", has_wa),
        ],
        grid=(W * Hq * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(W, Hq * 128)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def _default_tree_meta(W):
    v = _CHAIN_TREE_META.get(W)
    if v is None:
        tm_w = 9 if W <= 8 else 17
        rows = [[b] + list(range(b + 1)) + [0] * ((tm_w - 1) - (b + 1)) for b in range(W)]
        v = _CHAIN_TREE_META[W] = (tuple(range(W)), mx.array(rows, mx.int32))
    return v


def _get_causal_w_kernels():
    global _causal_w_kernels
    if _causal_w_kernels is None:
        s1_short = mx.fast.metal_kernel(
            name="causal_w_sdpa_short_t",
            input_names=["q", "kv", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta"],
            output_names=["out"],
            source=_CAUSAL_SHORT_SRC,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        s1_2p = mx.fast.metal_kernel(
            name="causal_w_sdpa_2p1_t",
            input_names=["q", "kv", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta"],
            output_names=["partials", "sums", "maxs"],
            source=_CAUSAL_2P1_SRC,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        s2_2p = mx.fast.metal_kernel(
            name="causal_w_sdpa_2p2",
            input_names=["partials", "sums", "maxs", "qkv"],
            output_names=["out"],
            source=_CAUSAL_2P2_SRC,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        _causal_w_kernels = (s1_short, s1_2p, s2_2p)
    return _causal_w_kernels


def _get_causal_w_dyn_kernels():
    global _causal_w_dyn_kernels
    if _causal_w_dyn_kernels is None:
        s_short = _CAUSAL_SHORT_SRC.replace(
            "    int bh = b * HQ + h;",
            "    if (b >= w_act[0]) return;\n    int bh = b * HQ + h;",
        )
        s_2p1 = _CAUSAL_2P1_SRC.replace(
            "    int b = BOFF + b_idx;",
            "    int b = BOFF + b_idx;\n    if (b >= w_act[0]) return;",
        )
        s_2p2 = _CAUSAL_2P2_SRC.replace(
            "    int b = BOFF + b_idx;",
            "    int b = BOFF + b_idx;\n    if (b >= w_act[0]) return;",
        )
        assert s_short != _CAUSAL_SHORT_SRC and s_2p1 != _CAUSAL_2P1_SRC and s_2p2 != _CAUSAL_2P2_SRC
        s1_short = mx.fast.metal_kernel(
            name="causal_w_sdpa_short_dyn",
            input_names=["q", "kv", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta", "w_act"],
            output_names=["out"],
            source=s_short,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        s1_2p = mx.fast.metal_kernel(
            name="causal_w_sdpa_2p1_dyn",
            input_names=["q", "kv", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta", "w_act"],
            output_names=["partials", "sums", "maxs"],
            source=s_2p1,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        s2_2p = mx.fast.metal_kernel(
            name="causal_w_sdpa_2p2_dyn",
            input_names=["partials", "sums", "maxs", "qkv", "w_act"],
            output_names=["out"],
            source=s_2p2,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        _causal_w_dyn_kernels = (s1_short, s1_2p, s2_2p)
    return _causal_w_dyn_kernels


_causal_w_2p_gqa3_kernels = None


def _get_causal_w_2p_gqa3_kernels():
    global _causal_w_2p_gqa3_kernels
    if _causal_w_2p_gqa3_kernels is None:
        s_dyn = _CAUSAL_2P1_GQA3_SRC.replace(
            "    int b = BOFF + b_idx;",
            "    int b = BOFF + b_idx;\n    if (b >= w_act[0]) return;",
        )
        assert s_dyn != _CAUSAL_2P1_GQA3_SRC
        k_std = mx.fast.metal_kernel(
            name="causal_w_sdpa_2p1_gqa3",
            input_names=["q", "kv", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta"],
            output_names=["partials", "sums", "maxs"],
            source=_CAUSAL_2P1_GQA3_SRC,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        k_dyn = mx.fast.metal_kernel(
            name="causal_w_sdpa_2p1_gqa3_dyn",
            input_names=["q", "kv", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta", "w_act"],
            output_names=["partials", "sums", "maxs"],
            source=s_dyn,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        _causal_w_2p_gqa3_kernels = (k_std, k_dyn)
    return _causal_w_2p_gqa3_kernels


_causal_w_2pfused_kernels = None


def _get_causal_w_2pfused_kernels():
    global _causal_w_2pfused_kernels
    if _causal_w_2pfused_kernels is None:
        s_dyn = _CAUSAL_2PFUSED_SRC.replace(
            "    int b = BOFF + b_idx;",
            "    int b = BOFF + b_idx;\n    if (b >= w_act[0]) return;",
        )
        assert s_dyn != _CAUSAL_2PFUSED_SRC
        k_std = mx.fast.metal_kernel(
            name="causal_w_sdpa_2pfused",
            input_names=["q", "kv", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta"],
            output_names=["out"],
            source=_CAUSAL_2PFUSED_SRC,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        k_dyn = mx.fast.metal_kernel(
            name="causal_w_sdpa_2pfused_dyn",
            input_names=["q", "kv", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta", "w_act"],
            output_names=["out"],
            source=s_dyn,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        _causal_w_2pfused_kernels = (k_std, k_dyn)
    return _causal_w_2pfused_kernels


def _get_causal_w_stage_kernels():
    global _causal_w_stage_kernels
    if _causal_w_stage_kernels is None:
        s1_short = mx.fast.metal_kernel(
            name="causal_w_sdpa_short_st",
            input_names=["q", "kv", "kv_st", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta"],
            output_names=["out"],
            source=_CAUSAL_SHORT_ST_SRC,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        s1_2p = mx.fast.metal_kernel(
            name="causal_w_sdpa_2p1_st",
            input_names=["q", "kv", "kv_st", "kv_new", "pb", "qkv", "meta", "scale", "tree_meta"],
            output_names=["partials", "sums", "maxs"],
            source=_CAUSAL_2P1_ST_SRC,
            header=_CAUSAL_HEADER,
            ensure_row_contiguous=True,
        )
        _, _, s2_2p = _get_causal_w_kernels()
        _causal_w_stage_kernels = (s1_short, s1_2p, s2_2p)
    return _causal_w_stage_kernels


def sdpa_causal_w_packed(
    q, kv_buf, kv_new, pb, qkv_flat, gate_off, sink_off, S0, end0, slot0, rot, cap, max_size, W, *,
    scale, window, Hq=24, Hkv=8, kv_layer_idx=0, tree_meta=None, depths=None, meta_arr=None, s_bounds=None,
    kv_stage=None, w_act=None, gate_stride=None, sink_stride=None,
):
    """Causal W-token/tree decode attention over pre-step packed KV + kv_new."""
    assert q.dtype == mx.bfloat16 and kv_buf.dtype == mx.bfloat16 and kv_new.dtype == mx.bfloat16 and qkv_flat.dtype == mx.bfloat16
    gqa = Hq // Hkv
    g_str = int(gate_stride or Hq * 128)
    s_str = int(sink_stride or Hq)
    assert Hq == gqa * Hkv and kv_new.shape == (2, Hkv, W, 128) and (
        (kv_buf.ndim == 4 and kv_buf.shape == (2, Hkv, cap, 128) and kv_layer_idx == 0)
        or (kv_buf.ndim == 5 and kv_buf.shape[1:] == (2, Hkv, cap, 128) and 0 <= kv_layer_idx < kv_buf.shape[0])
    )
    if tree_meta is None:
        depths_t, tree_meta = _default_tree_meta(W)
        if depths is not None:
            depths_t = tuple(int(d) for d in depths)
    else:
        tm_w = 9 if W <= 8 else 17
        assert tree_meta.shape == (W, tm_w) and tree_meta.dtype == mx.int32
        depths_t = tuple(int(d) for d in depths) if depths is not None else None
    R = pb.shape[-1]
    sck = float(scale)
    sc = _SCALE_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_CACHE[sck] = mx.array([sck], mx.float32)
    if meta_arr is not None:
        mta = meta_arr
        kv_lidx_t = int(kv_layer_idx)
    else:
        mkey = (int(S0), int(end0), int(slot0), int(cap), int(max_size), int(kv_layer_idx))
        mta = _CW_META_CACHE.get(mkey)
        if mta is None:
            if len(_CW_META_CACHE) >= 256:
                _CW_META_CACHE.clear()
            mta = _CW_META_CACHE[mkey] = mx.array(mkey, mx.int32)
        kv_lidx_t = -1
    has_wa = (w_act is not None)
    if s_bounds is not None:
        s_min_b, s_max_b = int(s_bounds[0]), int(s_bounds[1])
        if _LONG_SDPA and int(window) == 0 and int(rot) == 0 and s_min_b >= _LONG_SDPA_MIN_S:
            return sdpa_long_full_packed(
                q, kv_buf, kv_new, pb, qkv_flat, gate_off, sink_off, S0, int(cap) + W, cap, W,
                scale=scale, Hq=Hq, Hkv=Hkv, kv_layer_idx=kv_layer_idx, tree_meta=tree_meta,
                meta_arr=meta_arr, kv_stage=kv_stage, w_act=w_act,
                gate_stride=gate_stride, sink_stride=sink_stride,
            )
    else:
        s_vals = [min(end0 + d + 1, max_size) if rot else (S0 + d + 1) for d in depths_t]
        if _LONG_SDPA and int(window) == 0 and int(rot) == 0 and s_vals[0] >= _LONG_SDPA_MIN_S:
            return sdpa_long_full_packed(
                q, kv_buf, kv_new, pb, qkv_flat, gate_off, sink_off, S0, s_vals[-1], cap, W,
                scale=scale, Hq=Hq, Hkv=Hkv, kv_layer_idx=kv_layer_idx, tree_meta=tree_meta,
                meta_arr=meta_arr, kv_stage=kv_stage, w_act=w_act,
                gate_stride=gate_stride, sink_stride=sink_stride,
            )

    wa_arg = [w_act.reshape(-1)] if has_wa else []
    if kv_stage is not None:
        assert not has_wa
        assert meta_arr is not None and int(meta_arr.shape[0]) == 8
        assert kv_buf.ndim == 5 and kv_stage.shape[:3] == kv_buf.shape[:3] and kv_stage.shape[4] == 128 and kv_stage.dtype == mx.bfloat16
        scap = int(kv_stage.shape[3])
        s1_short, s1_2p, s2_2p = _get_causal_w_stage_kernels()
    elif has_wa:
        scap = 0
        s1_short, s1_2p, s2_2p = _get_causal_w_dyn_kernels()
    else:
        scap = 0
        s1_short, s1_2p, s2_2p = _get_causal_w_kernels()

    if s_bounds is not None:
        split = W if s_max_b < _TWO_PASS_N else 0
        s_vals = None
    else:
        split = 0
        while split < W and s_vals[split] < _TWO_PASS_N:
            split += 1

    def _run_short(b_off, b_cnt):
        ns_env = os.environ.get("LOOPLING_SDPA1_NS", "")
        ns = int(ns_env) if ns_env and ns_env != "0" else (16 if b_cnt >= 4 else 32)
        opass_env = os.environ.get("LOOPLING_CW_OPASS", "")
        opass = int(opass_env) if opass_env and opass_env != "0" else 1
        kvh_maj = os.environ.get("LOOPLING_CW_KVH_MAJ", "1") not in ("0", "")
        if kv_stage is not None:
            (o,) = s1_short(
                inputs=[q, kv_buf, kv_stage, kv_new, pb, qkv_flat, mta, sc, tree_meta],
                template=[
                    ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                    ("R", R), ("WIN", int(window)), ("NS", ns), ("ROT", int(rot)), ("WW", int(W)),
                    ("BOFF", int(b_off)), ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", g_str),
                    ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", s_str),
                    ("KV_LIDX", kv_lidx_t), ("SCAP", scap),
                ],
                grid=(b_cnt * Hq * 32 * ns, 1, 1),
                threadgroup=(32 * ns, 1, 1),
                output_shapes=[(b_cnt, Hq * 128)],
                output_dtypes=[mx.bfloat16],
            )
            return o
        (o,) = s1_short(
            inputs=[q, kv_buf, kv_new, pb, qkv_flat, mta, sc, tree_meta] + wa_arg,
            template=[
                ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                ("R", R), ("WIN", int(window)), ("NS", ns), ("ROT", int(rot)), ("WW", int(W)),
                ("BOFF", int(b_off)), ("B_CNT", int(b_cnt)), ("OPASS", opass), ("KVH_MAJ", kvh_maj),
                ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", g_str),
                ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", s_str),
                ("KV_LIDX", kv_lidx_t),
            ],
            grid=(b_cnt * Hq * 32 * ns, 1, 1),
            threadgroup=(32 * ns, 1, 1),
            output_shapes=[(b_cnt, Hq * 128)],
            output_dtypes=[mx.bfloat16],
        )
        return o

    def _run_2p(b_off, b_cnt):
        s_hi = s_max_b if s_bounds is not None else s_vals[-1]
        blocks = _blocks(s_hi, gqa)
        if (
            kv_stage is None
            and blocks == 128
            and s_hi <= 1536
            and b_cnt >= 6
            and os.environ.get("LOOPLING_CW_2PF", "1") not in ("0", "")
        ):
            k_std, k_dyn = _get_causal_w_2pfused_kernels()
            k_2pf = k_dyn if has_wa else k_std
            ns_f = 8
            (o,) = k_2pf(
                inputs=[q, kv_buf, kv_new, pb, qkv_flat, mta, sc, tree_meta] + wa_arg,
                template=[
                    ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                    ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns_f), ("OPASS", 4),
                    ("ROT", int(rot)), ("WW", int(W)), ("BOFF", int(b_off)), ("B_CNT", int(b_cnt)),
                    ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", g_str),
                    ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", s_str),
                    ("KV_LIDX", kv_lidx_t),
                ],
                grid=(b_cnt * Hq * 32 * ns_f, 1, 1),
                threadgroup=(32 * ns_f, 1, 1),
                output_shapes=[(b_cnt, Hq * 128)],
                output_dtypes=[mx.bfloat16],
            )
            return o
        ns = gqa
        tgpk = (blocks * gqa + ns - 1) // ns
        ntg = b_cnt * Hkv * tgpk
        if kv_stage is not None:
            partials, sums, maxs = s1_2p(
                inputs=[q, kv_buf, kv_stage, kv_new, pb, qkv_flat, mta, sc, tree_meta],
                template=[
                    ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                    ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns), ("ROT", int(rot)),
                    ("WW", int(W)), ("BOFF", int(b_off)), ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", s_str),
                    ("KV_LIDX", kv_lidx_t), ("SCAP", scap),
                ],
                grid=(ntg * 32 * ns, 1, 1),
                threadgroup=(32 * ns, 1, 1),
                output_shapes=[(b_cnt * Hq * blocks * 128,), (b_cnt * Hq * blocks,), (b_cnt * Hq * blocks,)],
                output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
            )
        elif gqa == 3 and 2 <= b_cnt <= 16 and os.environ.get("LOOPLING_CW_2P_GQA3", "1") not in ("0", ""):
            k_std, k_dyn = _get_causal_w_2p_gqa3_kernels()
            k_2p = k_dyn if has_wa else k_std
            ntg_g3 = Hkv * blocks
            partials, sums, maxs = k_2p(
                inputs=[q, kv_buf, kv_new, pb, qkv_flat, mta, sc, tree_meta] + wa_arg,
                template=[
                    ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv),
                    ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("ROT", int(rot)),
                    ("WW", int(W)), ("BOFF", int(b_off)), ("B_CNT", int(b_cnt)),
                    ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", s_str),
                    ("KV_LIDX", kv_lidx_t),
                ],
                grid=(ntg_g3 * 32 * b_cnt, 1, 1),
                threadgroup=(32 * b_cnt, 1, 1),
                output_shapes=[(b_cnt * Hq * blocks * 128,), (b_cnt * Hq * blocks,), (b_cnt * Hq * blocks,)],
                output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
            )
        else:
            partials, sums, maxs = s1_2p(
                inputs=[q, kv_buf, kv_new, pb, qkv_flat, mta, sc, tree_meta] + wa_arg,
                template=[
                    ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                    ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns), ("ROT", int(rot)),
                    ("WW", int(W)), ("BOFF", int(b_off)), ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", s_str),
                    ("KV_LIDX", kv_lidx_t),
                ],
                grid=(ntg * 32 * ns, 1, 1),
                threadgroup=(32 * ns, 1, 1),
                output_shapes=[(b_cnt * Hq * blocks * 128,), (b_cnt * Hq * blocks,), (b_cnt * Hq * blocks,)],
                output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
            )
        (o,) = s2_2p(
            inputs=[partials, sums, maxs, qkv_flat] + wa_arg,
            template=[("T", mx.bfloat16), ("HQ", Hq), ("BLOCKS", blocks), ("BOFF", int(b_off)), ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", g_str)],
            grid=(b_cnt * Hq * 1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(b_cnt, Hq * 128)],
            output_dtypes=[mx.bfloat16],
        )
        return o

    if split == W:
        return _run_short(0, W)
    if split == 0:
        return _run_2p(0, W)
    return mx.concatenate([_run_short(0, split), _run_2p(split, W - split)], axis=0)


# =============================================================================================== Staged batched SDPA
_SMW = 9  # S_now, end_now, idx_now, rot, kcap, vcap, slot_main, rot_wrap, max_size

_STAGED_HEADER = _HEADER + r"""
template <typename T, typename PBT, int D, int NS, bool HAS_GATE>
METAL_FUNC void sdpa1_head_v32_staged(
    const device T* queries, const device T* K_main, const device T* V_main,
    const device T* Kst, const device T* Vst, const device T* Knew, const device T* Vnew,
    const device T* gate, device T* out, const int N, const float scale, const float sink,
    const RowMask<T, PBT> msk, const int slot_main, const int rot_wrap, const int max_size,
    const int st_len, threadgroup float* outputs, threadgroup float* max_scores,
    threadgroup float* sum_exp_scores, uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int NV = BN / NS;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread];
  const int loff = (int)simd_lid * qk_per_thread;
  queries += loff;
  bfloat4 qb = as_type<bfloat4>(((const device uint2*)queries)[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  for (int v = 0; v < NV; v++) {
    int vg = (int)simd_gid + v * NS;
    for (int i = 0; i < v_per_thread; i++) o[i] = 0;
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    if (vg == 0) {
      max_score = sink;
      sum_exp_score = 1;
    }
    int i = vg;
    if (!rot_wrap) {
      const int n_fast = min(N, slot_main);
      for (; i < n_fast; i += BN) {
        float fm = 0;
        if (msk.at(i, fm)) {
          bfloat4 kb = as_type<bfloat4>(((const device uint2*)(K_main + (size_t)i * 128 + loff))[0]);
          U score = 0;
          score += q[0] * kb.x;
          score += q[1] * kb.y;
          score += q[2] * kb.z;
          score += q[3] * kb.w;
          score = simd_sum(score);
          score += static_cast<U>(fm);
          U new_max = max(max_score, score);
          U factor = fast::exp(max_score - new_max);
          U exp_score = fast::exp(score - new_max);
          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + exp_score;
          bfloat4 vb = as_type<bfloat4>(((const device uint2*)(V_main + (size_t)i * 128 + loff))[0]);
          o[0] = o[0] * factor + exp_score * vb.x;
          o[1] = o[1] * factor + exp_score * vb.y;
          o[2] = o[2] * factor + exp_score * vb.z;
          o[3] = o[3] * factor + exp_score * vb.w;
        }
      }
    }
    for (; i < N; i += BN) {
      float fm = 0;
      if (msk.at(i, fm)) {
        int ds = rot_wrap ? (i - slot_main + (i < slot_main ? max_size : 0)) : (i - slot_main);
        const device T* kp;
        const device T* vp;
        if (ds == st_len) {
          kp = Knew + loff;
          vp = Vnew + loff;
        } else if (ds >= 0 && ds < st_len) {
          kp = Kst + (size_t)ds * 128 + loff;
          vp = Vst + (size_t)ds * 128 + loff;
        } else {
          kp = K_main + (size_t)i * 128 + loff;
          vp = V_main + (size_t)i * 128 + loff;
        }
        bfloat4 kb = as_type<bfloat4>(((const device uint2*)kp)[0]);
        U score = 0;
        score += q[0] * kb.x;
        score += q[1] * kb.y;
        score += q[2] * kb.z;
        score += q[3] * kb.w;
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        bfloat4 vb = as_type<bfloat4>(((const device uint2*)vp)[0]);
        o[0] = o[0] * factor + exp_score * vb.x;
        o[1] = o[1] * factor + exp_score * vb.y;
        o[2] = o[2] * factor + exp_score * vb.z;
        o[3] = o[3] * factor + exp_score * vb.w;
      }
    }
    if (simd_lid == 0) {
      max_scores[vg] = max_score;
      sum_exp_scores[vg] = sum_exp_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
      outputs[(i * BD + simd_lid) * BN + vg] = o[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int f = (int)simd_gid; f < BD; f += NS) {
    for (int i = 0; i < v_per_thread; i++) {
      o[i] = simd_sum(outputs[(i * BD + f) * BN + simd_lid] * factor);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    }
    if (simd_lid == 0) {
      int base = f * v_per_thread;
      if constexpr (HAS_GATE) {
        bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
        bfloat4 ob = bfloat4(
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[0])), static_cast<float>(gb.x))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[1])), static_cast<float>(gb.y))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[2])), static_cast<float>(gb.z))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[3])), static_cast<float>(gb.w))));
        ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
      } else {
        bfloat4 ob = bfloat4(static_cast<T>(o[0]), static_cast<T>(o[1]), static_cast<T>(o[2]), static_cast<T>(o[3]));
        ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
      }
    }
  }
}

template <typename T, typename PBT, int D, int BLOCKS>
METAL_FUNC void sdpa2_block_staged(
    const device T* queries, const device T* K_main, const device T* V_main,
    const device T* Kst, const device T* Vst, const device T* Knew, const device T* Vnew,
    device T* out, device float* sums, device float* maxs, const int N, const float scale,
    const float sink, const RowMask<T, PBT> msk, const int slot_main, const int rot_wrap,
    const int max_size, const int st_len, const int block_idx, uint simd_lid) {
  constexpr int TK = 4;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int blocks = BLOCKS;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};
  const int loff = (int)simd_lid * qk_per_thread;
  queries += loff;
  out += loff;
  bfloat4 qb = as_type<bfloat4>(((const device uint2*)queries)[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (block_idx == 0) {
    max_score = sink;
    sum_exp_score = 1;
  }
  int i = block_idx;
  const int dirty_lo = (rot_wrap && slot_main + st_len >= max_size) ? (slot_main + st_len - max_size) : -1;
  const int dirty_end = rot_wrap ? min(N, slot_main + st_len + 1) : N;
  const int k_seq_stride = D;
  const int stride_k_tok = blocks * k_seq_stride;
  const int stride_v_tok = blocks * k_seq_stride;

  auto run_ds_chunk = [&](int ii_base) __attribute__((always_inline)) {
    bfloat4 k_reg[TK];
    bfloat4 v_reg[TK];
    for (int t = 0; t < TK; t++) {
      int ii = ii_base + t * blocks;
      int ds = rot_wrap ? (ii - slot_main + (ii < slot_main ? max_size : 0)) : (ii - slot_main);
      const device T* kp = (ds == st_len) ? (Knew + loff) : ((ds >= 0 && ds < st_len) ? (Kst + (size_t)ds * D + loff) : (K_main + (size_t)ii * D + loff));
      const device T* vp = (ds == st_len) ? (Vnew + loff) : ((ds >= 0 && ds < st_len) ? (Vst + (size_t)ds * D + loff) : (V_main + (size_t)ii * D + loff));
      k_reg[t] = as_type<bfloat4>(((const device uint2*)kp)[0]);
      v_reg[t] = as_type<bfloat4>(((const device uint2*)vp)[0]);
    }
    for (int t = 0; t < TK; t++) {
      int ii = ii_base + t * blocks;
      float fm = 0;
      if (msk.at(ii, fm)) {
        U score = 0;
        score += q[0] * static_cast<U>(k_reg[t].x);
        score += q[1] * static_cast<U>(k_reg[t].y);
        score += q[2] * static_cast<U>(k_reg[t].z);
        score += q[3] * static_cast<U>(k_reg[t].w);
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        o[0] = o[0] * factor + exp_score * static_cast<U>(v_reg[t].x);
        o[1] = o[1] * factor + exp_score * static_cast<U>(v_reg[t].y);
        o[2] = o[2] * factor + exp_score * static_cast<U>(v_reg[t].z);
        o[3] = o[3] * factor + exp_score * static_cast<U>(v_reg[t].w);
      }
    }
  };

  auto run_main_chunk = [&](int ii_base) __attribute__((always_inline)) {
    const device T* keys = K_main + (size_t)ii_base * k_seq_stride + loff;
    const device T* values = V_main + (size_t)ii_base * k_seq_stride + loff;
    bfloat4 k_reg[TK];
    bfloat4 v_reg[TK];
    for (int t = 0; t < TK; t++) {
      k_reg[t] = as_type<bfloat4>(((const device uint2*)(keys + t * stride_k_tok))[0]);
    }
    for (int t = 0; t < TK; t++) {
      v_reg[t] = as_type<bfloat4>(((const device uint2*)(values + t * stride_v_tok))[0]);
    }
    for (int t = 0; t < TK; t++) {
      int ii = ii_base + t * blocks;
      float fm = 0;
      if (msk.at(ii, fm)) {
        U score = 0;
        score += q[0] * static_cast<U>(k_reg[t].x);
        score += q[1] * static_cast<U>(k_reg[t].y);
        score += q[2] * static_cast<U>(k_reg[t].z);
        score += q[3] * static_cast<U>(k_reg[t].w);
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        o[0] = o[0] * factor + exp_score * static_cast<U>(v_reg[t].x);
        o[1] = o[1] * factor + exp_score * static_cast<U>(v_reg[t].y);
        o[2] = o[2] * factor + exp_score * static_cast<U>(v_reg[t].z);
        o[3] = o[3] * factor + exp_score * static_cast<U>(v_reg[t].w);
      }
    }
  };

  for (; i <= dirty_lo && i + (TK - 1) * blocks < N; i += blocks * TK) {
    run_ds_chunk(i);
  }
  for (; i + (TK - 1) * blocks < slot_main; i += blocks * TK) {
    run_main_chunk(i);
  }
  for (; i < dirty_end && i + (TK - 1) * blocks < N; i += blocks * TK) {
    run_ds_chunk(i);
  }
  for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
    run_main_chunk(i);
  }
  for (; i < N; i += blocks) {
    float fm = 0;
    if (msk.at(i, fm)) {
      int ds = rot_wrap ? (i - slot_main + (i < slot_main ? max_size : 0)) : (i - slot_main);
      const device T* kp = (ds == st_len) ? (Knew + loff) : ((ds >= 0 && ds < st_len) ? (Kst + (size_t)ds * D + loff) : (K_main + (size_t)i * D + loff));
      const device T* vp = (ds == st_len) ? (Vnew + loff) : ((ds >= 0 && ds < st_len) ? (Vst + (size_t)ds * D + loff) : (V_main + (size_t)i * D + loff));
      bfloat4 kb = as_type<bfloat4>(((const device uint2*)kp)[0]);
      bfloat4 vb = as_type<bfloat4>(((const device uint2*)vp)[0]);
      U score = 0;
      score += q[0] * kb.x;
      score += q[1] * kb.y;
      score += q[2] * kb.z;
      score += q[3] * kb.w;
      score = simd_sum(score);
      score += fm;
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      o[0] = o[0] * factor + exp_score * vb.x;
      o[1] = o[1] * factor + exp_score * vb.y;
      o[2] = o[2] * factor + exp_score * vb.z;
      o[3] = o[3] * factor + exp_score * vb.w;
    }
  }
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  bfloat4 ob = bfloat4(static_cast<T>(o[0]), static_cast<T>(o[1]), static_cast<T>(o[2]), static_cast<T>(o[3]));
  ((device uint2*)out)[0] = as_type<uint2>(ob);
}
"""

_STAGED_SHORT_SRC = r"""
    constexpr int D = 128;
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = (int)t / HQ;
    int h = (int)t % HQ;
    int kvh = h / GQA;
    int gb = B_OFF + b;
    int bh_in = gb * HQ + h;
    int bh_out = (int)t;
    const int S = meta[gb * MW + 0];
    const device T* K; const device T* V;
    SELECT_KV
    const int kcap = meta[gb * MW + 4];
    const int vcap = meta[gb * MW + 5];
    const int slot_main = meta[gb * MW + 6];
    const int rot_wrap = meta[gb * MW + 7];
    const int max_size = meta[gb * MW + 8];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[gb * MW + 1]; msk.idx = meta[gb * MW + 2]; msk.rot = meta[gb * MW + 3];
    msk.R = R; msk.window = WIN;
    msk.pb = pb + (size_t)bh_in * R;
    const int kv_lidx = st_info[0];
    const int scap = st_info[1];
    const int st_len = st_info[2];
    const device T* Kst = stage + ((((size_t)kv_lidx * ST_NROWS + gb) * 2 + 0) * HKV + kvh) * (size_t)scap * D;
    const device T* Vst = stage + ((((size_t)kv_lidx * ST_NROWS + gb) * 2 + 1) * HKV + kvh) * (size_t)scap * D;
    const device T* Knew = kv_new + ((size_t)(gb * 2 + 0) * HKV + kvh) * D;
    const device T* Vnew = kv_new + ((size_t)(gb * 2 + 1) * HKV + kvh) * D;
    float sink = float(static_cast<T>(qkv[SINK_OFF + (size_t)gb * SINK_STRIDE + h]));
    sdpa1_head_v32_staged<T, PBT, D, NS, true>(
        q + (size_t)bh_in * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
        Kst, Vst, Knew, Vnew,
        qkv + GATE_OFF + (size_t)gb * GATE_STRIDE + (size_t)h * D, o1 + (size_t)bh_out * D,
        S, scale[0], sink, msk, slot_main, rot_wrap, max_size, st_len,
        outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""

_STAGED_2PASS_SRC = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    constexpr int NTG2 = HKV * TGPK;
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = (int)t / NTG2;
    int tl = (int)t % NTG2;
    int gb = B_OFF + b;
    const int S = meta[gb * MW + 0];
    const device T* K; const device T* V;
    SELECT_KV
    const int kcap = meta[gb * MW + 4];
    const int vcap = meta[gb * MW + 5];
    const int slot_main = meta[gb * MW + 6];
    const int rot_wrap = meta[gb * MW + 7];
    const int max_size = meta[gb * MW + 8];
    const int kv_lidx = st_info[0];
    const int scap = st_info[1];
    const int st_len = st_info[2];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[gb * MW + 1]; msk.idx = meta[gb * MW + 2]; msk.rot = meta[gb * MW + 3];
    msk.R = R; msk.window = WIN;
    const float sc = scale[0];
    int kvh = tl / TGPK;
    int f = (tl % TGPK) * NS + (int)simd_gid;
    if (f < BLOCKS * GQA) {
        int block = f / GQA;
        int h = kvh * GQA + f % GQA;
        int bh_in = gb * HQ + h;
        int bh_out = b * HQ + h;
        msk.pb = pb + (size_t)bh_in * R;
        const device T* Kst = stage + ((((size_t)kv_lidx * ST_NROWS + gb) * 2 + 0) * HKV + kvh) * (size_t)scap * D;
        const device T* Vst = stage + ((((size_t)kv_lidx * ST_NROWS + gb) * 2 + 1) * HKV + kvh) * (size_t)scap * D;
        const device T* Knew = kv_new + ((size_t)(gb * 2 + 0) * HKV + kvh) * D;
        const device T* Vnew = kv_new + ((size_t)(gb * 2 + 1) * HKV + kvh) * D;
        sdpa2_block_staged<T, PBT, D, BLOCKS>(
            q + (size_t)bh_in * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
            Kst, Vst, Knew, Vnew,
            partials + ((size_t)bh_out * BLOCKS + block) * D, sums + (size_t)bh_out * BLOCKS + block,
            maxs + (size_t)bh_out * BLOCKS + block, S, sc,
            float(static_cast<T>(qkv[SINK_OFF + (size_t)gb * SINK_STRIDE + h])),
            msk, slot_main, rot_wrap, max_size, st_len, block, simd_lid);
    }
"""

_STAGED_MIXED_SRC = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    constexpr int NTG2 = HKV * TGPK;
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = 0, acc = 0;
    for (; b < NROWS; ++b) {
        int n = meta[(B_OFF + b) * MW] >= 1024 ? NTG2 : HQ;
        if ((int)t < acc + n) break;
        acc += n;
    }
    int tl = (int)t - acc;
    int gb = B_OFF + b;
    const int S = meta[gb * MW + 0];
    const device T* K; const device T* V;
    SELECT_KV
    const int kcap = meta[gb * MW + 4];
    const int vcap = meta[gb * MW + 5];
    const int slot_main = meta[gb * MW + 6];
    const int rot_wrap = meta[gb * MW + 7];
    const int max_size = meta[gb * MW + 8];
    const int kv_lidx = st_info[0];
    const int scap = st_info[1];
    const int st_len = st_info[2];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[gb * MW + 1]; msk.idx = meta[gb * MW + 2]; msk.rot = meta[gb * MW + 3];
    msk.R = R; msk.window = WIN;
    const float sc = scale[0];
    if (S < 1024) {
        int h = tl;
        int kvh = h / GQA;
        int bh_in = gb * HQ + h;
        int bh_out = b * HQ + h;
        msk.pb = pb + (size_t)bh_in * R;
        const device T* Kst = stage + ((((size_t)kv_lidx * ST_NROWS + gb) * 2 + 0) * HKV + kvh) * (size_t)scap * D;
        const device T* Vst = stage + ((((size_t)kv_lidx * ST_NROWS + gb) * 2 + 1) * HKV + kvh) * (size_t)scap * D;
        const device T* Knew = kv_new + ((size_t)(gb * 2 + 0) * HKV + kvh) * D;
        const device T* Vnew = kv_new + ((size_t)(gb * 2 + 1) * HKV + kvh) * D;
        sdpa1_head_v32_staged<T, PBT, D, 32, true>(
            q + (size_t)bh_in * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
            Kst, Vst, Knew, Vnew,
            qkv + GATE_OFF + (size_t)gb * GATE_STRIDE + (size_t)h * D, o1 + (size_t)bh_out * D,
            S, sc, float(static_cast<T>(qkv[SINK_OFF + (size_t)gb * SINK_STRIDE + h])),
            msk, slot_main, rot_wrap, max_size, st_len, outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
    } else {
        int kvh = tl / TGPK;
        int f = (tl % TGPK) * NS + (int)simd_gid;
        if (f < BLOCKS * GQA) {
            int block = f / GQA;
            int h = kvh * GQA + f % GQA;
            int bh_in = gb * HQ + h;
            int bh_out = b * HQ + h;
            msk.pb = pb + (size_t)bh_in * R;
            const device T* Kst = stage + ((((size_t)kv_lidx * ST_NROWS + gb) * 2 + 0) * HKV + kvh) * (size_t)scap * D;
            const device T* Vst = stage + ((((size_t)kv_lidx * ST_NROWS + gb) * 2 + 1) * HKV + kvh) * (size_t)scap * D;
            const device T* Knew = kv_new + ((size_t)(gb * 2 + 0) * HKV + kvh) * D;
            const device T* Vnew = kv_new + ((size_t)(gb * 2 + 1) * HKV + kvh) * D;
            sdpa2_block_staged<T, PBT, D, BLOCKS>(
                q + (size_t)bh_in * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
                Kst, Vst, Knew, Vnew,
                partials + ((size_t)bh_out * BLOCKS + block) * D, sums + (size_t)bh_out * BLOCKS + block,
                maxs + (size_t)bh_out * BLOCKS + block, S, sc,
                float(static_cast<T>(qkv[SINK_OFF + (size_t)gb * SINK_STRIDE + h])),
                msk, slot_main, rot_wrap, max_size, st_len, block, simd_lid);
        }
    }
"""

_STAGED_S2_SRC = r"""
    constexpr int D = 128;
    threadgroup float outputs[4 * 1024];
    uint bh = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint tix = thread_position_in_threadgroup.x;
    int b = int(bh) / HQ;
    int h = int(bh) % HQ;
    int gb = B_OFF + b;
    if (meta[gb * MW] >= 1024) {
        sdpa2_reduce<T, D, true>(
            partials + (size_t)bh * BLOCKS * D, sums + (size_t)bh * BLOCKS,
            maxs + (size_t)bh * BLOCKS,
            qkv + GATE_OFF + (size_t)gb * GATE_STRIDE + (size_t)h * D,
            out + (size_t)bh * D,
            BLOCKS, outputs, simd_gid, simd_lid);
    } else if (tix < D) {
        out[(size_t)bh * D + tix] = o1[(size_t)bh * D + tix];
    }
"""


_staged_kernels: dict = {}
_STAGED_META_CACHE: dict = {}
_ST_INFO_CACHE: dict = {}


def _get_staged_kernels(B):
    ks = _staged_kernels.get(B)
    if ks is None:
        sel = "switch (b) {\n" + "".join(
            f"        case {i}: K = k{i}; V = v{i}; break;\n" for i in range(B)
        ) + "        default: K = k0; V = v0; break;\n    }"
        kv_names = [f"k{i}" for i in range(B)] + [f"v{i}" for i in range(B)]
        ins_base = ["q", "pb", "qkv", "meta", "scale", "st_info", "stage", "kv_new"] + kv_names
        s1_short = mx.fast.metal_kernel(
            name=f"rows_sdpa_staged_short_b{B}",
            input_names=ins_base,
            output_names=["o1"],
            source=_STAGED_SHORT_SRC.replace("SELECT_KV", sel),
            header=_STAGED_HEADER,
            ensure_row_contiguous=False,
        )
        s1_2p = mx.fast.metal_kernel(
            name=f"rows_sdpa_staged_2p_b{B}",
            input_names=ins_base,
            output_names=["partials", "sums", "maxs"],
            source=_STAGED_2PASS_SRC.replace("SELECT_KV", sel),
            header=_STAGED_HEADER,
            ensure_row_contiguous=False,
        )
        s1_mix = mx.fast.metal_kernel(
            name=f"rows_sdpa_staged_mix_b{B}",
            input_names=ins_base,
            output_names=["o1", "partials", "sums", "maxs"],
            source=_STAGED_MIXED_SRC.replace("SELECT_KV", sel),
            header=_STAGED_HEADER,
            ensure_row_contiguous=False,
        )
        s2 = _staged_kernels.get("s2")
        if s2 is None:
            s2 = _staged_kernels["s2"] = mx.fast.metal_kernel(
                name="rows_sdpa_staged_s2",
                input_names=["partials", "sums", "maxs", "o1", "meta", "qkv"],
                output_names=["out"],
                source=_STAGED_S2_SRC,
                header=_HEADER,
                ensure_row_contiguous=False,
            )
        ks = _staged_kernels[B] = (s1_short, s1_2p, s1_mix, s2)
    return ks


def sdpa_rows_staged(
    q, ks_main, vs_main, stage, kv_new, pb, qkv_flat, gate_off, gate_stride, sink_off, sink_stride, meta,
    *, scale, window, kv_layer_idx, st_len, scap=32, meta_arr=None, Ss=None, st_info_arr=None,
):
    """Exact B-row (1..32) decode attention over unpacked main KV + stage + current kv_new, fused with gate epilogue."""
    B = q.shape[0]
    assert 1 <= B <= 32
    assert q.dtype == mx.bfloat16 and q.shape[-1] == 128
    Hq = q.shape[1]
    Hkv = ks_main[0].shape[1]
    gqa = Hq // Hkv
    assert Hq == gqa * Hkv and 3 <= gqa <= 5
    R = pb.shape[-1]
    if q.ndim != 3:
        q = q.reshape(B, Hq, 128)
    qc = mx.contiguous(q)
    pbc = mx.contiguous(pb.reshape(B, Hq, R) if pb.ndim != 3 else pb)
    if meta_arr is not None:
        mta = meta_arr
        assert Ss is not None and len(Ss) == B
    else:
        mt_key = tuple(meta)
        mta = _STAGED_META_CACHE.get(mt_key)
        if mta is None:
            if len(_STAGED_META_CACHE) >= 256:
                _STAGED_META_CACHE.clear()
            mt = np.empty((B, _SMW), np.int32)
            for i, row in enumerate(meta):
                mt[i] = row
            mta = _STAGED_META_CACHE[mt_key] = mx.array(mt)
        Ss = tuple(int(row[0]) for row in meta)
    st_nrows = int(stage.shape[1]) if stage.ndim >= 2 else B
    sck = float(scale)
    sc = _SCALE_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_CACHE[sck] = mx.array([sck], mx.float32)
    if st_info_arr is not None:
        st_info = st_info_arr
    else:
        st_key = (int(kv_layer_idx), int(scap), int(st_len))
        st_info = _ST_INFO_CACHE.get(st_key)
        if st_info is None:
            if len(_ST_INFO_CACHE) >= 4096:
                _ST_INFO_CACHE.clear()
            st_info = _ST_INFO_CACHE[st_key] = mx.array(st_key, mx.int32)

    def _run_chunk(b0, b1):
        bc = b1 - b0
        Ss_c = Ss[b0:b1]
        max_s = max(Ss_c)
        blocks = _blocks(max_s, gqa)
        assert blocks == 128
        has2 = max_s >= _TWO_PASS_N
        all2 = min(Ss_c) >= _TWO_PASS_N
        s1_short, s1_2p, s1_mix, s2 = _get_staged_kernels(bc)
        ins = [qc, pbc, qkv_flat, mta, sc, st_info, stage, kv_new] + list(ks_main[b0:b1]) + list(vs_main[b0:b1])
        if not has2:
            ns_env = os.environ.get("LOOPLING_SDPA1_NS", "0")
            ns = int(ns_env) if ns_env and ns_env != "0" else (32 if bc <= 2 else 8)
            tmpl = [
                ("T", mx.bfloat16), ("PBT", pb.dtype), ("NROWS", bc), ("ST_NROWS", st_nrows), ("B_OFF", b0),
                ("MW", _SMW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa), ("R", R), ("WIN", int(window)),
                ("NS", ns), ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", int(gate_stride)),
                ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", int(sink_stride)),
            ]
            (o1,) = s1_short(
                inputs=ins,
                template=tmpl,
                grid=(bc * Hq * 32 * ns, 1, 1),
                threadgroup=(32 * ns, 1, 1),
                output_shapes=[(bc, Hq * 128)],
                output_dtypes=[mx.bfloat16],
            )
            return o1
        n2 = bc * Hq * blocks
        if all2:
            ns2_env = os.environ.get("LOOPLING_SDPA2_STAGED_NS", "")
            if ns2_env and ns2_env != "0":
                ns = int(ns2_env)
            elif B >= 8:
                ns = min(32, 8 * gqa)
            else:
                ns = gqa
            tgpk = (blocks * gqa + ns - 1) // ns
            ntg = bc * Hkv * tgpk
            tmpl = [
                ("T", mx.bfloat16), ("PBT", pb.dtype), ("NROWS", bc), ("ST_NROWS", st_nrows), ("B_OFF", b0),
                ("MW", _SMW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa), ("BLOCKS", blocks), ("R", R),
                ("WIN", int(window)), ("NS", ns), ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", int(sink_stride)),
            ]
            partials, sums, maxs = s1_2p(
                inputs=ins,
                template=tmpl,
                grid=(ntg * 32 * ns, 1, 1),
                threadgroup=(32 * ns, 1, 1),
                output_shapes=[(n2 * 128,), (n2,), (n2,)],
                output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
            )
            o1 = _DUMMY_GATE
        else:
            ns = 32
            tgpk = (blocks * gqa + ns - 1) // ns
            ntg = sum(Hkv * tgpk if s >= _TWO_PASS_N else Hq for s in Ss_c)
            tmpl = [
                ("T", mx.bfloat16), ("PBT", pb.dtype), ("NROWS", bc), ("ST_NROWS", st_nrows), ("B_OFF", b0),
                ("MW", _SMW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa), ("BLOCKS", blocks), ("R", R),
                ("WIN", int(window)), ("NS", ns), ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", int(gate_stride)),
                ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", int(sink_stride)),
            ]
            o1, partials, sums, maxs = s1_mix(
                inputs=ins,
                template=tmpl,
                grid=(ntg * 32 * ns, 1, 1),
                threadgroup=(32 * ns, 1, 1),
                output_shapes=[(bc, Hq * 128), (n2 * 128,), (n2,), (n2,)],
                output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
            )
        (out,) = s2(
            inputs=[partials, sums, maxs, o1, mta, qkv_flat],
            template=[
                ("T", mx.bfloat16), ("B_OFF", b0), ("MW", _SMW), ("HQ", Hq), ("BLOCKS", blocks),
                ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", int(gate_stride)),
            ],
            grid=(bc * Hq * 1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(bc, Hq * 128)],
            output_dtypes=[mx.bfloat16],
        )
        return out

    # Metal 31-buffer cap: s1_mix binds 12+2*bc <= 31 => bc <= 9; s1_short/s1_2p bind <= 11+2*bc <= 31 => bc <= 10.
    is_mix = (max(Ss) >= _TWO_PASS_N) and (min(Ss) < _TWO_PASS_N)
    if B <= (9 if is_mix else MAX_ROWS):
        return _run_chunk(0, B)
    step = int(os.environ.get("LOOPLING_SDPA_STAGED_CHUNK", "8") or "8")
    step = max(1, min(9 if is_mix else MAX_ROWS, step))
    outs = [_run_chunk(b0, min(B, b0 + step)) for b0 in range(0, B, step)]
    return mx.concatenate(outs, axis=0)


# =============================================================================================== Multi-stream causal staged SDPA
_MCW = 8  # end_m, slot_main, rot, rot_wrap, max_size, kcap, vcap, st_len

_STAGED_MULTI_HEADER = _HEADER + r"""
template <typename T, typename PBT, int D, int NS, bool HAS_GATE>
METAL_FUNC void sdpa1_head_v32_multi_causal(
    const device T* queries, const device T* K_main, const device T* V_main,
    const device T* Kst, const device T* Vst, const device T* Knew, const device T* Vnew,
    const device T* gate, device T* out, const int N, const float scale, const float sink,
    const RowMask<T, PBT> msk, const int slot_main, const int rot_wrap, const int max_size,
    const int st_len, const int dd, threadgroup float* outputs, threadgroup float* max_scores,
    threadgroup float* sum_exp_scores, uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int NV = BN / NS;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread];
  const int loff = (int)simd_lid * qk_per_thread;
  queries += loff;
  bfloat4 qb = as_type<bfloat4>(((const device uint2*)queries)[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  for (int v = 0; v < NV; v++) {
    int vg = (int)simd_gid + v * NS;
    for (int i = 0; i < v_per_thread; i++) o[i] = 0;
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    if (vg == 0) {
      max_score = sink;
      sum_exp_score = 1;
    }
    for (int i = vg; i < N; i += BN) {
      float fm = 0;
      if (msk.at(i, fm)) {
        int ds = rot_wrap ? (i - slot_main + (i < slot_main ? max_size : 0)) : (i - slot_main);
        const device T* kp;
        const device T* vp;
        if (ds >= st_len && ds <= st_len + dd) {
          int off = ds - st_len;
          kp = Knew + (size_t)off * D + loff;
          vp = Vnew + (size_t)off * D + loff;
        } else if (ds >= 0 && ds < st_len) {
          kp = Kst + (size_t)ds * D + loff;
          vp = Vst + (size_t)ds * D + loff;
        } else {
          kp = K_main + (size_t)i * D + loff;
          vp = V_main + (size_t)i * D + loff;
        }
        bfloat4 kb = as_type<bfloat4>(((const device uint2*)kp)[0]);
        U score = 0;
        score += q[0] * kb.x;
        score += q[1] * kb.y;
        score += q[2] * kb.z;
        score += q[3] * kb.w;
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        bfloat4 vb = as_type<bfloat4>(((const device uint2*)vp)[0]);
        o[0] = o[0] * factor + exp_score * vb.x;
        o[1] = o[1] * factor + exp_score * vb.y;
        o[2] = o[2] * factor + exp_score * vb.z;
        o[3] = o[3] * factor + exp_score * vb.w;
      }
    }
    if (simd_lid == 0) {
      max_scores[vg] = max_score;
      sum_exp_scores[vg] = sum_exp_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
      outputs[(i * BD + simd_lid) * BN + vg] = o[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int f = (int)simd_gid; f < BD; f += NS) {
    for (int i = 0; i < v_per_thread; i++) {
      o[i] = simd_sum(outputs[(i * BD + f) * BN + simd_lid] * factor);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    }
    if (simd_lid == 0) {
      int base = f * v_per_thread;
      if constexpr (HAS_GATE) {
        bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
        bfloat4 ob = bfloat4(
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[0])), static_cast<float>(gb.x))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[1])), static_cast<float>(gb.y))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[2])), static_cast<float>(gb.z))),
            static_cast<T>(glue_gate(static_cast<float>(static_cast<T>(o[3])), static_cast<float>(gb.w))));
        ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
      } else {
        bfloat4 ob = bfloat4(static_cast<T>(o[0]), static_cast<T>(o[1]), static_cast<T>(o[2]), static_cast<T>(o[3]));
        ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
      }
    }
  }
}

template <typename T, typename PBT, int D, int BLOCKS>
METAL_FUNC void sdpa2_block_multi_causal(
    const device T* queries, const device T* K_main, const device T* V_main,
    const device T* Kst, const device T* Vst, const device T* Knew, const device T* Vnew,
    device T* out, device float* sums, device float* maxs, const int N, const float scale,
    const float sink, const RowMask<T, PBT> msk, const int slot_main, const int rot_wrap,
    const int max_size, const int st_len, const int dd, const int block_idx, uint simd_lid) {
  constexpr int TK = 4;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int blocks = BLOCKS;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};
  const int loff = (int)simd_lid * qk_per_thread;
  queries += loff;
  out += loff;
  bfloat4 qb = as_type<bfloat4>(((const device uint2*)queries)[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (block_idx == 0) {
    max_score = sink;
    sum_exp_score = 1;
  }
  int i = block_idx;
  for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
    bfloat4 k_reg[TK];
    bfloat4 v_reg[TK];
    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      int ds = rot_wrap ? (ii - slot_main + (ii < slot_main ? max_size : 0)) : (ii - slot_main);
      const device T* kp;
      if (ds >= st_len && ds <= st_len + dd) kp = Knew + (size_t)(ds - st_len) * D + loff;
      else if (ds >= 0 && ds < st_len) kp = Kst + (size_t)ds * D + loff;
      else kp = K_main + (size_t)ii * D + loff;
      k_reg[t] = as_type<bfloat4>(((const device uint2*)kp)[0]);
    }
    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      int ds = rot_wrap ? (ii - slot_main + (ii < slot_main ? max_size : 0)) : (ii - slot_main);
      const device T* vp;
      if (ds >= st_len && ds <= st_len + dd) vp = Vnew + (size_t)(ds - st_len) * D + loff;
      else if (ds >= 0 && ds < st_len) vp = Vst + (size_t)ds * D + loff;
      else vp = V_main + (size_t)ii * D + loff;
      v_reg[t] = as_type<bfloat4>(((const device uint2*)vp)[0]);
    }
    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      float fm = 0;
      if (msk.at(ii, fm)) {
        U score = 0;
        score += q[0] * static_cast<U>(k_reg[t].x);
        score += q[1] * static_cast<U>(k_reg[t].y);
        score += q[2] * static_cast<U>(k_reg[t].z);
        score += q[3] * static_cast<U>(k_reg[t].w);
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        o[0] = o[0] * factor + exp_score * static_cast<U>(v_reg[t].x);
        o[1] = o[1] * factor + exp_score * static_cast<U>(v_reg[t].y);
        o[2] = o[2] * factor + exp_score * static_cast<U>(v_reg[t].z);
        o[3] = o[3] * factor + exp_score * static_cast<U>(v_reg[t].w);
      }
    }
  }
  for (; i < N; i += blocks) {
    float fm = 0;
    if (msk.at(i, fm)) {
      int ds = rot_wrap ? (i - slot_main + (i < slot_main ? max_size : 0)) : (i - slot_main);
      const device T* kp;
      const device T* vp;
      if (ds >= st_len && ds <= st_len + dd) {
        int off = ds - st_len;
        kp = Knew + (size_t)off * D + loff;
        vp = Vnew + (size_t)off * D + loff;
      } else if (ds >= 0 && ds < st_len) {
        kp = Kst + (size_t)ds * D + loff;
        vp = Vst + (size_t)ds * D + loff;
      } else {
        kp = K_main + (size_t)i * D + loff;
        vp = V_main + (size_t)i * D + loff;
      }
      bfloat4 kb = as_type<bfloat4>(((const device uint2*)kp)[0]);
      U score = 0;
      score += q[0] * kb.x;
      score += q[1] * kb.y;
      score += q[2] * kb.z;
      score += q[3] * kb.w;
      score = simd_sum(score);
      score += fm;
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      bfloat4 vb = as_type<bfloat4>(((const device uint2*)vp)[0]);
      o[0] = o[0] * factor + exp_score * vb.x;
      o[1] = o[1] * factor + exp_score * vb.y;
      o[2] = o[2] * factor + exp_score * vb.z;
      o[3] = o[3] * factor + exp_score * vb.w;
    }
  }
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  bfloat4 ob = bfloat4(static_cast<T>(o[0]), static_cast<T>(o[1]), static_cast<T>(o[2]), static_cast<T>(o[3]));
  ((device uint2*)out)[0] = as_type<uint2>(ob);
}
"""

_MC_COMMON = r"""
    int c = r / WW;
    int dd = r % WW;
    const int end_m = meta[c * 8 + 0];
    const int slot_main = meta[c * 8 + 1];
    const int rot = meta[c * 8 + 2];
    const int rot_wrap = meta[c * 8 + 3];
    const int max_size = meta[c * 8 + 4];
    const int kcap = meta[c * 8 + 5];
    const int vcap = meta[c * 8 + 6];
    const int st_len = meta[c * 8 + 7];
    const int end_r = end_m + st_len + dd + 1;
    const int S = rot ? min(end_r, max_size) : end_r;
    const int idx_r = rot ? (rot_wrap ? (((slot_main + st_len + dd) % max_size) + 1) : end_r) : 0;
"""

_MULTI_CAUSAL_SHORT_SRC = r"""
    constexpr int D = 128;
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int r = (int)t / HQ;
    int h = (int)t % HQ;
    int kvh = h / GQA;
    int rh = (int)t;
""" + _MC_COMMON + r"""
    const device T* K; const device T* V;
    SELECT_KV
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = end_r; msk.idx = idx_r; msk.rot = rot;
    msk.R = R; msk.window = WIN;
    msk.pb = pb + (size_t)rh * R;
    const int kv_lidx = st_info[0];
    const int scap = st_info[1];
    const device T* Kst = stage + ((((size_t)kv_lidx * CC + c) * 2 + 0) * HKV + kvh) * (size_t)scap * D;
    const device T* Vst = stage + ((((size_t)kv_lidx * CC + c) * 2 + 1) * HKV + kvh) * (size_t)scap * D;
    const device T* Knew = kv_new + ((size_t)kvh * (CC * WW) + (size_t)c * WW) * D;
    const device T* Vnew = kv_new + ((size_t)(HKV + kvh) * (CC * WW) + (size_t)c * WW) * D;
    float sink = float(static_cast<T>(qkv[SINK_OFF + (size_t)r * SINK_STRIDE + h]));
    sdpa1_head_v32_multi_causal<T, PBT, D, NS, true>(
        q + (size_t)rh * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
        Kst, Vst, Knew, Vnew,
        qkv + GATE_OFF + (size_t)r * GATE_STRIDE + (size_t)h * D, o1 + (size_t)rh * D,
        S, scale[0], sink, msk, slot_main, rot_wrap, max_size, st_len, dd,
        outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""

_MULTI_CAUSAL_2P_SRC = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    constexpr int NTG2 = HKV * TGPK;
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int r = (int)t / NTG2;
    int tl = (int)t % NTG2;
""" + _MC_COMMON + r"""
    const device T* K; const device T* V;
    SELECT_KV
    const int kv_lidx = st_info[0];
    const int scap = st_info[1];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = end_r; msk.idx = idx_r; msk.rot = rot;
    msk.R = R; msk.window = WIN;
    const float sc = scale[0];
    int kvh = tl / TGPK;
    int f = (tl % TGPK) * NS + (int)simd_gid;
    if (f < BLOCKS * GQA) {
        int block = f / GQA;
        int h = kvh * GQA + f % GQA;
        int rh = r * HQ + h;
        msk.pb = pb + (size_t)rh * R;
        const device T* Kst = stage + ((((size_t)kv_lidx * CC + c) * 2 + 0) * HKV + kvh) * (size_t)scap * D;
        const device T* Vst = stage + ((((size_t)kv_lidx * CC + c) * 2 + 1) * HKV + kvh) * (size_t)scap * D;
        const device T* Knew = kv_new + ((size_t)kvh * (CC * WW) + (size_t)c * WW) * D;
        const device T* Vnew = kv_new + ((size_t)(HKV + kvh) * (CC * WW) + (size_t)c * WW) * D;
        sdpa2_block_multi_causal<T, PBT, D, BLOCKS>(
            q + (size_t)rh * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
            Kst, Vst, Knew, Vnew,
            partials + ((size_t)rh * BLOCKS + block) * D, sums + (size_t)rh * BLOCKS + block,
            maxs + (size_t)rh * BLOCKS + block, S, sc,
            float(static_cast<T>(qkv[SINK_OFF + (size_t)r * SINK_STRIDE + h])),
            msk, slot_main, rot_wrap, max_size, st_len, dd, block, simd_lid);
    }
"""

_MULTI_CAUSAL_MIX_SRC = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    constexpr int NTG2 = HKV * TGPK;
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int r = 0, acc = 0;
    for (; r < CC * WW; ++r) {
        int cr = r / WW, dr = r % WW;
        int er = meta[cr * 8 + 0] + meta[cr * 8 + 7] + dr + 1;
        int sr = meta[cr * 8 + 2] ? min(er, meta[cr * 8 + 4]) : er;
        int n = sr >= 1024 ? NTG2 : HQ;
        if ((int)t < acc + n) break;
        acc += n;
    }
    int tl = (int)t - acc;
""" + _MC_COMMON + r"""
    const device T* K; const device T* V;
    SELECT_KV
    const int kv_lidx = st_info[0];
    const int scap = st_info[1];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = end_r; msk.idx = idx_r; msk.rot = rot;
    msk.R = R; msk.window = WIN;
    const float sc = scale[0];
    if (S < 1024) {
        int h = tl;
        int kvh = h / GQA;
        int rh = r * HQ + h;
        msk.pb = pb + (size_t)rh * R;
        const device T* Kst = stage + ((((size_t)kv_lidx * CC + c) * 2 + 0) * HKV + kvh) * (size_t)scap * D;
        const device T* Vst = stage + ((((size_t)kv_lidx * CC + c) * 2 + 1) * HKV + kvh) * (size_t)scap * D;
        const device T* Knew = kv_new + ((size_t)kvh * (CC * WW) + (size_t)c * WW) * D;
        const device T* Vnew = kv_new + ((size_t)(HKV + kvh) * (CC * WW) + (size_t)c * WW) * D;
        sdpa1_head_v32_multi_causal<T, PBT, D, 32, true>(
            q + (size_t)rh * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
            Kst, Vst, Knew, Vnew,
            qkv + GATE_OFF + (size_t)r * GATE_STRIDE + (size_t)h * D, o1 + (size_t)rh * D,
            S, sc, float(static_cast<T>(qkv[SINK_OFF + (size_t)r * SINK_STRIDE + h])),
            msk, slot_main, rot_wrap, max_size, st_len, dd, outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
    } else {
        int kvh = tl / TGPK;
        int f = (tl % TGPK) * NS + (int)simd_gid;
        if (f < BLOCKS * GQA) {
            int block = f / GQA;
            int h = kvh * GQA + f % GQA;
            int rh = r * HQ + h;
            msk.pb = pb + (size_t)rh * R;
            const device T* Kst = stage + ((((size_t)kv_lidx * CC + c) * 2 + 0) * HKV + kvh) * (size_t)scap * D;
            const device T* Vst = stage + ((((size_t)kv_lidx * CC + c) * 2 + 1) * HKV + kvh) * (size_t)scap * D;
            const device T* Knew = kv_new + ((size_t)kvh * (CC * WW) + (size_t)c * WW) * D;
            const device T* Vnew = kv_new + ((size_t)(HKV + kvh) * (CC * WW) + (size_t)c * WW) * D;
            sdpa2_block_multi_causal<T, PBT, D, BLOCKS>(
                q + (size_t)rh * D, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D,
                Kst, Vst, Knew, Vnew,
                partials + ((size_t)rh * BLOCKS + block) * D, sums + (size_t)rh * BLOCKS + block,
                maxs + (size_t)rh * BLOCKS + block, S, sc,
                float(static_cast<T>(qkv[SINK_OFF + (size_t)r * SINK_STRIDE + h])),
                msk, slot_main, rot_wrap, max_size, st_len, dd, block, simd_lid);
        }
    }
"""

_MULTI_CAUSAL_S2_SRC = r"""
    constexpr int D = 128;
    threadgroup float outputs[4 * 1024];
    uint rh = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint tix = thread_position_in_threadgroup.x;
    int r = int(rh) / HQ;
    int h = int(rh) % HQ;
    int c = r / WW;
    int dd = r % WW;
    int end_r = meta[c * 8 + 0] + meta[c * 8 + 7] + dd + 1;
    int S = meta[c * 8 + 2] ? min(end_r, meta[c * 8 + 4]) : end_r;
    if (S >= 1024) {
        sdpa2_reduce<T, D, true>(
            partials + (size_t)rh * BLOCKS * D, sums + (size_t)rh * BLOCKS,
            maxs + (size_t)rh * BLOCKS,
            qkv + GATE_OFF + (size_t)r * GATE_STRIDE + (size_t)h * D,
            out + (size_t)rh * D,
            BLOCKS, outputs, simd_gid, simd_lid);
    } else if (tix < D) {
        out[(size_t)rh * D + tix] = o1[(size_t)rh * D + tix];
    }
"""

_multi_causal_kernels: dict = {}


def _get_multi_causal_kernels(C):
    ks = _multi_causal_kernels.get(C)
    if ks is None:
        sel = "switch (c) {\n" + "".join(
            f"        case {i}: K = k{i}; V = v{i}; break;\n" for i in range(C)
        ) + "        default: K = k0; V = v0; break;\n    }"
        kv_names = [f"k{i}" for i in range(C)] + [f"v{i}" for i in range(C)]
        ins_base = ["q", "pb", "qkv", "meta", "scale", "st_info", "stage", "kv_new"] + kv_names
        s1_short = mx.fast.metal_kernel(
            name=f"multi_causal_short_c{C}",
            input_names=ins_base,
            output_names=["o1"],
            source=_MULTI_CAUSAL_SHORT_SRC.replace("SELECT_KV", sel),
            header=_STAGED_MULTI_HEADER,
            ensure_row_contiguous=False,
        )
        s1_2p = mx.fast.metal_kernel(
            name=f"multi_causal_2p_c{C}",
            input_names=ins_base,
            output_names=["partials", "sums", "maxs"],
            source=_MULTI_CAUSAL_2P_SRC.replace("SELECT_KV", sel),
            header=_STAGED_MULTI_HEADER,
            ensure_row_contiguous=False,
        )
        s1_mix = mx.fast.metal_kernel(
            name=f"multi_causal_mix_c{C}",
            input_names=ins_base,
            output_names=["o1", "partials", "sums", "maxs"],
            source=_MULTI_CAUSAL_MIX_SRC.replace("SELECT_KV", sel),
            header=_STAGED_MULTI_HEADER,
            ensure_row_contiguous=False,
        )
        s2 = _multi_causal_kernels.get("s2")
        if s2 is None:
            s2 = _multi_causal_kernels["s2"] = mx.fast.metal_kernel(
                name="multi_causal_s2",
                input_names=["partials", "sums", "maxs", "o1", "meta", "qkv"],
                output_names=["out"],
                source=_MULTI_CAUSAL_S2_SRC,
                header=_HEADER,
                ensure_row_contiguous=False,
            )
        ks = _multi_causal_kernels[C] = (s1_short, s1_2p, s1_mix, s2)
    return ks


def sdpa_multi_causal_w_staged(
    q, ks_main, vs_main, stage, kv_new, pb, qkv_flat, gate_off, gate_stride, sink_off, sink_stride,
    meta_arr, Ss, C, W, *, scale, window, kv_layer_idx, scap=64, Hq=24, Hkv=8,
):
    """Multi-stream causal W-token staged SDPA over unpacked main KV + stage buffer + kv_new (B = C * W)."""
    B = int(C) * int(W)
    assert 2 <= C <= 8 and 2 <= W <= 8 and B <= 16
    assert len(ks_main) == C and len(vs_main) == C and len(Ss) == B
    assert q.dtype == mx.bfloat16 and q.shape[0] == B
    gqa = Hq // Hkv
    R = int(pb.shape[-1])
    qc = mx.contiguous(q.reshape(B, Hq, 128) if q.ndim != 3 else q)
    pbc = mx.contiguous(pb.reshape(B, Hq, R) if pb.ndim != 3 else pb)
    max_s = max(Ss)
    blocks = _blocks(max_s, gqa)
    assert blocks == 128
    has2 = max_s >= _TWO_PASS_N
    all2 = min(Ss) >= _TWO_PASS_N
    sck = float(scale)
    sc = _SCALE_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_CACHE[sck] = mx.array([sck], mx.float32)
    st_key = (int(kv_layer_idx), int(scap))
    st_info = _ST_INFO_CACHE.get(st_key)
    if st_info is None:
        if len(_ST_INFO_CACHE) >= 4096:
            _ST_INFO_CACHE.clear()
        st_info = _ST_INFO_CACHE[st_key] = mx.array(st_key, mx.int32)
    s1_short, s1_2p, s1_mix, s2 = _get_multi_causal_kernels(C)
    ins = [qc, pbc, qkv_flat, meta_arr, sc, st_info, stage, kv_new] + list(ks_main) + list(vs_main)
    if not has2:
        ns_env = os.environ.get("LOOPLING_SDPA1_NS", "0")
        ns = int(ns_env) if ns_env and ns_env != "0" else (32 if B <= 2 else 8)
        tmpl = [
            ("T", mx.bfloat16), ("PBT", pb.dtype), ("CC", int(C)), ("WW", int(W)),
            ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa), ("R", R), ("WIN", int(window)),
            ("NS", ns), ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", int(gate_stride)),
            ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", int(sink_stride)),
        ]
        (o1,) = s1_short(
            inputs=ins,
            template=tmpl,
            grid=(B * Hq * 32 * ns, 1, 1),
            threadgroup=(32 * ns, 1, 1),
            output_shapes=[(B, Hq * 128)],
            output_dtypes=[mx.bfloat16],
        )
        return o1
    n2 = B * Hq * blocks
    if all2:
        ns = gqa
        tgpk = (blocks * gqa + ns - 1) // ns
        ntg = B * Hkv * tgpk
        tmpl = [
            ("T", mx.bfloat16), ("PBT", pb.dtype), ("CC", int(C)), ("WW", int(W)),
            ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa), ("BLOCKS", blocks), ("R", R),
            ("WIN", int(window)), ("NS", ns), ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", int(sink_stride)),
        ]
        partials, sums, maxs = s1_2p(
            inputs=ins,
            template=tmpl,
            grid=(ntg * 32 * ns, 1, 1),
            threadgroup=(32 * ns, 1, 1),
            output_shapes=[(n2 * 128,), (n2,), (n2,)],
            output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
        )
        o1 = _DUMMY_GATE
    else:
        ns = 32
        tgpk = (blocks * gqa + ns - 1) // ns
        ntg = sum(Hkv * tgpk if s >= _TWO_PASS_N else Hq for s in Ss)
        tmpl = [
            ("T", mx.bfloat16), ("PBT", pb.dtype), ("CC", int(C)), ("WW", int(W)),
            ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa), ("BLOCKS", blocks), ("R", R),
            ("WIN", int(window)), ("NS", ns), ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", int(gate_stride)),
            ("SINK_OFF", int(sink_off)), ("SINK_STRIDE", int(sink_stride)),
        ]
        o1, partials, sums, maxs = s1_mix(
            inputs=ins,
            template=tmpl,
            grid=(ntg * 32 * ns, 1, 1),
            threadgroup=(32 * ns, 1, 1),
            output_shapes=[(B, Hq * 128), (n2 * 128,), (n2,), (n2,)],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
        )
    (out,) = s2(
        inputs=[partials, sums, maxs, o1, meta_arr, qkv_flat],
        template=[
            ("T", mx.bfloat16), ("CC", int(C)), ("WW", int(W)), ("HQ", Hq), ("BLOCKS", blocks),
            ("GATE_OFF", int(gate_off)), ("GATE_STRIDE", int(gate_stride)),
        ],
        grid=(B * Hq * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B, Hq * 128)],
        output_dtypes=[mx.bfloat16],
    )
    return out


# =============================================================================================== B=1 packed staged SDPA
# Eliminates 48x/step K7 slice_update copies on C=1 by reading main stacked/packed KV + small ring/full stage + kv_new.
# meta[0..10] = (S, end, idx, rot, cap, kv_lidx, slot_main, rot_wrap, max_size, scap, st_len)
_B1_STAGED_SHORT = r"""
    constexpr int D = 128;
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    int h = (int)threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int kvh = h / GQA;
    const int S = meta[0];
    const int cap = meta[4];
    const int lidx = meta[5];
    const int slot_main = meta[6];
    const int rot_wrap = meta[7];
    const int max_size = meta[8];
    const int scap = meta[9];
    const int st_len = meta[10];
    const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * (size_t)cap * D;
    const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * (size_t)cap * D;
    const device T* Kst = stage + ((size_t)lidx * 2 * HKV + kvh) * (size_t)scap * D;
    const device T* Vst = stage + ((size_t)lidx * 2 * HKV + HKV + kvh) * (size_t)scap * D;
    const device T* Knew = kv_new + (size_t)kvh * D;
    const device T* Vnew = kv_new + (size_t)(HKV + kvh) * D;
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[1]; msk.idx = meta[2]; msk.rot = meta[3];
    msk.R = R; msk.window = WIN;
    msk.pb = pb + (size_t)h * R;
    float sink = float(static_cast<T>(qkv[SINK_OFF + h]));
    sdpa1_head_v32_staged<T, PBT, D, NS, true>(
        q + (size_t)h * D, K, V, Kst, Vst, Knew, Vnew,
        qkv + GATE_OFF + (size_t)h * D, out + (size_t)h * D,
        S, scale[0], sink, msk, slot_main, rot_wrap, max_size, st_len,
        outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""

_B1_STAGED_2P1 = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    uint tl = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const int S = meta[0];
    const int cap = meta[4];
    const int lidx = meta[5];
    const int slot_main = meta[6];
    const int rot_wrap = meta[7];
    const int max_size = meta[8];
    const int scap = meta[9];
    const int st_len = meta[10];
    int kvh = (int)tl / TGPK;
    int f = ((int)tl % TGPK) * NS + (int)simd_gid;
    if (f < BLOCKS * GQA) {
        int block = f / GQA;
        int h = kvh * GQA + f % GQA;
        const device T* K = kv + ((size_t)lidx * 2 * HKV + kvh) * (size_t)cap * D;
        const device T* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * (size_t)cap * D;
        const device T* Kst = stage + ((size_t)lidx * 2 * HKV + kvh) * (size_t)scap * D;
        const device T* Vst = stage + ((size_t)lidx * 2 * HKV + HKV + kvh) * (size_t)scap * D;
        const device T* Knew = kv_new + (size_t)kvh * D;
        const device T* Vnew = kv_new + (size_t)(HKV + kvh) * D;
        RowMask<T, PBT> msk;
        msk.S = S; msk.end = meta[1]; msk.idx = meta[2]; msk.rot = meta[3];
        msk.R = R; msk.window = WIN;
        msk.pb = pb + (size_t)h * R;
        sdpa2_block_staged<T, PBT, D, BLOCKS>(
            q + (size_t)h * D, K, V, Kst, Vst, Knew, Vnew,
            partials + ((size_t)h * BLOCKS + block) * D, sums + (size_t)h * BLOCKS + block,
            maxs + (size_t)h * BLOCKS + block, S, scale[0],
            float(static_cast<T>(qkv[SINK_OFF + h])), msk,
            slot_main, rot_wrap, max_size, st_len, block, simd_lid);
    }
"""

_b1_staged_kernels = None


def _get_b1_staged_kernels():
    global _b1_staged_kernels
    if _b1_staged_kernels is None:
        s1_short = mx.fast.metal_kernel(
            name="b1_staged_sdpa_short",
            input_names=["q", "kv", "stage", "kv_new", "pb", "qkv", "meta", "scale"],
            output_names=["out"],
            source=_B1_STAGED_SHORT,
            header=_STAGED_HEADER,
            ensure_row_contiguous=True,
        )
        s1_2p = mx.fast.metal_kernel(
            name="b1_staged_sdpa_2p1",
            input_names=["q", "kv", "stage", "kv_new", "pb", "qkv", "meta", "scale"],
            output_names=["partials", "sums", "maxs"],
            source=_B1_STAGED_2P1,
            header=_STAGED_HEADER,
            ensure_row_contiguous=True,
        )
        s2_2p = _get_b1_packed_kernels()[2]
        _b1_staged_kernels = (s1_short, s1_2p, s2_2p)
    return _b1_staged_kernels


def sdpa_b1_staged(
    q, kv_buf, kv_stage, kv_new, pb, qkv_flat, gate_off, sink_off,
    S, end, idx, rot, cap, slot_main, rot_wrap, max_size, st_len,
    *, scale, window, Hq=24, Hkv=8, kv_layer_idx=0, scap=None, meta_arr=None,
):
    """Dedicated B=1 decode attention over packed/stacked main KV + staged KV + current kv_new (zero slice_update).

    kv_buf: (2, Hkv, cap, 128) or (NL, 2, Hkv, cap, 128) bf16.
    kv_stage: (2, Hkv, scap, 128) or (NL, 2, Hkv, scap, 128) bf16, or None (treated as st_len=0).
    kv_new: (2, Hkv, 1, 128) or (2, Hkv, 128) bf16.
    meta_arr: optional prebuilt int32 mx.array of length 11:
      (S, end, idx, rot, cap, kv_layer_idx, slot_main, rot_wrap, max_size, scap, st_len);
      S is still used as an int hint for 1-pass vs 2-pass dispatch.
    """
    assert q.dtype == mx.bfloat16 and kv_buf.dtype == mx.bfloat16 and kv_new.dtype == mx.bfloat16 and qkv_flat.dtype == mx.bfloat16
    gqa = Hq // Hkv
    assert Hq == gqa * Hkv and (
        (kv_buf.ndim == 4 and kv_buf.shape == (2, Hkv, cap, 128) and kv_layer_idx == 0)
        or (kv_buf.ndim == 5 and kv_buf.shape[1:] == (2, Hkv, cap, 128) and 0 <= kv_layer_idx < kv_buf.shape[0])
    )
    if kv_stage is None:
        kv_stage = kv_buf
        scap_i = int(cap)
        st_len_i = 0
    else:
        assert kv_stage.dtype == mx.bfloat16
        scap_i = int(kv_stage.shape[-2]) if scap is None else int(scap)
        st_len_i = int(st_len)
    R = pb.shape[-1]
    sck = float(scale)
    sc = _SCALE_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_CACHE[sck] = mx.array([sck], mx.float32)
    if meta_arr is not None:
        mta = meta_arr
    else:
        mkey = (
            "st", int(S), int(end), int(idx), int(rot), int(cap), int(kv_layer_idx),
            int(slot_main), int(rot_wrap), int(max_size), scap_i, st_len_i,
        )
        mta = _B1_META_CACHE.get(mkey)
        if mta is None:
            if len(_B1_META_CACHE) >= 256:
                _B1_META_CACHE.clear()
            mta = _B1_META_CACHE[mkey] = mx.array(mkey[1:], mx.int32)
    s1_short, s1_2p, s2_2p = _get_b1_staged_kernels()
    if S < _TWO_PASS_N:
        ns_env = os.environ.get("LOOPLING_SDPA1_NS", "")
        ns = int(ns_env) if ns_env and ns_env != "0" else 32
        (out,) = s1_short(
            inputs=[q, kv_buf, kv_stage, kv_new, pb, qkv_flat, mta, sc],
            template=[
                ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                ("R", R), ("WIN", int(window)), ("NS", ns), ("GATE_OFF", int(gate_off)),
                ("SINK_OFF", int(sink_off)),
            ],
            grid=(Hq * 32 * ns, 1, 1),
            threadgroup=(32 * ns, 1, 1),
            output_shapes=[(1, Hq * 128)],
            output_dtypes=[mx.bfloat16],
        )
        return out
    blocks = _blocks(S, gqa)
    ns = gqa
    tgpk = (blocks * gqa + ns - 1) // ns
    ntg = Hkv * tgpk
    partials, sums, maxs = s1_2p(
        inputs=[q, kv_buf, kv_stage, kv_new, pb, qkv_flat, mta, sc],
        template=[
            ("T", mx.bfloat16), ("PBT", pb.dtype), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
            ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns),
            ("SINK_OFF", int(sink_off)),
        ],
        grid=(ntg * 32 * ns, 1, 1),
        threadgroup=(32 * ns, 1, 1),
        output_shapes=[(Hq * blocks * 128,), (Hq * blocks,), (Hq * blocks,)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
    )
    (out,) = s2_2p(
        inputs=[partials, sums, maxs, qkv_flat],
        template=[("T", mx.bfloat16), ("BLOCKS", blocks), ("GATE_OFF", int(gate_off))],
        grid=(Hq * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, Hq * 128)],
        output_dtypes=[mx.bfloat16],
    )
    return out


# =============================================================================================== B=1 Fused K5 MID+REL + K8 SDPA
# Eliminates all 48x/step K5 MID+REL launches, q/pb DRAM round-trips, and moves K7 KV update off the critical path (S < 1024).
_SDPA_MID_HEADER = _HEADER + _MID_HEADER + r"""
struct RowMaskTG {
  const threadgroup bfloat16_t* pb;
  int S, end, idx, rot, R, window;
  METAL_FUNC bool at(int i, thread float& v) const {
    int kp;
    if (rot) {
      int m = i - idx;
      if (m < 0) m += S;
      kp = m + (end - S);
    } else {
      kp = i + (end - S);
    }
    int dist = (end - 1) - kp;
    bool visible = (dist >= 0) && (kp >= 0);
    if (window > 0) visible = visible && (dist < window);
    if (!visible) return false;
    bfloat16_t mv = (dist < R) ? pb[dist] : bfloat16_t(0.0f);
    v = static_cast<float>(mv);
    return mv >= Limits<bfloat16_t>::finite_min;
  }
};

template <int D, int NS>
METAL_FUNC void sdpa1_head_v32_mid(
    const threadgroup bfloat16_t* q_s,
    const device bfloat16_t* K_main,
    const device bfloat16_t* V_main,
    const threadgroup bfloat16_t* knew_s,
    const threadgroup bfloat16_t* vnew_s,
    const device bfloat16_t* gate,
    device bfloat16_t* out,
    const int N,
    const float scale,
    const float sink,
    const RowMaskTG msk,
    const int slot_main,
    const int rot_wrap,
    const int max_size,
    threadgroup float* outputs,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    uint simd_gid,
    uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int NV = BN / NS;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread];
  const int loff = (int)simd_lid * qk_per_thread;
  bfloat4 qb = as_type<bfloat4>(((const threadgroup uint2*)(q_s + loff))[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  for (int v = 0; v < NV; v++) {
    int vg = (int)simd_gid + v * NS;
    for (int i = 0; i < v_per_thread; i++) o[i] = 0;
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    if (vg == 0) {
      max_score = sink;
      sum_exp_score = 1;
    }
    for (int i = vg; i < N; i += BN) {
      float fm = 0;
      if (msk.at(i, fm)) {
        int ds = rot_wrap ? (i - slot_main + (i < slot_main ? max_size : 0)) : (i - slot_main);
        bfloat4 kb, vb;
        if (ds == 0) {
          kb = as_type<bfloat4>(((const threadgroup uint2*)(knew_s + loff))[0]);
          vb = as_type<bfloat4>(((const threadgroup uint2*)(vnew_s + loff))[0]);
        } else {
          kb = as_type<bfloat4>(((const device uint2*)(K_main + (size_t)i * 128 + loff))[0]);
          vb = as_type<bfloat4>(((const device uint2*)(V_main + (size_t)i * 128 + loff))[0]);
        }
        U score = 0;
        score += q[0] * kb.x;
        score += q[1] * kb.y;
        score += q[2] * kb.z;
        score += q[3] * kb.w;
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        o[0] = o[0] * factor + exp_score * vb.x;
        o[1] = o[1] * factor + exp_score * vb.y;
        o[2] = o[2] * factor + exp_score * vb.z;
        o[3] = o[3] * factor + exp_score * vb.w;
      }
    }
    if (simd_lid == 0) {
      max_scores[vg] = max_score;
      sum_exp_scores[vg] = sum_exp_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
      outputs[(i * BD + simd_lid) * BN + vg] = o[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int f = (int)simd_gid; f < BD; f += NS) {
    for (int i = 0; i < v_per_thread; i++) {
      o[i] = simd_sum(outputs[(i * BD + f) * BN + simd_lid] * factor);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    }
    if (simd_lid == 0) {
      int base = f * v_per_thread;
      bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
      bfloat4 ob = bfloat4(
          static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o[0])), static_cast<float>(gb.x))),
          static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o[1])), static_cast<float>(gb.y))),
          static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o[2])), static_cast<float>(gb.z))),
          static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o[3])), static_cast<float>(gb.w))));
      ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
    }
  }
}
"""


def _sdpa_mid_source(full, thr_q, thr_k, thr_r, eps_v):
    scale_conv = flit(np.float32(math.sqrt(5)))
    k_conv_expr = """
        float sw[4];
        for (int k = 0; k < 4; k++) sw[k] = float(k_static[k * 1024 + c]);
        y[i] = mid_conv_d(x0v, s0v, s1v, s2v, sw, dy);""" if full else f"""
        y[i] = mid_conv_s(x0v, s0v, s1v, s2v, float(cwb[i].x), float(cwb[i].y), float(cwb[i].z), float(cwb[i].w), {scale_conv});"""
    v_conv_expr = """
        float sw[4];
        for (int k = 0; k < 4; k++) sw[k] = float(v_static[k * 1024 + c]);
        y[i] = mid_conv_d(x0v, s0v, s1v, s2v, sw, dy);""" if full else f"""
        y[i] = mid_conv_s(x0v, s0v, s1v, s2v, float(cwb[i].x), float(cwb[i].y), float(cwb[i].z), float(cwb[i].w), {scale_conv});"""
    k_cw_load = "" if full else """
      uint4 cw0 = ((const device uint4*)(k_conv + c0 * 4))[0];
      uint4 cw1 = ((const device uint4*)(k_conv + c0 * 4 + 8))[0];
      bfloat4 cwb[4] = {as_type<bfloat4>(cw0.xy), as_type<bfloat4>(cw0.zw), as_type<bfloat4>(cw1.xy), as_type<bfloat4>(cw1.zw)};"""
    v_cw_load = "" if full else """
      uint4 cw0 = ((const device uint4*)(v_conv + c0 * 4))[0];
      uint4 cw1 = ((const device uint4*)(v_conv + c0 * 4 + 8))[0];
      bfloat4 cwb[4] = {as_type<bfloat4>(cw0.xy), as_type<bfloat4>(cw0.zw), as_type<bfloat4>(cw1.xy), as_type<bfloat4>(cw1.zw)};"""
    k_dy_load = """
      bfloat4 dyp = as_type<bfloat4>(((const device uint2*)(qkv + OFF_KW + kvh * 4))[0]);
      float dy[4] = {float(dyp.x), float(dyp.y), float(dyp.z), float(dyp.w)};""" if full else ""
    v_dy_load = """
      bfloat4 dyp = as_type<bfloat4>(((const device uint2*)(qkv + OFF_VW + kvh * 4))[0]);
      float dy[4] = {float(dyp.x), float(dyp.y), float(dyp.z), float(dyp.w)};""" if full else ""
    return f"""
    constexpr int D = 128;
    constexpr int P_SG = (NS >= 16) ? 8 : 4;
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    threadgroup uint2 q_raw[32];
    threadgroup uint2 kn_raw[32];
    threadgroup uint2 vn_raw[32];
    threadgroup bfloat16_t pb_s[1024];
    threadgroup bfloat16_t* q_s = (threadgroup bfloat16_t*)q_raw;
    threadgroup bfloat16_t* knew_s = (threadgroup bfloat16_t*)kn_raw;
    threadgroup bfloat16_t* vnew_s = (threadgroup bfloat16_t*)vn_raw;

    const int h = (int)threadgroup_position_in_grid.x;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int kvh = h / GQA;
    const bool kv_owner = (h == kvh * GQA);
    const int S = meta[0];
    {"const float tau_v = tau[0];" if full else ""}

    if (simd_gid == 0u) {{
      const uint qoff = (uint)h * 128u + simd_lid * 4u;
      bfloat4 xb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_Q + qoff))[0]);
      float s[4] = {{
          float(xb.x) * float(q_scale[qoff + 0u]),
          float(xb.y) * float(q_scale[qoff + 1u]),
          float(xb.z) * float(q_scale[qoff + 2u]),
          float(xb.w) * float(q_scale[qoff + 3u]),
      }};
      glue_l2_clip_lane128(s, {flit(thr_q)});
      bfloat16_t qo[4];
      for (int i = 0; i < 4; i++) {{
        bfloat16_t o = bfloat16_t(s[i]);
        {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
        qo[i] = o;
      }}
      bfloat4 ob = bfloat4(qo[0], qo[1], qo[2], qo[3]);
      q_raw[simd_lid] = as_type<uint2>(ob);
    }} else if (simd_gid == 1u) {{
      const int c0 = kvh * 128 + (int)simd_lid * 4;
      bfloat4 xb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_K + c0))[0]);
      float4 x0f = float4(float(xb.x), float(xb.y), float(xb.z), float(xb.w));
      float4 s0f = ((const device float4*)(cs0 + c0))[0];
      float4 s1f = ((const device float4*)(cs0 + 1024 + c0))[0];
      float4 s2f = ((const device float4*)(cs0 + 2048 + c0))[0];
      if (kv_owner) {{
        ((device float4*)(w0 + c0))[0] = s1f;
        ((device float4*)(w0 + 1024 + c0))[0] = s2f;
        ((device float4*)(w0 + 2048 + c0))[0] = x0f;
      }}{k_dy_load}{k_cw_load}
      float x0a[4] = {{x0f.x, x0f.y, x0f.z, x0f.w}};
      float s0a[4] = {{s0f.x, s0f.y, s0f.z, s0f.w}};
      float s1a[4] = {{s1f.x, s1f.y, s1f.z, s1f.w}};
      float s2a[4] = {{s2f.x, s2f.y, s2f.z, s2f.w}};
      float y[4];
      for (int i = 0; i < 4; i++) {{
        int c = c0 + i;
        float x0v = x0a[i], s0v = s0a[i], s1v = s1a[i], s2v = s2a[i];
        {k_conv_expr}
      }}
      glue_l2_clip_lane128(y, {flit(thr_k)});
      bfloat4 ob = bfloat4(bfloat16_t(y[0]), bfloat16_t(y[1]), bfloat16_t(y[2]), bfloat16_t(y[3]));
      uint2 pk = as_type<uint2>(ob);
      kn_raw[simd_lid] = pk;
      if (kv_owner) ((device uint2*)(kv_out + c0))[0] = pk;
    }} else if (simd_gid == 2u) {{
      const int c0 = kvh * 128 + (int)simd_lid * 4;
      bfloat4 xb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_V + c0))[0]);
      float4 x0f = float4(float(xb.x), float(xb.y), float(xb.z), float(xb.w));
      float4 s0f = ((const device float4*)(cs1 + c0))[0];
      float4 s1f = ((const device float4*)(cs1 + 1024 + c0))[0];
      float4 s2f = ((const device float4*)(cs1 + 2048 + c0))[0];
      if (kv_owner) {{
        ((device float4*)(w1 + c0))[0] = s1f;
        ((device float4*)(w1 + 1024 + c0))[0] = s2f;
        ((device float4*)(w1 + 2048 + c0))[0] = x0f;
      }}{v_dy_load}{v_cw_load}
      float x0a[4] = {{x0f.x, x0f.y, x0f.z, x0f.w}};
      float s0a[4] = {{s0f.x, s0f.y, s0f.z, s0f.w}};
      float s1a[4] = {{s1f.x, s1f.y, s1f.z, s1f.w}};
      float s2a[4] = {{s2f.x, s2f.y, s2f.z, s2f.w}};
      float y[4];
      for (int i = 0; i < 4; i++) {{
        int c = c0 + i;
        float x0v = x0a[i], s0v = s0a[i], s1v = s1a[i], s2v = s2a[i];
        {v_conv_expr}
      }}
      ((threadgroup float4*)(outputs + simd_lid * 4u))[0] = float4(y[0], y[1], y[2], y[3]);
      float inv_v = glue_rms_inv_sg<128, true>(outputs, {flit(eps_v)}, simd_lid);
      bfloat4 ob = bfloat4(
          glue_rms_apply_nw<bfloat16_t>(y[0], inv_v),
          glue_rms_apply_nw<bfloat16_t>(y[1], inv_v),
          glue_rms_apply_nw<bfloat16_t>(y[2], inv_v),
          glue_rms_apply_nw<bfloat16_t>(y[3], inv_v));
      uint2 pk = as_type<uint2>(ob);
      vn_raw[simd_lid] = pk;
      if (kv_owner) ((device uint2*)(kv_out + 1024 + c0))[0] = pk;
    }} else if (simd_gid < uint(3 + P_SG)) {{
      const int p = (int)simd_gid - 3;
      const int n_pb = min(S, (WIN > 0) ? min(R, WIN) : R);
      const int n_pb16 = min(R, (n_pb + 15) & ~15);
      if (p * 16 < n_pb16) {{
        threadgroup bfloat16_t* rel_tile = (threadgroup bfloat16_t*)(outputs + 256u + (uint)p * 64u);
        ((threadgroup uint2*)rel_tile)[simd_lid] = uint2(0u, 0u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lid == 0u) {{
          float s[16];
          for (int i = 0; i < 16; i++) s[i] = float(qkv[OFF_R + h * 16 + i]) * float(rel_scale[h * 16 + i]);
          glue_l2_clip_row16(s, {flit(thr_r)});
          for (int i = 0; i < 16; i++) {{
            bfloat16_t o = bfloat16_t(s[i]);
            {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
            rel_tile[i] = o;
          }}
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_bfloat8x8 ma0, ma1;
        simdgroup_load(ma0, rel_tile + 0, 16);
        simdgroup_load(ma1, rel_tile + 8, 16);
        threadgroup float* st = outputs + 1024u + (uint)p * 128u;
        for (int n0 = p * 16; n0 < n_pb16; n0 += P_SG * 16) {{
          const device bfloat16_t* Bm = rw + n0;
          simdgroup_bfloat8x8 mb00, mb10, mb01, mb11;
          simdgroup_load(mb00, Bm + 0 * R + 0, R);
          simdgroup_load(mb10, Bm + 8 * R + 0, R);
          simdgroup_load(mb01, Bm + 0 * R + 8, R);
          simdgroup_load(mb11, Bm + 8 * R + 8, R);
          simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
          simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
          simdgroup_multiply_accumulate(acc0, ma0, mb00, acc0);
          simdgroup_multiply_accumulate(acc0, ma1, mb10, acc0);
          simdgroup_multiply_accumulate(acc1, ma0, mb01, acc1);
          simdgroup_multiply_accumulate(acc1, ma1, mb11, acc1);
          simdgroup_store(acc0, st, 8);
          simdgroup_store(acc1, st + 64u, 8);
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_lid < 16u) {{
            int col = n0 + (int)simd_lid;
            if (col < 1024) pb_s[col] = bfloat16_t(st[(simd_lid < 8u) ? simd_lid : (64u + simd_lid - 8u)]);
          }}
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }}
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int cap = meta[4];
    const int lidx = meta[5];
    const int slot_main = meta[6];
    const int rot_wrap = meta[7];
    const int max_size = meta[8];
    const device bfloat16_t* K = kv + ((size_t)lidx * 2 * HKV + kvh) * (size_t)cap * D;
    const device bfloat16_t* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * (size_t)cap * D;
    RowMaskTG msk;
    msk.pb = pb_s;
    msk.S = S; msk.end = meta[1]; msk.idx = meta[2]; msk.rot = meta[3];
    msk.R = R; msk.window = WIN;
    float sink = float(qkv[SINK_OFF + h]);
    sdpa1_head_v32_mid<D, NS>(
        q_s, K, V, knew_s, vnew_s, qkv + GATE_OFF + (size_t)h * D, out + (size_t)h * D,
        S, scale[0], sink, msk, slot_main, rot_wrap, max_size,
        outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""


_sdpa_mid_kernels: dict = {}
_SDPA_MID_NS = int(os.environ.get("LOOPLING_SDPA_MID_NS", "32"))
if _SDPA_MID_NS not in (8, 16, 32):
    _SDPA_MID_NS = 32


def _get_sdpa_mid_layer_meta(layer):
    is_dict = isinstance(layer, dict)
    a = layer["self_attn"] if is_dict else layer.self_attn
    a_dict = isinstance(a, dict)
    ql2 = a["q_l2_clip"] if a_dict else a.q_l2_clip
    kl2 = a["k_l2_clip"] if a_dict else a.k_l2_clip
    rl2 = a["rel_l2_clip"] if a_dict else a.rel_l2_clip
    q_sc = ql2["scale"] if isinstance(ql2, dict) else ql2.scale
    r_sc = rl2["scale"] if isinstance(rl2, dict) else rl2.scale
    ksconv = a["k_sconv"] if a_dict else a.k_sconv
    vsconv = a["v_sconv"] if a_dict else a.v_sconv
    full = a.sliding_window == 0
    if full:
        cw0 = ksconv["static_weight"] if isinstance(ksconv, dict) else ksconv.static_weight
        cw1 = vsconv["static_weight"] if isinstance(vsconv, dict) else vsconv.static_weight
    else:
        cw0 = ksconv["conv1d"]["weight"] if isinstance(ksconv, dict) else ksconv.conv1d.weight
        cw1 = vsconv["conv1d"]["weight"] if isinstance(vsconv, dict) else vsconv.conv1d.weight
    return (full, q_sc, r_sc, cw0, cw1, ql2.threshold, kl2.threshold, rl2.threshold, a.eps, a.num_heads, a.num_key_value_heads)


def sdpa_mid_b1(
    layer, qkv_flat, qkv_lay, kv_buf, cs0, cs1, rel_w,
    S_now, end_now, idx_now, rot, cap, slot_main, rot_wrap, max_size,
    *, scale, window, tau=None, kv_layer_idx=0, meta_arr=None,
):
    """Fused B=1 K5 MID+REL + K8 SDPA (S_now < 1024) directly over K4's flat buffer + pre-update packed KV.
    Returns (attn_out (1, 3072) bf16, kv_new (2, 8, 1, 128) bf16, w0 (1, 3, 1024) f32, w1 (1, 3, 1024) f32)."""
    assert S_now < _TWO_PASS_N
    full, q_sc, r_sc, cw0, cw1, thr_q, thr_k, thr_r, eps_v, Hq, Hkv = _get_sdpa_mid_layer_meta(layer)
    R = int(rel_w.shape[1])
    sck = float(scale)
    sc = _SCALE_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_CACHE[sck] = mx.array([sck], mx.float32)
    if meta_arr is not None:
        mta = meta_arr
    else:
        mkey = ("mid", int(S_now), int(end_now), int(idx_now), int(rot), int(cap), int(kv_layer_idx), int(slot_main), int(rot_wrap), int(max_size))
        mta = _B1_META_CACHE.get(mkey)
        if mta is None:
            if len(_B1_META_CACHE) >= 256:
                _B1_META_CACHE.clear()
            mta = _B1_META_CACHE[mkey] = mx.array(mkey[1:], mx.int32)
    ins = [qkv_flat, kv_buf, cs0, cs1, q_sc, r_sc, rel_w, mta, sc]
    if full:
        ins += [tau.reshape(-1), cw0, cw1]
    else:
        ins += [cw0, cw1]
    kkey = (full, thr_q, thr_k, thr_r, eps_v)
    kern = _sdpa_mid_kernels.get(kkey)
    if kern is None:
        in_names = ["qkv", "kv", "cs0", "cs1", "q_scale", "rel_scale", "rw", "meta", "scale"]
        in_names += ["tau", "k_static", "v_static"] if full else ["k_conv", "v_conv"]
        kern = _sdpa_mid_kernels[kkey] = mx.fast.metal_kernel(
            name="b1_sdpa_mid_%s_%d" % ("F" if full else "S", abs(hash(kkey)) % 10**9),
            input_names=in_names,
            output_names=["out", "kv_out", "w0", "w1"],
            source=_sdpa_mid_source(full, thr_q, thr_k, thr_r, eps_v),
            header=_SDPA_MID_HEADER,
            ensure_row_contiguous=True,
        )
    ns = _SDPA_MID_NS
    tkey = ("b1", full, Hq, Hkv, R, int(window), ns, tuple(qkv_lay))
    tmpl = _sdpa_mid_tmpl_cache.get(tkey)
    if tmpl is None:
        if len(_sdpa_mid_tmpl_cache) >= 512:
            _sdpa_mid_tmpl_cache.clear()
        gqa = Hq // Hkv
        offs = {item[0]: item[1] for item in qkv_lay}
        tmpl = [
            ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa), ("R", R), ("WIN", int(window)), ("NS", ns),
            ("OFF_Q", offs["q_proj"]), ("OFF_R", offs["r_proj"]),
            ("OFF_K", offs["k_proj"]), ("OFF_V", offs["v_proj"]),
            ("GATE_OFF", offs["gate"]), ("SINK_OFF", offs["sinks"]),
        ]
        if full:
            tmpl += [("OFF_KW", offs["k_wproj"]), ("OFF_VW", offs["v_wproj"])]
        _sdpa_mid_tmpl_cache[tkey] = tmpl
    return kern(
        inputs=ins,
        template=tmpl,
        grid=(Hq * 32 * ns, 1, 1),
        threadgroup=(32 * ns, 1, 1),
        output_shapes=[(1, Hq * 128), (2, Hkv, 1, 128), (1, 3, Hkv * 128), (1, 3, Hkv * 128)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
    )


# =============================================================================================== Batched (B=1..8) Fused K5 MID+REL + K8 SDPA (S < 1024)
_SDPA_MID_BATCH_HEADER = _SDPA_MID_HEADER + r"""
template <int D, int NS, int OPASS = 1>
METAL_FUNC void sdpa1_head_v32_mid_staged(
    const threadgroup bfloat16_t* q_s,
    const device bfloat16_t* K_main,
    const device bfloat16_t* V_main,
    const device bfloat16_t* Kst,
    const device bfloat16_t* Vst,
    const threadgroup bfloat16_t* knew_s,
    const threadgroup bfloat16_t* vnew_s,
    const device bfloat16_t* gate,
    device bfloat16_t* out,
    const int N,
    const float scale,
    const float sink,
    const RowMaskTG msk,
    const int slot_main,
    const int rot_wrap,
    const int max_size,
    const int st_len,
    threadgroup float* outputs,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    uint simd_gid,
    uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int NV = BN / NS;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o_reg[(OPASS == 1) ? 1 : NV][v_per_thread];
  const int loff = (int)simd_lid * qk_per_thread;
  bfloat4 qb = as_type<bfloat4>(((const threadgroup uint2*)(q_s + loff))[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  for (int v = 0; v < NV; v++) {
    int vg = (int)simd_gid + v * NS;
    U o0 = 0, o1 = 0, o2 = 0, o3 = 0;
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    if (vg == 0) {
      max_score = sink;
      sum_exp_score = 1;
    }
    int i = vg;
    if (!rot_wrap) {
      const int n_fast = min(N, slot_main);
      for (; i < n_fast; i += BN) {
        float fm = 0;
        if (msk.at(i, fm)) {
          bfloat4 kb = as_type<bfloat4>(((const device uint2*)(K_main + (size_t)i * 128 + loff))[0]);
          U score = 0;
          score += q[0] * kb.x;
          score += q[1] * kb.y;
          score += q[2] * kb.z;
          score += q[3] * kb.w;
          score = simd_sum(score);
          score += static_cast<U>(fm);
          U new_max = max(max_score, score);
          U factor = fast::exp(max_score - new_max);
          U exp_score = fast::exp(score - new_max);
          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + exp_score;
          bfloat4 vb = as_type<bfloat4>(((const device uint2*)(V_main + (size_t)i * 128 + loff))[0]);
          o0 = o0 * factor + exp_score * vb.x;
          o1 = o1 * factor + exp_score * vb.y;
          o2 = o2 * factor + exp_score * vb.z;
          o3 = o3 * factor + exp_score * vb.w;
        }
      }
    }
    for (; i < N; i += BN) {
      float fm = 0;
      if (msk.at(i, fm)) {
        int ds = rot_wrap ? (i - slot_main + (i < slot_main ? max_size : 0)) : (i - slot_main);
        bfloat4 kb, vb;
        if (ds == st_len) {
          kb = as_type<bfloat4>(((const threadgroup uint2*)(knew_s + loff))[0]);
          vb = as_type<bfloat4>(((const threadgroup uint2*)(vnew_s + loff))[0]);
        } else if (ds >= 0 && ds < st_len) {
          kb = as_type<bfloat4>(((const device uint2*)(Kst + (size_t)ds * 128 + loff))[0]);
          vb = as_type<bfloat4>(((const device uint2*)(Vst + (size_t)ds * 128 + loff))[0]);
        } else {
          kb = as_type<bfloat4>(((const device uint2*)(K_main + (size_t)i * 128 + loff))[0]);
          vb = as_type<bfloat4>(((const device uint2*)(V_main + (size_t)i * 128 + loff))[0]);
        }
        U score = 0;
        score += q[0] * kb.x;
        score += q[1] * kb.y;
        score += q[2] * kb.z;
        score += q[3] * kb.w;
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        o0 = o0 * factor + exp_score * vb.x;
        o1 = o1 * factor + exp_score * vb.y;
        o2 = o2 * factor + exp_score * vb.z;
        o3 = o3 * factor + exp_score * vb.w;
      }
    }
    if (simd_lid == 0) {
      max_scores[vg] = max_score;
      sum_exp_scores[vg] = sum_exp_score;
    }
    if constexpr (OPASS == 1) {
      outputs[(0 * BD + simd_lid) * BN + vg] = o0;
      outputs[(1 * BD + simd_lid) * BN + vg] = o1;
      outputs[(2 * BD + simd_lid) * BN + vg] = o2;
      outputs[(3 * BD + simd_lid) * BN + vg] = o3;
    } else {
      o_reg[v][0] = o0;
      o_reg[v][1] = o1;
      o_reg[v][2] = o2;
      o_reg[v][3] = o3;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  if constexpr (OPASS == 1) {
    for (int f = (int)simd_gid; f < BD; f += NS) {
      U r0 = simd_sum(outputs[(0 * BD + f) * BN + simd_lid] * factor);
      U r1 = simd_sum(outputs[(1 * BD + f) * BN + simd_lid] * factor);
      U r2 = simd_sum(outputs[(2 * BD + f) * BN + simd_lid] * factor);
      U r3 = simd_sum(outputs[(3 * BD + f) * BN + simd_lid] * factor);
      r0 = sum_exp_score == 0 ? r0 : (r0 / sum_exp_score);
      r1 = sum_exp_score == 0 ? r1 : (r1 / sum_exp_score);
      r2 = sum_exp_score == 0 ? r2 : (r2 / sum_exp_score);
      r3 = sum_exp_score == 0 ? r3 : (r3 / sum_exp_score);
      if (simd_lid == 0) {
        int base = f * v_per_thread;
        bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
        bfloat4 ob = bfloat4(
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(r0)), static_cast<float>(gb.x))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(r1)), static_cast<float>(gb.y))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(r2)), static_cast<float>(gb.z))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(r3)), static_cast<float>(gb.w))));
        ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
      }
    }
  } else {
    for (int i = 0; i < v_per_thread; i++) {
      for (int v = 0; v < NV; v++) {
        int vg = (int)simd_gid + v * NS;
        outputs[simd_lid * BN + vg] = o_reg[v][i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int fi = 0; fi < NV; fi++) {
        int f = (int)simd_gid + fi * NS;
        U r = simd_sum(outputs[f * BN + simd_lid] * factor);
        o_reg[fi][i] = sum_exp_score == 0 ? r : (r / sum_exp_score);
      }
      if (i + 1 < v_per_thread) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    if (simd_lid == 0) {
      for (int fi = 0; fi < NV; fi++) {
        int base = ((int)simd_gid + fi * NS) * v_per_thread;
        bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
        bfloat4 ob = bfloat4(
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o_reg[fi][0])), static_cast<float>(gb.x))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o_reg[fi][1])), static_cast<float>(gb.y))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o_reg[fi][2])), static_cast<float>(gb.z))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o_reg[fi][3])), static_cast<float>(gb.w))));
        ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
      }
    }
  }
}
"""


def _sdpa_mid_batch_source(full, thr_q, thr_k, thr_r, eps_v, scale_lit, kv_select):
    scale_conv = flit(np.float32(math.sqrt(5)))
    k_conv_expr = """
        float sw[4];
        for (int k = 0; k < 4; k++) sw[k] = float(cw01[k * 1024 + c]);
        y[i] = mid_conv_d(x0v, s0v, s1v, s2v, sw, dy);""" if full else f"""
        y[i] = mid_conv_s(x0v, s0v, s1v, s2v, float(cwb[i].x), float(cwb[i].y), float(cwb[i].z), float(cwb[i].w), {scale_conv});"""
    v_conv_expr = """
        float sw[4];
        for (int k = 0; k < 4; k++) sw[k] = float(cw01[4096 + k * 1024 + c]);
        y[i] = mid_conv_d(x0v, s0v, s1v, s2v, sw, dy);""" if full else f"""
        y[i] = mid_conv_s(x0v, s0v, s1v, s2v, float(cwb[i].x), float(cwb[i].y), float(cwb[i].z), float(cwb[i].w), {scale_conv});"""
    k_cw_load = "" if full else """
      uint4 cw0 = ((const device uint4*)(cw01 + c0 * 4))[0];
      uint4 cw1 = ((const device uint4*)(cw01 + c0 * 4 + 8))[0];
      bfloat4 cwb[4] = {as_type<bfloat4>(cw0.xy), as_type<bfloat4>(cw0.zw), as_type<bfloat4>(cw1.xy), as_type<bfloat4>(cw1.zw)};"""
    v_cw_load = "" if full else """
      uint4 cw0 = ((const device uint4*)(cw01 + 4096 + c0 * 4))[0];
      uint4 cw1 = ((const device uint4*)(cw01 + 4096 + c0 * 4 + 8))[0];
      bfloat4 cwb[4] = {as_type<bfloat4>(cw0.xy), as_type<bfloat4>(cw0.zw), as_type<bfloat4>(cw1.xy), as_type<bfloat4>(cw1.zw)};"""
    k_dy_load = """
      bfloat4 dyp = as_type<bfloat4>(((const device uint2*)(qkv + OFF_KW + (size_t)gb * STR_KW + kvh * 4))[0]);
      float dy[4] = {float(dyp.x), float(dyp.y), float(dyp.z), float(dyp.w)};""" if full else ""
    v_dy_load = """
      bfloat4 dyp = as_type<bfloat4>(((const device uint2*)(qkv + OFF_VW + (size_t)gb * STR_VW + kvh * 4))[0]);
      float dy[4] = {float(dyp.x), float(dyp.y), float(dyp.z), float(dyp.w)};""" if full else ""
    return f"""
    constexpr int D = 128;
    constexpr int P_SG = (OPASS == 1 && NS >= 8) ? 5 : 4;
    threadgroup float outputs[(OPASS == 1 ? 4 : 1) * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    threadgroup uint2 q_raw[32];
    threadgroup uint2 kn_raw[32];
    threadgroup uint2 vn_raw[32];
    threadgroup bfloat16_t pb_s[1024];
    threadgroup bfloat16_t* q_s = (threadgroup bfloat16_t*)q_raw;
    threadgroup bfloat16_t* knew_s = (threadgroup bfloat16_t*)kn_raw;
    threadgroup bfloat16_t* vnew_s = (threadgroup bfloat16_t*)vn_raw;

    const int t = (int)threadgroup_position_in_grid.x;
    const int b = t / HQ;
    const int gb = B_OFF + b;
    const int h = t % HQ;
    const int kvh = h / GQA;
    const bool kv_owner = (h == kvh * GQA);
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int S = meta[gb * MW + 0];
    {"const float tau_v = tau[gb];" if full else ""}

    if (simd_gid == 0u) {{
      const uint qoff_w = (uint)h * 128u + simd_lid * 4u;
      const size_t qoff = OFF_Q + (size_t)gb * STR_Q + qoff_w;
      bfloat4 xb = as_type<bfloat4>(((const device uint2*)(qkv + qoff))[0]);
      bfloat4 qsb = as_type<bfloat4>(((const device uint2*)(qr_scale + qoff_w))[0]);
      float s[4] = {{
          float(xb.x) * float(qsb.x),
          float(xb.y) * float(qsb.y),
          float(xb.z) * float(qsb.z),
          float(xb.w) * float(qsb.w),
      }};
      glue_l2_clip_lane128(s, {flit(thr_q)});
      bfloat16_t qo[4];
      for (int i = 0; i < 4; i++) {{
        bfloat16_t o = bfloat16_t(s[i]);
        {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
        qo[i] = o;
      }}
      q_raw[simd_lid] = as_type<uint2>(bfloat4(qo[0], qo[1], qo[2], qo[3]));
    }} else if (simd_gid == 1u) {{
      const int c0 = kvh * 128 + (int)simd_lid * 4;
      bfloat4 xb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_K + (size_t)gb * STR_K + c0))[0]);
      float4 x0f = float4(float(xb.x), float(xb.y), float(xb.z), float(xb.w));
      const device float* cs0b = cs0 + (size_t)gb * 3072u;
      float4 s0f = ((const device float4*)(cs0b + c0))[0];
      float4 s1f = ((const device float4*)(cs0b + 1024 + c0))[0];
      float4 s2f = ((const device float4*)(cs0b + 2048 + c0))[0];
      if (kv_owner) {{
        device float* w0b = w0 + (size_t)b * 3072u;
        ((device float4*)(w0b + c0))[0] = s1f;
        ((device float4*)(w0b + 1024 + c0))[0] = s2f;
        ((device float4*)(w0b + 2048 + c0))[0] = x0f;
      }}{k_dy_load}{k_cw_load}
      float x0a[4] = {{x0f.x, x0f.y, x0f.z, x0f.w}};
      float s0a[4] = {{s0f.x, s0f.y, s0f.z, s0f.w}};
      float s1a[4] = {{s1f.x, s1f.y, s1f.z, s1f.w}};
      float s2a[4] = {{s2f.x, s2f.y, s2f.z, s2f.w}};
      float y[4];
      for (int i = 0; i < 4; i++) {{
        int c = c0 + i;
        float x0v = x0a[i], s0v = s0a[i], s1v = s1a[i], s2v = s2a[i];
        {k_conv_expr}
      }}
      glue_l2_clip_lane128(y, {flit(thr_k)});
      uint2 pk = as_type<uint2>(bfloat4(bfloat16_t(y[0]), bfloat16_t(y[1]), bfloat16_t(y[2]), bfloat16_t(y[3])));
      kn_raw[simd_lid] = pk;
      if (kv_owner) ((device uint2*)(kv_out + (size_t)b * 2048u + c0))[0] = pk;
    }} else if (simd_gid == 2u) {{
      const int c0 = kvh * 128 + (int)simd_lid * 4;
      bfloat4 xb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_V + (size_t)gb * STR_V + c0))[0]);
      float4 x0f = float4(float(xb.x), float(xb.y), float(xb.z), float(xb.w));
      const device float* cs1b = cs1 + (size_t)gb * 3072u;
      float4 s0f = ((const device float4*)(cs1b + c0))[0];
      float4 s1f = ((const device float4*)(cs1b + 1024 + c0))[0];
      float4 s2f = ((const device float4*)(cs1b + 2048 + c0))[0];
      if (kv_owner) {{
        device float* w1b = w1 + (size_t)b * 3072u;
        ((device float4*)(w1b + c0))[0] = s1f;
        ((device float4*)(w1b + 1024 + c0))[0] = s2f;
        ((device float4*)(w1b + 2048 + c0))[0] = x0f;
      }}{v_dy_load}{v_cw_load}
      float x0a[4] = {{x0f.x, x0f.y, x0f.z, x0f.w}};
      float s0a[4] = {{s0f.x, s0f.y, s0f.z, s0f.w}};
      float s1a[4] = {{s1f.x, s1f.y, s1f.z, s1f.w}};
      float s2a[4] = {{s2f.x, s2f.y, s2f.z, s2f.w}};
      float y[4];
      for (int i = 0; i < 4; i++) {{
        int c = c0 + i;
        float x0v = x0a[i], s0v = s0a[i], s1v = s1a[i], s2v = s2a[i];
        {v_conv_expr}
      }}
      ((threadgroup float4*)(outputs + simd_lid * 4u))[0] = float4(y[0], y[1], y[2], y[3]);
      float inv_v = glue_rms_inv_sg<128, true>(outputs, {flit(eps_v)}, simd_lid);
      uint2 pk = as_type<uint2>(bfloat4(
          glue_rms_apply_nw<bfloat16_t>(y[0], inv_v),
          glue_rms_apply_nw<bfloat16_t>(y[1], inv_v),
          glue_rms_apply_nw<bfloat16_t>(y[2], inv_v),
          glue_rms_apply_nw<bfloat16_t>(y[3], inv_v)));
      vn_raw[simd_lid] = pk;
      if (kv_owner) ((device uint2*)(kv_out + (size_t)b * 2048u + 1024u + c0))[0] = pk;
    }} else if (simd_gid < uint(3 + P_SG)) {{
      const int p = (int)simd_gid - 3;
      const int n_pb = min(S, (WIN > 0) ? min(R, WIN) : R);
      const int n_pb16 = min(R, (n_pb + 15) & ~15);
      if (p * 16 < n_pb16) {{
        threadgroup bfloat16_t* rel_tile = (threadgroup bfloat16_t*)(outputs + 128u + (uint)p * 64u);
        ((threadgroup uint2*)rel_tile)[simd_lid] = uint2(0u, 0u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lid == 0u) {{
          float s[16];
          const device bfloat16_t* rp = qkv + OFF_R + (size_t)gb * STR_R + (size_t)h * 16u;
          const device bfloat16_t* rsp = qr_scale + HQ * 128 + h * 16;
          for (int i = 0; i < 16; i++) s[i] = float(rp[i]) * float(rsp[i]);
          glue_l2_clip_row16(s, {flit(thr_r)});
          for (int i = 0; i < 16; i++) {{
            bfloat16_t o = bfloat16_t(s[i]);
            {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
            rel_tile[i] = o;
          }}
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_bfloat8x8 ma0, ma1;
        simdgroup_load(ma0, rel_tile + 0, 16);
        simdgroup_load(ma1, rel_tile + 8, 16);
        threadgroup float* st = outputs + ((P_SG > 4) ? 512u : 384u) + (uint)p * 128u;
        for (int n0 = p * 16; n0 < n_pb16; n0 += P_SG * 16) {{
          const device bfloat16_t* Bm = rw + n0;
          simdgroup_bfloat8x8 mb00, mb10, mb01, mb11;
          simdgroup_load(mb00, Bm + 0 * R + 0, R);
          simdgroup_load(mb10, Bm + 8 * R + 0, R);
          simdgroup_load(mb01, Bm + 0 * R + 8, R);
          simdgroup_load(mb11, Bm + 8 * R + 8, R);
          simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
          simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
          simdgroup_multiply_accumulate(acc0, ma0, mb00, acc0);
          simdgroup_multiply_accumulate(acc0, ma1, mb10, acc0);
          simdgroup_multiply_accumulate(acc1, ma0, mb01, acc1);
          simdgroup_multiply_accumulate(acc1, ma1, mb11, acc1);
          simdgroup_store(acc0, st, 8);
          simdgroup_store(acc1, st + 64u, 8);
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_lid < 16u) {{
            int col = n0 + (int)simd_lid;
            if (col < 1024) pb_s[col] = bfloat16_t(st[(simd_lid < 8u) ? simd_lid : (64u + simd_lid - 8u)]);
          }}
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }}
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device bfloat16_t* K;
    const device bfloat16_t* V;
    {kv_select}
    const int kcap = meta[gb * MW + 4];
    const int vcap = meta[gb * MW + 5];
    const int slot_main = meta[gb * MW + 6];
    const int rot_wrap = meta[gb * MW + 7];
    const int max_size = meta[gb * MW + 8];
    const int kv_lidx = st_info[0];
    const int scap = st_info[1];
    const int st_len = st_info[2];
    const device bfloat16_t* Kst = stage + ((((size_t)kv_lidx * ST_NROWS + gb) * 2 + 0) * HKV + kvh) * (size_t)scap * D;
    const device bfloat16_t* Vst = stage + ((((size_t)kv_lidx * ST_NROWS + gb) * 2 + 1) * HKV + kvh) * (size_t)scap * D;
    RowMaskTG msk;
    msk.pb = pb_s;
    msk.S = S; msk.end = meta[gb * MW + 1]; msk.idx = meta[gb * MW + 2]; msk.rot = meta[gb * MW + 3];
    msk.R = R; msk.window = WIN;
    float sink = float(static_cast<bfloat16_t>(qkv[SINK_OFF + (size_t)gb * SINK_STRIDE + h]));
    sdpa1_head_v32_mid_staged<D, NS, OPASS>(
        q_s, K + (size_t)kvh * kcap * D, V + (size_t)kvh * vcap * D, Kst, Vst,
        knew_s, vnew_s, qkv + GATE_OFF + (size_t)gb * GATE_STRIDE + (size_t)h * D,
        out + (size_t)t * D, S, {scale_lit}, sink, msk, slot_main, rot_wrap, max_size, st_len,
        outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""


_sdpa_mid_batch_kernels: dict = {}
_sdpa_mid_batch_layer_cache: dict = {}
_sdpa_mid_tmpl_cache: dict = {}
_SDPA_MID_LIDX_CACHE: dict = {}
_SDPA_MID_BATCH_NS_ENV = int(os.environ.get("LOOPLING_SDPA_MID_BATCH_NS", "0") or "0")
_SDPA_MID_CW_NS_ENV = int(os.environ.get("LOOPLING_SDPA_MID_CW_NS", "0") or "0")
_SDPA_MID_CW_MAX_W = int(os.environ.get("LOOPLING_SDPA_MID_CW_MAX_W", "4") or "4")


def _get_sdpa_mid_batch_layer_meta(layer):
    full, q_sc, r_sc, cw0, cw1, thr_q, thr_k, thr_r, eps_v, Hq, Hkv = _get_sdpa_mid_layer_meta(layer)
    key = id(layer)
    refs = (q_sc, r_sc, cw0, cw1)
    ent = _sdpa_mid_batch_layer_cache.get(key)
    if ent is not None and all(a is b for a, b in zip(ent[0], refs)):
        return ent[1]
    if len(_sdpa_mid_batch_layer_cache) >= 256:
        _sdpa_mid_batch_layer_cache.clear()
    qr_scale = mx.contiguous(mx.concatenate([q_sc.reshape(-1), r_sc.reshape(-1)]))
    cw01 = mx.contiguous(mx.concatenate([cw0.reshape(-1), cw1.reshape(-1)]))
    lparams = mx.array([float(thr_q), float(thr_k), float(thr_r), float(eps_v)], mx.float32)
    mx.eval(qr_scale, cw01, lparams)
    val = (full, qr_scale, cw01, lparams, thr_q, thr_k, thr_r, eps_v, Hq, Hkv)
    _sdpa_mid_batch_layer_cache[key] = (refs, val)
    if not isinstance(layer, dict) or hasattr(layer, "__weakref__"):
        try:
            weakref.finalize(layer, _sdpa_mid_batch_layer_cache.pop, key, None)
        except TypeError:
            pass
    return val


def sdpa_mid_rows_staged(
    layer, qkv_flat, qkv_lay, ks_main, vs_main, stage, cs0, cs1, rel_w,
    *, scale, window, kv_layer_idx, st_len, scap, meta_arr, Ss, st_info_arr=None, tau=None,
):
    """Batched (B=1..32) fused K5 MID+REL + K8 SDPA over main KV + stage buffer (S < 1024).
    Returns (attn_out (B, Hq*128) bf16, kv_new (B, 2, Hkv, 128) bf16, w0 (B, 3, Hkv*128) f32, w1 (B, 3, Hkv*128) f32)."""
    B = len(ks_main)
    assert 1 <= B <= 32 and max(Ss) < _TWO_PASS_N
    full, qr_scale, cw01, _lparams, thr_q, thr_k, thr_r, eps_v, Hq, Hkv = _get_sdpa_mid_batch_layer_meta(layer)
    R = int(rel_w.shape[1])
    st_nrows = int(stage.shape[1]) if stage.ndim >= 2 else B
    if st_info_arr is not None:
        st_info = st_info_arr
    else:
        st_key = (int(kv_layer_idx), int(scap), int(st_len))
        st_info = _ST_INFO_CACHE.get(st_key)
        if st_info is None:
            if len(_ST_INFO_CACHE) >= 4096:
                _ST_INFO_CACHE.clear()
            st_info = _ST_INFO_CACHE[st_key] = mx.array(st_key, mx.int32)
    scale_lit = flit(float(scale))
    base_ins = [qkv_flat, stage, cs0, cs1, qr_scale, rel_w, meta_arr, st_info, cw01]
    if full:
        base_ins.append(tau.reshape(-1))
    opass = int(os.environ.get("LOOPLING_SDPA_MID_BATCH_OPASS", "1") or "1")
    qkv_lay_t = tuple(qkv_lay)

    def _run_chunk(b0, b1):
        bc = b1 - b0
        ins = base_ins + list(ks_main[b0:b1]) + list(vs_main[b0:b1])
        kkey = (bc, full, thr_q, thr_k, thr_r, eps_v, scale_lit)
        kern = _sdpa_mid_batch_kernels.get(kkey)
        if kern is None:
            sel = "switch (b) {\n" + "".join(
                f"        case {i}: K = k{i}; V = v{i}; break;\n" for i in range(bc)
            ) + "        default: K = k0; V = v0; break;\n    }"
            in_names = ["qkv", "stage", "cs0", "cs1", "qr_scale", "rw", "meta", "st_info", "cw01"]
            if full:
                in_names.append("tau")
            in_names += [f"k{i}" for i in range(bc)] + [f"v{i}" for i in range(bc)]
            kern = _sdpa_mid_batch_kernels[kkey] = mx.fast.metal_kernel(
                name="batch_sdpa_mid_b%d_%s_%d" % (bc, "F" if full else "S", abs(hash(kkey)) % 10**9),
                input_names=in_names,
                output_names=["out", "kv_out", "w0", "w1"],
                source=_sdpa_mid_batch_source(full, thr_q, thr_k, thr_r, eps_v, scale_lit, sel),
                header=_SDPA_MID_BATCH_HEADER,
                ensure_row_contiguous=False,
            )
        ns = _SDPA_MID_BATCH_NS_ENV or (32 if bc <= 2 else 8)
        tkey = ("b", bc, st_nrows, b0, full, Hq, Hkv, R, int(window), ns, opass, qkv_lay_t)
        tmpl = _sdpa_mid_tmpl_cache.get(tkey)
        if tmpl is None:
            if len(_sdpa_mid_tmpl_cache) >= 1024:
                _sdpa_mid_tmpl_cache.clear()
            gqa = Hq // Hkv
            offs = {item[0]: (item[1], item[2]) for item in qkv_lay_t}
            tmpl = [
                ("NROWS", bc), ("ST_NROWS", st_nrows), ("B_OFF", b0),
                ("MW", _SMW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                ("R", R), ("WIN", int(window)), ("NS", ns), ("OPASS", opass),
                ("OFF_Q", offs["q_proj"][0]), ("STR_Q", offs["q_proj"][1]),
                ("OFF_R", offs["r_proj"][0]), ("STR_R", offs["r_proj"][1]),
                ("OFF_K", offs["k_proj"][0]), ("STR_K", offs["k_proj"][1]),
                ("OFF_V", offs["v_proj"][0]), ("STR_V", offs["v_proj"][1]),
                ("GATE_OFF", offs["gate"][0]), ("GATE_STRIDE", offs["gate"][1]),
                ("SINK_OFF", offs["sinks"][0]), ("SINK_STRIDE", offs["sinks"][1]),
            ]
            if full:
                tmpl += [
                    ("OFF_KW", offs["k_wproj"][0]), ("STR_KW", offs["k_wproj"][1]),
                    ("OFF_VW", offs["v_wproj"][0]), ("STR_VW", offs["v_wproj"][1]),
                ]
            _sdpa_mid_tmpl_cache[tkey] = tmpl
        return kern(
            inputs=ins,
            template=tmpl,
            grid=(bc * Hq * 32 * ns, 1, 1),
            threadgroup=(32 * ns, 1, 1),
            output_shapes=[(bc, Hq * 128), (bc, 2, Hkv, 128), (bc, 3, Hkv * 128), (bc, 3, Hkv * 128)],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
        )

    if B <= 8:
        return _run_chunk(0, B)
    chunks = [_run_chunk(b0, min(B, b0 + 8)) for b0 in range(0, B, 8)]
    return (
        mx.concatenate([c[0] for c in chunks], axis=0),
        mx.concatenate([c[1] for c in chunks], axis=0),
        mx.concatenate([c[2] for c in chunks], axis=0),
        mx.concatenate([c[3] for c in chunks], axis=0),
    )


# =============================================================================================== Causal-W (W=1..8) Fused K5 MID+REL + K8 SDPA (S < 1024)
_SDPA_MID_CW_HEADER = _SDPA_MID_HEADER + r"""
template <int D, int NS, int OPASS = 2>
METAL_FUNC void sdpa1_head_v32_mid_causal(
    const threadgroup bfloat16_t* q_s,
    const device bfloat16_t* K_main,
    const device bfloat16_t* V_main,
    const threadgroup bfloat16_t* knew_s,
    const threadgroup bfloat16_t* vnew_s,
    const device bfloat16_t* gate,
    device bfloat16_t* out,
    const int N,
    const float scale,
    const float sink,
    const RowMaskTG msk,
    const int db,
    const int S0,
    const int slot0,
    const int max_size,
    const int rot,
    threadgroup float* outputs,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    uint simd_gid,
    uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int NV = BN / NS;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int EP = v_per_thread / OPASS;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o_reg[(OPASS == 1) ? 1 : NV][v_per_thread];
  const int loff = (int)simd_lid * qk_per_thread;
  bfloat4 qb = as_type<bfloat4>(((const threadgroup uint2*)(q_s + loff))[0]);
  q[0] = static_cast<U>(scale) * qb.x;
  q[1] = static_cast<U>(scale) * qb.y;
  q[2] = static_cast<U>(scale) * qb.z;
  q[3] = static_cast<U>(scale) * qb.w;
  for (int v = 0; v < NV; v++) {
    int vg = (int)simd_gid + v * NS;
    U o0 = 0, o1 = 0, o2 = 0, o3 = 0;
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    if (vg == 0) {
      max_score = sink;
      sum_exp_score = 1;
    }
    for (int i = vg; i < N; i += BN) {
      float fm = 0;
      if (msk.at(i, fm)) {
        int d = rot ? (i - slot0 + (i < slot0 ? max_size : 0)) : (i - S0);
        bfloat4 kb, vb;
        if (d >= 0 && d <= db) {
          kb = as_type<bfloat4>(((const threadgroup uint2*)(knew_s + (size_t)d * 128 + loff))[0]);
          vb = as_type<bfloat4>(((const threadgroup uint2*)(vnew_s + (size_t)d * 128 + loff))[0]);
        } else {
          kb = as_type<bfloat4>(((const device uint2*)(K_main + (size_t)i * 128 + loff))[0]);
          vb = as_type<bfloat4>(((const device uint2*)(V_main + (size_t)i * 128 + loff))[0]);
        }
        U score = 0;
        score += q[0] * kb.x;
        score += q[1] * kb.y;
        score += q[2] * kb.z;
        score += q[3] * kb.w;
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        o0 = o0 * factor + exp_score * vb.x;
        o1 = o1 * factor + exp_score * vb.y;
        o2 = o2 * factor + exp_score * vb.z;
        o3 = o3 * factor + exp_score * vb.w;
      }
    }
    if (simd_lid == 0) {
      max_scores[vg] = max_score;
      sum_exp_scores[vg] = sum_exp_score;
    }
    if constexpr (OPASS == 1) {
      outputs[(0 * BD + simd_lid) * BN + vg] = o0;
      outputs[(1 * BD + simd_lid) * BN + vg] = o1;
      outputs[(2 * BD + simd_lid) * BN + vg] = o2;
      outputs[(3 * BD + simd_lid) * BN + vg] = o3;
    } else {
      o_reg[v][0] = o0;
      o_reg[v][1] = o1;
      o_reg[v][2] = o2;
      o_reg[v][3] = o3;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  if constexpr (OPASS == 1) {
    for (int f = (int)simd_gid; f < BD; f += NS) {
      U r0 = simd_sum(outputs[(0 * BD + f) * BN + simd_lid] * factor);
      U r1 = simd_sum(outputs[(1 * BD + f) * BN + simd_lid] * factor);
      U r2 = simd_sum(outputs[(2 * BD + f) * BN + simd_lid] * factor);
      U r3 = simd_sum(outputs[(3 * BD + f) * BN + simd_lid] * factor);
      r0 = sum_exp_score == 0 ? r0 : (r0 / sum_exp_score);
      r1 = sum_exp_score == 0 ? r1 : (r1 / sum_exp_score);
      r2 = sum_exp_score == 0 ? r2 : (r2 / sum_exp_score);
      r3 = sum_exp_score == 0 ? r3 : (r3 / sum_exp_score);
      if (simd_lid == 0) {
        int base = f * v_per_thread;
        bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
        bfloat4 ob = bfloat4(
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(r0)), static_cast<float>(gb.x))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(r1)), static_cast<float>(gb.y))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(r2)), static_cast<float>(gb.z))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(r3)), static_cast<float>(gb.w))));
        ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
      }
    }
  } else {
    for (int p = 0; p < OPASS; p++) {
      for (int e = 0; e < EP; e++) {
        int i = p * EP + e;
        for (int v = 0; v < NV; v++) {
          int vg = (int)simd_gid + v * NS;
          outputs[(e * BD + simd_lid) * BN + vg] = o_reg[v][i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int e = 0; e < EP; e++) {
        int i = p * EP + e;
        for (int fi = 0; fi < NV; fi++) {
          int f = (int)simd_gid + fi * NS;
          U r = simd_sum(outputs[(e * BD + f) * BN + simd_lid] * factor);
          o_reg[fi][i] = sum_exp_score == 0 ? r : (r / sum_exp_score);
        }
      }
      if (p + 1 < OPASS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    if (simd_lid == 0) {
      for (int fi = 0; fi < NV; fi++) {
        int base = ((int)simd_gid + fi * NS) * v_per_thread;
        bfloat4 gb = as_type<bfloat4>(((const device uint2*)(gate + base))[0]);
        bfloat4 ob = bfloat4(
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o_reg[fi][0])), static_cast<float>(gb.x))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o_reg[fi][1])), static_cast<float>(gb.y))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o_reg[fi][2])), static_cast<float>(gb.z))),
            static_cast<bfloat16_t>(glue_gate(static_cast<float>(static_cast<bfloat16_t>(o_reg[fi][3])), static_cast<float>(gb.w))));
        ((device uint2*)(out + base))[0] = as_type<uint2>(ob);
      }
    }
  }
}
"""


def _sdpa_mid_cw_source(full, scale_lit, has_wa):
    scale_conv = flit(np.float32(math.sqrt(5)))
    k_cw_hoist = """
      float sw[4][4];
      for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) sw[i][k] = float(cw01[k * 1024 + c0 + i]);
      }""" if full else """
      uint4 cw0 = ((const device uint4*)(cw01 + c0 * 4))[0];
      uint4 cw1 = ((const device uint4*)(cw01 + c0 * 4 + 8))[0];
      bfloat4 cwb[4] = {as_type<bfloat4>(cw0.xy), as_type<bfloat4>(cw0.zw), as_type<bfloat4>(cw1.xy), as_type<bfloat4>(cw1.zw)};"""
    v_cw_hoist = """
      float sw[4][4];
      for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) sw[i][k] = float(cw01[4096 + k * 1024 + c0 + i]);
      }""" if full else """
      uint4 cw0 = ((const device uint4*)(cw01 + 4096 + c0 * 4))[0];
      uint4 cw1 = ((const device uint4*)(cw01 + 4096 + c0 * 4 + 8))[0];
      bfloat4 cwb[4] = {as_type<bfloat4>(cw0.xy), as_type<bfloat4>(cw0.zw), as_type<bfloat4>(cw1.xy), as_type<bfloat4>(cw1.zw)};"""
    k_conv_expr = """
          y[i] = mid_conv_d(x0v, s0v, s1v, s2v, sw[i], dy);""" if full else f"""
          y[i] = mid_conv_s(x0v, s0v, s1v, s2v, float(cwb[i].x), float(cwb[i].y), float(cwb[i].z), float(cwb[i].w), {scale_conv});"""
    v_conv_expr = """
          y[i] = mid_conv_d(x0v, s0v, s1v, s2v, sw[i], dy);""" if full else f"""
          y[i] = mid_conv_s(x0v, s0v, s1v, s2v, float(cwb[i].x), float(cwb[i].y), float(cwb[i].z), float(cwb[i].w), {scale_conv});"""
    k_dy_load = """
        bfloat4 dyp = as_type<bfloat4>(((const device uint2*)(qkv + OFF_KW + (size_t)row_a * STR_KW + kvh * 4))[0]);
        float dy[4] = {float(dyp.x), float(dyp.y), float(dyp.z), float(dyp.w)};""" if full else ""
    v_dy_load = """
        bfloat4 dyp = as_type<bfloat4>(((const device uint2*)(qkv + OFF_VW + (size_t)row_a * STR_VW + kvh * 4))[0]);
        float dy[4] = {float(dyp.x), float(dyp.y), float(dyp.z), float(dyp.w)};""" if full else ""

    k_anc_body = f"""
        const int row_a = anc_ptr[d];
        const int a0 = conv_anc[row_a * 3 + 0];
        const int a1 = conv_anc[row_a * 3 + 1];
        const int a2 = conv_anc[row_a * 3 + 2];
        bfloat4 xb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_K + (size_t)row_a * STR_K + c0))[0]);
        float4 x0f = float4(float(xb.x), float(xb.y), float(xb.z), float(xb.w));
        float4 s0f, s1f, s2f;
        if (a0 >= 0) {{ bfloat4 tb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_K + (size_t)a0 * STR_K + c0))[0]); s0f = float4(float(tb.x), float(tb.y), float(tb.z), float(tb.w)); }}
        else {{ s0f = ((const device float4*)(cs0 + (size_t)(-1 - a0) * 1024u + c0))[0]; }}
        if (a1 >= 0) {{ bfloat4 tb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_K + (size_t)a1 * STR_K + c0))[0]); s1f = float4(float(tb.x), float(tb.y), float(tb.z), float(tb.w)); }}
        else {{ s1f = ((const device float4*)(cs0 + (size_t)(-1 - a1) * 1024u + c0))[0]; }}
        if (a2 >= 0) {{ bfloat4 tb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_K + (size_t)a2 * STR_K + c0))[0]); s2f = float4(float(tb.x), float(tb.y), float(tb.z), float(tb.w)); }}
        else {{ s2f = ((const device float4*)(cs0 + (size_t)(-1 - a2) * 1024u + c0))[0]; }}
        if (d == db && kv_owner) {{
          device float* w0b = w0 + (size_t)b * 3072u;
          ((device float4*)(w0b + c0))[0] = s1f;
          ((device float4*)(w0b + 1024 + c0))[0] = s2f;
          ((device float4*)(w0b + 2048 + c0))[0] = x0f;
        }}{k_dy_load}
        float x0a[4] = {{x0f.x, x0f.y, x0f.z, x0f.w}};
        float s0a[4] = {{s0f.x, s0f.y, s0f.z, s0f.w}};
        float s1a[4] = {{s1f.x, s1f.y, s1f.z, s1f.w}};
        float s2a[4] = {{s2f.x, s2f.y, s2f.z, s2f.w}};
        float y[4];
        for (int i = 0; i < 4; i++) {{
          float x0v = x0a[i], s0v = s0a[i], s1v = s1a[i], s2v = s2a[i];
          {k_conv_expr}
        }}
        glue_l2_clip_lane128(y, thr_k);
        uint2 pk = as_type<uint2>(bfloat4(bfloat16_t(y[0]), bfloat16_t(y[1]), bfloat16_t(y[2]), bfloat16_t(y[3])));
        kn_raw[(uint)d * 32u + simd_lid] = pk;
        if (d == db && kv_owner) ((device uint2*)(kv_out + ((size_t)kvh * WW + (size_t)b) * 128u + simd_lid * 4u))[0] = pk;
    """

    def v_anc_body(tmp_off):
        return f"""
        const int row_a = anc_ptr[d];
        const int a0 = conv_anc[row_a * 3 + 0];
        const int a1 = conv_anc[row_a * 3 + 1];
        const int a2 = conv_anc[row_a * 3 + 2];
        bfloat4 xb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_V + (size_t)row_a * STR_V + c0))[0]);
        float4 x0f = float4(float(xb.x), float(xb.y), float(xb.z), float(xb.w));
        float4 s0f, s1f, s2f;
        if (a0 >= 0) {{ bfloat4 tb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_V + (size_t)a0 * STR_V + c0))[0]); s0f = float4(float(tb.x), float(tb.y), float(tb.z), float(tb.w)); }}
        else {{ s0f = ((const device float4*)(cs1 + (size_t)(-1 - a0) * 1024u + c0))[0]; }}
        if (a1 >= 0) {{ bfloat4 tb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_V + (size_t)a1 * STR_V + c0))[0]); s1f = float4(float(tb.x), float(tb.y), float(tb.z), float(tb.w)); }}
        else {{ s1f = ((const device float4*)(cs1 + (size_t)(-1 - a1) * 1024u + c0))[0]; }}
        if (a2 >= 0) {{ bfloat4 tb = as_type<bfloat4>(((const device uint2*)(qkv + OFF_V + (size_t)a2 * STR_V + c0))[0]); s2f = float4(float(tb.x), float(tb.y), float(tb.z), float(tb.w)); }}
        else {{ s2f = ((const device float4*)(cs1 + (size_t)(-1 - a2) * 1024u + c0))[0]; }}
        if (d == db && kv_owner) {{
          device float* w1b = w1 + (size_t)b * 3072u;
          ((device float4*)(w1b + c0))[0] = s1f;
          ((device float4*)(w1b + 1024 + c0))[0] = s2f;
          ((device float4*)(w1b + 2048 + c0))[0] = x0f;
        }}{v_dy_load}
        float x0a[4] = {{x0f.x, x0f.y, x0f.z, x0f.w}};
        float s0a[4] = {{s0f.x, s0f.y, s0f.z, s0f.w}};
        float s1a[4] = {{s1f.x, s1f.y, s1f.z, s1f.w}};
        float s2a[4] = {{s2f.x, s2f.y, s2f.z, s2f.w}};
        float y[4];
        for (int i = 0; i < 4; i++) {{
          float x0v = x0a[i], s0v = s0a[i], s1v = s1a[i], s2v = s2a[i];
          {v_conv_expr}
        }}
        threadgroup float* vtmp = outputs + {tmp_off}u;
        ((threadgroup float4*)(vtmp + simd_lid * 4u))[0] = float4(y[0], y[1], y[2], y[3]);
        float inv_v = glue_rms_inv_sg<128, true>(vtmp, eps_v, simd_lid);
        uint2 pk = as_type<uint2>(bfloat4(
            glue_rms_apply_nw<bfloat16_t>(y[0], inv_v),
            glue_rms_apply_nw<bfloat16_t>(y[1], inv_v),
            glue_rms_apply_nw<bfloat16_t>(y[2], inv_v),
            glue_rms_apply_nw<bfloat16_t>(y[3], inv_v)));
        vn_raw[(uint)d * 32u + simd_lid] = pk;
        if (d == db && kv_owner) ((device uint2*)(kv_out + ((size_t)(HKV + kvh) * WW + (size_t)b) * 128u + simd_lid * 4u))[0] = pk;
    """

    wa_guard = "\n    if (b >= w_act[0]) return;" if has_wa else ""
    return f"""
    constexpr int D = 128;
    constexpr int P_SG = (NS >= 16) ? 8 : 4;
    constexpr int KV_SLOTS = (WW <= 4) ? 4 : 8;
    threadgroup float outputs[(4 / OPASS) * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    threadgroup uint2 q_raw[32];
    threadgroup uint2 kn_raw[KV_SLOTS * 32];
    threadgroup uint2 vn_raw[KV_SLOTS * 32];
    threadgroup bfloat16_t pb_s[1024];
    threadgroup bfloat16_t* q_s = (threadgroup bfloat16_t*)q_raw;
    threadgroup bfloat16_t* knew_s = (threadgroup bfloat16_t*)kn_raw;
    threadgroup bfloat16_t* vnew_s = (threadgroup bfloat16_t*)vn_raw;

    const int tg = (int)threadgroup_position_in_grid.x;
    const int gpk = WW * GQA;
    const int kvh = tg / gpk;
    const int rem = tg % gpk;
    const int b = rem / GQA;
    const int h = kvh * GQA + (rem % GQA);{wa_guard}
    const int bh = b * HQ + h;
    const bool kv_owner = (h == kvh * GQA);
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;

    const float thr_q = lparams[0], thr_k = lparams[1], thr_r = lparams[2], eps_v = lparams[3];
    const int S0 = meta[0], end0 = meta[1], slot0 = meta[2], cap = meta[3], max_size = meta[4];
    const int lidx = (lidx_in[0] >= 0) ? lidx_in[0] : meta[5];
    const int db = tree_meta[b * 9];
    const device int* anc_ptr = tree_meta + b * 9 + 1;
    const int S_b = ROT ? min(end0 + db + 1, max_size) : (S0 + db + 1);
    const int end_b = end0 + db + 1;
    const int idx_b = ROT ? (((slot0 + db) % max_size) + 1) : 0;
    {"const float tau_v = tau[b];" if full else ""}

    if (simd_gid == 0u) {{
      const uint qoff_w = (uint)h * 128u + simd_lid * 4u;
      const size_t qoff = OFF_Q + (size_t)b * STR_Q + qoff_w;
      bfloat4 xb = as_type<bfloat4>(((const device uint2*)(qkv + qoff))[0]);
      bfloat4 qsb = as_type<bfloat4>(((const device uint2*)(qr_scale + qoff_w))[0]);
      float s[4] = {{
          float(xb.x) * float(qsb.x),
          float(xb.y) * float(qsb.y),
          float(xb.z) * float(qsb.z),
          float(xb.w) * float(qsb.w),
      }};
      glue_l2_clip_lane128(s, thr_q);
      bfloat16_t qo[4];
      for (int i = 0; i < 4; i++) {{
        bfloat16_t o = bfloat16_t(s[i]);
        {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
        qo[i] = o;
      }}
      q_raw[simd_lid] = as_type<uint2>(bfloat4(qo[0], qo[1], qo[2], qo[3]));
      if (db >= 1) {{
        const int c0 = kvh * 128 + (int)simd_lid * 4;{k_cw_hoist}
        for (int d = 1; d <= db; d += 2) {{
          {k_anc_body}
        }}
      }}
    }} else if (simd_gid == 1u) {{
      const int c0 = kvh * 128 + (int)simd_lid * 4;{k_cw_hoist}
      for (int d = 0; d <= db; d += 2) {{
        {k_anc_body}
      }}
    }} else if (simd_gid == 2u) {{
      const int c0 = kvh * 128 + (int)simd_lid * 4;{v_cw_hoist}
      for (int d = 0; d <= db; d += 2) {{
        {v_anc_body(0)}
      }}
    }} else if (simd_gid == 7u) {{
      if (db >= 1) {{
        const int c0 = kvh * 128 + (int)simd_lid * 4;{v_cw_hoist}
        for (int d = 1; d <= db; d += 2) {{
          {v_anc_body(128)}
        }}
      }}
    }} else {{
      const int p = (simd_gid >= 3u && simd_gid <= 6u) ? ((int)simd_gid - 3)
                    : ((P_SG == 8 && simd_gid >= 8u && simd_gid < 12u) ? ((int)simd_gid - 4) : -1);
      if (p >= 0) {{
        const int n_pb = min(S_b, (WIN > 0) ? min(R, WIN) : R);
        const int n_pb16 = min(R, (n_pb + 15) & ~15);
        if (p * 16 < n_pb16) {{
          threadgroup bfloat16_t* rel_tile = (threadgroup bfloat16_t*)(outputs + 256u + (uint)p * 64u);
          ((threadgroup uint2*)rel_tile)[simd_lid] = uint2(0u, 0u);
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_lid == 0u) {{
            float s[16];
            const device bfloat16_t* rp = qkv + OFF_R + (size_t)b * STR_R + (size_t)h * 16u;
            const device bfloat16_t* rsp = qr_scale + HQ * 128 + h * 16;
            for (int i = 0; i < 16; i++) s[i] = float(rp[i]) * float(rsp[i]);
            glue_l2_clip_row16(s, thr_r);
            for (int i = 0; i < 16; i++) {{
              bfloat16_t o = bfloat16_t(s[i]);
              {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
              rel_tile[i] = o;
            }}
          }}
          simdgroup_barrier(mem_flags::mem_threadgroup);
          simdgroup_bfloat8x8 ma0, ma1;
          simdgroup_load(ma0, rel_tile + 0, 16);
          simdgroup_load(ma1, rel_tile + 8, 16);
          threadgroup float* st = outputs + 768u + (uint)p * 128u;
          for (int n0 = p * 16; n0 < n_pb16; n0 += P_SG * 16) {{
            const device bfloat16_t* Bm = rw + n0;
            simdgroup_bfloat8x8 mb00, mb10, mb01, mb11;
            simdgroup_load(mb00, Bm + 0 * R + 0, R);
            simdgroup_load(mb10, Bm + 8 * R + 0, R);
            simdgroup_load(mb01, Bm + 0 * R + 8, R);
            simdgroup_load(mb11, Bm + 8 * R + 8, R);
            simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_multiply_accumulate(acc0, ma0, mb00, acc0);
            simdgroup_multiply_accumulate(acc0, ma1, mb10, acc0);
            simdgroup_multiply_accumulate(acc1, ma0, mb01, acc1);
            simdgroup_multiply_accumulate(acc1, ma1, mb11, acc1);
            simdgroup_store(acc0, st, 8);
            simdgroup_store(acc1, st + 64u, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lid < 16u) {{
              int col = n0 + (int)simd_lid;
              if (col < 1024) pb_s[col] = bfloat16_t(st[(simd_lid < 8u) ? simd_lid : (64u + simd_lid - 8u)]);
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
          }}
        }}
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device bfloat16_t* K = kv + ((size_t)lidx * 2 * HKV + kvh) * (size_t)cap * D;
    const device bfloat16_t* V = kv + ((size_t)lidx * 2 * HKV + HKV + kvh) * (size_t)cap * D;
    RowMaskTG msk;
    msk.pb = pb_s;
    msk.S = S_b; msk.end = end_b; msk.idx = idx_b; msk.rot = ROT;
    msk.R = R; msk.window = WIN;
    float sink = float(static_cast<bfloat16_t>(qkv[SINK_OFF + (size_t)b * SINK_STRIDE + h]));
    sdpa1_head_v32_mid_causal<D, NS, OPASS>(
        q_s, K, V, knew_s, vnew_s,
        qkv + GATE_OFF + (size_t)b * GATE_STRIDE + (size_t)h * D,
        out + (size_t)bh * D,
        S_b, {scale_lit}, sink, msk, db, S0, slot0, max_size, ROT,
        outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
"""


_sdpa_mid_cw_kernels: dict = {}


def sdpa_mid_causal_w_packed(
    layer, qkv_flat, qkv_lay, kv_buf, cs0, cs1, rel_w,
    S0, end0, slot0, rot, cap, max_size, W, *,
    scale, window, tau=None, kv_layer_idx=0, tree_meta=None, conv_anc=None,
    depths=None, meta_arr=None, s_bounds=None, w_act=None,
):
    """Fused Causal-W (1 <= W <= 8, S_max < 1024) K5 MID+REL + K8 SDPA over pre-step packed KV.
    For W > _SDPA_MID_CW_MAX_W (default 4, where separate vectorized MID+SDPA beats fused inside compiled verify_w_gpu),
    delegates cleanly to mid_causal_w_flat + sdpa_causal_w_packed with identical return signature.
    Returns (attn_out (W, Hq*128) bf16, kv_new (2, Hkv, W, 128) bf16, w0 (W, 3, Hkv*128) f32, w1 (W, 3, Hkv*128) f32)."""
    assert 1 <= W <= 8 and kv_buf.dtype == mx.bfloat16 and qkv_flat.dtype == mx.bfloat16
    if W > _SDPA_MID_CW_MAX_W:
        q, kv_new, pb, w0, w1 = mid_causal_w_flat(
            layer, qkv_flat, qkv_lay, cs0, cs1, W, tau=tau, rel_w=rel_w, conv_anc=conv_anc, w_act=w_act,
        )
        offs_d = {item[0]: (item[1], item[2]) for item in qkv_lay}
        gate_off, gate_stride = offs_d["gate"]
        sink_off, sink_stride = offs_d["sinks"]
        attn_out = sdpa_causal_w_packed(
            q, kv_buf, kv_new, pb, qkv_flat, gate_off, sink_off,
            S0, end0, slot0, rot, cap, max_size, W,
            scale=scale, window=window, kv_layer_idx=kv_layer_idx, tree_meta=tree_meta, depths=depths,
            meta_arr=meta_arr, s_bounds=s_bounds, w_act=w_act,
            gate_stride=gate_stride, sink_stride=sink_stride,
        )
        return attn_out, kv_new, w0, w1
    full, qr_scale, cw01, lparams, _thr_q, _thr_k, _thr_r, _eps_v, Hq, Hkv = _get_sdpa_mid_batch_layer_meta(layer)
    R = int(rel_w.shape[1])
    if tree_meta is None:
        _, tree_meta = _default_tree_meta(W)
    if conv_anc is None:
        conv_anc = _default_conv_anc(W)
    if meta_arr is not None:
        mta = meta_arr
        kv_lidx_v = int(kv_layer_idx)
    else:
        mkey = (int(S0), int(end0), int(slot0), int(cap), int(max_size), int(kv_layer_idx))
        mta = _CW_META_CACHE.get(mkey)
        if mta is None:
            if len(_CW_META_CACHE) >= 256:
                _CW_META_CACHE.clear()
            mta = _CW_META_CACHE[mkey] = mx.array(mkey, mx.int32)
        kv_lidx_v = -1
    lidx_arr = _SDPA_MID_LIDX_CACHE.get(kv_lidx_v)
    if lidx_arr is None:
        lidx_arr = _SDPA_MID_LIDX_CACHE[kv_lidx_v] = mx.array([kv_lidx_v], mx.int32)
    has_wa = w_act is not None
    scale_lit = flit(float(scale))
    ins = [qkv_flat, kv_buf, cs0, cs1, qr_scale, rel_w, mta, cw01, tree_meta, conv_anc, lparams, lidx_arr]
    if full:
        ins.append(tau.reshape(-1))
    if has_wa:
        ins.append(w_act.reshape(-1))
    kkey = (full, has_wa, scale_lit)
    kern = _sdpa_mid_cw_kernels.get(kkey)
    if kern is None:
        in_names = ["qkv", "kv", "cs0", "cs1", "qr_scale", "rw", "meta", "cw01", "tree_meta", "conv_anc", "lparams", "lidx_in"]
        if full:
            in_names.append("tau")
        if has_wa:
            in_names.append("w_act")
        kern = _sdpa_mid_cw_kernels[kkey] = mx.fast.metal_kernel(
            name="cw_sdpa_mid_%s_%s_%d" % ("F" if full else "S", "D" if has_wa else "S", abs(hash(kkey)) % 10**9),
            input_names=in_names,
            output_names=["out", "kv_out", "w0", "w1"],
            source=_sdpa_mid_cw_source(full, scale_lit, has_wa),
            header=_SDPA_MID_CW_HEADER,
            ensure_row_contiguous=True,
        )
    ns = _SDPA_MID_CW_NS_ENV or (16 if W <= 4 else 8)
    opass = int(os.environ.get("LOOPLING_SDPA_MID_CW_OPASS", "1") or "1")
    tkey = ("cw", full, Hq, Hkv, R, int(window), ns, opass, int(rot), int(W), tuple(qkv_lay))
    tmpl = _sdpa_mid_tmpl_cache.get(tkey)
    if tmpl is None:
        if len(_sdpa_mid_tmpl_cache) >= 512:
            _sdpa_mid_tmpl_cache.clear()
        gqa = Hq // Hkv
        offs = {item[0]: (item[1], item[2]) for item in qkv_lay}
        tmpl = [
            ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa), ("R", R), ("WIN", int(window)), ("NS", ns),
            ("OPASS", opass), ("ROT", int(rot)), ("WW", int(W)),
            ("OFF_Q", offs["q_proj"][0]), ("STR_Q", offs["q_proj"][1]),
            ("OFF_R", offs["r_proj"][0]), ("STR_R", offs["r_proj"][1]),
            ("OFF_K", offs["k_proj"][0]), ("STR_K", offs["k_proj"][1]),
            ("OFF_V", offs["v_proj"][0]), ("STR_V", offs["v_proj"][1]),
            ("GATE_OFF", offs["gate"][0]), ("GATE_STRIDE", offs["gate"][1]),
            ("SINK_OFF", offs["sinks"][0]), ("SINK_STRIDE", offs["sinks"][1]),
        ]
        if full:
            tmpl += [
                ("OFF_KW", offs["k_wproj"][0]), ("STR_KW", offs["k_wproj"][1]),
                ("OFF_VW", offs["v_wproj"][0]), ("STR_VW", offs["v_wproj"][1]),
            ]
        _sdpa_mid_tmpl_cache[tkey] = tmpl
    return kern(
        inputs=ins,
        template=tmpl,
        grid=(W * Hq * 32 * ns, 1, 1),
        threadgroup=(32 * ns, 1, 1),
        output_shapes=[(W, Hq * 128), (2, Hkv, W, 128), (W, 3, Hkv * 128), (W, 3, Hkv * 128)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
    )
