"""Batched KV store for B decode streams of ONE layer visit (lane F5 + Phase 4 PagedRowKV).

Sliding layers (`RotatingKVCache`, max_size=1024): bounded double-buffered ring (constant size per visit) with
dynamic per-step `has2` and `ns = gqa if min(lens) >= 1024 else 32`.

Full-attention layers (`KVCache`, ROWKV_PAGED=1 default ON): 256-token paged KV store:
  * immutable `base_K, base_V (n_base, Hkv, 256, D)` bf16 holding complete prefill pages (zero padding waste, 1x RAM),
  * small double-buffered `ov_K, ov_V` allocated lazily if decoding crosses page boundaries (no multi-GB pool copies),
  * double-buffered active tail slab `act_K, act_V (B * Hkv * 256, D)` bf16 (~4 MB per buffer; always donatable).

ROWKV_PAGED=0 falls back to the un-paged RowKV store. Supports optional fused `attn_out * sigmoid(gate)` store
epilogue via `sdpa(..., gate=gate)` so o_proj can run as plain `qmv_rows`. Every row is bitwise == `fastdecode._core`.
"""
import os
from types import SimpleNamespace

import numpy as np

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.inkling2.rows.sdpa import _DUMMY_GATE, _HEADER, _STAGE2, _TWO_PASS_N, _blocks

_MW = 4  # per-row table columns: S, end, idx, rot
D = 128
PAGE = 256
_DUMMY_BF = mx.zeros((256,), mx.bfloat16)  # >= 512B ensures Metal device address space

_STAGE1_FLAT = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;  // threadgroups per kv head in the 2-pass stage 1
    constexpr int NTG2 = HKV * TGPK;
    constexpr int NG = HAS2 ? NTG2 : HQ;                // threadgroups reserved per row (static)
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = (int)t / NG;
    int tl = (int)t % NG;
    const int S = meta[b * MW + 0];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[b * MW + 1]; msk.idx = meta[b * MW + 2]; msk.rot = meta[b * MW + 3];
    msk.R = R; msk.window = WIN;
    const float sc = scale[0];
    if (S < 1024) {
        if (tl >= HQ) return;
        int h = tl;
        int kvh = h / GQA;
        int bh = b * HQ + h;
        size_t base = ((size_t)(b * HKV + kvh) * CAP) * D;
        msk.pb = pb + (size_t)bh * R;
        sdpa1_head<T, PBT, D, HAS_GATE>(
            q + (size_t)bh * D, kk + base, vv + base, gate + (size_t)bh * D, o1 + (size_t)bh * D, S, sc,
            float(static_cast<T>(sinks[bh])), msk, outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
    } else {
        int kvh = tl / TGPK;
        int f = (tl % TGPK) * NS + (int)simd_gid;
        if (f < BLOCKS * GQA) {
            int block = f / GQA;
            int h = kvh * GQA + f % GQA;
            int bh = b * HQ + h;
            size_t base = ((size_t)(b * HKV + kvh) * CAP) * D;
            msk.pb = pb + (size_t)bh * R;
            sdpa2_block<T, PBT, D, BLOCKS>(
                q + (size_t)bh * D, kk + base, vv + base,
                partials + ((size_t)bh * BLOCKS + block) * D, sums + (size_t)bh * BLOCKS + block,
                maxs + (size_t)bh * BLOCKS + block, S, sc, float(static_cast<T>(sinks[bh])), msk, block, simd_lid);
        }
    }
"""

_STAGE1_FLAT_2PASS = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    constexpr int NTG2 = HKV * TGPK;
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = (int)t / NTG2;
    int tl = (int)t % NTG2;
    const int S = meta[b * MW + 0];
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
        size_t base = ((size_t)(b * HKV + kvh) * CAP) * D;
        msk.pb = pb + (size_t)bh * R;
        sdpa2_block<T, PBT, D, BLOCKS>(
            q + (size_t)bh * D, kk + base, vv + base,
            partials + ((size_t)bh * BLOCKS + block) * D, sums + (size_t)bh * BLOCKS + block,
            maxs + (size_t)bh * BLOCKS + block, S, sc, float(static_cast<T>(sinks[bh])), msk, block, simd_lid);
    }
"""

_PAGED_HEADER = _HEADER + r"""
template <typename T>
struct PagedKvPair {
  const device T* k;
  const device T* v;
};

template <typename T>
METAL_FUNC PagedKvPair<T> paged_kv_pair(
    const device T* base_k, const device T* ov_k, const device T* act_k,
    const device T* base_v, const device T* ov_v, const device T* act_v,
    const device int* pt_row, int n_base, int hkv, int kvh, int b, int page_idx, int tok_off) {
  int pid = pt_row[page_idx];
  if (pid >= 0) {
    if (pid < n_base) {
      size_t bo = ((((size_t)pid * hkv + kvh) << 8) * 128) + tok_off;
      return {base_k + bo, base_v + bo};
    } else {
      size_t bo = ((((size_t)(pid - n_base) * hkv + kvh) << 8) * 128) + tok_off;
      return {ov_k + bo, ov_v + bo};
    }
  }
  size_t ao = (((size_t)(b * hkv + kvh) << 8) * 128) + tok_off;
  return {act_k + ao, act_v + ao};
}

template <typename T, typename PBT, int D, bool HAS_GATE, typename GT>
METAL_FUNC void sdpa1_head_paged(
    const device T* queries, const device T* base_k, const device T* ov_k, const device T* act_k,
    const device T* base_v, const device T* ov_v, const device T* act_v, const device int* pt_row,
    const int n_base, const int hkv, const int kvh, const int b, GT gate, device T* out, const int N,
    const float scale, const float sink, const RowMask<T, PBT> msk, threadgroup float* outputs,
    threadgroup float* max_scores, threadgroup float* sum_exp_scores, uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  typedef float U;
  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];
  const int loff = (int)simd_lid * qk_per_thread;
  queries += loff;
  gate += (int)simd_gid * v_per_thread;
  out += (int)simd_gid * v_per_thread;
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
      PagedKvPair<T> kv = paged_kv_pair(
          base_k, ov_k, act_k, base_v, ov_v, act_v, pt_row, n_base, hkv, kvh, b, i >> 8, (i & 255) * 128 + loff);
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = kv.k[j];
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
        o[j] = o[j] * factor + exp_score * kv.v[j];
      }
    }
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

template <typename T, typename PBT, int D, int BLOCKS>
METAL_FUNC void sdpa2_block_paged(
    const device T* queries, const device T* base_k, const device T* ov_k, const device T* act_k,
    const device T* base_v, const device T* ov_v, const device T* act_v, const device int* pt_row,
    const int n_base, const int hkv, const int kvh, const int b, device T* out, device float* sums,
    device float* maxs, const int N, const float scale, const float sink, const RowMask<T, PBT> msk,
    const int block_idx, uint simd_lid) {
  static_assert(BLOCKS == 128, "sdpa2_block_paged 2-page hoisting assumes BLOCKS == 128");
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
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (block_idx == 0) {
    max_score = sink;
    sum_exp_score = 1;
  }
  const int blk_off = block_idx * 128 + loff;
  int i = block_idx;
  int p = 0;
  for (; i + (TK - 1) * blocks < N; i += blocks * TK, p += 2) {
    PagedKvPair<T> kv0 = paged_kv_pair(base_k, ov_k, act_k, base_v, ov_v, act_v, pt_row, n_base, hkv, kvh, b, p, blk_off);
    PagedKvPair<T> kv1 = paged_kv_pair(base_k, ov_k, act_k, base_v, ov_v, act_v, pt_row, n_base, hkv, kvh, b, p + 1, blk_off);
    T k_reg[TK][qk_per_thread];
    T v_reg[TK][v_per_thread];
    for (int j = 0; j < qk_per_thread; j++) {
      k_reg[0][j] = kv0.k[j];
      k_reg[1][j] = kv0.k[128 * 128 + j];
      k_reg[2][j] = kv1.k[j];
      k_reg[3][j] = kv1.k[128 * 128 + j];
    }
    for (int j = 0; j < v_per_thread; j++) {
      v_reg[0][j] = kv0.v[j];
      v_reg[1][j] = kv0.v[128 * 128 + j];
      v_reg[2][j] = kv1.v[j];
      v_reg[3][j] = kv1.v[128 * 128 + j];
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
  for (; i < N; i += blocks) {
    float fm = 0;
    bool use_key = msk.at(i, fm);
    if (use_key) {
      PagedKvPair<T> kv = paged_kv_pair(
          base_k, ov_k, act_k, base_v, ov_v, act_v, pt_row, n_base, hkv, kvh, b, i >> 8, (i & 255) * 128 + loff);
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * kv.k[j];
      }
      score = simd_sum(score);
      score += fm;
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * kv.v[j];
      }
    }
  }
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i2 = 0; i2 < v_per_thread; i2++) {
    out[i2] = static_cast<T>(o[i2]);
  }
}
"""

_STAGE1_PAGED = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    constexpr int NTG2 = HKV * TGPK;
    constexpr int NG = HAS2 ? NTG2 : HQ;
    threadgroup float outputs[4 * 1024];
    threadgroup float max_scores[32];
    threadgroup float sum_exp_scores[32];
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = (int)t / NG;
    int tl = (int)t % NG;
    const int S = meta[b * MW + 0];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[b * MW + 1]; msk.idx = meta[b * MW + 2]; msk.rot = meta[b * MW + 3];
    msk.R = R; msk.window = WIN;
    const float sc = scale[0];
    const device int* pt_row = pt + b * CAP_PAGES;
    if (S < 1024) {
        if (tl >= HQ) return;
        int h = tl;
        int kvh = h / GQA;
        int bh = b * HQ + h;
        msk.pb = pb + (size_t)bh * R;
        sdpa1_head_paged<T, PBT, D, HAS_GATE>(
            q + (size_t)bh * D, base_k, ov_k, act_k, base_v, ov_v, act_v, pt_row, N_BASE, HKV, kvh, b,
            gate + (size_t)bh * D, o1 + (size_t)bh * D, S, sc, float(static_cast<T>(sinks[bh])),
            msk, outputs, max_scores, sum_exp_scores, simd_gid, simd_lid);
    } else {
        int kvh = tl / TGPK;
        int f = (tl % TGPK) * NS + (int)simd_gid;
        if (f < BLOCKS * GQA) {
            int block = f / GQA;
            int h = kvh * GQA + f % GQA;
            int bh = b * HQ + h;
            msk.pb = pb + (size_t)bh * R;
            sdpa2_block_paged<T, PBT, D, BLOCKS>(
                q + (size_t)bh * D, base_k, ov_k, act_k, base_v, ov_v, act_v, pt_row, N_BASE, HKV, kvh, b,
                partials + ((size_t)bh * BLOCKS + block) * D, sums + (size_t)bh * BLOCKS + block,
                maxs + (size_t)bh * BLOCKS + block, S, sc, float(static_cast<T>(sinks[bh])), msk, block, simd_lid);
        }
    }
"""

_STAGE1_PAGED_2PASS = r"""
    constexpr int D = 128;
    constexpr int TGPK = (BLOCKS * GQA + NS - 1) / NS;
    constexpr int NTG2 = HKV * TGPK;
    uint t = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int b = (int)t / NTG2;
    int tl = (int)t % NTG2;
    const int S = meta[b * MW + 0];
    RowMask<T, PBT> msk;
    msk.S = S; msk.end = meta[b * MW + 1]; msk.idx = meta[b * MW + 2]; msk.rot = meta[b * MW + 3];
    msk.R = R; msk.window = WIN;
    const float sc = scale[0];
    const device int* pt_row = pt + b * CAP_PAGES;
    int kvh = tl / TGPK;
    int f = (tl % TGPK) * NS + (int)simd_gid;
    if (f < BLOCKS * GQA) {
        int block = f / GQA;
        int h = kvh * GQA + f % GQA;
        int bh = b * HQ + h;
        msk.pb = pb + (size_t)bh * R;
        sdpa2_block_paged<T, PBT, D, BLOCKS>(
            q + (size_t)bh * D, base_k, ov_k, act_k, base_v, ov_v, act_v, pt_row, N_BASE, HKV, kvh, b,
            partials + ((size_t)bh * BLOCKS + block) * D, sums + (size_t)bh * BLOCKS + block,
            maxs + (size_t)bh * BLOCKS + block, S, sc, float(static_cast<T>(sinks[bh])), msk, block, simd_lid);
    }
"""

_kernels: dict = {}


def _kernel_pair(paged=False):
    key = 1 if paged else 0
    ks = _kernels.get(key)
    if ks is None:
        if paged:
            s1 = mx.fast.metal_kernel(
                name="kvstore_sdpa_paged_s1",
                input_names=["q", "pb", "sinks", "meta", "scale", "gate", "pt",
                             "act_k", "act_v", "base_k", "base_v", "ov_k", "ov_v"],
                output_names=["o1", "partials", "sums", "maxs"],
                source=_STAGE1_PAGED, header=_PAGED_HEADER, ensure_row_contiguous=False,
            )
            s1_2pass = mx.fast.metal_kernel(
                name="kvstore_sdpa_paged_s1_2pass",
                input_names=["q", "pb", "sinks", "meta", "scale", "pt",
                             "act_k", "act_v", "base_k", "base_v", "ov_k", "ov_v"],
                output_names=["partials", "sums", "maxs"],
                source=_STAGE1_PAGED_2PASS, header=_PAGED_HEADER, ensure_row_contiguous=False,
            )
        else:
            s1 = mx.fast.metal_kernel(
                name="kvstore_sdpa_s1", input_names=["q", "pb", "sinks", "meta", "scale", "gate", "kk", "vv"],
                output_names=["o1", "partials", "sums", "maxs"], source=_STAGE1_FLAT, header=_HEADER,
                ensure_row_contiguous=False,
            )
            s1_2pass = mx.fast.metal_kernel(
                name="kvstore_sdpa_s1_2pass", input_names=["q", "pb", "sinks", "meta", "scale", "kk", "vv"],
                output_names=["partials", "sums", "maxs"], source=_STAGE1_FLAT_2PASS, header=_HEADER,
                ensure_row_contiguous=False,
            )
        s2 = _kernels.get("s2")
        if s2 is None:
            s2 = _kernels["s2"] = mx.fast.metal_kernel(
                name="rows_sdpa_s2", input_names=["partials", "sums", "maxs", "o1", "meta", "gate"],
                output_names=["out"], source=_STAGE2, header=_HEADER, ensure_row_contiguous=False,
            )
        ks = _kernels[key] = (s1, s1_2pass, s2)
    return ks


def _advance_full(off, idx, cap, step):
    """KVCache.update_and_fetch(S=1) on host state -> (write slot, off', idx', cap')."""
    if off + 1 > cap:
        cap = off + step
    return off, off + 1, 0, cap


def _advance_ring(off, idx, cap, step, max_size, keep):
    """RotatingKVCache._update_in_place(S=1) on host state -> (write slot, off', idx', cap')."""
    if off >= cap and cap < max_size:
        cap += min(step, max_size - off)
        idx = off
    assert cap <= max_size
    if idx == max_size:
        idx = keep
    return idx, off + 1, idx + 1, cap


class RowKV:
    """B streams' keys/values of one layer visit; paged (256-tok pages) on full attention, bounded ring on sliding."""

    def __init__(self, caches, *, paged=None):
        assert len(caches) >= 1
        kinds = {type(c) for c in caches}
        assert kinds <= {KVCache} or kinds <= {RotatingKVCache}, f"mixed or unsupported cache classes {kinds}"
        self.caches = list(caches)
        self.B = B = len(caches)
        self.ring = isinstance(caches[0], RotatingKVCache)
        self.step = caches[0].step
        assert all(c.step == self.step for c in caches)
        if self.ring:
            self.max_size = caches[0].max_size
            assert all(c.max_size == self.max_size and c.keep == 0 for c in caches), "ring rows need keep=0, equal max_size"
        use_paged = (os.environ.get("ROWKV_PAGED", "1") not in ("0", "")) if paged is None else bool(paged)
        self.paged = (not self.ring) and use_paged

        ks, vs, st = [], [], []
        for c in caches:
            assert c.keys is not None and c.keys.dtype == mx.bfloat16 and c.values.dtype == mx.bfloat16
            kb, vb = c.keys, c.values
            assert kb.shape[0] == 1 and kb.shape[3] == D and vb.shape == kb.shape, (kb.shape, vb.shape)
            off = int(c.offset)
            if self.ring:
                idx = int(c._idx)
                if kb.shape[2] > self.max_size:
                    t = kb.shape[2] - self.max_size
                    kb, vb = kb[..., t:, :], vb[..., t:, :]
                    idx = self.max_size
            else:
                idx = 0
                assert off <= kb.shape[2]
            ks.append(kb)
            vs.append(vb)
            st.append((off, idx, kb.shape[2]))
        self.Hkv = ks[0].shape[1]
        assert all(k.shape[1] == self.Hkv for k in ks)
        self.state = tuple(st)
        self.double = os.environ.get("ROWKV_DOUBLE", "1") not in ("0", "")
        self._plan = None
        self._scale = {}

        if self.paged:
            self.cap = max(s[2] for s in st)
            self.pages = [[] for _ in range(B)]
            base_k_list, base_v_list, act_k_list, act_v_list = [], [], [], []
            pid = 0
            for b, ((off, _, _), kb, vb) in enumerate(zip(st, ks, vs)):
                n_full = off // PAGE
                rem = off % PAGE
                if n_full > 0:
                    kp = kb[0, :, : n_full * PAGE, :].reshape(self.Hkv, n_full, PAGE, D).transpose(1, 0, 2, 3)
                    vp = vb[0, :, : n_full * PAGE, :].reshape(self.Hkv, n_full, PAGE, D).transpose(1, 0, 2, 3)
                    base_k_list.append(kp)
                    base_v_list.append(vp)
                    self.pages[b] = list(range(pid, pid + n_full))
                    pid += n_full
                if rem == 0:
                    act_k_list.append(mx.zeros((1, self.Hkv, PAGE, D), mx.bfloat16))
                    act_v_list.append(mx.zeros((1, self.Hkv, PAGE, D), mx.bfloat16))
                else:
                    pad_k = mx.zeros((1, self.Hkv, PAGE - rem, D), mx.bfloat16)
                    act_k_list.append(mx.concatenate([kb[:, :, n_full * PAGE : off, :], pad_k], axis=2))
                    act_v_list.append(mx.concatenate([vb[:, :, n_full * PAGE : off, :], pad_k], axis=2))
            self._n_base = pid
            if base_k_list:
                self.base_K = mx.contiguous(base_k_list[0]) if len(base_k_list) == 1 else mx.concatenate(base_k_list, axis=0)
                self.base_V = mx.contiguous(base_v_list[0]) if len(base_v_list) == 1 else mx.concatenate(base_v_list, axis=0)
            else:
                self.base_K = _DUMMY_BF
                self.base_V = _DUMMY_BF
            self.act_K = mx.concatenate(act_k_list, axis=0).reshape(B * self.Hkv * PAGE, D)
            self.act_V = mx.concatenate(act_v_list, axis=0).reshape(B * self.Hkv * PAGE, D)
            self.ov_K = _DUMMY_BF
            self.ov_V = _DUMMY_BF
            self._ov_cap = 0
            self._ov_used = 0
            self._ov_alt = None
            self._ov_pend = []
            mx.eval(self.base_K, self.base_V, self.act_K, self.act_V)
            self._alt_act = (self.act_K + 0, self.act_V + 0) if self.double else None
            if self._alt_act is not None:
                mx.eval(*self._alt_act)
            self._act_pend = []
        else:
            self.cap = self.max_size if self.ring else max(s[2] for s in st)
            self.K = self._pack(ks)
            self.V = self._pack(vs)
            mx.eval(self.K, self.V)
            self._alt = None
            self._pend = []
            if self.double:
                self._alt = (self.K + 0, self.V + 0)
                mx.eval(*self._alt)

    @property
    def pages_sig(self):
        return tuple(len(p) for p in self.pages) if self.paged else None

    # ---------------------------------------------------------------- flat (ring / fallback) helpers
    def _pack(self, bufs):
        Hkv, cap = self.Hkv, self.cap
        rows = []
        for b in bufs:
            pad = cap - b.shape[2]
            if pad:
                b = mx.concatenate([b, mx.zeros((1, Hkv, pad, D), b.dtype)], axis=2)
            rows.append(b)
        a = rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=0)
        return a.reshape(len(bufs) * Hkv * cap, D)

    def _grow(self, new_cap):
        B, Hkv, cap = self.B, self.Hkv, self.cap
        z = mx.zeros((B, Hkv, new_cap - cap, D), mx.bfloat16)
        g = lambda a: mx.concatenate([a.reshape(B, Hkv, cap, D), z], axis=2).reshape(-1, D)
        self.K, self.V = g(self.K), g(self.V)
        if self._alt is not None:
            self._alt = (self.K + 0, self.V + 0)
            self._pend = []
            mx.eval(*self._alt)
        self.cap = new_cap
        mx.eval(self.K, self.V)

    # ---------------------------------------------------------------- paged overflow helpers
    def _ensure_ov(self, need_extra):
        if self._ov_used + need_extra <= self._ov_cap:
            return
        if self._ov_cap > 0 and self._ov_used > 0:
            u0 = self._ov_used
            done_k = self.ov_K[:u0]
            done_v = self.ov_V[:u0]
            self.base_K = done_k if self._n_base == 0 else mx.concatenate([self.base_K, done_k], axis=0)
            self.base_V = done_v if self._n_base == 0 else mx.concatenate([self.base_V, done_v], axis=0)
            mx.eval(self.base_K, self.base_V)
            self._n_base += u0
            self._ov_used = 0
        new_cap = max(16, self.B * 4, need_extra * 2)
        self.ov_K = mx.zeros((new_cap, self.Hkv, PAGE, D), mx.bfloat16)
        self.ov_V = mx.zeros((new_cap, self.Hkv, PAGE, D), mx.bfloat16)
        self._ov_cap = new_cap
        self._ov_pend = []
        if self.double:
            self._ov_alt = (mx.zeros_like(self.ov_K), mx.zeros_like(self.ov_V))
            mx.eval(self.ov_K, self.ov_V, *self._ov_alt)
        else:
            mx.eval(self.ov_K, self.ov_V)

    # ---------------------------------------------------------------- per-step host metadata
    def plan(self):
        new, wpos, tbl = [], [], np.empty((self.B, _MW), np.int32)
        if self.paged:
            flush_rows = []
            cur_plen = self.pages_sig
            for i, (off, idx, cap) in enumerate(self.state):
                w, off2, idx2, cap2 = _advance_full(off, idx, cap, self.step)
                new.append((off2, idx2, cap2))
                wpos.append(w)
                tbl[i] = (min(off2, cap2), off2, 0, 0)
                if off // PAGE > cur_plen[i]:
                    flush_rows.append(i)
            new_ov_pids = tuple(self._n_base + self._ov_used + j for j in range(len(flush_rows)))
            max_pages = max((o + PAGE - 1) // PAGE for o, _, _ in new)
            cap_pages = max(64, ((max_pages + 31) // 32) * 32)
            pt_np = np.full((self.B, cap_pages), -1, dtype=np.int32)
            fmap = dict(zip(flush_rows, new_ov_pids))
            for b in range(self.B):
                pgs = self.pages[b]
                if pgs:
                    pt_np[b, : len(pgs)] = pgs
                if b in fmap:
                    pt_np[b, len(pgs)] = fmap[b]
            base = (np.arange(self.B * self.Hkv, dtype=np.int64) * PAGE).reshape(self.B, self.Hkv)
            widx = (base + (np.asarray(wpos, np.int64) % PAGE)[:, None]).reshape(-1).astype(np.int32)
            return SimpleNamespace(
                prev=self.state, prev_pages=cur_plen, new=tuple(new), cap=max(c for _, _, c in new),
                mt=mx.array(tbl), pt=mx.array(pt_np), cap_pages=cap_pages, widx=mx.array(widx),
                flush_rows=tuple(flush_rows), new_ov_pids=new_ov_pids,
                lens=tuple(int(t) for t in tbl[:, 0]), paged=True,
            )

        for i, (off, idx, cap) in enumerate(self.state):
            if self.ring:
                w, off2, idx2, cap2 = _advance_ring(off, idx, cap, self.step, self.max_size, 0)
            else:
                w, off2, idx2, cap2 = _advance_full(off, idx, cap, self.step)
            new.append((off2, idx2, cap2))
            wpos.append(w)
            tbl[i] = (min(off2, cap2), off2, idx2 if self.ring else 0, 1 if self.ring else 0)
        need = max(c for _, _, c in new)
        if need > self.cap:
            self._grow(need)
        base = (np.arange(self.B * self.Hkv, dtype=np.int64) * self.cap).reshape(self.B, self.Hkv)
        widx = (base + np.asarray(wpos, np.int64)[:, None]).reshape(-1).astype(np.int32)
        return SimpleNamespace(prev=self.state, prev_pages=None, new=tuple(new), cap=self.cap, mt=mx.array(tbl),
                               widx=mx.array(widx), lens=tuple(int(t) for t in tbl[:, 0]), paged=False)

    def _apply(self, p):
        assert p.prev == self.state, "plan was made for a different row state"
        if self.paged:
            assert p.paged and p.prev_pages == self.pages_sig, "plan pages mismatch"
            self.cap = max(self.cap, p.cap)
        else:
            assert not p.paged
            if p.cap > self.cap:
                self._grow(p.cap)
            assert p.cap == self.cap, "plan was made for a store with a different capacity"
        self.state = p.new
        self._plan = p

    def _apply_planned(self, p):
        if self.paged:
            self.cap = max(self.cap, p.cap)
        self.state = p.new
        self._plan = p

    # ---------------------------------------------------------------- K7: update
    def update(self, k, v, plan=None):
        """k, v: (B, Hkv, D) bf16 (the new token of every row). In place (donated scatter)."""
        assert k.shape == (self.B, self.Hkv, D) and v.shape == k.shape
        assert k.dtype == mx.bfloat16 and v.dtype == mx.bfloat16
        if plan is None:
            plan = self.plan()
            self._apply_planned(plan)
        else:
            self._apply(plan)
        n = self.B * self.Hkv
        k2, v2 = k.reshape(n, D), v.reshape(n, D)

        if self.paged:
            if plan.flush_rows:
                self._ensure_ov(len(plan.flush_rows))
                act_k4 = self.act_K.reshape(self.B, self.Hkv, PAGE, D)
                act_v4 = self.act_V.reshape(self.B, self.Hkv, PAGE, D)
                idx_b = mx.array(plan.flush_rows, mx.int32)
                fk, fv = act_k4[idx_b], act_v4[idx_b]
                ov_slots = mx.array([pid - self._n_base for pid in plan.new_ov_pids], mx.int32)
                if not self.double:
                    self.ov_K[ov_slots] = fk
                    self.ov_V[ov_slots] = fv
                else:
                    ok, ov = self._ov_alt
                    self._ov_alt = (self.ov_K, self.ov_V)
                    if self._ov_pend:
                        si = mx.concatenate([s for s, _, _ in self._ov_pend] + [ov_slots])
                        kk = mx.concatenate([a for _, a, _ in self._ov_pend] + [fk])
                        vv = mx.concatenate([b for _, _, b in self._ov_pend] + [fv])
                    else:
                        si, kk, vv = ov_slots, fk, fv
                    ok[si] = kk
                    ov[si] = vv
                    self.ov_K, self.ov_V = ok, ov
                    self._ov_pend = [(ov_slots, fk, fv)]
                self._ov_used += len(plan.flush_rows)
                for b, pid in zip(plan.flush_rows, plan.new_ov_pids):
                    self.pages[b].append(pid)

            if not self.double:
                self.act_K[plan.widx] = k2
                self.act_V[plan.widx] = v2
                return plan
            ak, av = self._alt_act
            self._alt_act = (self.act_K, self.act_V)
            if self._act_pend:
                wi = mx.concatenate([w for w, _, _ in self._act_pend] + [plan.widx])
                kk = mx.concatenate([a for _, a, _ in self._act_pend] + [k2])
                vv = mx.concatenate([b for _, _, b in self._act_pend] + [v2])
            else:
                wi, kk, vv = plan.widx, k2, v2
            ak[wi] = kk
            av[wi] = vv
            self.act_K, self.act_V = ak, av
            self._act_pend = [(plan.widx, k2, v2)]
            return plan

        if not self.double:
            self.K[plan.widx] = k2
            self.V[plan.widx] = v2
            return plan
        K, V = self._alt
        self._alt = (self.K, self.V)
        if self._pend:
            wi = mx.concatenate([w for w, _, _ in self._pend] + [plan.widx])
            kk = mx.concatenate([a for _, a, _ in self._pend] + [k2])
            vv = mx.concatenate([b for _, _, b in self._pend] + [v2])
        else:
            wi, kk, vv = plan.widx, k2, v2
        K[wi] = kk
        V[wi] = vv
        self.K, self.V = K, V
        self._pend = [(plan.widx, k2, v2)]
        return plan

    # ---------------------------------------------------------------- K8: sdpa
    def sdpa(self, q, pb, sinks, plan=None, *, scale, window, gate=None):
        """q (B, Hq, 1, D) bf16; pb (B, 1, Hq, R) or (B, Hq, R); sinks (B, 1, Hq) or (B, Hq);
        optional gate (B, 1, Hq * D) bf16 to fuse `bf16(out) * sigmoid(gate)` on store.
        Returns (B, 1, Hq, D) bf16, row b bitwise == fastdecode._core (and _attend_post gate if given)."""
        plan = plan or self._plan
        assert plan is not None and plan.new == self.state, "call update() for this step first"
        B, Hkv = self.B, self.Hkv
        assert q.dtype == mx.bfloat16 and q.shape[0] == B and q.shape[2] == 1 and q.shape[3] == D
        Hq = q.shape[1]
        gqa = Hq // Hkv
        assert Hq == gqa * Hkv and 3 <= gqa <= 5, "block count rule implemented for n_simds in 3..5 (blocks=128)"
        max_len = max(plan.lens)
        min_len = min(plan.lens)
        blocks = _blocks(max_len, gqa)
        assert blocks == 128
        R = pb.shape[-1]
        has2 = max_len >= _TWO_PASS_N
        all2 = has2 and min_len >= _TWO_PASS_N
        ns = gqa if all2 else 32
        tgpk = (blocks * gqa + ns - 1) // ns
        ng = Hkv * tgpk if has2 else Hq
        s1, s1_2pass, s2 = _kernel_pair(paged=self.paged)
        key = (float(scale),)
        sc = self._scale.get(key)
        if sc is None:
            sc = self._scale[key] = mx.array([scale], mx.float32)
        has_gate = gate is not None
        gt = mx.contiguous(gate) if has_gate else _DUMMY_GATE
        n2 = B * Hq * blocks if has2 else 1
        qc, pbc, sk = mx.contiguous(q), mx.contiguous(pb), mx.contiguous(sinks)

        if self.paged:
            if all2:
                tmpl = [
                    ("T", mx.bfloat16), ("PBT", pb.dtype), ("MW", _MW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                    ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns),
                    ("N_BASE", self._n_base), ("CAP_PAGES", plan.cap_pages),
                ]
                partials, sums, maxs = s1_2pass(
                    inputs=[qc, pbc, sk, plan.mt, sc, plan.pt,
                            self.act_K, self.act_V, self.base_K, self.base_V, self.ov_K, self.ov_V],
                    template=tmpl, grid=(B * ng * 32 * ns, 1, 1), threadgroup=(32 * ns, 1, 1),
                    output_shapes=[(n2 * D,), (n2,), (n2,)],
                    output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
                )
                o1 = _DUMMY_GATE
            else:
                tmpl = [
                    ("T", mx.bfloat16), ("PBT", pb.dtype), ("MW", _MW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                    ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns), ("HAS2", 1 if has2 else 0),
                    ("HAS_GATE", has_gate), ("N_BASE", self._n_base), ("CAP_PAGES", plan.cap_pages),
                ]
                o1, partials, sums, maxs = s1(
                    inputs=[qc, pbc, sk, plan.mt, sc, gt, plan.pt,
                            self.act_K, self.act_V, self.base_K, self.base_V, self.ov_K, self.ov_V],
                    template=tmpl, grid=(B * ng * 32 * ns, 1, 1), threadgroup=(32 * ns, 1, 1),
                    output_shapes=[(B, 1, Hq, D), (n2 * D if has2 else 1,), (n2,), (n2,)],
                    output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
                )
        else:
            if all2:
                tmpl = [
                    ("T", mx.bfloat16), ("PBT", pb.dtype), ("MW", _MW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                    ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns), ("CAP", self.cap),
                ]
                partials, sums, maxs = s1_2pass(
                    inputs=[qc, pbc, sk, plan.mt, sc, self.K, self.V],
                    template=tmpl, grid=(B * ng * 32 * ns, 1, 1), threadgroup=(32 * ns, 1, 1),
                    output_shapes=[(n2 * D,), (n2,), (n2,)],
                    output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
                )
                o1 = _DUMMY_GATE
            else:
                tmpl = [
                    ("T", mx.bfloat16), ("PBT", pb.dtype), ("MW", _MW), ("HQ", Hq), ("HKV", Hkv), ("GQA", gqa),
                    ("BLOCKS", blocks), ("R", R), ("WIN", int(window)), ("NS", ns), ("CAP", self.cap),
                    ("HAS2", 1 if has2 else 0), ("HAS_GATE", has_gate),
                ]
                o1, partials, sums, maxs = s1(
                    inputs=[qc, pbc, sk, plan.mt, sc, gt, self.K, self.V],
                    template=tmpl, grid=(B * ng * 32 * ns, 1, 1), threadgroup=(32 * ns, 1, 1),
                    output_shapes=[(B, 1, Hq, D), (n2 * D if has2 else 1,), (n2,), (n2,)],
                    output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
                )
        if not has2:
            return o1
        (out,) = s2(
            inputs=[partials, sums, maxs, o1, plan.mt, gt],
            template=[("T", mx.bfloat16), ("MW", _MW), ("HQ", Hq), ("BLOCKS", blocks), ("HAS_GATE", has_gate)],
            grid=(B * Hq * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(B, 1, Hq, D)],
            output_dtypes=[mx.bfloat16],
        )
        return out

    # ---------------------------------------------------------------- write back
    def flush(self, i):
        """Write row i's state into its own cache object, as if that cache had run alone."""
        c = self.caches[i]
        off, idx, cap = self.state[i]
        if self.paged:
            parts = []
            for pid in self.pages[i]:
                src = self.base_K[pid : pid + 1] if pid < self._n_base else self.ov_K[pid - self._n_base : pid - self._n_base + 1]
                parts.append(src.transpose(1, 0, 2, 3).reshape(1, self.Hkv, PAGE, D))
            rem = off - len(self.pages[i]) * PAGE
            if rem > 0:
                act_k4 = self.act_K.reshape(self.B, self.Hkv, PAGE, D)
                parts.append(act_k4[i : i + 1, :, :rem, :])
            if cap > off:
                parts.append(mx.zeros((1, self.Hkv, cap - off, D), mx.bfloat16))
            c.keys = mx.contiguous(parts[0]) if len(parts) == 1 else mx.concatenate(parts, axis=2)

            parts_v = []
            for pid in self.pages[i]:
                src = self.base_V[pid : pid + 1] if pid < self._n_base else self.ov_V[pid - self._n_base : pid - self._n_base + 1]
                parts_v.append(src.transpose(1, 0, 2, 3).reshape(1, self.Hkv, PAGE, D))
            if rem > 0:
                act_v4 = self.act_V.reshape(self.B, self.Hkv, PAGE, D)
                parts_v.append(act_v4[i : i + 1, :, :rem, :])
            if cap > off:
                parts_v.append(mx.zeros((1, self.Hkv, cap - off, D), mx.bfloat16))
            c.values = mx.contiguous(parts_v[0]) if len(parts_v) == 1 else mx.concatenate(parts_v, axis=2)
            c.offset = off
            mx.eval(c.keys, c.values)
            return c

        K4 = self.K.reshape(self.B, self.Hkv, self.cap, D)
        V4 = self.V.reshape(self.B, self.Hkv, self.cap, D)
        c.keys = mx.contiguous(K4[i : i + 1, :, :cap, :])
        c.values = mx.contiguous(V4[i : i + 1, :, :cap, :])
        c.offset = off
        if self.ring:
            c._idx = idx
        mx.eval(c.keys, c.values)
        return c

    def to_caches(self):
        return [self.flush(i) for i in range(self.B)]


PagedRowKV = RowKV
