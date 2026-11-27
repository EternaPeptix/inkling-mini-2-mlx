"""Fused LM head tail kernel: bf16 softcap -> logsumexp -> (softcapped - lse) argmax (Phase 3b).

Avoids materializing (B, V) f32 tensors for softcap and logsumexp and launches 2 kernels instead of
cast + div + tanh + mul + slice + C*(logsumexp + sub + argmax) + concat.
Matches MLX's `tanh(logits.f32 / sc) * sc`, `logsumexp_looped` (rows/glue.py), and `argmax` bitwise.
"""
import os

import mlx.core as mx

from .glue import GLUE_HEADER, flit

__all__ = [
    "softcap_argmax_rows",
    "softcap_logprobs_rows",
    "softcap_lse_rows",
    "softcap_sample_rows",
    "softcap_sample_rows_nosync",
    "softcap_coupled_verify_rows",
]

_HEAD_TG_LSE = int(os.environ.get("HEAD_TG_LSE", "1024"))
_HEAD_TG_ARG = int(os.environ.get("HEAD_TG_ARG", "1024"))
_HEAD_SPLIT = os.environ.get("HEAD_SPLIT", "")
_HEAD_FAST_ARG = os.environ.get("HEAD_FAST_ARG", "1") not in ("0", "")

_HEAD_HEADER = GLUE_HEADER + r"""
template <bool HAS_SC>
inline float head_sc(bfloat16_t v, float sc) {
  GL_NOFMA
  float x = float(v);
  return HAS_SC ? (metal::precise::tanh(x / sc) * sc) : x;
}

template <int TG, bool HAS_SC, bool ALIGN4 = true>
inline float head_logsumexp_bf16(const device bfloat16_t* in, int n, float sc, uint lid,
                                 threadgroup float* lmax, threadgroup float* lnrm) {
  GL_NOFMA
  constexpr int VT = 1024, NV = VT / TG, RS = TG / 32;
  uint lane = lid & 31, sg = lid >> 5;
  float nrm[NV], gmx[NV];
  int nr = (n + 4 * VT - 1) / (4 * VT);
  for (int k = 0; k < NV; k++) {
    int g = int(sg) + k * RS;
    int v = g * 32 + int(lane);
    float prevmax;
    float maxval = -3.4028234663852886e+38f;
    float normalizer = 0.0f;
    for (int r = 0; r < nr; r++) {
      int offset = r * VT * 4 + v * 4;
      float vals[4];
      if (offset + 4 <= n) {
        const device bfloat16_t* p = in + offset;
        if constexpr (ALIGN4) {
          bfloat4 bv = as_type<bfloat4>(((const device uint2*)p)[0]);
          vals[0] = head_sc<HAS_SC>(bv.x, sc);
          vals[1] = head_sc<HAS_SC>(bv.y, sc);
          vals[2] = head_sc<HAS_SC>(bv.z, sc);
          vals[3] = head_sc<HAS_SC>(bv.w, sc);
        } else {
          for (int i = 0; i < 4; i++) vals[i] = head_sc<HAS_SC>(p[i], sc);
        }
      } else {
        for (int i = 0; i < 4; i++) vals[i] = (offset + i < n) ? head_sc<HAS_SC>(in[offset + i], sc) : -INFINITY;
      }
      prevmax = maxval;
      for (int i = 0; i < 4; i++) maxval = (maxval < vals[i]) ? vals[i] : maxval;
      normalizer *= metal::fast::exp(prevmax - maxval);
      for (int i = 0; i < 4; i++) normalizer += metal::fast::exp(vals[i] - maxval);
    }
    prevmax = maxval;
    maxval = simd_max(maxval);
    normalizer *= metal::fast::exp(prevmax - maxval);
    normalizer = simd_sum(normalizer);
    gmx[k] = maxval;
    nrm[k] = normalizer;
    if (lane == 0) lmax[g] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gm = simd_max(lmax[lane]);
  for (int k = 0; k < NV; k++) {
    float nn = nrm[k] * metal::fast::exp(gmx[k] - gm);
    if (lane == 0) lnrm[int(sg) + k * RS] = nn;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = simd_sum(lnrm[lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return metal::isinf(gm) ? gm : (metal::log(tot) + gm);
}
"""

_LSE_SRC = r"""
    uint b = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    threadgroup float lmax[32];
    threadgroup float lnrm[32];
    float out_v = head_logsumexp_bf16<TG, HAS_SC, (PP & 3) == 0>(logits + size_t(b) * PP, VV, SC, lid, lmax, lnrm);
    if (lid == 0) lse[b] = out_v;
"""

_ARG1_SRC = r"""
    uint b = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31, sg = lid >> 5;
    threadgroup GlueIV buf[32];
    const device bfloat16_t* row = logits + size_t(b) * PP;
    float sub = lse[b];
    GlueIV best;
    best.val = -INFINITY;
    best.idx = 0;
    for (int i = int(lid); i < VV; i += TG) {
      GlueIV c;
      c.val = head_sc<HAS_SC>(row[i], SC) - sub;
      c.idx = uint(i);
      best = glue_iv_reduce(best, c);
    }
    for (uint d = 16; d > 0; d /= 2) {
      GlueIV o;
      o.val = simd_shuffle_down(best.val, d);
      o.idx = simd_shuffle_down(best.idx, d);
      best = glue_iv_reduce(best, o);
    }
    if (lane == 0) buf[sg] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    GlueIV f;
    f.val = -INFINITY;
    f.idx = 0;
    if (lane < TG / 32) f = buf[lane];
    for (uint d = 16; d > 0; d /= 2) {
      GlueIV o;
      o.val = simd_shuffle_down(f.val, d);
      o.idx = simd_shuffle_down(f.idx, d);
      f = glue_iv_reduce(f, o);
    }
    if (lid == 0) out[b] = f.idx;
"""

_ARGS_P1_SRC = r"""
    uint b = threadgroup_position_in_grid.y;
    uint sp = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31, sg = lid >> 5;
    threadgroup GlueIV buf[32];
    const device bfloat16_t* row = logits + size_t(b) * PP;
    float sub = lse[b];
    const int chunk = (VV + SP - 1) / SP;
    const int s0 = int(sp) * chunk;
    const int s1 = min(VV, s0 + chunk);
    GlueIV best;
    best.val = -INFINITY;
    best.idx = 0;
    for (int i = s0 + int(lid); i < s1; i += TG) {
      GlueIV c;
      c.val = head_sc<HAS_SC>(row[i], SC) - sub;
      c.idx = uint(i);
      best = glue_iv_reduce(best, c);
    }
    for (uint d = 16; d > 0; d /= 2) {
      GlueIV o;
      o.val = simd_shuffle_down(best.val, d);
      o.idx = simd_shuffle_down(best.idx, d);
      best = glue_iv_reduce(best, o);
    }
    if (lane == 0) buf[sg] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    GlueIV f;
    f.val = -INFINITY;
    f.idx = 0;
    if (lane < TG / 32) f = buf[lane];
    for (uint d = 16; d > 0; d /= 2) {
      GlueIV o;
      o.val = simd_shuffle_down(f.val, d);
      o.idx = simd_shuffle_down(f.idx, d);
      f = glue_iv_reduce(f, o);
    }
    if (lid == 0) {
      pval[b * SP + sp] = f.val;
      pidx[b * SP + sp] = f.idx;
    }
"""

_ARGS_P2_SRC = r"""
    uint b = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    GlueIV f;
    f.val = -INFINITY;
    f.idx = 0;
    if ((int)lane < SP) {
      f.val = pval[b * SP + lane];
      f.idx = pidx[b * SP + lane];
    }
    for (uint d = 16; d > 0; d /= 2) {
      GlueIV o;
      o.val = simd_shuffle_down(f.val, d);
      o.idx = simd_shuffle_down(f.idx, d);
      f = glue_iv_reduce(f, o);
    }
    if (lane == 0) out[b] = f.idx;
"""

_FAST_ARG_P1_SRC = r"""
    uint sp = threadgroup_position_in_grid.x;
    uint b = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31, sg = lid >> 5;
    const device bfloat16_t* row = logits + size_t(b) * PP;
    const int chunk = (((VV + SP - 1) / SP) + 7) & ~7;
    const int s0 = int(sp) * chunk;
    const int s1 = min(VV, s0 + chunk);
    float bval = -INFINITY, sval = -INFINITY;
    uint bidx = 0xFFFFFFFFu, nanf = 0u;
    if constexpr ((PP & 7) == 0) {
      int i = s0 + int(lid) * 8;
      for (; i + 8 <= s1; i += TG * 8) {
        uint4 pk = ((const device uint4*)(row + i))[0];
        bfloat4 v0 = as_type<bfloat4>(pk.xy), v1 = as_type<bfloat4>(pk.zw);
        float vs[8] = { float(v0.x), float(v0.y), float(v0.z), float(v0.w), float(v1.x), float(v1.y), float(v1.z), float(v1.w) };
        #pragma unroll
        for (int k = 0; k < 8; k++) {
          float v = vs[k];
          uint idx = uint(i + k);
          if (metal::isnan(v)) { nanf = 1u; }
          else if (v > bval || (v == bval && idx < bidx)) { sval = bval; bval = v; bidx = idx; }
          else if (v > sval) { sval = v; }
        }
      }
      for (; i < s1; i++) {
        float v = float(row[i]);
        if (metal::isnan(v)) { nanf = 1u; }
        else if (v > bval || (v == bval && uint(i) < bidx)) { sval = bval; bval = v; bidx = uint(i); }
        else if (v > sval) { sval = v; }
      }
    } else {
      for (int i = s0 + int(lid); i < s1; i += TG) {
        float v = float(row[i]);
        if (metal::isnan(v)) { nanf = 1u; }
        else if (v > bval || (v == bval && uint(i) < bidx)) { sval = bval; bval = v; bidx = uint(i); }
        else if (v > sval) { sval = v; }
      }
    }
    for (uint d = 16; d > 0; d /= 2) {
      float ov = simd_shuffle_down(bval, d);
      uint oi = simd_shuffle_down(bidx, d);
      float os = simd_shuffle_down(sval, d);
      nanf |= simd_shuffle_down(nanf, d);
      if (ov > bval || (ov == bval && oi < bidx)) { sval = max(bval, os); bval = ov; bidx = oi; }
      else { sval = max(sval, ov); }
    }
    threadgroup float sb[8], ss[8];
    threadgroup uint si[8], sn[8];
    if (lane == 0) { sb[sg] = bval; si[sg] = bidx; ss[sg] = sval; sn[sg] = nanf; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float fv = -INFINITY, fs = -INFINITY;
      uint fi = 0xFFFFFFFFu, fn = 0u;
      if (lane < TG / 32) { fv = sb[lane]; fi = si[lane]; fs = ss[lane]; fn = sn[lane]; }
      for (uint d = 16; d > 0; d /= 2) {
        float ov = simd_shuffle_down(fv, d);
        uint oi = simd_shuffle_down(fi, d);
        float os = simd_shuffle_down(fs, d);
        fn |= simd_shuffle_down(fn, d);
        if (ov > fv || (ov == fv && oi < fi)) { fs = max(fv, os); fv = ov; fi = oi; }
        else { fs = max(fs, ov); }
      }
      if (lane == 0) {
        size_t o = size_t(b) * SP + sp;
        pbv[o] = fv; pbi[o] = fi; psv[o] = fs; pnf[o] = fn;
      }
    }
"""

_FAST_ARG_P2_SRC = r"""
    uint b = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31, sg = lid >> 5;
    threadgroup uint need_fb[1];
    if (sg == 0) {
      float fv = -INFINITY, fs = -INFINITY;
      uint fi = 0xFFFFFFFFu, fn = 0u;
      for (int i = int(lane); i < SP; i += 32) {
        size_t o = size_t(b) * SP + uint(i);
        float ov = pbv[o];
        uint oi = pbi[o];
        float os = psv[o];
        fn |= pnf[o];
        if (ov > fv || (ov == fv && oi < fi)) { fs = max(fv, os); fv = ov; fi = oi; }
        else { fs = max(fs, ov); }
      }
      for (uint d = 16; d > 0; d /= 2) {
        float ov = simd_shuffle_down(fv, d);
        uint oi = simd_shuffle_down(fi, d);
        float os = simd_shuffle_down(fs, d);
        fn |= simd_shuffle_down(fn, d);
        if (ov > fv || (ov == fv && oi < fi)) { fs = max(fv, os); fv = ov; fi = oi; }
        else { fs = max(fs, ov); }
      }
      if (lane == 0) {
        float b_sc = head_sc<HAS_SC>(bfloat16_t(fv), SC);
        float s_sc = (fs == -INFINITY) ? -INFINITY : head_sc<HAS_SC>(bfloat16_t(fs), SC);
        float gap = (!HAS_SC && metal::abs(b_sc) > 100.0f) ? (metal::abs(b_sc) * 1e-5f) : 1e-3f;
        bool ok = (fn == 0u) && metal::isfinite(b_sc) && (s_sc == -INFINITY || (b_sc - s_sc) > gap);
        need_fb[0] = ok ? 0u : 1u;
        if (ok) out[b] = fi;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!need_fb[0]) return;
    const device bfloat16_t* row = logits + size_t(b) * PP;
    threadgroup float lmax[32], lnrm[32];
    float sub = head_logsumexp_bf16<256, HAS_SC, (PP & 3) == 0>(row, VV, SC, lid, lmax, lnrm);
    GlueIV best;
    best.val = -INFINITY;
    best.idx = 0;
    for (int i = int(lid); i < VV; i += 256) {
      GlueIV c;
      c.val = head_sc<HAS_SC>(row[i], SC) - sub;
      c.idx = uint(i);
      best = glue_iv_reduce(best, c);
    }
    for (uint d = 16; d > 0; d /= 2) {
      GlueIV o;
      o.val = simd_shuffle_down(best.val, d);
      o.idx = simd_shuffle_down(best.idx, d);
      best = glue_iv_reduce(best, o);
    }
    threadgroup GlueIV buf[8];
    if (lane == 0) buf[sg] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      GlueIV f;
      f.val = -INFINITY;
      f.idx = 0;
      if (lane < 8) f = buf[lane];
      for (uint d = 16; d > 0; d /= 2) {
        GlueIV o;
        o.val = simd_shuffle_down(f.val, d);
        o.idx = simd_shuffle_down(f.idx, d);
        f = glue_iv_reduce(f, o);
      }
      if (lane == 0) out[b] = f.idx;
    }
"""

_LSE32_P1_SRC = r"""
    GL_NOFMA
    uint g = threadgroup_position_in_grid.x;
    uint b = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    const device bfloat16_t* in = logits + size_t(b) * PP;
    constexpr int VT = 1024;
    int nr = (VV + 4 * VT - 1) / (4 * VT);
    int v = int(g) * 32 + int(lane);
    float prevmax;
    float maxval = -3.4028234663852886e+38f;
    float normalizer = 0.0f;
    for (int r = 0; r < nr; r++) {
      int offset = r * VT * 4 + v * 4;
      float vals[4];
      if (offset + 4 <= VV) {
        const device bfloat16_t* p = in + offset;
        if constexpr ((PP & 3) == 0) {
          bfloat4 bv = as_type<bfloat4>(((const device uint2*)p)[0]);
          vals[0] = head_sc<HAS_SC>(bv.x, SC);
          vals[1] = head_sc<HAS_SC>(bv.y, SC);
          vals[2] = head_sc<HAS_SC>(bv.z, SC);
          vals[3] = head_sc<HAS_SC>(bv.w, SC);
        } else {
          for (int i = 0; i < 4; i++) vals[i] = head_sc<HAS_SC>(p[i], SC);
        }
      } else {
        for (int i = 0; i < 4; i++) vals[i] = (offset + i < VV) ? head_sc<HAS_SC>(in[offset + i], SC) : -INFINITY;
      }
      prevmax = maxval;
      for (int i = 0; i < 4; i++) maxval = (maxval < vals[i]) ? vals[i] : maxval;
      normalizer *= metal::fast::exp(prevmax - maxval);
      for (int i = 0; i < 4; i++) normalizer += metal::fast::exp(vals[i] - maxval);
    }
    prevmax = maxval;
    maxval = simd_max(maxval);
    normalizer *= metal::fast::exp(prevmax - maxval);
    normalizer = simd_sum(normalizer);
    if (lane == 0) {
      gmax[b * 32 + g] = maxval;
      gnrm[b * 32 + g] = normalizer;
    }
"""

_LP32_ARG_P1_SRC = r"""
    GL_NOFMA
    uint sp = threadgroup_position_in_grid.x;
    uint b = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31, sg = lid >> 5;
    float gmv = gmax[b * 32 + lane];
    float gm = simd_max(gmv);
    float nn = gnrm[b * 32 + lane] * metal::fast::exp(gmv - gm);
    float tot = simd_sum(nn);
    float sub = metal::isinf(gm) ? gm : (metal::log(tot) + gm);
    const device bfloat16_t* row = logits + size_t(b) * PP;
    device float* lp_row = lp + size_t(b) * VV;
    const int chunk = (((VV + SP - 1) / SP) + 3) & ~3;
    const int s0 = int(sp) * chunk;
    const int s1 = min(VV, s0 + chunk);
    const bool lp_aligned = ((size_t(b) * uint(VV)) & 3u) == 0u;
    if constexpr (!DO_ARG) {
      if constexpr ((PP & 3) == 0) {
        int i = s0 + int(lid) * 4;
        for (; i + 4 <= s1; i += TG * 4) {
          bfloat4 bv = as_type<bfloat4>(((const device uint2*)(row + i))[0]);
          float4 fv = float4(head_sc<HAS_SC>(bv.x, SC) - sub, head_sc<HAS_SC>(bv.y, SC) - sub,
                             head_sc<HAS_SC>(bv.z, SC) - sub, head_sc<HAS_SC>(bv.w, SC) - sub);
          if (lp_aligned) { ((device float4*)(lp_row + i))[0] = fv; }
          else { lp_row[i] = fv.x; lp_row[i + 1] = fv.y; lp_row[i + 2] = fv.z; lp_row[i + 3] = fv.w; }
        }
        for (; i < s1; i++) lp_row[i] = head_sc<HAS_SC>(row[i], SC) - sub;
      } else {
        for (int i = s0 + int(lid); i < s1; i += TG) lp_row[i] = head_sc<HAS_SC>(row[i], SC) - sub;
      }
      return;
    }
    threadgroup GlueIV buf[32];
    GlueIV best;
    best.val = -INFINITY;
    best.idx = 0;
    if constexpr ((PP & 3) == 0) {
      int i = s0 + int(lid) * 4;
      for (; i + 4 <= s1; i += TG * 4) {
        bfloat4 bv = as_type<bfloat4>(((const device uint2*)(row + i))[0]);
        float4 fv = float4(head_sc<HAS_SC>(bv.x, SC) - sub, head_sc<HAS_SC>(bv.y, SC) - sub,
                           head_sc<HAS_SC>(bv.z, SC) - sub, head_sc<HAS_SC>(bv.w, SC) - sub);
        if (lp_aligned) { ((device float4*)(lp_row + i))[0] = fv; }
        else { lp_row[i] = fv.x; lp_row[i + 1] = fv.y; lp_row[i + 2] = fv.z; lp_row[i + 3] = fv.w; }
        GlueIV c;
        c.val = fv.x; c.idx = uint(i + 0); best = glue_iv_reduce(best, c);
        c.val = fv.y; c.idx = uint(i + 1); best = glue_iv_reduce(best, c);
        c.val = fv.z; c.idx = uint(i + 2); best = glue_iv_reduce(best, c);
        c.val = fv.w; c.idx = uint(i + 3); best = glue_iv_reduce(best, c);
      }
      for (; i < s1; i++) {
        float v = head_sc<HAS_SC>(row[i], SC) - sub;
        lp_row[i] = v;
        GlueIV c; c.val = v; c.idx = uint(i); best = glue_iv_reduce(best, c);
      }
    } else {
      for (int i = s0 + int(lid); i < s1; i += TG) {
        float v = head_sc<HAS_SC>(row[i], SC) - sub;
        lp_row[i] = v;
        GlueIV c;
        c.val = v;
        c.idx = uint(i);
        best = glue_iv_reduce(best, c);
      }
    }
    for (uint d = 16; d > 0; d /= 2) {
      GlueIV o;
      o.val = simd_shuffle_down(best.val, d);
      o.idx = simd_shuffle_down(best.idx, d);
      best = glue_iv_reduce(best, o);
    }
    if (lane == 0) buf[sg] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    GlueIV f;
    f.val = -INFINITY;
    f.idx = 0;
    if (lane < TG / 32) f = buf[lane];
    for (uint d = 16; d > 0; d /= 2) {
      GlueIV o;
      o.val = simd_shuffle_down(f.val, d);
      o.idx = simd_shuffle_down(f.idx, d);
      f = glue_iv_reduce(f, o);
    }
    if (lid == 0) {
      pval[b * SP + sp] = f.val;
      pidx[b * SP + sp] = f.idx;
    }
"""

_LP32_ARG_P2_SRC = r"""
    uint b = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    GlueIV f;
    f.val = -INFINITY;
    f.idx = 0;
    for (int i = int(lane); i < SP; i += 32) {
      GlueIV c;
      c.val = pval[b * SP + i];
      c.idx = pidx[b * SP + i];
      f = glue_iv_reduce(f, c);
    }
    for (uint d = 16; d > 0; d /= 2) {
      GlueIV o;
      o.val = simd_shuffle_down(f.val, d);
      o.idx = simd_shuffle_down(f.idx, d);
      f = glue_iv_reduce(f, o);
    }
    if (lane == 0) out[b] = f.idx;
"""

_kernels = {}


def _get(kind, has_sc, sc_lit):
    key = (kind, has_sc, sc_lit)
    k = _kernels.get(key)
    if k is not None:
        return k
    sc_def = f"\n#define SC ({sc_lit})\n"
    h = abs(hash(sc_lit)) % 10**6
    if kind == "lse":
        k = mx.fast.metal_kernel(
            name=f"head_lse_{int(has_sc)}_{h}",
            input_names=["logits"], output_names=["lse"],
            source=sc_def + _LSE_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "arg1":
        k = mx.fast.metal_kernel(
            name=f"head_arg1_{int(has_sc)}_{h}",
            input_names=["logits", "lse"], output_names=["out"],
            source=sc_def + _ARG1_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "argsp1":
        k = mx.fast.metal_kernel(
            name=f"head_argsp1_{int(has_sc)}_{h}",
            input_names=["logits", "lse"], output_names=["pval", "pidx"],
            source=sc_def + _ARGS_P1_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "argsp2":
        k = mx.fast.metal_kernel(
            name="head_argsp2",
            input_names=["pval", "pidx"], output_names=["out"],
            source=_ARGS_P2_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "farg1":
        k = mx.fast.metal_kernel(
            name="head_farg1",
            input_names=["logits"], output_names=["pbv", "pbi", "psv", "pnf"],
            source=_FAST_ARG_P1_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "farg2":
        k = mx.fast.metal_kernel(
            name=f"head_farg2_{int(has_sc)}_{h}",
            input_names=["logits", "pbv", "pbi", "psv", "pnf"], output_names=["out"],
            source=sc_def + _FAST_ARG_P2_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "lse32":
        k = mx.fast.metal_kernel(
            name=f"head_lse32_{int(has_sc)}_{h}",
            input_names=["logits"], output_names=["gmax", "gnrm"],
            source=sc_def + _LSE32_P1_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "lp32_arg":
        k = mx.fast.metal_kernel(
            name=f"head_lp32_arg_{int(has_sc)}_{h}",
            input_names=["logits", "gmax", "gnrm"], output_names=["lp", "pval", "pidx"],
            source=sc_def + _LP32_ARG_P1_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "lp32_only":
        k = mx.fast.metal_kernel(
            name=f"head_lp32_only_{int(has_sc)}_{h}",
            input_names=["logits", "gmax", "gnrm"], output_names=["lp", "pval", "pidx"],
            source=sc_def + _LP32_ARG_P1_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "lp32_p2":
        k = mx.fast.metal_kernel(
            name="head_lp32_p2",
            input_names=["pval", "pidx"], output_names=["out"],
            source=_LP32_ARG_P2_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    _kernels[key] = k
    return k


def _norm_logits(logits, unpadded_vocab):
    if logits.dtype != mx.bfloat16:
        raise ValueError(f"softcap_argmax_rows requires bfloat16 logits, got {logits.dtype}")
    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits.reshape(logits.shape[0], logits.shape[2])
    elif logits.ndim == 3:
        logits = logits[:, -1, :]
    elif logits.ndim != 2:
        raise ValueError(f"logits must be (B, P) or (B, T, P), got {logits.shape}")
    B, P = logits.shape
    V = P if unpadded_vocab is None else int(unpadded_vocab)
    if V <= 0 or V > P:
        raise ValueError(f"invalid unpadded_vocab={V} for P={P}")
    return logits, B, P, V


def softcap_lse_rows(logits, softcap, unpadded_vocab=None, *, tg=None):
    """Return per-row exact logsumexp (B,) f32 of softcapped logits[..., :unpadded_vocab]."""
    logits, B, P, V = _norm_logits(logits, unpadded_vocab)
    has_sc = softcap is not None and float(softcap) > 0.0
    sc_lit = flit(float(softcap)) if has_sc else "1.0f"
    tg = int(tg or _HEAD_TG_LSE)
    assert tg in (256, 512, 1024), f"HEAD_TG_LSE must divide 1024 (>=256), got {tg}"
    kl = _get("lse", has_sc, sc_lit)
    (lse,) = kl(
        inputs=[logits],
        template=[("HAS_SC", has_sc), ("PP", P), ("VV", V), ("TG", tg)],
        grid=(B * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B,)],
        output_dtypes=[mx.float32],
    )
    return lse


def softcap_logprobs_rows(logits, softcap, unpadded_vocab=None, *, argmax=False):
    """Exact 32-TG+80-TG fused tail returning logprobs (B, V) f32, or (toks (B,) u32, logprobs (B, V) f32)."""
    logits, B, P, V = _norm_logits(logits, unpadded_vocab)
    has_sc = softcap is not None and float(softcap) > 0.0
    sc_lit = flit(float(softcap)) if has_sc else "1.0f"
    kl = _get("lse32", has_sc, sc_lit)
    gmax, gnrm = kl(
        inputs=[logits],
        template=[("HAS_SC", has_sc), ("PP", P), ("VV", V)],
        grid=(32 * 32, B, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, 32), (B, 32)],
        output_dtypes=[mx.float32, mx.float32],
    )
    sp = 80 if B == 1 else max(16, 80 // B)
    tg = 128
    kp1 = _get("lp32_arg" if argmax else "lp32_only", has_sc, sc_lit)
    p_shp = (B, sp) if argmax else (1, 1)
    lp, pv, pi = kp1(
        inputs=[logits, gmax, gnrm],
        template=[("HAS_SC", has_sc), ("PP", P), ("VV", V), ("TG", tg), ("SP", sp), ("DO_ARG", bool(argmax))],
        grid=(sp * tg, B, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, V), p_shp, p_shp],
        output_dtypes=[mx.float32, mx.float32, mx.uint32],
    )
    if not argmax:
        return lp
    kp2 = _get("lp32_p2", False, "1.0f")
    (out,) = kp2(
        inputs=[pv, pi],
        template=[("SP", sp)],
        grid=(B * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B,)],
        output_dtypes=[mx.uint32],
    )
    return out, lp


def softcap_argmax_rows(logits, softcap, unpadded_vocab=None, *, tg_lse=None, tg_arg=None, split=None):
    """Exact fused tail: (logits -> softcap -> logsumexp -> logits_sc - lse -> argmax) returning (B,) uint32."""
    logits, B, P, V = _norm_logits(logits, unpadded_vocab)
    has_sc = softcap is not None and float(softcap) > 0.0
    sc_lit = flit(float(softcap)) if has_sc else "1.0f"
    use_fast = (split is None and _HEAD_SPLIT in ("0", "") and tg_lse is None and tg_arg is None and _HEAD_FAST_ARG)
    if use_fast:
        sp = 80 if B == 1 else max(16, 80 // B)
        tg = 128
        kf1 = _get("farg1", False, "1.0f")
        pbv, pbi, psv, pnf = kf1(
            inputs=[logits],
            template=[("PP", P), ("VV", V), ("SP", sp), ("TG", tg)],
            grid=(sp * tg, B, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(B, sp), (B, sp), (B, sp), (B, sp)],
            output_dtypes=[mx.float32, mx.uint32, mx.float32, mx.uint32],
        )
        kf2 = _get("farg2", has_sc, sc_lit)
        (out,) = kf2(
            inputs=[logits, pbv, pbi, psv, pnf],
            template=[("HAS_SC", has_sc), ("PP", P), ("VV", V), ("SP", sp)],
            grid=(B * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B,)],
            output_dtypes=[mx.uint32],
        )
        return out
    lse = softcap_lse_rows(logits, softcap, V, tg=tg_lse)
    sp = int((_HEAD_SPLIT or "1") if split is None else split)
    if sp <= 1:
        tg = int(tg_arg or _HEAD_TG_ARG)
        ka = _get("arg1", has_sc, sc_lit)
        (out,) = ka(
            inputs=[logits, lse],
            template=[("HAS_SC", has_sc), ("PP", P), ("VV", V), ("TG", tg)],
            grid=(B * tg, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(B,)],
            output_dtypes=[mx.uint32],
        )
        return out
    assert sp <= 32
    tg = int(tg_arg or 256)
    kp1 = _get("argsp1", has_sc, sc_lit)
    pval, pidx = kp1(
        inputs=[logits, lse],
        template=[("HAS_SC", has_sc), ("PP", P), ("VV", V), ("TG", tg), ("SP", sp)],
        grid=(sp * tg, B, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, sp), (B, sp)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    kp2 = _get("argsp2", False, "1.0f")
    (out,) = kp2(
        inputs=[pval, pidx],
        template=[("SP", sp)],
        grid=(B * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B,)],
        output_dtypes=[mx.uint32],
    )
    return out


# ---------------------------------------------------------------------------------------------------- Fused Sparse-Nucleus Top-P + Gumbel-Max & Coupled Speculative Verify
# Reads bf16 logits twice total (lse32 + sparse_cand_p1). Only tokens with lp >= -10.5f (~100 tokens
# out of 200,058 on real LLM outputs) enter SRAM for O(K log K) exact top-p/min-p + Gumbel-max/coupled sampling.
_SAMP_SPARSE_P1_SRC = r"""
    GL_NOFMA
    const float MINP = s_cfg[1];
    uint sp = threadgroup_position_in_grid.x;
    uint b = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31, sg = lid >> 5;
    float gmv = gmax[b * 32 + lane];
    float gm = simd_max(gmv);
    float nn = gnrm[b * 32 + lane] * metal::fast::exp(gmv - gm);
    float tot = simd_sum(nn);
    float sub = metal::isinf(gm) ? gm : (metal::log(tot) + gm);
    float lp_max = gm - sub;
    float scan_thr = max(-10.5f, (MINP > 0.0f) ? (lp_max + metal::log(MINP)) : -1e30f);
    scan_thr = min(scan_thr, lp_max - 1e-5f);

    threadgroup atomic_uint tg_cnt;
    if (lid == 0) atomic_store_explicit(&tg_cnt, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device bfloat16_t* row = logits + size_t(b) * PP;
    const int chunk = (((VV + SP - 1) / SP) + 3) & ~3;
    const int s0 = int(sp) * chunk;
    const int s1 = min(VV, s0 + chunk);
    size_t dst_base = (size_t(b) * SP + sp) * MAX_CSP;
    float mass_loc = 0.0f;
    if constexpr ((PP & 3) == 0) {
      int i = s0 + int(lid) * 4;
      for (; i + 4 <= s1; i += TG * 4) {
        bfloat4 bv = as_type<bfloat4>(((const device uint2*)(row + i))[0]);
        float lps4[4] = {
          head_sc<HAS_SC>(bv.x, SC) - sub,
          head_sc<HAS_SC>(bv.y, SC) - sub,
          head_sc<HAS_SC>(bv.z, SC) - sub,
          head_sc<HAS_SC>(bv.w, SC) - sub
        };
        #pragma unroll
        for (int k = 0; k < 4; k++) {
          float lp = lps4[k];
          if (lp >= scan_thr) {
            mass_loc += metal::fast::exp(lp);
            uint pos = atomic_fetch_add_explicit(&tg_cnt, 1u, memory_order_relaxed);
            if (pos < uint(MAX_CSP)) {
              cand_lp[dst_base + pos] = lp;
              cand_id[dst_base + pos] = uint(i + k);
            }
          }
        }
      }
      for (; i < s1; i++) {
        float lp = head_sc<HAS_SC>(row[i], SC) - sub;
        if (lp >= scan_thr) {
          mass_loc += metal::fast::exp(lp);
          uint pos = atomic_fetch_add_explicit(&tg_cnt, 1u, memory_order_relaxed);
          if (pos < uint(MAX_CSP)) {
            cand_lp[dst_base + pos] = lp;
            cand_id[dst_base + pos] = uint(i);
          }
        }
      }
    } else {
      for (int i = s0 + int(lid); i < s1; i += TG) {
        float lp = head_sc<HAS_SC>(row[i], SC) - sub;
        if (lp >= scan_thr) {
          mass_loc += metal::fast::exp(lp);
          uint pos = atomic_fetch_add_explicit(&tg_cnt, 1u, memory_order_relaxed);
          if (pos < uint(MAX_CSP)) {
            cand_lp[dst_base + pos] = lp;
            cand_id[dst_base + pos] = uint(i);
          }
        }
      }
    }
    mass_loc = simd_sum(mass_loc);
    threadgroup float sbuf[32];
    if (lane == 0) sbuf[sg] = mass_loc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float tm = (lane < TG / 32) ? sbuf[lane] : 0.0f;
      tm = simd_sum(tm);
      if (lane == 0) {
        mass_part[size_t(b) * SP + sp] = tm;
        c_cnt[size_t(b) * SP + sp] = atomic_load_explicit(&tg_cnt, memory_order_relaxed);
        if (sp == 0) lpmax_out[b] = lp_max;
      }
    }
"""

_SAMP_SRAM_P2_SRC = r"""
    const float TOPP = s_cfg[0];
    const float INVT = s_cfg[2];
    uint b = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;

    float mass_loc = 0.0f;
    uint cnt_loc = 0u;
    uint ovf_loc = 0u;
    for (int s = int(lane); s < SP; s += 32) {
      mass_loc += mass_part[size_t(b) * SP + s];
      uint cs = c_cnt[size_t(b) * SP + s];
      if (cs > uint(MAX_CSP)) { ovf_loc = 1u; cs = uint(MAX_CSP); }
      cnt_loc += cs;
    }
    float tot_mass = simd_sum(mass_loc);
    int total_c = int(simd_sum(cnt_loc));
    bool ovf = (simd_sum(ovf_loc) != 0u);
    if (ovf || total_c > MAX_TOT || total_c == 0 || (USE_TOPP && tot_mass < TOPP)) {
      if (lane == 0u) {
        overflow[b] = 1u;
        out_bonus[b] = 0u;
        if constexpr (COUPLED) { out_reject[b] = 0u; if (int(b) < N_Q) child_pt[b] = 0.0f; }
      }
      return;
    }

    threadgroup float lps[MAX_TOT];
    threadgroup uint ids[MAX_TOT];
    threadgroup int k_cut_tg;
    if (lane == 0u) {
      int n = 0;
      for (int s = 0; s < SP; s++) {
        uint cnt_s = min(uint(MAX_CSP), c_cnt[size_t(b) * SP + s]);
        size_t src_base = (size_t(b) * SP + s) * MAX_CSP;
        for (uint j = 0; j < cnt_s; j++) {
          lps[n] = cand_lp[src_base + j];
          ids[n] = cand_id[src_base + j];
          n++;
        }
      }
      // Insertion sort descending on lps (n is typically 10..200)
      for (int i = 1; i < n; i++) {
        float kl = lps[i];
        uint ki = ids[i];
        int j = i - 1;
        while (j >= 0 && lps[j] < kl) {
          lps[j + 1] = lps[j];
          ids[j + 1] = ids[j];
          j--;
        }
        lps[j + 1] = kl;
        ids[j + 1] = ki;
      }
      int k_cut = n;
      if (USE_TOPP) {
        float cum = 0.0f;
        for (int i = 0; i < n; i++) {
          cum += metal::fast::exp(lps[i]);
          if (cum >= TOPP) {
            float thr_lp = lps[i];
            k_cut = i + 1;
            while (k_cut < n && lps[k_cut] >= thr_lp) k_cut++;
            break;
          }
        }
      }
      k_cut_tg = k_cut;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int k_cut = k_cut_tg;
    const float lp_max = lps[0];
    float zt_loc = 0.0f;
    for (int i = int(lane); i < k_cut; i += 32) {
      zt_loc += metal::fast::exp((lps[i] - lp_max) * INVT);
    }
    float zt = simd_sum(zt_loc);
    float inv_zt = 1.0f / max(zt, 1e-30f);

    const uint seed0 = seeds[b * 2 + 0];
    const uint seed1 = seeds[b * 2 + 1];
    auto hash_u01 = [](uint k0, uint k1, uint idx) -> float {
      uint x = idx ^ k0;
      x *= 0x7feb352du;
      x ^= x >> 15;
      x += k1;
      x *= 0x846ca68bu;
      x ^= x >> 16;
      return clamp((float(x & 0xFFFFFFu) + 0.5f) * 5.960464477539063e-8f, 1e-7f, 1.0f - 1e-7f);
    };

    uint qid_loc[16];
    float qp_loc[16];
    uint cid = 0u;
    if constexpr (COUPLED) {
      if (int(b) < N_Q) {
        size_t qoff = size_t(b) * 16;
        for (int k = 0; k < 16; k++) { qid_loc[k] = q_ids[qoff + k]; qp_loc[k] = q_probs[qoff + k]; }
        cid = child_toks[b];
      }
    }

    float best_b_sc = -INFINITY;
    uint best_b_id = ids[0];
    float best_r_sc = -INFINITY;
    uint best_r_id = ids[0];
    float c_pt = 0.0f;

    for (int i = int(lane); i < k_cut; i += 32) {
      uint id_i = ids[i];
      float pt = metal::fast::exp((lps[i] - lp_max) * INVT) * inv_zt;
      float u = hash_u01(seed0, seed1, id_i);
      float g = -metal::log(-metal::log(u));
      float sc_b = metal::log(max(pt, 1e-38f)) + g;
      if (sc_b > best_b_sc || (sc_b == best_b_sc && id_i < best_b_id)) { best_b_sc = sc_b; best_b_id = id_i; }
      if constexpr (COUPLED) {
        if (int(b) < N_Q) {
          if (id_i == cid) c_pt = pt;
          float qv = 0.0f;
          for (int k = 0; k < 16; k++) {
            if (qid_loc[k] == id_i) qv += qp_loc[k];
          }
          float rw = pt - qv;
          if (rw > 0.0f) {
            float sc_r = metal::log(rw) + g;
            if (sc_r > best_r_sc || (sc_r == best_r_sc && id_i < best_r_id)) { best_r_sc = sc_r; best_r_id = id_i; }
          }
        }
      }
    }

    for (uint d = 16u; d > 0u; d >>= 1) {
      float ob_sc = simd_shuffle_down(best_b_sc, d);
      uint ob_id = simd_shuffle_down(best_b_id, d);
      if (ob_sc > best_b_sc || (ob_sc == best_b_sc && ob_id < best_b_id)) { best_b_sc = ob_sc; best_b_id = ob_id; }
      if constexpr (COUPLED) {
        float or_sc = simd_shuffle_down(best_r_sc, d);
        uint or_id = simd_shuffle_down(best_r_id, d);
        if (or_sc > best_r_sc || (or_sc == best_r_sc && or_id < best_r_id)) { best_r_sc = or_sc; best_r_id = or_id; }
      }
    }
    if constexpr (COUPLED) {
      c_pt = simd_max(c_pt);
    }

    if (lane == 0u) {
      overflow[b] = 0u;
      out_bonus[b] = best_b_id;
      if constexpr (COUPLED) {
        out_reject[b] = (best_r_sc > -1e30f) ? best_r_id : best_b_id;
        if (int(b) < N_Q) child_pt[b] = c_pt;
      }
    }
"""


_SAMP_PURE_P1_SRC = r"""
    GL_NOFMA
    const float INVT = s_cfg[2];
    uint sp = threadgroup_position_in_grid.x;
    uint b = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31, sg = lid >> 5;
    const uint seed0 = seeds[b * 2 + 0];
    const uint seed1 = seeds[b * 2 + 1];
    auto hash_u01 = [](uint k0, uint k1, uint idx) -> float {
      uint x = idx ^ k0;
      x *= 0x7feb352du;
      x ^= x >> 15;
      x += k1;
      x *= 0x846ca68bu;
      x ^= x >> 16;
      return clamp((float(x & 0xFFFFFFu) + 0.5f) * 5.960464477539063e-8f, 1e-7f, 1.0f - 1e-7f);
    };
    const device bfloat16_t* row = logits + size_t(b) * PP;
    const int chunk = (((VV + SP - 1) / SP) + 7) & ~7;
    const int s0 = int(sp) * chunk;
    const int s1 = min(VV, s0 + chunk);
    float bsc = -INFINITY;
    uint bid = 0xFFFFFFFFu;
    if constexpr ((PP & 7) == 0) {
      int i = s0 + int(lid) * 8;
      for (; i + 8 <= s1; i += TG * 8) {
        uint4 pk = ((const device uint4*)(row + i))[0];
        bfloat4 v0 = as_type<bfloat4>(pk.xy), v1 = as_type<bfloat4>(pk.zw);
        bfloat16_t bs[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
        #pragma unroll
        for (int k = 0; k < 8; k++) {
          uint idx = uint(i + k);
          float u = hash_u01(seed0, seed1, idx);
          float sc = head_sc<HAS_SC>(bs[k], SC) * INVT - metal::log(-metal::log(u));
          if (sc > bsc || (sc == bsc && idx < bid)) { bsc = sc; bid = idx; }
        }
      }
      for (; i < s1; i++) {
        uint idx = uint(i);
        float u = hash_u01(seed0, seed1, idx);
        float sc = head_sc<HAS_SC>(row[i], SC) * INVT - metal::log(-metal::log(u));
        if (sc > bsc || (sc == bsc && idx < bid)) { bsc = sc; bid = idx; }
      }
    } else {
      for (int i = s0 + int(lid); i < s1; i += TG) {
        uint idx = uint(i);
        float u = hash_u01(seed0, seed1, idx);
        float sc = head_sc<HAS_SC>(row[i], SC) * INVT - metal::log(-metal::log(u));
        if (sc > bsc || (sc == bsc && idx < bid)) { bsc = sc; bid = idx; }
      }
    }
    for (uint d = 16u; d > 0u; d >>= 1) {
      float osc = simd_shuffle_down(bsc, d);
      uint oid = simd_shuffle_down(bid, d);
      if (osc > bsc || (osc == bsc && oid < bid)) { bsc = osc; bid = oid; }
    }
    threadgroup float sb[8];
    threadgroup uint si[8];
    if (lane == 0u) { sb[sg] = bsc; si[sg] = bid; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      float fv = -INFINITY;
      uint fi = 0xFFFFFFFFu;
      if (lane < TG / 32u) { fv = sb[lane]; fi = si[lane]; }
      for (uint d = 16u; d > 0u; d >>= 1) {
        float ov = simd_shuffle_down(fv, d);
        uint oi = simd_shuffle_down(fi, d);
        if (ov > fv || (ov == fv && oi < fi)) { fv = ov; fi = oi; }
      }
      if (lane == 0u) {
        size_t o = size_t(b) * SP + sp;
        pval[o] = fv;
        pidx[o] = fi;
      }
    }
"""

_SAMP_PURE_P2_SRC = r"""
    uint b = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    float fv = -INFINITY;
    uint fi = 0xFFFFFFFFu;
    for (int i = int(lane); i < SP; i += 32) {
      size_t o = size_t(b) * SP + uint(i);
      float ov = pval[o];
      uint oi = pidx[o];
      if (ov > fv || (ov == fv && oi < fi)) { fv = ov; fi = oi; }
    }
    for (uint d = 16u; d > 0u; d >>= 1) {
      float ov = simd_shuffle_down(fv, d);
      uint oi = simd_shuffle_down(fi, d);
      if (ov > fv || (ov == fv && oi < fi)) { fv = ov; fi = oi; }
    }
    if (lane == 0u) out[b] = fi;
"""

_SAMP_SRAM_P2_FB_SRC = r"""
    GL_NOFMA
    const float TOPP = s_cfg[0];
    const float MINP = s_cfg[1];
    const float INVT = s_cfg[2];
    uint b = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;

    const uint seed0 = seeds[b * 2 + 0];
    const uint seed1 = seeds[b * 2 + 1];
    auto hash_u01 = [](uint k0, uint k1, uint idx) -> float {
      uint x = idx ^ k0;
      x *= 0x7feb352du;
      x ^= x >> 15;
      x += k1;
      x *= 0x846ca68bu;
      x ^= x >> 16;
      return clamp((float(x & 0xFFFFFFu) + 0.5f) * 5.960464477539063e-8f, 1e-7f, 1.0f - 1e-7f);
    };

    float mass_loc = 0.0f;
    uint cnt_loc = 0u;
    uint ovf_loc = 0u;
    for (int s = int(lane); s < SP; s += 32) {
      mass_loc += mass_part[size_t(b) * SP + s];
      uint cs = c_cnt[size_t(b) * SP + s];
      if (cs > uint(MAX_CSP)) { ovf_loc = 1u; cs = uint(MAX_CSP); }
      cnt_loc += cs;
    }
    float tot_mass = simd_sum(mass_loc);
    int total_c = int(simd_sum(cnt_loc));
    bool ovf = (simd_sum(ovf_loc) != 0u);
    if (ovf || total_c > MAX_TOT || total_c == 0 || (USE_TOPP && tot_mass < TOPP)) {
      // Exact 0-sync GPU fallback directly over bf16 logits (triggers on <0.1% high-entropy rows).
      float gmv = gmax[b * 32 + lane];
      float gm = simd_max(gmv);
      float nn = gnrm[b * 32 + lane] * metal::fast::exp(gmv - gm);
      float tot = simd_sum(nn);
      float sub = metal::isinf(gm) ? gm : (metal::log(tot) + gm);
      float lp_max = gm - sub;
      const device bfloat16_t* row = logits + size_t(b) * PP;

      float thr = (MINP > 0.0f) ? (lp_max + metal::log(MINP)) : -1e30f;
      if constexpr (USE_TOPP) {
        float lmin = lp_max;
        for (int i = int(lane); i < VV; i += 32) {
          float lp = head_sc<HAS_SC>(row[i], SC) - sub;
          if (lp < lmin) lmin = lp;
        }
        lmin = simd_min(lmin);
        float lo = lmin, hi = lp_max;
        for (int it = 0; it < 26; it++) {
          float mid = 0.5f * (lo + hi);
          float m_loc = 0.0f;
          for (int i = int(lane); i < VV; i += 32) {
            float lp = head_sc<HAS_SC>(row[i], SC) - sub;
            if (lp >= mid) m_loc += metal::fast::exp(lp);
          }
          float m_sum = simd_sum(m_loc);
          if (m_sum >= TOPP) lo = mid;
          else hi = mid;
        }
        if (lo > thr) thr = lo;
      }
      if (thr > lp_max) thr = lp_max;

      float best_sc = -INFINITY;
      uint best_id = 0u;
      for (int i = int(lane); i < VV; i += 32) {
        float lp = head_sc<HAS_SC>(row[i], SC) - sub;
        if (lp >= thr) {
          float u = hash_u01(seed0, seed1, uint(i));
          float sc = (lp - lp_max) * INVT - metal::log(-metal::log(u));
          if (sc > best_sc || (sc == best_sc && uint(i) < best_id)) { best_sc = sc; best_id = uint(i); }
        }
      }
      for (uint d = 16u; d > 0u; d >>= 1) {
        float osc = simd_shuffle_down(best_sc, d);
        uint oid = simd_shuffle_down(best_id, d);
        if (osc > best_sc || (osc == best_sc && oid < best_id)) { best_sc = osc; best_id = oid; }
      }
      if (lane == 0u) out_bonus[b] = best_id;
      return;
    }

    threadgroup float lps[MAX_TOT];
    threadgroup uint ids[MAX_TOT];
    threadgroup int k_cut_tg;
    if (lane == 0u) {
      int n = 0;
      for (int s = 0; s < SP; s++) {
        uint cnt_s = min(uint(MAX_CSP), c_cnt[size_t(b) * SP + s]);
        size_t src_base = (size_t(b) * SP + s) * MAX_CSP;
        for (uint j = 0; j < cnt_s; j++) {
          lps[n] = cand_lp[src_base + j];
          ids[n] = cand_id[src_base + j];
          n++;
        }
      }
      for (int i = 1; i < n; i++) {
        float kl = lps[i];
        uint ki = ids[i];
        int j = i - 1;
        while (j >= 0 && lps[j] < kl) {
          lps[j + 1] = lps[j];
          ids[j + 1] = ids[j];
          j--;
        }
        lps[j + 1] = kl;
        ids[j + 1] = ki;
      }
      int k_cut = n;
      if (USE_TOPP) {
        float cum = 0.0f;
        for (int i = 0; i < n; i++) {
          cum += metal::fast::exp(lps[i]);
          if (cum >= TOPP) {
            float thr_lp = lps[i];
            k_cut = i + 1;
            while (k_cut < n && lps[k_cut] >= thr_lp) k_cut++;
            break;
          }
        }
      }
      k_cut_tg = k_cut;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int k_cut = k_cut_tg;
    const float lp_max = lps[0];
    float best_b_sc = -INFINITY;
    uint best_b_id = ids[0];
    for (int i = int(lane); i < k_cut; i += 32) {
      uint id_i = ids[i];
      float u = hash_u01(seed0, seed1, id_i);
      float sc_b = (lps[i] - lp_max) * INVT - metal::log(-metal::log(u));
      if (sc_b > best_b_sc || (sc_b == best_b_sc && id_i < best_b_id)) { best_b_sc = sc_b; best_b_id = id_i; }
    }
    for (uint d = 16u; d > 0u; d >>= 1) {
      float ob_sc = simd_shuffle_down(best_b_sc, d);
      uint ob_id = simd_shuffle_down(best_b_id, d);
      if (ob_sc > best_b_sc || (ob_sc == best_b_sc && ob_id < best_b_id)) { best_b_sc = ob_sc; best_b_id = ob_id; }
    }
    if (lane == 0u) out_bonus[b] = best_b_id;
"""


def _get_samp(kind, has_sc=False, sc_lit="1.0f"):
    key = ("samp", kind, has_sc, sc_lit)
    k = _kernels.get(key)
    if k is not None:
        return k
    sc_def = f"\n#define SC ({sc_lit})\n"
    h = abs(hash(sc_lit)) % 10**6
    if kind == "sparse_p1":
        k = mx.fast.metal_kernel(
            name=f"head_samp_sparse_p1_{int(has_sc)}_{h}",
            input_names=["logits", "gmax", "gnrm", "s_cfg"],
            output_names=["cand_lp", "cand_id", "mass_part", "c_cnt", "lpmax_out"],
            source=sc_def + _SAMP_SPARSE_P1_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "sram_p2":
        k = mx.fast.metal_kernel(
            name="head_samp_sram_p2",
            input_names=["cand_lp", "cand_id", "mass_part", "c_cnt", "seeds", "s_cfg", "q_ids", "q_probs", "child_toks"],
            output_names=["out_bonus", "out_reject", "child_pt", "overflow"],
            source=_SAMP_SRAM_P2_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "sram_p2_fb":
        k = mx.fast.metal_kernel(
            name=f"head_samp_sram_p2_fb_{int(has_sc)}_{h}",
            input_names=["cand_lp", "cand_id", "mass_part", "c_cnt", "seeds", "s_cfg", "logits", "gmax", "gnrm"],
            output_names=["out_bonus"],
            source=sc_def + _SAMP_SRAM_P2_FB_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "pure_p1":
        k = mx.fast.metal_kernel(
            name=f"head_samp_pure_p1_{int(has_sc)}_{h}",
            input_names=["logits", "seeds", "s_cfg"],
            output_names=["pval", "pidx"],
            source=sc_def + _SAMP_PURE_P1_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    elif kind == "pure_p2":
        k = mx.fast.metal_kernel(
            name="head_samp_pure_p2",
            input_names=["pval", "pidx"],
            output_names=["out"],
            source=_SAMP_PURE_P2_SRC, header=_HEAD_HEADER, ensure_row_contiguous=True)
    _kernels[key] = k
    return k


_SAMP_CFG_CACHE: dict = {}
_DUMMY_SAMP_U32 = mx.zeros((256,), mx.uint32)
_DUMMY_SAMP_F32 = mx.zeros((256,), mx.float32)


def _get_samp_cfg(top_p, min_p, temp):
    inv_t = 1.0 / max(float(temp), 1e-6)
    use_topp = 0.0 < float(top_p) < 1.0
    tp_val = float(top_p) if use_topp else 1.0
    mp_val = float(min_p) if min_p else 0.0
    ckey = (tp_val, mp_val, inv_t)
    s_cfg = _SAMP_CFG_CACHE.get(ckey)
    if s_cfg is None:
        if len(_SAMP_CFG_CACHE) >= 64:
            _SAMP_CFG_CACHE.clear()
        s_cfg = _SAMP_CFG_CACHE[ckey] = mx.array([tp_val, mp_val, inv_t], mx.float32)
    return s_cfg, use_topp, mp_val


def _run_sample_core(logits, softcap, unpadded_vocab, *, temp, top_p=1.0, min_p=0.0, seeds=None, coupled=False, q_ids=None, q_probs=None, child_toks=None):
    logits, B, P, V = _norm_logits(logits, unpadded_vocab)
    has_sc = softcap is not None and float(softcap) > 0.0
    sc_lit = flit(float(softcap)) if has_sc else "1.0f"
    s_cfg, use_topp, _ = _get_samp_cfg(top_p, min_p, temp)
    if seeds is None:
        seeds = mx.random.randint(0, 0x7FFFFFFF, (B, 2), dtype=mx.int32).astype(mx.uint32)

    kl = _get("lse32", has_sc, sc_lit)
    gmax, gnrm = kl(
        inputs=[logits],
        template=[("HAS_SC", has_sc), ("PP", P), ("VV", V)],
        grid=(32 * 32, B, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, 32), (B, 32)],
        output_dtypes=[mx.float32, mx.float32],
    )
    sp = 80 if B == 1 else max(16, 80 // B)
    tg = 128
    max_csp = 128
    max_tot = 1536
    ks1 = _get_samp("sparse_p1", has_sc, sc_lit)
    cand_lp, cand_id, mass_part, c_cnt, _ = ks1(
        inputs=[logits, gmax, gnrm, s_cfg],
        template=[("HAS_SC", has_sc), ("PP", P), ("VV", V), ("SP", sp), ("TG", tg), ("MAX_CSP", max_csp)],
        grid=(sp * tg, B, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, sp, max_csp), (B, sp, max_csp), (B, sp), (B, sp), (B,)],
        output_dtypes=[mx.float32, mx.uint32, mx.float32, mx.uint32, mx.float32],
    )
    ks2 = _get_samp("sram_p2")
    n_q = int(q_ids.shape[0]) if coupled else 0
    bonus, reject, child_pt, ovf = ks2(
        inputs=[
            cand_lp, cand_id, mass_part, c_cnt, seeds, s_cfg,
            q_ids.reshape(-1) if coupled else _DUMMY_SAMP_U32,
            q_probs.reshape(-1) if coupled else _DUMMY_SAMP_F32,
            child_toks.reshape(-1) if coupled else _DUMMY_SAMP_U32,
        ],
        template=[("SP", sp), ("MAX_CSP", max_csp), ("MAX_TOT", max_tot), ("USE_TOPP", use_topp), ("COUPLED", coupled), ("N_Q", n_q)],
        grid=(B * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B,), (B,) if coupled else (1,), (max(1, n_q),), (B,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.float32, mx.uint32],
    )
    if coupled:
        return bonus, reject, child_pt, ovf
    return bonus, ovf


def softcap_sample_rows(logits, softcap, unpadded_vocab=None, *, temp=1.0, top_p=1.0, min_p=0.0, seeds=None):
    """Fused GPU softcap + lse + sparse-nucleus top-p + Gumbel-max returning `(toks (B,) u32, overflow (B,) u32)`."""
    return _run_sample_core(logits, softcap, unpadded_vocab, temp=temp, top_p=top_p, min_p=min_p, seeds=seeds, coupled=False)


def softcap_sample_rows_nosync(logits, softcap, unpadded_vocab=None, *, temp=1.0, top_p=1.0, min_p=0.0, seeds=None):
    """Zero-sync fused GPU exact-distribution sampler returning `(B,) uint32` (never requires CPU overflow check)."""
    logits, B, P, V = _norm_logits(logits, unpadded_vocab)
    has_sc = softcap is not None and float(softcap) > 0.0
    sc_lit = flit(float(softcap)) if has_sc else "1.0f"
    s_cfg, use_topp, mp_val = _get_samp_cfg(top_p, min_p, temp)
    if seeds is None:
        seeds = mx.random.randint(0, 0x7FFFFFFF, (B, 2), dtype=mx.int32).astype(mx.uint32)
    sp = 80 if B == 1 else max(16, 80 // B)
    tg = 128

    if not use_topp and mp_val <= 0.0:
        kp1 = _get_samp("pure_p1", has_sc, sc_lit)
        pv, pi = kp1(
            inputs=[logits, seeds, s_cfg],
            template=[("HAS_SC", has_sc), ("PP", P), ("VV", V), ("SP", sp), ("TG", tg)],
            grid=(sp * tg, B, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(B, sp), (B, sp)],
            output_dtypes=[mx.float32, mx.uint32],
        )
        kp2 = _get_samp("pure_p2")
        (out,) = kp2(
            inputs=[pv, pi],
            template=[("SP", sp)],
            grid=(B * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(B,)],
            output_dtypes=[mx.uint32],
        )
        return out

    kl = _get("lse32", has_sc, sc_lit)
    gmax, gnrm = kl(
        inputs=[logits],
        template=[("HAS_SC", has_sc), ("PP", P), ("VV", V)],
        grid=(32 * 32, B, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, 32), (B, 32)],
        output_dtypes=[mx.float32, mx.float32],
    )
    max_csp = 128
    max_tot = 1536
    ks1 = _get_samp("sparse_p1", has_sc, sc_lit)
    cand_lp, cand_id, mass_part, c_cnt, _ = ks1(
        inputs=[logits, gmax, gnrm, s_cfg],
        template=[("HAS_SC", has_sc), ("PP", P), ("VV", V), ("SP", sp), ("TG", tg), ("MAX_CSP", max_csp)],
        grid=(sp * tg, B, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, sp, max_csp), (B, sp, max_csp), (B, sp), (B, sp), (B,)],
        output_dtypes=[mx.float32, mx.uint32, mx.float32, mx.uint32, mx.float32],
    )
    ks2 = _get_samp("sram_p2_fb", has_sc, sc_lit)
    (out,) = ks2(
        inputs=[cand_lp, cand_id, mass_part, c_cnt, seeds, s_cfg, logits, gmax, gnrm],
        template=[("HAS_SC", has_sc), ("PP", P), ("VV", V), ("SP", sp), ("MAX_CSP", max_csp), ("MAX_TOT", max_tot), ("USE_TOPP", use_topp)],
        grid=(B * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B,)],
        output_dtypes=[mx.uint32],
    )
    return out


def softcap_coupled_verify_rows(
    logits, softcap, unpadded_vocab, q_ids, q_probs, child_toks, *, temp=1.0, top_p=1.0, min_p=0.0, seeds=None
):
    """Return `(bonus_tok (B,) u32, reject_tok (B,) u32, child_pt (B-1,) f32, overflow (B,) u32)`."""
    return _run_sample_core(
        logits, softcap, unpadded_vocab, temp=temp, top_p=top_p, min_p=min_p, seeds=seeds,
        coupled=True, q_ids=q_ids, q_probs=q_probs, child_toks=child_toks,
    )
