"""Fused LM head tail kernel: bf16 softcap -> logsumexp -> (softcapped - lse) argmax (Phase 3b).

Avoids materializing (B, V) f32 tensors for softcap and logsumexp and launches 2 kernels instead of
cast + div + tanh + mul + slice + C*(logsumexp + sub + argmax) + concat.
Matches MLX's `tanh(logits.f32 / sc) * sc`, `logsumexp_looped` (rows/glue.py), and `argmax` bitwise.
"""
import os

import mlx.core as mx

from .glue import GLUE_HEADER, flit

__all__ = ["softcap_argmax_rows", "softcap_logprobs_rows", "softcap_lse_rows"]

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

template <int TG, bool HAS_SC>
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
        for (int i = 0; i < 4; i++) vals[i] = head_sc<HAS_SC>(p[i], sc);
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
    float out_v = head_logsumexp_bf16<TG, HAS_SC>(logits + size_t(b) * PP, VV, SC, lid, lmax, lnrm);
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
    const int chunk = (VV + SP - 1) / SP;
    const int s0 = int(sp) * chunk;
    const int s1 = min(VV, s0 + chunk);
    float bval = -INFINITY, sval = -INFINITY;
    uint bidx = 0xFFFFFFFFu, nanf = 0u;
    for (int i = s0 + int(lid); i < s1; i += TG) {
      float v = float(row[i]);
      if (metal::isnan(v)) { nanf = 1u; }
      else if (v > bval || (v == bval && uint(i) < bidx)) { sval = bval; bval = v; bidx = uint(i); }
      else if (v > sval) { sval = v; }
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
    float sub = head_logsumexp_bf16<256, HAS_SC>(row, VV, SC, lid, lmax, lnrm);
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
        for (int i = 0; i < 4; i++) vals[i] = head_sc<HAS_SC>(p[i], SC);
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
    const int chunk = (VV + SP - 1) / SP;
    const int s0 = int(sp) * chunk;
    const int s1 = min(VV, s0 + chunk);
    if constexpr (!DO_ARG) {
      for (int i = s0 + int(lid); i < s1; i += TG) {
        lp_row[i] = head_sc<HAS_SC>(row[i], SC) - sub;
      }
      return;
    }
    threadgroup GlueIV buf[32];
    GlueIV best;
    best.val = -INFINITY;
    best.idx = 0;
    for (int i = s0 + int(lid); i < s1; i += TG) {
      float v = head_sc<HAS_SC>(row[i], SC) - sub;
      lp_row[i] = v;
      GlueIV c;
      c.val = v;
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
