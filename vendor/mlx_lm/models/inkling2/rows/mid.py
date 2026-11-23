"""Lane f2: fused kernel K5 MID (docs/FUSED_MID.md, docs/FUSED.md).

One Metal launch for B rows: everything between the q/r/k/v projections and attention.
  k, v: short conv (sliding: the default fused conv kernel arithmetic; full: DynamicShortConvolution), new conv states;
  q: ScaledL2Clip, k: L2Clip, v: unweighted rms (eps from args), rel: ScaledL2Clip over d_rel = 16;
  full layers: q and rel scaled by tau (f32 multiply, bf16).
Each row is bitwise equal to fused_ref.visit_ref's q, k, v, rel, w0, w1 for that row alone.  Not wired into decode.
"""
import math

import mlx.core as mx
import numpy as np

from .glue import GLUE_HEADER, flit

__all__ = ["mid_rows"]

_MID_HEADER = r"""
// sliding ShortConvolution (fastdecode._CONV_SRC arithmetic, per channel): state cast to the weight dtype (bf16)
inline float mid_conv_s(float x0, float s0, float s1, float s2, float w0, float w1, float w2, float w3, float scale) {
  GL_NOFMA
  float t0 = float(bfloat16_t(s0)), t1 = float(bfloat16_t(s1)), t2 = float(bfloat16_t(s2));
  float acc = 0.0f;
  acc = fma(w0, t0, acc);
  acc = fma(w1, t1, acc);
  acc = fma(w2, t2, acc);
  acc = fma(w3, x0, acc);
  float conv = float(bfloat16_t(acc));
  return float(bfloat16_t((conv + x0) / scale));
}
// DynamicShortConvolution: weights = static + dynamic (f32); windows lag order (x, s2, s1, s0); Sum over the kernel axis is MLX
// row_reduce_small (sequential from 0: acc = p + acc); then (out + x) / 3 -> bf16.
inline float mid_conv_d(float x0, float s0, float s1, float s2, thread float* st, thread float* dy) {
  GL_NOFMA
  float p0 = x0 * (st[0] + dy[0]);
  float p1 = s2 * (st[1] + dy[1]);
  float p2 = s1 * (st[2] + dy[2]);
  float p3 = s0 * (st[3] + dy[3]);
  float acc = p0 + 0.0f;
  acc = p1 + acc;
  acc = p2 + acc;
  acc = p3 + acc;
  return float(bfloat16_t((acc + x0) / 3.0f));
}
"""


def _source(full, H, HK, thr_q, thr_k, thr_r, eps, debug):
    C = HK * 128
    src = f"""
    constexpr int H = {H}, HK = {HK}, C = {C};
    uint b = threadgroup_position_in_grid.y;
    uint role = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31, sg = lid >> 5;
    threadgroup float vbuf[8 * 128];
    float tau_v = {"tau[b]" if full else "1.0f"};
    if (role < 3) {{                          // q: head = 8*role + sg
      uint h = role * 8 + sg;
      const device bfloat16_t* x = q_proj + b * (H * 128) + h * 128 + lane * 4;
      float s[4];
      for (int i = 0; i < 4; i++) s[i] = float(x[i]) * float(q_scale[h * 128 + lane * 4 + i]);
      glue_l2_clip_lane128(s, {flit(thr_q)});
      for (int i = 0; i < 4; i++) {{
        bfloat16_t o = bfloat16_t(s[i]);
        {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
        q_out[b * (H * 128) + h * 128 + lane * 4 + i] = o;
      }}
    }} else if (role < 5) {{                  // k (role 3) / v (role 4): head = sg, channels c = head*128 + 4*lane + i
      bool isv = role == 4;
      uint c0 = sg * 128 + lane * 4;
      const device bfloat16_t* xp = (isv ? v_proj : k_proj) + b * C;
      const device float* st = (isv ? cs1 : cs0) + b * 3 * C;
      device float* nst = (isv ? w1 : w0) + b * 3 * C;
      float y[4];
      for (int i = 0; i < 4; i++) {{
        uint c = c0 + i;
        float x0 = float(xp[c]);
        float s0 = st[c], s1 = st[C + c], s2 = st[2 * C + c];
"""
    if full:
        src += f"""
        const device bfloat16_t* wp = (isv ? v_wproj : k_wproj) + b * {HK * 4} + sg * 4;
        float dy[4], sw[4];
        for (int k = 0; k < 4; k++) {{ dy[k] = float(wp[k]); sw[k] = float((isv ? v_static : k_static)[k * C + c]); }}
        y[i] = mid_conv_d(x0, s0, s1, s2, sw, dy);
"""
    else:
        src += f"""
        const device bfloat16_t* cw = (isv ? v_conv : k_conv) + c * 4;
        y[i] = mid_conv_s(x0, s0, s1, s2, float(cw[0]), float(cw[1]), float(cw[2]), float(cw[3]), {flit(np.float32(math.sqrt(5)))});
"""
    src += f"""
        nst[c] = s1; nst[C + c] = s2; nst[2 * C + c] = x0;
      }}
""" + ("""
      for (int i = 0; i < 4; i++) (isv ? vconv_dbg : kconv_dbg)[b * C + c0 + i] = bfloat16_t(y[i]);
""" if debug else "") + f"""
      if (!isv) {{
        glue_l2_clip_lane128(y, {flit(thr_k)});
        for (int i = 0; i < 4; i++) k_out[b * C + c0 + i] = bfloat16_t(y[i]);
      }} else {{
        threadgroup float* vb = vbuf + sg * 128;
        for (int i = 0; i < 4; i++) vb[lane * 4 + i] = y[i];
        float inv = glue_rms_inv_sg<128, true>(vb, {flit(eps)}, lane);
        for (int i = 0; i < 4; i++) v_out[b * C + c0 + i] = glue_rms_apply_nw<bfloat16_t>(y[i], inv);
      }}
    }} else if (sg == 0 && lane < H) {{        // rel: one lane per head, row of 16
      uint h = lane;
      float s[16];
      for (int i = 0; i < 16; i++) s[i] = float(r_proj[b * (H * 16) + h * 16 + i]) * float(rel_scale[h * 16 + i]);
      glue_l2_clip_row16(s, {flit(thr_r)});
      for (int i = 0; i < 16; i++) {{
        bfloat16_t o = bfloat16_t(s[i]);
        {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
        rel_out[b * (H * 16) + h * 16 + i] = o;
      }}
    }}
"""
    return src


_cache = {}


def mid_rows(layer, q_proj, r_proj, k_proj, v_proj, cs0, cs1, tau=None, k_wproj=None, v_wproj=None, debug=False):
    """K5 MID for B rows in one launch.

    layer: the decoder layer (or its Attention).  Shapes (B rows): q_proj (B, H*128) bf16, r_proj (B, H*16) bf16, k_proj, v_proj
    (B, HK*128) bf16, cs0, cs1 (B, 3, HK*128) f32.  Full-attention layers also: tau (B,) f32 (per-row fastdecode._full_tau value),
    k_wproj, v_wproj (B, HK*4) bf16 (= k/v_sconv.weight_proj outputs).
    Returns (q (B,H,128), k (B,HK,128), v (B,HK,128), rel (B,H,16)) bf16 and new conv states (w0, w1) (B,3,HK*128) f32
    [+ k_conv, v_conv (B, HK*128) bf16 if debug].  Row b equals visit_ref's q/k/v/rel/w0/w1 for that row alone (q/k/v before the
    head transpose).  Unsupported shapes/dtypes raise.
    """
    a = getattr(layer, "self_attn", layer)
    full = a.sliding_window == 0
    H, HK = a.num_heads, a.num_key_value_heads
    C = HK * 128
    B = q_proj.shape[0]
    assert a.head_dim == 128 and a.rel_l2_clip.scale.shape == (H, 16) and a.k_sconv.kernel_size == 4 and H <= 32, "unsupported config"
    assert HK <= 8 and H <= 24
    for x, shp, dt in ((q_proj, (B, H * 128), mx.bfloat16), (r_proj, (B, H * 16), mx.bfloat16), (k_proj, (B, C), mx.bfloat16),
                       (v_proj, (B, C), mx.bfloat16), (cs0, (B, 3, C), mx.float32), (cs1, (B, 3, C), mx.float32)):
        if x.shape != shp or x.dtype != dt:
            raise ValueError("bad input %s %s, want %s %s" % (x.shape, x.dtype, shp, dt))
    if a.q_l2_clip.scale.shape != (H, 128):
        raise ValueError("q scale shape")
    ins = [q_proj, r_proj, k_proj, v_proj, cs0, cs1, a.q_l2_clip.scale, a.rel_l2_clip.scale]
    names = ["q_proj", "r_proj", "k_proj", "v_proj", "cs0", "cs1", "q_scale", "rel_scale"]
    if full:
        if tau is None or k_wproj is None or v_wproj is None:
            raise ValueError("full-attention layer needs tau, k_wproj, v_wproj")
        tau = tau.reshape(-1)
        if tau.shape != (B,) or tau.dtype != mx.float32:
            raise ValueError("tau must be (B,) f32")
        for w in (k_wproj, v_wproj):
            if w.shape != (B, HK * 4) or w.dtype != mx.bfloat16:
                raise ValueError("wproj must be (B, %d) bf16" % (HK * 4))
        for m in (a.k_sconv, a.v_sconv):
            assert m.static_weight.shape == (4, C) and m.num_heads == HK
        ins += [tau, k_wproj, v_wproj, a.k_sconv.static_weight, a.v_sconv.static_weight]
        names += ["tau", "k_wproj", "v_wproj", "k_static", "v_static"]
    else:
        if tau is not None:
            raise ValueError("sliding layer takes no tau")
        for m in (a.k_sconv, a.v_sconv):
            if m.conv1d.weight.dtype != mx.bfloat16 or m.conv1d.weight.shape != (C, 4, 1):
                raise ValueError("conv weight must be (C,4,1) bf16")
        ins += [a.k_sconv.conv1d.weight, a.v_sconv.conv1d.weight]
        names += ["k_conv", "v_conv"]
    outs = ["q_out", "k_out", "v_out", "rel_out", "w0", "w1"] + (["kconv_dbg", "vconv_dbg"] if debug else [])
    key = (full, H, HK, a.q_l2_clip.threshold, a.k_l2_clip.threshold, a.rel_l2_clip.threshold, a.eps, debug,
           a.q_l2_clip.scale.dtype, a.rel_l2_clip.scale.dtype, tuple(str(x.dtype) for x in ins))
    k = _cache.get(key)
    if k is None:
        src = _source(full, H, HK, a.q_l2_clip.threshold, a.k_l2_clip.threshold, a.rel_l2_clip.threshold, a.eps, debug)
        k = _cache[key] = mx.fast.metal_kernel(name="mid_rows_%s_%d" % ("F" if full else "S", abs(hash(key)) % 10**9),
                                               input_names=names, output_names=outs, source=src, header=GLUE_HEADER + _MID_HEADER)
    shapes = [(B, H, 128), (B, HK, 128), (B, HK, 128), (B, H, 16), (B, 3, C), (B, 3, C)] + ([(B, C)] * 2 if debug else [])
    dts = [mx.bfloat16] * 4 + [mx.float32] * 2 + ([mx.bfloat16] * 2 if debug else [])
    return k(inputs=ins, grid=(256 * 6, B, 1), threadgroup=(256, 1, 1), output_shapes=shapes, output_dtypes=dts)
