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

__all__ = ["mid_rows", "mid_rows_flat", "mid_causal_w_flat"]

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


def _source(full, H, HK, thr_q, thr_k, thr_r, eps, debug, flat=False, rel_r=0, packed_kv=False, causal_w=False, has_w_act=False, packed_kv_b=False):
    C = HK * 128
    fused_rel = rel_r > 0
    flat_ptrs = ""
    if flat:
        flat_ptrs = """
    const device bfloat16_t* q_proj = qkv + OFF_Q;
    const device bfloat16_t* r_proj = qkv + OFF_R;
    const device bfloat16_t* k_proj = qkv + OFF_K;
    const device bfloat16_t* v_proj = qkv + OFF_V;""" + ("""
    const device bfloat16_t* k_wproj = qkv + OFF_KW;
    const device bfloat16_t* v_wproj = qkv + OFF_VW;""" if full else "")
    wa_guard = "\n    if (int(b) >= w_act[0]) return;" if has_w_act else ""
    str_q = "STR_Q" if flat else "(H * 128)"
    str_r = "STR_R" if flat else "(H * 16)"
    str_k = "STR_K" if flat else "C"
    str_v = "STR_V" if flat else "C"
    str_kw = "STR_KW" if flat else f"{HK * 4}"
    str_vw = "STR_VW" if flat else f"{HK * 4}"
    src = f"""
    constexpr int H = {H}, HK = {HK}, C = {C};
    uint b = threadgroup_position_in_grid.y;{wa_guard}
    uint role = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid & 31, sg = lid >> 5;
    threadgroup float vbuf[8 * {384 if fused_rel else 128}];
    float tau_v = {"tau[b]" if full else "1.0f"};{flat_ptrs}
    if (role < 3) {{                          // q: head = 8*role + sg
      uint h = role * 8 + sg;
      const device bfloat16_t* x = q_proj + b * {str_q} + h * 128 + lane * 4;
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
      uint c0 = sg * 128 + lane * 4;"""
    if causal_w:
        src += f"""
      const device bfloat16_t* xp = (isv ? v_proj : k_proj);
      const uint st_x = isv ? {str_v} : {str_k};
      const device float* st = (isv ? cs1 : cs0);
      device float* nst = (isv ? w1 : w0) + b * 3 * C;
      int a0 = conv_anc[b * 3 + 0];
      int a1 = conv_anc[b * 3 + 1];
      int a2 = conv_anc[b * 3 + 2];
      float y[4];
      for (int i = 0; i < 4; i++) {{
        uint c = c0 + i;
        float x0 = float(xp[b * st_x + c]);
        float s0 = (a0 >= 0) ? float(xp[a0 * st_x + c]) : st[(-1 - a0) * C + c];
        float s1 = (a1 >= 0) ? float(xp[a1 * st_x + c]) : st[(-1 - a1) * C + c];
        float s2 = (a2 >= 0) ? float(xp[a2 * st_x + c]) : st[(-1 - a2) * C + c];
"""
    else:
        src += f"""
      const device bfloat16_t* xp = isv ? (v_proj + b * {str_v}) : (k_proj + b * {str_k});
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
        const device bfloat16_t* wp = (isv ? (v_wproj + b * {str_vw}) : (k_wproj + b * {str_kw})) + sg * 4;
        float dy[4], sw[4];
        for (int k = 0; k < 4; k++) {{ dy[k] = float(wp[k]); sw[k] = float((isv ? v_static : k_static)[k * C + c]); }}
        y[i] = mid_conv_d(x0, s0, s1, s2, sw, dy);
"""
    else:
        src += f"""
        const device bfloat16_t* cw = (isv ? v_conv : k_conv) + c * 4;
        y[i] = mid_conv_s(x0, s0, s1, s2, float(cw[0]), float(cw[1]), float(cw[2]), float(cw[3]), {flit(np.float32(math.sqrt(5)))});
"""
    if causal_w:
        k_dst = "kv_out[(sg * WW + b) * 128 + lane * 4 + i]"
        v_dst = "kv_out[((HK + sg) * WW + b) * 128 + lane * 4 + i]"
    elif packed_kv_b:
        k_dst = "kv_out[((b * 2) * HK + sg) * 128 + lane * 4 + i]"
        v_dst = "kv_out[((b * 2 + 1) * HK + sg) * 128 + lane * 4 + i]"
    else:
        k_dst = "kv_out[c0 + i]" if packed_kv else "k_out[b * C + c0 + i]"
        v_dst = f"kv_out[{C} + c0 + i]" if packed_kv else "v_out[b * C + c0 + i]"
    src += f"""
        nst[c] = s1; nst[C + c] = s2; nst[2 * C + c] = x0;
      }}
""" + ("""
      for (int i = 0; i < 4; i++) (isv ? vconv_dbg : kconv_dbg)[b * C + c0 + i] = bfloat16_t(y[i]);
""" if debug else "") + f"""
      if (!isv) {{
        glue_l2_clip_lane128(y, {flit(thr_k)});
        for (int i = 0; i < 4; i++) {k_dst} = bfloat16_t(y[i]);
      }} else {{
        threadgroup float* vb = vbuf + sg * 128;
        for (int i = 0; i < 4; i++) vb[lane * 4 + i] = y[i];
        float inv = glue_rms_inv_sg<128, true>(vb, {flit(eps)}, lane);
        for (int i = 0; i < 4; i++) {v_dst} = glue_rms_apply_nw<bfloat16_t>(y[i], inv);
      }}
"""
    if fused_rel:
        src += f"""    }} else {{
      constexpr int R = {rel_r};
      threadgroup bfloat16_t rel_s[H * 16];
      if (sg == 0 && lane < H) {{
        uint h = lane;
        float s[16];
        for (int i = 0; i < 16; i++) s[i] = float(r_proj[b * {str_r} + h * 16 + i]) * float(rel_scale[h * 16 + i]);
        glue_l2_clip_row16(s, {flit(thr_r)});
        for (int i = 0; i < 16; i++) {{
          bfloat16_t o = bfloat16_t(s[i]);
          {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
          rel_s[h * 16 + i] = o;
        }}
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint n0 = (role - 5) * 128;
      const device bfloat16_t* Bm = rw + n0 + sg * 16;
      device bfloat16_t* Cm = pb_out + b * (H * R) + n0 + sg * 16;
      simdgroup_bfloat8x8 ma[3][2];
      simdgroup_bfloat8x8 mb[2][2];
      for (int mi = 0; mi < 3; mi++) {{
        simdgroup_load(ma[mi][0], rel_s + (mi * 8) * 16 + 0, 16);
        simdgroup_load(ma[mi][1], rel_s + (mi * 8) * 16 + 8, 16);
      }}
      for (int ni = 0; ni < 2; ni++) {{
        simdgroup_load(mb[0][ni], Bm + 0 * R + ni * 8, R);
        simdgroup_load(mb[1][ni], Bm + 8 * R + ni * 8, R);
      }}
      threadgroup float* st = vbuf + sg * 384;
      for (int mi = 0; mi < 3; mi++) {{
        for (int ni = 0; ni < 2; ni++) {{
          simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
          simdgroup_multiply_accumulate(acc, ma[mi][0], mb[0][ni], acc);
          simdgroup_multiply_accumulate(acc, ma[mi][1], mb[1][ni], acc);
          simdgroup_store(acc, st + mi * 128 + ni * 8, 16);
        }}
      }}
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = lane; i < 384; i += 32) {{
        int r = i >> 4, c = i & 15;
        Cm[r * R + c] = bfloat16_t(st[i]);
      }}
    }}
"""
    else:
        src += f"""    }} else if (sg == 0 && lane < H) {{        // rel: one lane per head, row of 16
      uint h = lane;
      float s[16];
      for (int i = 0; i < 16; i++) s[i] = float(r_proj[b * {str_r} + h * 16 + i]) * float(rel_scale[h * 16 + i]);
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
_CHAIN_CONV_ANC = {}


def _default_conv_anc(W):
    a = _CHAIN_CONV_ANC.get(W)
    if a is None:
        rows = [[b - 3 if b >= 3 else -(b + 1), b - 2 if b >= 2 else -(b + 2), b - 1 if b >= 1 else -3] for b in range(W)]
        a = _CHAIN_CONV_ANC[W] = mx.array(rows, mx.int32)
    return a


def _run_mid(
    layer, q_proj, r_proj, k_proj, v_proj, flat_buf, flat_offs, cs0, cs1, tau, k_wproj, v_wproj, debug,
    rel_w=None, packed_kv=False, causal_w_B=0, conv_anc=None, w_act=None, flat_strs=None, packed_kv_b=False,
):
    a = dict.get(layer, "self_attn", layer) if isinstance(layer, dict) else getattr(layer, "self_attn", layer)
    if isinstance(a, dict):
        ql2, kl2, rl2 = a["q_l2_clip"], a["k_l2_clip"], a["rel_l2_clip"]
        q_sc, r_sc = ql2["scale"], rl2["scale"]
        ksconv, vsconv = a["k_sconv"], a["v_sconv"]
    else:
        ql2, kl2, rl2 = a.q_l2_clip, a.k_l2_clip, a.rel_l2_clip
        q_sc, r_sc = ql2.scale, rl2.scale
        ksconv, vsconv = a.k_sconv, a.v_sconv
    full = a.sliding_window == 0
    H, HK = a.num_heads, a.num_key_value_heads
    C = HK * 128
    flat = flat_buf is not None
    causal_w = causal_w_B > 0
    if causal_w:
        B = int(causal_w_B)
        if not flat or debug or packed_kv_b:
            raise ValueError("causal_w requires flat=True and debug=False")
        packed_kv = True
        if cs0.ndim != 3 or cs0.shape[1:] != (3, C) or cs0.dtype != mx.float32 or cs1.shape != cs0.shape or cs1.dtype != mx.float32:
            raise ValueError("causal_w cs0/cs1 must be (S, 3, C) f32")
    else:
        B = cs0.shape[0]
        if packed_kv and packed_kv_b:
            raise ValueError("choose packed_kv or packed_kv_b")
        if packed_kv and (B != 1 or debug):
            raise ValueError("packed_kv requires B == 1 and debug=False")
        if packed_kv_b and debug:
            raise ValueError("packed_kv_b requires debug=False")
        if cs0.shape != (B, 3, C) or cs0.dtype != mx.float32 or cs1.shape != (B, 3, C) or cs1.dtype != mx.float32:
            raise ValueError("cs0/cs1 must be (B, 3, C) f32")
    assert a.head_dim == 128 and r_sc.shape == (H, 16) and ksconv.kernel_size == 4 and H <= 32, "unsupported config"
    assert HK <= 8 and H <= 24
    if flat:
        if flat_buf.ndim != 1 or flat_buf.dtype != mx.bfloat16:
            raise ValueError("flat_buf must be 1D bf16")
        ins = [flat_buf, cs0, cs1, q_sc, r_sc]
        names = ["qkv", "cs0", "cs1", "q_scale", "rel_scale"]
    else:
        for x, shp, dt in ((q_proj, (B, H * 128), mx.bfloat16), (r_proj, (B, H * 16), mx.bfloat16), (k_proj, (B, C), mx.bfloat16),
                           (v_proj, (B, C), mx.bfloat16)):
            if x.shape != shp or x.dtype != dt:
                raise ValueError("bad input %s %s, want %s %s" % (x.shape, x.dtype, shp, dt))
        ins = [q_proj, r_proj, k_proj, v_proj, cs0, cs1, q_sc, r_sc]
        names = ["q_proj", "r_proj", "k_proj", "v_proj", "cs0", "cs1", "q_scale", "rel_scale"]
    if q_sc.shape != (H, 128):
        raise ValueError("q scale shape")
    tmpl = []
    if causal_w:
        if conv_anc is None:
            conv_anc = _default_conv_anc(B)
        else:
            assert conv_anc.shape == (B, 3) and conv_anc.dtype == mx.int32
        ins.append(conv_anc)
        names.append("conv_anc")
        tmpl += [("WW", B)]
    if flat:
        fs = flat_strs or {}
        tmpl += [
            ("OFF_Q", flat_offs["q_proj"]), ("OFF_R", flat_offs["r_proj"]),
            ("OFF_K", flat_offs["k_proj"]), ("OFF_V", flat_offs["v_proj"]),
            ("STR_Q", fs.get("q_proj", H * 128)), ("STR_R", fs.get("r_proj", H * 16)),
            ("STR_K", fs.get("k_proj", C)), ("STR_V", fs.get("v_proj", C)),
        ]
    if full:
        if tau is None or (not flat and (k_wproj is None or v_wproj is None)):
            raise ValueError("full-attention layer needs tau, k_wproj, v_wproj")
        tau = tau.reshape(-1)
        if tau.shape != (B,) or tau.dtype != mx.float32:
            raise ValueError("tau must be (B,) f32")
        ksw = ksconv["static_weight"] if isinstance(ksconv, dict) else ksconv.static_weight
        vsw = vsconv["static_weight"] if isinstance(vsconv, dict) else vsconv.static_weight
        assert ksw.shape == (4, C) and ksconv.num_heads == HK and vsw.shape == (4, C) and vsconv.num_heads == HK
        if flat:
            fs = flat_strs or {}
            tmpl += [
                ("OFF_KW", flat_offs["k_wproj"]), ("OFF_VW", flat_offs["v_wproj"]),
                ("STR_KW", fs.get("k_wproj", HK * 4)), ("STR_VW", fs.get("v_wproj", HK * 4)),
            ]
            ins += [tau, ksw, vsw]
            names += ["tau", "k_static", "v_static"]
        else:
            for w in (k_wproj, v_wproj):
                if w.shape != (B, HK * 4) or w.dtype != mx.bfloat16:
                    raise ValueError("wproj must be (B, %d) bf16" % (HK * 4))
            ins += [tau, k_wproj, v_wproj, ksw, vsw]
            names += ["tau", "k_wproj", "v_wproj", "k_static", "v_static"]
    else:
        if tau is not None:
            raise ValueError("sliding layer takes no tau")
        kcw = ksconv["conv1d"]["weight"] if isinstance(ksconv, dict) else ksconv.conv1d.weight
        vcw = vsconv["conv1d"]["weight"] if isinstance(vsconv, dict) else vsconv.conv1d.weight
        if kcw.dtype != mx.bfloat16 or kcw.shape != (C, 4, 1) or vcw.dtype != mx.bfloat16 or vcw.shape != (C, 4, 1):
            raise ValueError("conv weight must be (C,4,1) bf16")
        ins += [kcw, vcw]
        names += ["k_conv", "v_conv"]
    rel_r = 0
    kv_w = B if causal_w else 1
    kv_shape = (B, 2, HK, 128) if packed_kv_b else (2, HK, kv_w, 128)
    any_pkv = packed_kv or packed_kv_b
    if rel_w is not None:
        if debug:
            raise ValueError("mid_rows rel_w fusion does not support debug=True")
        if rel_w.dtype != mx.bfloat16 or rel_w.ndim != 2 or rel_w.shape[0] != 16 or rel_w.shape[1] % 128 != 0 or H != 24:
            raise ValueError(f"unsupported rel_w shape/dtype {rel_w.shape} {rel_w.dtype} for H={H}")
        rel_r = rel_w.shape[1]
        ins.append(rel_w)
        names.append("rw")
        if any_pkv:
            outs = ["q_out", "kv_out", "pb_out", "w0", "w1"]
            shapes = [(B, H, 128), kv_shape, (B, H, rel_r), (B, 3, C), (B, 3, C)]
            dts = [mx.bfloat16] * 3 + [mx.float32] * 2
        else:
            outs = ["q_out", "k_out", "v_out", "pb_out", "w0", "w1"]
            shapes = [(B, H, 128), (B, HK, 128), (B, HK, 128), (B, H, rel_r), (B, 3, C), (B, 3, C)]
            dts = [mx.bfloat16] * 4 + [mx.float32] * 2
        ntg = 5 + rel_r // 128
    else:
        if any_pkv:
            outs = ["q_out", "kv_out", "rel_out", "w0", "w1"]
            shapes = [(B, H, 128), kv_shape, (B, H, 16), (B, 3, C), (B, 3, C)]
            dts = [mx.bfloat16] * 3 + [mx.float32] * 2
        else:
            outs = ["q_out", "k_out", "v_out", "rel_out", "w0", "w1"] + (["kconv_dbg", "vconv_dbg"] if debug else [])
            shapes = [(B, H, 128), (B, HK, 128), (B, HK, 128), (B, H, 16), (B, 3, C), (B, 3, C)] + ([(B, C)] * 2 if debug else [])
            dts = [mx.bfloat16] * 4 + [mx.float32] * 2 + ([mx.bfloat16] * 2 if debug else [])
        ntg = 6
    has_wa = w_act is not None
    if has_wa:
        ins.append(w_act.reshape(-1))
        names.append("w_act")
    key = (flat, full, rel_r, packed_kv, causal_w, has_wa, packed_kv_b, H, HK, ql2.threshold, kl2.threshold, rl2.threshold, a.eps, debug,
           q_sc.dtype, r_sc.dtype, tuple(x.dtype for x in ins))
    k = _cache.get(key)
    if k is None:
        src = _source(full, H, HK, ql2.threshold, kl2.threshold, rl2.threshold,
                      a.eps, debug, flat=flat, rel_r=rel_r, packed_kv=packed_kv, causal_w=causal_w, has_w_act=has_wa, packed_kv_b=packed_kv_b)
        k = _cache[key] = mx.fast.metal_kernel(
            name="mid_rows_%s%s%s%s%s%s%s_%d" % ("F" if full else "S", "_flat" if flat else "", "_rel" if rel_r else "", "_pkv" if packed_kv else "", "_pkb" if packed_kv_b else "", "_cw" if causal_w else "", "_wa" if has_wa else "", abs(hash(key)) % 10**9),
            input_names=names, output_names=outs, source=src, header=GLUE_HEADER + _MID_HEADER,
            ensure_row_contiguous=True)
    kw = {"template": tmpl} if tmpl else {}
    return k(inputs=ins, grid=(256 * ntg, B, 1), threadgroup=(256, 1, 1), output_shapes=shapes, output_dtypes=dts, **kw)


def mid_rows(layer, q_proj, r_proj, k_proj, v_proj, cs0, cs1, tau=None, k_wproj=None, v_wproj=None, debug=False, rel_w=None):
    """K5 MID for B rows in one launch.

    layer: the decoder layer (or its Attention).  Shapes (B rows): q_proj (B, H*128) bf16, r_proj (B, H*16) bf16, k_proj, v_proj
    (B, HK*128) bf16, cs0, cs1 (B, 3, HK*128) f32.  Full-attention layers also: tau (B,) f32 (per-row fastdecode._full_tau value),
    k_wproj, v_wproj (B, HK*4) bf16 (= k/v_sconv.weight_proj outputs).
    Returns (q (B,H,128), k (B,HK,128), v (B,HK,128), rel (B,H,16) [or pb (B,H,R) if rel_w is given]) bf16 and new conv
    states (w0, w1) (B,3,HK*128) f32 [+ k_conv, v_conv (B, HK*128) bf16 if debug]."""
    return _run_mid(layer, q_proj, r_proj, k_proj, v_proj, None, None, cs0, cs1, tau, k_wproj, v_wproj, debug, rel_w=rel_w)


def _unpack_flat(layout):
    offs, strs = {}, {}
    for item in layout:
        nm = item[0]
        if len(item) == 3:
            offs[nm], strs[nm] = item[1], item[2]
        else:
            offs[nm], strs[nm] = item[1], item[2]
    return offs, strs


def mid_rows_flat(layer, flat_buf, layout, cs0, cs1, tau=None, debug=False, rel_w=None, packed_kv=False, packed_kv_b=False):
    """K5 MID reading directly from K4's flat output buffer `(flat_buf, layout)` with zero intermediate slice views."""
    offs, strs = _unpack_flat(layout)
    return _run_mid(
        layer, None, None, None, None, flat_buf, offs, cs0, cs1, tau, None, None, debug,
        rel_w=rel_w, packed_kv=packed_kv, flat_strs=strs, packed_kv_b=packed_kv_b,
    )


def mid_causal_w_flat(layer, flat_buf, layout, cs0, cs1, W, *, tau=None, rel_w=None, conv_anc=None, w_act=None):
    """Causal W-token/tree K5 MID reading W rows from K4's flat buffer and single pre-step conv states cs0/cs1 (1, 3, C)."""
    offs, strs = _unpack_flat(layout)
    return _run_mid(
        layer, None, None, None, None, flat_buf, offs, cs0, cs1, tau, None, None, False,
        rel_w=rel_w, packed_kv=True, causal_w_B=W, conv_anc=conv_anc, w_act=w_act, flat_strs=strs,
    )
