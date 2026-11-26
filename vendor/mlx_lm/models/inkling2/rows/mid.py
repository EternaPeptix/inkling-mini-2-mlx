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
    threadgroup float vbuf[8 * {"256" if fused_rel else "128"}];
    float tau_v = {"tau[b]" if full else "1.0f"};{flat_ptrs}
    if (role < 3) {{                          // q: head = 8*role + sg
      uint h = role * 8 + sg;
      const device bfloat16_t* x = q_proj + b * {str_q} + h * 128 + lane * 4;
      bfloat4 xb = as_type<bfloat4>(((const device uint2*)x)[0]);
      uint qoff = h * 128 + lane * 4;
      float s[4] = {{
          float(xb.x) * float(q_scale[qoff + 0]),
          float(xb.y) * float(q_scale[qoff + 1]),
          float(xb.z) * float(q_scale[qoff + 2]),
          float(xb.w) * float(q_scale[qoff + 3]),
      }};
      glue_l2_clip_lane128(s, {flit(thr_q)});
      bfloat16_t qo[4];
      for (int i = 0; i < 4; i++) {{
        bfloat16_t o = bfloat16_t(s[i]);
        {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
        qo[i] = o;
      }}
      bfloat4 ob = bfloat4(qo[0], qo[1], qo[2], qo[3]);
      ((device uint2*)(q_out + b * (H * 128) + qoff))[0] = as_type<uint2>(ob);
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
      bfloat4 x0b = as_type<bfloat4>(((const device uint2*)(xp + b * st_x + c0))[0]);
      float4 x0f = float4(float(x0b.x), float(x0b.y), float(x0b.z), float(x0b.w));
      float4 s0f, s1f, s2f;
      if (a0 >= 0) {{ bfloat4 tb = as_type<bfloat4>(((const device uint2*)(xp + uint(a0) * st_x + c0))[0]); s0f = float4(float(tb.x), float(tb.y), float(tb.z), float(tb.w)); }}
      else {{ s0f = ((const device float4*)(st + uint(-1 - a0) * C + c0))[0]; }}
      if (a1 >= 0) {{ bfloat4 tb = as_type<bfloat4>(((const device uint2*)(xp + uint(a1) * st_x + c0))[0]); s1f = float4(float(tb.x), float(tb.y), float(tb.z), float(tb.w)); }}
      else {{ s1f = ((const device float4*)(st + uint(-1 - a1) * C + c0))[0]; }}
      if (a2 >= 0) {{ bfloat4 tb = as_type<bfloat4>(((const device uint2*)(xp + uint(a2) * st_x + c0))[0]); s2f = float4(float(tb.x), float(tb.y), float(tb.z), float(tb.w)); }}
      else {{ s2f = ((const device float4*)(st + uint(-1 - a2) * C + c0))[0]; }}
      ((device float4*)(nst + c0))[0] = s1f;
      ((device float4*)(nst + C + c0))[0] = s2f;
      ((device float4*)(nst + 2 * C + c0))[0] = x0f;
"""
    else:
        src += f"""
      const device bfloat16_t* xp = isv ? (v_proj + b * {str_v}) : (k_proj + b * {str_k});
      const device float* st = (isv ? cs1 : cs0) + b * 3 * C;
      device float* nst = (isv ? w1 : w0) + b * 3 * C;
      bfloat4 x0b = as_type<bfloat4>(((const device uint2*)(xp + c0))[0]);
      float4 x0f = float4(float(x0b.x), float(x0b.y), float(x0b.z), float(x0b.w));
      float4 s0f = ((const device float4*)(st + c0))[0];
      float4 s1f = ((const device float4*)(st + C + c0))[0];
      float4 s2f = ((const device float4*)(st + 2 * C + c0))[0];
      ((device float4*)(nst + c0))[0] = s1f;
      ((device float4*)(nst + C + c0))[0] = s2f;
      ((device float4*)(nst + 2 * C + c0))[0] = x0f;
"""
    if full:
        src += f"""
      const device bfloat16_t* wp = (isv ? (v_wproj + b * {str_vw}) : (k_wproj + b * {str_kw})) + sg * 4;
      bfloat4 dyp = as_type<bfloat4>(((const device uint2*)wp)[0]);
      float dy[4] = {{float(dyp.x), float(dyp.y), float(dyp.z), float(dyp.w)}};
      float x0a[4] = {{x0f.x, x0f.y, x0f.z, x0f.w}};
      float s0a[4] = {{s0f.x, s0f.y, s0f.z, s0f.w}};
      float s1a[4] = {{s1f.x, s1f.y, s1f.z, s1f.w}};
      float s2a[4] = {{s2f.x, s2f.y, s2f.z, s2f.w}};
      float y[4];
      for (int i = 0; i < 4; i++) {{
        uint c = c0 + i;
        float sw[4];
        for (int k = 0; k < 4; k++) sw[k] = float((isv ? v_static : k_static)[k * C + c]);
        y[i] = mid_conv_d(x0a[i], s0a[i], s1a[i], s2a[i], sw, dy);
      }}
"""
    else:
        src += f"""
      float x0a[4] = {{x0f.x, x0f.y, x0f.z, x0f.w}};
      float s0a[4] = {{s0f.x, s0f.y, s0f.z, s0f.w}};
      float s1a[4] = {{s1f.x, s1f.y, s1f.z, s1f.w}};
      float s2a[4] = {{s2f.x, s2f.y, s2f.z, s2f.w}};
      const device bfloat16_t* cwp = (isv ? v_conv : k_conv) + c0 * 4;
      uint4 cwq0 = ((const device uint4*)cwp)[0];
      uint4 cwq1 = ((const device uint4*)cwp)[1];
      bfloat4 cwa[4] = {{ as_type<bfloat4>(cwq0.xy), as_type<bfloat4>(cwq0.zw), as_type<bfloat4>(cwq1.xy), as_type<bfloat4>(cwq1.zw) }};
      float y[4];
      for (int i = 0; i < 4; i++) {{
        bfloat4 cwb = cwa[i];
        y[i] = mid_conv_s(x0a[i], s0a[i], s1a[i], s2a[i], float(cwb.x), float(cwb.y), float(cwb.z), float(cwb.w), {flit(np.float32(math.sqrt(5)))});
      }}
"""
    if causal_w:
        k_dst_ptr = "kv_out + (sg * WW + b) * 128 + lane * 4"
        v_dst_ptr = "kv_out + ((HK + sg) * WW + b) * 128 + lane * 4"
    elif packed_kv_b:
        k_dst_ptr = "kv_out + ((b * 2) * HK + sg) * 128 + lane * 4"
        v_dst_ptr = "kv_out + ((b * 2 + 1) * HK + sg) * 128 + lane * 4"
    else:
        k_dst_ptr = "kv_out + c0" if packed_kv else "k_out + b * C + c0"
        v_dst_ptr = f"kv_out + {C} + c0" if packed_kv else "v_out + b * C + c0"
    src += ("""
      { bfloat4 db = bfloat4(bfloat16_t(y[0]), bfloat16_t(y[1]), bfloat16_t(y[2]), bfloat16_t(y[3]));
        ((device uint2*)((isv ? vconv_dbg : kconv_dbg) + b * C + c0))[0] = as_type<uint2>(db); }
""" if debug else "") + f"""
      if (!isv) {{
        glue_l2_clip_lane128(y, {flit(thr_k)});
        bfloat4 kb = bfloat4(bfloat16_t(y[0]), bfloat16_t(y[1]), bfloat16_t(y[2]), bfloat16_t(y[3]));
        ((device uint2*)({k_dst_ptr}))[0] = as_type<uint2>(kb);
      }} else {{
        threadgroup float* vb = vbuf + sg * 128;
        ((threadgroup float4*)(vb + lane * 4))[0] = float4(y[0], y[1], y[2], y[3]);
        float inv = glue_rms_inv_sg<128, true>(vb, {flit(eps)}, lane);
        bfloat4 vb4 = bfloat4(
            glue_rms_apply_nw<bfloat16_t>(y[0], inv),
            glue_rms_apply_nw<bfloat16_t>(y[1], inv),
            glue_rms_apply_nw<bfloat16_t>(y[2], inv),
            glue_rms_apply_nw<bfloat16_t>(y[3], inv));
        ((device uint2*)({v_dst_ptr}))[0] = as_type<uint2>(vb4);
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
      simdgroup_bfloat8x8 mb00, mb10, mb01, mb11;
      simdgroup_load(mb00, Bm + 0 * R + 0, R);
      simdgroup_load(mb10, Bm + 8 * R + 0, R);
      simdgroup_load(mb01, Bm + 0 * R + 8, R);
      simdgroup_load(mb11, Bm + 8 * R + 8, R);
      int r = int(lane >> 2), c = int(lane & 3) * 4;
      for (int mi = 0; mi < 3; mi++) {{
        simdgroup_bfloat8x8 ma0, ma1;
        simdgroup_load(ma0, rel_s + (mi * 8) * 16 + 0, 16);
        simdgroup_load(ma1, rel_s + (mi * 8) * 16 + 8, 16);
        simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_multiply_accumulate(acc0, ma0, mb00, acc0);
        simdgroup_multiply_accumulate(acc0, ma1, mb10, acc0);
        simdgroup_multiply_accumulate(acc1, ma0, mb01, acc1);
        simdgroup_multiply_accumulate(acc1, ma1, mb11, acc1);
        threadgroup float* st = vbuf + (sg * 2 + (mi & 1)) * 128;
        simdgroup_store(acc0, st + 0, 16);
        simdgroup_store(acc1, st + 8, 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float4 fv = ((const threadgroup float4*)(st + r * 16 + c))[0];
        bfloat4 bv = bfloat4(bfloat16_t(fv.x), bfloat16_t(fv.y), bfloat16_t(fv.z), bfloat16_t(fv.w));
        ((device uint2*)(Cm + (mi * 8 + r) * R + c))[0] = as_type<uint2>(bv);
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
_FLAT_TMPL_CACHE = {}
_MID_FAST_CACHE = {}
_MID_DTS_PKV_REL = [mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32, mx.float32]
_MID_DTS_UNP_REL = [mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32, mx.float32]
_MID_TG = (256, 1, 1)


def _default_conv_anc(W):
    a = _CHAIN_CONV_ANC.get(W)
    if a is None:
        rows = [[b - 3 if b >= 3 else -(b + 1), b - 2 if b >= 2 else -(b + 2), b - 1 if b >= 1 else -3] for b in range(W)]
        a = _CHAIN_CONV_ANC[W] = mx.array(rows, mx.int32)
    return a


def _flat_tmpl(layout, full, H, HK, C, causal_w_B):
    lt = tuple(layout) if isinstance(layout, list) else layout
    ck = (lt, full, H, HK, causal_w_B)
    t = _FLAT_TMPL_CACHE.get(ck)
    if t is None:
        offs, strs = _unpack_flat(lt)
        t = ([("WW", causal_w_B)] if causal_w_B > 0 else []) + [
            ("OFF_Q", offs["q_proj"]), ("OFF_R", offs["r_proj"]),
            ("OFF_K", offs["k_proj"]), ("OFF_V", offs["v_proj"]),
            ("STR_Q", strs.get("q_proj", H * 128)), ("STR_R", strs.get("r_proj", H * 16)),
            ("STR_K", strs.get("k_proj", C)), ("STR_V", strs.get("v_proj", C)),
        ]
        if full:
            t = t + [
                ("OFF_KW", offs["k_wproj"]), ("OFF_VW", offs["v_wproj"]),
                ("STR_KW", strs.get("k_wproj", HK * 4)), ("STR_VW", strs.get("v_wproj", HK * 4)),
            ]
        _FLAT_TMPL_CACHE[ck] = t
    return t


def _run_mid(
    layer, q_proj, r_proj, k_proj, v_proj, flat_buf, flat_layout, cs0, cs1, tau, k_wproj, v_wproj, debug,
    rel_w=None, packed_kv=False, causal_w_B=0, conv_anc=None, w_act=None, packed_kv_b=False,
):
    if flat_buf is not None and not debug:
        B_f = int(causal_w_B) if causal_w_B > 0 else cs0.shape[0]
        rel_r_f = rel_w.shape[1] if rel_w is not None else 0
        fkey = (id(layer), id(flat_layout), B_f, packed_kv, packed_kv_b, causal_w_B, rel_r_f, w_act is not None)
        ent = _MID_FAST_CACHE.get(fkey)
        if ent is not None and ent[0] is flat_layout:
            (_lay_ref, a_ref, full_f, ql2, rl2, ksconv, vsconv, w_refs, k, tmpl, grid, shapes, dts, C_f) = ent
            if isinstance(a_ref, dict):
                q_sc = ql2["scale"]
                r_sc = rl2["scale"]
                cw0 = ksconv["static_weight"] if full_f else ksconv["conv1d"]["weight"]
                cw1 = vsconv["static_weight"] if full_f else vsconv["conv1d"]["weight"]
            else:
                q_sc = ql2.scale
                r_sc = rl2.scale
                cw0 = ksconv.static_weight if full_f else ksconv.conv1d.weight
                cw1 = vsconv.static_weight if full_f else vsconv.conv1d.weight
            if (q_sc is w_refs[0] and r_sc is w_refs[1] and cw0 is w_refs[2] and cw1 is w_refs[3]
                and flat_buf.ndim == 1 and flat_buf.dtype == mx.bfloat16
                and cs0.dtype == mx.float32 and cs1.dtype == mx.float32):
                ins = [flat_buf, cs0, cs1, q_sc, r_sc]
                if causal_w_B > 0:
                    ins.append(_default_conv_anc(B_f) if conv_anc is None else conv_anc)
                if full_f:
                    if tau.ndim != 1:
                        tau = tau.reshape(-1)
                    ins.append(tau)
                ins.append(cw0)
                ins.append(cw1)
                if rel_w is not None:
                    ins.append(rel_w)
                if w_act is not None:
                    ins.append(w_act if w_act.ndim == 1 else w_act.reshape(-1))
                return k(inputs=ins, grid=grid, threadgroup=_MID_TG, output_shapes=shapes, output_dtypes=dts, template=tmpl)
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
        if cs0.ndim != 3 or cs0.shape[1] != 3 or cs0.shape[2] != C or cs0.dtype != mx.float32 or cs1.shape != cs0.shape or cs1.dtype != mx.float32:
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
    if flat:
        if flat_buf.ndim != 1 or flat_buf.dtype != mx.bfloat16:
            raise ValueError("flat_buf must be 1D bf16")
        ins = [flat_buf, cs0, cs1, q_sc, r_sc]
    else:
        for x, shp, dt in ((q_proj, (B, H * 128), mx.bfloat16), (r_proj, (B, H * 16), mx.bfloat16), (k_proj, (B, C), mx.bfloat16),
                           (v_proj, (B, C), mx.bfloat16)):
            if x.shape != shp or x.dtype != dt:
                raise ValueError("bad input %s %s, want %s %s" % (x.shape, x.dtype, shp, dt))
        ins = [q_proj, r_proj, k_proj, v_proj, cs0, cs1, q_sc, r_sc]
    if q_sc.shape != (H, 128):
        raise ValueError("q scale shape")
    if causal_w:
        if conv_anc is None:
            conv_anc = _default_conv_anc(B)
        elif conv_anc.shape != (B, 3) or conv_anc.dtype != mx.int32:
            raise ValueError("conv_anc must be (B, 3) int32")
        ins.append(conv_anc)
    tmpl = _flat_tmpl(flat_layout, full, H, HK, C, causal_w_B) if flat else None
    sw_dt = None
    if full:
        if tau is None or (not flat and (k_wproj is None or v_wproj is None)):
            raise ValueError("full-attention layer needs tau, k_wproj, v_wproj")
        if tau.ndim != 1:
            tau = tau.reshape(-1)
        if tau.shape != (B,) or tau.dtype != mx.float32:
            raise ValueError("tau must be (B,) f32")
        ksw = ksconv["static_weight"] if isinstance(ksconv, dict) else ksconv.static_weight
        vsw = vsconv["static_weight"] if isinstance(vsconv, dict) else vsconv.static_weight
        sw_dt = (ksw.dtype, vsw.dtype)
        if flat:
            ins.append(tau)
            ins.append(ksw)
            ins.append(vsw)
        else:
            for w in (k_wproj, v_wproj):
                if w.shape != (B, HK * 4) or w.dtype != mx.bfloat16:
                    raise ValueError("wproj must be (B, %d) bf16" % (HK * 4))
            ins.append(tau)
            ins.append(k_wproj)
            ins.append(v_wproj)
            ins.append(ksw)
            ins.append(vsw)
    else:
        if tau is not None:
            raise ValueError("sliding layer takes no tau")
        kcw = ksconv["conv1d"]["weight"] if isinstance(ksconv, dict) else ksconv.conv1d.weight
        vcw = vsconv["conv1d"]["weight"] if isinstance(vsconv, dict) else vsconv.conv1d.weight
        if kcw.dtype != mx.bfloat16 or kcw.shape != (C, 4, 1) or vcw.dtype != mx.bfloat16 or vcw.shape != (C, 4, 1):
            raise ValueError("conv weight must be (C,4,1) bf16")
        ins.append(kcw)
        ins.append(vcw)
    rel_r = 0
    kv_w = B if causal_w else 1
    kv_shape = (B, 2, HK, 128) if packed_kv_b else (2, HK, kv_w, 128)
    any_pkv = packed_kv or packed_kv_b
    if rel_w is not None:
        if debug:
            raise ValueError("mid_rows rel_w fusion does not support debug=True")
        rel_r = rel_w.shape[1]
        ins.append(rel_w)
        if any_pkv:
            shapes = [(B, H, 128), kv_shape, (B, H, rel_r), (B, 3, C), (B, 3, C)]
            dts = _MID_DTS_PKV_REL
        else:
            shapes = [(B, H, 128), (B, HK, 128), (B, HK, 128), (B, H, rel_r), (B, 3, C), (B, 3, C)]
            dts = _MID_DTS_UNP_REL
        ntg = 5 + (rel_r >> 7)
    else:
        if any_pkv:
            shapes = [(B, H, 128), kv_shape, (B, H, 16), (B, 3, C), (B, 3, C)]
            dts = _MID_DTS_PKV_REL
        else:
            shapes = [(B, H, 128), (B, HK, 128), (B, HK, 128), (B, H, 16), (B, 3, C), (B, 3, C)] + ([(B, C)] * 2 if debug else [])
            dts = _MID_DTS_UNP_REL if not debug else (_MID_DTS_UNP_REL + [mx.bfloat16, mx.bfloat16])
        ntg = 6
    has_wa = w_act is not None
    if has_wa:
        ins.append(w_act if w_act.ndim == 1 else w_act.reshape(-1))
    key = (flat, full, rel_r, packed_kv, causal_w, has_wa, packed_kv_b, H, HK, ql2.threshold, kl2.threshold, rl2.threshold, a.eps, debug,
           q_sc.dtype, r_sc.dtype, sw_dt)
    k = _cache.get(key)
    if k is None:
        assert a.head_dim == 128 and r_sc.shape == (H, 16) and ksconv.kernel_size == 4 and H <= 32, "unsupported config"
        assert HK <= 8 and H <= 24
        if full:
            assert ksw.shape == (4, C) and ksconv.num_heads == HK and vsw.shape == (4, C) and vsconv.num_heads == HK
        if rel_w is not None and (rel_w.dtype != mx.bfloat16 or rel_w.ndim != 2 or rel_w.shape[0] != 16 or rel_w.shape[1] % 128 != 0 or H != 24):
            raise ValueError(f"unsupported rel_w shape/dtype {rel_w.shape} {rel_w.dtype} for H={H}")
        names = (["qkv", "cs0", "cs1", "q_scale", "rel_scale"] if flat else ["q_proj", "r_proj", "k_proj", "v_proj", "cs0", "cs1", "q_scale", "rel_scale"])
        if causal_w:
            names.append("conv_anc")
        if full:
            names += ["tau", "k_static", "v_static"] if flat else ["tau", "k_wproj", "v_wproj", "k_static", "v_static"]
        else:
            names += ["k_conv", "v_conv"]
        if rel_w is not None:
            names.append("rw")
            outs = ["q_out", "kv_out", "pb_out", "w0", "w1"] if any_pkv else ["q_out", "k_out", "v_out", "pb_out", "w0", "w1"]
        else:
            outs = ["q_out", "kv_out", "rel_out", "w0", "w1"] if any_pkv else (["q_out", "k_out", "v_out", "rel_out", "w0", "w1"] + (["kconv_dbg", "vconv_dbg"] if debug else []))
        if has_wa:
            names.append("w_act")
        src = _source(full, H, HK, ql2.threshold, kl2.threshold, rl2.threshold,
                      a.eps, debug, flat=flat, rel_r=rel_r, packed_kv=packed_kv, causal_w=causal_w, has_w_act=has_wa, packed_kv_b=packed_kv_b)
        k = _cache[key] = mx.fast.metal_kernel(
            name="mid_rows_%s%s%s%s%s%s%s_%d" % ("F" if full else "S", "_flat" if flat else "", "_rel" if rel_r else "", "_pkv" if packed_kv else "", "_pkb" if packed_kv_b else "", "_cw" if causal_w else "", "_wa" if has_wa else "", abs(hash(key)) % 10**9),
            input_names=names, output_names=outs, source=src, header=GLUE_HEADER + _MID_HEADER,
            ensure_row_contiguous=True)
    grid = (256 * ntg, B, 1)
    if flat and not debug:
        fkey = (id(layer), id(flat_layout), B, packed_kv, packed_kv_b, causal_w_B, rel_r, has_wa)
        if len(_MID_FAST_CACHE) >= 512:
            _MID_FAST_CACHE.clear()
        cw0 = ksw if full else kcw
        cw1 = vsw if full else vcw
        _MID_FAST_CACHE[fkey] = (flat_layout, a, full, ql2, rl2, ksconv, vsconv, (q_sc, r_sc, cw0, cw1), k, tmpl, grid, shapes, dts, C)
    if tmpl is not None:
        return k(inputs=ins, grid=grid, threadgroup=_MID_TG, output_shapes=shapes, output_dtypes=dts, template=tmpl)
    return k(inputs=ins, grid=grid, threadgroup=_MID_TG, output_shapes=shapes, output_dtypes=dts)


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
        offs[nm], strs[nm] = item[1], item[2]
    return offs, strs


def mid_rows_flat(layer, flat_buf, layout, cs0, cs1, tau=None, debug=False, rel_w=None, packed_kv=False, packed_kv_b=False):
    """K5 MID reading directly from K4's flat output buffer `(flat_buf, layout)` with zero intermediate slice views."""
    return _run_mid(
        layer, None, None, None, None, flat_buf, layout, cs0, cs1, tau, None, None, debug,
        rel_w=rel_w, packed_kv=packed_kv, packed_kv_b=packed_kv_b,
    )


def mid_causal_w_flat(layer, flat_buf, layout, cs0, cs1, W, *, tau=None, rel_w=None, conv_anc=None, w_act=None):
    """Causal W-token/tree K5 MID reading W rows from K4's flat buffer and single pre-step conv states cs0/cs1 (1, 3, C)."""
    return _run_mid(
        layer, None, None, None, None, flat_buf, layout, cs0, cs1, tau, None, None, False,
        rel_w=rel_w, packed_kv=True, causal_w_B=W, conv_anc=conv_anc, w_act=w_act,
    )
