"""Lane f1: row-local fused visit kernels K1 PRE, K3b NORM, K10 POST (docs/FUSED.md, docs/FUSED_ROWLOCAL.md).

One launch each, B rows (B = 1..), one threadgroup per row (K3b: per row and state).  Every output row is bitwise equal to
fused_ref.visit_ref evaluated on that row alone.  Nothing here is wired into the decode path.

Layout: D = 2304.  K1/K10 use a threadgroup of RL_TG = 288 threads = row_threads(2304), i.e. exactly the threadgroup of the
repo's `depth_attention` kernel (thread t owns channels t, t+288, ... t+2016 of the row), so the depth-attention arithmetic is
that kernel's code unchanged.  The rms_norms use the glue `glue_rms_inv_tg` (MLX's rms_single_row with 576 virtual threads, two
virtual simdgroups per real one).  Per-step values are arrays only; per-layer constants (block count, kind, eps) are in the
kernel source, so one kernel exists per (layer kind, block count).
"""

import functools
import math

import mlx.core as mx
import numpy as np

from .glue import GLUE_HEADER, flit

__all__ = ["k1_pre", "k3b_norm", "k10_post", "k1_pre_layer", "k10_post_layer", "k3b_norm_layer"]

D = 2304
TG = 288  # depth-attention threadgroup (row_threads(2304)); also the K1/K10 threadgroup
ND = D // TG  # 8 channels per thread
EPS_UW = 1e-6  # unweighted_rms_norm default
_BF = mx.bfloat16
_F32 = mx.float32

RL_HEADER = (
    GLUE_HEADER
    + r"""
#define RL_D 2304
#define RL_TG 288
#define RL_ND 8
#define RL_NSG 9

// Depth attention of Q queries over N sources for ONE row, thread tid of the 288-thread group.  x[i*RL_ND+k] = source i at
// channel tid + k*RL_TG.  Arithmetic copied from kernels._DEPTH_ATTENTION_SOURCE (default fp contraction, as the original; no
// GL_NOFMA here on purpose).  buf: threadgroup float[N*Q*RL_NSG].  Ends with a barrier (buf reusable).
template <int N, int Q>
inline void rl_depth(thread const float* x, const device float* queries, thread float* o, threadgroup float* buf,
                     uint tid, uint lane, uint sg, float scale) {
  float acc[N * Q];
  for (int i = 0; i < N * Q; i++) acc[i] = 0.0f;
  for (int k = 0; k < RL_ND; k++) {
    int d = int(tid) + k * RL_TG;
    for (int j = 0; j < Q; j++) {
      float qv = queries[j * RL_D + d];
      for (int i = 0; i < N; i++) acc[i * Q + j] += x[i * RL_ND + k] * qv;
    }
  }
  for (int i = 0; i < N * Q; i++) {
    float v = simd_sum(acc[i]);
    if (lane == 0) buf[i * RL_NSG + sg] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float w[N * Q];
  for (int i = 0; i < N * Q; i++) {
    float t = 0.0f;
    for (int s = 0; s < RL_NSG; s++) t += buf[i * RL_NSG + s];
    w[i] = t * scale;
  }
  for (int j = 0; j < Q; j++) {
    float m = -INFINITY;
    for (int i = 0; i < N; i++) m = max(m, w[i * Q + j]);
    float z = 0.0f;
    for (int i = 0; i < N; i++) {
      w[i * Q + j] = metal::exp(w[i * Q + j] - m);
      z += w[i * Q + j];
    }
    for (int i = 0; i < N; i++) w[i * Q + j] /= z;
  }
  for (int k = 0; k < RL_ND; k++)
    for (int j = 0; j < Q; j++) {
      float oo = 0.0f;
      for (int i = 0; i < N; i++) oo += w[i * Q + j] * x[i * RL_ND + k];
      o[j * RL_ND + k] = oo;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Several rms inverses of rows of 2304 in one pass: MLX rms_single_row (576 virtual threads) for R rows, TGN real threads
// (multiple of 32).  Row r is x[r] (pointer array of the same type).  part: threadgroup float[R * 32].  Same bits as R calls of
// glue_rms_inv_tg (the per-row arithmetic is identical); ends with a barrier.
template <int R, int TGN, typename P>
inline void rl_rms_inv_multi(thread P* x, thread float* inv, float eps, threadgroup float* part, uint lid) {
  GL_NOFMA
  constexpr int NSG = 18;
  uint lane = lid & 31, sg = lid >> 5;
  for (int g = sg; g < NSG; g += TGN / 32) {
    for (int r = 0; r < R; r++) {
      float acc = 0.0f;
      for (int i = 0; i < 4; i++) {
        int idx = (g * 32 + int(lane)) * 4 + i;
        float xi = (idx < RL_D) ? float(x[r][idx]) : 0.0f;
        acc = fma(xi, xi, acc);
      }
      acc = simd_sum(acc);
      if (lane == 0) part[r * 32 + g] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = 0; r < R; r++) {
    float tot = simd_sum(lane < uint(NSG) ? part[r * 32 + lane] : 0.0f);
    inv[r] = metal::precise::rsqrt(tot / float(RL_D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""
)


@functools.lru_cache(maxsize=None)
def _kernel(name, source, inputs, outputs):
    return mx.fast.metal_kernel(
        name=name, input_names=list(inputs), output_names=list(outputs), source=source, header=RL_HEADER
    )


def _check(x, shape_tail, dtype, name):
    if x.dtype != dtype or tuple(x.shape[-len(shape_tail):]) != tuple(shape_tail):
        raise ValueError("%s: expected dtype %s tail shape %s, got %s %s" % (name, dtype, shape_tail, x.dtype, x.shape))


_SCALE_DEPTH = flit(np.float32(D**-0.5))  # python float `D ** -0.5` -> f32 scalar array in the original
_SCALE_CONV = repr(float(np.float32(math.sqrt(5)))) + "f"  # fastdecode.short_conv_decode (kernel_size 4)


# ===================================================================================================== K1 PRE
def k1_pre(h, bs, msg, query, ln_w, *, sliding, eps_ln=1e-6, debug=False):
    """K1 PRE for B rows.

    h (B,D) bf16; bs (n,B,D) f32 or None (layer 0: no attention_input); msg (B,4,D) f32 or None (looped pass > 0);
    query: the layer's attention_input.query (f32; (D,) sliding / (3,D) full) or None iff bs is None;
    ln_w: layernorm weights, bf16 (D,): [input_layernorm] for sliding, [q, k, v input_layernorm] for full (ignored when msg).
    Returns a dict:
      normed_residual (B,D) f32                       always
      no msg:  q_in, k_in, v_in (B,D) bf16            sliding: ONE array (the same object three times)
      msg:     s_q, s_k, s_v (B,D) f32 (sliding: one array three times), q_gate_in, k_gate_in, v_gate_in (B,4608) bf16
      debug:   attn_input (B,G,D) f32 (G = 1 sliding, 3 full) when bs is given
    """
    B = h.shape[0]
    _check(h, (D,), _BF, "h")
    has_ai = query is not None
    if has_ai != (bs is not None):
        raise ValueError("query and bs must both be given or both None")
    if msg is not None:
        if not has_ai:
            raise ValueError("message visit without attention_input is not supported")
        _check(msg, (4, D), _F32, "msg")
    G = 1 if sliding else 3
    if has_ai:
        _check(bs, (B, D), _F32, "bs")
        query = query.reshape(-1, D)
        if query.shape[0] != G or query.dtype != _F32:
            raise ValueError("query shape/dtype")
    NB = bs.shape[0] if has_ai else 0
    if msg is None:
        if len(ln_w) != G:
            raise ValueError("need %d layernorm weights" % G)
        for w in ln_w:
            _check(w, (D,), _BF, "ln_w")

    ins = ["h"]
    args = [h]
    if has_ai:
        ins += ["bs", "query"]
        args += [bs, query]
    if msg is not None:
        ins += ["msg"]
        args += [msg]
    else:
        for g in range(G):
            ins.append("w%d" % g)
            args.append(ln_w[g])
    outs = ["nr"]
    shapes = [(B, D)]
    dts = [_F32]
    if msg is None:
        for g in range(G):
            outs.append("x%d" % g)
            shapes.append((B, D))
            dts.append(_BF)
    else:
        for g in range(G):
            outs.append("s%d" % g)
            shapes.append((B, D))
            dts.append(_F32)
        for g in range(3):
            outs.append("gi%d" % g)
            shapes.append((B, 2 * D))
            dts.append(_BF)
    if debug and has_ai:
        outs.append("ai")
        shapes.append((B, G, D))
        dts.append(_F32)

    s = [
        "uint row = threadgroup_position_in_grid.x;",
        "uint B = threadgroups_per_grid.x; (void)B;",
        "uint tid = thread_position_in_threadgroup.x;",
        "uint lane = thread_index_in_simdgroup; uint sg = simdgroup_index_in_threadgroup;",
        "threadgroup float part[96];",
        "threadgroup float dbuf[%d];" % max(1, (NB + 1) * G * 9),
        "const device bfloat16_t* hr = h + size_t(row) * RL_D;",
        "float inv0[1]; const device bfloat16_t* px[1] = {hr};",
        "rl_rms_inv_multi<1, RL_TG>(px, inv0, %s, part, tid);" % flit(EPS_UW),
        "float nrk[RL_ND];",
        "for (int k = 0; k < RL_ND; k++) { int d = int(tid) + k * RL_TG; nrk[k] = glue_rms_apply_nw<float>(float(hr[d]), inv0[0]);"
        " nr[size_t(row) * RL_D + d] = nrk[k]; }",
    ]
    if has_ai:
        s += [
            "threadgroup float sm[%d];" % (G * D),
            "float xs[%d];" % ((NB + 1) * ND),
            "for (int k = 0; k < RL_ND; k++) { int d = int(tid) + k * RL_TG;",
            "  for (int i = 0; i < %d; i++) xs[i * RL_ND + k] = bs[(size_t(i) * B + row) * RL_D + d];" % NB,
            "  xs[%d * RL_ND + k] = nrk[k]; }" % NB,
            "float o[%d];" % (G * ND),
            "rl_depth<%d, %d>(xs, query, o, dbuf, tid, lane, sg, %s);" % (NB + 1, G, _SCALE_DEPTH),
            "for (int j = 0; j < %d; j++) for (int k = 0; k < RL_ND; k++) sm[j * RL_D + int(tid) + k * RL_TG] = o[j * RL_ND + k];"
            % G,
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]
        if debug:
            s += [
                "for (int j = 0; j < %d; j++) for (int k = 0; k < RL_ND; k++) ai[(size_t(row) * %d + j) * RL_D + int(tid) + k * RL_TG]"
                " = o[j * RL_ND + k];" % (G, G)
            ]
    if msg is None:
        if has_ai:
            s += ["float invl[%d]; const threadgroup float* sp[%d] = {%s};"
                  % (G, G, ", ".join("sm + %d * RL_D" % g for g in range(G)))]
            s += ["rl_rms_inv_multi<%d, RL_TG>(sp, invl, %s, part, tid);" % (G, flit(eps_ln))]
            for g in range(G):
                s += ["for (int i = int(tid); i < RL_D; i += RL_TG) x%d[size_t(row) * RL_D + i] = bfloat16_t(glue_rms_apply<float>("
                      "sm[%d * RL_D + i], invl[%d], w%d[i]));" % (g, g, g, g)]
        else:  # layer 0 style: x = h (bf16 path, output dtype bf16)
            s += ["float invl[%d]; const device bfloat16_t* sp[%d] = {%s};" % (G, G, ", ".join(["hr"] * G))]
            s += ["rl_rms_inv_multi<%d, RL_TG>(sp, invl, %s, part, tid);" % (G, flit(eps_ln))]
            for g in range(G):
                s += ["for (int i = int(tid); i < RL_D; i += RL_TG) x%d[size_t(row) * RL_D + i] = glue_rms_apply<bfloat16_t>("
                      "float(hr[i]), invl[%d], w%d[i]);" % (g, g, g)]
    else:
        # messages: s_g, gate_in_i = bf16(concat(rms(s_g), rms(msg_i)))
        s += ["for (int j = 0; j < %d; j++) for (int k = 0; k < RL_ND; k++) { int d = int(tid) + k * RL_TG;"
              " if (j == 0) s0[size_t(row) * RL_D + d] = o[k];" % G]
        for g in range(1, G):
            s += ["  if (j == %d) s%d[size_t(row) * RL_D + d] = o[%d * RL_ND + k];" % (g, g, g)]
        s += ["}"]
        s += ["float ins_[%d]; const threadgroup float* sp[%d] = {%s};"
              % (G, G, ", ".join("sm + %d * RL_D" % g for g in range(G)))]
        s += ["rl_rms_inv_multi<%d, RL_TG>(sp, ins_, %s, part, tid);" % (G, flit(EPS_UW))]
        s += ["float inm[3]; const device float* mp[3] = {msg + size_t(row) * 4 * RL_D, msg + size_t(row) * 4 * RL_D + RL_D,"
              " msg + size_t(row) * 4 * RL_D + 2 * RL_D};",
              "rl_rms_inv_multi<3, RL_TG>(mp, inm, %s, part, tid);" % flit(EPS_UW)]
        for i in range(3):
            g = 0 if sliding else i
            s += ["for (int d = int(tid); d < RL_D; d += RL_TG) {",
                  "  gi%d[size_t(row) * 2 * RL_D + d] = bfloat16_t(glue_rms_apply_nw<float>(sm[%d * RL_D + d], ins_[%d]));" % (i, g, g),
                  "  gi%d[size_t(row) * 2 * RL_D + RL_D + d] = bfloat16_t(glue_rms_apply_nw<float>(mp[%d][d], inm[%d])); }" % (i, i, i)]
    src = "\n".join(s)
    name = "rl_k1_%s_%d_%d_%d_%s_%s" % ("S" if sliding else "F", NB, int(has_ai), int(msg is not None), flit(eps_ln),
                                       "dbg" if debug else "")
    k = _kernel(name.replace(".", "p").replace("-", "m"), src, tuple(ins), tuple(outs))
    r = k(inputs=args, grid=(B * TG, 1, 1), threadgroup=(TG, 1, 1), output_shapes=shapes, output_dtypes=dts)
    o = dict(zip(outs, r))
    res = {"normed_residual": o["nr"]}
    if msg is None:
        if sliding:
            res.update(q_in=o["x0"], k_in=o["x0"], v_in=o["x0"])
        else:
            res.update(q_in=o["x0"], k_in=o["x1"], v_in=o["x2"])
    else:
        if sliding:
            res.update(s_q=o["s0"], s_k=o["s0"], s_v=o["s0"])
        else:
            res.update(s_q=o["s0"], s_k=o["s1"], s_v=o["s2"])
        res.update(q_gate_in=o["gi0"], k_gate_in=o["gi1"], v_gate_in=o["gi2"])
    if "ai" in o:
        res["attn_input"] = o["ai"]
    return res


def k1_pre_layer(layer, h, bs, msg, *, debug=False):
    """K1 on a decoder layer module (h (B,D) bf16, bs (n,B,D) f32 or None, msg (B,4,D) f32 or None)."""
    from mlx_lm.models.inkling2 import layers as L

    sliding = isinstance(layer, L.SlidingDecoderLayer)
    q = layer.attention_input.query if layer.attention_input is not None else None
    if sliding:
        ln = [layer.input_layernorm]
    else:
        ln = [layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm]
    return k1_pre(h, bs, msg, q, [m.weight for m in ln], sliding=sliding, eps_ln=ln[0].eps, debug=debug)


# ===================================================================================================== K3b NORM
def k3b_norm(mixed, ln_w, *, eps_ln=1e-6):
    """mixed (B,G,D) f32 (G = 3: q,k,v states; G = 1: mlp state) -> tuple of G arrays (B,D) bf16,
    out_g = bf16(rms_norm(mixed[:,g], ln_w[g])).  ln_w: list of G bf16 (D,) weights (repeat the object for sliding layers)."""
    B, G, _ = mixed.shape
    _check(mixed, (G, D), _F32, "mixed")
    if len(ln_w) != G:
        raise ValueError("need %d weights" % G)
    for w in ln_w:
        _check(w, (D,), _BF, "ln_w")
    NT = 576
    ins = ["x"] + ["w%d" % g for g in range(G)]
    outs = ["o%d" % g for g in range(G)]
    s = [
        "uint t = threadgroup_position_in_grid.x; uint lid = thread_position_in_threadgroup.x;",
        "uint row = t / %d, g = t %% %d;" % (G, G),
        "threadgroup float part[32];",
        "const device float* xr = x + (size_t(row) * %d + g) * RL_D;" % G,
        "float inv = glue_rms_inv_tg<RL_D, %d, true>(xr, %s, part, lid);" % (NT, flit(eps_ln)),
    ]
    for g in range(G):
        s += ["if (g == %d) { for (int i = int(lid); i < RL_D; i += %d) o%d[size_t(row) * RL_D + i] = bfloat16_t(glue_rms_apply<float>("
              "xr[i], inv, w%d[i])); }" % (g, NT, g, g)]
    k = _kernel(("rl_k3b_%d_%s" % (G, flit(eps_ln))).replace(".", "p").replace("-", "m"), "\n".join(s), tuple(ins), tuple(outs))
    r = k(inputs=[mixed] + list(ln_w), grid=(B * G * NT, 1, 1), threadgroup=(NT, 1, 1),
          output_shapes=[(B, D)] * G, output_dtypes=[_BF] * G)
    return tuple(r)


def k3b_norm_layer(layer, mixed, *, mlp=False):
    """mixed (B,3,D) for q/k/v (sliding: input_layernorm for all three) or (B,1,D) with mlp=True (post_attention_layernorm)."""
    from mlx_lm.models.inkling2 import layers as L

    if mlp:
        m = layer.post_attention_layernorm
        return k3b_norm(mixed, [m.weight], eps_ln=m.eps)
    if isinstance(layer, L.SlidingDecoderLayer):
        m = layer.input_layernorm
        return k3b_norm(mixed, [m.weight] * 3, eps_ln=m.eps)
    ms = [layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm]
    return k3b_norm(mixed, [m.weight for m in ms], eps_ln=ms[0].eps)


# ===================================================================================================== K10 POST
def k10_post(o_proj, h, nr, bs, cs2, conv_w, query, ln_w, msg, *, boundary, eps_ln=1e-6):
    """K10 POST for B rows.

    o_proj (B,D) bf16 (o_proj output), h (B,D) bf16, nr (B,D) f32 (K1's normed_residual), bs (n,B,D) f32 or None (layer 0),
    cs2 (B,3,D) f32 post-attention conv state, conv_w post_attention_convolution.conv1d.weight (D,4,1) bf16,
    query mlp_input.query (D,) f32, ln_w post_attention_layernorm.weight (D,) bf16, msg (B,4,D) f32 or None.
    boundary: layer is a block boundary (residual := conv out; the new block is nr, appended to bs for mlp_input).
    Returns dict: post_attn_conv (B,D) bf16, w2 (B,3,D) f32, residual (B,D) f32, normed (B,D) f32, mlp_src (B,D) f32, and
      no msg: mlp_x (B,D) bf16;  msg: mlp_gate_in (B,4608) bf16.
    bs_new (boundary) = concat(bs, nr[None]) is NOT materialized (the caller stacks it if it needs it).
    """
    B = h.shape[0]
    for x, dt, nm in ((o_proj, _BF, "o_proj"), (h, _BF, "h"), (nr, _F32, "nr"), (ln_w, _BF, "ln_w")):
        _check(x, (D,), dt, nm)
    _check(cs2, (3, D), _F32, "cs2")
    if conv_w.dtype != _BF or conv_w.size != D * 4:
        raise ValueError("conv weight must be bf16 (D,4,1)")
    NB = bs.shape[0] if bs is not None else 0
    if bs is not None:
        _check(bs, (B, D), _F32, "bs")
    if NB == 0 and not boundary:
        raise ValueError("no blocks to attend over")
    NS = NB + (1 if boundary else 0) + 1  # sources: blocks, [nr], partial
    if msg is not None:
        _check(msg, (4, D), _F32, "msg")
    query = query.reshape(-1, D)
    if query.shape[0] != 1 or query.dtype != _F32:
        raise ValueError("query")

    ins = ["op", "h", "nr"] + (["bs"] if NB else []) + ["cs", "cw", "query", "w"] + (["msg"] if msg is not None else [])
    args = [o_proj, h, nr] + ([bs] if NB else []) + [cs2, conv_w, query, ln_w] + ([msg] if msg is not None else [])
    outs = ["conv", "w2", "res", "normed", "src"] + (["gi"] if msg is not None else ["mx"])
    shapes = [(B, D), (B, 3, D), (B, D), (B, D), (B, D), (B, 2 * D) if msg is not None else (B, D)]
    dts = [_BF, _F32, _F32, _F32, _F32, _BF]

    s = [
        "uint row = threadgroup_position_in_grid.x;",
        "uint B = threadgroups_per_grid.x; (void)B;",
        "uint tid = thread_position_in_threadgroup.x;",
        "uint lane = thread_index_in_simdgroup; uint sg = simdgroup_index_in_threadgroup;",
        "threadgroup float part[64];",
        "threadgroup float dbuf[%d];" % (NS * 9),
        "threadgroup float sm[RL_D];",
        "float res[RL_ND];",
        # short conv (fastdecode._CONV_SRC, T = bf16), per channel
        "for (int k = 0; k < RL_ND; k++) {",
        "  int c = int(tid) + k * RL_TG;",
        "  float x0 = float(op[size_t(row) * RL_D + c]);",
        "  const device float* st = cs + size_t(row) * 3 * RL_D;",
        "  float r1 = st[RL_D + c], r2 = st[2 * RL_D + c];",
        "  float s0 = float(bfloat16_t(st[c])), s1 = float(bfloat16_t(r1)), s2 = float(bfloat16_t(r2));",
        "  float acc = 0.0f;",
        "  acc = fma(float(cw[c * 4 + 0]), s0, acc);",
        "  acc = fma(float(cw[c * 4 + 1]), s1, acc);",
        "  acc = fma(float(cw[c * 4 + 2]), s2, acc);",
        "  acc = fma(float(cw[c * 4 + 3]), x0, acc);",
        "  float cv = float(bfloat16_t(acc));",
        "  bfloat16_t co = bfloat16_t((cv + x0) / %s);" % _SCALE_CONV,
        "  conv[size_t(row) * RL_D + c] = co;",
        "  w2[size_t(row) * 3 * RL_D + c] = r1; w2[size_t(row) * 3 * RL_D + RL_D + c] = r2; w2[size_t(row) * 3 * RL_D + 2 * RL_D + c] = x0;",
        "  float hid = float(co);",
        "  res[k] = %s;" % ("hid" if boundary else "float(h[size_t(row) * RL_D + c]) + hid"),
        "  res_o[size_t(row) * RL_D + c] = res[k];",
        "  sm[c] = res[k];",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "float inv1[1]; const threadgroup float* p1[1] = {sm};",
        "rl_rms_inv_multi<1, RL_TG>(p1, inv1, %s, part, tid);" % flit(EPS_UW),
        "float xs[%d];" % (NS * ND),
        "for (int k = 0; k < RL_ND; k++) { int d = int(tid) + k * RL_TG;",
        "  float nm = glue_rms_apply_nw<float>(res[k], inv1[0]);",
        "  normed[size_t(row) * RL_D + d] = nm;",
    ]
    if NB:
        s += ["  for (int i = 0; i < %d; i++) xs[i * RL_ND + k] = bs[(size_t(i) * B + row) * RL_D + d];" % NB]
    if boundary:
        s += ["  xs[%d * RL_ND + k] = nr[size_t(row) * RL_D + d];" % NB]
    s += [
        "  xs[%d * RL_ND + k] = nm; }" % (NS - 1),
        "float o[RL_ND];",
        "rl_depth<%d, 1>(xs, query, o, dbuf, tid, lane, sg, %s);" % (NS, _SCALE_DEPTH),
        "for (int k = 0; k < RL_ND; k++) { int d = int(tid) + k * RL_TG; src[size_t(row) * RL_D + d] = o[k]; sm[d] = o[k]; }",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
    ]
    if msg is None:
        s += [
            "float inv2[1]; const threadgroup float* p2[1] = {sm};",
            "rl_rms_inv_multi<1, RL_TG>(p2, inv2, %s, part, tid);" % flit(eps_ln),
            "for (int d = int(tid); d < RL_D; d += RL_TG) mx[size_t(row) * RL_D + d] = bfloat16_t(glue_rms_apply<float>("
            "sm[d], inv2[0], w[d]));",
        ]
    else:
        s += [
            "float inv2[2]; const threadgroup float* p2[1] = {sm};",
            "rl_rms_inv_multi<1, RL_TG>(p2, inv2, %s, part, tid);" % flit(EPS_UW),
            "const device float* mp[1] = {msg + size_t(row) * 4 * RL_D + 3 * RL_D};",
            "rl_rms_inv_multi<1, RL_TG>(mp, inv2 + 1, %s, part, tid);" % flit(EPS_UW),
            "for (int d = int(tid); d < RL_D; d += RL_TG) {",
            "  gi[size_t(row) * 2 * RL_D + d] = bfloat16_t(glue_rms_apply_nw<float>(sm[d], inv2[0]));",
            "  gi[size_t(row) * 2 * RL_D + RL_D + d] = bfloat16_t(glue_rms_apply_nw<float>(mp[0][d], inv2[1])); }",
        ]
    src = "\n".join(s)
    outs = ["conv", "w2", "res_o", "normed", "src"] + (["gi"] if msg is not None else ["mx"])
    name = "rl_k10_%d_%d_%d_%s" % (NB, int(boundary), int(msg is not None), flit(eps_ln))
    k = _kernel(name.replace(".", "p").replace("-", "m"), src, tuple(ins), tuple(outs))
    r = k(inputs=args, grid=(B * TG, 1, 1), threadgroup=(TG, 1, 1), output_shapes=shapes, output_dtypes=dts)
    o = dict(zip(outs, r))
    res = dict(post_attn_conv=o["conv"], w2=o["w2"], residual=o["res_o"], normed=o["normed"], mlp_src=o["src"])
    if msg is None:
        res["mlp_x"] = o["mx"]
    else:
        res["mlp_gate_in"] = o["gi"]
    return res


def k10_post_layer(layer, o_proj, h, nr, bs, cs2, msg):
    m = layer.post_attention_layernorm
    return k10_post(o_proj, h, nr, bs, cs2, layer.post_attention_convolution.conv1d.weight, layer.mlp_input.query, m.weight,
                    msg, boundary=layer.is_block_boundary, eps_ln=m.eps)
