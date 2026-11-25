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

__all__ = [
    "k1_pre", "k3b_norm", "k10_post",
    "k1_pre_layer", "k10_post_layer", "k3b_norm_layer",
    "k1_from_down_causal", "k1_from_down_causal_layer",
]

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
}

// Compute R1 + R2 rms inverses (possibly different pointer address spaces) with a single barrier.
template <int R1, int R2, int TGN, typename P1, typename P2>
inline void rl_rms_inv_two(thread P1* x1, thread float* inv1, thread P2* x2, thread float* inv2,
                           float eps, threadgroup float* part, uint lid) {
  GL_NOFMA
  constexpr int NSG = 18;
  uint lane = lid & 31, sg = lid >> 5;
  for (int g = sg; g < NSG; g += TGN / 32) {
    for (int r = 0; r < R1; r++) {
      float acc = 0.0f;
      for (int i = 0; i < 4; i++) {
        int idx = (g * 32 + int(lane)) * 4 + i;
        float xi = (idx < RL_D) ? float(x1[r][idx]) : 0.0f;
        acc = fma(xi, xi, acc);
      }
      acc = simd_sum(acc);
      if (lane == 0) part[r * 32 + g] = acc;
    }
    for (int r = 0; r < R2; r++) {
      float acc = 0.0f;
      for (int i = 0; i < 4; i++) {
        int idx = (g * 32 + int(lane)) * 4 + i;
        float xi = (idx < RL_D) ? float(x2[r][idx]) : 0.0f;
        acc = fma(xi, xi, acc);
      }
      acc = simd_sum(acc);
      if (lane == 0) part[(R1 + r) * 32 + g] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = 0; r < R1; r++) {
    float tot = simd_sum(lane < uint(NSG) ? part[r * 32 + lane] : 0.0f);
    inv1[r] = metal::precise::rsqrt(tot / float(RL_D) + eps);
  }
  for (int r = 0; r < R2; r++) {
    float tot = simd_sum(lane < uint(NSG) ? part[(R1 + r) * 32 + lane] : 0.0f);
    inv2[r] = metal::precise::rsqrt(tot / float(RL_D) + eps);
  }
}

// Contiguous rms_inv directly from registers (no threadgroup float[D] roundtrip):
// thread (sg, lane) owns pass=0..1, i=0..3 at base = ((sg + pass*9)*32 + lane)*4.
// Identical virtual-576-thread reduction order as rl_rms_inv_multi / MLX rms_single_row.
inline float rl_rms_inv_reg(float v00, float v01, float v02, float v03,
                            float v10, float v11, float v12, float v13,
                            float eps, threadgroup float* part, uint sg, uint lane) {
  GL_NOFMA
  float a0 = 0.0f;
  a0 = fma(v00, v00, a0); a0 = fma(v01, v01, a0); a0 = fma(v02, v02, a0); a0 = fma(v03, v03, a0);
  a0 = simd_sum(a0);
  if (lane == 0) part[sg] = a0;
  float a1 = 0.0f;
  a1 = fma(v10, v10, a1); a1 = fma(v11, v11, a1); a1 = fma(v12, v12, a1); a1 = fma(v13, v13, a1);
  a1 = simd_sum(a1);
  if (lane == 0) part[sg + 9] = a1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = simd_sum(lane < 18u ? part[lane] : 0.0f);
  return metal::precise::rsqrt(tot / float(RL_D) + eps);
}

inline void rl_rms_inv_reg2(float v00, float v01, float v02, float v03,
                            float v10, float v11, float v12, float v13,
                            float m00, float m01, float m02, float m03,
                            float m10, float m11, float m12, float m13,
                            float eps, threadgroup float* part, uint sg, uint lane,
                            thread float& inv_v, thread float& inv_m) {
  GL_NOFMA
  float a0 = 0.0f;
  a0 = fma(v00, v00, a0); a0 = fma(v01, v01, a0); a0 = fma(v02, v02, a0); a0 = fma(v03, v03, a0);
  a0 = simd_sum(a0);
  if (lane == 0) part[sg] = a0;
  float a1 = 0.0f;
  a1 = fma(v10, v10, a1); a1 = fma(v11, v11, a1); a1 = fma(v12, v12, a1); a1 = fma(v13, v13, a1);
  a1 = simd_sum(a1);
  if (lane == 0) part[sg + 9] = a1;
  float b0 = 0.0f;
  b0 = fma(m00, m00, b0); b0 = fma(m01, m01, b0); b0 = fma(m02, m02, b0); b0 = fma(m03, m03, b0);
  b0 = simd_sum(b0);
  if (lane == 0) part[32 + sg] = b0;
  float b1 = 0.0f;
  b1 = fma(m10, m10, b1); b1 = fma(m11, m11, b1); b1 = fma(m12, m12, b1); b1 = fma(m13, m13, b1);
  b1 = simd_sum(b1);
  if (lane == 0) part[32 + sg + 9] = b1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tv = simd_sum(lane < 18u ? part[lane] : 0.0f);
  float tm = simd_sum(lane < 18u ? part[32 + lane] : 0.0f);
  inv_v = metal::precise::rsqrt(tv / float(RL_D) + eps);
  inv_m = metal::precise::rsqrt(tm / float(RL_D) + eps);
}

inline bfloat4 rl_load_raw_bf4(const device bfloat16_t* p) {
  return as_type<bfloat4>(((const device uint2*)p)[0]);
}

inline float4 rl_load_bf4(const device bfloat16_t* p) {
  bfloat4 v = as_type<bfloat4>(((const device uint2*)p)[0]);
  return float4(v.x, v.y, v.z, v.w);
}

inline void rl_store_bf4(device bfloat16_t* p, float4 v) {
  bfloat4 b = bfloat4(bfloat16_t(v.x), bfloat16_t(v.y), bfloat16_t(v.z), bfloat16_t(v.w));
  ((device uint2*)p)[0] = as_type<uint2>(b);
}

inline bfloat16_t rl_conv_pmc_resid(bfloat4 cww, float sr0, float sr1, float sr2, float sx0, float rs, float scale) {
  float acc = 0.0f;
  acc = fma(float(cww.x), float(bfloat16_t(sr0)), acc);
  acc = fma(float(cww.y), float(bfloat16_t(sr1)), acc);
  acc = fma(float(cww.z), float(bfloat16_t(sr2)), acc);
  acc = fma(float(cww.w), sx0, acc);
  float cv = float(bfloat16_t(acc));
  bfloat16_t pmc = bfloat16_t((cv + sx0) / scale);
  return bfloat16_t(rs + float(pmc));
}
"""
)


@functools.lru_cache(maxsize=None)
def _kernel(name, source, inputs, outputs, ensure_row_contiguous=False):
    return mx.fast.metal_kernel(
        name=name, input_names=list(inputs), output_names=list(outputs), source=source, header=RL_HEADER,
        ensure_row_contiguous=ensure_row_contiguous,
    )


def _check(x, shape_tail, dtype, name):
    if x.dtype != dtype or tuple(x.shape[-len(shape_tail):]) != tuple(shape_tail):
        raise ValueError("%s: expected dtype %s tail shape %s, got %s %s" % (name, dtype, shape_tail, x.dtype, x.shape))


_SCALE_DEPTH = flit(np.float32(D**-0.5))  # python float `D ** -0.5` -> f32 scalar array in the original
_SCALE_CONV = repr(float(np.float32(math.sqrt(5)))) + "f"  # fastdecode.short_conv_decode (kernel_size 4)
_K1_CACHE: dict = {}
_K1_DOWN_CACHE: dict = {}
_K10_CACHE: dict = {}
_K3B_CACHE: dict = {}


# ===================================================================================================== K1 PRE
def _check_bs(bs, B, name="bs"):
    if bs is None:
        return False, 0
    if isinstance(bs, (list, tuple)):
        for i, b in enumerate(bs):
            if b.dtype != _F32 or b.shape != (B, D):
                raise ValueError(f"{name}[{i}]: expected float32 {(B, D)}, got {b.dtype} {b.shape}")
        return True, len(bs)
    _check(bs, (B, D), _F32, name)
    return False, bs.shape[0]


def k1_pre(h, bs, msg, query, ln_w, *, sliding, eps_ln=1e-6, debug=False, w_act=None):
    """K1 PRE for B rows.

    h (B,D) bf16; bs (n,B,D) f32 or list/tuple of n (B,D) f32 arrays or None (layer 0: no attention_input);
    msg (B,4,D) f32 or None (looped pass > 0);
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
    bs_is_list, NB = _check_bs(bs, B, "bs") if has_ai else (False, 0)
    if has_ai:
        query = query.reshape(-1, D)
        if query.shape[0] != G or query.dtype != _F32:
            raise ValueError("query shape/dtype")
    if msg is None:
        if len(ln_w) != G:
            raise ValueError("need %d layernorm weights" % G)
        for w in ln_w:
            _check(w, (D,), _BF, "ln_w")

    args = [h]
    if has_ai:
        if bs_is_list:
            args += list(bs) + [query]
        else:
            args += [bs, query]
    if msg is not None:
        args.append(msg)
    else:
        args.extend(ln_w)

    has_wa = w_act is not None
    if has_wa:
        args.append(w_act.reshape(-1))
    shapes = [(B, D)]
    if msg is None:
        shapes += [(B, D)] * G
    else:
        shapes += [(B, D)] * G + [(B, 3, 2 * D)]
    if debug and has_ai:
        shapes.append((B, G, D))

    ckey = (sliding, NB, bs_is_list, has_ai, msg is not None, eps_ln, debug, has_wa)
    cached = _K1_CACHE.get(ckey)
    if cached is not None:
        k, outs, dts, gpar = cached
        r = k(inputs=args, grid=(B * gpar * TG, 1, 1), threadgroup=(TG, 1, 1), output_shapes=shapes, output_dtypes=dts)
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
            gi = o["gi"]
            res.update(qkv_gate_in=gi, q_gate_in=gi[:, 0], k_gate_in=gi[:, 1], v_gate_in=gi[:, 2])
        if "ai" in o:
            res["attn_input"] = o["ai"]
        return res

    ins = ["h"]
    if has_ai:
        if bs_is_list:
            ins += [f"bs{i}" for i in range(NB)] + ["query"]
        else:
            ins += ["bs", "query"]
    if msg is not None:
        ins.append("msg")
    else:
        for g in range(G):
            ins.append("w%d" % g)
    if has_wa:
        ins.append("w_act")
    outs = ["nr"]
    dts = [_F32]
    if msg is None:
        for g in range(G):
            outs.append("x%d" % g)
            dts.append(_BF)
    else:
        for g in range(G):
            outs.append("s%d" % g)
            dts.append(_F32)
        outs.append("gi")
        dts.append(_BF)
    if debug and has_ai:
        outs.append("ai")
        dts.append(_F32)

    gpar = G if (has_ai and msg is None) else 1
    s = [
        "uint tgx = threadgroup_position_in_grid.x;",
        f"uint row = tgx / {gpar}; uint g_id = tgx % {gpar}; (void)g_id;",
        *([f"if (int(row) >= w_act[0]) return;"] if has_wa else []),
        f"uint B = threadgroups_per_grid.x / {gpar}; (void)B;",
        "uint tid = thread_position_in_threadgroup.x;",
        "uint lane = thread_index_in_simdgroup; uint sg = simdgroup_index_in_threadgroup;",
        "threadgroup float part[64];",
        "const device bfloat16_t* hr = h + size_t(row) * RL_D;",
        "int b0 = (int(sg) * 32 + int(lane)) * 4;",
        "int b1 = ((int(sg) + 9) * 32 + int(lane)) * 4;",
        "float4 hv0 = rl_load_bf4(hr + b0); float4 hv1 = rl_load_bf4(hr + b1);",
        f"float inv0 = rl_rms_inv_reg(hv0.x, hv0.y, hv0.z, hv0.w, hv1.x, hv1.y, hv1.z, hv1.w, {flit(EPS_UW)}, part, sg, lane);",
        "float4 nr0 = float4(glue_rms_apply_nw<float>(hv0.x, inv0), glue_rms_apply_nw<float>(hv0.y, inv0), glue_rms_apply_nw<float>(hv0.z, inv0), glue_rms_apply_nw<float>(hv0.w, inv0));",
        "float4 nr1 = float4(glue_rms_apply_nw<float>(hv1.x, inv0), glue_rms_apply_nw<float>(hv1.y, inv0), glue_rms_apply_nw<float>(hv1.z, inv0), glue_rms_apply_nw<float>(hv1.w, inv0));",
    ]
    def _bs_d(i):
        return f"bs{i}[size_t(row) * RL_D + d]" if bs_is_list else f"bs[(size_t({i}) * B + row) * RL_D + d]"

    def _bs_f4(i, off):
        return (
            f"((const device float4*)(bs{i} + size_t(row) * RL_D + {off}))[0]"
            if bs_is_list
            else f"((const device float4*)(bs + (size_t({i}) * B + row) * RL_D + {off}))[0]"
        )

    if not has_ai:
        s += [
            "((device float4*)(nr + size_t(row) * RL_D + b0))[0] = nr0;",
            "((device float4*)(nr + size_t(row) * RL_D + b1))[0] = nr1;",
            "float invl = inv0;" if eps_ln == EPS_UW else f"float invl = rl_rms_inv_reg(hv0.x, hv0.y, hv0.z, hv0.w, hv1.x, hv1.y, hv1.z, hv1.w, {flit(eps_ln)}, part, sg, lane);",
        ]
        for g in range(G):
            s += [
                f"bfloat4 ww0_{g} = rl_load_raw_bf4(w{g} + b0); bfloat4 ww1_{g} = rl_load_raw_bf4(w{g} + b1);",
                f"bfloat4 xo0_{g} = bfloat4(glue_rms_apply<bfloat16_t>(hv0.x, invl, ww0_{g}.x), glue_rms_apply<bfloat16_t>(hv0.y, invl, ww0_{g}.y), glue_rms_apply<bfloat16_t>(hv0.z, invl, ww0_{g}.z), glue_rms_apply<bfloat16_t>(hv0.w, invl, ww0_{g}.w));",
                f"bfloat4 xo1_{g} = bfloat4(glue_rms_apply<bfloat16_t>(hv1.x, invl, ww1_{g}.x), glue_rms_apply<bfloat16_t>(hv1.y, invl, ww1_{g}.y), glue_rms_apply<bfloat16_t>(hv1.z, invl, ww1_{g}.z), glue_rms_apply<bfloat16_t>(hv1.w, invl, ww1_{g}.w));",
                f"((device uint2*)(x{g} + size_t(row) * RL_D + b0))[0] = as_type<uint2>(xo0_{g});",
                f"((device uint2*)(x{g} + size_t(row) * RL_D + b1))[0] = as_type<uint2>(xo1_{g});",
            ]
    elif msg is None:
        s += [
            "if (g_id == 0) { ((device float4*)(nr + size_t(row) * RL_D + b0))[0] = nr0; ((device float4*)(nr + size_t(row) * RL_D + b1))[0] = nr1; }",
            "const device float* qg = query + g_id * RL_D;",
            f"threadgroup float dbuf[{(NB + 1) * 9}];",
            f"float acc[{NB + 1}]; for (int i = 0; i < {NB + 1}; i++) acc[i] = 0.0f;",
            "for (int k = 0; k < 8; k++) { int d = int(tid) + k * RL_TG; float qv = qg[d];",
        ]
        for i in range(NB):
            s += [f"  acc[{i}] += {_bs_d(i)} * qv;"]
        s += [
            f"  acc[{NB}] += glue_rms_apply_nw<float>(float(hr[d]), inv0) * qv; }}",
            f"for (int i = 0; i < {NB + 1}; i++) {{ float v = simd_sum(acc[i]); if (lane == 0) dbuf[i * 9 + sg] = v; }}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"float wt[{NB + 1}]; float mxw = -INFINITY;",
            f"for (int i = 0; i < {NB + 1}; i++) {{ float t = 0.0f; for (int ss = 0; ss < 9; ss++) t += dbuf[i * 9 + ss]; wt[i] = t * {_SCALE_DEPTH}; mxw = max(mxw, wt[i]); }}",
            f"float zz = 0.0f; for (int i = 0; i < {NB + 1}; i++) {{ wt[i] = metal::exp(wt[i] - mxw); zz += wt[i]; }}",
            f"for (int i = 0; i < {NB + 1}; i++) wt[i] /= zz;",
            "float4 o0 = float4(0.0f); float4 o1 = float4(0.0f);",
        ]
        for i in range(NB):
            s += [f"o0 += wt[{i}] * {_bs_f4(i, 'b0')}; o1 += wt[{i}] * {_bs_f4(i, 'b1')};"]
        s += [f"o0 += wt[{NB}] * nr0; o1 += wt[{NB}] * nr1;"]
        if debug:
            s += [
                f"((device float4*)(ai + (size_t(row) * {G} + g_id) * RL_D + b0))[0] = o0;",
                f"((device float4*)(ai + (size_t(row) * {G} + g_id) * RL_D + b1))[0] = o1;",
            ]
        s += [f"float invl = rl_rms_inv_reg(o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w, {flit(eps_ln)}, part, sg, lane);"]
        for g in range(G):
            cond = f"if (g_id == {g})" if G > 1 else ""
            s += [
                f"{cond} {{",
                f"  bfloat4 ww0 = rl_load_raw_bf4(w{g} + b0); bfloat4 ww1 = rl_load_raw_bf4(w{g} + b1);",
                f"  bfloat4 xo0 = bfloat4(bfloat16_t(glue_rms_apply<float>(o0.x, invl, ww0.x)), bfloat16_t(glue_rms_apply<float>(o0.y, invl, ww0.y)), bfloat16_t(glue_rms_apply<float>(o0.z, invl, ww0.z)), bfloat16_t(glue_rms_apply<float>(o0.w, invl, ww0.w)));",
                f"  bfloat4 xo1 = bfloat4(bfloat16_t(glue_rms_apply<float>(o1.x, invl, ww1.x)), bfloat16_t(glue_rms_apply<float>(o1.y, invl, ww1.y)), bfloat16_t(glue_rms_apply<float>(o1.z, invl, ww1.z)), bfloat16_t(glue_rms_apply<float>(o1.w, invl, ww1.w)));",
                f"  ((device uint2*)(x{g} + size_t(row) * RL_D + b0))[0] = as_type<uint2>(xo0);",
                f"  ((device uint2*)(x{g} + size_t(row) * RL_D + b1))[0] = as_type<uint2>(xo1);",
                "}",
            ]
    else:
        s += [
            "((device float4*)(nr + size_t(row) * RL_D + b0))[0] = nr0;",
            "((device float4*)(nr + size_t(row) * RL_D + b1))[0] = nr1;",
            f"threadgroup float dbuf[{(NB + 1) * G * 9}];",
            f"float acc[{(NB + 1) * G}]; for (int i = 0; i < {(NB + 1) * G}; i++) acc[i] = 0.0f;",
            "for (int k = 0; k < 8; k++) { int d = int(tid) + k * RL_TG;",
            f"  float qv[{G}]; for (int j = 0; j < {G}; j++) qv[j] = query[j * RL_D + d];",
        ]
        for i in range(NB):
            s += [f"  {{ float sv = {_bs_d(i)}; for (int j = 0; j < {G}; j++) acc[{i} * {G} + j] += sv * qv[j]; }}"]
        s += [
            f"  {{ float sv = glue_rms_apply_nw<float>(float(hr[d]), inv0); for (int j = 0; j < {G}; j++) acc[{NB} * {G} + j] += sv * qv[j]; }} }}",
            f"for (int i = 0; i < {(NB + 1) * G}; i++) {{ float v = simd_sum(acc[i]); if (lane == 0) dbuf[i * 9 + sg] = v; }}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"float wt[{(NB + 1) * G}];",
            f"for (int i = 0; i < {(NB + 1) * G}; i++) {{ float t = 0.0f; for (int ss = 0; ss < 9; ss++) t += dbuf[i * 9 + ss]; wt[i] = t * {_SCALE_DEPTH}; }}",
            f"for (int j = 0; j < {G}; j++) {{ float m = -INFINITY; for (int i = 0; i < {NB + 1}; i++) m = max(m, wt[i * {G} + j]); float z = 0.0f; for (int i = 0; i < {NB + 1}; i++) {{ wt[i * {G} + j] = metal::exp(wt[i * {G} + j] - m); z += wt[i * {G} + j]; }} for (int i = 0; i < {NB + 1}; i++) wt[i * {G} + j] /= z; }}",
            f"float4 o0[{G}], o1[{G}]; for (int j = 0; j < {G}; j++) {{ o0[j] = float4(0.0f); o1[j] = float4(0.0f); }}",
        ]
        for i in range(NB):
            s += [
                f"{{ float4 bv0 = {_bs_f4(i, 'b0')}; float4 bv1 = {_bs_f4(i, 'b1')};",
                f"  for (int j = 0; j < {G}; j++) {{ o0[j] += wt[{i} * {G} + j] * bv0; o1[j] += wt[{i} * {G} + j] * bv1; }} }}",
            ]
        s += [f"for (int j = 0; j < {G}; j++) {{ o0[j] += wt[{NB} * {G} + j] * nr0; o1[j] += wt[{NB} * {G} + j] * nr1; }}"]
        for g in range(G):
            s += [
                f"((device float4*)(s{g} + size_t(row) * RL_D + b0))[0] = o0[{g}];",
                f"((device float4*)(s{g} + size_t(row) * RL_D + b1))[0] = o1[{g}];",
            ]
            if debug:
                s += [
                    f"((device float4*)(ai + (size_t(row) * {G} + {g}) * RL_D + b0))[0] = o0[{g}];",
                    f"((device float4*)(ai + (size_t(row) * {G} + {g}) * RL_D + b1))[0] = o1[{g}];",
                ]
        s += [
            f"threadgroup float partm[{(G + 3) * 32}];",
            "const device float* mp0 = msg + size_t(row) * 4 * RL_D;",
            "float4 mv0[3], mv1[3];",
            "for (int m = 0; m < 3; m++) { mv0[m] = ((const device float4*)(mp0 + m * RL_D + b0))[0]; mv1[m] = ((const device float4*)(mp0 + m * RL_D + b1))[0]; }",
            "{ GL_NOFMA",
            f"  for (int g = 0; g < {G}; g++) {{",
            "    float a0 = 0.0f; a0 = fma(o0[g].x, o0[g].x, a0); a0 = fma(o0[g].y, o0[g].y, a0); a0 = fma(o0[g].z, o0[g].z, a0); a0 = fma(o0[g].w, o0[g].w, a0); a0 = simd_sum(a0); if (lane == 0) partm[g * 32 + sg] = a0;",
            "    float a1 = 0.0f; a1 = fma(o1[g].x, o1[g].x, a1); a1 = fma(o1[g].y, o1[g].y, a1); a1 = fma(o1[g].z, o1[g].z, a1); a1 = fma(o1[g].w, o1[g].w, a1); a1 = simd_sum(a1); if (lane == 0) partm[g * 32 + sg + 9] = a1;",
            "  }",
            "  for (int m = 0; m < 3; m++) {",
            f"    float b0m = 0.0f; b0m = fma(mv0[m].x, mv0[m].x, b0m); b0m = fma(mv0[m].y, mv0[m].y, b0m); b0m = fma(mv0[m].z, mv0[m].z, b0m); b0m = fma(mv0[m].w, mv0[m].w, b0m); b0m = simd_sum(b0m); if (lane == 0) partm[({G} + m) * 32 + sg] = b0m;",
            f"    float b1m = 0.0f; b1m = fma(mv1[m].x, mv1[m].x, b1m); b1m = fma(mv1[m].y, mv1[m].y, b1m); b1m = fma(mv1[m].z, mv1[m].z, b1m); b1m = fma(mv1[m].w, mv1[m].w, b1m); b1m = simd_sum(b1m); if (lane == 0) partm[({G} + m) * 32 + sg + 9] = b1m;",
            "  }",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"float ins_[{G}], inm[3];",
            "{ GL_NOFMA",
            f"  for (int g = 0; g < {G}; g++) {{ float tot = simd_sum(lane < 18u ? partm[g * 32 + lane] : 0.0f); ins_[g] = metal::precise::rsqrt(tot / float(RL_D) + {flit(EPS_UW)}); }}",
            f"  for (int m = 0; m < 3; m++) {{ float tot = simd_sum(lane < 18u ? partm[({G} + m) * 32 + lane] : 0.0f); inm[m] = metal::precise::rsqrt(tot / float(RL_D) + {flit(EPS_UW)}); }}",
            "}",
        ]
        for i in range(3):
            g = 0 if sliding else i
            s += [
                f"{{ size_t gio = (size_t(row) * 3 + {i}) * 2 * RL_D;",
                f"  rl_store_bf4(gi + gio + b0, float4(glue_rms_apply_nw<float>(o0[{g}].x, ins_[{g}]), glue_rms_apply_nw<float>(o0[{g}].y, ins_[{g}]), glue_rms_apply_nw<float>(o0[{g}].z, ins_[{g}]), glue_rms_apply_nw<float>(o0[{g}].w, ins_[{g}])));",
                f"  rl_store_bf4(gi + gio + b1, float4(glue_rms_apply_nw<float>(o1[{g}].x, ins_[{g}]), glue_rms_apply_nw<float>(o1[{g}].y, ins_[{g}]), glue_rms_apply_nw<float>(o1[{g}].z, ins_[{g}]), glue_rms_apply_nw<float>(o1[{g}].w, ins_[{g}])));",
                f"  rl_store_bf4(gi + gio + RL_D + b0, float4(glue_rms_apply_nw<float>(mv0[{i}].x, inm[{i}]), glue_rms_apply_nw<float>(mv0[{i}].y, inm[{i}]), glue_rms_apply_nw<float>(mv0[{i}].z, inm[{i}]), glue_rms_apply_nw<float>(mv0[{i}].w, inm[{i}])));",
                f"  rl_store_bf4(gi + gio + RL_D + b1, float4(glue_rms_apply_nw<float>(mv1[{i}].x, inm[{i}]), glue_rms_apply_nw<float>(mv1[{i}].y, inm[{i}]), glue_rms_apply_nw<float>(mv1[{i}].z, inm[{i}]), glue_rms_apply_nw<float>(mv1[{i}].w, inm[{i}]))); }}",
            ]
    src = "\n".join(s)
    name = "rl_k1_%s_%d_%d_%d_%d_%s%s_%s" % (
        "S" if sliding else "F", NB, int(bs_is_list), int(has_ai), int(msg is not None), flit(eps_ln), "_wa" if has_wa else "", "dbg" if debug else ""
    )
    outs_t = tuple(outs)
    k = _kernel(name.replace(".", "p").replace("-", "m"), src, tuple(ins), outs_t)
    _K1_CACHE[ckey] = (k, outs_t, dts, gpar)
    r = k(inputs=args, grid=(B * gpar * TG, 1, 1), threadgroup=(TG, 1, 1), output_shapes=shapes, output_dtypes=dts)
    o = dict(zip(outs_t, r))
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
        gi = o["gi"]
        res.update(qkv_gate_in=gi, q_gate_in=gi[:, 0], k_gate_in=gi[:, 1], v_gate_in=gi[:, 2])
    if "ai" in o:
        res["attn_input"] = o["ai"]
    return res


def k1_pre_layer(layer, h, bs, msg, *, debug=False, w_act=None):
    """K1 on a decoder layer module (h (B,D) bf16, bs (n,B,D) f32 or None, msg (B,4,D) f32 or None)."""
    if isinstance(layer, dict):
        ai = dict.get(layer, "attention_input")
        q = ai["query"] if ai is not None else None
        if "input_layernorm" in layer:
            m = layer["input_layernorm"]
            return k1_pre(h, bs, msg, q, [m["weight"]], sliding=True, eps_ln=m.eps, debug=debug, w_act=w_act)
        m0, m1, m2 = layer["q_input_layernorm"], layer["k_input_layernorm"], layer["v_input_layernorm"]
        return k1_pre(h, bs, msg, q, [m0["weight"], m1["weight"], m2["weight"]], sliding=False, eps_ln=m0.eps, debug=debug, w_act=w_act)
    from mlx_lm.models.inkling2 import layers as L

    sliding = isinstance(layer, L.SlidingDecoderLayer)
    q = layer.attention_input.query if layer.attention_input is not None else None
    if sliding:
        ln = [layer.input_layernorm]
    else:
        ln = [layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm]
    return k1_pre(h, bs, msg, q, [m.weight for m in ln], sliding=sliding, eps_ln=ln[0].eps, debug=debug, w_act=w_act)


def _k1_from_down_impl(
    down, wts, resid, cs3, conv_w, conv_anc,
    bs, msg, query, ln_w, *, causal, sliding, eps_ln=1e-6, emit_h=False, debug=False,
):
    """Fused MoE combine + short conv + K1 PRE (eliminates combine-conv launch + optional h DRAM roundtrip)."""
    B = int(down.shape[0])
    if B < 1 or B > 8 or down.dtype != _BF or down.shape != (B, 8, D):
        raise ValueError(f"down: expected bfloat16 (B<=8,8,{D}), got {down.dtype} {down.shape}")
    _check(wts, (8,), _F32, "wts")
    if wts.shape[0] != B:
        raise ValueError("wts batch mismatch")
    _check(resid, (D,), _F32, "resid")
    if resid.shape[0] != B:
        raise ValueError("resid batch mismatch")
    cs3_b = 1 if causal else B
    if cs3.dtype != _F32 or cs3.shape != (cs3_b, 3, D):
        raise ValueError(f"cs3: expected float32 ({cs3_b},3,{D}), got {cs3.dtype} {cs3.shape}")
    if conv_w.dtype != _BF or conv_w.size != D * 4:
        raise ValueError("conv_w: expected bfloat16 of size D*4")
    if causal:
        if conv_anc is None or conv_anc.dtype != mx.int32 or conv_anc.shape != (B, 3):
            raise ValueError("conv_anc: expected int32 (B,3)")

    has_ai = query is not None
    if has_ai != (bs is not None):
        raise ValueError("query and bs must both be given or both None")
    if msg is not None:
        if not has_ai:
            raise ValueError("message visit without attention_input is not supported")
        _check(msg, (4, D), _F32, "msg")
    G = 1 if sliding else 3
    bs_is_list, NB = _check_bs(bs, B, "bs") if has_ai else (False, 0)
    if has_ai:
        query = query.reshape(-1, D)
        if query.shape[0] != G or query.dtype != _F32:
            raise ValueError("query shape/dtype")
    if msg is None:
        if len(ln_w) != G:
            raise ValueError("need %d layernorm weights" % G)
        for w in ln_w:
            _check(w, (D,), _BF, "ln_w")

    args = [down, wts, resid, cs3, conv_w] + ([conv_anc] if causal else [])
    if has_ai:
        if bs_is_list:
            args += list(bs) + [query]
        else:
            args += [bs, query]
    if msg is not None:
        args.append(msg)
    else:
        args.extend(ln_w)

    shapes = ([(B, D)] if emit_h else []) + [(B, 3, D), (B, D)]
    if msg is None:
        shapes += [(B, D)] * G
    else:
        shapes += [(B, D)] * G + [(B, 3, 2 * D)]
    if debug and has_ai:
        shapes.append((B, G, D))

    ckey = (causal, B, sliding, NB, bs_is_list, has_ai, msg is not None, eps_ln, emit_h, debug)
    cached = _K1_DOWN_CACHE.get(ckey)
    if cached is not None:
        k, outs_t, dts, gpar = cached
    else:
        ins = ["down", "wts", "resid", "cs3", "cw"] + (["conv_anc"] if causal else [])
        if has_ai:
            if bs_is_list:
                ins += [f"bs{i}" for i in range(NB)] + ["query"]
            else:
                ins += ["bs", "query"]
        if msg is not None:
            ins.append("msg")
        else:
            for g in range(G):
                ins.append("w%d" % g)
        outs = (["h_out"] if emit_h else []) + ["w3", "nr"]
        dts = ([_BF] if emit_h else []) + [_F32, _F32]
        if msg is None:
            for g in range(G):
                outs.append("x%d" % g)
                dts.append(_BF)
        else:
            for g in range(G):
                outs.append("s%d" % g)
                dts.append(_F32)
            outs.append("gi")
            dts.append(_BF)
        if debug and has_ai:
            outs.append("ai")
            dts.append(_F32)

        gpar = G if (has_ai and msg is None) else 1
        s = [
            "uint tgx = threadgroup_position_in_grid.x;",
            f"uint row = tgx / {gpar}; uint g_id = tgx % {gpar}; (void)g_id;",
            f"const int B = {B};",
            "uint tid = thread_position_in_threadgroup.x;",
            "uint lane = thread_index_in_simdgroup; uint sg = simdgroup_index_in_threadgroup;",
            "threadgroup float part[64];",
        ]
        if causal:
            s += [
                "threadgroup float ww_s[64];",
                f"if (int(tid) < {B} * 8) ww_s[tid] = wts[tid];",
                "threadgroup bfloat16_t h_s[RL_D];",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
                "int a0 = conv_anc[row * 3 + 0];",
                "int a1 = conv_anc[row * 3 + 1];",
                "int a2 = conv_anc[row * 3 + 2];",
            ]
        else:
            s += [
                "threadgroup float ww_s[8];",
                "if (int(tid) < 8) ww_s[tid] = wts[row * 8 + tid];",
                "threadgroup bfloat16_t h_s[RL_D];",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
                "const device float* st = cs3 + size_t(row) * 3 * RL_D;",
            ]
        s += [
            "int b0 = (int(sg) * 32 + int(lane)) * 4;",
            "int b1 = ((int(sg) + 9) * 32 + int(lane)) * 4;",
            "float4 hv0, hv1;",
            "float hv[8];",
            "for (int h_i = 0; h_i < 2; h_i++) {",
            "  int bc = (h_i == 0) ? b0 : b1;",
            "  bfloat4 hb;",
            "  float4 r1v, r2v, x0v;",
        ]
        if causal:
            s += [
                "  for (int m = 0; m < 4; m++) {",
                "    int c = bc + m;",
                "    float ev[8], wv[8];",
                "    for (int j = 0; j < 8; j++) { ev[j] = float(down[(row * 8 + j) * RL_D + c]); wv[j] = ww_s[row * 8 + j]; }",
                "    float x0 = float(bfloat16_t(glue_moe_combine8(ev, wv)));",
                "    float r0, r1, r2;",
                "    if (a0 >= 0) {",
                "      for (int j = 0; j < 8; j++) { ev[j] = float(down[(a0 * 8 + j) * RL_D + c]); wv[j] = ww_s[a0 * 8 + j]; }",
                "      r0 = float(bfloat16_t(glue_moe_combine8(ev, wv)));",
                "    } else r0 = cs3[(-1 - a0) * RL_D + c];",
                "    if (a1 >= 0) {",
                "      for (int j = 0; j < 8; j++) { ev[j] = float(down[(a1 * 8 + j) * RL_D + c]); wv[j] = ww_s[a1 * 8 + j]; }",
                "      r1 = float(bfloat16_t(glue_moe_combine8(ev, wv)));",
                "    } else r1 = cs3[(-1 - a1) * RL_D + c];",
                "    if (a2 >= 0) {",
                "      for (int j = 0; j < 8; j++) { ev[j] = float(down[(a2 * 8 + j) * RL_D + c]); wv[j] = ww_s[a2 * 8 + j]; }",
                "      r2 = float(bfloat16_t(glue_moe_combine8(ev, wv)));",
                "    } else r2 = cs3[(-1 - a2) * RL_D + c];",
                "    bfloat4 cww = rl_load_raw_bf4(cw + c * 4);",
                f"    bfloat16_t hch = rl_conv_pmc_resid(cww, r0, r1, r2, x0, resid[size_t(row) * RL_D + c], {_SCALE_CONV});",
                "    hb[m] = hch; hv[h_i * 4 + m] = float(hch);",
                "    r1v[m] = r1; r2v[m] = r2; x0v[m] = x0;",
                "  }",
            ]
        else:
            s += [
                "  bfloat4 dv[8];",
                "  for (int j = 0; j < 8; j++) dv[j] = rl_load_raw_bf4(down + (row * 8 + j) * RL_D + bc);",
                "  float4 r0s = ((const device float4*)(st + bc))[0];",
                "  float4 r1s = ((const device float4*)(st + RL_D + bc))[0];",
                "  float4 r2s = ((const device float4*)(st + 2 * RL_D + bc))[0];",
                "  float4 ress = ((const device float4*)(resid + size_t(row) * RL_D + bc))[0];",
                "  for (int m = 0; m < 4; m++) {",
                "    int c = bc + m;",
                "    float ev[8], wv[8];",
                "    for (int j = 0; j < 8; j++) { ev[j] = float(dv[j][m]); wv[j] = ww_s[j]; }",
                "    float x0 = float(bfloat16_t(glue_moe_combine8(ev, wv)));",
                "    float r0 = r0s[m], r1 = r1s[m], r2 = r2s[m];",
                "    bfloat4 cww = rl_load_raw_bf4(cw + c * 4);",
                f"    bfloat16_t hch = rl_conv_pmc_resid(cww, r0, r1, r2, x0, ress[m], {_SCALE_CONV});",
                "    hb[m] = hch; hv[h_i * 4 + m] = float(hch);",
                "    r1v[m] = r1; r2v[m] = r2; x0v[m] = x0;",
                "  }",
            ]
        s += [
            "  ((threadgroup uint2*)(h_s + bc))[0] = as_type<uint2>(hb);",
            "  if (g_id == 0) {",
            "    ((device float4*)(w3 + (size_t(row) * 3 + 0) * RL_D + bc))[0] = r1v;",
            "    ((device float4*)(w3 + (size_t(row) * 3 + 1) * RL_D + bc))[0] = r2v;",
            "    ((device float4*)(w3 + (size_t(row) * 3 + 2) * RL_D + bc))[0] = x0v;",
        ]
        if emit_h:
            s.append("    ((device uint2*)(h_out + size_t(row) * RL_D + bc))[0] = as_type<uint2>(hb);")
        s += [
            "  }",
            "}",
            "hv0 = float4(hv[0], hv[1], hv[2], hv[3]);",
            "hv1 = float4(hv[4], hv[5], hv[6], hv[7]);",
        ]
        s += [
            f"float inv0 = rl_rms_inv_reg(hv0.x, hv0.y, hv0.z, hv0.w, hv1.x, hv1.y, hv1.z, hv1.w, {flit(EPS_UW)}, part, sg, lane);",
            "float4 nr0 = float4(glue_rms_apply_nw<float>(hv0.x, inv0), glue_rms_apply_nw<float>(hv0.y, inv0), glue_rms_apply_nw<float>(hv0.z, inv0), glue_rms_apply_nw<float>(hv0.w, inv0));",
            "float4 nr1 = float4(glue_rms_apply_nw<float>(hv1.x, inv0), glue_rms_apply_nw<float>(hv1.y, inv0), glue_rms_apply_nw<float>(hv1.z, inv0), glue_rms_apply_nw<float>(hv1.w, inv0));",
        ]

        def _bs_d(i):
            return f"bs{i}[size_t(row) * RL_D + d]" if bs_is_list else f"bs[(size_t({i}) * B + row) * RL_D + d]"

        def _bs_f4(i, off):
            return (
                f"((const device float4*)(bs{i} + size_t(row) * RL_D + {off}))[0]"
                if bs_is_list
                else f"((const device float4*)(bs + (size_t({i}) * B + row) * RL_D + {off}))[0]"
            )

        if not has_ai:
            s += [
                "((device float4*)(nr + size_t(row) * RL_D + b0))[0] = nr0;",
                "((device float4*)(nr + size_t(row) * RL_D + b1))[0] = nr1;",
                "float invl = inv0;" if eps_ln == EPS_UW else f"float invl = rl_rms_inv_reg(hv0.x, hv0.y, hv0.z, hv0.w, hv1.x, hv1.y, hv1.z, hv1.w, {flit(eps_ln)}, part, sg, lane);",
            ]
            for g in range(G):
                s += [
                    f"bfloat4 ww0_{g} = rl_load_raw_bf4(w{g} + b0); bfloat4 ww1_{g} = rl_load_raw_bf4(w{g} + b1);",
                    f"bfloat4 xo0_{g} = bfloat4(glue_rms_apply<bfloat16_t>(hv0.x, invl, ww0_{g}.x), glue_rms_apply<bfloat16_t>(hv0.y, invl, ww0_{g}.y), glue_rms_apply<bfloat16_t>(hv0.z, invl, ww0_{g}.z), glue_rms_apply<bfloat16_t>(hv0.w, invl, ww0_{g}.w));",
                    f"bfloat4 xo1_{g} = bfloat4(glue_rms_apply<bfloat16_t>(hv1.x, invl, ww1_{g}.x), glue_rms_apply<bfloat16_t>(hv1.y, invl, ww1_{g}.y), glue_rms_apply<bfloat16_t>(hv1.z, invl, ww1_{g}.z), glue_rms_apply<bfloat16_t>(hv1.w, invl, ww1_{g}.w));",
                    f"((device uint2*)(x{g} + size_t(row) * RL_D + b0))[0] = as_type<uint2>(xo0_{g});",
                    f"((device uint2*)(x{g} + size_t(row) * RL_D + b1))[0] = as_type<uint2>(xo1_{g});",
                ]
        elif msg is None:
            s += [
                "if (g_id == 0) { ((device float4*)(nr + size_t(row) * RL_D + b0))[0] = nr0; ((device float4*)(nr + size_t(row) * RL_D + b1))[0] = nr1; }",
                "const device float* qg = query + g_id * RL_D;",
                f"threadgroup float dbuf[{(NB + 1) * 9}];",
                f"float acc[{NB + 1}]; for (int i = 0; i < {NB + 1}; i++) acc[i] = 0.0f;",
                "for (int k = 0; k < 8; k++) { int d = int(tid) + k * RL_TG; float qv = qg[d];",
            ]
            for i in range(NB):
                s += [f"  acc[{i}] += {_bs_d(i)} * qv;"]
            s += [
                f"  acc[{NB}] += glue_rms_apply_nw<float>(float(h_s[d]), inv0) * qv; }}",
                f"for (int i = 0; i < {NB + 1}; i++) {{ float v = simd_sum(acc[i]); if (lane == 0) dbuf[i * 9 + sg] = v; }}",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
                f"float wt[{NB + 1}]; float mxw = -INFINITY;",
                f"for (int i = 0; i < {NB + 1}; i++) {{ float t = 0.0f; for (int ss = 0; ss < 9; ss++) t += dbuf[i * 9 + ss]; wt[i] = t * {_SCALE_DEPTH}; mxw = max(mxw, wt[i]); }}",
                f"float zz = 0.0f; for (int i = 0; i < {NB + 1}; i++) {{ wt[i] = metal::exp(wt[i] - mxw); zz += wt[i]; }}",
                f"for (int i = 0; i < {NB + 1}; i++) wt[i] /= zz;",
                "float4 o0 = float4(0.0f); float4 o1 = float4(0.0f);",
            ]
            for i in range(NB):
                s += [f"o0 += wt[{i}] * {_bs_f4(i, 'b0')}; o1 += wt[{i}] * {_bs_f4(i, 'b1')};"]
            s += [f"o0 += wt[{NB}] * nr0; o1 += wt[{NB}] * nr1;"]
            if debug:
                s += [
                    f"((device float4*)(ai + (size_t(row) * {G} + g_id) * RL_D + b0))[0] = o0;",
                    f"((device float4*)(ai + (size_t(row) * {G} + g_id) * RL_D + b1))[0] = o1;",
                ]
            s += [f"float invl = rl_rms_inv_reg(o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w, {flit(eps_ln)}, part, sg, lane);"]
            for g in range(G):
                cond = f"if (g_id == {g})" if G > 1 else ""
                s += [
                    f"{cond} {{",
                    f"  bfloat4 ww0 = rl_load_raw_bf4(w{g} + b0); bfloat4 ww1 = rl_load_raw_bf4(w{g} + b1);",
                    f"  bfloat4 xo0 = bfloat4(bfloat16_t(glue_rms_apply<float>(o0.x, invl, ww0.x)), bfloat16_t(glue_rms_apply<float>(o0.y, invl, ww0.y)), bfloat16_t(glue_rms_apply<float>(o0.z, invl, ww0.z)), bfloat16_t(glue_rms_apply<float>(o0.w, invl, ww0.w)));",
                    f"  bfloat4 xo1 = bfloat4(bfloat16_t(glue_rms_apply<float>(o1.x, invl, ww1.x)), bfloat16_t(glue_rms_apply<float>(o1.y, invl, ww1.y)), bfloat16_t(glue_rms_apply<float>(o1.z, invl, ww1.z)), bfloat16_t(glue_rms_apply<float>(o1.w, invl, ww1.w)));",
                    f"  ((device uint2*)(x{g} + size_t(row) * RL_D + b0))[0] = as_type<uint2>(xo0);",
                    f"  ((device uint2*)(x{g} + size_t(row) * RL_D + b1))[0] = as_type<uint2>(xo1);",
                    "}",
                ]
        else:
            s += [
                "((device float4*)(nr + size_t(row) * RL_D + b0))[0] = nr0;",
                "((device float4*)(nr + size_t(row) * RL_D + b1))[0] = nr1;",
                f"threadgroup float dbuf[{(NB + 1) * G * 9}];",
                f"float acc[{(NB + 1) * G}]; for (int i = 0; i < {(NB + 1) * G}; i++) acc[i] = 0.0f;",
                "for (int k = 0; k < 8; k++) { int d = int(tid) + k * RL_TG;",
                f"  float qv[{G}]; for (int j = 0; j < {G}; j++) qv[j] = query[j * RL_D + d];",
            ]
            for i in range(NB):
                s += [f"  {{ float sv = {_bs_d(i)}; for (int j = 0; j < {G}; j++) acc[{i} * {G} + j] += sv * qv[j]; }}"]
            s += [
                f"  {{ float sv = glue_rms_apply_nw<float>(float(h_s[d]), inv0); for (int j = 0; j < {G}; j++) acc[{NB} * {G} + j] += sv * qv[j]; }} }}",
                f"for (int i = 0; i < {(NB + 1) * G}; i++) {{ float v = simd_sum(acc[i]); if (lane == 0) dbuf[i * 9 + sg] = v; }}",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
                f"float wt[{(NB + 1) * G}];",
                f"for (int i = 0; i < {(NB + 1) * G}; i++) {{ float t = 0.0f; for (int ss = 0; ss < 9; ss++) t += dbuf[i * 9 + ss]; wt[i] = t * {_SCALE_DEPTH}; }}",
                f"for (int j = 0; j < {G}; j++) {{ float m = -INFINITY; for (int i = 0; i < {NB + 1}; i++) m = max(m, wt[i * {G} + j]); float z = 0.0f; for (int i = 0; i < {NB + 1}; i++) {{ wt[i * {G} + j] = metal::exp(wt[i * {G} + j] - m); z += wt[i * {G} + j]; }} for (int i = 0; i < {NB + 1}; i++) wt[i * {G} + j] /= z; }}",
                f"float4 o0[{G}], o1[{G}]; for (int j = 0; j < {G}; j++) {{ o0[j] = float4(0.0f); o1[j] = float4(0.0f); }}",
            ]
            for i in range(NB):
                s += [
                    f"{{ float4 bv0 = {_bs_f4(i, 'b0')}; float4 bv1 = {_bs_f4(i, 'b1')};",
                    f"  for (int j = 0; j < {G}; j++) {{ o0[j] += wt[{i} * {G} + j] * bv0; o1[j] += wt[{i} * {G} + j] * bv1; }} }}",
                ]
            s += [f"for (int j = 0; j < {G}; j++) {{ o0[j] += wt[{NB} * {G} + j] * nr0; o1[j] += wt[{NB} * {G} + j] * nr1; }}"]
            for g in range(G):
                s += [
                    f"((device float4*)(s{g} + size_t(row) * RL_D + b0))[0] = o0[{g}];",
                    f"((device float4*)(s{g} + size_t(row) * RL_D + b1))[0] = o1[{g}];",
                ]
                if debug:
                    s += [
                        f"((device float4*)(ai + (size_t(row) * {G} + {g}) * RL_D + b0))[0] = o0[{g}];",
                        f"((device float4*)(ai + (size_t(row) * {G} + {g}) * RL_D + b1))[0] = o1[{g}];",
                    ]
            s += [
                f"threadgroup float partm[{(G + 3) * 32}];",
                "const device float* mp0 = msg + size_t(row) * 4 * RL_D;",
                "float4 mv0[3], mv1[3];",
                "for (int m = 0; m < 3; m++) { mv0[m] = ((const device float4*)(mp0 + m * RL_D + b0))[0]; mv1[m] = ((const device float4*)(mp0 + m * RL_D + b1))[0]; }",
                "{ GL_NOFMA",
                f"  for (int g = 0; g < {G}; g++) {{",
                "    float a0 = 0.0f; a0 = fma(o0[g].x, o0[g].x, a0); a0 = fma(o0[g].y, o0[g].y, a0); a0 = fma(o0[g].z, o0[g].z, a0); a0 = fma(o0[g].w, o0[g].w, a0); a0 = simd_sum(a0); if (lane == 0) partm[g * 32 + sg] = a0;",
                "    float a1 = 0.0f; a1 = fma(o1[g].x, o1[g].x, a1); a1 = fma(o1[g].y, o1[g].y, a1); a1 = fma(o1[g].z, o1[g].z, a1); a1 = fma(o1[g].w, o1[g].w, a1); a1 = simd_sum(a1); if (lane == 0) partm[g * 32 + sg + 9] = a1;",
                "  }",
                "  for (int m = 0; m < 3; m++) {",
                f"    float b0m = 0.0f; b0m = fma(mv0[m].x, mv0[m].x, b0m); b0m = fma(mv0[m].y, mv0[m].y, b0m); b0m = fma(mv0[m].z, mv0[m].z, b0m); b0m = fma(mv0[m].w, mv0[m].w, b0m); b0m = simd_sum(b0m); if (lane == 0) partm[({G} + m) * 32 + sg] = b0m;",
                f"    float b1m = 0.0f; b1m = fma(mv1[m].x, mv1[m].x, b1m); b1m = fma(mv1[m].y, mv1[m].y, b1m); b1m = fma(mv1[m].z, mv1[m].z, b1m); b1m = fma(mv1[m].w, mv1[m].w, b1m); b1m = simd_sum(b1m); if (lane == 0) partm[({G} + m) * 32 + sg + 9] = b1m;",
                "  }",
                "}",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
                f"float ins_[{G}], inm[3];",
                "{ GL_NOFMA",
                f"  for (int g = 0; g < {G}; g++) {{ float tot = simd_sum(lane < 18u ? partm[g * 32 + lane] : 0.0f); ins_[g] = metal::precise::rsqrt(tot / float(RL_D) + {flit(EPS_UW)}); }}",
                f"  for (int m = 0; m < 3; m++) {{ float tot = simd_sum(lane < 18u ? partm[({G} + m) * 32 + lane] : 0.0f); inm[m] = metal::precise::rsqrt(tot / float(RL_D) + {flit(EPS_UW)}); }}",
                "}",
            ]
            for i in range(3):
                g = 0 if sliding else i
                s += [
                    f"{{ size_t gio = (size_t(row) * 3 + {i}) * 2 * RL_D;",
                    f"  rl_store_bf4(gi + gio + b0, float4(glue_rms_apply_nw<float>(o0[{g}].x, ins_[{g}]), glue_rms_apply_nw<float>(o0[{g}].y, ins_[{g}]), glue_rms_apply_nw<float>(o0[{g}].z, ins_[{g}]), glue_rms_apply_nw<float>(o0[{g}].w, ins_[{g}])));",
                    f"  rl_store_bf4(gi + gio + b1, float4(glue_rms_apply_nw<float>(o1[{g}].x, ins_[{g}]), glue_rms_apply_nw<float>(o1[{g}].y, ins_[{g}]), glue_rms_apply_nw<float>(o1[{g}].z, ins_[{g}]), glue_rms_apply_nw<float>(o1[{g}].w, ins_[{g}])));",
                    f"  rl_store_bf4(gi + gio + RL_D + b0, float4(glue_rms_apply_nw<float>(mv0[{i}].x, inm[{i}]), glue_rms_apply_nw<float>(mv0[{i}].y, inm[{i}]), glue_rms_apply_nw<float>(mv0[{i}].z, inm[{i}]), glue_rms_apply_nw<float>(mv0[{i}].w, inm[{i}])));",
                    f"  rl_store_bf4(gi + gio + RL_D + b1, float4(glue_rms_apply_nw<float>(mv1[{i}].x, inm[{i}]), glue_rms_apply_nw<float>(mv1[{i}].y, inm[{i}]), glue_rms_apply_nw<float>(mv1[{i}].z, inm[{i}]), glue_rms_apply_nw<float>(mv1[{i}].w, inm[{i}]))); }}",
                ]
        src = "\n".join(s)
        prefix = "rl_k1d" if causal else "rl_k1b"
        name = "%s_%d_%s_%d_%d_%d_%d_%s_%d_%s" % (
            prefix, B, "S" if sliding else "F", NB, int(bs_is_list), int(has_ai), int(msg is not None), flit(eps_ln), int(emit_h), "dbg" if debug else ""
        )
        outs_t = tuple(outs)
        k = _kernel(name.replace(".", "p").replace("-", "m"), src, tuple(ins), outs_t, ensure_row_contiguous=True)
        _K1_DOWN_CACHE[ckey] = (k, outs_t, dts, gpar)

    r = k(inputs=args, grid=(B * gpar * TG, 1, 1), threadgroup=(TG, 1, 1), output_shapes=shapes, output_dtypes=dts)
    o = dict(zip(outs_t, r))
    res = {"w3": o["w3"], "normed_residual": o["nr"]}
    if emit_h:
        res["h"] = o["h_out"]
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
        gi = o["gi"]
        res.update(qkv_gate_in=gi, q_gate_in=gi[:, 0], k_gate_in=gi[:, 1], v_gate_in=gi[:, 2])
    if "ai" in o:
        res["attn_input"] = o["ai"]
    return res


def k1_from_down_causal(
    down, wts, resid, cs3, conv_w, conv_anc,
    bs, msg, query, ln_w, *, sliding, eps_ln=1e-6, emit_h=False, debug=False,
):
    """Fused causal MoE combine + short conv + K1 PRE (eliminates comb_causal launch + h DRAM roundtrip)."""
    return _k1_from_down_impl(
        down, wts, resid, cs3, conv_w, conv_anc,
        bs, msg, query, ln_w, causal=True, sliding=sliding, eps_ln=eps_ln, emit_h=emit_h, debug=debug,
    )


def k1_from_down_batch(
    down, wts, resid, cs3, conv_w,
    bs, msg, query, ln_w, *, sliding, eps_ln=1e-6, emit_h=True, debug=False,
):
    """Fused independent-batch MoE combine + short conv + K1 PRE (eliminates moe_comb launch + h DRAM read)."""
    return _k1_from_down_impl(
        down, wts, resid, cs3, conv_w, None,
        bs, msg, query, ln_w, causal=False, sliding=sliding, eps_ln=eps_ln, emit_h=emit_h, debug=debug,
    )


def _layer_norm_meta(layer):
    if isinstance(layer, dict):
        ai = dict.get(layer, "attention_input")
        q = ai["query"] if ai is not None else None
        if "input_layernorm" in layer:
            m = layer["input_layernorm"]
            return True, q, [m["weight"]], m.eps
        m0, m1, m2 = layer["q_input_layernorm"], layer["k_input_layernorm"], layer["v_input_layernorm"]
        return False, q, [m0["weight"], m1["weight"], m2["weight"]], m0.eps
    from mlx_lm.models.inkling2 import layers as L

    sliding = isinstance(layer, L.SlidingDecoderLayer)
    q = layer.attention_input.query if layer.attention_input is not None else None
    ln = [layer.input_layernorm] if sliding else [layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm]
    return sliding, q, [m.weight for m in ln], ln[0].eps


def k1_from_down_causal_layer(layer, prev_moe, conv_anc, bs, msg, *, emit_h=False, debug=False):
    """Fused causal MoE combine + short conv + K1 PRE from `prev_moe` dict produced by `moe_rows(..., skip_combine=True)`."""
    sliding, q, weights, eps_ln = _layer_norm_meta(layer)
    return k1_from_down_causal(
        prev_moe["down"], prev_moe["expert_w"], prev_moe["residual"], prev_moe["cs3"], prev_moe["conv_w"], conv_anc,
        bs, msg, q, weights, sliding=sliding, eps_ln=eps_ln, emit_h=emit_h, debug=debug,
    )


def k1_from_down_batch_layer(layer, prev_moe, bs, msg, *, emit_h=True, debug=False):
    """Fused independent-batch MoE combine + short conv + K1 PRE from `prev_moe` dict."""
    sliding, q, weights, eps_ln = _layer_norm_meta(layer)
    return k1_from_down_batch(
        prev_moe["down"], prev_moe["expert_w"], prev_moe["residual"], prev_moe["cs3"], prev_moe["conv_w"],
        bs, msg, q, weights, sliding=sliding, eps_ln=eps_ln, emit_h=emit_h, debug=debug,
    )


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
    NT = TG
    ckey = ("vec", G, eps_ln)
    k = _K3B_CACHE.get(ckey)
    if k is None:
        ins = ["x"] + ["w%d" % g for g in range(G)]
        outs = ["o%d" % g for g in range(G)]
        s = [
            "uint t = threadgroup_position_in_grid.x;",
            "uint lane = thread_index_in_simdgroup; uint sg = simdgroup_index_in_threadgroup;",
            "uint row = t / %d, g = t %% %d;" % (G, G),
            "threadgroup float part[64];",
            "const device float* xr = x + (size_t(row) * %d + g) * RL_D;" % G,
            "int b0 = (int(sg) * 32 + int(lane)) * 4;",
            "int b1 = ((int(sg) + 9) * 32 + int(lane)) * 4;",
            "float4 v0 = ((const device float4*)(xr + b0))[0];",
            "float4 v1 = ((const device float4*)(xr + b1))[0];",
            "float inv = rl_rms_inv_reg(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w, %s, part, sg, lane);" % flit(eps_ln),
        ]
        for g in range(G):
            cond = ("if (g == %d)" % g) if G > 1 else ""
            s += [
                "%s {" % cond if cond else "{",
                "  bfloat4 ww0 = rl_load_raw_bf4(w%d + b0); bfloat4 ww1 = rl_load_raw_bf4(w%d + b1);" % (g, g),
                "  bfloat4 xo0 = bfloat4(bfloat16_t(glue_rms_apply<float>(v0.x, inv, ww0.x)), bfloat16_t(glue_rms_apply<float>(v0.y, inv, ww0.y)), bfloat16_t(glue_rms_apply<float>(v0.z, inv, ww0.z)), bfloat16_t(glue_rms_apply<float>(v0.w, inv, ww0.w)));",
                "  bfloat4 xo1 = bfloat4(bfloat16_t(glue_rms_apply<float>(v1.x, inv, ww1.x)), bfloat16_t(glue_rms_apply<float>(v1.y, inv, ww1.y)), bfloat16_t(glue_rms_apply<float>(v1.z, inv, ww1.z)), bfloat16_t(glue_rms_apply<float>(v1.w, inv, ww1.w)));",
                "  ((device uint2*)(o%d + size_t(row) * RL_D + b0))[0] = as_type<uint2>(xo0);" % g,
                "  ((device uint2*)(o%d + size_t(row) * RL_D + b1))[0] = as_type<uint2>(xo1);" % g,
                "}",
            ]
        k = _K3B_CACHE[ckey] = _kernel(
            ("rl_k3b_v_%d_%s" % (G, flit(eps_ln))).replace(".", "p").replace("-", "m"),
            "\n".join(s),
            tuple(ins),
            tuple(outs),
            ensure_row_contiguous=True,
        )
    r = k(inputs=[mixed] + list(ln_w), grid=(B * G * NT, 1, 1), threadgroup=(NT, 1, 1),
          output_shapes=[(B, D)] * G, output_dtypes=[_BF] * G)
    return tuple(r)


def k3b_norm_layer(layer, mixed, *, mlp=False):
    """mixed (B,3,D) for q/k/v (sliding: input_layernorm for all three) or (B,1,D) with mlp=True (post_attention_layernorm)."""
    if isinstance(layer, dict):
        if mlp:
            m = layer["post_attention_layernorm"]
            return k3b_norm(mixed, [m["weight"]], eps_ln=m.eps)
        if "input_layernorm" in layer:
            m = layer["input_layernorm"]
            w = m["weight"]
            return k3b_norm(mixed, [w, w, w], eps_ln=m.eps)
        m0, m1, m2 = layer["q_input_layernorm"], layer["k_input_layernorm"], layer["v_input_layernorm"]
        return k3b_norm(mixed, [m0["weight"], m1["weight"], m2["weight"]], eps_ln=m0.eps)
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
_K10_CHAIN_CONV_ANC = {}


def _k10_default_conv_anc(W):
    a = _K10_CHAIN_CONV_ANC.get(W)
    if a is None:
        rows = [[b - 3 if b >= 3 else -(b + 1), b - 2 if b >= 2 else -(b + 2), b - 1 if b >= 1 else -3] for b in range(W)]
        a = _K10_CHAIN_CONV_ANC[W] = mx.array(rows, mx.int32)
    return a


def k10_post(o_proj, h, nr, bs, cs2, conv_w, query, ln_w, msg, *, boundary, eps_ln=1e-6, debug=True, causal=False, conv_anc=None, w_act=None):
    """K10 POST for B rows.

    o_proj (B,D) bf16 (o_proj output), h (B,D) bf16, nr (B,D) f32 (K1's normed_residual),
    bs (n,B,D) f32 or list/tuple of n (B,D) f32 arrays or None (layer 0),
    cs2 (B,3,D) f32 [or (1,3,D) f32 when causal=True] post-attention conv state,
    conv_w post_attention_convolution.conv1d.weight (D,4,1) bf16,
    query mlp_input.query (D,) f32, ln_w post_attention_layernorm.weight (D,) bf16, msg (B,4,D) f32 or None.
    boundary: layer is a block boundary (residual := conv out; the new block is nr, appended to bs for mlp_input).
    Returns dict: w2 (B,3,D) f32, residual (B,D) f32, [mlp_src (B,D) f32 if debug or msg],
      [post_attn_conv, normed if debug], and no msg: mlp_x (B,D) bf16; msg: mlp_gate_in (B,4608) bf16.
    """
    B = h.shape[0]
    for x, dt, nm in ((o_proj, _BF, "o_proj"), (h, _BF, "h"), (nr, _F32, "nr"), (ln_w, _BF, "ln_w")):
        _check(x, (D,), dt, nm)
    if causal:
        if cs2.dtype != _F32 or cs2.shape != (1, 3, D):
            raise ValueError(f"cs2 causal: expected float32 {(1, 3, D)}, got {cs2.dtype} {cs2.shape}")
    elif cs2.dtype != _F32 or cs2.shape != (B, 3, D):
        raise ValueError(f"cs2: expected float32 {(B, 3, D)}, got {cs2.dtype} {cs2.shape}")
    if conv_w.dtype != _BF or conv_w.size != D * 4:
        raise ValueError("conv weight must be bf16 (D,4,1)")
    bs_is_list, NB = _check_bs(bs, B, "bs")
    if NB == 0 and not boundary:
        raise ValueError("no blocks to attend over")
    NS = NB + (1 if boundary else 0) + 1  # sources: blocks, [nr], partial
    if msg is not None:
        _check(msg, (4, D), _F32, "msg")
    query = query.reshape(-1, D)
    if query.shape[0] != 1 or query.dtype != _F32:
        raise ValueError("query")

    has_src = debug or msg is not None
    bs_args = list(bs) if bs_is_list else ([bs] if NB else [])
    if causal:
        if conv_anc is None:
            conv_anc = _k10_default_conv_anc(B)
        else:
            assert conv_anc.shape == (B, 3) and conv_anc.dtype == mx.int32
    has_wa = w_act is not None
    args = [o_proj, h, nr] + bs_args + [cs2, conv_w, query, ln_w] + ([msg] if msg is not None else []) + ([conv_anc] if causal else []) + ([w_act.reshape(-1)] if has_wa else [])
    shapes = ([(B, D)] if debug else []) + [(B, 3, D), (B, D)] + ([(B, D)] if debug else []) + ([(B, D)] if has_src else []) + [(B, 2 * D) if msg is not None else (B, D)]

    ckey = (NB, bs_is_list, boundary, msg is not None, eps_ln, debug, causal, has_wa)
    cached = _K10_CACHE.get(ckey)
    if cached is not None:
        k, outs, dts = cached
        r = k(inputs=args, grid=(B * TG, 1, 1), threadgroup=(TG, 1, 1), output_shapes=shapes, output_dtypes=dts)
        o = dict(zip(outs, r))
        res = dict(w2=o["w2"], residual=o["res_o"])
        if has_src:
            res["mlp_src"] = o["src"]
        if debug:
            res.update(post_attn_conv=o["conv"], normed=o["normed"])
        if msg is None:
            res["mlp_x"] = o["mx"]
        else:
            res["mlp_gate_in"] = o["gi"]
        return res

    bs_ins = [f"bs{i}" for i in range(NB)] if bs_is_list else (["bs"] if NB else [])
    ins = ["op", "h", "nr"] + bs_ins + ["cs", "cw", "query", "w"] + (["msg"] if msg is not None else []) + (["conv_anc"] if causal else []) + (["w_act"] if has_wa else [])
    outs = tuple((["conv"] if debug else []) + ["w2", "res_o"] + (["normed"] if debug else []) + (["src"] if has_src else []) + ["gi" if msg is not None else "mx"])
    dts = ([_BF] if debug else []) + [_F32, _F32] + ([_F32] if debug else []) + ([_F32] if has_src else []) + [_BF]

    def _bs_d(i):
        return f"bs{i}[size_t(row) * RL_D + d]" if bs_is_list else f"bs[(size_t({i}) * B + row) * RL_D + d]"

    def _bs_f4(i, off):
        return (
            f"((const device float4*)(bs{i} + size_t(row) * RL_D + {off}))[0]"
            if bs_is_list
            else f"((const device float4*)(bs + (size_t({i}) * B + row) * RL_D + {off}))[0]"
        )

    res_expr = "hid" if boundary else "ha[i] + hid"
    if causal:
        conv_loads = [
            "  float4 r0v = (a0 >= 0) ? rl_load_bf4(op + size_t(a0) * RL_D + base) : ((const device float4*)(cs + size_t(-1 - a0) * RL_D + base))[0];",
            "  float4 r1v = (a1 >= 0) ? rl_load_bf4(op + size_t(a1) * RL_D + base) : ((const device float4*)(cs + size_t(-1 - a1) * RL_D + base))[0];",
            "  float4 r2v = (a2 >= 0) ? rl_load_bf4(op + size_t(a2) * RL_D + base) : ((const device float4*)(cs + size_t(-1 - a2) * RL_D + base))[0];",
        ]
        anc_decl = [
            "int a0 = conv_anc[row * 3 + 0];",
            "int a1 = conv_anc[row * 3 + 1];",
            "int a2 = conv_anc[row * 3 + 2];",
        ]
    else:
        conv_loads = [
            "  float4 r0v = ((const device float4*)(st + base))[0];",
            "  float4 r1v = ((const device float4*)(st + RL_D + base))[0];",
            "  float4 r2v = ((const device float4*)(st + 2 * RL_D + base))[0];",
        ]
        anc_decl = []
    s = [
        "uint row = threadgroup_position_in_grid.x;",
        *(["if (int(row) >= w_act[0]) return;"] if has_wa else []),
        "uint B = threadgroups_per_grid.x; (void)B;",
        "uint tid = thread_position_in_threadgroup.x;",
        "uint lane = thread_index_in_simdgroup; uint sg = simdgroup_index_in_threadgroup;",
        "threadgroup float part[64];",
        f"threadgroup float dbuf[{NS * 9}];",
        "threadgroup float sm[RL_D];",
        "int b0 = (int(sg) * 32 + int(lane)) * 4;",
        "int b1 = ((int(sg) + 9) * 32 + int(lane)) * 4;",
        "const device bfloat16_t* opr = op + size_t(row) * RL_D;",
        "const device bfloat16_t* hr = h + size_t(row) * RL_D;",
        "const device float* st = cs" + ("" if causal else " + size_t(row) * 3 * RL_D") + "; (void)st;",
        "device float* w2r = w2 + size_t(row) * 3 * RL_D;",
    ] + anc_decl + [
        "float rv[2][4];",
        "for (int pass = 0; pass < 2; pass++) {",
        "  int base = (pass == 0) ? b0 : b1;",
        "  float4 x0v = rl_load_bf4(opr + base);",
    ] + conv_loads + [
        "  ((device float4*)(w2r + base))[0] = r1v;",
        "  ((device float4*)(w2r + RL_D + base))[0] = r2v;",
        "  ((device float4*)(w2r + 2 * RL_D + base))[0] = x0v;",
        "  float4 hv = rl_load_bf4(hr + base);",
        "  float x0a[4] = {x0v.x, x0v.y, x0v.z, x0v.w};",
        "  float r0a[4] = {r0v.x, r0v.y, r0v.z, r0v.w};",
        "  float r1a[4] = {r1v.x, r1v.y, r1v.z, r1v.w};",
        "  float r2a[4] = {r2v.x, r2v.y, r2v.z, r2v.w};",
        "  float ha[4] = {hv.x, hv.y, hv.z, hv.w};",
        "  for (int i = 0; i < 4; i++) {",
        "    int c = base + i;",
        "    float s0 = float(bfloat16_t(r0a[i])), s1 = float(bfloat16_t(r1a[i])), s2 = float(bfloat16_t(r2a[i]));",
        "    bfloat4 cwb = rl_load_raw_bf4(cw + c * 4);",
        "    float acc = 0.0f;",
        "    acc = fma(float(cwb.x), s0, acc);",
        "    acc = fma(float(cwb.y), s1, acc);",
        "    acc = fma(float(cwb.z), s2, acc);",
        "    acc = fma(float(cwb.w), x0a[i], acc);",
        "    float cv = float(bfloat16_t(acc));",
        f"    bfloat16_t co = bfloat16_t((cv + x0a[i]) / {_SCALE_CONV});",
    ]
    if debug:
        s += ["    conv[size_t(row) * RL_D + c] = co;"]
    s += [
        "    float hid = float(co);",
        f"    rv[pass][i] = {res_expr};",
        "  }",
        "  float4 rvf = float4(rv[pass][0], rv[pass][1], rv[pass][2], rv[pass][3]);",
        "  ((device float4*)(res_o + size_t(row) * RL_D + base))[0] = rvf;",
        "  ((threadgroup float4*)(sm + base))[0] = rvf;",
        "}",
        # Barrier inside rl_rms_inv_reg also makes sm[] visible to all 288 threads
        f"float inv1 = rl_rms_inv_reg(rv[0][0], rv[0][1], rv[0][2], rv[0][3], rv[1][0], rv[1][1], rv[1][2], rv[1][3], {flit(EPS_UW)}, part, sg, lane);",
        f"float acc[{NS}]; for (int i = 0; i < {NS}; i++) acc[i] = 0.0f;",
        "for (int k = 0; k < 8; k++) {",
        "  int d = int(tid) + k * RL_TG;",
        "  float qv = query[d];",
    ]
    for i in range(NB):
        s += [f"  acc[{i}] += {_bs_d(i)} * qv;"]
    if boundary:
        s += [f"  acc[{NB}] += nr[size_t(row) * RL_D + d] * qv;"]
    s += [
        f"  acc[{NS - 1}] += glue_rms_apply_nw<float>(sm[d], inv1) * qv;",
        "}",
        f"for (int i = 0; i < {NS}; i++) {{ float v = simd_sum(acc[i]); if (lane == 0) dbuf[i * 9 + sg] = v; }}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"float wt[{NS}]; float mxw = -INFINITY;",
        f"for (int i = 0; i < {NS}; i++) {{ float t = 0.0f; for (int ss = 0; ss < 9; ss++) t += dbuf[i * 9 + ss]; wt[i] = t * {_SCALE_DEPTH}; mxw = max(mxw, wt[i]); }}",
        f"float zz = 0.0f; for (int i = 0; i < {NS}; i++) {{ wt[i] = metal::exp(wt[i] - mxw); zz += wt[i]; }}",
        f"for (int i = 0; i < {NS}; i++) wt[i] /= zz;",
        "float4 o0 = float4(0.0f); float4 o1 = float4(0.0f);",
    ]
    for i in range(NB):
        s += [f"o0 += wt[{i}] * {_bs_f4(i, 'b0')}; o1 += wt[{i}] * {_bs_f4(i, 'b1')};"]
    if boundary:
        s += [
            f"o0 += wt[{NB}] * ((const device float4*)(nr + size_t(row) * RL_D + b0))[0];",
            f"o1 += wt[{NB}] * ((const device float4*)(nr + size_t(row) * RL_D + b1))[0];",
        ]
    s += [
        "float4 nm0 = float4(glue_rms_apply_nw<float>(rv[0][0], inv1), glue_rms_apply_nw<float>(rv[0][1], inv1), glue_rms_apply_nw<float>(rv[0][2], inv1), glue_rms_apply_nw<float>(rv[0][3], inv1));",
        "float4 nm1 = float4(glue_rms_apply_nw<float>(rv[1][0], inv1), glue_rms_apply_nw<float>(rv[1][1], inv1), glue_rms_apply_nw<float>(rv[1][2], inv1), glue_rms_apply_nw<float>(rv[1][3], inv1));",
    ]
    if debug:
        s += [
            "((device float4*)(normed + size_t(row) * RL_D + b0))[0] = nm0;",
            "((device float4*)(normed + size_t(row) * RL_D + b1))[0] = nm1;",
        ]
    s += [f"o0 += wt[{NS - 1}] * nm0; o1 += wt[{NS - 1}] * nm1;"]
    if has_src:
        s += [
            "((device float4*)(src + size_t(row) * RL_D + b0))[0] = o0;",
            "((device float4*)(src + size_t(row) * RL_D + b1))[0] = o1;",
        ]
    if msg is None:
        s += [
            f"float inv2 = rl_rms_inv_reg(o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w, {flit(eps_ln)}, part, sg, lane);",
            "bfloat4 ww0 = rl_load_raw_bf4(w + b0); bfloat4 ww1 = rl_load_raw_bf4(w + b1);",
            "bfloat4 xo0 = bfloat4(bfloat16_t(glue_rms_apply<float>(o0.x, inv2, ww0.x)), bfloat16_t(glue_rms_apply<float>(o0.y, inv2, ww0.y)), bfloat16_t(glue_rms_apply<float>(o0.z, inv2, ww0.z)), bfloat16_t(glue_rms_apply<float>(o0.w, inv2, ww0.w)));",
            "bfloat4 xo1 = bfloat4(bfloat16_t(glue_rms_apply<float>(o1.x, inv2, ww1.x)), bfloat16_t(glue_rms_apply<float>(o1.y, inv2, ww1.y)), bfloat16_t(glue_rms_apply<float>(o1.z, inv2, ww1.z)), bfloat16_t(glue_rms_apply<float>(o1.w, inv2, ww1.w)));",
            "((device uint2*)(mx + size_t(row) * RL_D + b0))[0] = as_type<uint2>(xo0);",
            "((device uint2*)(mx + size_t(row) * RL_D + b1))[0] = as_type<uint2>(xo1);",
        ]
    else:
        s += [
            "const device float* mp = msg + size_t(row) * 4 * RL_D + 3 * RL_D;",
            "float4 mv0 = ((const device float4*)(mp + b0))[0]; float4 mv1 = ((const device float4*)(mp + b1))[0];",
            "float inv2, invm;",
            f"rl_rms_inv_reg2(o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w, mv0.x, mv0.y, mv0.z, mv0.w, mv1.x, mv1.y, mv1.z, mv1.w, {flit(EPS_UW)}, part, sg, lane, inv2, invm);",
            "size_t gio = size_t(row) * 2 * RL_D;",
            "rl_store_bf4(gi + gio + b0, float4(glue_rms_apply_nw<float>(o0.x, inv2), glue_rms_apply_nw<float>(o0.y, inv2), glue_rms_apply_nw<float>(o0.z, inv2), glue_rms_apply_nw<float>(o0.w, inv2)));",
            "rl_store_bf4(gi + gio + b1, float4(glue_rms_apply_nw<float>(o1.x, inv2), glue_rms_apply_nw<float>(o1.y, inv2), glue_rms_apply_nw<float>(o1.z, inv2), glue_rms_apply_nw<float>(o1.w, inv2)));",
            "rl_store_bf4(gi + gio + RL_D + b0, float4(glue_rms_apply_nw<float>(mv0.x, invm), glue_rms_apply_nw<float>(mv0.y, invm), glue_rms_apply_nw<float>(mv0.z, invm), glue_rms_apply_nw<float>(mv0.w, invm)));",
            "rl_store_bf4(gi + gio + RL_D + b1, float4(glue_rms_apply_nw<float>(mv1.x, invm), glue_rms_apply_nw<float>(mv1.y, invm), glue_rms_apply_nw<float>(mv1.z, invm), glue_rms_apply_nw<float>(mv1.w, invm)));",
        ]
    src = "\n".join(s)
    name = "rl_k10_%d_%d_%d_%d_%s%s%s%s" % (
        NB, int(bs_is_list), int(boundary), int(msg is not None), flit(eps_ln), "_c" if causal else "", "_wa" if has_wa else "", "_dbg" if debug else ""
    )
    k = _kernel(name.replace(".", "p").replace("-", "m"), src, tuple(ins), outs)
    _K10_CACHE[ckey] = (k, outs, dts)
    r = k(inputs=args, grid=(B * TG, 1, 1), threadgroup=(TG, 1, 1), output_shapes=shapes, output_dtypes=dts)
    o = dict(zip(outs, r))
    res = dict(w2=o["w2"], residual=o["res_o"])
    if has_src:
        res["mlp_src"] = o["src"]
    if debug:
        res.update(post_attn_conv=o["conv"], normed=o["normed"])
    if msg is None:
        res["mlp_x"] = o["mx"]
    else:
        res["mlp_gate_in"] = o["gi"]
    return res


def k10_post_layer(layer, o_proj, h, nr, bs, cs2, msg, *, debug=True, causal=False, conv_anc=None, w_act=None):
    if isinstance(layer, dict):
        m = layer["post_attention_layernorm"]
        return k10_post(
            o_proj, h, nr, bs, cs2,
            layer["post_attention_convolution"]["conv1d"]["weight"],
            layer["mlp_input"]["query"],
            m["weight"],
            msg,
            boundary=layer.is_block_boundary,
            eps_ln=m.eps,
            debug=debug,
            causal=causal,
            conv_anc=conv_anc,
            w_act=w_act,
        )
    m = layer.post_attention_layernorm
    return k10_post(o_proj, h, nr, bs, cs2, layer.post_attention_convolution.conv1d.weight, layer.mlp_input.query, m.weight,
                    msg, boundary=layer.is_block_boundary, eps_ln=m.eps, debug=debug, causal=causal, conv_anc=conv_anc, w_act=w_act)
