"""Lane e5: exact "glue" device functions between the row kernels.

Everything in GLUE_HEADER is Metal source meant to be pasted (``header=``) into an ``mx.fast.metal_kernel`` and called from
a larger fused kernel.  Every function reproduces the bits MLX 0.32.4 produces for the B=1 decode shapes, for any number of
rows in one launch.  Nothing here is wired into the decode path.  Details, thread layouts and test results: docs/GLUE.md.

Conventions
  * ``lane`` = thread_index_in_simdgroup, ``lid`` = thread_position_in_threadgroup.x, TG = real threadgroup size
    (a multiple of 32).  Functions named ``*_sg`` need one simdgroup (all 32 lanes, no barrier); ``*_tg`` need the whole
    threadgroup (barriers) and a ``threadgroup float[32]`` scratch.
  * Every function starts with GL_NOFMA: the compiler may not contract a*b+c into an fma unless MLX's own kernel does,
    in which case the function calls fma() explicitly.
  * The functions emulate MLX's *virtual* thread layout, so the real threadgroup can be any multiple of 32.
"""

import functools

import mlx.core as mx
import numpy as np

__all__ = ["GLUE_HEADER", "flit"]


def flit(x) -> str:
    """A float literal that Metal parses to exactly float32(x)."""
    s = "%.9g" % float(np.float32(x))
    if "." not in s and "e" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s + "f"


GLUE_HEADER = r"""
#define GL_NOFMA _Pragma("clang fp contract(off)")

// ------------------------------------------------------------------ (c) unary ops (kernels/unary_ops.h)
inline float glue_sigmoid(float x) { GL_NOFMA float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x))); return (x < 0) ? y : 1.0f - y; }
inline float glue_tanh(float x) { return metal::precise::tanh(x); }
inline float glue_exp(float x) { return metal::precise::exp(x); }
inline float glue_log(float x) { return metal::precise::log(x); }
inline float glue_sqrt(float x) { return metal::precise::sqrt(x); }
// binary_ops.h Minimum / Maximum for floats
inline float glue_min(float x, float y) { return metal::isnan(x) ? x : (x < y ? x : y); }
inline float glue_max(float x, float y) { return metal::isnan(x) ? x : (x > y ? x : y); }

// ------------------------------------------------------------------ (a) mx.fast.rms_norm (kernels/rms_norm.metal rms_single_row)
// MLX: threadgroup of VT = 32*ceil(ceil(N/4)/32) virtual threads, thread v reads x[4v..4v+3], acc = sum x*x sequentially from 0
// (RF=true: acc = fma(x,x,acc), the metallib contracts), simd_sum, per-simdgroup partials -> simd_sum(lane < NSG ? part : 0),
// inv = precise::rsqrt(acc / N + eps).  N <= 4096.
//
// _sg: one simdgroup computes the whole row (each lane plays NSG virtual lanes in a loop).  No barrier, no scratch.
template <int N, bool RF, typename P>
inline float glue_rms_inv_sg(P x, float eps, uint lane) {
  GL_NOFMA
  constexpr int NSG = (((N + 3) / 4 + 31) / 32);
  float v = 0.0f;
  for (int g = 0; g < NSG; g++) {
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) {
      int idx = (g * 32 + int(lane)) * 4 + i;
      float xi = (idx < N) ? float(x[idx]) : 0.0f;
      acc = RF ? fma(xi, xi, acc) : (acc + xi * xi);
    }
    acc = simd_sum(acc);
    if (lane == uint(g)) v = acc;
  }
  float tot = simd_sum(v);
  return metal::precise::rsqrt(tot / float(N) + eps);
}
// _tg: the whole threadgroup (TG threads, multiple of 32) computes the row; virtual simdgroup g runs on real simdgroup g % (TG/32).
// part: threadgroup float[32] scratch.  Ends with a barrier so part may be reused immediately.
template <int N, int TG, bool RF, typename P>
inline float glue_rms_inv_tg(P x, float eps, threadgroup float* part, uint lid) {
  GL_NOFMA
  constexpr int NSG = (((N + 3) / 4 + 31) / 32);
  uint lane = lid & 31, sg = lid >> 5;
  for (int g = sg; g < NSG; g += TG / 32) {
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) {
      int idx = (g * 32 + int(lane)) * 4 + i;
      float xi = (idx < N) ? float(x[idx]) : 0.0f;
      acc = RF ? fma(xi, xi, acc) : (acc + xi * xi);
    }
    acc = simd_sum(acc);
    if (lane == 0) part[g] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = simd_sum(lane < uint(NSG) ? part[lane] : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return metal::precise::rsqrt(tot / float(N) + eps);
}
// output element: w * T(x * inv), T = output dtype (= promoted dtype of x and w; w is converted to T first).
template <typename T, typename W>
inline T glue_rms_apply(float xi, float inv, W w) { return T(w) * T(xi * inv); }
// unweighted (weight array(1)): T(x * inv)
template <typename T>
inline T glue_rms_apply_nw(float xi, float inv) { return T(xi * inv); }

// ------------------------------------------------------------------ (b) _l2_clip  (layers.py)
// states = f32(x) * scale; ss = sum(states*states) over the last axis (separate materialized Multiply, then MLX Reduce);
// norm = sqrt(ss); out = bf16(states * min(thr / (norm + 1e-6), 1)).
// Row of 128 (q: 24 rows, k: 8 rows): MLX row_reduce_looped with a 32-thread group: lane l sums elements 4l..4l+3
// sequentially (total = x + total from 0), then simd_sum.  s = the lane's 4 (already scaled) states.  Needs one full simdgroup.
inline void glue_l2_clip_lane128(thread float* s, float thr) {
  GL_NOFMA
  float acc = 0.0f;
  for (int i = 0; i < 4; i++) { float p = s[i] * s[i]; acc = p + acc; }
  float tot = simd_sum(0.0f + acc);
  float norm = metal::precise::sqrt(tot);
  float f = glue_min(thr / (norm + 1e-6f), 1.0f);
  for (int i = 0; i < 4; i++) s[i] = s[i] * f;
}
// Row of 16 (rel: 24 rows): MLX row_reduce_small, one thread per row, sequential sum.  Single lane, no simd ops.
inline void glue_l2_clip_row16(thread float* s, float thr) {
  GL_NOFMA
  float acc = 0.0f;
  for (int i = 0; i < 16; i++) { float p = s[i] * s[i]; acc = p + acc; }
  float norm = metal::precise::sqrt(acc);
  float f = glue_min(thr / (norm + 1e-6f), 1.0f);
  for (int i = 0; i < 16; i++) s[i] = s[i] * f;
}

// ------------------------------------------------------------------ (d) MoE combine
// (f32(es) * w[...,None]).sum(axis=-2): MLX col_reduce_small (8x2304): thread y holds p_y = es_y * w_y (a materialized f32
// product), totals = p_y + 0, then thread y == 0 adds the 8 partials in order: t = p0; t = p1 + t; ... p7 + t.
inline float glue_moe_combine8(thread float* es, thread float* w) {
  GL_NOFMA
  float t = (es[0] * w[0]) + 0.0f;
  for (int j = 1; j < 8; j++) { float p = es[j] * w[j]; t = p + t; }
  return t;
}

// ------------------------------------------------------------------ (e) fused elementwise
// _softcapped_swiglu: gate/up are bf16 -> f32; (tanh(g/GC)*GC * sigmoid(g)) * (tanh(u/UC)*UC)
inline float glue_swiglu(float g, float u, float gc, float uc) {
  GL_NOFMA
  float a = glue_tanh(g / gc) * gc;
  float b = glue_tanh(u / uc) * uc;
  return (a * glue_sigmoid(g)) * b;
}
// MessageGate combine: (1 - tanh(f)) * h + tanh(m) * msg.  MV selects the contraction MLX's compiled kernel performs:
// 0 = none, 1 = fma(tanh(m), msg, (1-tanh f)*h), 2 = fma(1-tanh f, h, tanh(m)*msg).
template <int MV>
inline float glue_msg_combine(float f, float m, float h, float msg) {
  GL_NOFMA
  float tf = glue_tanh(f), tm = glue_tanh(m);
  float omf = 1.0f - tf;
  if (MV == 1) return fma(tm, msg, omf * h);
  if (MV == 2) return fma(omf, h, tm * msg);
  return omf * h + tm * msg;
}
// attn * sigmoid(gate), both f32
inline float glue_gate(float a, float g) { GL_NOFMA return a * glue_sigmoid(g); }
// tau scaling (q and rel): f32(x) * tau
inline float glue_tau(float x, float tau) { GL_NOFMA return x * tau; }

// ------------------------------------------------------------------ (g) logsumexp (kernels/logsumexp.h logsumexp_looped, N > 4096)
// MLX runs 1024 virtual threads per row.  Real TG threads (multiple of 32 dividing 1024) emulate them: virtual simdgroup
// g = sg + k*(TG/32).  lmax/lnrm: threadgroup float[32] each.  Result valid in every thread.
template <int TG>
inline float glue_logsumexp(const device float* in, int n, uint lid, threadgroup float* lmax, threadgroup float* lnrm) {
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
      if (offset + 4 <= n) { for (int i = 0; i < 4; i++) vals[i] = in[offset + i]; }
      else { for (int i = 0; i < 4; i++) vals[i] = (offset + i < n) ? in[offset + i] : -INFINITY; }
      prevmax = maxval;
      for (int i = 0; i < 4; i++) maxval = (maxval < vals[i]) ? vals[i] : maxval;
      normalizer *= metal::fast::exp(prevmax - maxval);
      for (int i = 0; i < 4; i++) normalizer += metal::fast::exp(vals[i] - maxval);
    }
    prevmax = maxval;
    maxval = simd_max(maxval);
    normalizer *= metal::fast::exp(prevmax - maxval);
    normalizer = simd_sum(normalizer);
    gmx[k] = maxval; nrm[k] = normalizer;
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

// argmax over (value, index): MLX ArgMax semantics (arg_reduce.metal): larger value wins, ties -> lower index, the lowest
// index NaN beats everything.  The reduction is a total order, so any partition of the row gives the same answer.
struct GlueIV { float val; uint idx; };
inline GlueIV glue_iv_reduce(GlueIV best, GlueIV cur) {
  if ((metal::isnan(cur.val) && (!metal::isnan(best.val) || best.idx > cur.idx)) ||
      (!metal::isnan(best.val) && (best.val < cur.val || (best.val == cur.val && best.idx > cur.idx)))) return cur;
  return best;
}
// row of n values lp[i] = in[i] - sub  (sub = lse; pass 0 and the f32 subtraction is still exact: x - 0 == x except -0)
// buf: threadgroup GlueIV[32].  Result valid in every thread.
template <int TG>
inline uint glue_argmax_sub(const device float* in, int n, float sub, bool do_sub, uint lid, threadgroup GlueIV* buf) {
  uint lane = lid & 31, sg = lid >> 5;
  GlueIV best; best.val = -INFINITY; best.idx = 0;
  for (int i = int(lid); i < n; i += TG) {
    GlueIV c; c.val = do_sub ? in[i] - sub : in[i]; c.idx = uint(i);
    best = glue_iv_reduce(best, c);
  }
  for (uint d = 16; d > 0; d /= 2) {
    GlueIV o; o.val = simd_shuffle_down(best.val, d); o.idx = simd_shuffle_down(best.idx, d);
    best = glue_iv_reduce(best, o);
  }
  if (lane == 0) buf[sg] = best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  GlueIV f; f.val = -INFINITY; f.idx = 0;
  if (lane < TG / 32) f = buf[lane];
  for (uint d = 16; d > 0; d /= 2) {
    GlueIV o; o.val = simd_shuffle_down(f.val, d); o.idx = simd_shuffle_down(f.idx, d);
    f = glue_iv_reduce(f, o);
  }
  uint r = simd_broadcast_first(f.idx);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}
"""


@functools.lru_cache(maxsize=None)
def _kernel(name, source, inputs, outputs, header=GLUE_HEADER):
    return mx.fast.metal_kernel(
        name=name, input_names=list(inputs), output_names=list(outputs), source=source, header=header
    )


def _ident(x) -> str:
    return "%08x" % int(np.float32(x).view(np.uint32))


def _tname(dt):
    return {mx.float32: "float", mx.bfloat16: "bfloat16_t", mx.float16: "half"}[dt]


# ---------------------------------------------------------------------------------------------------- test kernels
def rms_norm_rows(x, w, eps, *, out_dtype, TG=256, mode="tg", rf=True):
    """x: (B, N) any float dtype, w: (N,) or None; one threadgroup (mode tg, TG threads) or one simdgroup (mode sg) per row.
    Reproduces mx.fast.rms_norm(x.astype(out_dtype), w.astype(out_dtype) or None, eps) on each row alone."""
    B, N = x.shape
    if mode == "sg":
        TG = 32
    inv = (
        f"glue_rms_inv_tg<N, TG, RF>(xr, eps, part, lid)"
        if mode == "tg"
        else "glue_rms_inv_sg<N, RF>(xr, eps, lid & 31)"
    )
    if w is None:
        store = "out[size_t(row) * N + i] = glue_rms_apply_nw<TO>(xi, inv);"
        ins = ("x", "eps")
    else:
        store = "out[size_t(row) * N + i] = glue_rms_apply<TO>(xi, inv, w[i]);"
        ins = ("x", "w", "eps")
    src = f"""
        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        threadgroup float part[32];
        const device TX* xr = x + size_t(row) * N;
        float inv = {inv};
        for (uint i = lid; i < N; i += TG) {{
            float xi = float(xr[i]);
            {store}
        }}
    """
    k = _kernel("glue_rms_" + mode + ("_w" if w is not None else "_nw"), src, ins, ("out",))
    args = [x] + ([w] if w is not None else []) + [mx.array(eps, mx.float32)]
    return k(
        inputs=args,
        template=[("TX", x.dtype), ("TO", out_dtype), ("N", N), ("TG", TG), ("RF", rf)],
        grid=(B * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(B, N)],
        output_dtypes=[out_dtype],
    )[0]


def l2_clip_rows(x, scale, thr, *, nrows_per_b):
    """x: (B*H, D) bf16, D in (128, 16); scale: (H, D) bf16 or None (scale 1.0).  Returns bf16 (B*H, D).  D=128: one simdgroup
    per row; D=16: one thread per row.  H = nrows_per_b (heads)."""
    R, D = x.shape
    H = nrows_per_b
    sc = ("float(sc[(row % H) * 128 + lane * 4 + i])" if scale is not None else "1.0f") if D == 128 else None
    if D == 128:
        src = f"""
            uint row = threadgroup_position_in_grid.x;
            uint lane = thread_index_in_simdgroup;
            float s[4];
            for (int i = 0; i < 4; i++) s[i] = float(x[size_t(row) * 128 + lane * 4 + i]) * {sc};
            glue_l2_clip_lane128(s, THR);
            for (int i = 0; i < 4; i++) out[size_t(row) * 128 + lane * 4 + i] = bfloat16_t(s[i]);
        """
        grid, tg = (R * 32, 1, 1), (32, 1, 1)
    else:
        scl = "float(sc[(row % H) * 16 + i])" if scale is not None else "1.0f"
        src = f"""
            uint row = thread_position_in_grid.x;
            if (row >= R) return;
            float s[16];
            for (int i = 0; i < 16; i++) s[i] = float(x[size_t(row) * 16 + i]) * {scl};
            glue_l2_clip_row16(s, THR);
            for (int i = 0; i < 16; i++) out[size_t(row) * 16 + i] = bfloat16_t(s[i]);
        """
        grid, tg = (((R + 31) // 32) * 32, 1, 1), (32, 1, 1)
    src = src.replace("THR", flit(thr))
    ins = ("x", "sc") if scale is not None else ("x",)
    k = _kernel(f"glue_l2_{D}_{'s' if scale is not None else 'n'}_{_ident(thr)}", src, ins, ("out",))
    return k(
        inputs=[x] + ([scale] if scale is not None else []),
        template=[("H", H), ("R", R)],
        grid=grid,
        threadgroup=tg,
        output_shapes=[(R, D)],
        output_dtypes=[mx.bfloat16],
    )[0]


def elementwise(name, expr, inputs, out_dtype, *, ctypes=None):
    """out[i] = TO(expr) with the arrays in `inputs` (dict name -> flat array) as a0.. via their names; expr uses
    float(<name>[i]).  Used for the unary / elementwise tests."""
    names = tuple(inputs)
    n = inputs[names[0]].size
    src = f"""
        uint i = thread_position_in_grid.x;
        if (i >= n) return;
        {''.join(f'float v_{k} = float({k}[i]); ' for k in names)}
        out[i] = TO({expr});
    """
    k = _kernel("glue_ew_" + name, src, names, ("out",))
    return k(
        inputs=[inputs[k_] for k_ in names],
        template=[("TO", out_dtype), ("n", n)],
        grid=(((n + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[inputs[names[0]].shape],
        output_dtypes=[out_dtype],
    )[0]


def convert_roundtrip(bits_u32):
    """f32 bit patterns -> bf16 (as uint16 bits) and bf16 bits -> f32 (as uint32 bits), through the kernel."""
    src = """
        uint i = thread_position_in_grid.x;
        if (i >= n) return;
        float f = as_type<float>(a[i]);
        bfloat16_t b = bfloat16_t(f);
        o1[i] = as_type<ushort>(b);
        o2[i] = as_type<uint>(float(b));
    """
    k = _kernel("glue_convert", src, ("a",), ("o1", "o2"))
    n = bits_u32.size
    return k(
        inputs=[bits_u32],
        template=[("n", n)],
        grid=(((n + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,), (n,)],
        output_dtypes=[mx.uint16, mx.uint32],
    )


def moe_combine(es, w):
    """es: (B, 8, C) bf16, w: (B, 8) f32 -> (B, C) bf16; one thread per output column."""
    B, _, C = es.shape
    src = """
        uint t = thread_position_in_grid.x;
        uint b = t / C, c = t % C;
        if (b >= B) return;
        float e[8], ww[8];
        for (int j = 0; j < 8; j++) { e[j] = float(es[(size_t(b) * 8 + j) * C + c]); ww[j] = w[b * 8 + j]; }
        out[size_t(b) * C + c] = bfloat16_t(glue_moe_combine8(e, ww));
    """
    k = _kernel("glue_moe_combine", src, ("es", "w"), ("out",))
    n = B * C
    return k(
        inputs=[es, w],
        template=[("B", B), ("C", C)],
        grid=(((n + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, C)],
        output_dtypes=[mx.bfloat16],
    )[0]


def swiglu(g, u, gc, uc):
    """g, u bf16 (n,) -> bf16 (n,)"""
    n = g.size
    src = f"""
        uint i = thread_position_in_grid.x;
        if (i >= n) return;
        out[i] = bfloat16_t(glue_swiglu(float(g[i]), float(u[i]), {flit(gc)}, {flit(uc)}));
    """
    k = _kernel(f"glue_swiglu_{_ident(gc)}_{_ident(uc)}", src, ("g", "u"), ("out",))
    return k(
        inputs=[g, u],
        template=[("n", n)],
        grid=(((n + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[g.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def msg_combine(f, m, h, msg, mv):
    """f, m, h, msg f32 (n,) -> f32 (n,)"""
    n = f.size
    src = """
        uint i = thread_position_in_grid.x;
        if (i >= n) return;
        out[i] = glue_msg_combine<MV>(f[i], m[i], h[i], msg[i]);
    """
    k = _kernel("glue_msg_combine", src, ("f", "m", "h", "msg"), ("out",))
    return k(
        inputs=[f, m, h, msg],
        template=[("n", n), ("MV", mv)],
        grid=(((n + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[f.shape],
        output_dtypes=[mx.float32],
    )[0]


def head_rows(lg, *, TG=256, do_sub=True):
    """lg: (B, V) f32 softcapped logits.  Returns (lse (B,) f32, argmax(lg - lse) (B,) uint32) as the sampler path computes
    it (logprobs = logits - logsumexp(logits, keepdims=True); argmax(logprobs))."""
    B, V = lg.shape
    src = """
        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        threadgroup float lmax[32];
        threadgroup float lnrm[32];
        threadgroup GlueIV buf[32];
        const device float* r = lg + size_t(row) * V;
        float lse = glue_logsumexp<TG>(r, V, lid, lmax, lnrm);
        uint idx = glue_argmax_sub<TG>(r, V, lse, DO_SUB, lid, buf);
        if (lid == 0) { lse_o[row] = lse; idx_o[row] = idx; }
    """
    k = _kernel("glue_head", src, ("lg",), ("lse_o", "idx_o"))
    return k(
        inputs=[lg],
        template=[("V", V), ("TG", TG), ("DO_SUB", do_sub)],
        grid=(B * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(B,), (B,)],
        output_dtypes=[mx.float32, mx.uint32],
    )


def softcap_rows(x, cap, V):
    """x: (B, Vp) f32 raw logits -> (B, V) f32  tanh(x / cap) * cap (the slice [..., :V] fused)."""
    B, Vp = x.shape
    src = f"""
        uint i = thread_position_in_grid.x;
        uint b = i / V, c = i % V;
        if (b >= B) return;
        float v = x[size_t(b) * Vp + c];
        out[i] = glue_tanh(v / {flit(cap)}) * {flit(cap)};
    """
    k = _kernel(f"glue_softcap_{_ident(cap)}", src, ("x",), ("out",))
    n = B * V
    return k(
        inputs=[x],
        template=[("B", B), ("V", V), ("Vp", Vp)],
        grid=(((n + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, V)],
        output_dtypes=[mx.float32],
    )[0]
