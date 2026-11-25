"""Fused bit-exact Metal kernels for L > 1 prefill (LOOPLING_FAST_PREFILL=1 default)."""
import math
import os

import mlx.core as mx
import numpy as np

from mlx_lm.models.inkling2.rows.glue import GLUE_HEADER

_FAST = os.environ.get("LOOPLING_FAST_PREFILL", "1") not in ("0", "")
_SPLASH_MSGGATE = os.environ.get("LOOPLING_SPLASH_MSGGATE", "0") not in ("0", "", "false", "False")
_D = 2304
_TG_RMS = 576
_DUMMY_F32 = mx.zeros((256,), mx.float32)  # >= 512 B ensures Metal device address space
_S5_LIT = repr(float(np.float32(math.sqrt(5.0)))) + "f"
_S9_LIT = repr(float(np.float32(math.sqrt(9.0)))) + "f"


def active(dtype: mx.Dtype, x: mx.array) -> bool:
    return (
        _FAST
        and dtype == mx.bfloat16
        and x.ndim >= 3
        and x.shape[1] > 1
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


def enabled(x: mx.array) -> bool:
    return active(x.dtype, x)


def msg_gate_ok(mod, h: mx.array, msg: mx.array, ln) -> bool:
    return (
        h.dtype == mx.float32
        and msg.dtype == mx.float32
        and h.shape[-1] == _D
        and getattr(mod.down_proj, "weight", None) is not None
        and mod.down_proj.weight.dtype == mx.bfloat16
        and mod.forget_gate_proj.weight.dtype == mx.float32
        and mod.message_gate_proj.weight.dtype == mx.float32
        and ln.weight.dtype == mx.bfloat16
    )


_kernels: dict = {}


def _k(name, ins, outs, src):
    k = _kernels.get(name)
    if k is None:
        k = _kernels[name] = mx.fast.metal_kernel(
            name=name,
            input_names=list(ins),
            output_names=list(outs),
            source=src,
            header=GLUE_HEADER,
            ensure_row_contiguous=True,
        )
    return k


# ---------------------------------------------------------------------------------------------------- MoE combine
_UC_SRC = r"""
    GL_NOFMA
    uint d = thread_position_in_grid.x;
    uint tok = thread_position_in_grid.y;
    if (d >= DD) return;
    float ev[8], ww[8];
    for (uint e = 0; e < 8; e++) {
        uint p = tok * 8 + e;
        ev[e] = float(dn[(size_t)inv[p] * DD + d]);
        ww[e] = wts[p];
    }
    out[(size_t)tok * DD + d] = bfloat16_t(glue_moe_combine8(ev, ww));
"""


def unsort_combine(dn_sorted: mx.array, inv_order: mx.array, routing_weights: mx.array) -> mx.array:
    """Fuse _scatter_unsort + (es.f32 * wts[...,None]).sum(-2).astype(bf16) for K=8."""
    B, L, E = routing_weights.shape
    D = dn_sorted.shape[-1]
    assert E == 8 and dn_sorted.dtype == mx.bfloat16
    k = _k("pf_unsort_combine", ("dn", "inv", "wts"), ("out",), _UC_SRC)
    return k(
        inputs=[dn_sorted, inv_order.astype(mx.uint32), routing_weights.astype(mx.float32)],
        template=[("DD", D)],
        grid=(((D + 255) // 256) * 256, B * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, D)],
        output_dtypes=[mx.bfloat16],
    )[0]


# ---------------------------------------------------------------------------------------------------- ShortConvolution
_SC_SRC = f"""
    uint c = thread_position_in_grid.x;
    uint tok = thread_position_in_grid.y;
    uint b = tok / LL;
    uint t = tok % LL;
    if (c >= DD) return;
    float x3 = float(x[(size_t)tok * DD + c]);
    auto tap = [&](int idx) -> float {{
        int p = int(t) - idx;
        if (p >= 0) return float(x[(size_t)(b * LL + p) * DD + c]);
        int sidx = 3 + p;
        return HAS_CS ? float(bfloat16_t(cs[((size_t)b * 3 + sidx) * DD + c])) : 0.0f;
    }};
    float acc = 0.0f;
    acc = fma(float(cw[c * 4 + 0]), tap(3), acc);
    acc = fma(float(cw[c * 4 + 1]), tap(2), acc);
    acc = fma(float(cw[c * 4 + 2]), tap(1), acc);
    acc = fma(float(cw[c * 4 + 3]), x3, acc);
    float conv = float(bfloat16_t(acc));
    out[(size_t)tok * DD + c] = bfloat16_t((conv + x3) / {_S5_LIT});
    if (t == 0) {{
        for (int r = 0; r < 3; r++) {{
            int p = int(LL) - 3 + r;
            ncs[((size_t)b * 3 + r) * DD + c] = (p >= 0)
                ? float(x[(size_t)(b * LL + p) * DD + c])
                : (HAS_CS ? cs[((size_t)b * 3 + (3 + p)) * DD + c] : 0.0f);
        }}
    }}
"""


def short_conv(x: mx.array, conv_state: mx.array | None, weight: mx.array) -> tuple[mx.array, mx.array]:
    """K=4 causal depthwise conv + residual/sqrt(5) -> (out bf16, new_cs (B,3,C) f32)."""
    B, L, C = x.shape
    has_cs = int(conv_state is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    k = _k("pf_short_conv", ("x", "cs", "cw"), ("out", "ncs"), _SC_SRC)
    return k(
        inputs=[x, cs_in, weight],
        template=[("DD", C), ("LL", L), ("HAS_CS", has_cs)],
        grid=(((C + 255) // 256) * 256, B * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, C), (B, 3, C)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


# ---------------------------------------------------------------------------------------------------- DynamicShortConv
_DSC_SRC = f"""
    uint c = thread_position_in_grid.x;
    uint tok = thread_position_in_grid.y;
    uint b = tok / LL;
    uint t = tok % LL;
    if (c >= CC) return;
    uint h = c / HD;
    float x0 = float(x[(size_t)tok * CC + c]);
    auto tap = [&](int lag) -> float {{
        int p = int(t) - lag;
        if (p >= 0) return float(x[(size_t)(b * LL + p) * CC + c]);
        int sidx = 3 + p;
        return HAS_CS ? cs[((size_t)b * 3 + sidx) * CC + c] : 0.0f;
    }};
    float p0 = tap(0) * (float(sw[0 * CC + c]) + float(dw[((size_t)tok * HH + h) * 4 + 0]));
    float p1 = tap(1) * (float(sw[1 * CC + c]) + float(dw[((size_t)tok * HH + h) * 4 + 1]));
    float p2 = tap(2) * (float(sw[2 * CC + c]) + float(dw[((size_t)tok * HH + h) * 4 + 2]));
    float p3 = tap(3) * (float(sw[3 * CC + c]) + float(dw[((size_t)tok * HH + h) * 4 + 3]));
    float s = ((p0 + p1) + p2) + p3;
    out[(size_t)tok * CC + c] = bfloat16_t((s + x0) / {_S9_LIT});
    if (t == 0) {{
        for (int r = 0; r < 3; r++) {{
            int p = int(LL) - 3 + r;
            ncs[((size_t)b * 3 + r) * CC + c] = (p >= 0)
                ? float(x[(size_t)(b * LL + p) * CC + c])
                : (HAS_CS ? cs[((size_t)b * 3 + (3 + p)) * CC + c] : 0.0f);
        }}
    }}
"""


def dyn_short_conv(
    x: mx.array, dyn_weights: mx.array, static_weight: mx.array, conv_state: mx.array | None, num_heads: int
) -> tuple[mx.array, mx.array]:
    """K=4 dynamic short conv + residual/sqrt(9) -> (out bf16, new_cs (B,3,C) f32)."""
    B, L, C = x.shape
    has_cs = int(conv_state is not None)
    cs_in = conv_state if conv_state is not None else _DUMMY_F32
    k = _k("pf_dyn_short_conv", ("x", "dw", "sw", "cs"), ("out", "ncs"), _DSC_SRC)
    return k(
        inputs=[x, dyn_weights, static_weight, cs_in],
        template=[("CC", C), ("LL", L), ("HH", num_heads), ("HD", C // num_heads), ("HAS_CS", has_cs)],
        grid=(((C + 255) // 256) * 256, B * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, C), (B, 3, C)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


# ---------------------------------------------------------------------------------------------------- MessageGate fused
_MGIN_SINGLE_SRC = r"""
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    threadgroup float part[32];
    const device float* hr = h + size_t(row) * DD;
    const device float* mr = msg + (size_t(row) * MSTRIDE + SLOT) * DD;
    float invh = glue_rms_inv_tg<DD, TG, true>(hr, 1e-6f, part, lid);
    float invm = glue_rms_inv_tg<DD, TG, true>(mr, 1e-6f, part, lid);
    for (uint d = lid; d < DD; d += TG) {
        out[size_t(row) * 2 * DD + d] = glue_rms_apply_nw<bfloat16_t>(hr[d], invh);
        out[size_t(row) * 2 * DD + DD + d] = glue_rms_apply_nw<bfloat16_t>(mr[d], invm);
    }
"""

_MGIN_QKV_SRC = r"""
    uint row = threadgroup_position_in_grid.x;
    uint g = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    threadgroup float part[32];
    const device float* hr = h + (SHARED_H ? size_t(row) : size_t(row) * 3 + g) * DD;
    const device float* mr = msg + (size_t(row) * 4 + g) * DD;
    float invh = glue_rms_inv_tg<DD, TG, true>(hr, 1e-6f, part, lid);
    float invm = glue_rms_inv_tg<DD, TG, true>(mr, 1e-6f, part, lid);
    device bfloat16_t* dst = (g == 0 ? o0 : (g == 1 ? o1 : o2)) + size_t(row) * 2 * DD;
    for (uint d = lid; d < DD; d += TG) {
        dst[d] = glue_rms_apply_nw<bfloat16_t>(hr[d], invh);
        dst[DD + d] = glue_rms_apply_nw<bfloat16_t>(mr[d], invm);
    }
"""

_MIXNORM_SINGLE_SRC = r"""
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    threadgroup float part[32];
    threadgroup float buf[DD];
    size_t base = size_t(row) * DD;
    size_t mbase = (size_t(row) * 4 + SLOT) * DD;
    for (uint d = lid; d < DD; d += TG) {
        buf[d] = glue_msg_combine<0>(float(fl[base + d]), float(ml[base + d]), h[base + d], msg[mbase + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = glue_rms_inv_tg<DD, TG, true>(buf, eps[0], part, lid);
    for (uint d = lid; d < DD; d += TG) {
        out[base + d] = bfloat16_t(glue_rms_apply<float>(buf[d], inv, nw[d]));
    }
"""

_MIXNORM_QKV_SRC = r"""
    uint row = threadgroup_position_in_grid.x;
    uint g = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    threadgroup float part[32];
    threadgroup float buf[DD];
    size_t base = size_t(row) * DD;
    size_t hbase = (SHARED_H ? size_t(row) : size_t(row) * 3 + g) * DD;
    size_t mbase = (size_t(row) * 4 + g) * DD;
    const device float* flp = (g == 0 ? fl0 : (g == 1 ? fl1 : fl2)) + base;
    const device float* mlp = (g == 0 ? ml0 : (g == 1 ? ml1 : ml2)) + base;
    const device bfloat16_t* nwp = SHARED_LN ? nw0 : (g == 0 ? nw0 : (g == 1 ? nw1 : nw2));
    device bfloat16_t* dst = (g == 0 ? o0 : (g == 1 ? o1 : o2)) + base;
    for (uint d = lid; d < DD; d += TG) {
        buf[d] = glue_msg_combine<0>(flp[d], mlp[d], h[hbase + d], msg[mbase + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = glue_rms_inv_tg<DD, TG, true>(buf, eps[0], part, lid);
    for (uint d = lid; d < DD; d += TG) {
        dst[d] = bfloat16_t(glue_rms_apply<float>(buf[d], inv, nwp[d]));
    }
"""


# ---------------------------------------------------------------------------------------------------- Splash tiled MessageGate
_SPLASH_MG_HEADER = (
    "#pragma clang fp reassociate(off)\n"
    "#include <metal_simdgroup_matrix>\n"
    + GLUE_HEADER
    + r"""
namespace mg_sg {
struct Lane { ushort fm; ushort fn; };
inline Lane lane_map(uint lane) {
  const uint qid = lane >> 2;
  return { ushort((qid & 4) | ((lane >> 1) & 3)), ushort(((qid & 2) << 1) | ((lane & 1) << 1)) };
}
template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2>& te(thread metal::simdgroup_matrix<T, 8, 8>& m) {
  return reinterpret_cast<thread vec<T, 2>&>(m.thread_elements());
}
__attribute__((always_inline)) inline void mma_acc(thread float2& c, float2 a, float2 b) {
  metal::simdgroup_matrix<float, 8, 8> A, B, C, D;
  te(A) = a; te(B) = b; te(C) = c;
  metal::simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}
}  // namespace mg_sg

constant constexpr uint kMgHidden = 2304;
constant constexpr uint kMgRank = 64;
constant constexpr uint kMgTileRows = 32;
constant constexpr uint kMgTileSg = 8;
constant constexpr uint kMgTileThreads = 256;
constant constexpr uint kMgTileStep = 64;
constant constexpr uint kMgTileStride = 72;
constant constexpr uint kMgFrags = 4;

struct MgEight {
  float4 low, high;
  float operator[](uint i) const { return i < 4 ? low[i] : high[i - 4]; }
};
inline MgEight mg_eight_bf16(const device bfloat16_t* row, uint k) {
  const device bfloat4* q = reinterpret_cast<const device bfloat4*>(row + k);
  return {float4(q[0]), float4(q[1])};
}
inline MgEight mg_eight_f32(const device float* row, uint k) {
  const device float4* q = reinterpret_cast<const device float4*>(row + k);
  return {q[0], q[1]};
}
struct MgShared {
  float scales[2 * kMgTileRows];
  union {
    bfloat16_t stages[2][kMgTileRows * kMgTileStride];
    struct {
      bfloat16_t lows[kMgTileRows * kMgTileStride];
      float4 partials[kMgTileSg * 32];
    } gated;
  };
  float inverses[kMgTileRows];
};
struct MgRun { float4 low, high; };
struct MgChains { float2 even0, odd0, even1, odd1, even2, odd2, even3, odd3; };

inline const device float* mg_row(const device float* x, const device float* msg, uint x_stride, uint x_col, uint m_col,
                                  uint first, uint count, uint r, bool own) {
  const ulong row = first + min(r, count - 1);
  return own ? (x + row * x_stride + x_col) : (msg + row * (4u * kMgHidden) + m_col);
}

inline void mg_read(threadgroup const bfloat16_t* row, thread MgEight& low, thread MgEight& high) {
  threadgroup const bfloat4* a = reinterpret_cast<threadgroup const bfloat4*>(row);
  low = {float4(a[0]), float4(a[1])};
  high = {float4(a[2]), float4(a[3])};
}

inline void mg_mma(threadgroup const bfloat16_t* row, MgEight b0, MgEight b1, thread float2& even, thread float2& odd) {
  MgEight low, high;
  mg_read(row, low, high);
  for (uint j = 0; j < 8; j += 2) {
    mg_sg::mma_acc(even, float2(low[j], high[j]), float2(b0[j], b1[j]));
    mg_sg::mma_acc(odd, float2(low[j + 1], high[j + 1]), float2(b0[j + 1], b1[j + 1]));
  }
}

METAL_FUNC void mg_tile_run(
    const device float* x, const device float* msg,
    const device bfloat16_t* down, const device float* forget, const device float* gain,
    const device bfloat16_t* ln_w, device float* gated, device bfloat16_t* out,
    uint x_stride, uint x_col, uint m_col, uint first, uint count, float eps,
    uint tid, uint lane, uint sg, threadgroup MgShared& shared) {
  // Phase 1: row & message inv-RMS
  const uint quads = kMgHidden / 4;
  for (uint i = sg; i < 2 * kMgTileRows; i += kMgTileSg) {
    const device float4* row = reinterpret_cast<const device float4*>(
        mg_row(x, msg, x_stride, x_col, m_col, first, count, i / 2, (i & 1) == 0));
    float squares = 0.0f;
    if (i / 2 < count) {
      for (uint d = lane; d < quads; d += 32) {
        const float4 v = row[d];
        squares += v.x * v.x;
        squares += v.y * v.y;
        squares += v.z * v.z;
        squares += v.w * v.w;
      }
    }
    squares = metal::simd_sum(squares);
    if (lane == 0) shared.scales[i] = (i / 2 < count) ? metal::precise::rsqrt(squares / kMgHidden + 1e-6f) : 0.0f;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // Phase 2: down_proj (4608 -> 64) via double-buffered 32x64 stages
  const mg_sg::Lane l = mg_sg::lane_map(lane);
  const uint steps = kMgHidden / kMgTileStep;  // 36
  const uint column = 8 * sg + l.fn;
  const device bfloat16_t* w0 = down + ulong(column) * (2 * kMgHidden);
  const device bfloat16_t* w1 = down + ulong(column + 1) * (2 * kMgHidden);

  auto load_run = [&](uint step) -> MgRun {
    const bool own = step < steps;
    const device float4* run = reinterpret_cast<const device float4*>(
        mg_row(x, msg, x_stride, x_col, m_col, first, count, tid / 8, own)
        + (own ? step : step - steps) * kMgTileStep + 8 * (tid & 7));
    return {run[0], run[1]};
  };
  auto stage_run = [&](MgRun run, bool own, threadgroup bfloat16_t* st) {
    const uint r = tid / 8;
    const float sc = shared.scales[2 * r + (own ? 0u : 1u)];
    threadgroup bfloat4* dst = reinterpret_cast<threadgroup bfloat4*>(st + r * kMgTileStride + 8 * (tid & 7));
    dst[0] = bfloat4(run.low * sc);
    dst[1] = bfloat4(run.high * sc);
  };

  MgChains own{}, other{};
  stage_run(load_run(0), true, shared.stages[0]);
  MgEight next0 = mg_eight_bf16(w0, 8 * l.fm);
  MgEight next1 = mg_eight_bf16(w1, 8 * l.fm);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  const auto run_step = [&](uint step, thread MgChains& chains) __attribute__((always_inline)) {
    const bool has_next = (step + 1) < (2 * steps);
    const MgRun ahead = has_next ? load_run(step + 1) : MgRun{};
    const MgEight b0 = next0, b1 = next1;
    if (has_next) {
      next0 = mg_eight_bf16(w0, (step + 1) * kMgTileStep + 8 * l.fm);
      next1 = mg_eight_bf16(w1, (step + 1) * kMgTileStep + 8 * l.fm);
    }
    threadgroup const bfloat16_t* row = shared.stages[step & 1] + l.fm * kMgTileStride + 8 * l.fn;
    mg_mma(row, b0, b1, chains.even0, chains.odd0);
    mg_mma(row + 8 * kMgTileStride, b0, b1, chains.even1, chains.odd1);
    mg_mma(row + 16 * kMgTileStride, b0, b1, chains.even2, chains.odd2);
    mg_mma(row + 24 * kMgTileStride, b0, b1, chains.even3, chains.odd3);
    if (has_next) stage_run(ahead, (step + 1) < steps, shared.stages[(step + 1) & 1]);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  };
  for (uint step = 0; step < steps; ++step) run_step(step, own);
  for (uint step = steps; step < 2 * steps; ++step) run_step(step, other);

  auto store_low = [&](uint f, float2 sum) {
    const uint r = 8 * f + l.fm;
    shared.gated.lows[r * kMgTileStride + column] = bfloat16_t(sum.x);
    shared.gated.lows[r * kMgTileStride + column + 1] = bfloat16_t(sum.y);
  };
  store_low(0, (own.even0 + own.odd0) + (other.even0 + other.odd0));
  store_low(1, (own.even1 + own.odd1) + (other.even1 + other.odd1));
  store_low(2, (own.even2 + own.odd2) + (other.even2 + other.odd2));
  store_low(3, (own.even3 + own.odd3) + (other.even3 + other.odd3));
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // Phase 3: forget + gain projections + gated combine
  const uint blocks = kMgHidden / kMgTileStep;  // 36
  const uint k8 = 8 * l.fm;
  float squares[kMgFrags] = {0.0f, 0.0f, 0.0f, 0.0f};
  auto load_fg = [&](uint block, thread MgEight (&w)[4]) {
    const uint col = block * kMgTileStep + 8 * sg + l.fn;
    w[0] = mg_eight_f32(forget + ulong(col) * kMgRank, k8);
    w[1] = mg_eight_f32(forget + ulong(col + 1) * kMgRank, k8);
    w[2] = mg_eight_f32(gain + ulong(col) * kMgRank, k8);
    w[3] = mg_eight_f32(gain + ulong(col + 1) * kMgRank, k8);
  };
  MgEight next_fg[4];
  load_fg(0, next_fg);
  for (uint block = 0; block < blocks; ++block) {
    const uint col = block * kMgTileStep + 8 * sg + l.fn;
    const MgEight f0 = next_fg[0], f1 = next_fg[1], e0 = next_fg[2], e1 = next_fg[3];
    if (block + 1 < blocks) load_fg(block + 1, next_fg);
    float2 xs[kMgFrags], ms[kMgFrags];
    for (uint f = 0; f < kMgFrags; ++f) {
      const ulong row = first + min(8 * f + l.fm, count - 1);
      xs[f] = *reinterpret_cast<const device float2*>(x + row * x_stride + x_col + col);
      ms[f] = *reinterpret_cast<const device float2*>(msg + row * (4u * kMgHidden) + m_col + col);
    }
    float2 forgets[kMgFrags], gains[kMgFrags];
    for (uint f = 0; f < kMgFrags; ++f) {
      MgEight low, high;
      mg_read(shared.gated.lows + (8 * f + l.fm) * kMgTileStride + 8 * l.fn, low, high);
      forgets[f] = float2(0.0f);
      gains[f] = float2(0.0f);
      for (uint j = 0; j < 8; ++j) {
        mg_sg::mma_acc(forgets[f], float2(low[j], high[j]), float2(f0[j], f1[j]));
        mg_sg::mma_acc(gains[f], float2(low[j], high[j]), float2(e0[j], e1[j]));
      }
    }
    for (uint f = 0; f < kMgFrags; ++f) {
      const uint r = 8 * f + l.fm;
      if (r < count) {
        const ulong row = first + r;
        const float2 xv = xs[f], mv = ms[f];
        float2 v;
        v.x = (1.0f - metal::precise::tanh(forgets[f].x)) * xv.x + metal::precise::tanh(gains[f].x) * mv.x;
        v.y = (1.0f - metal::precise::tanh(forgets[f].y)) * xv.y + metal::precise::tanh(gains[f].y) * mv.y;
        *reinterpret_cast<device float2*>(gated + row * kMgHidden + col) = v;
        squares[f] += v.x * v.x;
        squares[f] += v.y * v.y;
      }
    }
  }
  shared.gated.partials[sg * 32 + lane] = float4(squares[0], squares[1], squares[2], squares[3]);

  // Phase 4: RMSNorm into bf16
  threadgroup_barrier(metal::mem_flags::mem_device | metal::mem_flags::mem_threadgroup);
  if (tid < count) {
    const uint f = tid / 8, fm = tid & 7;
    float total = 0.0f;
    for (uint s = 0; s < kMgTileSg; ++s) {
      for (uint ln = 0; ln < 32; ++ln) {
        if (mg_sg::lane_map(ln).fm == fm) total += shared.gated.partials[s * 32 + ln][f];
      }
    }
    shared.inverses[tid] = metal::precise::rsqrt(total / kMgHidden + eps);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const device float4* rows = reinterpret_cast<const device float4*>(gated + ulong(first) * kMgHidden);
  device bfloat4* dst = reinterpret_cast<device bfloat4*>(out + ulong(first) * kMgHidden);
  for (uint q = tid; q < count * quads; q += kMgTileThreads) {
    const uint r = q / quads, e = 4 * (q % quads);
    const bfloat4 wbf = *reinterpret_cast<const device bfloat4*>(ln_w + e);
    const float4 sc = rows[q] * shared.inverses[r];
    dst[q] = bfloat4(
        bfloat16_t(wbf.x) * bfloat16_t(sc.x),
        bfloat16_t(wbf.y) * bfloat16_t(sc.y),
        bfloat16_t(wbf.z) * bfloat16_t(sc.z),
        bfloat16_t(wbf.w) * bfloat16_t(sc.w));
  }
}
"""
)

_SPLASH_SINGLE_SRC = r"""
    threadgroup MgShared shared;
    const uint first = threadgroup_position_in_grid.x * kMgTileRows;
    if (first >= NN) return;
    const uint count = min(kMgTileRows, NN - first);
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    mg_tile_run(
        h, msg, dn, wf, wg, nw, gated, out,
        kMgHidden, 0u, SLOT * kMgHidden, first, count, eps[0],
        tid, lane, sg, shared);
"""

_SPLASH_QKV_SRC = r"""
    threadgroup MgShared shared;
    const uint first = threadgroup_position_in_grid.x * kMgTileRows;
    const uint g = threadgroup_position_in_grid.y;
    if (first >= NN || g >= 3) return;
    const uint count = min(kMgTileRows, NN - first);
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const device bfloat16_t* dn = (g == 0 ? dn0 : (g == 1 ? dn1 : dn2));
    const device float* wf = (g == 0 ? wf0 : (g == 1 ? wf1 : wf2));
    const device float* wg = (g == 0 ? wg0 : (g == 1 ? wg1 : wg2));
    const device bfloat16_t* nw = SHARED_LN ? nw0 : (g == 0 ? nw0 : (g == 1 ? nw1 : nw2));
    device bfloat16_t* dst = (g == 0 ? o0 : (g == 1 ? o1 : o2));
    device float* gt = gated + ulong(g) * NN * kMgHidden;
    mg_tile_run(
        h, msg, dn, wf, wg, nw, gt, dst,
        SHARED_H ? kMgHidden : 3u * kMgHidden,
        SHARED_H ? 0u : g * kMgHidden,
        g * kMgHidden,
        first, count, eps[0],
        tid, lane, sg, shared);
"""


def _k_splash(name, ins, outs, src):
    k = _kernels.get(name)
    if k is None:
        k = _kernels[name] = mx.fast.metal_kernel(
            name=name,
            input_names=list(ins),
            output_names=list(outs),
            source=src,
            header=_SPLASH_MG_HEADER,
            ensure_row_contiguous=True,
        )
    return k


def _splash_ok(gates, N: int) -> bool:
    return (
        _SPLASH_MSGGATE
        and len(gates) == 3
        and N >= 32
        and all(g.down_proj.weight.shape == (64, 4608) for g in gates)
    )


def _gate_logits(gate_mod, x_bf16):
    st = gate_mod.down_proj(x_bf16).astype(mx.float32)
    return gate_mod.forget_gate_proj(st), gate_mod.message_gate_proj(st)


def msg_gate_norm_single(gate_mod, h: mx.array, msg: mx.array, slot: int, ln) -> mx.array:
    """Single MessageGate + downstream RMSNorm -> (B, L, D) bf16."""
    B, L, D = h.shape
    N = B * L
    if _splash_ok((gate_mod,), N):
        nt = (N + 31) // 32
        k = _k_splash(
            "pf_splash_mg_s",
            ("h", "msg", "dn", "wf", "wg", "nw", "eps"),
            ("out", "gated"),
            _SPLASH_SINGLE_SRC,
        )
        out, _ = k(
            inputs=[
                h, msg, gate_mod.down_proj.weight, gate_mod.forget_gate_proj.weight,
                gate_mod.message_gate_proj.weight, ln.weight, mx.array([ln.eps], mx.float32),
            ],
            template=[("NN", N), ("SLOT", slot)],
            grid=(nt * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D), (N, D)],
            output_dtypes=[mx.bfloat16, mx.float32],
        )
        return out
    k_in = _k("pf_mgin_s", ("h", "msg"), ("out",), _MGIN_SINGLE_SRC)
    g_in = k_in(
        inputs=[h, msg],
        template=[("DD", D), ("TG", _TG_RMS), ("MSTRIDE", 4), ("SLOT", slot)],
        grid=(N * _TG_RMS, 1, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, 2 * D)],
        output_dtypes=[mx.bfloat16],
    )[0]
    fl, ml = _gate_logits(gate_mod, g_in)
    k_mn = _k("pf_mn_s", ("fl", "ml", "h", "msg", "nw", "eps"), ("out",), _MIXNORM_SINGLE_SRC)
    return k_mn(
        inputs=[fl, ml, h, msg, ln.weight, mx.array([ln.eps], mx.float32)],
        template=[("DD", D), ("TG", _TG_RMS), ("SLOT", slot)],
        grid=(N * _TG_RMS, 1, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, D)],
        output_dtypes=[mx.bfloat16],
    )[0]


def msg_gate_norm_qkv(gates, h_or_ai: mx.array, msg: mx.array, lns, *, shared_h: bool, shared_ln: bool):
    """Batched Q/K/V MessageGate + input RMSNorm -> (q_in, k_in, v_in) bf16 in 2 fused launches + projections."""
    B, L = msg.shape[0], msg.shape[1]
    D = msg.shape[-1]
    N = B * L
    if _splash_ok(gates, N):
        nt = (N + 31) // 32
        k = _k_splash(
            "pf_splash_mg_qkv",
            ("h", "msg", "dn0", "dn1", "dn2", "wf0", "wf1", "wf2", "wg0", "wg1", "wg2", "nw0", "nw1", "nw2", "eps"),
            ("o0", "o1", "o2", "gated"),
            _SPLASH_QKV_SRC,
        )
        o0, o1, o2, _ = k(
            inputs=[
                h_or_ai, msg,
                gates[0].down_proj.weight, gates[1].down_proj.weight, gates[2].down_proj.weight,
                gates[0].forget_gate_proj.weight, gates[1].forget_gate_proj.weight, gates[2].forget_gate_proj.weight,
                gates[0].message_gate_proj.weight, gates[1].message_gate_proj.weight, gates[2].message_gate_proj.weight,
                lns[0].weight, lns[1].weight, lns[2].weight,
                mx.array([lns[0].eps], mx.float32),
            ],
            template=[("NN", N), ("SHARED_H", int(shared_h)), ("SHARED_LN", int(shared_ln))],
            grid=(nt * 256, 3, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D), (B, L, D), (B, L, D), (3, N, D)],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32],
        )
        return (o0, o1, o2)
    k_in = _k("pf_mgin_qkv", ("h", "msg"), ("o0", "o1", "o2"), _MGIN_QKV_SRC)
    in0, in1, in2 = k_in(
        inputs=[h_or_ai, msg],
        template=[("DD", D), ("TG", _TG_RMS), ("SHARED_H", int(shared_h))],
        grid=(N * _TG_RMS, 3, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, 2 * D)] * 3,
        output_dtypes=[mx.bfloat16] * 3,
    )
    fl0, ml0 = _gate_logits(gates[0], in0)
    fl1, ml1 = _gate_logits(gates[1], in1)
    fl2, ml2 = _gate_logits(gates[2], in2)
    k_mn = _k(
        "pf_mn_qkv",
        ("fl0", "ml0", "fl1", "ml1", "fl2", "ml2", "h", "msg", "nw0", "nw1", "nw2", "eps"),
        ("o0", "o1", "o2"),
        _MIXNORM_QKV_SRC,
    )
    return k_mn(
        inputs=[
            fl0, ml0, fl1, ml1, fl2, ml2,
            h_or_ai, msg,
            lns[0].weight, lns[1].weight, lns[2].weight,
            mx.array([lns[0].eps], mx.float32),
        ],
        template=[("DD", D), ("TG", _TG_RMS), ("SHARED_H", int(shared_h)), ("SHARED_LN", int(shared_ln))],
        grid=(N * _TG_RMS, 3, 1),
        threadgroup=(_TG_RMS, 1, 1),
        output_shapes=[(B, L, D)] * 3,
        output_dtypes=[mx.bfloat16] * 3,
    )
