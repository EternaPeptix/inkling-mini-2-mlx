"""Fused bit-exact Metal kernels for L > 1 prefill (LOOPLING_FAST_PREFILL=1 default)."""
import math
import os

import mlx.core as mx
import numpy as np

from mlx_lm.models.inkling2.rows.glue import GLUE_HEADER

_FAST = os.environ.get("LOOPLING_FAST_PREFILL", "1") not in ("0", "")
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


def _gate_logits(gate_mod, x_bf16):
    st = gate_mod.down_proj(x_bf16).astype(mx.float32)
    return gate_mod.forget_gate_proj(st), gate_mod.message_gate_proj(st)


def msg_gate_norm_single(gate_mod, h: mx.array, msg: mx.array, slot: int, ln) -> mx.array:
    """Single MessageGate + downstream RMSNorm -> (B, L, D) bf16."""
    B, L, D = h.shape
    N = B * L
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
