"""Staged B=1 reference of one fast-decode layer visit (fastdecode s1 -> _core -> s2) that exposes every tensor at the
fused-kernel stage boundaries of docs/FUSED.md.

Same ops in the same order as fastdecode.make_s1 / make_s2 (uncompiled; compile only fuses element-wise runs and does
not change bits).  bench/fused_capture.py --check verifies on captured real visits that the outputs of visit_ref equal the
compiled fast path bitwise.  Every fused stage kernel is tested row by row against these tensors.
"""
import mlx.core as mx

from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import layers as L

f32 = mx.float32


def _gate(g, s, m, T, p):
    """MessageGate.__call__ split at its stage boundaries: gate_in (bf16, 2D) -> gate_down (bf16, rank) -> mixed (f32)."""
    gin = mx.concatenate([L.unweighted_rms_norm(s), L.unweighted_rms_norm(m)], axis=-1)
    gin = gin.astype(g.down_proj.weight.dtype)
    d = g.down_proj(gin)
    st = d.astype(f32)
    fg = mx.tanh(g.forget_gate_proj(st))
    mg = mx.tanh(g.message_gate_proj(st))
    out = (1 - fg) * s + mg * m
    T[p + "gate_in"], T[p + "gate_down"], T[p + "forget"], T[p + "message"], T[p + "mixed"] = gin, d, fg, mg, out
    return out


def visit_ref(layer, h, bs, msg, cs, attn_out, tau=None):
    """One layer visit at B=1, L=1.

    h (1,1,D) bf16; bs (n,1,1,D) f32 or None; msg (1,1,4,D) f32 or None; cs = 4 conv states (1,3,C) f32;
    attn_out = output of fastdecode._core for this visit (1,1,H,hd) bf16 (attention itself is not a fused stage here);
    tau = fastdecode._full_tau(layer, end) for full-attention layers.  Returns dict name -> array (see docs/FUSED.md).
    """
    a = layer.self_attn
    dtype = h.dtype
    T = {}
    sliding = isinstance(layer, L.SlidingDecoderLayer)

    # ---- stage A (pre): row-local
    nr = L.unweighted_rms_norm(h.astype(f32))
    T["normed_residual"] = nr
    ai = layer.attention_input(bs, nr) if layer.attention_input is not None else None
    T["attn_input"] = ai
    gates = (getattr(layer, "q_message_gate", None), getattr(layer, "k_message_gate", None),
             getattr(layer, "v_message_gate", None))
    if sliding:
        hs = ai[..., 0, :] if ai is not None else h
        if msg is None:
            q_in = k_in = v_in = layer.input_layernorm(hs).astype(dtype)
        else:
            st = [_gate(g, hs, msg[..., i, :], T, "qkv"[i] + "_") for i, g in enumerate(gates)]
            q_in, k_in, v_in = [layer.input_layernorm(s).astype(dtype) for s in st]
    else:
        if ai is None:
            qs = ks = vs = h
        else:
            qs, ks, vs = (ai[..., i, :] for i in range(3))
        if msg is not None:
            qs, ks, vs = [_gate(g, s, msg[..., i, :], T, "qkv"[i] + "_") for i, (g, s) in enumerate(zip(gates, (qs, ks, vs)))]
        q_in = layer.q_input_layernorm(qs).astype(dtype)
        k_in = layer.k_input_layernorm(ks).astype(dtype)
        v_in = layer.v_input_layernorm(vs).astype(dtype)
    T.update(q_in=q_in, k_in=k_in, v_in=v_in)

    # ---- stage Q (projections, weights: 8-bit affine gs64)
    q = a.q_proj(q_in)
    r = a.r_proj(q_in)
    gate = a.gate_proj(q_in)
    sinks = a.sink_proj(q_in)
    kr = a.k_proj(k_in)
    vr = a.v_proj(v_in)
    T.update(q_proj=q, r_proj=r, gate=gate, sinks=sinks, k_proj=kr, v_proj=vr)
    if not sliding:
        T["k_wproj"] = a.k_sconv.weight_proj(k_in)
        T["v_wproj"] = a.v_sconv.weight_proj(v_in)

    # ---- stage M (mid: convs, clips, value rms, tau, relative logits)
    if sliding:
        key, w0 = FD._sc(a.k_sconv, kr, cs[0])
        value, w1 = FD._sc(a.v_sconv, vr, cs[1])
    else:
        key, w0 = FD._dsc(a.k_sconv, kr, k_in, cs[0])
        value, w1 = FD._dsc(a.v_sconv, vr, v_in, cs[1])
    T.update(k_conv=key, v_conv=value, w0=w0, w1=w1)
    B, Lq = 1, 1
    qh = a.q_l2_clip(q.reshape(B, Lq, a.num_heads, -1))
    kh = a.k_l2_clip(key.reshape(B, Lq, a.num_key_value_heads, -1))
    vh = value.reshape(B, Lq, a.num_key_value_heads, -1)
    vh = L.unweighted_rms_norm(vh.astype(f32), a.eps).astype(vh.dtype)
    rh = a.rel_l2_clip(r.reshape(B, Lq, a.num_heads, -1))
    if tau is not None:
        qh = (qh.astype(f32) * tau).astype(qh.dtype)
        rh = (rh.astype(f32) * tau).astype(rh.dtype)
    T.update(q=qh, k=kh, v=vh, rel=rh)
    pb = a.rel_logits_proj(rh)
    T["position_bias"] = pb  # (1,1,H,R) before the transpose

    # ---- attention (not fused here): fastdecode._core(layer, q^T, k^T, v^T, pb^T, sinks, kv) -> attn_out
    T["attn_out"] = attn_out

    # ---- stage O (gate * attn -> o_proj -> post-attention conv -> residual)
    attn = a._attend_post(attn_out, gate)
    T["o_proj"] = attn
    hidden, w2 = FD._sc(layer.post_attention_convolution, attn, cs[2])
    T["post_attn_conv"], T["w2"] = hidden, w2
    residual = hidden = hidden.astype(f32)
    if layer.is_block_boundary:
        block = nr[None]
        bs = block if bs is None else mx.concatenate([bs, block])
    else:
        residual = hidden = h.astype(f32) + hidden
    T["residual"], T["bs_new"] = residual, bs

    # ---- stage P (rms, mlp_input depth attention, [mlp message gate], post_attention_layernorm)
    normed = L.unweighted_rms_norm(hidden)
    T["normed"] = normed
    hidden = layer.mlp_input(bs, normed)[..., 0, :]
    T["mlp_src"] = hidden
    if msg is not None:
        hidden = _gate(layer.mlp_message_gate, hidden, msg[..., 3, :], T, "mlp_")
    x = layer.post_attention_layernorm(hidden).astype(dtype)
    T["mlp_x"] = x

    # ---- stage E (router, routing, experts, combine, post-mlp conv, output)
    moe = layer.mlp
    T["router_logits"] = x.astype(f32) @ moe.gate.weight.T
    inds, wts = moe.gate(x)
    T["expert_inds"], T["expert_w"] = inds, wts
    ex = moe.experts
    xe = mx.expand_dims(x, (-2, -3))
    idx = mx.stop_gradient(inds)
    assert idx.size < 64  # B=1: no sort
    gu = ex.gate_up_proj(xe, idx, sorted_indices=False)
    T["gate_up"] = gu
    act = ex.act_fn(gu[..., 0::2], gu[..., 1::2])
    T["swiglu"] = act
    dn = ex.down_proj(act, idx, sorted_indices=False).squeeze(-2)
    T["down"] = dn  # (1,1,8,D) bf16
    comb = (dn.astype(f32) * wts[..., None]).sum(axis=-2).astype(x.dtype)
    T["moe_out"] = comb
    hidden, w3 = FD._sc(layer.post_mlp_convolution, comb, cs[3])
    T["post_mlp_conv"], T["w3"] = hidden, w3
    T["out"] = (residual + hidden.astype(f32)).astype(dtype)
    return T
