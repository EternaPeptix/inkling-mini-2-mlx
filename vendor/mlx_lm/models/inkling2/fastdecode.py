"""Lane A: low-overhead decode path (B=1, L=1) for Inkling2 decoder layers. Flag: LOOPLING_COMPILE_DECODE.

  unset/0  off (reference path)
  2        same segmentation + hoisted attention masks, but no mx.compile (isolates the compile effect)
  1        segments compiled with mx.compile (fixed shapes, one trace per distinct input shape)

The math is the reference math: the layer is split in S1 (norms, attention-input mix, projections, short convs, l2 clips),
the cache update + attention core (shapes vary with the cache, never compiled), and S2 (output projection, MLP/MoE,
convs, residual). Elementwise chains are fused by mx.compile; everything else is the same primitive as the reference.
"""
import os

import mlx.core as mx

_MODE = os.environ.get("LOOPLING_COMPILE_DECODE", "0")
_COMPILE = _MODE == "1"
_FUSE = os.environ.get("LOOPLING_FUSE_QKV", "0") not in ("0", "")
_FCONV = os.environ.get("LOOPLING_FUSED_CONV", "0") not in ("0", "")
_ctx: dict = {}
_tau: dict = {}


def eligible(h, cache, conv_mask) -> bool:
    if cache is None or conv_mask is not None or h.shape[0] != 1 or h.shape[1] != 1:
        return False
    conv = cache[1]
    if conv.lengths is not None or conv.left_padding is not None:
        return False
    if any(c is None for c in conv.cache):
        return False
    return cache[0].keys is not None


def _fn(f):
    return mx.compile(f) if _COMPILE else f


_CONV_SRC = """
    uint c = thread_position_in_grid.x;
    if (c >= C) return;
    float x0 = float(x[c]);
    float s0 = st[c], s1 = st[C + c], s2 = st[2 * C + c];
    float acc = 0.0f;
    acc = fma(float(w[c * 4 + 0]), s0, acc);
    acc = fma(float(w[c * 4 + 1]), s1, acc);
    acc = fma(float(w[c * 4 + 2]), s2, acc);
    acc = fma(float(w[c * 4 + 3]), x0, acc);
    float conv = float(T(acc));
    out[c] = T((conv + x0) / SCALE);
    nst[c] = s1;
    nst[C + c] = s2;
    nst[2 * C + c] = x0;
"""
_conv_kernels: dict = {}


def short_conv_decode(mod, x, state):
    """ShortConvolution for one decode token: (out, new_state), bit-identical to the reference (checked by bench/convtest_a.py)."""
    import math

    import numpy as np

    C = x.shape[-1]
    K = mod.kernel_size
    assert K == 4
    scale = repr(float(np.float32(math.sqrt(K + 1)))) + "f"
    k = _conv_kernels.get(scale)
    if k is None:
        k = _conv_kernels[scale] = mx.fast.metal_kernel(
            name="short_conv_decode", input_names=["x", "st", "w"], output_names=["out", "nst"],
            source=_CONV_SRC.replace("SCALE", scale),
        )
    out, nst = k(
        inputs=[x, state, mod.conv1d.weight], template=[("T", x.dtype), ("C", C)], grid=(C, 1, 1),
        threadgroup=(min(256, C), 1, 1), output_shapes=[x.shape, state.shape], output_dtypes=[x.dtype, mx.float32],
    )
    return out, nst


def _conv_ok(mod, x):
    return _FCONV and x.dtype == mx.bfloat16 and mod.conv1d.weight.dtype == mx.bfloat16


def _sc(mod, x, st):
    """ShortConvolution step returning (out, new 3-row state)."""
    if _conv_ok(mod, x):
        return short_conv_decode(mod, x, st)
    o, w = mod(x, st, None)
    return o, w[:, w.shape[1] - 3 :, :]


def _dsc(mod, x, dyn, st):
    o, w = mod(x, dyn, st, None)
    return o, w[:, w.shape[1] - 3 :, :]


class _Fused:
    """Several quantized projections of one input as a single quantized matmul (rows are independent, so the result
    is bit-identical to the separate matmuls)."""

    def __init__(self, mods):
        from mlx import nn

        m0 = mods[0]
        assert all(isinstance(m, nn.QuantizedLinear) for m in mods)
        assert all((m.group_size, m.bits, m.mode) == (m0.group_size, m0.bits, m0.mode) for m in mods)
        assert all(("biases" in m) == ("biases" in m0) for m in mods)
        self.w = mx.concatenate([m["weight"] for m in mods], axis=0)
        self.s = mx.concatenate([m["scales"] for m in mods], axis=0)
        self.b = mx.concatenate([m["biases"] for m in mods], axis=0) if "biases" in m0 else None
        self.cfg = (m0.group_size, m0.bits, m0.mode)
        self.splits = []
        o = 0
        for m in mods[:-1]:
            o += m["weight"].shape[0]
            self.splits.append(o)
        mx.eval([x for x in (self.w, self.s, self.b) if x is not None])

    def __call__(self, x):
        g, b, mode = self.cfg
        y = mx.quantized_matmul(x, self.w, scales=self.s, biases=self.b, transpose=True, group_size=g, bits=b, mode=mode)
        return mx.split(y, self.splits, axis=-1)


def _build(layer):
    from mlx_lm.models.inkling2 import layers as L

    f32 = mx.float32
    sliding = isinstance(layer, L.SlidingDecoderLayer)
    a = layer.self_attn
    out = {}
    fq = fqkv = fk = fv = None
    if _FUSE:
        fq = _Fused([a.q_proj, a.r_proj, a.gate_proj, a.sink_proj])
        if sliding:
            fqkv = _Fused([a.q_proj, a.r_proj, a.gate_proj, a.sink_proj, a.k_proj, a.v_proj])

    def make_s1(has_bs, has_msg):
        def s1(h, *rest):
            rest = list(rest)
            bs = rest.pop(0) if has_bs else None
            msg = rest.pop(0) if has_msg else None
            tau = rest.pop(0) if (not sliding) else None
            cs0, cs1 = rest
            dtype = h.dtype
            normed_residual = L.unweighted_rms_norm(h.astype(f32))
            ai = layer.attention_input(bs, normed_residual) if layer.attention_input is not None else None
            if sliding:
                hs = ai[..., 0, :] if ai is not None else h
                if msg is None:
                    q_in = k_in = v_in = layer.input_layernorm(hs).astype(dtype)
                else:
                    st = layer._mix_messages((hs,) * 3, msg)
                    q_in, k_in, v_in = [layer.input_layernorm(s).astype(dtype) for s in st]
                if fq is None:
                    key, w0 = _sc(a.k_sconv, a.k_proj(k_in), cs0)
                    value, w1 = _sc(a.v_sconv, a.v_proj(v_in), cs1)
                    pre = a._attend_pre(q_in, a.q_proj(q_in), key, value, a.r_proj(q_in), None)
                else:
                    if msg is None:
                        qp, rp, gp, sp, kp, vp = fqkv(q_in)
                    else:
                        qp, rp, gp, sp = fq(q_in)
                        kp, vp = a.k_proj(k_in), a.v_proj(v_in)
                    key, w0 = _sc(a.k_sconv, kp, cs0)
                    value, w1 = _sc(a.v_sconv, vp, cs1)
                    pre = a._attend_pre(q_in, qp, key, value, rp, None, gp, sp)
            else:
                if ai is None:
                    qs = ks = vs = h
                else:
                    qs, ks, vs = (ai[..., i, :] for i in range(3))
                if msg is not None:
                    qs, ks, vs = layer._mix_messages((qs, ks, vs), msg)
                q_in = layer.q_input_layernorm(qs).astype(dtype)
                k_in = layer.k_input_layernorm(ks).astype(dtype)
                v_in = layer.v_input_layernorm(vs).astype(dtype)
                key, w0 = _dsc(a.k_sconv, a.k_proj(k_in), k_in, cs0)
                value, w1 = _dsc(a.v_sconv, a.v_proj(v_in), v_in, cs1)
                if fq is None:
                    pre = a._attend_pre(q_in, a.q_proj(q_in), key, value, a.r_proj(q_in), tau)
                else:
                    qp, rp, gp, sp = fq(q_in)
                    pre = a._attend_pre(q_in, qp, key, value, rp, tau, gp, sp)
            return (*pre, w0, w1, normed_residual)

        return _fn(s1)

    def make_s2(has_bs, has_msg):
        def s2(h, normed_residual, attn_out, gate, w0, w1, cs2, cs3, *rest):
            rest = list(rest)
            bs = rest.pop(0) if has_bs else None
            msg = rest.pop(0) if has_msg else None
            dtype = h.dtype
            attn = a._attend_post(attn_out, gate)
            hidden, w2 = _sc(layer.post_attention_convolution, attn, cs2)
            residual = hidden = hidden.astype(f32)
            if layer.is_block_boundary:
                block = normed_residual[None]
                bs = block if bs is None else mx.concatenate([bs, block])
            else:
                residual = hidden = h.astype(f32) + hidden
            normed = L.unweighted_rms_norm(hidden)
            hidden = layer.mlp_input(bs, normed)[..., 0, :]
            if msg is not None:
                hidden = layer.mlp_message_gate(hidden, msg[..., 3, :])
            hidden = layer.mlp(layer.post_attention_layernorm(hidden).astype(dtype))
            hidden, w3 = _sc(layer.post_mlp_convolution, hidden, cs3)
            states = [w0, w1, w2, w3]
            o = (residual + hidden.astype(f32)).astype(dtype)
            return (o, bs, *states)

        return _fn(s2)

    out["s1"] = {}
    out["s2"] = {}
    out["make_s1"], out["make_s2"] = make_s1, make_s2
    return out


def _full_tau(layer, offset):
    t = _tau.get(offset)
    if t is None:
        if len(_tau) > 8:
            _tau.clear()
        a = layer.self_attn
        pos = mx.arange(1) + offset
        pos = pos.astype(mx.float32)
        t = 1.0 + a.log_scaling_alpha * mx.log(mx.maximum((pos + 1) / a.log_scaling_n_floor, 1.0))
        t = t[..., None, None]
        _tau[offset] = t
    return t


def _core(layer, q, k, v, pb, sinks, kv):
    from mlx_lm.models.inkling2 import layers as L

    a = layer.self_attn
    k_all, v_all = kv.update_and_fetch(k, v)
    S = k_all.shape[2]
    end = kv.offset
    R = pb.shape[-1]
    idx = getattr(kv, "_idx", None) if isinstance(kv, L.RotatingKVCache) else None
    key = (S, end, idx, R, a.sliding_window)
    c = _ctx.get(key)
    if c is None:
        if len(_ctx) > 16:
            _ctx.clear()
        qp = L._positions(end - 1, 1)
        kp = L._key_positions(kv, S, 1, end)
        distance = qp.reshape(1) - kp.reshape(S)
        clip = mx.clip(distance, 0, R - 1)
        visible = (distance >= 0) & (kp.reshape(S) >= 0)
        if a.sliding_window:
            visible = visible & (distance < a.sliding_window)
        use_bias = visible & (distance < R)
        fill = mx.where(visible, 0.0, -mx.inf)
        c = (clip, use_bias, fill)
        _ctx[key] = c
    clip, use_bias, fill = c
    bias = mx.take(pb[0, 0], clip, axis=-1)
    mask = mx.where(use_bias, bias, fill.astype(bias.dtype)).astype(q.dtype)
    o = mx.fast.scaled_dot_product_attention(
        q, k_all, v_all, scale=a.scaling, mask=mask[None, :, None, :], sinks=sinks[0, 0].astype(q.dtype)
    )
    return o.transpose(0, 2, 1, 3)


def decode_layer(layer, h, bs, cache, messages):
    fd = getattr(layer, "_fd", None)
    if fd is None:
        fd = layer._fd = _build(layer)
    from mlx_lm.models.inkling2 import layers as L

    sliding = isinstance(layer, L.SlidingDecoderLayer)
    conv = cache[1]
    kv = cache[0]
    cs = list(conv.cache)
    has_bs, has_msg = bs is not None, messages is not None
    k1 = (has_bs, has_msg)
    s1 = fd["s1"].get(k1)
    if s1 is None:
        s1 = fd["s1"][k1] = fd["make_s1"](*k1)
        fd["s2"][k1] = fd["make_s2"](*k1)
    s2 = fd["s2"][k1]
    args = [h]
    if has_bs:
        args.append(bs)
    if has_msg:
        args.append(messages)
    if not sliding:
        args.append(_full_tau(layer, kv.offset))
    args += [cs[0], cs[1]]
    q, k, v, pb, sinks, gate, w0, w1, normed_residual = s1(*args)
    attn_out = _core(layer, q, k, v, pb, sinks, kv)
    args2 = [h, normed_residual, attn_out, gate, w0, w1, cs[2], cs[3]]
    if has_bs:
        args2.append(bs)
    if has_msg:
        args2.append(messages)
    o, bs_new, *states = s2(*args2)
    for i, s in enumerate(states):
        conv[i] = s
    conv.advance(1)
    return o, bs_new
