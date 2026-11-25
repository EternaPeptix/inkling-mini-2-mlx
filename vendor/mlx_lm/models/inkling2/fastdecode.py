"""Lane A: low-overhead decode path (B=1, L=1) for Inkling2 decoder layers. Flags (default ON, `=0` disables):

  LOOPLING_COMPILE_DECODE   1 (default) segments compiled with mx.compile (fixed shapes, one trace per distinct input shape)
                            2 same segmentation + hoisted attention masks, but no mx.compile (isolates the compile effect)
                            0 off (reference path)
  LOOPLING_FUSED_CONV       1 (default) one Metal kernel per decode ShortConvolution; 0 reference conv

The math is the reference math: the layer is split in S1 (norms, attention-input mix, projections, short convs, l2 clips),
the cache update + attention core (shapes vary with the cache, never compiled), and S2 (output projection, MLP/MoE,
convs, residual). Elementwise chains are fused by mx.compile; everything else is the same primitive as the reference.

Weights are never frozen into a compiled closure: the layer module itself is an argument of every compiled call, so
its arrays are graph inputs (reassigning / load_weights / update / adapter fuse take effect on the next decode step).
Compiled state lives in a registry keyed by id(layer) and holds only a weakref to the layer (no reference cycle).
"""
import os
import weakref

import mlx.core as mx

_MODE = os.environ.get("LOOPLING_COMPILE_DECODE", "1")
_COMPILE = _MODE == "1"
_FCONV = os.environ.get("LOOPLING_FUSED_CONV", "1") not in ("0", "")
_FD_FUSED = os.environ.get("LOOPLING_FD_FUSED", "1") not in ("0", "")
_ctx: dict = {}
_tau: dict = {}
_wpos_cache: dict = {}
_fds: dict = {}  # id(layer) -> compiled segments; the entry is removed when the layer dies
_RM_MODS = None


def _mods():
    global _RM_MODS
    if _RM_MODS is None:
        from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache
        from mlx_lm.models.inkling2 import layers as L
        from mlx_lm.models.inkling2.rows import mid as RM
        from mlx_lm.models.inkling2.rows import moe as RMo
        from mlx_lm.models.inkling2.rows import proj as RP
        from mlx_lm.models.inkling2.rows import qmv as RQ
        from mlx_lm.models.inkling2.rows import rowlocal as RL
        from mlx_lm.models.inkling2.rows import sdpa as RS
        from mlx_lm.models.inkling2.rows import visit as RV
        _RM_MODS = (KVCache, RotatingKVCache, BatchKVCache, BatchRotatingKVCache, L, RM, RMo, RP, RQ, RL, RS, RV)
    return _RM_MODS


def _wpos(pos: int) -> mx.array:
    a = _wpos_cache.get(pos)
    if a is None:
        if len(_wpos_cache) >= 256:
            _wpos_cache.clear()
        a = _wpos_cache[pos] = mx.array([pos], mx.int32)
    return a


def _rel_w_bf(rel_proj) -> mx.array:
    p = rel_proj.proj
    c = getattr(rel_proj, "_fd_bf_cache", None)
    if c is None or c[0] is not p:
        c = rel_proj._fd_bf_cache = (p, mx.contiguous(p.astype(mx.bfloat16)))
    return c[1]


def _is_aff8(m) -> bool:
    return (
        getattr(m, "mode", None) == "affine"
        and getattr(m, "bits", None) == 8
        and getattr(m, "group_size", None) == 64
        and "bias" not in m
    )


def eligible(h, cache, conv_mask) -> bool:
    if cache is None or h.shape[0] != 1 or h.shape[1] != 1:
        return False
    conv = cache[1]
    if conv.lengths is not None:
        return False
    if conv.left_padding is not None or conv_mask is not None:
        # B=1 batch caches (server path): left_padding starts at [0] and is decremented per token, so the conv mask is
        # all True (an identity `where`). cache.py flags such caches; any prepare/extend/filter drops the flag.
        if not getattr(conv, "_zero_pad", False):
            return False
    if any(c is None for c in conv.cache):
        return False
    kv = cache[0]
    if isinstance(kv.offset, mx.array) and getattr(kv, "_fd_delta", None) is None:
        return False
    return getattr(kv, "_packed", None) is not None or isinstance(kv.keys, mx.array)


def _end(kv):
    """Host-int token count (the `offset` of a plain KV/Rotating cache), also for flagged B=1 batch caches."""
    d = getattr(kv, "_fd_delta", None)
    if d is None:
        return kv.offset
    return (kv._offset if hasattr(kv, "_offset") else kv._idx) + d


def _swap(dst, src, saved):
    items = src.items() if isinstance(src, dict) else enumerate(src)
    for k, v in items:
        if isinstance(v, (dict, list)):
            _swap(dst[k], v, saved)
        else:
            saved.append((dst, k, dst[k]))
            dst[k] = v


def _fn(ref, body):
    """body(layer, *args). Compiled: the layer module is passed as an argument, so all its arrays are graph inputs
    (read at every call); during tracing they are swapped into the real module and restored afterwards."""
    if not _COMPILE:
        return lambda *a: body(ref(), *a)

    def traced(tree, *a):
        layer = ref()
        saved = []
        _swap(layer, tree, saved)
        try:
            return body(layer, *a)
        finally:
            for d, k, v in reversed(saved):
                d[k] = v

    cf = mx.compile(traced)
    return lambda *a: cf(ref(), *a)


_CONV_SRC = """
    uint c = thread_position_in_grid.x;
    if (c >= C) return;
    float x0 = float(x[c]);
    float r1 = st[C + c], r2 = st[2 * C + c];
    // the reference casts the (f32) conv state to the weight dtype before the conv
    float s0 = float(T(st[c])), s1 = float(T(r1)), s2 = float(T(r2));
    float acc = 0.0f;
    acc = fma(float(w[c * 4 + 0]), s0, acc);
    acc = fma(float(w[c * 4 + 1]), s1, acc);
    acc = fma(float(w[c * 4 + 2]), s2, acc);
    acc = fma(float(w[c * 4 + 3]), x0, acc);
    float conv = float(T(acc));
    out[c] = T((conv + x0) / SCALE);
    nst[c] = r1;
    nst[C + c] = r2;
    nst[2 * C + c] = x0;
"""
# Batched variant (B rows, grid.y = row): the per-element arithmetic is the same as _CONV_SRC (used by batchdecode.py).
_CONV_ROWS_SRC = """
    uint c = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (c >= C) return;
    x += b * C; st += b * 3 * C; out += b * C; nst += b * 3 * C;
""" + _CONV_SRC.split("if (c >= C) return;", 1)[1]
_conv_kernels: dict = {}


def short_conv_decode(mod, x, state):
    """ShortConvolution for one decode token: (out, new_state), bit-identical to the reference (checked by bench/convtest_a.py)."""
    import math

    import numpy as np

    C = x.shape[-1]
    K = mod.kernel_size
    assert K == 4
    scale = repr(float(np.float32(math.sqrt(K + 1)))) + "f"
    B = x.shape[0]
    rows = B > 1
    k = _conv_kernels.get((scale, rows))
    if k is None:
        k = _conv_kernels[(scale, rows)] = mx.fast.metal_kernel(
            name="short_conv_decode_rows" if rows else "short_conv_decode", input_names=["x", "st", "w"],
            output_names=["out", "nst"], source=(_CONV_ROWS_SRC if rows else _CONV_SRC).replace("SCALE", scale),
        )
    out, nst = k(
        inputs=[x, state, mod.conv1d.weight], template=[("T", x.dtype), ("C", C)], grid=(C, B, 1),
        threadgroup=(min(256, C), 1, 1), output_shapes=[x.shape, state.shape], output_dtypes=[x.dtype, mx.float32],
    )
    return out, nst


def _conv_ok(mod, x):
    return _FCONV and mod.kernel_size == 4 and x.dtype == mx.bfloat16 and mod.conv1d.weight.dtype == mx.bfloat16


def _sc(mod, x, st):
    """ShortConvolution step returning (out, new 3-row state)."""
    if _conv_ok(mod, x):
        return short_conv_decode(mod, x, st)
    o, w = mod(x, st, None)
    return o, w[:, w.shape[1] - 3 :, :]


def _dsc(mod, x, dyn, st):
    o, w = mod(x, dyn, st, None)
    return o, w[:, w.shape[1] - 3 :, :]


def _build(layer):
    from mlx_lm.models.inkling2 import layers as L

    f32 = mx.float32
    sliding = isinstance(layer, L.SlidingDecoderLayer)
    ref = weakref.ref(layer)
    out = {}

    def make_s1(has_bs, has_msg, wrap=None):
        def s1(layer, h, *rest):
            a = layer.self_attn
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
                key, w0 = _sc(a.k_sconv, a.k_proj(k_in), cs0)
                value, w1 = _sc(a.v_sconv, a.v_proj(v_in), cs1)
                pre = a._attend_pre(q_in, a.q_proj(q_in), key, value, a.r_proj(q_in), None)
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
                pre = a._attend_pre(q_in, a.q_proj(q_in), key, value, a.r_proj(q_in), tau)
            return (*pre, w0, w1, normed_residual)

        return wrap(s1) if wrap else _fn(ref, s1)

    def make_s2(has_bs, has_msg, wrap=None):
        def s2(layer, h, normed_residual, attn_out, gate, w0, w1, cs2, cs3, *rest):
            a = layer.self_attn
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

        return wrap(s2) if wrap else _fn(ref, s2)

    out["s1"] = {}
    out["s2"] = {}
    out["make_s1"], out["make_s2"] = make_s1, make_s2
    return out


def _get_fd(layer):
    fd = _fds.get(id(layer))
    if fd is None:
        fd = _fds[id(layer)] = _build(layer)
        weakref.finalize(layer, _fds.pop, id(layer), None)
    return fd


def _full_tau(layer, offset):
    a = layer.self_attn
    key = (offset, a.log_scaling_alpha, a.log_scaling_n_floor)
    t = _tau.get(key)
    if t is None:
        if len(_tau) > 8:
            _tau.clear()
        pos = mx.arange(1) + offset
        pos = pos.astype(mx.float32)
        t = 1.0 + a.log_scaling_alpha * mx.log(mx.maximum((pos + 1) / a.log_scaling_n_floor, 1.0))
        t = t[..., None, None]
        _tau[key] = t
    return t


def _core(layer, q, k, v, pb, sinks, kv):
    from mlx_lm.models.inkling2 import layers as L

    a = layer.self_attn
    k_all, v_all = kv.update_and_fetch(k, v)
    S = k_all.shape[2]
    end = _end(kv)
    R = pb.shape[-1]
    idx = getattr(kv, "_idx", None) if isinstance(kv, (L.RotatingKVCache, L.BatchRotatingKVCache)) else None
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


def _fused_b1_ok_layer(layer) -> bool:
    L = _mods()[4]
    if not L._use_kernels():
        return False
    a = layer["self_attn"]
    if a.num_heads != 24 or a.num_key_value_heads != 8 or a.head_dim != 128:
        return False
    if (
        a["q_l2_clip"]["scale"].shape != (24, 128)
        or a["rel_l2_clip"]["scale"].shape != (24, 16)
        or a["rel_logits_proj"]["proj"].shape != (16, 1024)
    ):
        return False
    ks, vs = a["k_sconv"], a["v_sconv"]
    pac, pmc = layer["post_attention_convolution"], layer["post_mlp_convolution"]
    if ks.kernel_size != 4 or vs.kernel_size != 4 or pac.kernel_size != 4 or pmc.kernel_size != 4:
        return False
    ai = dict.get(layer, "attention_input")
    if (
        pac["conv1d"]["weight"].dtype != mx.bfloat16
        or pmc["conv1d"]["weight"].dtype != mx.bfloat16
        or layer["post_attention_layernorm"]["weight"].dtype != mx.bfloat16
        or layer["mlp_input"]["query"].dtype != mx.float32
        or (ai is not None and ai["query"].dtype != mx.float32)
    ):
        return False
    projs = [a["q_proj"], a["r_proj"], a["gate_proj"], a["sink_proj"], a["k_proj"], a["v_proj"], a["o_proj"]]
    if isinstance(layer, L.SlidingDecoderLayer):
        if (
            layer["input_layernorm"]["weight"].dtype != mx.bfloat16
            or ks["conv1d"]["weight"].dtype != mx.bfloat16
            or vs["conv1d"]["weight"].dtype != mx.bfloat16
        ):
            return False
    else:
        if (
            layer["q_input_layernorm"]["weight"].dtype != mx.bfloat16
            or layer["k_input_layernorm"]["weight"].dtype != mx.bfloat16
            or layer["v_input_layernorm"]["weight"].dtype != mx.bfloat16
        ):
            return False
        projs += [ks["weight_proj"], vs["weight_proj"]]
    if not all(_is_aff8(m) for m in projs):
        return False
    mlp = layer["mlp"]
    if getattr(mlp, "sharding_group", None) is not None:
        return False
    gate = mlp["gate"]
    if gate["weight"].dtype != mx.float32 or gate["weight"].shape != (128, 2304) or gate.top_k != 8:
        return False
    ex = mlp["experts"]
    gu_m, dn_m = ex["gate_up_proj"], ex["down_proj"]
    if (
        getattr(gu_m, "mode", None) not in ("nvfp4", "mxfp8")
        or getattr(dn_m, "mode", None) != gu_m.mode
        or "biases" in gu_m
        or "biases" in dn_m
    ):
        return False
    return True


def _fused_b1_ok(layer, h, conv) -> bool:
    return (
        _FD_FUSED
        and _FCONV
        and h.dtype == mx.bfloat16
        and h.shape == (1, 1, 2304)
        and not getattr(conv, "record", False)
        and all(c.dtype == mx.float32 for c in conv.cache)
        and _fused_b1_ok_layer(layer)
    )


def _decode_layer_fused_b1(layer, h, bs, cache, messages):
    KVCache, RotatingKVCache, BatchKVCache, BatchRotatingKVCache, L, RM, RMo, RP, RQ, RL, RS, RV = _mods()
    conv = cache[1]
    kv = cache[0]
    cs = conv.cache
    a = layer["self_attn"]
    sliding = "input_layernorm" in layer
    h2 = h.reshape(1, 2304)
    bs_is_list = isinstance(bs, list)
    bs_in = None if (bs is None or (bs_is_list and not bs)) else (bs if bs_is_list else bs.reshape(bs.shape[0], 1, 2304))
    m2 = None if messages is None else messages.reshape(1, 4, 2304)

    r1 = RL.k1_pre_layer(layer, h2, bs_in, m2)
    nr = r1["normed_residual"]
    if m2 is None:
        q_in, k_in, v_in = r1["q_in"], r1["k_in"], r1["v_in"]
    else:
        gs = (layer["q_message_gate"], layer["k_message_gate"], layer["v_message_gate"])
        mixed = RV._gates(gs, (r1["s_q"], r1["s_k"], r1["s_v"]), m2, (0, 1, 2), r1["qkv_gate_in"])
        q_in, k_in, v_in = RL.k3b_norm_layer(layer, mixed)

    qkv_flat, qkv_lay = RP.qkv_rows(a, q_in, k_in, v_in, flat=True)
    if qkv_lay[2][0] == "gate" and qkv_lay[3][0] == "sinks":
        gate_off, sink_off = qkv_lay[2][1], qkv_lay[3][1]
    else:
        dm = {nm: off for nm, off, _ in qkv_lay}
        gate_off, sink_off = dm["gate"], dm["sinks"]
    tau_1d = None if sliding else _full_tau(layer, _end(kv)).reshape(-1)
    rel_w_bf = _rel_w_bf(a["rel_logits_proj"])

    use_packed_kv = (
        type(kv) in (KVCache, RotatingKVCache)
        and getattr(kv, "keep", 0) == 0
        and getattr(kv, "_fd_delta", None) is None
        and (
            (kv._packed is not None and kv._packed.dtype == mx.bfloat16 and kv._packed.shape[1] == 8 and kv._packed.shape[3] == 128)
            or (kv._keys is not None and kv._keys.dtype == mx.bfloat16 and kv._keys.shape[0] == 1 and kv._keys.shape[1] == 8 and kv._keys.shape[3] == 128)
        )
    )
    op_m = a["o_proj"]
    if use_packed_kv:
        if kv._packed is None:
            kv.pack_b1()
        q, kv_new, pb, w0, w1 = RM.mid_rows_flat(
            layer, qkv_flat, qkv_lay, cs[0], cs[1], tau=tau_1d, rel_w=rel_w_bf, packed_kv=True
        )
        kv_buf, S, end, idx, rot, cap = kv.update_packed_b1(kv_new, _wpos)
        attn_out = RS.sdpa_b1_packed(
            q, kv_buf, pb, qkv_flat, gate_off, sink_off, S, end, idx, rot, cap,
            scale=a.scaling, window=a.sliding_window or 0,
        )
        op = RQ.qmv_rows(attn_out, op_m["weight"], op_m["scales"], op_m["biases"])
    else:
        q, kk, vv, pb, w0, w1 = RM.mid_rows_flat(
            layer, qkv_flat, qkv_lay, cs[0], cs[1], tau=tau_1d, rel_w=rel_w_bf, packed_kv=False
        )
        gate = qkv_flat[gate_off : gate_off + 3072].reshape(1, 3072)
        sinks = qkv_flat[sink_off : sink_off + 24].reshape(1, 1, 24)
        use_sdpa_rows = (
            type(kv) in (KVCache, RotatingKVCache, BatchKVCache, BatchRotatingKVCache)
            and getattr(kv, "keep", 0) == 0
            and q.dtype == mx.bfloat16
            and kk.dtype == mx.bfloat16
            and (kv.keys is None or kv.keys.dtype == mx.bfloat16)
            and q.shape[0] == 1
            and q.shape[1] == 24
            and q.shape[2] == 128
            and kk.shape[1] == 8
        )
        if use_sdpa_rows:
            k_all, v_all = kv.update_and_fetch(kk[:, :, None, :], vv[:, :, None, :])
            is_rot = isinstance(kv, (RotatingKVCache, BatchRotatingKVCache))
            meta = [
                (
                    int(_end(kv)),
                    int(kv._idx) if is_rot else 0,
                    1 if is_rot else 0,
                    int(kv.keys.shape[2]),
                    int(kv.values.shape[2]),
                )
            ]
            attn_out = RS.sdpa_rows(
                q[:, :, None, :], [k_all], [v_all], pb[:, None], sinks,
                meta, scale=a.scaling, window=a.sliding_window or 0, gate=gate,
            ).reshape(1, 3072)
            op = RQ.qmv_rows(attn_out, op_m["weight"], op_m["scales"], op_m["biases"])
        else:
            attn_out = _core(layer, q[:, :, None, :], kk[:, :, None, :], vv[:, :, None, :], pb[:, None], sinks, kv)
            op = RP.oproj_rows(attn_out, gate, op_m["weight"], op_m["scales"], op_m["biases"])

    r10 = RL.k10_post_layer(layer, op, h2, nr, bs_in, cs[2], m2, debug=False)
    if layer.is_block_boundary:
        if bs_is_list:
            bs_new = bs + [nr]
        else:
            block = nr.reshape(1, 1, 1, 2304)
            bs_new = block if bs is None else mx.concatenate([bs, block])
    else:
        bs_new = bs

    if m2 is None:
        x_mlp = r10["mlp_x"]
    else:
        mixed = RV._gates((layer["mlp_message_gate"],), (r10["mlp_src"],), m2, (3,), (r10["mlp_gate_in"],))
        (x_mlp,) = RL.k3b_norm_layer(layer, mixed, mlp=True)
    e = RMo.moe_rows(layer, x_mlp, r10["residual"], cs[3], route=RV.ROUTE)
    conv[0], conv[1], conv[2], conv[3] = w0, w1, r10["w2"], e["w3"]
    conv.advance(1)
    return e["out"].reshape(1, 1, 2304), bs_new


def decode_layer(layer, h, bs, cache, messages):
    if _fused_b1_ok(layer, h, cache[1]):
        return _decode_layer_fused_b1(layer, h, bs, cache, messages)
    if isinstance(bs, list):
        bs = None if not bs else mx.stack(bs, axis=0)[:, :, None, :]
    if messages is not None and messages.ndim == 3:
        messages = messages[:, None, :, :]
    fd = _get_fd(layer)
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
        args.append(_full_tau(layer, _end(kv)))
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
