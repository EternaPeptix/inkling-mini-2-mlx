"""Fused-visit assembly for the batched decoder (docs/FUSED.md, lead integration).

Each make_* returns a body(layer, *args) with the same signature and output layout as fastdecode.make_s1 / make_s2, built
from the exact row kernels (rows/*.py) instead of the per-op module calls.  batchdecode.BatchDecoder picks them per stage
with LOOPLING_BD_FUSED (comma list; empty = off).  Every row stays bitwise equal to the B=1 fast path (checked end to end
by bench/batch_dec.py).
"""
import mlx.core as mx

from mlx_lm.models.inkling2 import layers as L
from mlx_lm.models.inkling2.rows import gemv as RG
from mlx_lm.models.inkling2.rows import mid as RM
from mlx_lm.models.inkling2.rows import moe as RMo
from mlx_lm.models.inkling2.rows import proj as RP
from mlx_lm.models.inkling2.rows import qmv as RQ
from mlx_lm.models.inkling2.rows import rowlocal as RL

f32 = mx.float32
import os as _os
K1 = "k1" in set(filter(None, _os.environ.get("LOOPLING_BD_FUSED", "pre,k1,post,head").split(",")))
ROUTE = _os.environ.get("LOOPLING_BD_ROUTE", "auto")
# timing-only ablation (NOT exact; needs LOOPLING_COMPILE_DECODE=2): stages listed in LOOPLING_BD_FREEZE return their
# first output forever (so their kernels drop out of later steps).  Stages: k1 qkv mid rel oproj k10 gates moe sdpa
FREEZE = set(filter(None, _os.environ.get("LOOPLING_BD_FREEZE", "").split(",")))
_FRZ: dict = {}
DUP = set(filter(None, _os.environ.get("LOOPLING_BD_DUP", "").split(",")))


# profiling only (needs LOOPLING_COMPILE_DECODE=2): eval after every stage and log its name, so a GPU trace can attribute
# kernel time per stage (bench/stage_trace.py)
SYNC = _os.environ.get("LOOPLING_BD_SYNC", "") not in ("0", "")
LOG: list = []


def frz(name, key, fn, *a, **k):
    if SYNC:
        out = fn(*a, **k)
        t0 = __import__("time").perf_counter_ns()
        mx.eval(out)
        LOG.append((name, t0, __import__("time").perf_counter_ns()))
        __import__("time").sleep(float(_os.environ.get("LOOPLING_BD_SYNC_SLEEP", "0.0004")))
        return out
    if name in DUP:  # timing only: run the stage twice (exact output; the copy's kernels add the stage's in-pipeline cost)
        out = fn(*a, **k)
        extra = fn(*a, **k)
        leaves = [x for x in (extra if isinstance(extra, (tuple, list)) else
                              (extra.values() if isinstance(extra, dict) else [extra])) if isinstance(x, mx.array)]
        if isinstance(out, dict):
            ks = [q for q in out if isinstance(out[q], mx.array)]
            if not ks or not leaves:
                return out
            res = dict(out)
            res.update(zip(ks, mx.depends([out[q] for q in ks], leaves)))
            return res
        if isinstance(out, (tuple, list)):
            arr_i = [i for i, x in enumerate(out) if isinstance(x, mx.array)]
            dep = mx.depends([out[i] for i in arr_i], leaves) if arr_i and leaves else []
            res = list(out)
            for i, d in zip(arr_i, dep):
                res[i] = d
            return type(out)(res)
        return mx.depends([out], leaves)[0]
    if name not in FREEZE:
        return fn(*a, **k)
    kk = (name, key)
    if kk not in _FRZ:
        out = fn(*a, **k)
        mx.eval(out)
        _FRZ[kk] = out
    return _FRZ[kk]


MOE = _os.environ.get("LOOPLING_BD_MOE", "1") not in ("0", "")  # timing ablation: 0 = MLX MoE ops in the fused post


def _qw(m):
    if "bias" in m or m.mode != "affine" or m.bits != 8 or m.group_size != 64:
        raise ValueError("qmv rows: affine 8-bit gs64 without bias only")
    return (m["weight"], m["scales"], m["biases"])


def _inputs(layer, h, bs, msg, sliding):
    """Stage A (+ message gates): same ops as fastdecode.make_s1 (module calls; gates use the patched row gemv)."""
    dtype = h.dtype
    nr = L.unweighted_rms_norm(h.astype(f32))
    ai = layer.attention_input(bs, nr) if layer.attention_input is not None else None
    if sliding:
        hs = ai[..., 0, :] if ai is not None else h
        if msg is None:
            x = layer.input_layernorm(hs).astype(dtype)
            return nr, x, x, x
        st = layer._mix_messages((hs,) * 3, msg)
        q_in, k_in, v_in = [layer.input_layernorm(s).astype(dtype) for s in st]
        return nr, q_in, k_in, v_in
    if ai is None:
        qs = ks = vs = h
    else:
        qs, ks, vs = (ai[..., i, :] for i in range(3))
    if msg is not None:
        qs, ks, vs = layer._mix_messages((qs, ks, vs), msg)
    return (nr, layer.q_input_layernorm(qs).astype(dtype), layer.k_input_layernorm(ks).astype(dtype),
            layer.v_input_layernorm(vs).astype(dtype))


def _gates(gs, srcs, msg2, slots, gate_ins):
    """MessageGates for G gates: K2 down (one launch) + K3 forget/message + combine (one launch). Returns (B,G,D) f32."""
    x = gate_ins if isinstance(gate_ins, mx.array) else (
        gate_ins[0][:, None] if len(gate_ins) == 1 else mx.stack(gate_ins, axis=1)
    )
    down = RP.gate_down_rows(x, [g["down_proj"]["weight"] for g in gs])
    return RP.gate_mix_rows(down, list(srcs), msg2, slots, [g["forget_gate_proj"]["weight"] for g in gs],
                            [g["message_gate_proj"]["weight"] for g in gs])


def _inputs_k1(layer, h, bs, msg):
    """K1 PRE (one launch) [+ message gates + K3b NORM]. Returns nr (B,1,D) f32, q_in, k_in, v_in (B,D) bf16."""
    B, D = h.shape[0], h.shape[-1]
    bs2 = None if bs is None else bs.reshape(bs.shape[0], B, D)
    m2 = None if msg is None else msg.reshape(B, 4, D)
    r = RL.k1_pre_layer(layer, h.reshape(B, D), bs2, m2)
    nr = r["normed_residual"].reshape(B, 1, D)
    if msg is None:
        return nr, r["q_in"], r["k_in"], r["v_in"]
    gs = (layer.q_message_gate, layer.k_message_gate, layer.v_message_gate)
    mixed = _gates(gs, (r["s_q"], r["s_k"], r["s_v"]), m2, (0, 1, 2), r["qkv_gate_in"])
    q_in, k_in, v_in = RL.k3b_norm_layer(layer, mixed)
    return nr, q_in, k_in, v_in


def make_post(layer, has_bs, has_msg, *, gated=False):
    """K9 (o_proj) + K10 POST + [mlp gate + K3b] + K11-13 moe_rows.
    When `gated=True`, SDPA already multiplied attn_out by sigmoid(gate) on store, so o_proj uses plain qmv_rows."""

    def post(layer, h, nr, attn_out, gate, w0, w1, cs2, cs3, *rest):
        a = layer.self_attn
        rest = list(rest)
        bs = rest.pop(0) if has_bs else None
        msg = rest.pop(0) if has_msg else None
        B, D = h.shape[0], h.shape[-1]
        lk = (id(layer), has_msg)
        if gated:
            op = frz("oproj", lk, RQ.qmv_rows, attn_out.reshape(B, -1), *_qw(a.o_proj))
        else:
            op = frz("oproj", lk, RP.oproj_rows, attn_out.reshape(B, -1), gate.reshape(B, -1), *_qw(a.o_proj))
        bs2 = None if bs is None else bs.reshape(bs.shape[0], B, D)
        m2 = None if msg is None else msg.reshape(B, 4, D)
        r = frz("k10", lk, RL.k10_post_layer, layer, op, h.reshape(B, D), nr.reshape(B, D), bs2, cs2, m2, debug=False)
        if layer.is_block_boundary:
            block = nr[None]
            bs = block if bs is None else mx.concatenate([bs, block])
        if msg is None:
            e = frz("moe", lk, RMo.moe_rows, layer, r["mlp_x"], r["residual"], cs3, route=ROUTE)
            return (e["out"].reshape(B, 1, D), bs, w0, w1, r["w2"], e["w3"])
        mixed = frz("gates", lk + ("mlp",), _gates, (layer.mlp_message_gate,), (r["mlp_src"],), m2, (3,),
                    (r["mlp_gate_in"],))
        (x,) = frz("gates", lk + ("mlpn",), RL.k3b_norm_layer, layer, mixed, mlp=True)
        if not MOE:
            from mlx_lm.models.inkling2 import fastdecode as FD
            hid, w3 = FD._sc(layer.post_mlp_convolution, layer.mlp(x[:, None]), cs3)
            o = (r["residual"][:, None] + hid.astype(f32)).astype(h.dtype)
            return (o, bs, w0, w1, r["w2"], w3)
        e = frz("moe", lk, RMo.moe_rows, layer, x, r["residual"], cs3, route=ROUTE)
        return (e["out"].reshape(B, 1, D), bs, w0, w1, r["w2"], e["w3"])

    return post


def make_pre(layer, has_bs, has_msg):
    """K1 (module ops) + K4 (qmv_rows_multi, 1-3 launches) + K5 (mid_rows, 1 launch) + K6 (rel_logits_rows, 1 launch)."""
    sliding = isinstance(layer, L.SlidingDecoderLayer)
    rel_w_bf = mx.contiguous(layer.self_attn.rel_logits_proj.proj.astype(mx.bfloat16))
    fuse_rel = (
        rel_w_bf.dtype == mx.bfloat16 and rel_w_bf.shape == (16, 1024) and layer.self_attn.num_heads == 24
        and "rel" not in DUP and "rel" not in FREEZE and not SYNC
    )

    def pre(layer, h, *rest):
        a = layer.self_attn
        rest = list(rest)
        bs = rest.pop(0) if has_bs else None
        msg = rest.pop(0) if has_msg else None
        tau = rest.pop(0) if not sliding else None
        cs0, cs1 = rest
        B, D = h.shape[0], h.shape[-1]
        lk = (id(layer), has_msg)
        if K1:
            nr, q_in, k_in, v_in = frz("k1", lk, _inputs_k1, layer, h, bs, msg)
        else:
            nr, q_in, k_in, v_in = _inputs(layer, h, bs, msg, sliding)
        kw = vw = None
        if K1:
            qkv_flat, qkv_lay = frz("qkv", lk, RP.qkv_rows, a, q_in.reshape(B, D), k_in.reshape(B, D), v_in.reshape(B, D),
                                    flat=True)
            dm = {item[0]: (item[1], item[2], item[-1]) for item in qkv_lay}
            def _ext_qkv(nm):
                off, st, nc = dm[nm]
                if B == 1 or st == nc:
                    return qkv_flat[off:off + B * nc].reshape(B, nc)
                return mx.stack([qkv_flat[off + b * st : off + b * st + nc] for b in range(B)], axis=0)
            gate = _ext_qkv("gate")
            sinks = _ext_qkv("sinks")
            if fuse_rel:
                q, k, v, pb, w0, w1 = frz(
                    "mid", lk, RM.mid_rows_flat, layer, qkv_flat, qkv_lay, cs0, cs1,
                    tau=None if sliding else tau.reshape(-1), rel_w=rel_w_bf)
            else:
                q, k, v, rel, w0, w1 = frz(
                    "mid", lk, RM.mid_rows_flat, layer, qkv_flat, qkv_lay, cs0, cs1,
                    tau=None if sliding else tau.reshape(-1))
                rw = rel_w_bf if rel.dtype == mx.bfloat16 else a.rel_logits_proj.proj.astype(rel.dtype)
                pb = frz("rel", lk, RG.rel_logits_rows, rel, rw)
        else:
            if sliding and q_in is k_in and k_in is v_in:
                xq = q_in.reshape(B, D)
                qp, rp, gate, sinks, kp, vp = RQ.qmv_rows_multi(
                    xq, [_qw(a.q_proj), _qw(a.r_proj), _qw(a.gate_proj), _qw(a.sink_proj), _qw(a.k_proj), _qw(a.v_proj)])
            else:
                xq = q_in.reshape(B, D)
                qp, rp, gate, sinks = RQ.qmv_rows_multi(xq, [_qw(a.q_proj), _qw(a.r_proj), _qw(a.gate_proj), _qw(a.sink_proj)])
                if sliding:
                    (kp,) = RQ.qmv_rows_multi(k_in.reshape(B, D), [_qw(a.k_proj)])
                    (vp,) = RQ.qmv_rows_multi(v_in.reshape(B, D), [_qw(a.v_proj)])
                else:
                    kp, kw = RQ.qmv_rows_multi(k_in.reshape(B, D), [_qw(a.k_proj), _qw(a.k_sconv.weight_proj)])
                    vp, vw = RQ.qmv_rows_multi(v_in.reshape(B, D), [_qw(a.v_proj), _qw(a.v_sconv.weight_proj)])
            q, k, v, rel, w0, w1 = frz("mid", lk, RM.mid_rows, layer, qp, rp, kp, vp, cs0, cs1,
                                       tau=None if sliding else tau.reshape(-1), k_wproj=kw, v_wproj=vw)
            rw = rel_w_bf if rel.dtype == mx.bfloat16 else a.rel_logits_proj.proj.astype(rel.dtype)
            pb = frz("rel", lk, RG.rel_logits_rows, rel, rw)
        # layouts of fastdecode s1: q/k/v (B,H,1,128), pb (B,1,H,R), sinks (B,1,H), gate (B,1,q_dim)
        return (q[:, :, None, :], k[:, :, None, :], v[:, :, None, :], pb[:, None], sinks[:, None], gate[:, None],
                w0, w1, nr)

    return pre
