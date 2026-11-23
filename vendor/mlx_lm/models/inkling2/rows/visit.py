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
from mlx_lm.models.inkling2.rows import qmv as RQ

f32 = mx.float32


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


def make_pre(layer, has_bs, has_msg):
    """K1 (module ops) + K4 (qmv_rows_multi, 1-3 launches) + K5 (mid_rows, 1 launch) + K6 (rel_logits_rows, 1 launch)."""
    sliding = isinstance(layer, L.SlidingDecoderLayer)

    def pre(layer, h, *rest):
        a = layer.self_attn
        rest = list(rest)
        bs = rest.pop(0) if has_bs else None
        msg = rest.pop(0) if has_msg else None
        tau = rest.pop(0) if not sliding else None
        cs0, cs1 = rest
        B, D = h.shape[0], h.shape[-1]
        nr, q_in, k_in, v_in = _inputs(layer, h, bs, msg, sliding)
        xq = q_in.reshape(B, D)
        kw = vw = None
        if sliding and q_in is k_in and k_in is v_in:
            qp, rp, gate, sinks, kp, vp = RQ.qmv_rows_multi(
                xq, [_qw(a.q_proj), _qw(a.r_proj), _qw(a.gate_proj), _qw(a.sink_proj), _qw(a.k_proj), _qw(a.v_proj)])
        else:
            qp, rp, gate, sinks = RQ.qmv_rows_multi(xq, [_qw(a.q_proj), _qw(a.r_proj), _qw(a.gate_proj), _qw(a.sink_proj)])
            if sliding:
                (kp,) = RQ.qmv_rows_multi(k_in.reshape(B, D), [_qw(a.k_proj)])
                (vp,) = RQ.qmv_rows_multi(v_in.reshape(B, D), [_qw(a.v_proj)])
            else:
                kp, kw = RQ.qmv_rows_multi(k_in.reshape(B, D), [_qw(a.k_proj), _qw(a.k_sconv.weight_proj)])
                vp, vw = RQ.qmv_rows_multi(v_in.reshape(B, D), [_qw(a.v_proj), _qw(a.v_sconv.weight_proj)])
        q, k, v, rel, w0, w1 = RM.mid_rows(layer, qp, rp, kp, vp, cs0, cs1,
                                           tau=None if sliding else tau.reshape(-1), k_wproj=kw, v_wproj=vw)
        pb = RG.rel_logits_rows(rel, a.rel_logits_proj.proj.astype(rel.dtype))
        # layouts of fastdecode s1: q/k/v (B,H,1,128), pb (B,1,H,R), sinks (B,1,H), gate (B,1,q_dim)
        return (q[:, :, None, :], k[:, :, None, :], v[:, :, None, :], pb[:, None], sinks[:, None], gate[:, None],
                w0, w1, nr)

    return pre
