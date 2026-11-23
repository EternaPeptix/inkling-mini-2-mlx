"""Speed of rows/rowlocal.py K1 / K3b / K10 vs the MLX ops they replace (the compiled fast-decode segment portions), B=1 and 8.

  LANE=f1 RHOST=512S1 scripts/r --gpu 'python bench/fused_rowlocal_speed.py'

ref   = the current B=1 path portion (mx.compile'd closure over the layer, short conv via the fused conv kernel), called once
        per row (B calls for B rows; the current path cannot batch rows).
fused = one launch for all B rows.
queued: N independent calls queued, one eval.   chain: N dependent calls (output feeds the next input; only where the output is
directly usable as the next input without an extra op: K1 non-message, K10).  Median us per call over REPS evals.
"""
import json, os, statistics, time

import mlx_lm
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import layers as L
from mlx_lm.models.inkling2.rows import rowlocal as RL

V = os.path.expanduser("~/loopling-opt/runs/visits_p3")
idx = json.load(open(V + "/index.json"))
model, _ = load(os.path.expanduser("~/loopling/models/") + idx["model"])
layers = list(model.model.language_model.layers)
D = 2304
f32, bf = mx.float32, mx.bfloat16
REPS, N = 40, 48
rms = L.unweighted_rms_norm


def visit(layer, msg):
    v = [x for x in idx["visits"] if x["layer"] == layer and x["has_msg"] == msg][0]
    return mx.load(V + "/" + v["file"])


def ref_k1(layer, has_msg):
    sliding = isinstance(layer, L.SlidingDecoderLayer)

    def f(h, bs, msg):
        nr = rms(h.astype(f32))
        ai = layer.attention_input(bs, nr) if layer.attention_input is not None else None
        if sliding:
            hs = ai[..., 0, :] if ai is not None else h
            sts = (hs,) * 3
        else:
            sts = (h,) * 3 if ai is None else tuple(ai[..., i, :] for i in range(3))
        if not has_msg:
            if sliding:
                return nr, layer.input_layernorm(hs).astype(h.dtype)
            lns = (layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm)
            return (nr,) + tuple(m(s).astype(h.dtype) for m, s in zip(lns, sts))
        return (nr,) + tuple(sts) + tuple(
            mx.concatenate([rms(s), rms(msg[..., i, :])], axis=-1).astype(bf) for i, s in enumerate(sts))

    return mx.compile(f)


def ref_k10(layer, has_msg):
    def f(op, h, nr, bs, cs2, msg):
        hidden, w2 = FD._sc(layer.post_attention_convolution, op, cs2)
        residual = hidden = hidden.astype(f32)
        if layer.is_block_boundary:
            bs = nr[None] if bs is None else mx.concatenate([bs, nr[None]])
        else:
            residual = hidden = h.astype(f32) + hidden
        normed = rms(hidden)
        src = layer.mlp_input(bs, normed)[..., 0, :]
        if has_msg:
            return hidden, w2, residual, normed, src, mx.concatenate([rms(src), rms(msg[..., 3, :])], axis=-1).astype(bf)
        return hidden, w2, residual, normed, src, layer.post_attention_layernorm(src).astype(bf)

    return mx.compile(f)


def fresh(x):
    """a new array buffer with the same values (distinct graph node, so the compiled program cannot merge calls)."""
    y = x + mx.zeros((), x.dtype)
    mx.eval(y)
    return y


def bench(step, mk, chain):
    """step(primary) -> (result, feed); mk() -> a fresh primary input (the ref's is a list of B per-row arrays, so its B calls are
    distinct).  The N-call program is mx.compile'd (graph built once; per-eval CPU cost = kernel encode only).  queued: N calls on
    N distinct inputs; chain: call i+1 consumes the feed of call i."""
    if chain:
        def prog(x):
            rs = []
            for _ in range(N):
                r, x = step(x)
                rs.append(r)
            return rs
        args = [mk()]
    else:
        def prog(*xs):
            return [step(x)[0] for x in xs]
        args = [mk() for _ in range(N)]
    f = mx.compile(prog)
    for _ in range(4):
        mx.eval(f(*args))
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); mx.eval(f(*args)); ts.append((time.perf_counter() - t) / N * 1e6)
    return statistics.median(ts)


def bc(x, B):  # tile a single-row array (leading dim 1 after reshape) to B rows, materialized
    return mx.contiguous(mx.broadcast_to(x, (B,) + x.shape[1:]) + mx.zeros((1,), x.dtype))


KINDS = [("S", 1, False), ("Sb", 6, False), ("F", 5, False), ("Sb0", 0, False), ("Sm", 7, True), ("Fm", 11, True), ("Sbm", 12, True)]
print("%-4s %-4s %-2s | %9s %9s %6s | %9s %9s %6s" % ("stg", "kind", "B", "ref que", "fus que", "x", "ref chn", "fus chn", "x"))
for stage in ("K1", "K10", "K3b"):
    for kind, li, hm in KINDS:
        if stage == "K3b" and not hm:
            continue
        layer = layers[li]
        sliding = isinstance(layer, L.SlidingDecoderLayer)
        t = visit(li, hm)
        h1, bs1, msg1 = t["h"], t.get("bs"), t.get("msg")
        cs1 = t["cs2"]
        op1 = mx.random.normal((1, 1, D)).astype(bf)
        nr1 = mx.random.normal((1, 1, D))
        mx.eval(op1, nr1)
        for B in (1, 8):
            h = bc(h1.reshape(1, D), B)
            bs = None if bs1 is None else mx.contiguous(mx.broadcast_to(bs1.reshape(-1, 1, D), (bs1.shape[0], B, D)) + 0.0)
            msg = None if msg1 is None else bc(msg1.reshape(1, 4, D), B)
            nrB, csB, opB = bc(nr1.reshape(1, D), B), bc(cs1.reshape(1, 3, D), B), bc(op1.reshape(1, D), B)
            mixed = mx.random.normal((B, 3, D))
            m1 = mx.random.normal((1, 1, D))
            mx.eval(h, bs, msg, nrB, csB, opB, mixed, m1)
            chain = True
            if stage == "K1":
                rf = ref_k1(layer, hm)
                chain = not hm
                x_ref, x_fus = h1, h

                def ref_step(xs, rf=rf):
                    rows = [rf(x, bs1, msg1) for x in xs]
                    return rows, [r[1] for r in rows]

                def fus_step(x):
                    o = RL.k1_pre_layer(layer, x, bs, msg)
                    return o, o.get("q_in")
            elif stage == "K10":
                rf = ref_k10(layer, hm)
                x_ref, x_fus = op1, opB

                def ref_step(xs, rf=rf):
                    rows = [rf(x, h1, nr1, bs1, cs1, msg1) for x in xs]
                    return rows, [r[0] for r in rows]

                def fus_step(x):
                    o = RL.k10_post_layer(layer, x, h, nrB, bs, csB, msg)
                    return o, o["post_attn_conv"]
            else:
                chain = False
                lns = [layer.input_layernorm] * 3 if sliding else [layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm]
                rf = lambda x, lns=lns: tuple(ln(x).astype(bf) for ln in lns)
                x_ref, x_fus = m1, mixed

                def ref_step(xs, rf=rf):
                    rows = [rf(x) for x in xs]
                    return rows, None

                def fus_step(x):
                    return RL.k3b_norm_layer(layer, x), None
            rq, fq = bench(ref_step, lambda: [fresh(x_ref) for _ in range(B)], False), bench(fus_step, lambda: fresh(x_fus), False)
            if chain:
                rc, fc = bench(ref_step, lambda: [fresh(x_ref) for _ in range(B)], True), bench(fus_step, lambda: fresh(x_fus), True)
                tail = "| %9.1f %9.1f %5.2fx" % (rc, fc, rc / fc)
            else:
                tail = "|"
            print("%-4s %-4s %-2d | %9.1f %9.1f %5.2fx %s" % (stage, kind, B, rq, fq, rq / fq, tail), flush=True)
