"""Row invariance probe: for each op of a decode step, is a C-row batch (C,1,...) bitwise equal row by row to C separate
B=1 calls, and which formulation makes it so. Prints one line per (op, formulation, C)."""
import os, sys
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

P = os.path.expanduser("~/loopling/models/loopling-p3")
model, tok = load(P)
TM = model.model.language_model
mx.random.seed(0)
CS = [2, 4, 8]


def bits(a):
    return a.view({mx.float32: mx.uint32, mx.bfloat16: mx.uint16, mx.float16: mx.uint16}.get(a.dtype, a.dtype))


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.all(bits(a) == bits(b)).item())


def check(name, f_one, f_batch, make_x, Cs=CS):
    res = []
    for C in Cs:
        xs = make_x(C)
        one = mx.concatenate([f_one(x[i:i + 1]) if not isinstance(x, tuple) else f_one(*(t[i:i + 1] for t in x)) for i in range(C)]
                             if False else [f_one(*(t[i:i + 1] for t in xs)) for i in range(C)], axis=0)
        bat = f_batch(*xs)
        mx.eval(one, bat)
        ok = same(one, bat)
        if not ok and one.shape == bat.shape:
            d = mx.max(mx.abs(one.astype(mx.float32) - bat.astype(mx.float32))).item()
            res.append("C%d:DIFF(%.3g)" % (C, d))
        else:
            res.append("C%d:%s" % (C, "ok" if ok else "SHAPE%s/%s" % (one.shape, bat.shape)))
    print("%-34s %s" % (name, " ".join(res)), flush=True)


bf = mx.bfloat16
D = 2304
rnd = lambda *s, dt=bf: (mx.random.normal(s) * 1.0).astype(dt)

L5 = TM.layers[5]       # full attention, prelude
L7 = TM.layers[7]       # sliding, loop core (message gates)
a7 = L7.self_attn

# 1. 8-bit QuantizedLinear (q_proj 3072x2304)
ql = a7.q_proj
QL_DONE=True
if not QL_DONE: check("qlinear plain", lambda x: ql(x), lambda x: ql(x), lambda C: (rnd(C, 1, D),))


def ql_bcast(m, x):
    C = x.shape[0]
    bc = lambda a: None if a is None else mx.broadcast_to(a[None], (C,) + a.shape)
    return mx.quantized_matmul(x, bc(m["weight"]), scales=bc(m["scales"]), biases=bc(m.get("biases")), transpose=True,
                               group_size=m.group_size, bits=m.bits, mode=m.mode)


if not QL_DONE:
    check("qlinear bcast-batch", lambda x: ql(x), lambda x: ql_bcast(ql, x), lambda C: (rnd(C, 1, D),))
    check("qlinear o_proj bcast", lambda x: a7.o_proj(x), lambda x: ql_bcast(a7.o_proj, x), lambda C: (rnd(C, 1, 3072),))
    check("qlinear sink bcast", lambda x: a7.sink_proj(x), lambda x: ql_bcast(a7.sink_proj, x), lambda C: (rnd(C, 1, D),))
    check("lm_head bcast", lambda x: model.lm_head(x), lambda x: ql_bcast(model.lm_head, x), lambda C: (rnd(C, 1, D),), Cs=[2, 8])

# 2. float / bf16 Linear
router = L7.mlp.gate
W = router.weight  # f32 (128, 2304)
print("router weight", W.dtype, W.shape, "| msg down", L7.q_message_gate.down_proj.weight.dtype, L7.q_message_gate.down_proj.weight.shape,
      "| forget", L7.q_message_gate.forget_gate_proj.weight.dtype, "| wproj", L5.self_attn.k_sconv.weight_proj.weight.dtype,
      L5.self_attn.k_sconv.weight_proj.weight.shape, flush=True)
_mat = {}


def mat_w(Wt, C):
    k = (id(Wt), C)
    if k not in _mat:
        _mat[k] = mx.contiguous(mx.broadcast_to(Wt[None], (C,) + Wt.shape))
        mx.eval(_mat[k])
    return _mat[k]


if not QL_DONE:
    for nm, Wt, dt, K in [("router f32", W, mx.float32, D), ("msg down bf16", L7.q_message_gate.down_proj.weight, bf, 2 * D),
                          ("msg forget f32", L7.q_message_gate.forget_gate_proj.weight, mx.float32, 64)]:
        xdt = mx.float32 if Wt.dtype == mx.float32 else bf
        mk = lambda C, K=K, xdt=xdt: (rnd(C, 1, K, dt=xdt),)
        check(nm + " plain", lambda x, Wt=Wt: x @ Wt.T, lambda x, Wt=Wt: x @ Wt.T, mk)
        check(nm + " materialized", lambda x, Wt=Wt: x @ Wt.T, lambda x, Wt=Wt: x @ mat_w(Wt, x.shape[0]).transpose(0, 2, 1), mk)
        check(nm + " gather_mm", lambda x, Wt=Wt: x @ Wt.T,
              lambda x, Wt=Wt: mx.gather_mm(x, Wt.T[None], rhs_indices=mx.zeros((x.shape[0], 1), mx.uint32)), mk)
        check(nm + " addmm0", lambda x, Wt=Wt: x @ Wt.T,
              lambda x, Wt=Wt: mx.concatenate([x[i:i + 1] @ Wt.T for i in range(x.shape[0])]), mk)

# 3. relative logits (C,1,24,16) @ (16,1024)
rl = a7.rel_logits_proj
check("rel_logits plain", lambda x: rl(x), lambda x: rl(x), lambda C: (rnd(C, 1, 24, 16),))

# 4. norms
from mlx_lm.models.inkling2.layers import unweighted_rms_norm, _depth_attention
check("unweighted_rms_norm f32", lambda x: unweighted_rms_norm(x), lambda x: unweighted_rms_norm(x), lambda C: (rnd(C, 1, D, dt=mx.float32),))
check("RMSNorm bf16-out", lambda x: L7.post_attention_layernorm(x), lambda x: L7.post_attention_layernorm(x), lambda C: (rnd(C, 1, D, dt=mx.float32),))
check("rms(value) heads", lambda x: unweighted_rms_norm(x, 1e-6), lambda x: unweighted_rms_norm(x, 1e-6), lambda C: (rnd(C, 1, 8, 128, dt=mx.float32),))
check("q_l2_clip", lambda x: a7.q_l2_clip(x), lambda x: a7.q_l2_clip(x), lambda C: (rnd(C, 1, 24, 128),))
check("k_l2_clip", lambda x: a7.k_l2_clip(x), lambda x: a7.k_l2_clip(x), lambda C: (rnd(C, 1, 8, 128),))

# 5. depth attention: block_states (nb, C, 1, D) f32, partial (C,1,D)
q3 = L7.attention_input.query.reshape(-1, D)
check("depth_attn (attn input, 3q)", lambda bs, p: _depth_attention(bs.transpose(1, 0, 2, 3)[:, 0:1][:, :, :, :].transpose(1, 0, 2, 3) if False else bs, p, q3),
      lambda bs, p: _depth_attention(bs, p, q3), lambda C: (rnd(C, 2, 1, D, dt=mx.float32), rnd(C, 1, D, dt=mx.float32)), Cs=[])
# block states are stacked on axis 0 (nb, B, L, D): give f_one/f_batch the batch-major tensor and swap inside


def da(bsT, p, q):
    return _depth_attention(bsT.transpose(1, 0, 2, 3), p, q)


check("depth_attn 3q", lambda b, p: da(b, p, q3), lambda b, p: da(b, p, q3), lambda C: (rnd(C, 2, 1, D, dt=mx.float32), rnd(C, 1, D, dt=mx.float32)))
check("depth_attn 1q", lambda b, p: da(b, p, L7.mlp_input.query.reshape(-1, D)), lambda b, p: da(b, p, L7.mlp_input.query.reshape(-1, D)),
      lambda C: (rnd(C, 3, 1, D, dt=mx.float32), rnd(C, 1, D, dt=mx.float32)))

# 6. router kernel + experts
check("router", lambda x: mx.concatenate(router(x), axis=-1) if False else router(x)[0].astype(mx.float32) + 0 * router(x)[1],
      lambda x: router(x)[0].astype(mx.float32) + 0 * router(x)[1], lambda C: (rnd(C, 1, D),))
check("router weights", lambda x: router(x)[1], lambda x: router(x)[1], lambda C: (rnd(C, 1, D),))
ex = L7.mlp.experts
idx = lambda C: mx.stack([mx.random.permutation(128)[:8] for _ in range(C)]).reshape(C, 1, 8).astype(mx.uint32)


def experts_nosort(x, ind):
    x = mx.expand_dims(x, (-2, -3))
    g = ex.gate_up_proj(x, ind, sorted_indices=False)
    y = ex.act_fn(g[..., 0::2], g[..., 1::2])
    return ex.down_proj(y, ind, sorted_indices=False).squeeze(-2)


check("experts default", lambda x, i: ex(x, i), lambda x, i: ex(x, i), lambda C: (rnd(C, 1, D), idx(C)))
check("experts nosort", lambda x, i: ex(x, i), lambda x, i: experts_nosort(x, i), lambda C: (rnd(C, 1, D), idx(C)))
L23 = TM.layers[23]
ex23 = L23.mlp.experts
check("experts L23 mxfp8 default", lambda x, i: ex23(x, i), lambda x, i: ex23(x, i), lambda C: (rnd(C, 1, D), idx(C)))

# 7. message gate whole (with materialized weights later); dynamic conv sum
from mlx_lm.models.inkling2 import fastdecode as FD
sc = L5.self_attn.k_sconv
check("dyn wproj qlinear bcast", lambda x: sc.weight_proj(x), lambda x: ql_bcast(sc.weight_proj, x), lambda C: (rnd(C, 1, D),))
check("dyn conv (plain)", lambda x, d, s: FD._dsc(sc, x, d, s)[0], lambda x, d, s: FD._dsc(sc, x, d, s)[0],
      lambda C: (rnd(C, 1, 1024), rnd(C, 1, D), rnd(C, 3, 1024, dt=mx.float32)))
# softcap + argmax
check("softcap", lambda x: mx.tanh(x / 30.0) * 30.0, lambda x: mx.tanh(x / 30.0) * 30.0, lambda C: (rnd(C, 1, 4096, dt=mx.float32),))
check("argmax 201k", lambda x: mx.argmax(x, axis=-1), lambda x: mx.argmax(x, axis=-1), lambda C: (rnd(C, 1, 200058, dt=mx.float32),))
