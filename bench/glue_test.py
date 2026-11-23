"""Lane e5: bitwise tests of vendor/mlx_lm/models/inkling2/rows/glue.py against MLX evaluated on each row alone (decode shapes).

Run (512S2):  LANE=e5 RHOST=512S2 scripts/r --gpu 'python bench/glue_test.py'
Prints one PASS/FAIL line per case; exit status 1 if anything differs.  Only "required" cases decide the status; cases
named "probe:" print which contraction / variant matches MLX and never fail the run.
"""

import math
import os
import sys
import time

import mlx.core as mx
import numpy as np

from mlx_lm import load
from mlx_lm.models import cache as C
from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import kernels as K
from mlx_lm.models.inkling2 import layers as L
from mlx_lm.models.inkling2.rows import glue as G

BS = [1, 2, 3, 5, 8, 16]
RESULTS = []
mx.random.seed(0)


def bits(a):
    a = mx.array(a) if not isinstance(a, mx.array) else a
    if a.dtype in (mx.float32, mx.uint32, mx.int32):
        return np.array(a.view(mx.uint32))
    if a.dtype in (mx.bfloat16, mx.float16, mx.uint16):
        return np.array(a.view(mx.uint16))
    return np.array(a)


def check(name, ref_rows, mine, required=True):
    """ref_rows: list of B arrays (each the reference on one row); mine: array with leading dim B (or list)."""
    mx.eval(ref_rows, mine)
    B = len(ref_rows)
    bad = 0
    n = 0
    for b in range(B):
        r = ref_rows[b]
        m = mine[b]
        if r.shape != m.shape:
            m = m.reshape(r.shape)
        if r.dtype != m.dtype:
            bad += r.size
            n += r.size
            continue
        d = bits(r) != bits(m)
        bad += int(d.sum())
        n += d.size
    ok = bad == 0
    RESULTS.append((name, ok, required))
    print("%-4s %s%-62s elems=%-9d mismatches=%d" % ("PASS" if ok else "FAIL", "" if required else "probe: ", name, n, bad), flush=True)
    return ok


def rnd(shape, dtype=mx.float32, scale=1.0):
    return (mx.random.normal(shape) * scale).astype(dtype)


# ----------------------------------------------------------------------------------------------------- model + capture
P = os.path.expanduser("~/loopling/models/loopling-p3")
model, tok = load(P)
TM = model.model.language_model
args = model.args.text if hasattr(model, "args") else None
cfg_eps = 1e-6
captured = {}  # id(layer) -> list of (h, messages)
steps_raw = []  # raw bf16 lm_head outputs (1, Vp) per step
orig_decode_layer = FD.decode_layer


def hook(layer, h, bs, cache, messages):
    captured.setdefault(id(layer), []).append((h, messages))
    return orig_decode_layer(layer, h, bs, cache, messages)


FD.decode_layer = hook
ids = tok.encode("Write a Python function that parses an ISO-8601 duration like P3DT4H12M into seconds, then explain it.")
cache = C.make_prompt_cache(model)
out = model(mx.array(ids)[None], cache=cache)
y = mx.argmax(out[:, -1, :], axis=-1)
NSTEP = 20
for s in range(NSTEP):
    hid = TM(y[None], cache=cache)
    raw = model.lm_head(hid)
    steps_raw.append(raw[0, -1][None])  # (1, Vp) bf16
    lg = mx.tanh(raw.astype(mx.float32) / 30.0) * 30.0
    y = mx.argmax(lg[:, -1, :200058], axis=-1)
    mx.eval(y, raw)
FD.decode_layer = orig_decode_layer
print("captured", {i: len(v) for i, v in list(captured.items())[:3]}, "steps", len(steps_raw), flush=True)
LAY = TM.layers


def real_h(layer, n, msg=False):
    """n real bf16 activations (1,1,2304) captured at this layer (with messages if msg)."""
    es = [e for e in captured[id(layer)] if (e[1] is not None) == msg]
    return [es[i % len(es)] for i in range(n)]


# ============================================================================================ (a) rms norm
def rms_cases():
    L1, L5, L7 = LAY[1], LAY[5], LAY[7]
    cases = []
    # name, x_dtype, weight (None/array), eps
    cases.append(("unweighted f32 N=2304", mx.float32, None))
    cases.append(("weighted w=bf16 x=f32 (input/post_attn layernorm)", mx.float32, L1.input_layernorm.weight))
    cases.append(("weighted w=bf16 x=bf16 (layer-0 style)", mx.bfloat16, L1.input_layernorm.weight))
    cases.append(("weighted w=bf16 x=f32 post_attention_layernorm", mx.float32, L7.post_attention_layernorm.weight))
    cases.append(("weighted w=bf16 x=f32 q_input_layernorm (full)", mx.float32, L5.q_input_layernorm.weight))
    cases.append(("weighted w=bf16 x=f32 final norm", mx.float32, TM.norm.weight))
    cases.append(("weighted w=bf16 x=bf16 embed_norm", mx.bfloat16, TM.embed_norm.weight))
    for name, xdt, w in cases:
        for src in ("random", "real"):
            for B in BS:
                if src == "random":
                    xs = [(rnd((1, 1, 2304), mx.float32, 3.0) + 0.5).astype(xdt) for _ in range(B)]
                else:
                    xs = [h.astype(xdt) for h, _ in real_h(L7, B)]
                eps = cfg_eps
                odt = mx.float32 if (xdt == mx.float32) else mx.bfloat16
                if w is None:
                    ref = [mx.fast.rms_norm(x.astype(mx.float32), None, eps) for x in xs]
                else:
                    ref = [mx.fast.rms_norm(x, w, eps) for x in xs]
                    odt = ref[0].dtype
                X = mx.concatenate([x.reshape(1, 2304) for x in xs])
                ref = [r.reshape(2304) for r in ref]
                for mode, TG in (("tg", 64), ("tg", 256), ("tg", 576), ("sg", 32)):
                    for rf in (True, False):
                        out = G.rms_norm_rows(X, w, eps, out_dtype=odt, TG=TG, mode=mode, rf=rf)
                        ok = check("rms %s | %s | B=%d TG=%d %s rf=%d" % (name, src, B, TG, mode, rf), ref, out, required=rf)
    # value-head rms: (1,1,8,128) f32 -> cast back to bf16
    L1a = L1.self_attn
    for B in BS:
        xs = [a for a in (rnd((1, 1, 8, 128), mx.bfloat16, 2.0) for _ in range(B))]
        ref = [L.unweighted_rms_norm(x.astype(mx.float32), cfg_eps).astype(mx.bfloat16).reshape(8, 128) for x in xs]
        X = mx.concatenate([x.reshape(8, 128) for x in xs])
        out = G.rms_norm_rows(X, None, cfg_eps, out_dtype=mx.float32, mode="sg", rf=True).astype(mx.bfloat16)
        check("rms value-head N=128 f32->bf16 | B=%d (8 rows each)" % B, ref, out.reshape(B, 8, 128))


# ============================================================================================ (b) l2 clip
def l2_cases():
    L7 = LAY[7]
    a = L7.self_attn
    thr_qk = float(a.q_l2_clip.threshold) if hasattr(a.q_l2_clip, "threshold") else 22.627416997969522
    thr_rel = float(a.rel_l2_clip.threshold) if hasattr(a.rel_l2_clip, "threshold") else 8.0
    print("l2 thresholds", thr_qk, thr_rel, flush=True)
    for src in ("random", "real"):
        for B in BS:
            hs = [h for h, _ in real_h(L7, B)]
            qin = [L7.input_layernorm(h).astype(mx.bfloat16) for h in hs]
            if src == "real":
                qs = [a.q_proj(x).reshape(1, 1, 24, 128) for x in qin]
                ks = [a.k_proj(x).reshape(1, 1, 8, 128) for x in qin]
                rs = [a.r_proj(x).reshape(1, 1, 24, 16) for x in qin]
            else:
                qs = [rnd((1, 1, 24, 128), mx.bfloat16, 20.0) for _ in range(B)]
                ks = [rnd((1, 1, 8, 128), mx.bfloat16, 20.0) for _ in range(B)]
                rs = [rnd((1, 1, 24, 16), mx.bfloat16, 8.0) for _ in range(B)]
            sc_q = a.q_l2_clip.scale
            sc_r = a.rel_l2_clip.scale
            ref = [a.q_l2_clip(q) for q in qs]
            out = G.l2_clip_rows(mx.concatenate([q.reshape(24, 128) for q in qs]), sc_q, thr_qk, nrows_per_b=24)
            check("l2 q  (24x128, scale) %s | B=%d" % (src, B), [r.reshape(24, 128) for r in ref], out.reshape(B, 24, 128))
            ref = [a.k_l2_clip(k) for k in ks]
            out = G.l2_clip_rows(mx.concatenate([k.reshape(8, 128) for k in ks]), None, thr_qk, nrows_per_b=8)
            check("l2 k  (8x128, no scale) %s | B=%d" % (src, B), [r.reshape(8, 128) for r in ref], out.reshape(B, 8, 128))
            ref = [a.rel_l2_clip(r) for r in rs]
            out = G.l2_clip_rows(mx.concatenate([r.reshape(24, 16) for r in rs]), sc_r, thr_rel, nrows_per_b=24)
            check("l2 rel (24x16, scale) %s | B=%d" % (src, B), [r.reshape(24, 16) for r in ref], out.reshape(B, 24, 16))
            # the sum-of-squares order matters most when many elements are large: also huge-dynamic-range inputs
    # saturating inputs (norm well above threshold) and tiny ones
    for B in (1, 8):
        qs = [rnd((1, 1, 24, 128), mx.bfloat16, 300.0) * (10.0 ** (i % 3 - 1)) for i in range(B)]
        qs = [q.astype(mx.bfloat16) for q in qs]
        ref = [a.q_l2_clip(q) for q in qs]
        out = G.l2_clip_rows(mx.concatenate([q.reshape(24, 128) for q in qs]), a.q_l2_clip.scale, thr_qk, nrows_per_b=24)
        check("l2 q wide-range | B=%d" % B, [r.reshape(24, 128) for r in ref], out.reshape(B, 24, 128))


# ============================================================================================ (c) unary / conversions
def unary_cases():
    n = 1 << 20
    x = mx.random.uniform(-40, 40, (n,))
    xs = mx.random.uniform(-6, 6, (n,))
    xp = mx.random.uniform(1e-8, 1e4, (n,))
    specials = mx.array([0.0, -0.0, 1e-30, -1e-30, 88.0, -88.0, 100.0, -100.0, 1e38, -1e38, float("inf"), -float("inf"), float("nan")], dtype=mx.float32)
    for tag, v in (("U[-40,40]", x), ("U[-6,6]", xs), ("specials", mx.concatenate([specials, x[:243]]))):
        for nm, ex, rf in (
            ("sigmoid", "glue_sigmoid(v_a)", mx.sigmoid),
            ("tanh", "glue_tanh(v_a)", mx.tanh),
            ("exp", "glue_exp(v_a)", mx.exp),
            ("sqrt(|x|)", "glue_sqrt(metal::abs(v_a))", lambda t: mx.sqrt(mx.abs(t))),
            ("tanh(x/4)*4", "glue_tanh(v_a / 4.0f) * 4.0f", lambda t: mx.tanh(t / 4.0) * 4.0),
            ("tanh(x/25)*25", "glue_tanh(v_a / 25.0f) * 25.0f", lambda t: mx.tanh(t / 25.0) * 25.0),
            ("tanh(x/30)*30", "glue_tanh(v_a / 30.0f) * 30.0f", lambda t: mx.tanh(t / 30.0) * 30.0),
        ):
            out = G.elementwise("u_" + nm.replace("/", "_").replace("*", "x").replace("(", "").replace(")", "").replace("|", ""), ex, {"a": v}, mx.float32)
            check("unary %s %s" % (nm, tag), [rf(v)], out[None])
    lg = mx.abs(xp)
    check("unary log x>0", [mx.log(lg)], G.elementwise("log", "glue_log(v_a)", {"a": lg}, mx.float32)[None])
    # f32 <-> bf16 over random bit patterns (includes NaN/inf/subnormal/ties)
    rb = mx.random.randint(0, 2**31 - 1, (1 << 22,)).astype(mx.uint32) * mx.array(2, mx.uint32) + mx.random.randint(0, 2, (1 << 22,)).astype(mx.uint32)
    f = rb.view(mx.float32)
    b16, back = G.convert_roundtrip(rb)
    ref16 = f.astype(mx.bfloat16)
    check("f32->bf16 random bit patterns", [ref16.view(mx.uint16)], b16[None])
    check("bf16->f32 (of the above)", [ref16.astype(mx.float32).view(mx.uint32)], back[None])


# ============================================================================================ (d) moe combine
def moe_cases():
    L7 = LAY[7]
    for src in ("random", "real"):
        for B in BS:
            if src == "random":
                es = [rnd((1, 1, 8, 2304), mx.bfloat16, 2.0) for _ in range(B)]
                ws = [mx.abs(rnd((1, 1, 8), mx.float32, 1.0)) * 4 for _ in range(B)]
            else:
                es, ws = [], []
                for h, _ in real_h(L7, B):
                    x = L7.post_attention_layernorm(h.astype(mx.float32)).astype(mx.bfloat16)
                    inds, w = L7.mlp.gate(x)
                    es.append(L7.mlp.experts(x, inds))
                    ws.append(w)
                mx.eval(es, ws)
            comb = mx.compile(lambda e, w: (e.astype(mx.float32) * w[..., None]).sum(axis=-2).astype(mx.bfloat16))
            ref = [comb(e, w) for e, w in zip(es, ws)]
            eager = [(e.astype(mx.float32) * w[..., None]).sum(axis=-2).astype(mx.bfloat16) for e, w in zip(es, ws)]
            mine = G.moe_combine(mx.concatenate([e.reshape(1, 8, 2304) for e in es]), mx.concatenate([w.reshape(1, 8) for w in ws]))
            check("moe combine %s | B=%d vs compiled" % (src, B), [r.reshape(2304) for r in ref], mine)
            check("moe combine %s | B=%d vs eager" % (src, B), [r.reshape(2304) for r in eager], mine)


# ============================================================================================ (e) elementwise chains
def ew_cases():
    L7 = LAY[7]
    ex = L7.mlp.experts
    for src in ("random", "real"):
        for B in BS:
            if src == "random":
                gs = [rnd((1, 1, 8, 1152), mx.bfloat16, 3.0) for _ in range(B)]
                us = [rnd((1, 1, 8, 1152), mx.bfloat16, 8.0) for _ in range(B)]
            else:
                gs, us = [], []
                for h, _ in real_h(L7, B):
                    x = L7.post_attention_layernorm(h.astype(mx.float32)).astype(mx.bfloat16)
                    inds, _w = L7.mlp.gate(x)
                    gu = ex.gate_up_proj(mx.expand_dims(x, (-2, -3)), inds, sorted_indices=False)
                    gs.append(gu[..., 0::2].reshape(1, 1, 8, 1152))
                    us.append(gu[..., 1::2].reshape(1, 1, 8, 1152))
                mx.eval(gs, us)
            ref = [L._softcapped_swiglu(g, u, 4.0, 25.0) for g, u in zip(gs, us)]
            mine = G.swiglu(mx.concatenate([g.reshape(-1) for g in gs]), mx.concatenate([u.reshape(-1) for u in us]), 4.0, 25.0)
            check("swiglu %s | B=%d" % (src, B), [r.reshape(-1) for r in ref], mine.reshape(B, -1))
    # message-gate combine: probe the contraction variants against a compiled mini function, then the real module.
    for B in (1, 8):
        f, m, h, g = (rnd((B, 2304), mx.float32, s) for s in (1.0, 1.0, 3.0, 3.0))
        mini = mx.compile(lambda f, m, h, g: (1 - mx.tanh(f)) * h + mx.tanh(m) * g)
        ref = [mini(f[i], m[i], h[i], g[i]) for i in range(B)]
        eager = [(1 - mx.tanh(f[i])) * h[i] + mx.tanh(m[i]) * g[i] for i in range(B)]
        for mv in (0, 1, 2):
            out = G.msg_combine(f.reshape(-1), m.reshape(-1), h.reshape(-1), g.reshape(-1), mv)
            check("probe: msg combine variant %d vs compiled mini | B=%d" % (mv, B), ref, out.reshape(B, -1), required=False)
            check("probe: msg combine variant %d vs eager | B=%d" % (mv, B), eager, out.reshape(B, -1), required=False)
    # real MessageGate modules (compiled, as the fast-decode segment runs them), loop-core layer 7 and 11, q/k/v/mlp gates
    for lay_i in (7, 11):
        lay = LAY[lay_i]
        for gname in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
            mg = getattr(lay, gname)
            for B in BS:
                es = real_h(lay, B, msg=True)
                hs = [h.astype(mx.float32) for h, _ in es]
                ms = [mm[..., 0, :] if mm.ndim == 3 else mm[..., 0, :] for _, mm in es]
                mi = {"q_message_gate": 0, "k_message_gate": 1, "v_message_gate": 2, "mlp_message_gate": 3}[gname]
                ms = [mm[..., mi, :] for _, mm in es]
                cfn = mx.compile(lambda h, m: mg(h, m))
                ref = [cfn(h, m) for h, m in zip(hs, ms)]
                efn = [mg(h, m) for h, m in zip(hs, ms)]
                fs, mls = [], []
                for h, m in zip(hs, ms):
                    st = mx.concatenate([L.unweighted_rms_norm(h), L.unweighted_rms_norm(m)], axis=-1)
                    st = mg.down_proj(st.astype(mg.down_proj.weight.dtype)).astype(mx.float32)
                    fs.append(mg.forget_gate_proj(st))
                    mls.append(mg.message_gate_proj(st))
                mx.eval(fs, mls)
                cat = lambda xs: mx.concatenate([x.reshape(1, -1) for x in xs]).reshape(-1)
                for mv in (0, 1, 2):
                    out = G.msg_combine(cat(fs), cat(mls), cat(hs), cat(ms), mv)
                    ok = check("probe: msg gate L%d.%s variant %d vs compiled module | B=%d" % (lay_i, gname, mv, B), [r.reshape(-1) for r in ref], out.reshape(B, -1), required=False)
    # gate: attn * sigmoid(gate) -> bf16 ; tau ; residual adds
    L5 = LAY[5]
    a5 = L5.self_attn
    for B in BS:
        hs = [h for h, _ in real_h(L5, B)]
        gates = [a5.gate_proj(L5.q_input_layernorm(h.astype(mx.float32)).astype(mx.bfloat16)) for h in hs]
        attn = [rnd((1, 1, 24, 128), mx.bfloat16, 1.0) for _ in range(B)]
        mx.eval(gates, attn)
        gfn = mx.compile(lambda a, g: (a.reshape(1, 1, -1).astype(mx.float32) * mx.sigmoid(g.astype(mx.float32))).astype(g.dtype))
        ref = [gfn(a, g) for a, g in zip(attn, gates)]
        eager = [(a.reshape(1, 1, -1).astype(mx.float32) * mx.sigmoid(g.astype(mx.float32))).astype(g.dtype) for a, g in zip(attn, gates)]
        mine = G.elementwise("gate", "glue_gate(v_a, v_g)", {"a": mx.concatenate([a.reshape(-1) for a in attn]), "g": mx.concatenate([g.reshape(-1) for g in gates])}, mx.bfloat16)
        check("gate attn*sigmoid(gate) vs compiled | B=%d" % B, [r.reshape(-1) for r in ref], mine.reshape(B, -1))
        check("gate attn*sigmoid(gate) vs eager | B=%d" % B, [r.reshape(-1) for r in eager], mine.reshape(B, -1))
        # tau scaling: positions differ per row
        taus = [FD._full_tau(L5, 40 + 997 * i + 8192 * (i % 2)) for i in range(B)]
        qs = [rnd((1, 1, 24, 128), mx.bfloat16, 5.0) for _ in range(B)]
        mx.eval(taus, qs)
        tfn = mx.compile(lambda q, t: (q.astype(mx.float32) * t).astype(q.dtype))
        ref = [tfn(q, t) for q, t in zip(qs, taus)]
        eager = [(q.astype(mx.float32) * t).astype(q.dtype) for q, t in zip(qs, taus)]
        tv = mx.concatenate([mx.broadcast_to(t.reshape(1), (24 * 128,)) for t in taus])
        mine = G.elementwise("tau", "glue_tau(v_a, v_t)", {"a": mx.concatenate([q.reshape(-1) for q in qs]), "t": tv}, mx.bfloat16)
        check("tau scaling vs compiled | B=%d" % B, [r.reshape(-1) for r in ref], mine.reshape(B, -1))
        check("tau scaling vs eager | B=%d" % B, [r.reshape(-1) for r in eager], mine.reshape(B, -1))
        # residual: (residual_f32 + hidden.f32).astype(bf16), and h.f32 + hidden
        r_ = [rnd((1, 1, 2304), mx.float32, 30.0) for _ in range(B)]
        hh = [rnd((1, 1, 2304), mx.bfloat16, 30.0) for _ in range(B)]
        mx.eval(r_, hh)
        rfn = mx.compile(lambda r, h: (r + h.astype(mx.float32)).astype(mx.bfloat16))
        ref = [rfn(r, h) for r, h in zip(r_, hh)]
        mine = G.elementwise("resid", "v_a + v_b", {"a": mx.concatenate([r.reshape(-1) for r in r_]), "b": mx.concatenate([h.reshape(-1) for h in hh])}, mx.bfloat16)
        check("residual add f32+bf16 -> bf16 | B=%d" % B, [r.reshape(-1) for r in ref], mine.reshape(B, -1))


# ============================================================================================ (f) existing custom kernels
def custom_cases():
    from mlx_lm.kernels.expert_routing import expert_routing, _ROUTING_SOURCE
    from mlx_lm.kernels.common import scalar, row_threads

    L7 = LAY[7]
    # short conv rows kernel vs the reference ShortConvolution (and vs the B=1 kernel) on real weights
    for lay_i, cname in ((1, "post_attention_convolution"), (7, "post_mlp_convolution"), (1, "self_attn.k_sconv")):
        mod = LAY[lay_i]
        for part in cname.split("."):
            mod = getattr(mod, part)
        Cn = mod.conv1d.weight.shape[0]
        for B in BS:
            xs = [rnd((1, 1, Cn), mx.bfloat16, 2.0) for _ in range(B)]
            sts = [rnd((1, 3, Cn), mx.float32, 2.0).astype(mx.bfloat16).astype(mx.float32) for _ in range(B)]
            ref = [mod(x, st, None) for x, st in zip(xs, sts)]
            one = [FD.short_conv_decode(mod, x, st) for x, st in zip(xs, sts)]
            rows = FD.short_conv_decode(mod, mx.concatenate(xs), mx.concatenate(sts))
            check("conv %s L%d out rows-kernel vs reference | B=%d" % (cname, lay_i, B), [r[0] for r in ref], rows[0])
            check("conv %s L%d out rows-kernel vs B=1 kernel | B=%d" % (cname, lay_i, B), [r[0] for r in one], rows[0])
            check("conv %s L%d state rows-kernel vs B=1 kernel | B=%d" % (cname, lay_i, B), [r[1] for r in one], rows[1])
    # expert routing: B rows in one launch vs each alone, and vs the pure-MLX fallback
    for B in BS:
        lg = [rnd((1, 128), mx.float32, 2.0) for _ in range(B)]
        bias = L7.mlp.gate.e_score_correction_bias
        kw = dict(top_k=8, multiplier=1.0, score_eps=L._ROUTER_SCORE_EPS, route_scale=8.0)
        one = [expert_routing(l, bias, **kw) for l in lg]
        both = expert_routing(mx.concatenate(lg), bias, **kw)
        check("routing inds B rows vs alone | B=%d" % B, [o[0] for o in one], both[0])
        check("routing weights B rows vs alone | B=%d" % B, [o[1] for o in one], both[1])
    # depth attention: B rows vs alone, Q in {1,3,4}
    for Q, qn in ((1, "mlp_input"), (3, "attention_input L5"), (4, "message_input")):
        for B in BS:
            D = 2304
            for Kn in (1, 5):
                blocks = rnd((Kn, B, D), mx.float32, 1.0)
                part = rnd((B, D), mx.float32, 1.0)
                q = rnd((Q, D), mx.float32, 1.0)
                sc = D ** -0.5
                rows = K.depth_attention(blocks, part, q, scale=sc)
                one = [K.depth_attention(blocks[:, i : i + 1], part[i : i + 1], q, scale=sc) for i in range(B)]
                check("depth_attention Q=%d K=%d rows vs alone | B=%d" % (Q, Kn, B), [o.reshape(Q, D) for o in one], rows.reshape(B, Q, D))
    # inlining: same sources compiled WITH the glue header in front must give the same bits
    for B in (1, 8):
        D, Q, Kn = 2304, 4, 5
        blocks, part, q = rnd((Kn, B, D)), rnd((B, D)), rnd((Q, D))
        ref = K.depth_attention(blocks, part, q, scale=D ** -0.5)
        kern = mx.fast.metal_kernel(name="depth_attention_g", input_names=["blocks", "partial", "queries", "scale"], output_names=["out"], source=K._DEPTH_ATTENTION_SOURCE, header=G.GLUE_HEADER)
        TG = row_threads(D)
        out = kern(inputs=[blocks, part, q, scalar(D ** -0.5)], template=[("K", Kn), ("D", D), ("Q", Q), ("TG", TG)], grid=(B * TG, 1, 1), threadgroup=(TG, 1, 1), output_shapes=[(B, Q, D)], output_dtypes=[mx.float32])[0]
        check("inline depth_attention with glue header | B=%d" % B, [ref[i] for i in range(B)], out)
        lg = rnd((B, 128), mx.float32, 2.0)
        bias = L7.mlp.gate.e_score_correction_bias
        kw = dict(top_k=8, multiplier=1.0, score_eps=L._ROUTER_SCORE_EPS, route_scale=8.0)
        ref = expert_routing(lg, bias, **kw)
        kern = mx.fast.metal_kernel(name="expert_routing_g", input_names=["logits", "bias", "multiplier", "eps", "scale"], output_names=["inds", "weights"], source=_ROUTING_SOURCE, header=G.GLUE_HEADER)
        out = kern(inputs=[lg, bias, scalar(1.0), scalar(L._ROUTER_SCORE_EPS), scalar(8.0)], template=[("E", 128), ("K", 8)], grid=(B * 32, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(B, 8)] * 2, output_dtypes=[mx.uint32, mx.float32])
        check("inline expert_routing with glue header (inds) | B=%d" % B, [ref[0][i] for i in range(B)], out[0])
        check("inline expert_routing with glue header (weights) | B=%d" % B, [ref[1][i] for i in range(B)], out[1])


# ============================================================================================ (g) head
def head_cases():
    V, Vp = 200058, 201024
    raws = [s.astype(mx.float32) for s in steps_raw]  # (1, Vp) f32 raw lm_head
    mx.eval(raws)
    for B in BS:
        xs = [raws[i % len(raws)] for i in range(B)]
        ref_lg = [mx.tanh(x[None] / 30.0) * 30.0 for x in xs]
        ref_lg = [r[..., :V][:, -1, :] for r in ref_lg]
        mine = G.softcap_rows(mx.concatenate(xs), 30.0, V)
        check("softcap tanh(x/30)*30 + slice | B=%d" % B, [r.reshape(V) for r in ref_lg], mine)
        lse = [mx.logsumexp(r, keepdims=True) for r in ref_lg]
        lp = [r - l for r, l in zip(ref_lg, lse)]
        am = [mx.argmax(p, axis=-1) for p in lp]
        LG = mx.concatenate(ref_lg)
        for TG in (256, 1024):
            l_o, i_o = G.head_rows(LG, TG=TG)
            check("logsumexp (real logits) | B=%d TG=%d" % (B, TG), [l.reshape(()) for l in lse], l_o)
            check("argmax(logprobs) (real logits) | B=%d TG=%d" % (B, TG), [a.reshape(()) for a in am], i_o)
    # adversarial: ties, near ties after the lse subtraction, NaN, -inf, huge logits, all equal
    mx.random.seed(7)
    rows = []
    base = mx.random.normal((V,)) * 5
    r = base
    r = mx.where(mx.arange(V) % 9973 == 5, mx.max(r) + 1.0, r)  # many equal maxima
    rows.append(r)
    r2 = mx.full((V,), 3.25)  # all equal
    rows.append(r2)
    r3 = base + mx.where(mx.arange(V) == 150000, float("nan"), 0.0)
    r3 = mx.where(mx.arange(V) == 3, float("nan"), r3)
    rows.append(r3)
    r4 = mx.where(mx.arange(V) % 2 == 0, -float("inf"), base)
    rows.append(r4)
    r5 = mx.full((V,), -float("inf"))
    rows.append(r5)
    r6 = (base * 1e-4).astype(mx.bfloat16).astype(mx.float32) + 11.0  # coarse values: lots of exact ties; lse subtraction can merge neighbours
    rows.append(r6)
    r7 = mx.round(base * 4096.0) / 4096.0 + 1e3
    rows.append(r7)
    r8 = base.at[199999].add(60.0)
    rows.append(r8)
    for i, r in enumerate(rows):
        r = r[None]
        lse = mx.logsumexp(r, keepdims=True)
        lp = r - lse
        am = mx.argmax(lp, axis=-1)
        for TG in (256, 1024):
            l_o, i_o = G.head_rows(r, TG=TG)
            check("adversarial row %d logsumexp TG=%d" % (i, TG), [lse.reshape(())], l_o)
            check("adversarial row %d argmax TG=%d" % (i, TG), [am.reshape(())], i_o)
    # raw argmax (no subtraction) tie rule
    for i, r in enumerate(rows):
        am = mx.argmax(r[None], axis=-1)
        _, i_o = G.head_rows(r[None], TG=256, do_sub=False)
        check("argmax on raw row %d" % i, [am.reshape(())], i_o)


if __name__ == "__main__":
    which = sys.argv[1:] or ["rms", "l2", "unary", "moe", "ew", "custom", "head"]
    fns = dict(rms=rms_cases, l2=l2_cases, unary=unary_cases, moe=moe_cases, ew=ew_cases, custom=custom_cases, head=head_cases)
    t0 = time.time()
    for w in which:
        print("=====", w, flush=True)
        fns[w]()
    req = [r for r in RESULTS if r[2]]
    fails = [r for r in req if not r[1]]
    print("\nSUMMARY: %d required cases, %d FAIL; %d probes (%d match); %.0fs" % (len(req), len(fails), len(RESULTS) - len(req), sum(1 for r in RESULTS if not r[2] and r[1]), time.time() - t0))
    for f in fails:
        print("  FAIL", f[0])
    sys.exit(1 if fails else 0)
