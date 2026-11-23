"""Exactness test for rows/gemv.py: per-row bit compare against the reference op on that row alone (B=1 decode shapes).
Real Loopling-p3 weights, all layers; random inputs at realistic scale and activations captured from a short decode."""
import os, sys
os.environ["LOOPLING_COMPILE_DECODE"] = "0"
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.inkling2 import layers as LY
from mlx_lm.models.inkling2.rows.gemv import gemv_rows, gemv_rows_multi, gemv_rows_group, rel_logits_rows

P = os.path.expanduser("~/loopling/models/loopling-p3")
model, tok = load(P)
TM = model.model.language_model.layers
BS = [1, 2, 3, 5, 8, 16]
GATES = ["q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"]

# ---- capture real decode activations (reference path, un-compiled) ----
CAP = {}
def rec(key, a):
    CAP.setdefault(key, []).append(a)
_mg, _tk, _rl = LY.MessageGate.__call__, LY.TopKRouter.__call__, LY.RelativeLogits.__call__
def mg(self, hidden_states, message):
    states = mx.concatenate([LY.unweighted_rms_norm(hidden_states), LY.unweighted_rms_norm(message)], axis=-1)
    if states.shape[1] == 1:
        rec(("down", id(self)), states.astype(self.down_proj.weight.dtype))
        gate_in = self.down_proj(states.astype(self.down_proj.weight.dtype)).astype(mx.float32)
        rec(("gate", id(self)), gate_in)
    return _mg(self, hidden_states, message)
def tk(self, h):
    if h.shape[1] == 1: rec(("router", id(self)), h)
    return _tk(self, h)
def rl(self, r):
    if r.shape[1] == 1: rec(("rel", id(self)), r)
    return _rl(self, r)
LY.MessageGate.__call__, LY.TopKRouter.__call__, LY.RelativeLogits.__call__ = mg, tk, rl
ids = tok.encode("Write a short poem about the sea and the stars.", add_special_tokens=False)
n = 0
for resp in stream_generate(model, tok, ids, max_tokens=10, sampler=lambda x: mx.argmax(x, axis=-1)):
    n += 1
LY.MessageGate.__call__, LY.TopKRouter.__call__, LY.RelativeLogits.__call__ = _mg, _tk, _rl
for k, v in CAP.items(): mx.eval(v)
print("captured %d decode steps; keys %d" % (len(CAP[("router", id(TM[0].mlp.gate))]), len(CAP)), flush=True)

def bits(a):
    return a.view(mx.uint32) if a.dtype == mx.float32 else a.view(mx.uint16)
def nmis(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    return int(mx.sum(bits(a) != bits(b)).item())

mx.random.seed(0)
rnd = lambda *s: mx.random.normal(s)
def rms(x): return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)

def ref_lin(x, w):  # x (B,K) -> (B,N); each row alone, shapes (1,1,K) @ w.T like nn.Linear
    return mx.concatenate([(x[b].reshape(1, 1, -1) @ w.T).reshape(1, -1) for b in range(x.shape[0])], axis=0)
def ref_rel(r, proj):
    return mx.concatenate([(r[b][None, None] @ proj)[0] for b in range(r.shape[0])], axis=0)

def real_rows(key, B, shape):
    """B rows cycling through the captured decode steps, shape (B,)+shape."""
    caps = CAP[key]
    return mx.concatenate([caps[i % len(caps)].reshape((1,) + shape) for i in range(B)], axis=0)

results = {}  # (kind, source, B) -> [cases, mismatching_elems, failing layers]
def report(kind, src, B, layer, mis, cnt):
    r = results.setdefault((kind, src, B), [0, 0, 0, []])
    r[0] += 1; r[1] += mis; r[2] += cnt
    if mis: r[3].append(layer)

def check(kind, src, B, layer, outs, refs):
    mis = cnt = 0
    for o, r in zip(outs, refs):
        mis += nmis(o, r); cnt += r.size
    report(kind, src, B, layer, mis, cnt)

for src in ("random", "real"):
    for B in BS:
        for li, L in enumerate(TM):
            # router
            w = L.mlp.gate.weight
            if src == "random": h = (rnd(B, 2304) * 3).astype(mx.bfloat16)
            else: h = real_rows(("router", id(L.mlp.gate)), B, (2304,))
            x = h.astype(mx.float32)
            ref = ref_lin(x, w); out = gemv_rows(x, w); mx.eval(ref, out)
            check("router(128x2304 f32)", src, B, li, [out], [ref])
            # relative logits
            proj = L.self_attn.rel_logits_proj.proj
            if src == "random": r = rnd(B, 24, 16).astype(mx.bfloat16)
            else: r = real_rows(("rel", id(L.self_attn.rel_logits_proj)), B, (24, 16))
            ref = ref_rel(r, proj); out = rel_logits_rows(r, proj); mx.eval(ref, out)
            check("rel_logits(24x16@16x1024 bf16)", src, B, li, [out], [ref])
            if not hasattr(L, "q_message_gate"): continue
            xd, xg, wd, wf, wm, refd, reff, refm = [], [], [], [], [], [], [], []
            for g in GATES:
                G = getattr(L, g)
                if src == "random":
                    d = mx.concatenate([rms(rnd(B, 2304)), rms(rnd(B, 2304))], axis=-1).astype(mx.bfloat16)
                    gi = (rnd(B, 64) * 2).astype(mx.float32)
                else:
                    d = real_rows(("down", id(G)), B, (4608,)); gi = real_rows(("gate", id(G)), B, (64,))
                rd = ref_lin(d, G.down_proj.weight); od = gemv_rows(d, G.down_proj.weight)
                rf = ref_lin(gi, G.forget_gate_proj.weight); rm = ref_lin(gi, G.message_gate_proj.weight)
                of = gemv_rows(gi, G.forget_gate_proj.weight); om = gemv_rows(gi, G.message_gate_proj.weight)
                omf, omm = gemv_rows_multi(gi, [G.forget_gate_proj.weight, G.message_gate_proj.weight])
                mx.eval(rd, od, rf, rm, of, om, omf, omm)
                check("gate.down(64x4608 bf16)", src, B, li, [od], [rd])
                check("gate.forget(2304x64 f32)", src, B, li, [of], [rf])
                check("gate.message(2304x64 f32)", src, B, li, [om], [rm])
                check("gate.forget+message one launch", src, B, li, [omf, omm], [rf, rm])
                xd.append(d); xg.append(gi); wd.append(G.down_proj.weight)
                wf.append(G.forget_gate_proj.weight); wm.append(G.message_gate_proj.weight)
                refd.append(rd); reff.append(rf); refm.append(rm)
            od = gemv_rows_group(xd, wd)
            of = gemv_rows_group(xg + xg, wf + wm)
            mx.eval(od, of)
            check("4 gates down one launch", src, B, li, od, refd)
            check("4 gates forget+message one launch (8 problems)", src, B, li, of, reff + refm)
            od3 = gemv_rows_group(xd[:3], wd[:3]); of3 = gemv_rows_group((xg + xg)[:3] + xg[:3], wf[:3] + wm[:3]); mx.eval(od3, of3)
            check("3 gates (q,k,v) down one launch", src, B, li, od3, refd[:3])
            check("3 gates (q,k,v) forget+message one launch", src, B, li, of3, reff[:3] + refm[:3])
    print("done", src, flush=True)

# negative control: the naive zero-copy batched GEMM must NOT be exact (proves the compare is sensitive)
ctl = 0
for B in (5, 8):
    G = TM[7].q_message_gate; gi = (rnd(B, 64) * 2).astype(mx.float32); d = mx.concatenate([rms(rnd(B, 2304)), rms(rnd(B, 2304))], -1).astype(mx.bfloat16)
    for x, w in ((gi, G.forget_gate_proj.weight), (d, G.down_proj.weight)):
        a = x @ w.T; b = ref_lin(x, w); mx.eval(a, b); ctl += nmis(a, b)
print("control (naive M=B matmul) mismatching elements: %d (must be > 0)" % ctl)

fails = 0
for (kind, src, B), (n_, mis, cnt, bad) in sorted(results.items()):
    ok = mis == 0
    fails += not ok
    print("%s %-52s %-6s B=%-2d cases=%-3d elems=%-9d mismatches=%d%s" % ("PASS" if ok else "FAIL", kind, src, B, n_, cnt, mis, "" if ok else " layers=%s" % sorted(set(bad))))
tot = sum(v[1] for v in results.values())
print("SUMMARY: %d groups, %d failing, total mismatching elements %d" % (len(results), fails, tot))
sys.exit(1 if fails else 0)
