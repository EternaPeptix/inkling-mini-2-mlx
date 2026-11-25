"""Exactness of rows/moe.py (K11 router, K12 routing+gate_up+swiglu, K13 down+combine+conv+residual) vs fused_ref stage E.

  python bench/fused_moe_test.py [--layers 0,1,..] [--visits ~/loopling-opt/runs/visits_p3] [--bs 1,2,3,5,8]

Rows = B different captured visits of the same layer (reference = fused_ref.visit_ref per visit alone) and random rows
(independent / identical / correlated so experts are shared). All tensors compared by bit pattern per row; both routing
modes (route in kernel, route as its own launch), debug and production variants. 0 mismatches required.
"""
import argparse, json, os, sys

import mlx_lm  # noqa: F401  (before mlx.core)
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import fused_ref as FR
from mlx_lm.models.inkling2.rows import moe as M
from mlx_lm.models.inkling2.rows import rowlocal as RL

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default=",".join(str(i) for i in range(24)))
ap.add_argument("--visits", default=os.path.expanduser("~/loopling-opt/runs/visits_p3"))
ap.add_argument("--bs", default="1,2,3,5,8")
ap.add_argument("--model", default=os.path.expanduser("~/loopling/models/loopling-p3"))
a = ap.parse_args()
LAYERS = [int(i) for i in a.layers.split(",")]
BS = [int(i) for i in a.bs.split(",")]
f32 = mx.float32

model, tok = load(a.model)
layers = list(model.model.language_model.layers)
index = json.load(open(os.path.join(a.visits, "index.json")))["visits"]

NAMES = ["router_logits", "expert_inds", "expert_w", "gate_up", "swiglu", "down", "moe_out", "post_mlp_conv", "w3", "out"]


def bits(x):
    return x.view(mx.uint32) if x.dtype in (mx.float32, mx.uint32) else x.view(mx.uint16)


def ref_E(layer, x, residual, cs3):
    """Stage E of fused_ref.visit_ref for one row: x (1,1,D) bf16, residual (1,1,D) f32, cs3 (1,3,D) f32."""
    moe = layer.mlp
    T = {}
    T["router_logits"] = x.astype(f32) @ moe.gate.weight.T
    inds, wts = moe.gate(x)
    T["expert_inds"], T["expert_w"] = inds, wts
    ex = moe.experts
    xe = mx.expand_dims(x, (-2, -3))
    gu = ex.gate_up_proj(xe, inds, sorted_indices=False)
    T["gate_up"] = gu
    act = ex.act_fn(gu[..., 0::2], gu[..., 1::2])
    T["swiglu"] = act
    dn = ex.down_proj(act, inds, sorted_indices=False).squeeze(-2)
    T["down"] = dn
    comb = (dn.astype(f32) * wts[..., None]).sum(axis=-2).astype(x.dtype)
    T["moe_out"] = comb
    hidden, w3 = FD._sc(layer.post_mlp_convolution, comb, cs3)
    T["post_mlp_conv"], T["w3"] = hidden, w3
    T["out"] = (residual + hidden.astype(f32)).astype(x.dtype)
    return T


# ---- reference tensors of every captured visit (visit_ref, stage E part)
vis = {li: [] for li in LAYERS}
for n, v in enumerate(index):
    li = v["layer"]
    if li not in vis:
        continue
    d = mx.load(os.path.join(a.visits, v["file"]))
    cs = [d["cs%d" % i] for i in range(4)]
    T = FR.visit_ref(layers[li], d["h"], d.get("bs"), d.get("msg"), cs, d["attn_out"], d.get("tau"))
    rec = {k: T[k] for k in NAMES + ["mlp_x", "residual"]}
    rec["cs3"] = cs[3]
    mx.eval(list(rec.values()))
    vis[li].append(rec)
print("reference visits:", {li: len(v) for li, v in vis.items()}, flush=True)

tot = {"elems": 0, "bad": 0, "cases": 0, "fail": 0}
bad_by = {}


def check(tag, got, refs, B, li):
    """got: dict name -> (B,...) ; refs: list of B ref dicts."""
    nbad = 0
    for k in got:
        g = got[k].reshape(B, -1)
        r = mx.concatenate([rr[k].reshape(1, -1) for rr in refs])
        if g.dtype != r.dtype or g.shape != r.shape:
            print("  DTYPE/SHAPE %s %s %s vs %s" % (tag, k, g.shape, r.shape))
            nbad += g.size
            bad_by[k] = bad_by.get(k, 0) + g.size
            continue
        m = int(mx.sum(bits(g) != bits(r)).item())
        tot["elems"] += g.size
        if m:
            nbad += m
            bad_by[k] = bad_by.get(k, 0) + m
            print("  MISMATCH L%02d %s B=%d %s: %d/%d" % (li, tag, B, k, m, g.size))
    tot["cases"] += 1
    tot["bad"] += nbad
    tot["fail"] += nbad > 0
    return nbad


def run_case(li, tag, x, resid, cs3, refs):
    B = x.shape[0]
    layer = layers[li]
    for route in ("kernel", "launch"):
        o = M.moe_rows(layer, x, resid, cs3, route=route, debug=True)
        mx.eval(list(o.values()))
        check("%s/%s" % (tag, route), o, refs, B, li)
        if route == "kernel":
            o2 = M.moe_rows(layer, x, resid, cs3, route=route, debug=False)
            mx.eval(list(o2.values()))
            check("%s/prod" % tag, {k: o2[k] for k in ("out", "w3", "expert_inds", "expert_w", "swiglu")}, refs, B, li)


mx.random.seed(1)
for li in LAYERS:
    V = vis[li]
    nv = len(V)
    D = V[0]["mlp_x"].shape[-1]
    before = tot["bad"]
    for B in BS:
        for off in range(0, min(nv, 3)):
            sel = [V[(off + b * max(1, nv // B)) % nv] for b in range(B)]
            x = mx.concatenate([s["mlp_x"].reshape(1, D) for s in sel])
            resid = mx.concatenate([s["residual"].reshape(1, D) for s in sel])
            cs3 = mx.concatenate([s["cs3"] for s in sel])
            run_case(li, "real%d" % off, x, resid, cs3, sel)
        # random rows: independent, identical, correlated, large
        for kind in ("indep", "same", "corr", "scaled"):
            if kind == "indep":
                x = mx.random.normal((B, D)) * 1.5
            elif kind == "same":
                x = mx.broadcast_to(mx.random.normal((1, D)) * 1.5, (B, D))
            elif kind == "corr":
                x = mx.random.normal((1, D)) * 1.5 + 0.2 * mx.random.normal((B, D))
            else:
                x = mx.random.normal((B, D)) * 6.0
            x = x.astype(mx.bfloat16)
            resid = mx.random.normal((B, D)) * 3
            cs3 = mx.random.normal((B, 3, D)) * 2
            mx.eval(x, resid, cs3)
            refs = []
            for b in range(B):
                T = ref_E(layers[li], x[b][None, None], resid[b][None, None], cs3[b][None])
                mx.eval(list(T.values()))
                refs.append(T)
            run_case(li, kind, x, resid, cs3, refs)
    print("layer %02d (%s) mismatches %d" % (li, layers[li].mlp.experts.gate_up_proj.mode, tot["bad"] - before), flush=True)

print("SUMMARY cases=%d failing=%d elements=%d mismatches=%d by tensor=%s -> %s" % (
    tot["cases"], tot["fail"], tot["elems"], tot["bad"], bad_by, "PASS" if tot["bad"] == 0 else "FAIL"))
sys.exit(0 if tot["bad"] == 0 else 1)
