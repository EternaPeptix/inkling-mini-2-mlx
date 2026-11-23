"""Bitwise tests of rows/rowlocal.py (K1 PRE, K3b NORM, K10 POST) against fused_ref.visit_ref evaluated on each row alone.

  LANE=f1 RHOST=512S1 scripts/r --gpu 'python bench/fused_rowlocal_test.py [--quick] [--bs 1,2,3,5,8]'

Rows = different captured visits (runs/visits_p3) of the SAME layer and kind (has_msg); plus random rows (random h, bs, msg,
conv state, o_proj, attn_out through visit_ref).  Every compared tensor: shape, dtype and bit pattern.  Exit 1 on any mismatch.
"""
import argparse, json, os, sys

import mlx_lm  # first
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.inkling2 import fused_ref as FR
from mlx_lm.models.inkling2 import layers as L
from mlx_lm.models.inkling2.rows import rowlocal as RL

ap = argparse.ArgumentParser()
ap.add_argument("--bs", default="1,2,3,5,8")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--visits", default=os.path.expanduser("~/loopling-opt/runs/visits_p3"))
a = ap.parse_args()
BS = [int(x) for x in a.bs.split(",")]
D = 2304

idx = json.load(open(a.visits + "/index.json"))
model, _ = load(os.path.expanduser("~/loopling/models/") + idx["model"])
layers = list(model.model.language_model.layers)

STATS = {}  # (kernel, kind) -> [cases, elements, mismatches]
BAD = []


def bits(x):
    return np.array(x.view(mx.uint32) if x.dtype == mx.float32 else x.view(mx.uint16))


def cmp(kern, kind, name, ref_rows, mine, tag):
    """ref_rows: list of B arrays (row-alone reference, any leading singleton dims); mine: (B, ...) array."""
    B = len(ref_rows)
    mx.eval(ref_rows + [mine])
    r = mx.concatenate([x.reshape(1, -1) for x in ref_rows], axis=0)
    m = mine.reshape(B, -1)
    st = STATS.setdefault((kern, kind), [0, 0, 0])
    st[0] += 1
    st[1] += r.size
    if r.dtype != m.dtype or r.shape != m.shape:
        st[2] += r.size
        BAD.append("%s %s %s dtype/shape %s %s vs %s %s" % (kern, kind, tag, r.dtype, r.shape, m.dtype, m.shape))
        return
    nb = int((bits(r) != bits(m)).sum())
    st[2] += nb
    if nb:
        BAD.append("%s %s %s %s: %d mismatches" % (kern, kind, tag, name, nb))


def kind_of(layer_i, full, msg):
    L_ = layers[layer_i]
    return ("F" if full else "S") + ("b" if L_.is_block_boundary else "") + ("m" if msg else "") + ("0" if layer_i == 0 else "")


def run(layer_i, rows, tag):
    """rows: list of dict(h, bs, msg, cs (4), attn_out, tau)."""
    layer = layers[layer_i]
    full = not isinstance(layer, L.SlidingDecoderLayer)
    has_msg = rows[0]["msg"] is not None
    kind = kind_of(layer_i, full, has_msg)
    Ts = [FR.visit_ref(layer, r["h"], r["bs"], r["msg"], r["cs"], r["attn_out"], r["tau"]) for r in rows]
    mx.eval([[v for v in T.values() if v is not None] for T in Ts])
    B = len(rows)
    h = mx.concatenate([r["h"].reshape(1, D) for r in rows])
    bs = None if rows[0]["bs"] is None else mx.concatenate([r["bs"].reshape(-1, 1, D) for r in rows], axis=1)
    msg = None if not has_msg else mx.concatenate([r["msg"].reshape(1, 4, D) for r in rows])
    ref = lambda name: [T[name] for T in Ts]

    # ---- K1
    for dbg in (False, True):
        o = RL.k1_pre_layer(layer, h, bs, msg, debug=dbg)
        cmp("K1", kind, "normed_residual", ref("normed_residual"), o["normed_residual"], tag)
        if dbg and bs is not None:
            cmp("K1", kind, "attn_input", ref("attn_input"), o["attn_input"], tag)
        if not has_msg:
            for n in ("q_in", "k_in", "v_in"):
                cmp("K1", kind, n, ref(n), o[n], tag)
        else:
            for i, c in enumerate("qkv"):
                cmp("K1", kind, c + "_gate_in", ref(c + "_gate_in"), o[c + "_gate_in"], tag)
            # states = attention_input rows
            for i, c in enumerate("qkv"):
                src = [T["attn_input"][..., 0 if not full else i, :] for T in Ts]
                cmp("K1", kind, "s_" + c, src, o["s_" + c], tag)
        if not a.quick or not dbg:
            pass
    # ---- K3b
    if has_msg:
        mixed = mx.stack([mx.concatenate([T[c + "_mixed"].reshape(1, D) for T in Ts]) for c in "qkv"], axis=1)  # (B,3,D)
        outs = RL.k3b_norm_layer(layer, mixed)
        for n, ot in zip(("q_in", "k_in", "v_in"), outs):
            cmp("K3b", kind, n, ref(n), ot, tag)
        mm = mx.concatenate([T["mlp_mixed"].reshape(1, 1, D) for T in Ts])
        cmp("K3b", kind, "mlp_x", ref("mlp_x"), RL.k3b_norm_layer(layer, mm, mlp=True)[0], tag)
    else:
        # random states through the same weights (all visits of the kind): test with the reference's own layernorm
        mixed = mx.random.normal((B, 3, D)) * 2 + 0.3
        outs = RL.k3b_norm_layer(layer, mixed)
        if full:
            lns = [layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm]
        else:
            lns = [layer.input_layernorm] * 3
        for g, (ln, ot) in enumerate(zip(lns, outs)):
            cmp("K3b", kind + "r", "rand%d" % g, [ln(mixed[b, g][None, None]).astype(mx.bfloat16) for b in range(B)], ot, tag)
        cmp("K3b", kind + "r", "rand_mlp", [layer.post_attention_layernorm(mixed[b, :1]).astype(mx.bfloat16) for b in range(B)],
            RL.k3b_norm_layer(layer, mixed[:, :1], mlp=True)[0], tag)
    # ---- K10
    nr = mx.concatenate([T["normed_residual"].reshape(1, D) for T in Ts])
    op = mx.concatenate([T["o_proj"].reshape(1, D) for T in Ts])
    cs2 = mx.concatenate([r["cs"][2].reshape(1, 3, D) for r in rows])
    o = RL.k10_post_layer(layer, op, h, nr, bs, cs2, msg)
    for n in ("post_attn_conv", "w2", "residual", "normed", "mlp_src"):
        cmp("K10", kind, n, ref(n), o[n], tag)
    if has_msg:
        cmp("K10", kind, "mlp_gate_in", ref("mlp_gate_in"), o["mlp_gate_in"], tag)
    else:
        cmp("K10", kind, "mlp_x", ref("mlp_x"), o["mlp_x"], tag)
    if layer.is_block_boundary:  # bs_new = concat(bs, nr): the new block row is normed_residual (as passed)
        cmp("K10", kind, "bs_new_row", [T["bs_new"][-1] for T in Ts], nr, tag)


# ---------------------------------------------------------------------------------------------- real visits
groups = {}
for v in idx["visits"]:
    groups.setdefault((v["layer"], v["has_msg"]), []).append(v)


def load_visit(v):
    t = mx.load(a.visits + "/" + v["file"])
    return dict(h=t["h"], bs=t.get("bs"), msg=t.get("msg"), cs=[t["cs%d" % i] for i in range(4)], attn_out=t["attn_out"], tau=t.get("tau"))


rng = np.random.RandomState(0)
for (li, hm), vs in sorted(groups.items()):
    data = [load_visit(v) for v in vs]
    for B in BS:
        if B > len(vs):
            # repeat visits (rows may repeat; still B separate rows)
            sel = [i % len(vs) for i in range(B)]
        else:
            sel = list(rng.permutation(len(vs))[:B])
        run(li, [data[i] for i in sel], "real L%d B%d" % (li, B))
    print("layer %2d msg=%d done (%d visits)" % (li, hm, len(vs)), flush=True)

# ---------------------------------------------------------------------------------------------- random rows
mx.random.seed(1)
tpl = {}
for (li, hm), vs in groups.items():
    tpl[(li, hm)] = load_visit(vs[0])


def rand_row(t, sc):
    r = dict(t)
    r["h"] = (mx.random.normal(t["h"].shape) * sc + 0.2).astype(mx.bfloat16)
    if t["bs"] is not None:
        r["bs"] = mx.random.normal(t["bs"].shape) * sc * 0.7
    if t["msg"] is not None:
        r["msg"] = mx.random.normal(t["msg"].shape) * sc
    r["cs"] = [mx.random.normal(c.shape) * sc for c in t["cs"]]
    r["attn_out"] = (mx.random.normal(t["attn_out"].shape) * sc).astype(mx.bfloat16)
    return r


for (li, hm), t in sorted(tpl.items()):
    if a.quick and li not in (0, 1, 5, 6, 7, 11, 18, 23):
        continue
    for B in (1, 3, 8):
        rows = [rand_row(t, [0.1, 1.0, 5.0, 30.0][b % 4]) for b in range(B)]
        run(li, rows, "rand L%d B%d" % (li, B))

print("\n%-5s %-6s %6s %12s %10s" % ("kern", "kind", "cases", "elements", "mismatch"))
tot = 0
for (k, kind), (c, n, m) in sorted(STATS.items()):
    print("%-5s %-6s %6d %12d %10d" % (k, kind, c, n, m))
    tot += m
for b in BAD[:40]:
    print("MISMATCH", b)
print("TOTAL mismatching elements:", tot, "PASS" if tot == 0 else "FAIL")
sys.exit(0 if tot == 0 else 1)
