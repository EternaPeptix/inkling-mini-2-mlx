"""Exactness of rows/proj.py (K4 QKV, K9 OPROJ, K2 GATE-DOWN, K3 GATE-MIX) against fused_ref.visit_ref on real captured visits.

  python bench/fused_proj_test.py [--visits ~/loopling-opt/runs/visits_p3] [--layers 0-23]

B rows = B different captured visits of the same layer (windows starting at offsets 0 and 1), B in {1,2,3,5,8}; message
visits (layers 6..17) for K2/K3 (also K2 -> K3 chained); plus random rows. Reference per row = visit_ref on that visit
alone (tensor names of docs/FUSED.md).  Bit patterns compared; any mismatch fails.
"""
import argparse, collections, json, os, sys

os.environ.setdefault("LOOPLING_COMPILE_DECODE", "0")
import mlx_lm  # noqa: F401  (before mlx.core)
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import fused_ref as FR
from mlx_lm.models.inkling2.rows import proj as P

ap = argparse.ArgumentParser()
ap.add_argument("--visits", default=os.path.expanduser("~/loopling-opt/runs/visits_p3"))
ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
ap.add_argument("--layers", default="0-23")
ap.add_argument("--per-layer", type=int, default=9)
a = ap.parse_args()
lo, hi = map(int, a.layers.split("-"))
BS = [1, 2, 3, 5, 8]

model, tok = load(os.path.join(a.models, "loopling-p3"))
layers = list(model.model.language_model.layers)
idx = json.load(open(os.path.join(a.visits, "index.json")))["visits"]

STAT = collections.defaultdict(lambda: [0, 0, 0])  # name -> [cases, elements, mismatching elements]


def bits(x):
    return x.view(mx.uint32) if x.dtype == mx.float32 else x.view(mx.uint16)


def cmp(name, got, refs, tag=""):
    """got (B, ...), refs list of B arrays each with the same total size."""
    B = got.shape[0]
    ref = mx.stack([r.reshape(got.shape[1:]) for r in refs])
    assert ref.dtype == got.dtype, (name, ref.dtype, got.dtype)
    bad = int(mx.sum(bits(got) != bits(ref)).item())
    s = STAT[name]
    s[0] += 1; s[1] += got.size; s[2] += bad
    if bad:
        print("FAIL", name, tag, "B", B, "bad", bad, flush=True)


def load_visit(v):
    d = mx.load(os.path.join(a.visits, v["file"]))
    cs = [d["cs%d" % i] for i in range(4)]
    T = FR.visit_ref(layers[v["layer"]], d["h"], d.get("bs"), d.get("msg"), cs, d["attn_out"], d.get("tau"))
    keep = {}
    for k in ("q_in", "k_in", "v_in", "q_proj", "r_proj", "gate", "sinks", "k_proj", "v_proj", "k_wproj", "v_wproj",
              "attn_out", "o_proj", "attn_input", "mlp_src"):
        if T.get(k) is not None:
            keep[k] = T[k]
    for p in ("q_", "k_", "v_", "mlp_"):
        for n in ("gate_in", "gate_down", "mixed"):
            if p + n in T:
                keep[p + n] = T[p + n]
    keep["msg"] = d.get("msg")
    mx.eval(list(v2 for v2 in keep.values() if v2 is not None))
    return keep


def stackrows(rows, key):
    return mx.concatenate([r[key].reshape(1, -1) for r in rows])


def windows(rows, B):
    for off in (0, 1):
        if off + B <= len(rows):
            yield off, rows[off:off + B]


def test_layer(li, rows):
    layer = layers[li]
    at = layer.self_attn
    full = hasattr(at.k_sconv, "weight_proj")
    names = ["q_proj", "r_proj", "gate", "sinks", "k_proj", "v_proj"] + (["k_wproj", "v_wproj"] if full else [])
    # ---- K4, K9 (any visit of the layer)
    for B in BS:
        for off, rs in windows(rows, B):
            qi, ki, vi = (stackrows(rs, k) for k in ("q_in", "k_in", "v_in"))
            o = P.qkv_rows(at, qi, ki, vi)
            for n in names:
                cmp("K4 " + n, o[n], [r[n] for r in rs], "L%d off%d" % (li, off))
            if not full and all(r['msg'] is None for r in rs):  # sliding, no message gates: q_in == k_in == v_in as one aliased array
                o = P.qkv_rows(at, qi, qi, qi)
                for n in names:
                    cmp("K4 " + n + " (aliased)", o[n], [r[n] for r in rs], "L%d off%d" % (li, off))
            ao = mx.concatenate([r["attn_out"].reshape(1, -1) for r in rs])
            g = stackrows(rs, "gate")
            out = P.oproj_rows(ao, g, at.o_proj.weight, at.o_proj.scales, at.o_proj.biases)
            cmp("K9 o_proj", out, [r["o_proj"] for r in rs], "L%d off%d" % (li, off))
    # ---- random rows
    for B in BS:
        mx.random.seed(100 * li + B)
        xq, xk, xv = (mx.random.normal((B, 2304)).astype(mx.bfloat16) * s for s in (1.0, 2.0, 0.5))
        o = P.qkv_rows(at, xq, xk, xv)
        refs = {"q_proj": at.q_proj, "r_proj": at.r_proj, "gate": at.gate_proj, "sinks": at.sink_proj}
        for n, m in refs.items():
            cmp("K4 " + n + " random", o[n], [m(xq[b].reshape(1, 1, -1)) for b in range(B)], "L%d" % li)
        cmp("K4 k_proj random", o["k_proj"], [at.k_proj(xk[b].reshape(1, 1, -1)) for b in range(B)], "L%d" % li)
        cmp("K4 v_proj random", o["v_proj"], [at.v_proj(xv[b].reshape(1, 1, -1)) for b in range(B)], "L%d" % li)
        if full:
            cmp("K4 k_wproj random", o["k_wproj"], [at.k_sconv.weight_proj(xk[b].reshape(1, 1, -1)) for b in range(B)])
            cmp("K4 v_wproj random", o["v_wproj"], [at.v_sconv.weight_proj(xv[b].reshape(1, 1, -1)) for b in range(B)])
        ao = (mx.random.normal((B, 3072)) * 2).astype(mx.bfloat16)
        gt = (mx.random.normal((B, 3072)) * 4).astype(mx.bfloat16)
        out = P.oproj_rows(ao, gt, at.o_proj.weight, at.o_proj.scales, at.o_proj.biases)
        cmp("K9 o_proj random", out, [at._attend_post(ao[b].reshape(1, 1, 24, 128), gt[b].reshape(1, 1, -1)) for b in range(B)], "L%d" % li)
    # ---- K2, K3 (message visits)
    mrows = [r for r in rows if r["msg"] is not None]
    if not mrows:
        return
    gq = [layer.q_message_gate, layer.k_message_gate, layer.v_message_gate]
    for B in BS:
        for off, rs in windows(mrows, B):
            msg = mx.concatenate([r["msg"].reshape(1, 4, 2304) for r in rs])
            for label, pre, gates, slots in (("qkv", ("q_", "k_", "v_"), gq, (0, 1, 2)), ("mlp", ("mlp_",), [layer.mlp_message_gate], (3,))):
                G = len(gates)
                x = mx.stack([mx.concatenate([r[p + "gate_in"].reshape(1, -1) for r in rs]) for p in pre], axis=1)  # (B,G,4608)
                dn = P.gate_down_rows(x, [g.down_proj.weight for g in gates])
                for gi, p in enumerate(pre):
                    cmp("K2 gate_down " + label, dn[:, gi], [r[p + "gate_down"] for r in rs], "L%d off%d g%d" % (li, off, gi))
                # K3 inputs: s per gate
                ai = [r["attn_input"] for r in rs]
                if label == "qkv":
                    if full:
                        ss = [mx.concatenate([r["attn_input"][..., i, :].reshape(1, -1) for r in rs]) for i in range(3)]
                    else:
                        s0 = mx.concatenate([r["attn_input"][..., 0, :].reshape(1, -1) for r in rs])
                        ss = [s0, s0, s0]
                else:
                    ss = [stackrows(rs, "mlp_src")]
                for chain in (False, True):
                    d3 = dn if chain else mx.stack([mx.concatenate([r[p + "gate_down"].reshape(1, -1) for r in rs]) for p in pre], axis=1)
                    mixed = P.gate_mix_rows(d3, ss, msg, slots, [g.forget_gate_proj.weight for g in gates],
                                            [g.message_gate_proj.weight for g in gates])
                    for gi, p in enumerate(pre):
                        cmp("K3 gate_mix " + label + (" (K2->K3)" if chain else ""), mixed[:, gi], [r[p + "mixed"] for r in rs],
                            "L%d off%d g%d" % (li, off, gi))
    # random
    for B in BS:
        mx.random.seed(7000 + 100 * li + B)
        for label, gates, slots in (("qkv", gq, (0, 1, 2)), ("mlp", [layer.mlp_message_gate], (3,))):
            G = len(gates)
            x = (mx.random.normal((B, G, 4608)) * 1.5).astype(mx.bfloat16)
            dn = P.gate_down_rows(x, [g.down_proj.weight for g in gates])
            for gi, g in enumerate(gates):
                cmp("K2 gate_down random " + label, dn[:, gi], [g.down_proj(x[b, gi].reshape(1, 1, -1)) for b in range(B)])
            ss = [mx.random.normal((B, 2304)) * 2 for _ in range(G)]
            msg = mx.random.normal((B, 4, 2304)) * 3
            dd = (mx.random.normal((B, G, 64)) * 3).astype(mx.bfloat16)
            mixed = P.gate_mix_rows(dd, ss, msg, slots, [g.forget_gate_proj.weight for g in gates],
                                    [g.message_gate_proj.weight for g in gates])
            for gi, g in enumerate(gates):
                refs = []
                for b in range(B):
                    st = dd[b, gi].reshape(1, 1, 64).astype(mx.float32)
                    fg = mx.tanh(g.forget_gate_proj(st)); mg = mx.tanh(g.message_gate_proj(st))
                    refs.append((1 - fg) * ss[gi][b].reshape(1, 1, -1) + mg * msg[b, slots[gi]].reshape(1, 1, -1))
                cmp("K3 gate_mix random " + label, mixed[:, gi], refs)


byl = collections.defaultdict(list)
for v in idx:
    if lo <= v["layer"] <= hi and len(byl[v["layer"]]) < a.per_layer:
        byl[v["layer"]].append(v)
# prefer visits that carry msg for looped layers (all visits of layers 6..17 pass > 0 have it; the first pass does not)
for li in sorted(byl):
    allv = [v for v in idx if v["layer"] == li]
    if 6 <= li <= 17:
        msgv = [v for v in allv if v["has_msg"]]
        plain = [v for v in allv if not v["has_msg"]]
        pick = (msgv[: a.per_layer - 1] + plain[:1]) if msgv else plain[: a.per_layer]
        # interleave so windows mix prompts/steps
        byl[li] = pick
    else:
        byl[li] = allv[: a.per_layer]
    rows = [load_visit(v) for v in byl[li]]
    test_layer(li, rows)
    print("layer %d done (%d visits, %d with msg)" % (li, len(rows), sum(r["msg"] is not None for r in rows)), flush=True)

tot_c = tot_e = tot_b = 0
print("%-34s %6s %12s %6s" % ("case", "cases", "elements", "bad"))
for k in sorted(STAT):
    c, e, b = STAT[k]
    tot_c += c; tot_e += e; tot_b += b
    print("%-34s %6d %12d %6d" % (k, c, e, b))
print("SUMMARY: cases %d elements %d mismatching %d %s" % (tot_c, tot_e, tot_b, "ALL PASS" if tot_b == 0 else "FAIL"))
sys.exit(1 if tot_b else 0)
