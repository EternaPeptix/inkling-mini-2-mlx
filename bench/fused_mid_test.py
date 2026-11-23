"""Exactness test for rows/mid.py (K5 MID): per-row bit compare vs fused_ref.visit_ref (stage M) on real captured visits
(all 24 layers, message and non-message visits) for B in {1,2,3,5,8}, plus random rows.
  python bench/fused_mid_test.py [visits_dir]"""
import json, os, sys
os.environ["LOOPLING_COMPILE_DECODE"] = "0"
import mlx_lm
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import fused_ref as FR
from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import layers as L
from mlx_lm.models.inkling2.rows.mid import mid_rows

VD = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/loopling-opt/runs/visits_p3")
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
layers = list(model.model.language_model.layers)
idx = json.load(open(os.path.join(VD, "index.json")))["visits"]
f32, bf16 = mx.float32, mx.bfloat16
BS = [1, 2, 3, 5, 8]


def bits(x):
    return x.view(mx.uint32) if x.dtype == f32 else x.view(mx.uint16)


def nmis(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    return int(mx.sum(bits(a) != bits(b)).item())


def mid_ref(layer, q, r, kr, vr, k_in, v_in, cs0, cs1, tau):
    """stage M of visit_ref for one row (shapes as in visit_ref: (1,1,.) and cs (1,3,C))."""
    a = layer.self_attn
    if isinstance(layer, L.SlidingDecoderLayer):
        key, w0 = FD._sc(a.k_sconv, kr, cs0)
        value, w1 = FD._sc(a.v_sconv, vr, cs1)
    else:
        key, w0 = FD._dsc(a.k_sconv, kr, k_in, cs0)
        value, w1 = FD._dsc(a.v_sconv, vr, v_in, cs1)
    qh = a.q_l2_clip(q.reshape(1, 1, a.num_heads, -1))
    kh = a.k_l2_clip(key.reshape(1, 1, a.num_key_value_heads, -1))
    vh = value.reshape(1, 1, a.num_key_value_heads, -1)
    vh = L.unweighted_rms_norm(vh.astype(f32), a.eps).astype(vh.dtype)
    rh = a.rel_l2_clip(r.reshape(1, 1, a.num_heads, -1))
    if tau is not None:
        qh = (qh.astype(f32) * tau).astype(qh.dtype)
        rh = (rh.astype(f32) * tau).astype(rh.dtype)
    return dict(q=qh, k=kh, v=vh, rel=rh, w0=w0, w1=w1, k_conv=key, v_conv=value)


_cacheT = {}
def real_row(i):
    """inputs + visit_ref tensors of visit i of the index."""
    if i in _cacheT:
        return _cacheT[i]
    v = idx[i]
    d = mx.load(os.path.join(VD, v["file"]))
    cs = [d["cs%d" % j] for j in range(4)]
    T = FR.visit_ref(layers[v["layer"]], d["h"], d.get("bs"), d.get("msg"), cs, d["attn_out"], d.get("tau"))
    row = dict(q=T["q_proj"], r=T["r_proj"], kr=T["k_proj"], vr=T["v_proj"], k_in=T["k_in"], v_in=T["v_in"],
               cs0=cs[0], cs1=cs[1], tau=d.get("tau"), T=T)
    mx.eval([x for x in row.values() if isinstance(x, mx.array)] + [T[k] for k in ("q", "k", "v", "rel", "w0", "w1")])
    _cacheT[i] = row
    return row


def run(layer, rows):
    a = layer.self_attn
    full = isinstance(layer, L.FullDecoderLayer)
    B = len(rows)
    cat = lambda k, shp: mx.concatenate([r[k].reshape((1,) + shp) for r in rows], axis=0)
    H, HK = a.num_heads, a.num_key_value_heads
    kw = {}
    if full:
        kw = dict(tau=mx.concatenate([r["tau"].reshape(1) for r in rows]),
                  k_wproj=mx.concatenate([a.k_sconv.weight_proj(r["k_in"]).reshape(1, -1) for r in rows]),
                  v_wproj=mx.concatenate([a.v_sconv.weight_proj(r["v_in"]).reshape(1, -1) for r in rows]))
    out = mid_rows(layer, cat("q", (H * 128,)), cat("r", (H * 16,)), cat("kr", (HK * 128,)), cat("vr", (HK * 128,)),
                   cat("cs0", (3, HK * 128)), cat("cs1", (3, HK * 128)), debug=True, **kw)
    mx.eval(out)
    return out


def compare(layer, rows, refs):
    out = run(layer, rows)
    names = ["q", "k", "v", "rel", "w0", "w1"]
    mis = cnt = 0
    bad = []
    for j, nm in enumerate(names):
        for b, ref in enumerate(refs):
            x = out[j][b].reshape(ref[nm].shape)
            m = nmis(x, ref[nm]); mis += m; cnt += x.size
            if m: bad.append((nm, b, m))
    for j, nm in ((6, "k_conv"), (7, "v_conv")):
        for b, ref in enumerate(refs):
            m = nmis(out[j][b].reshape(ref[nm].shape), ref[nm]); mis += m; cnt += ref[nm].size
            if m: bad.append((nm, b, m))
    return mis, cnt, bad


total = {"cases": 0, "mis": 0, "elems": 0}
table = {}
def rec(kind, B, mis, cnt, bad, tag):
    t = table.setdefault((kind, B), [0, 0, 0])
    t[0] += 1; t[1] += mis; t[2] += cnt
    total["cases"] += 1; total["mis"] += mis; total["elems"] += cnt
    if mis: print("MISMATCH", tag, kind, "B=%d" % B, bad[:6], flush=True)

# ---- real visits
by = {}
for i, v in enumerate(idx):
    by.setdefault((v["layer"], v["has_msg"]), []).append(i)
for (li, hm), ids in sorted(by.items()):
    layer = layers[li]
    kind = ("F" if idx[ids[0]]["full"] else "S") + ("m" if hm else "")
    for i in ids[:2]:  # validate the reference helper against visit_ref's own tensors
        r = real_row(i)
        m = mid_ref(layer, r["q"], r["r"], r["kr"], r["vr"], r["k_in"], r["v_in"], r["cs0"], r["cs1"], r["tau"])
        for k in ("q", "k", "v", "rel", "w0", "w1"):
            assert nmis(m[k], r["T"][k]) == 0, ("mid_ref != visit_ref", li, k)
    for B in BS:
        for off in range(0, min(len(ids), 3)):
            sel = [ids[(off + 2 * j) % len(ids)] for j in range(B)]
            rows = [real_row(i) for i in sel]
            refs = [mid_ref(layer, r["q"], r["r"], r["kr"], r["vr"], r["k_in"], r["v_in"], r["cs0"], r["cs1"], r["tau"]) for r in rows]
            mis, cnt, bad = compare(layer, rows, refs)
            rec("real " + kind, B, mis, cnt, bad, "layer %d off %d" % (li, off))
    print("layer %2d %s done (visits %d): cum mismatches %d" % (li, kind, len(ids), total["mis"]), flush=True)

# ---- random rows (random tau, conv state, input scale), every layer
mx.random.seed(1)
rn = lambda *s: mx.random.normal(s)
for li, layer in enumerate(layers):
    a = layer.self_attn
    full = isinstance(layer, L.FullDecoderLayer)
    H, HK = a.num_heads, a.num_key_value_heads
    for B in BS:
        for scale in (1.0, 6.0, 40.0):
            rows = []
            for _ in range(B):
                rows.append(dict(q=(rn(1, 1, H * 128) * scale).astype(bf16), r=(rn(1, 1, H * 16) * scale).astype(bf16),
                                 kr=(rn(1, 1, HK * 128) * scale).astype(bf16), vr=(rn(1, 1, HK * 128) * scale).astype(bf16),
                                 k_in=(rn(1, 1, 2304) * 2).astype(bf16), v_in=(rn(1, 1, 2304) * 2).astype(bf16),
                                 cs0=rn(1, 3, HK * 128) * scale, cs1=rn(1, 3, HK * 128) * scale,
                                 tau=(1.0 + mx.random.uniform(shape=(1, 1, 1)) * 1.5) if full else None))
            refs = [mid_ref(layer, r["q"], r["r"], r["kr"], r["vr"], r["k_in"], r["v_in"], r["cs0"], r["cs1"], r["tau"]) for r in rows]
            mis, cnt, bad = compare(layer, rows, refs)
            rec("random " + ("F" if full else "S"), B, mis, cnt, bad, "layer %d scale %g" % (li, scale))

print("\n%-12s %3s %6s %12s %10s" % ("source/kind", "B", "cases", "elements", "mismatch"))
for (kind, B), (c, m, e) in sorted(table.items()):
    print("%-12s %3d %6d %12d %10d" % (kind, B, c, e, m))
print("TOTAL cases %d elements %d mismatches %d -> %s" % (total["cases"], total["elems"], total["mis"], "PASS" if total["mis"] == 0 else "FAIL"))
