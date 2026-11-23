"""Exactness of rows/qmv.py vs the per-row M=1 mx.quantized_matmul on every affine-8bit linear of loopling-p3."""
import os, sys, collections
os.environ.setdefault("LOOPLING_COMPILE_DECODE", "0")
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.models.inkling2.rows.qmv import qmv_rows, qmv_rows_multi

BS = [1, 2, 3, 5, 8, 16]
mx.random.seed(7)
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))

mods = {}
for name, m in model.named_modules():
    if isinstance(m, nn.QuantizedLinear) and m.bits == 8 and m.mode == "affine" and m.group_size == 64:
        mods[name] = m
print("affine-8bit QuantizedLinear modules:", len(mods))
shapes = collections.Counter((m.weight.shape[0], m.weight.shape[1] * 4) for m in mods.values())
print("distinct (N, K):", dict(shapes))

# capture real decode activations
cap = collections.defaultdict(list)
idmap = {id(m): n for n, m in mods.items()}
orig = nn.QuantizedLinear.__call__
def hooked(self, x):
    n = idmap.get(id(self))
    if n is not None and x.shape[-2] == 1 and len(cap[n]) < 16:
        cap[n].append(x.reshape(-1))
    return orig(self, x)
nn.QuantizedLinear.__call__ = hooked
generate(model, tok, "Write a short poem about the sea and explain its rhyme scheme.", max_tokens=24, verbose=False)
nn.QuantizedLinear.__call__ = orig
print("captured activations for", len(cap), "modules; min rows", min(len(v) for v in cap.values()) if cap else 0)

def bits(a): return a.view(mx.uint16)
def ref_rows(m, X):
    K = X.shape[1]
    outs = [mx.quantized_matmul(X[i].reshape(1, 1, K), m.weight, m.scales, m.biases, transpose=True,
                                group_size=64, bits=8, mode="affine").reshape(1, -1) for i in range(X.shape[0])]
    return mx.concatenate(outs)

def inputs(name, K):
    g = mx.random.normal((16, K)).astype(mx.bfloat16)
    z = mx.random.normal((16, K)) * 3.0
    z = z * mx.rsqrt(mx.mean(z * z, axis=-1, keepdims=True) + 1e-6)
    lnw = 0.5 + mx.random.uniform(shape=(K,)) * 1.5
    d = {"randn": g, "rmsnorm*w": (z * lnw).astype(mx.bfloat16)}
    if name in cap and len(cap[name]) >= 1:
        rows = cap[name]
        while len(rows) < 16: rows = rows + rows
        d["real"] = mx.stack(rows[:16]).astype(mx.bfloat16)
    return d

total_bad = 0; ncase = 0; per_shape = collections.defaultdict(lambda: [0, 0])
fails = []
for name, m in mods.items():
    N = m.weight.shape[0]; K = m.weight.shape[1] * 4
    for kind, X in inputs(name, K).items():
        ref = ref_rows(m, X)
        outs = {B: qmv_rows(X[:B], m.weight, m.scales, m.biases) for B in BS}
        mx.eval(ref, *outs.values())
        for B in BS:
            bad = int(mx.sum(bits(outs[B]) != bits(ref[:B])).item())
            ncase += 1; total_bad += bad
            per_shape[(N, K)][0] += 1; per_shape[(N, K)][1] += bad
            if bad: fails.append((name, kind, B, bad)); print("FAIL", name, kind, "B", B, "bad", bad)
    print("PASS" if not any(f[0] == name for f in fails) else "FAIL", name, (N, K), flush=True)

# multi variants: share the input of the first weight
def multi_case(label, names):
    global total_bad, ncase
    ms = [mods[n] for n in names]
    K = ms[0].weight.shape[1] * 4
    for kind, X in inputs(names[0], K).items():
        refs = [ref_rows(m, X) for m in ms]
        for B in BS:
            outs = qmv_rows_multi(X[:B], [(m.weight, m.scales, m.biases) for m in ms])
            mx.eval(refs, outs)
            bad = sum(int(mx.sum(bits(o) != bits(r[:B])).item()) for o, r in zip(outs, refs))
            ncase += 1; total_bad += bad
            if bad: fails.append((label, kind, B, bad)); print("FAIL multi", label, kind, B, bad)
    print("PASS multi", label, flush=True)
L = "model.language_model.layers.%d.self_attn."
for i in range(24):
    multi_case("qrgs L%d" % i, [L % i + p for p in ("q_proj", "r_proj", "gate_proj", "sink_proj")])
    multi_case("kv L%d" % i, [L % i + p for p in ("k_proj", "v_proj")])
    kw = [n for n in (L % i + "k_sconv.weight_proj", L % i + "v_sconv.weight_proj") if n in mods]
    if kw: multi_case("kvsconv L%d" % i, kw)

# synthetic: partial_rows (N % 8 != 0), tiny N -- not present in the model, covered for generality
class M: pass
for N, K in [(36, 2304), (5, 2304), (1, 512), (13, 3072), (24, 2304)]:
    w = (mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=8)
    m = M(); m.weight, m.scales, m.biases = wq, s, b
    X = mx.random.normal((16, K)).astype(mx.bfloat16)
    ref = ref_rows(m, X)
    nb = 0
    for B in BS:
        o = qmv_rows(X[:B], wq, s, b); mx.eval(o, ref)
        bad = int(mx.sum(bits(o) != bits(ref[:B])).item()); ncase += 1; total_bad += bad; nb += bad
        if bad: fails.append(("synthetic", N, K, B, bad))
    print("PASS" if nb == 0 else "FAIL", "synthetic", (N, K), flush=True)

print("--- per-shape (cases, mismatching elements)")
for k, v in sorted(per_shape.items()): print(k, v)
print("SUMMARY: cases %d, mismatching elements %d, %s" % (ncase, total_bad, "ALL PASS" if total_bad == 0 else "FAIL"))
sys.exit(1 if total_bad else 0)
