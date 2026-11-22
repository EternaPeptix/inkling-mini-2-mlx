"""Lane D: batch-invariance comparison tool.
Runs a T-token forward (decode time, existing KV cache) and T sequential 1-token forwards from identical cache
states, traces every op (module __call__) of both, and reports per op class the first call where the T-row result
differs bitwise from the per-row result, flagging SOURCES (inputs bitwise equal, output differs).
usage: bi_compare.py --passes 3 -T 4 [--ctx 300] [--window N]   (honours LOOPLING_BATCH_INVARIANT)
"""
import argparse, os, collections
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.layers import moe as moe_mod
from mlx_lm.models.inkling2 import layers as L

ap = argparse.ArgumentParser(); ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
ap.add_argument("--passes", type=int, default=3); ap.add_argument("-T", type=int, default=4)
ap.add_argument("--ctx", type=int, default=300); ap.add_argument("--window", type=int, default=0)
ap.add_argument("--quiet", action="store_true"); a = ap.parse_args()
model, tok = load("%s/loopling-p%d" % (a.models, a.passes)); mx.eval(model.parameters())
if a.window:
    model.args.text.sliding_window_size = a.window
    for l in model.model.language_model.layers:
        if hasattr(l.self_attn, "sliding_window"): l.self_attn.sliding_window = a.window
T = a.T
ids = mx.array([(17 * i * i + 3 * i) % 40000 + 1000 for i in range(a.ctx)], mx.uint32)
text = "Write a short story about a lighthouse keeper."
tids = tok.encode(text, add_special_tokens=False)[: T + 2]

REC = None
def arrays(x):
    if isinstance(x, mx.array): return [x]
    if isinstance(x, (list, tuple)): return [y for e in x for y in arrays(e)]
    return []
PATH = {id(m): n for n, m in model.named_modules()}
def wrap(cls, name):
    orig = cls.__call__
    def f(self, *args, **kw):
        out = orig(self, *args, **kw)
        if REC is not None:
            REC.append((name + " " + PATH.get(id(self), "?"), arrays(list(args)), arrays(out)))
        return out
    cls.__call__ = f
def wrap_m(cls, meth, name):
    orig = getattr(cls, meth)
    def f(self, *args, **kw):
        out = orig(self, *args, **kw)
        if REC is not None:
            REC.append((name, [x for x in arrays(list(args)) if x.ndim >= 3][:3], arrays(out)))
        return out
    setattr(cls, meth, f)
for cls in (nn.Linear, nn.QuantizedLinear, nn.RMSNorm, nn.Embedding, nn.QuantizedEmbedding, nn.Conv1d,
            moe_mod.QuantizedSwitchLinear, moe_mod.SwitchLinear, L.ShortConvolution, L.DynamicShortConvolution,
            L.MessageGate, L.BlockAttentionResidual, L.MessageInput, L.TopKRouter, L.Experts, L.SparseMoeBlock,
            L.ScaledL2Clip, L.L2Clip, L.RelativeLogits):
    wrap(cls, cls.__name__)
wrap_m(L.Attention, "_attend", "Attention._attend")
wrap_m(L.DecoderLayer, "__call__", "DecoderLayer")

def fresh():
    c = make_prompt_cache(model)
    for s in range(0, a.ctx, 512):
        model(ids[s:s + 512][None], cache=c); mx.eval([x.state for x in c])
    return c
def fwd(c, toks):
    global REC
    REC = []
    from mlx_lm.models.inkling2 import batch_invariant as bi
    with (bi.active() if len(toks) > 1 else __import__('contextlib').nullcontext()):
        lg = model(mx.array(toks, mx.uint32)[None], cache=c)
    recs = REC; REC = None
    flat = [x for r in recs for x in r[1] + r[2]]
    mx.eval(lg, flat)
    return lg, recs

toks = tids[:T]
cs = fresh(); seq = []
for t in toks: seq.append(fwd(cs, [t]))
ct = fresh(); lgT, recT = fwd(ct, toks)

def rowaxis(sT, s1):
    if len(sT) != len(s1): return None
    for ax, (x, y) in enumerate(zip(sT, s1)):
        if x == T and y == 1 and all(p == q for i, (p, q) in enumerate(zip(sT, s1)) if i != ax): return ax
    return None
def rowof(x, ax, j): return mx.take(x, mx.array([j]), axis=ax)
def eq(x, y): return x.shape == y.shape and bool(mx.array_equal(x, y).item())

nrec = len(recT); assert all(len(s[1]) == nrec for s in seq), "op trace length differs"
first = collections.OrderedDict(); sources = collections.OrderedDict(); ndiff = collections.Counter(); nsrc = collections.Counter()
for i in range(nrec):
    name, insT, outT = recT[i]; full = name; name = name.split()[0]
    for j in range(T):
        _, ins1, out1 = seq[j][1][i]
        if len(outT) != len(out1): continue
        diff = False; inok = True
        for xT, x1 in zip(outT, out1):
            ax = rowaxis(xT.shape, x1.shape)
            if ax is not None and not eq(rowof(xT, ax, j), x1): diff = True
        if not diff: continue
        for xT, x1 in zip(insT, ins1):
            ax = rowaxis(xT.shape, x1.shape)
            if ax is not None and not eq(rowof(xT, ax, j), x1): inok = False
        ndiff[name] += 1; first.setdefault(name, (i, j, full))
        if inok: nsrc[name] += 1; sources.setdefault(name, (i, j, full))
print("T=%d passes=%d ctx=%d flag=%s ops=%d" % (T, a.passes, a.ctx, os.environ.get("LOOPLING_BATCH_INVARIANT", "0"), nrec))
for n in first: print("  %-24s first diff #%d row %d %s; pairs %d; SOURCE (equal inputs) %d %s" % (n, *first[n], ndiff[n], nsrc[n], sources.get(n, ("", "", ""))[2]))
# final logits per row
res = []
for j in range(T):
    l1 = seq[j][0][0, 0]; lt = lgT[0, j]
    res.append("row%d: %s maxdiff %.4f" % (j, "BITWISE" if eq(lt, l1) else "DIFF", mx.abs(lt - l1).max().item()))
print("  logits:", "; ".join(res))
# cache writes
bad = 0
for k, (c1, c2) in enumerate(zip(cs, ct)):
    for part in (0,):
        k1, k2 = c1[0].state[0], c2[0].state[0]
        v1, v2 = c1[0].state[1], c2[0].state[1]
        n = min(k1.shape[2], k2.shape[2])
        if not (eq(k1[..., -T:, :], k2[..., -T:, :]) and eq(v1[..., -T:, :], v2[..., -T:, :])): bad += 1
    for w1, w2 in zip(c1[1].cache, c2[1].cache):
        if not eq(w1, w2): bad += 1
print("  cache entries (layer visits) with mismatching last-T KV or conv state: %d / %d" % (bad, len(cs)))
