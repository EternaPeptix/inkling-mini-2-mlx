"""Exactness test for rows/gather.py: every (token, expert) output of gather_rows is bitwise == gather_qmm for that token alone
(the B=1 decode shapes of Experts.__call__). Real Loopling weights, random + real activations, random/adversarial routing."""
import os, sys, itertools
os.environ["LOOPLING_COMPILE_DECODE"] = "0"
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.inkling2 import layers as L
from mlx_lm.models.inkling2.rows.gather import gather_rows

P = os.path.expanduser("~/loopling/models/loopling-p3")
LAYERS = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 6, 12, 19, 20, 23]
BS = [1, 2, 3, 5, 8, 16]
model, tok = load(P)
TM = model.model.language_model
experts = {i: TM.layers[i].mlp.experts for i in LAYERS}

# ---- capture real decode activations (x (1,1,K) and indices (1,1,8)) per layer ----
cap = {i: [] for i in LAYERS}
orig = L.Experts.__call__
mod2layer = {id(experts[i]): i for i in LAYERS}


def hook(self, x, indices):
    i = mod2layer.get(id(self))
    if i is not None and x.shape[:2] == (1, 1) and len(cap[i]) < 40:
        mx.eval(x, indices)
        cap[i].append((x.reshape(-1), indices.reshape(-1).astype(mx.uint32)))
    return orig(self, x, indices)


L.Experts.__call__ = hook
ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Explain how a bloom filter works, briefly."}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>", add_special_tokens=False)
for _ in stream_generate(model, tok, ids, max_tokens=24):
    pass
L.Experts.__call__ = orig
print("captured", {i: len(v) for i, v in cap.items()}, flush=True)


def bits(a):
    return a.view(mx.uint16)


def ref_one(proj, xrow, irow):
    """xrow (K,) [gate] or (E,K) [down]; irow (E,) -> (E,N), exactly the decode-path call for one token."""
    E = irow.shape[0]
    xx = xrow.reshape(1, 1, 1, 1, -1) if xrow.ndim == 1 else xrow.reshape(1, 1, E, 1, -1)
    o = proj(xx, irow.reshape(1, 1, E), sorted_indices=False)
    return o.reshape(E, -1)


def rows(proj, x, idx):
    return gather_rows(x, proj["weight"], proj["scales"], proj.get("biases"), idx, mode=proj.mode,
                       group_size=proj.group_size, bits=proj.bits, global_scale=proj.global_scale)


def route(kind, B, rng):
    E = 8
    if kind == "random":
        return mx.stack([mx.array(rng.choice(128, E, replace=False).astype("uint32")) for _ in range(B)])
    if kind == "same":
        r = mx.array(rng.choice(128, E, replace=False).astype("uint32"))
        return mx.stack([r] * B)
    if kind == "distinct":
        perm = rng.permutation(128).astype("uint32")
        return mx.stack([mx.array(perm[b * E:(b + 1) * E]) for b in range(B)])
    if kind == "partial":
        pool = rng.choice(128, 12, replace=False)
        return mx.stack([mx.array(rng.choice(pool, E, replace=False).astype("uint32")) for _ in range(B)])
    if kind == "dupintok":  # same expert twice inside one token (not produced by top-k; robustness)
        return mx.stack([mx.array(rng.choice(128, 4, replace=False).astype("uint32").repeat(2)) for _ in range(B)])


import numpy as np

rng = np.random.default_rng(0)
mx.random.seed(0)
tot = bad = ncase = 0
for li in LAYERS:
    ex = experts[li]
    print("layer %d: mode=%s bits=%d gs=%d global_scale=%s biases=%s" % (li, ex.gate_up_proj.mode, ex.gate_up_proj.bits, ex.gate_up_proj.group_size, ex.gate_up_proj.global_scale is not None, ex.gate_up_proj.get("biases") is not None), flush=True)
    for pname in ("gate_up_proj", "down_proj"):
        proj = getattr(ex, pname)
        K = proj.input_dims
        scale_gate = 1.0
        for src in ("rand", "real"):
            if src != "rand" and not cap[li]:
                continue
            for kind in (("random", "same", "distinct", "partial", "dupintok") if src == "rand" else ("real",)):
                for B in BS:
                    if src == "rand":
                        idx = route(kind, B, rng)
                        if pname == "gate_up_proj":
                            x = (mx.random.normal((B, K)) * 1.0).astype(mx.bfloat16)
                        else:
                            x = (mx.random.normal((B, 8, K)) * 0.5).astype(mx.bfloat16)
                    else:
                        cs = [cap[li][(b * 2) % len(cap[li])] for b in range(B)]
                        idx = mx.stack([c[1] for c in cs])
                        xg = mx.stack([c[0] for c in cs])
                        if pname == "gate_up_proj":
                            x = xg
                        else:  # real down input = reference swiglu of reference gate_up
                            acts = []
                            for b in range(B):
                                gu = ref_one(ex.gate_up_proj, xg[b], idx[b])
                                acts.append(ex.act_fn(gu[..., 0::2], gu[..., 1::2]))
                            x = mx.stack(acts)
                    out = rows(proj, x, idx)
                    refo = mx.stack([ref_one(proj, x[b], idx[b]) for b in range(B)])
                    mx.eval(out, refo)
                    mism = int(mx.sum(bits(out) != bits(refo)).item())
                    ok = mism == 0 and out.shape == refo.shape
                    tot += out.size
                    bad += mism
                    ncase += 1
                    if not ok or B in (1, 5, 16):
                        print("%s L%02d %-12s %-9s %-8s B=%-2d mism=%d/%d" % ("PASS" if ok else "FAIL", li, pname, src, kind, B, mism, out.size), flush=True)
                    if not ok:
                        sys.stdout.flush()
print("SUMMARY cases=%d elements=%d mismatches=%d -> %s" % (ncase, tot, bad, "PASS" if bad == 0 else "FAIL"))
sys.exit(0 if bad == 0 else 1)
