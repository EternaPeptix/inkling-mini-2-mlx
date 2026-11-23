"""Capture real layer visits of the B=1 fast decode path and check fused_ref.visit_ref against them.

  python bench/fused_capture.py [--passes 3] [--steps 6] [--out ~/loopling-opt/runs/<lane>/visits] [--check]

For each of 3 prompts, prefills, then decodes `--steps` tokens with the default fast path while recording, for every
layer visit of the last 2 steps: the visit inputs (h, bs, msg, 4 conv states, tau) and outputs (o, bs_new, 4 new conv
states) plus attn_out of fastdecode._core.  Saves one safetensors file per visit + index.json in --out.
--check: re-run fused_ref.visit_ref on every captured visit and require bitwise equality of out / bs_new / w0..w3.
"""
import argparse, json, os, sys

import mlx_lm  # first: applies LOOPLING_CMDBUF before mlx.core initializes the device
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import fused_ref as FR
from mlx_lm.models.inkling2 import layers as L

PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
    "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
    "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
ap.add_argument("--passes", type=int, default=3)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--keep", type=int, default=2, help="record the last N decode steps")
ap.add_argument("--out", default="")
ap.add_argument("--check", action="store_true")
a = ap.parse_args()

model, tok = load(os.path.join(a.models, "loopling-p%d" % a.passes))
TM = model.model.language_model
layers = list(TM.layers)
lidx = {id(l): i for i, l in enumerate(layers)}

REC = []  # list of dict
recording = [False]
_orig_dl = FD.decode_layer
_orig_core = FD._core
last_attn = [None]


def core(layer, q, k, v, pb, sinks, kv):
    o = _orig_core(layer, q, k, v, pb, sinks, kv)
    last_attn[0] = o
    return o


def decode_layer(layer, h, bs, cache, messages):
    if not recording[0]:
        return _orig_dl(layer, h, bs, cache, messages)
    conv, kv = cache[1], cache[0]
    cs = [mx.array(c) for c in conv.cache]
    tau = None if isinstance(layer, L.SlidingDecoderLayer) else FD._full_tau(layer, FD._end(kv))
    o, bs_new = _orig_dl(layer, h, bs, cache, messages)
    rec = dict(layer=lidx[id(layer)], h=h, bs=bs, msg=messages, cs=cs, tau=tau, attn_out=last_attn[0],
               o=o, bs_new=bs_new, ns=[mx.array(c) for c in conv.cache])
    mx.eval([v for v in rec.values() if isinstance(v, mx.array)] + rec["cs"] + rec["ns"])
    REC.append(rec)
    return o, bs_new


FD.decode_layer = decode_layer
FD._core = core
# layers.py calls fastdecode.decode_layer through the module, so the patch is seen.

for pi, p in enumerate(PROMPTS):
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                     + "<|message_model|><|content_text|>", add_special_tokens=False)
    cache = model.make_cache()
    x = mx.array(ids)
    while x.size > 1:
        n = min(2048, x.size - 1)
        model(x[:n][None], cache=cache)
        mx.eval([c.state for c in cache])
        x = x[n:]
    y = x
    for s in range(a.steps):
        recording[0] = s >= a.steps - a.keep
        start = len(REC)
        lg = model(y[None], cache=cache)
        y = mx.argmax(lg[0, -1:], axis=-1)
        mx.eval(y)
        for j, r in enumerate(REC[start:]):
            r.update(prompt=pi, step=s, visit=j)
recording[0] = False
print("captured %d visits" % len(REC))

if a.out:
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    index = []
    for n, r in enumerate(REC):
        arrs = {"h": r["h"], "attn_out": r["attn_out"], "o": r["o"], "bs_new": r["bs_new"]}
        for i in range(4):
            arrs["cs%d" % i] = r["cs"][i]
            arrs["ns%d" % i] = r["ns"][i]
        for k in ("bs", "msg", "tau"):
            if r[k] is not None:
                arrs[k] = r[k]
        fn = "v%04d.safetensors" % n
        mx.save_safetensors(os.path.join(out, fn), arrs)
        index.append(dict(file=fn, layer=r["layer"], prompt=r["prompt"], step=r["step"], visit=r["visit"],
                          has_bs=r["bs"] is not None, has_msg=r["msg"] is not None, full=r["tau"] is not None,
                          boundary=bool(layers[r["layer"]].is_block_boundary)))
    json.dump(dict(passes=a.passes, model="loopling-p%d" % a.passes, visits=index), open(os.path.join(out, "index.json"), "w"), indent=1)
    print("saved", out)


def bits(x):
    return x.view(mx.uint32) if x.dtype == mx.float32 else x.view(mx.uint16)


def same(x, y):
    return x.shape == y.shape and x.dtype == y.dtype and bool(mx.array_equal(bits(x), bits(y)))


if a.check:
    bad = 0
    kinds = {}
    for r in REC:
        layer = layers[r["layer"]]
        T = FR.visit_ref(layer, r["h"], r["bs"], r["msg"], r["cs"], r["attn_out"], r["tau"])
        ok = same(T["out"], r["o"]) and same(T["bs_new"], r["bs_new"])
        ok = ok and all(same(T["w%d" % i], r["ns"][i]) for i in range(4))
        kind = ("F" if r["tau"] is not None else "S") + ("b" if layer.is_block_boundary else "") + ("m" if r["msg"] is not None else "")
        kinds.setdefault(kind, [0, 0])[0 if ok else 1] += 1
        if not ok:
            bad += 1
            if bad <= 5:
                diff = [k for k, x, y in [("out", T["out"], r["o"]), ("bs", T["bs_new"], r["bs_new"])] + [
                    ("w%d" % i, T["w%d" % i], r["ns"][i]) for i in range(4)] if not same(x, y)]
                print("MISMATCH layer %d kind %s: %s" % (r["layer"], kind, diff))
    print("visit kinds (ok, bad):", kinds)
    print("CHECK", "PASS" if bad == 0 else "FAIL", "%d/%d" % (len(REC) - bad, len(REC)))
