"""Primitive inventory of one batched decode step (export_to_dot, counted by type)."""
import os, re, sys, collections
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import batchdecode as BD
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3")); mx.eval(model.parameters())
ids = tok.encode("Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.")
for C in [1, 8]:
    caches = []
    for i in range(C):
        c = make_prompt_cache(model); model(mx.array(ids[i:-1])[None], cache=c); mx.eval([x.state for x in c]); caches.append(c)
    dec = BD.BatchDecoder(model, caches)
    y = mx.array([ids[-1]] * C)
    for _ in range(3):
        y = dec.step(y); mx.eval(y)
    ny = dec.step(y)
    outs = [ny] + [a for v in dec.conv for a in v] + [c[v][0].keys for c in caches for v in range(len(caches[0]))]
    f = "/tmp/bd_C%d.dot" % C
    mx.export_to_dot(f, *outs)
    txt = open(f).read()
    labels = re.findall(r'label\s*=\s*"([^"]+)"', txt)
    cnt = collections.Counter(re.sub(r"[\s(].*", "", l) for l in labels)
    tot = sum(v for k, v in cnt.items() if k and not k[0].islower())
    print("C=%d nodes %d" % (C, len(labels)))
    print("  " + ", ".join("%s:%d" % kv for kv in cnt.most_common(40)), flush=True)
    mx.eval(ny)
