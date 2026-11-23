"""Timing-only ablation of the batched step (NOT exact): ABL=moe|experts|core|none."""
import os, sys, time, statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import batchdecode as BD, layers as L, fastdecode as FD
abl = os.environ.get("ABL", "none")
if abl == "moe":
    L.SparseMoeBlock.__call__ = lambda self, x: x
if abl == "experts":
    L.Experts.__call__ = lambda self, x, idx: mx.broadcast_to(x[..., None, :], x.shape[:-1] + (idx.shape[-1], x.shape[-1]))
if abl == "core":
    FD._core = lambda layer, q, k, v, pb, sinks, kv: q.transpose(0, 2, 1, 3)
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3")); mx.eval(model.parameters())
ids = tok.encode("Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.")
for C in [1, 8]:
    caches = []
    for i in range(C):
        c = make_prompt_cache(model); model(mx.array(ids[i:-1])[None], cache=c); mx.eval([x.state for x in c]); caches.append(c)
    dec = BD.BatchDecoder(model, caches)
    y = mx.array([ids[-1]] * C)
    for _ in range(4):
        y = dec.step(y); mx.eval(y)
    ts = []
    for _ in range(16):
        t0 = time.perf_counter(); y = dec.step(y); mx.eval(y); ts.append(time.perf_counter() - t0)
    print("ABL=%s C=%d %.2f ms/step" % (abl, C, 1e3 * statistics.median(ts)), flush=True)
