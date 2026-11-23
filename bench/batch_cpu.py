"""Pipelined step anatomy: build (python graph) / encode (async_eval) / wait (GPU tail)."""
import os, sys, time, statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import batchdecode as BD
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p%s" % os.environ.get("P", "3"))); mx.eval(model.parameters())
ids = tok.encode("Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.")
for C in [1, 4, 8]:
    caches = []
    for i in range(C):
        c = make_prompt_cache(model); model(mx.array(ids[i:-1])[None], cache=c); mx.eval([x.state for x in c]); caches.append(c)
    dec = BD.BatchDecoder(model, caches)
    y = mx.array([ids[-1]] * C)
    for _ in range(4):
        y = dec.step(y); mx.eval(y)
    y = dec.step(y); mx.async_eval(y)
    B, E, W, T = [], [], [], []
    for _ in range(24):
        t0 = time.perf_counter(); ny = dec.step(y); t1 = time.perf_counter(); mx.async_eval(ny); t2 = time.perf_counter()
        y.tolist(); t3 = time.perf_counter(); y = ny
        B.append(t1 - t0); E.append(t2 - t1); W.append(t3 - t2); T.append(t3 - t0)
    m = lambda v: 1e3 * statistics.median(v)
    print("C=%d build %.2f encode %.2f wait %.2f | step %.2f ms" % (C, m(B), m(E), m(W), m(T)), flush=True)
