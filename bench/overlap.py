"""Does GPU work of step k run while the CPU builds/encodes step k+1? Sleep S ms after async_eval and see if wait drops."""
import os, sys, time, statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import batchdecode as BD
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p%s" % os.environ.get("P", "3"))); mx.eval(model.parameters())
ids = tok.encode("Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.")
C = int(os.environ.get("C", "1"))
caches = []
for i in range(C):
    c = make_prompt_cache(model); model(mx.array(ids[i:-1])[None], cache=c); mx.eval([x.state for x in c]); caches.append(c)
dec = BD.BatchDecoder(model, caches)
y = mx.array([ids[-1]] * C)
for _ in range(4):
    y = dec.step(y); mx.eval(y)
for sl in [0, 5, 10, 20]:
    y = dec.step(y); mx.async_eval(y)
    W, T = [], []
    for _ in range(16):
        t0 = time.perf_counter(); ny = dec.step(y); mx.async_eval(ny); t2 = time.perf_counter()
        if sl: time.sleep(sl / 1e3)
        t2b = time.perf_counter(); y.tolist(); t3 = time.perf_counter(); y = ny
        W.append(t3 - t2b); T.append(t3 - t0)
    print("C=%d sleep %2d ms: wait after sleep %.2f ms, step %.2f ms" % (C, sl, 1e3 * statistics.median(W), 1e3 * statistics.median(T)), flush=True)
    mx.eval(y)
# single isolated step: encode then wait
for _ in range(3):
    mx.eval(y)
    t0 = time.perf_counter(); ny = dec.step(y); t1 = time.perf_counter(); mx.async_eval(ny); t2 = time.perf_counter(); ny.tolist(); t3 = time.perf_counter()
    print("isolated: build %.2f encode %.2f gpu-tail %.2f" % (1e3*(t1-t0), 1e3*(t2-t1), 1e3*(t3-t2)))
    y = ny
