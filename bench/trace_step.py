"""Short batched decode for GPU tracing (xctrace Metal System Trace): warm up, then STEPS steps with a marker gap."""
import mlx_lm  # first
import os, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import batchdecode as BD
C = int(os.environ.get("C", "1")); STEPS = int(os.environ.get("STEPS", "6"))
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
BD.install()
ids = tok.encode("Write a Python function that merges two sorted lists.")
caches = []
for i in range(C):
    c = model.make_cache(); p = mx.array(ids)
    model(p[:-1][None], cache=c); mx.eval([x.state for x in c]); caches.append(c)
dec = BD.BatchDecoder(model, caches)
y = mx.array([ids[-1]] * C)
for _ in range(8):
    y = dec.step(y); mx.eval(y)
time.sleep(0.5)
t = time.perf_counter()
for _ in range(STEPS):
    y = dec.step(y); mx.async_eval(y)
mx.eval(y)
print("C=%d %.2f ms/step" % (C, (time.perf_counter() - t) * 1e3 / STEPS))
