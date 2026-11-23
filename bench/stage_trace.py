"""Per-stage GPU time of the fused batched decode (profiling only, NOT the production path).

Run under `xcrun xctrace record --template "Metal System Trace"` with LOOPLING_COMPILE_DECODE=2 LOOPLING_BD_SYNC=1
MLX_MAX_OPS_PER_BUFFER=0: every stage is evaluated alone followed by a 400 us idle gap; the stage names of the timed
steps go to $LOGOUT (json). bench/stage_parse.py then attributes GPU intervals to stages.
"""
import mlx_lm  # first
import json, os, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import batchdecode as BD
from mlx_lm.models.inkling2.rows import visit as RV
C = int(os.environ.get("C", "1")); STEPS = int(os.environ.get("STEPS", "2"))
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
BD.install()
ids = tok.encode("Write a Python function that merges two sorted lists.")
caches = []
for i in range(C):
    c = model.make_cache(); p = mx.array(ids)
    model(p[:-1][None], cache=c); mx.eval([x.state for x in c]); caches.append(c)
dec = BD.BatchDecoder(model, caches)
y = mx.array([ids[-1]] * C)
for _ in range(6):
    y = dec.step(y); mx.eval(y)
time.sleep(0.5)
RV.LOG.clear()
for _ in range(STEPS):
    y = dec.step(y); t0 = time.perf_counter_ns(); mx.eval(y)
    RV.LOG.append(("tail", t0, time.perf_counter_ns())); time.sleep(0.0004)
json.dump(RV.LOG, open(os.environ["LOGOUT"], "w"))
print("logged", len(RV.LOG))
