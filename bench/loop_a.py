"""Lane A: steady async decode loop for external sampling (sample(1))."""
import argparse, os, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
ap = argparse.ArgumentParser(); ap.add_argument("--passes", type=int, default=3); ap.add_argument("--steps", type=int, default=600); a = ap.parse_args()
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p%d" % a.passes)); mx.eval(model.parameters())
c = make_prompt_cache(model); ids = [(17 * i * i + 3 * i) % 40000 + 1000 for i in range(300)]
model(mx.array([ids]), cache=c); mx.eval([x.state for x in c])
def step(y):
    lg = model(y, cache=c)[:, -1, :]; return mx.argmax(lg - mx.logsumexp(lg, keepdims=True), axis=-1).reshape(1, 1)
y = step(mx.array([[5]])); mx.async_eval(y); print("READY", flush=True); t0 = time.time()
for i in range(a.steps):
    n = step(y); mx.async_eval(n); y.item(); y = n
print("ms/tok", 1e3 * (time.time() - t0) / a.steps, flush=True)
