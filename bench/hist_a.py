"""Lane A: primitive histogram of one decode graph, per layer type (sliding / full) and for the whole step."""
import argparse, collections, os, re, tempfile
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser(); ap.add_argument("--passes", type=int, default=3); ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
a = ap.parse_args()
model, tok = load("%s/loopling-p%d" % (a.models, a.passes)); mx.eval(model.parameters())
cache = make_prompt_cache(model)
ids = [(17 * i * i + 3 * i) % 40000 + 1000 for i in range(300)]
model(mx.array([ids]), cache=cache); mx.eval([c.state for c in cache])
y = mx.array([[5]])
def hist(out):
    f = tempfile.NamedTemporaryFile("w+", suffix=".dot"); mx.export_to_dot(f.name, out); f.seek(0); txt = f.read()
    if os.environ.get("DOTDBG"): print(txt[:1200])
    c = collections.Counter()
    for l in txt.splitlines():
        m = re.match(r"\s*\{ (\d+) \[label =\"([^\"]*)\"", l)
        if m: c[m.group(2)] += 1
    return c
for _ in range(3): mx.eval(model(y, cache=cache))
tot = hist(model(y, cache=cache))
print("WHOLE STEP nodes", sum(tot.values())); print(tot.most_common(40))
