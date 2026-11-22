"""Lane A: reference vs fast decode path. Same prompt, N greedy steps; reports max |logit| diff and token equality per step."""
import argparse, os, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import layers as L
ap = argparse.ArgumentParser(); ap.add_argument("--passes", type=int, default=3); ap.add_argument("--ctx", type=int, default=300); ap.add_argument("--steps", type=int, default=40)
a = ap.parse_args()
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p%d" % a.passes)); mx.eval(model.parameters())
ids = [(17 * i * i + 3 * i) % 40000 + 1000 for i in range(a.ctx)]
def pf():
    c = make_prompt_cache(model); L._FAST_ON = False
    lg = model(mx.array([ids]), cache=c); mx.eval([x.state for x in c]); return c, int(mx.argmax(lg[0, -1]).item())
cr, t0 = pf(); cf, _ = pf(); tr = tf = t0; worst = 0.0; same = True
for s in range(a.steps):
    L._FAST_ON = False; lr = model(mx.array([[tr]]), cache=cr)[0, -1]; mx.eval(lr)
    L._FAST_ON = True; lf = model(mx.array([[tf]]), cache=cf)[0, -1]; mx.eval(lf)
    d = mx.abs(lr - lf).max().item(); worst = max(worst, d); tr = int(mx.argmax(lr).item()); tf = int(mx.argmax(lf).item())
    if s < 6 or tr != tf: print(s, "maxdiff", d, tr, tf)
    same &= tr == tf
    if tr != tf: break
print("passes", a.passes, "worst max|dlogit|", worst, "tokens", "SAME" if same else "DIFF")
