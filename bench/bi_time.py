"""Lane D: time T x 1-token forwards vs the default T-row forward vs the batch-invariant T-row forward (ms, median).
usage: bi_time.py --passes 3 --Ts 2,3,4,5 [--profile-ops]"""
import argparse, os, statistics, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import batch_invariant as bi

ap = argparse.ArgumentParser(); ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
ap.add_argument("--passes", type=int, default=3); ap.add_argument("--Ts", default="2,3,4,5"); ap.add_argument("--reps", type=int, default=12)
ap.add_argument("--ctx", type=int, default=300); ap.add_argument("--ops", default=None, help="LOOPLING_BI_OPS subset to time")
a = ap.parse_args()
if a.ops is not None: os.environ["LOOPLING_BI_OPS"] = a.ops
model, tok = load("%s/loopling-p%d" % (a.models, a.passes)); mx.eval(model.parameters())
ids = mx.array([(17 * i * i + 3 * i) % 40000 + 1000 for i in range(a.ctx)], mx.uint32)
c = make_prompt_cache(model); model(ids[None], cache=c); mx.eval([x.state for x in c])
toks = [1234, 5678, 910, 1112, 1314, 1516, 1718, 1920]


def timeit(fn):
    for _ in range(3): fn()
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts), max(ts)


def seq(T):
    def f():
        outs = [model(mx.array([[toks[j]]], mx.uint32), cache=c)[0, 0] for j in range(T)]
        mx.eval(mx.stack(outs).argmax(-1))
    return f


def rows(T, inv):
    def f():
        if inv:
            with bi.active(): lg = model(mx.array([toks[:T]], mx.uint32), cache=c)
        else:
            lg = model(mx.array([toks[:T]], mx.uint32), cache=c)
        mx.eval(lg[0].argmax(-1))
    return f


print("passes %d ctx %d flag %s fast %s ops %s" % (a.passes, a.ctx, os.environ.get("LOOPLING_BATCH_INVARIANT"), os.environ.get("LOOPLING_COMPILE_DECODE"), a.ops))
m1 = timeit(seq(1))
print("1 token: %.1f ms (min %.1f max %.1f)" % m1)
for T in map(int, a.Ts.split(",")):
    s, d = timeit(seq(T)), timeit(rows(T, False))
    r = {"seq": s, "default": d}
    if os.environ.get("LOOPLING_BATCH_INVARIANT") == "1": r["invariant"] = timeit(rows(T, True))
    print("T=%d  " % T + "  ".join("%s %.1f ms (%.1f-%.1f)" % (k, *v) for k, v in r.items()), flush=True)
