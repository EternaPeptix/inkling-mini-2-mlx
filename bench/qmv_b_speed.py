"""Speed of the qmv-family kernels vs B (lane g1). median us/call, queued (32 launches per eval) and dependent chain.

usage: qmv_b_speed.py [--cfg rps,nsg,rb[,stg] ...] [--only substr,...] [--bs 1,2,4,8,16]
(--cfg sets LOOPLING_QMV_CFG for tuning; without it the module's table is used)
"""
import argparse, os, statistics, time
import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2.rows import proj as P, qmv as Q

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", action="append", default=[])
ap.add_argument("--only", default="")
ap.add_argument("--bs", default="1,2,4,8,16")
ap.add_argument("--reps", type=int, default=30)
args = ap.parse_args()
Bs = [int(b) for b in args.bs.split(",")]
only = [t for t in args.only.split(",") if t]

model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
Ls = model.model.language_model.layers
sl, fl = Ls[7].self_attn, Ls[5].self_attn
W = lambda m: (m.weight, m.scales, m.biases)
mx.random.seed(0)


def rnd(*s):
    return mx.random.normal(s).astype(mx.bfloat16)


def queued(fn, reps, q=32):
    for _ in range(3): mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval([fn() for _ in range(q)]); ts.append((time.perf_counter() - t) / q * 1e6)
    return statistics.median(ts)


def chain(step, reps, n=32):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); y = None
        for _ in range(n): y = step(y)
        mx.eval(y); ts.append((time.perf_counter() - t) / n * 1e6)
    return statistics.median(ts)


def bump(base, y):  # cheap dependent perturbation: base + first column of y
    if y is None: return base
    return base + y.reshape(-1)[:1]


cases = []  # (name, make(B) -> (fn, step))
def add(name, mk): cases.append((name, mk))

for nm, m, at in [("q", sl.q_proj, sl), ("k", sl.k_proj, sl), ("r", sl.r_proj, sl), ("gate", sl.gate_proj, sl),
                  ("sink", sl.sink_proj, sl), ("o", sl.o_proj, sl)]:
    def mk(B, m=m):
        K = m.weight.shape[1] * 4; x = rnd(B, K); mx.eval(x)
        return (lambda: Q.qmv_rows(x, *W(m))), (lambda y: Q.qmv_rows(bump(x, y), *W(m)))
    add("qmv_rows " + nm, mk)

def mk_multi(B):
    ms = [sl.q_proj, sl.r_proj, sl.gate_proj, sl.sink_proj]; x = rnd(B, 2304); mx.eval(x)
    return (lambda: Q.qmv_rows_multi(x, [W(m) for m in ms])), (lambda y: Q.qmv_rows_multi(bump(x, y[0] if y else None), [W(m) for m in ms]))
def _multi_step(B):
    f, s = mk_multi(B)
    return f, (lambda y: s(None if y is None else y))
add("qmv_rows_multi q+r+gate+sink", lambda B: (lambda f, x=None: (f[0], lambda y, f=f: (lambda xx: f[0]())(0)))(mk_multi(B)))

def mk_multi2(B):
    ms = [sl.q_proj, sl.r_proj, sl.gate_proj, sl.sink_proj]; x = rnd(B, 2304); mx.eval(x)
    return (lambda: Q.qmv_rows_multi(x, [W(m) for m in ms])[0]), (lambda y: Q.qmv_rows_multi(bump(x, y), [W(m) for m in ms])[0])
cases[-1] = ("qmv_rows_multi q+r+gate+sink", mk_multi2)

def mk_mx(B):
    ms = [sl.k_proj, sl.v_proj]; xs = [rnd(B, 2304), rnd(B, 2304)]; mx.eval(xs)
    return (lambda: P.qmv_rows_multi_x(xs, [W(m) for m in ms], flat=True)), \
        (lambda y: P.qmv_rows_multi_x([bump(xs[0], y), xs[1]], [W(m) for m in ms], flat=True))
add("qmv_rows_multi_x k+v", mk_mx)

for tag, at in (("sliding", sl), ("full", fl)):
    def mk(B, at=at):
        a, b, c = rnd(B, 2304), rnd(B, 2304), rnd(B, 2304); mx.eval(a, b, c)
        return (lambda: P.qkv_rows(at, a, b, c, flat=True)[0]), (lambda y: P.qkv_rows(at, bump(a, y), b, c, flat=True)[0])
    add("qkv_rows " + tag, mk)

def mk_o(B):
    ao, gt = rnd(B, 3072), rnd(B, 3072) * 3; mx.eval(ao, gt)
    m = sl.o_proj
    return (lambda: P.oproj_rows(ao, gt, *W(m))), (lambda y: P.oproj_rows(bump(ao, y), gt, *W(m)))
add("oproj_rows", mk_o)

cfgs = args.cfg or [None]
for cfg in cfgs:
    if cfg is None: os.environ.pop("LOOPLING_QMV_CFG", None)
    else: os.environ["LOOPLING_QMV_CFG"] = cfg
    for name, mk in cases:
        if only and not any(t in name for t in only): continue
        for B in Bs:
            fn, step = mk(B)
            q = queued(fn, args.reps); c = chain(step, max(8, args.reps // 2))
            print("cfg=%-10s %-32s B=%2d queued %7.1f chain %7.1f" % (cfg or "table", name, B, q, c), flush=True)
