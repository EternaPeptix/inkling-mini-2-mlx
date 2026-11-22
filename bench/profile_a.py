"""Lane A: decode-step profile. Per-token wall (sync / async), CPU-vs-wall, build-vs-eval, per layer-group timing, graph size."""
import argparse, json, os, resource, statistics as st, tempfile, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2.layers import unweighted_rms_norm

ap = argparse.ArgumentParser(); ap.add_argument("--passes", type=int, default=3); ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--steps", type=int, default=40); ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
a = ap.parse_args()
model, tok = load("%s/loopling-p%d" % (a.models, a.passes)); mx.eval(model.parameters())
tm = model.model.language_model; core = tm.args.loop_core
ids = [(17 * i * i + 3 * i) % 40000 + 1000 for i in range(a.ctx)]
cache = make_prompt_cache(model)
for s in range(0, a.ctx, 512):
    model(mx.array([ids[s:s + 512]]), cache=cache); mx.eval([c.state for c in cache])
cpu = lambda: (lambda r: r.ru_utime + r.ru_stime)(resource.getrusage(resource.RUSAGE_SELF))
y = mx.array([[5]])
def full_step(y):
    lg = model(y, cache=cache)[:, -1, :]
    lp = lg - mx.logsumexp(lg, keepdims=True)
    return mx.argmax(lp, axis=-1)
for _ in range(5): y = full_step(y).reshape(1, 1); mx.eval(y)
R = {"passes": a.passes, "ctx": a.ctx}
# 1. sync: build (graph construction) vs eval, CPU time
b, e, c0 = [], [], cpu(); w0 = time.perf_counter()
for _ in range(a.steps):
    t0 = time.perf_counter(); n = full_step(y); t1 = time.perf_counter(); mx.eval(n); t2 = time.perf_counter(); y = n.reshape(1, 1)
    b.append(t1 - t0); e.append(t2 - t1)
W = time.perf_counter() - w0
R["sync_ms_per_tok"] = 1e3 * W / a.steps; R["build_ms"] = 1e3 * st.median(b); R["eval_ms"] = 1e3 * st.median(e); R["sync_cpu_ms_per_tok"] = 1e3 * (cpu() - c0) / a.steps
# 2. async pipelined (what generate does)
c0 = cpu(); w0 = time.perf_counter(); n = full_step(y); mx.async_eval(n)
for _ in range(a.steps):
    nn_ = full_step(n.reshape(1, 1)); mx.async_eval(nn_); n.item(); n = nn_
W = time.perf_counter() - w0
R["async_ms_per_tok"] = 1e3 * W / a.steps; R["async_cpu_ms_per_tok"] = 1e3 * (cpu() - c0) / a.steps; y = n.reshape(1, 1); mx.eval(y)
# 3. graph size
n = full_step(y); f = tempfile.NamedTemporaryFile("w+", suffix=".dot"); mx.export_to_dot(f.name, n); f.seek(0); txt = f.read()
R["graph_dot_lines"] = len(txt.splitlines()); R["graph_nodes"] = sum(1 for l in txt.splitlines() if "[label" in l and "->" not in l); mx.eval(n); y = n.reshape(1, 1)
# 4. per group, synced (build, build+eval) medians
x = mx.zeros((1,)); mx.eval(x); fl = []
for _ in range(200):
    t0 = time.perf_counter(); o = x + 1; mx.eval(o); fl.append(time.perf_counter() - t0)
R["sync_floor_ms"] = 1e3 * st.median(fl)
def run_groups():
    conv_mask = cache[0][1].make_mask(1); caches = iter(cache)
    h = tm.embed(y); mx.eval(h); recs = []
    def timed(name, fn):
        t0 = time.perf_counter(); out = fn(); t1 = time.perf_counter(); mx.eval(out); t2 = time.perf_counter(); recs.append((name, t1 - t0, t2 - t0)); return out
    def run(layers, h, bs, bank=None):
        for i, layer in enumerate(layers):
            msg = None if bank is None else tm.message_input(bank, i)
            h, bs = layer(h, bs, cache=next(caches), conv_mask=conv_mask, messages=msg)
        return h, bs
    pre = timed("prelude(L0-5)", lambda: run(tm.layers[:core.start], h, None)); h, bs = pre
    for p in range(tm.args.loop_config["num_passes"]):
        bank = None
        if p > 0:
            bank = (bs[pre[1].shape[0] + 1:], unweighted_rms_norm(h.astype(mx.float32))); h, bs = pre
        h, bs = timed("loop pass %d (L6-17)" % p, lambda: run(tm.layers[core.start:core.stop], h, bs, bank))
    h, bs = timed("tail(L18-23)", lambda: run(tm.layers[core.stop:], h, bs))
    def head():
        x = unweighted_rms_norm(h.astype(mx.float32)); x = tm.final_input(bs, x)[..., 0, :]; return tm.norm(x).astype(h.dtype)
    hh = timed("final_input+norm", head)
    lg = timed("lm_head+softcap", lambda: (lambda l: mx.tanh(l.astype(mx.float32) / 30.0) * 30.0)(model.lm_head(hh)))
    timed("logsumexp+argmax", lambda: mx.argmax(lg[:, -1, :] - mx.logsumexp(lg[:, -1, :], keepdims=True), axis=-1))
    return recs
acc = {}
for _ in range(15):
    for name, bt, tt in run_groups(): acc.setdefault(name, []).append((bt, tt))
R["groups_ms[build,build+eval]"] = {k: [round(1e3 * st.median(x[0] for x in v), 3), round(1e3 * st.median(x[1] for x in v), 3)] for k, v in acc.items()}
print(json.dumps(R, indent=1))
