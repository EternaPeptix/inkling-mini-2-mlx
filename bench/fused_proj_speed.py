"""Speed of rows/proj.py kernels vs the MLX ops they replace (median us per call, 32 launches queued per eval)."""
import os, statistics, time
os.environ.setdefault("LOOPLING_COMPILE_DECODE", "0")
import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2.rows import proj as P

model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
Ls = model.model.language_model.layers
mx.random.seed(0)


def queued(fn, reps=60, q=32):
    for _ in range(5): mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval([fn() for _ in range(q)]); ts.append((time.perf_counter() - t) / q * 1e6)
    return statistics.median(ts)


def rnd(*s, dt=mx.bfloat16):
    return mx.random.normal(s).astype(dt)


def row(name, B, fused, ref, ref2=None):
    f = queued(fused); r = queued(ref)
    extra = ""
    if ref2 is not None:
        r2 = queued(ref2); extra = " | ref compiled %7.1f (%.2fx)" % (r2, r2 / f)
    print("%-26s B=%d | fused %7.1f | ref %7.1f (%.2fx)%s" % (name, B, f, r, r / f, extra), flush=True)


for B in (1, 8):
    for li, tag in ((7, "sliding L7"), (5, "full L5")):
        L = Ls[li]; at = L.self_attn; full = li == 5
        xq, xk, xv = rnd(B, 2304), rnd(B, 2304), rnd(B, 2304); mx.eval(xq, xk, xv)
        def ref4():
            outs = []
            for b in range(B):
                q, k, v = (x[b].reshape(1, 1, -1) for x in (xq, xk, xv))
                outs += [at.q_proj(q), at.r_proj(q), at.gate_proj(q), at.sink_proj(q), at.k_proj(k), at.v_proj(v)]
                if full: outs += [at.k_sconv.weight_proj(k), at.v_sconv.weight_proj(v)]
            return outs
        row("K4 QKV " + tag, B, lambda: list(P.qkv_rows(at, xq, xk, xv).values()), ref4)
    at = Ls[7].self_attn
    ao = rnd(B, 3072); gt = rnd(B, 3072) * 3; mx.eval(ao, gt)
    def ref9():
        return [at._attend_post(ao[b].reshape(1, 1, 24, 128), gt[b].reshape(1, 1, -1)) for b in range(B)]
    def ref9c():
        return [cpost(ao[b].reshape(1, 1, 24, 128), gt[b].reshape(1, 1, -1)) for b in range(B)]
    def post(a, g):
        return at.o_proj((a.reshape(1, 1, -1).astype(mx.float32) * mx.sigmoid(g.astype(mx.float32))).astype(g.dtype))
    cpost = mx.compile(post)
    row("K9 OPROJ", B, lambda: P.oproj_rows(ao, gt, at.o_proj.weight, at.o_proj.scales, at.o_proj.biases), ref9, ref9c)

    L = Ls[7]
    for label, gates, slots in (("G=3 qkv", [L.q_message_gate, L.k_message_gate, L.v_message_gate], (0, 1, 2)),
                                ("G=1 mlp", [L.mlp_message_gate], (3,))):
        G = len(gates)
        x = rnd(B, G, 4608); dd = rnd(B, G, 64); ss = [rnd(B, 2304, dt=mx.float32) for _ in range(G)]
        msg = rnd(B, 4, 2304, dt=mx.float32); mx.eval(x, dd, ss, msg)
        wd = [g.down_proj.weight for g in gates]
        wf = [g.forget_gate_proj.weight for g in gates]; wm = [g.message_gate_proj.weight for g in gates]
        row("K2 GATE-DOWN " + label, B, lambda: P.gate_down_rows(x, wd),
            lambda: [g.down_proj(x[b, i].reshape(1, 1, -1)) for b in range(B) for i, g in enumerate(gates)])
        def mixref(g, d, s, m):
            st = d.astype(mx.float32)
            fg = mx.tanh(g.forget_gate_proj(st)); mg = mx.tanh(g.message_gate_proj(st))
            return (1 - fg) * s + mg * m
        cm = [mx.compile(lambda d, s, m, g=g: mixref(g, d, s, m)) for g in gates]
        row("K3 GATE-MIX " + label, B, lambda: P.gate_mix_rows(dd, ss, msg, slots, wf, wm),
            lambda: [mixref(g, dd[b, i].reshape(1, 1, 64), ss[i][b].reshape(1, 1, -1), msg[b, slots[i]].reshape(1, 1, -1))
                     for b in range(B) for i, g in enumerate(gates)],
            lambda: [cm[i](dd[b, i].reshape(1, 1, 64), ss[i][b].reshape(1, 1, -1), msg[b, slots[i]].reshape(1, 1, -1))
                     for b in range(B) for i, g in enumerate(gates)])

# ---- dependent chains at B=1 (output feeds the next input; every output consumed) -- exposes per-launch latency
def chain(step, reps=40, n=32):
    ts = []
    for _ in range(reps):
        x = rnd(1, 2304); mx.eval(x)
        t = time.perf_counter(); y = x
        for _ in range(n): y = step(y)
        mx.eval(y); ts.append((time.perf_counter() - t) / n * 1e6)
    return statistics.median(ts)

def fit(outs):
    return mx.concatenate([o.reshape(1, -1) for o in outs], axis=1)[:, :2304].astype(mx.bfloat16)

at = Ls[7].self_attn
print("--- dependent chains, B=1 (includes one concatenate per step in both variants)")
print("K4 sliding chain: fused (flat buffer) %.1f us | fused (6 slice views) %.1f us | 6 module calls %.1f us" % (
    chain(lambda y: P.qkv_rows(at, y, y, y, flat=True)[0].reshape(1, -1)[:, :2304]),
    chain(lambda y: fit(P.qkv_rows(at, y, y, y).values())),
    chain(lambda y: fit([at.q_proj(y), at.r_proj(y), at.gate_proj(y), at.sink_proj(y), at.k_proj(y), at.v_proj(y)]))))
gt1 = rnd(1, 3072) * 3; ao1 = rnd(1, 3072)
mx.eval(gt1, ao1)
def k9f(y):
    return P.oproj_rows(mx.concatenate([y, y[:, :768]], axis=1), gt1, at.o_proj.weight, at.o_proj.scales, at.o_proj.biases)
def k9r(y):
    return at._attend_post(mx.concatenate([y, y[:, :768]], axis=1).reshape(1, 1, 24, 128), gt1.reshape(1, 1, -1)).reshape(1, -1)
print("K9 chain: fused %.1f us | ref ops %.1f us" % (chain(k9f), chain(k9r)))
L = Ls[7]; gates = [L.q_message_gate, L.k_message_gate, L.v_message_gate]
x3 = rnd(1, 3, 4608); m3 = rnd(1, 4, 2304, dt=mx.float32); mx.eval(x3, m3)
wd = [g.down_proj.weight for g in gates]
wf = [g.forget_gate_proj.weight for g in gates]; wm = [g.message_gate_proj.weight for g in gates]
def g_f(y):  # K2 -> K3; y (1,2304) bf16 is the hidden state of all three gates and perturbs the gate input
    yf = y.astype(mx.float32)
    dn = P.gate_down_rows(x3 + y[:, :1][:, None], wd)
    return P.gate_mix_rows(dn, [yf, yf, yf], m3, (0, 1, 2), wf, wm).sum(axis=1).astype(mx.bfloat16)
def g_r(y):
    yf = y.astype(mx.float32).reshape(1, 1, -1)
    xin = x3 + y[:, :1][:, None]
    outs = []
    for i, g in enumerate(gates):
        d = g.down_proj(xin[:, i].reshape(1, 1, -1)); st = d.astype(mx.float32)
        fg = mx.tanh(g.forget_gate_proj(st)); mg = mx.tanh(g.message_gate_proj(st))
        outs.append((1 - fg) * yf + mg * m3[:, i].reshape(1, 1, -1))
    return mx.stack(outs, axis=1).sum(axis=1).reshape(1, -1).astype(mx.bfloat16)
print("K2->K3 (G=3) chain: fused %.1f us | ref ops %.1f us" % (chain(g_f), chain(g_r)))
