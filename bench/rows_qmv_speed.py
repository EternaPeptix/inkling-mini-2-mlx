import os, time, statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2.rows.qmv import qmv_rows, qmv_rows_multi

model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
lay = model.model.language_model.layers[3].self_attn
P = lambda m: (m.weight, m.scales, m.biases)
mx.random.seed(0)

def ref1(x, m):
    return mx.quantized_matmul(x.reshape(1, 1, -1), m.weight, m.scales, m.biases, transpose=True,
                               group_size=64, bits=8, mode="affine")

def queued(fn, reps=60, q=32):
    """median us per call with q launches queued per single eval."""
    for _ in range(5): mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval([fn() for _ in range(q)]); ts.append((time.perf_counter() - t) / q * 1e6)
    return statistics.median(ts)

def chain(step, K, reps=50, n=32):
    """median us per call of a dependent chain: output feeds the next input."""
    ts = []
    for _ in range(reps):
        x = mx.random.normal((1, K)).astype(mx.bfloat16); mx.eval(x)
        t = time.perf_counter(); y = x
        for _ in range(n): y = step(y)
        mx.eval(y); ts.append((time.perf_counter() - t) / n * 1e6)
    return statistics.median(ts)

def fit(y, K):
    y = y.reshape(1, -1); N = y.shape[1]
    return (mx.tile(y, (1, (K + N - 1) // N))[:, :K] if N < K else y[:, :K]).astype(mx.bfloat16)

cases = [("q_proj", lay.q_proj), ("k_proj", lay.k_proj), ("o_proj", lay.o_proj), ("sink_proj", lay.sink_proj), ("lm_head", model.lm_head)]
print("%-10s %2s | %10s %10s %7s | %10s %10s" % ("weight", "B", "rows(us)", "ref(us)", "ratio", "rows chain", "ref chain"))
for name, m in cases:
    N = m.weight.shape[0]; K = m.weight.shape[1] * 4
    for B in (1, 8):
        X = mx.random.normal((B, K)).astype(mx.bfloat16); mx.eval(X)
        rq = queued(lambda: qmv_rows(X, *P(m)))
        fq = queued(lambda: [ref1(X[b], m) for b in range(B)])
        if B == 1:
            rc = chain(lambda y: fit(qmv_rows(y, *P(m)), K), K)
            fc = chain(lambda y: fit(ref1(y, m), K), K)
        else:
            rc = fc = float("nan")
        print("%-10s %2d | %10.1f %10.1f %7.2f | %10.1f %10.1f" % (name, B, rq, fq, fq / rq, rc, fc), flush=True)

ms = [lay.q_proj, lay.r_proj, lay.gate_proj, lay.sink_proj]
K = 2304
for B in (1, 8):
    X = mx.random.normal((B, K)).astype(mx.bfloat16); mx.eval(X)
    mq = queued(lambda: qmv_rows_multi(X, [P(m) for m in ms]))
    sr = queued(lambda: [qmv_rows(X, *P(m)) for m in ms])
    sep = queued(lambda: [ref1(X[b], m) for m in ms for b in range(B)])
    print("multi q+r+gate+sink B=%d: 1 launch %.1f us | 4 qmv_rows launches %.1f us | %d ref calls %.1f us" % (B, mq, sr, 4 * B, sep), flush=True)
    if B == 1:
        def step(y):
            # every output must be consumed, otherwise lazy eval skips the unused ones
            o = qmv_rows_multi(y, [P(m) for m in ms]); return fit(mx.concatenate([a.reshape(1, -1) for a in o], axis=1), K)
        def step_ref(y):
            o = [ref1(y, m) for m in ms]; return fit(mx.concatenate([a.reshape(1, -1) for a in o], axis=1), K)
        print("   dependent chain: multi %.1f us | 4 ref calls %.1f us" % (chain(step, K), chain(step_ref, K)), flush=True)
