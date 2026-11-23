"""Speed of rows/gemv.py vs the reference single call. Median us over >= 50 reps (many launches queued, one eval),
plus a dependent chain. Run inside the GPU lock."""
import os, statistics, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2.rows.gemv import gemv_rows, gemv_rows_multi, gemv_rows_group, rel_logits_rows

model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
L = model.model.language_model.layers
REPS, Q = 60, 48   # timed reps; launches queued per rep
mx.random.seed(0)
rnd = lambda *s: mx.random.normal(s)

def timeit(fn, n=Q):
    """median us per call: queue n launches (cycling over the layers' weights), single eval."""
    for _ in range(5): mx.eval(fn(n))
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); mx.eval(fn(n)); ts.append((time.perf_counter() - t) / n * 1e6)
    return statistics.median(ts)

def chain(step, n=Q):
    """dependent chain: each call's output feeds the next input; median us per link."""
    return None

loop = [l for l in L if hasattr(l, "q_message_gate")]
rows = []
def case(name, mk_ref, mk_rows, B_list=(1, 8), nl=len(loop)):
    for B in B_list:
        ref = timeit(lambda n: [mk_ref(i % nl, B) for i in range(n)]) if True else 0
        row = timeit(lambda n: [mk_rows(i % nl, B) for i in range(n)])
        rows.append((name, B, ref, row))
        print("%-44s B=%-2d ref(B x 1-row calls)=%8.1f us  rows=%8.1f us  speedup %.2fx" % (name, B, ref, row, ref / row), flush=True)

def lin_refs(w, x):  # B separate single-row reference calls
    return [(x[b].reshape(1, 1, -1) @ w.T) for b in range(x.shape[0])]

# ---- inputs per loop layer
xd = {B: [(rnd(B, 4608)).astype(mx.bfloat16) for _ in loop] for B in (1, 8)}
xg = {B: [rnd(B, 64) for _ in loop] for B in (1, 8)}
xr = {B: rnd(B, 2304).astype(mx.float32) for B in (1, 8)}
rr = {B: rnd(B, 24, 16).astype(mx.bfloat16) for B in (1, 8)}
mx.eval(xd, xg, xr, rr)
G0 = lambda i: loop[i].q_message_gate
case("gate.down 64x4608 bf16", lambda i, B: lin_refs(G0(i).down_proj.weight, xd[B][i]), lambda i, B: gemv_rows(xd[B][i], G0(i).down_proj.weight))
case("gate.forget 2304x64 f32", lambda i, B: lin_refs(G0(i).forget_gate_proj.weight, xg[B][i]), lambda i, B: gemv_rows(xg[B][i], G0(i).forget_gate_proj.weight))
case("gate.forget+message (1 launch vs 2 refs)", lambda i, B: lin_refs(G0(i).forget_gate_proj.weight, xg[B][i]) + lin_refs(G0(i).message_gate_proj.weight, xg[B][i]),
     lambda i, B: gemv_rows_multi(xg[B][i], [G0(i).forget_gate_proj.weight, G0(i).message_gate_proj.weight]))
GS = ["q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"]
case("4 gates down (1 launch vs 4 refs)", lambda i, B: sum([lin_refs(getattr(loop[i], g).down_proj.weight, xd[B][i]) for g in GS], []),
     lambda i, B: gemv_rows_group([xd[B][i]] * 4, [getattr(loop[i], g).down_proj.weight for g in GS]))
case("4 gates forget+message (8 problems)", lambda i, B: sum([lin_refs(getattr(loop[i], g).forget_gate_proj.weight, xg[B][i]) + lin_refs(getattr(loop[i], g).message_gate_proj.weight, xg[B][i]) for g in GS], []),
     lambda i, B: gemv_rows_group([xg[B][i]] * 8, [getattr(loop[i], g).forget_gate_proj.weight for g in GS] + [getattr(loop[i], g).message_gate_proj.weight for g in GS]))
nl = len(L)
case("router 128x2304 f32", lambda i, B: lin_refs(L[i].mlp.gate.weight, xr[B]), lambda i, B: gemv_rows(xr[B], L[i].mlp.gate.weight), nl=nl)
case("rel_logits 24x16@16x1024 bf16", lambda i, B: [rr[B][b][None, None] @ L[i].self_attn.rel_logits_proj.proj for b in range(B)],
     lambda i, B: rel_logits_rows(rr[B], L[i].self_attn.rel_logits_proj.proj), nl=nl)

# ---- dependent chain: B=1 and B=8, router -> feed back (f32), 24 layers x Q rounds
print("dependent chains (median us per link, %d reps):" % REPS)
def dep(name, ref_step, rows_step, B):
    def run(f):
        ts = []
        for r in range(REPS + 5):
            x = xr[B]; t = time.perf_counter()
            for _ in range(8):
                for i in range(nl): x = f(i, x)
            mx.eval(x); ts.append((time.perf_counter() - t) / (8 * nl) * 1e6)
        return statistics.median(ts[5:])
    a, b = run(ref_step), run(rows_step)
    print("%-44s B=%-2d ref=%8.1f us  rows=%8.1f us  speedup %.2fx" % (name, B, a, b, a / b), flush=True)
def mix(y, x):  # (B,128) -> (B,2304) cheap dependent re-expansion so the chain keeps K=2304 shape
    return mx.tile(y, (1, 18)) * 0.01 + x * 0.5
for B in (1, 8):
    dep("router chain", lambda i, x: mix(mx.concatenate(lin_refs(L[i].mlp.gate.weight, x), 0).reshape(B, -1), x),
        lambda i, x: mix(gemv_rows(x, L[i].mlp.gate.weight), x), B)
