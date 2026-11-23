"""Speed of rows/gather.py vs the MLX reference (gather_qmm). Run inside the GPU lock. Median us per launch over >= 50 reps.
queued = REPS reps of (NQ independent launches, then one eval) / NQ; chain = dependent chain (output feeds next x)."""
import os, sys, time, statistics
os.environ["LOOPLING_COMPILE_DECODE"] = "0"
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.inkling2 import layers as L
from mlx_lm.models.inkling2.rows.gather import gather_rows

P = os.path.expanduser("~/loopling/models/loopling-p3")
LAYERS = [0, 20]
REPS, NQ = 60, 16
model, tok = load(P)
TM = model.model.language_model
experts = {i: TM.layers[i].mlp.experts for i in LAYERS}
cap = {i: [] for i in LAYERS}
orig = L.Experts.__call__
m2l = {id(experts[i]): i for i in LAYERS}


def hook(self, x, indices):
    i = m2l.get(id(self))
    if i is not None and x.shape[:2] == (1, 1) and len(cap[i]) < 40:
        mx.eval(x, indices)
        cap[i].append((x.reshape(-1), indices.reshape(-1).astype(mx.uint32)))
    return orig(self, x, indices)


L.Experts.__call__ = hook
ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Explain how a bloom filter works, briefly."}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>", add_special_tokens=False)
for _ in stream_generate(model, tok, ids, max_tokens=24):
    pass
L.Experts.__call__ = orig


def med_queued(f, nq=NQ, reps=REPS):
    for _ in range(5):
        mx.eval([f() for _ in range(nq)])
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval([f() for _ in range(nq)])
        ts.append((time.perf_counter() - t) / nq * 1e6)
    return statistics.median(ts)


def med_chain(step, x0, n=16, reps=REPS):
    def run():
        x = x0
        for _ in range(n):
            x = step(x)
        return x
    for _ in range(5):
        mx.eval(run())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(run())
        ts.append((time.perf_counter() - t) / n * 1e6)
    return statistics.median(ts)


def rows(proj, x, idx):
    return gather_rows(x, proj["weight"], proj["scales"], proj.get("biases"), idx, mode=proj.mode,
                       group_size=proj.group_size, bits=proj.bits, global_scale=proj.global_scale)


def ref1(proj, xrow, irow):
    E = irow.shape[0]
    xx = xrow.reshape(1, 1, 1, 1, -1) if xrow.ndim == 1 else xrow.reshape(1, 1, E, 1, -1)
    return proj(xx, irow.reshape(1, 1, E), sorted_indices=False)


print("%-5s %-9s %-4s %-34s %9s %9s %7s" % ("layer", "proj", "B", "case", "ref_us", "rows_us", "ratio"), flush=True)
for li in LAYERS:
    ex = experts[li]
    mode = ex.gate_up_proj.mode
    for pname in ("gate_up_proj", "down_proj"):
        proj = getattr(ex, pname)
        K = proj.input_dims
        for B in (1, 8):
            cs = [cap[li][b * 2] for b in range(B)]
            idx = mx.stack([c[1] for c in cs])
            xg = mx.stack([c[0] for c in cs])
            if pname == "gate_up_proj":
                x = xg
            else:
                x = mx.stack([ex.act_fn(g[..., 0::2], g[..., 1::2]) for g in [ref1(ex.gate_up_proj, xg[b], idx[b]).reshape(8, -1) for b in range(B)]])
            mx.eval(x, idx)
            nuniq = len(set(idx.reshape(-1).tolist()))
            tag = "L%d(%s)" % (li, mode)
            # --- rows kernel ---
            t_rows = med_queued(lambda: rows(proj, x, idx))
            # --- 1 / 8 separate reference calls ---
            t_ref = med_queued(lambda: mx.stack([ref1(proj, x[b], idx[b]) for b in range(B)]) if B > 1 else ref1(proj, x[0], idx[0]), nq=NQ // B if B > 1 else NQ)
            case = "queued, %d uniq experts" % nuniq if B > 1 else "queued"
            print("%-5s %-9s %-4d %-34s %9.1f %9.1f %6.2fx" % (tag, pname[:-5], B, case + " vs %d ref calls" % B, t_ref, t_rows, t_ref / t_rows), flush=True)
            if B > 1:
                # --- one reference call on all 8*B pairs, sorted path (what the batched decoder does) ---
                if pname == "gate_up_proj":
                    def sorted_ref():
                        xs, ids_, inv = L._gather_sort(mx.expand_dims(x, (-2, -3)), idx)
                        o = proj(xs, ids_, sorted_indices=True)
                        return L._scatter_unsort(o, inv, idx.shape)
                    lab = "sorted ref (incl sort+unsort)"
                else:
                    xs_in = x.reshape(B * 8, 1, 1, -1)
                    order = mx.argsort(idx.reshape(-1))
                    ids_s = idx.reshape(-1)[order]
                    xs_s = xs_in[order]
                    mx.eval(ids_s, xs_s)

                    def sorted_ref():
                        return proj(xs_s, ids_s, sorted_indices=True)
                    lab = "sorted ref (gather_qmm only)"
                t_sr = med_queued(sorted_ref)
                print("%-5s %-9s %-4d %-34s %9.1f %9.1f %6.2fx" % (tag, pname[:-5], B, "queued vs " + lab, t_sr, t_rows, t_sr / t_rows), flush=True)
            # --- dependent chain (gate_up only: output (B,E,2304) == K feeds next x via slice) ---
            if pname == "gate_up_proj":
                c_rows = med_chain(lambda xx: rows(proj, xx, idx)[:, 0, :], x)
                if B == 1:
                    c_ref = med_chain(lambda xx: ref1(proj, xx[0], idx[0]).reshape(8, -1)[0:1, :], x)
                else:
                    c_ref = med_chain(lambda xx: mx.stack([ref1(proj, xx[b], idx[b]).reshape(8, -1)[0] for b in range(B)]), x)
                print("%-5s %-9s %-4d %-34s %9.1f %9.1f %6.2fx" % (tag, pname[:-5], B, "dependent chain vs %d ref calls" % B, c_ref, c_rows, c_ref / c_rows), flush=True)
