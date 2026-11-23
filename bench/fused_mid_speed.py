"""Speed of rows/mid.py (K5 MID) vs the MLX ops it replaces (the compiled stage-M segment of the default fast path, B separate
single-row calls). Median us per call over REPS, Q launches queued per eval, one layer's weights per call (cycling layers of the
same kind). Run inside the GPU lock:  python bench/fused_mid_speed.py"""
import os, statistics, time
os.environ["LOOPLING_COMPILE_DECODE"] = "0"
import mlx_lm
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import layers as L
from mlx_lm.models.inkling2.rows.mid import mid_rows

model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
layers = list(model.model.language_model.layers)
f32, bf16 = mx.float32, mx.bfloat16
REPS, Q = 60, 48
mx.random.seed(0)
rn = lambda *s: mx.random.normal(s)


def dsc(mod, x, wproj, st):  # DynamicShortConvolution given wproj (the weight_proj matmul belongs to K4)
    B, Ln, C = x.shape
    K, H = 4, mod.num_heads
    dyn = wproj.astype(f32).reshape(B, Ln, H, 1, K)
    stat = mod.static_weight.astype(f32).T.reshape(H, -1, K)
    w = stat + dyn
    window = L._conv_input(x, st, None, K)
    windows = mx.stack([window[:, K - 1 - lag: K - 1 - lag + Ln] for lag in range(K)], axis=-1).reshape(B, Ln, H, -1, K)
    out = (windows * w).sum(axis=-1).reshape(B, Ln, C)
    out = (out + x.astype(f32)) / 3.0
    return out.astype(x.dtype), window[:, 3:]


def make_ref(layer):
    a = layer.self_attn
    full = isinstance(layer, L.FullDecoderLayer)

    def f(q, r, kr, vr, cs0, cs1, tau, kw, vw):
        if full:
            key, w0 = dsc(a.k_sconv, kr, kw, cs0)
            value, w1 = dsc(a.v_sconv, vr, vw, cs1)
        else:
            key, w0 = FD._sc(a.k_sconv, kr, cs0)
            value, w1 = FD._sc(a.v_sconv, vr, cs1)
        qh = a.q_l2_clip(q.reshape(1, 1, a.num_heads, -1))
        kh = a.k_l2_clip(key.reshape(1, 1, a.num_key_value_heads, -1))
        vh = value.reshape(1, 1, a.num_key_value_heads, -1)
        vh = L.unweighted_rms_norm(vh.astype(f32), a.eps).astype(vh.dtype)
        rh = a.rel_l2_clip(r.reshape(1, 1, a.num_heads, -1))
        if full:
            qh = (qh.astype(f32) * tau).astype(qh.dtype)
            rh = (rh.astype(f32) * tau).astype(rh.dtype)
        return qh, kh, vh, rh, w0, w1
    return mx.compile(f)


def timeit(fn):
    for _ in range(5): mx.eval(fn(Q))
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); mx.eval(fn(Q)); ts.append((time.perf_counter() - t) / Q * 1e6)
    return statistics.median(ts)


for kind, ids in (("sliding", [i for i in range(24) if not isinstance(layers[i], L.FullDecoderLayer)][:8]),
                  ("full", [i for i in range(24) if isinstance(layers[i], L.FullDecoderLayer)])):
    full = kind == "full"
    refs = {i: make_ref(layers[i]) for i in ids}
    for B in (1, 8):
        inp = dict(q=rn(B, 3072).astype(bf16), r=rn(B, 384).astype(bf16), kr=rn(B, 1024).astype(bf16), vr=rn(B, 1024).astype(bf16),
                   cs0=rn(B, 3, 1024), cs1=rn(B, 3, 1024), tau=1.0 + mx.random.uniform(shape=(B,)),
                   kw=rn(B, 32).astype(bf16), vw=rn(B, 32).astype(bf16))
        mx.eval(inp)

        def ref_call(i, B=B):
            f = refs[ids[i % len(ids)]]
            outs = []
            for b in range(B):  # B separate single-row calls, as the decode path does today
                outs.append(f(inp["q"][b:b + 1].reshape(1, 1, -1), inp["r"][b:b + 1].reshape(1, 1, -1), inp["kr"][b:b + 1].reshape(1, 1, -1),
                              inp["vr"][b:b + 1].reshape(1, 1, -1), inp["cs0"][b:b + 1], inp["cs1"][b:b + 1], inp["tau"][b:b + 1].reshape(1, 1, 1),
                              inp["kw"][b:b + 1].reshape(1, 1, -1), inp["vw"][b:b + 1].reshape(1, 1, -1)))
            return outs

        def row_call(i, B=B):
            kw = dict(tau=inp["tau"], k_wproj=inp["kw"], v_wproj=inp["vw"]) if full else {}
            return mid_rows(layers[ids[i % len(ids)]], inp["q"], inp["r"], inp["kr"], inp["vr"], inp["cs0"], inp["cs1"], **kw)

        ref = timeit(lambda n: [ref_call(i) for i in range(n)])
        row = timeit(lambda n: [row_call(i) for i in range(n)])
        print("%-8s B=%d  ref (B single-row compiled calls) %8.1f us   mid_rows %8.1f us   speedup %.2fx" % (kind, B, ref, row, ref / row), flush=True)
