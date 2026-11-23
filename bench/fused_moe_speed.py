"""Speed of rows/moe.py (3 launches) vs the MLX op chain of fused_ref stage E (per row, mx.compile'd like the fast path).
Median us per call over REPS reps; queued = NQ independent calls then one eval; chain = dependent (out feeds next mlp_x).
Run inside the GPU lock:  python bench/fused_moe_speed.py [--layers 0,20]"""
import argparse, json, os, statistics, time

import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import fused_ref as FR
from mlx_lm.models.inkling2.rows import moe as M

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="0,20")
ap.add_argument("--visits", default=os.path.expanduser("~/loopling-opt/runs/visits_p3"))
ap.add_argument("--model", default=os.path.expanduser("~/loopling/models/loopling-p3"))
ap.add_argument("--reps", type=int, default=60)
ap.add_argument("--nq", type=int, default=16)
a = ap.parse_args()
f32 = mx.float32
model, tok = load(a.model)
layers = list(model.model.language_model.layers)
index = json.load(open(os.path.join(a.visits, "index.json")))["visits"]


def ref_chain(layer):
    moe, ex = layer.mlp, layer.mlp.experts

    def f(x, residual, cs3):  # x (1,1,D) bf16
        logits = x.astype(f32) @ moe.gate.weight.T
        inds, wts = moe.gate(x)
        xe = mx.expand_dims(x, (-2, -3))
        gu = ex.gate_up_proj(xe, inds, sorted_indices=False)
        act = ex.act_fn(gu[..., 0::2], gu[..., 1::2])
        dn = ex.down_proj(act, inds, sorted_indices=False).squeeze(-2)
        comb = (dn.astype(f32) * wts[..., None]).sum(axis=-2).astype(x.dtype)
        hidden, w3 = FD._sc(layer.post_mlp_convolution, comb, cs3)
        return (residual + hidden.astype(f32)).astype(x.dtype), w3

    return f


def med(f, nq=a.nq, reps=a.reps):
    for _ in range(5):
        mx.eval(f(nq))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(f(nq))
        ts.append((time.perf_counter() - t) / nq * 1e6)
    return statistics.median(ts)


def fused_fn(layer, route, ngu, ndn):
    moe, ex = layer.mlp, layer.mlp.experts
    gu, dn = ex.gate_up_proj, ex.down_proj
    cw = layer.post_mlp_convolution.conv1d.weight

    def f(x, resid, cs3):
        lg = M.router_rows(x, moe.gate.weight)
        r = M.gateup_rows(lg, moe.gate.e_score_correction_bias, x, gu["weight"], gu["scales"], gu.get("global_scale"),
                          mode=gu.mode, group_size=gu.group_size, bits=gu.bits, route_scale=moe.gate.routed_scaling_factor,
                          gate_softcap=ex.act_fn.gate_softcap, up_softcap=ex.act_fn.up_softcap, route=route, nsg=ngu)
        o = M.down_rows(r[2], r[0], r[1], resid, cs3, cw, dn["weight"], dn["scales"], dn.get("global_scale"),
                        mode=dn.mode, group_size=dn.group_size, bits=dn.bits, nsg=ndn)
        return o[0], o[1]

    return f


for li in [int(i) for i in a.layers.split(",")]:
    layer = layers[li]
    D = 2304
    xs, rs, cs = [], [], []
    for v in [v for v in index if v["layer"] == li][:8]:
        d = mx.load(os.path.join(a.visits, v["file"]))
        c = [d["cs%d" % i] for i in range(4)]
        T = FR.visit_ref(layer, d["h"], d.get("bs"), d.get("msg"), c, d["attn_out"], d.get("tau"))
        xs.append(T["mlp_x"].reshape(1, D)); rs.append(T["residual"].reshape(1, D)); cs.append(c[3])
    X, R, C = mx.concatenate(xs), mx.concatenate(rs), mx.concatenate(cs)
    mx.eval(X, R, C)
    mode = layer.mlp.experts.gate_up_proj.mode
    ref = mx.compile(ref_chain(layer))
    ref_u = ref_chain(layer)
    print("== layer %d (%s)" % (li, mode), flush=True)
    for B in (1, 8):
        xb, rb, cb = X[:B], R[:B], C[:B]
        x1 = [xb[b][None, None] for b in range(B)]
        r1 = [rb[b][None, None] for b in range(B)]
        c1 = [cb[b][None] for b in range(B)]
        mx.eval(x1, r1, c1)

        def run_ref(nq, fn=ref):
            outs = []
            for _ in range(nq):
                for b in range(B):
                    outs.extend(fn(x1[b], r1[b], c1[b]))
            return outs

        res = {"ref compiled (B x chain)": med(run_ref, nq=max(1, a.nq // B)) , "ref uncompiled (B x chain)": med(lambda nq: run_ref(nq, ref_u), nq=max(1, a.nq // B))}
        for route, ngu, ndn in [("kernel", 8, 8), ("kernel", 4, 8), ("kernel", 2, 8), ("launch", 8, 8), ("launch", 4, 8), ("launch", 2, 8), ("kernel", 8, 4), ("kernel", 4, 4)]:
            f = fused_fn(layer, route, ngu, ndn)
            res["fused route=%s gu_nsg=%d dn_nsg=%d" % (route, ngu, ndn)] = med(lambda nq: [o for _ in range(nq) for o in f(xb, rb, cb)])
        # per-kernel (each alone, queued; includes ~9 us of host encode per launch when the GPU time is shorter)
        moe, ex = layer.mlp, layer.mlp.experts
        gu_, dn_ = ex.gate_up_proj, ex.down_proj
        lg = M.router_rows(xb, moe.gate.weight)
        kw = dict(mode=gu_.mode, group_size=gu_.group_size, bits=gu_.bits, route_scale=moe.gate.routed_scaling_factor,
                  gate_softcap=ex.act_fn.gate_softcap, up_softcap=ex.act_fn.up_softcap)
        g1 = lambda route: M.gateup_rows(lg, moe.gate.e_score_correction_bias, xb, gu_["weight"], gu_["scales"], gu_.get("global_scale"), route=route, **kw)
        r_ = g1("kernel")
        cw_ = layer.post_mlp_convolution.conv1d.weight
        print("  per-kernel: K11 router %.1f | K12 route-in-kernel %.1f | K12 route-launch (2 launches) %.1f | K13 down %.1f us" % (
            med(lambda nq: [M.router_rows(xb, moe.gate.weight) for _ in range(nq)]),
            med(lambda nq: [g1("kernel") for _ in range(nq)]),
            med(lambda nq: [g1("launch") for _ in range(nq)]),
            med(lambda nq: [M.down_rows(r_[2], r_[0], r_[1], rb, cb, cw_, dn_["weight"], dn_["scales"], dn_.get("global_scale"),
                                        mode=dn_.mode, group_size=dn_.group_size, bits=dn_.bits) for _ in range(nq)])), flush=True)
        # chain: out (B,D) bf16 feeds the next x
        best = min((k for k in res if k.startswith("fused")), key=res.get)
        print("B=%d" % B, flush=True)
        for k, v in res.items():
            print("  %-44s %8.1f us%s" % (k, v, "   <- best" if k == best else ""), flush=True)
        f = fused_fn(layer, "kernel", 8, 8) if best is None else None
        bf = best.split()
        route = bf[1].split("=")[1]; ngu = int(bf[2].split("=")[1]); ndn = int(bf[3].split("=")[1])
        f = fused_fn(layer, route, ngu, ndn)

        def chain_f(n):
            x = xb
            for _ in range(n):
                x = f(x, rb, cb)[0]
            return x

        def chain_r(n, fn=ref):
            x = x1
            for _ in range(n):
                x = [fn(x[b], r1[b], c1[b])[0] for b in range(B)]
            return x

        n = 8
        print("  chain (dependent) fused best: %.1f us/link   ref compiled: %.1f us/link" % (
            med(lambda nq: chain_f(n), nq=n), med(lambda nq: chain_r(n), nq=n)), flush=True)
