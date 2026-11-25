"""C=8 stage breakdown with 8 distinct prompts (LOOPLING_BD_FREEZE + gather vs moe microbenchmarks).

Usage: LANE=opt RHOST=512S2 zsh scripts/r --gpu 'python bench/c8_stage_prof.py'
"""
import json
import os
import statistics
import subprocess
import sys
import time

import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import fused_ref as FR
from mlx_lm.models.inkling2.rows import gather as RGa
from mlx_lm.models.inkling2.rows import moe as RMo

PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
    "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
    "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
    "Explain how a B-tree insertion works, step by step, with an example of a node split.",
    "Write a bash script that finds the ten largest files under a directory and prints their sizes in MB.",
    "Summarize the causes of the French Revolution in five bullet points, then give one counterargument.",
    "Implement binary search in Rust and explain the off-by-one pitfalls.",
    "Describe the water cycle to a ten-year-old, then list three common misconceptions.",
]


def run_bd(fused, freeze_stage="", route="kernel", compile_mode="2"):
    env = dict(
        os.environ,
        LOOPLING_BD_FUSED=fused,
        LOOPLING_COMPILE_DECODE=compile_mode,
        LOOPLING_BD_FREEZE=freeze_stage,
        LOOPLING_BD_ROUTE=route,
        ROWKV_DOUBLE="1",
    )
    code = """
import mlx_lm, os, statistics, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2.batchdecode import BatchDecoder
PROMPTS = __PROMPTS__
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
mx.eval(model.parameters())
ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False)
                  + "<|message_model|><|content_text|>", add_special_tokens=False) for t in PROMPTS]
def pre(i):
    c = make_prompt_cache(model); model(mx.array(ids[i][:-1])[None], cache=c); mx.eval([x.state for x in c]); return c
dec = BatchDecoder(model, [pre(i) for i in range(8)])
y = mx.array([ids[i][-1] for i in range(8)])
y = dec.step(y); mx.async_eval(y)
ts = []
for n in range(44):
    ny = None
    if n + 1 < 44: ny = dec.step(y); mx.async_eval(ny)
    y.tolist()
    ts.append(time.perf_counter())
    y = ny
dt = [1e3*(b-a) for a,b in zip(ts[8:], ts[9:])]
print("RESULT=%.3f" % statistics.median(dt))
""".replace("__PROMPTS__", repr(PROMPTS))
    out = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    for line in out.splitlines():
        if line.startswith("RESULT="):
            return float(line.split("=")[1])
    raise RuntimeError(out)


def moe_micro():
    print("\n=== MoE B=8 real-distinct-visit microbenchmark: gather.py chain vs moe.py (x48 visits/step) ===")
    model, _ = load(os.path.expanduser("~/loopling/models/loopling-p3"))
    mx.eval(model.parameters())
    layers = list(model.model.language_model.layers)
    vdir = os.path.expanduser("~/loopling-opt/runs/visits_p3")
    index = json.load(open(os.path.join(vdir, "index.json")))["visits"]

    def get_batch(li):
        rows = [v for v in index if v["layer"] == li][:8]
        xs, rs, cs = [], [], []
        for v in rows:
            d = mx.load(os.path.join(vdir, v["file"]))
            c = [d["cs%d" % i] for i in range(4)]
            T = FR.visit_ref(layers[li], d["h"], d.get("bs"), d.get("msg"), c, d["attn_out"], d.get("tau"))
            xs.append(T["mlp_x"].reshape(1, 2304))
            rs.append(T["residual"].reshape(1, 2304))
            cs.append(c[3])
        X, R, C = mx.concatenate(xs), mx.concatenate(rs), mx.concatenate(cs)
        mx.eval(X, R, C)
        return X, R, C

    def timed(fn, reps=30, nq=16):
        for _ in range(4):
            mx.eval([fn() for _ in range(nq)])
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            mx.eval([fn() for _ in range(nq)])
            ts.append((time.perf_counter() - t0) / nq * 1e6)
        return statistics.median(ts)

    for li in (0, 7, 20):
        layer = layers[li]
        moe, ex = layer.mlp, layer.mlp.experts
        gu, dn = ex.gate_up_proj, ex.down_proj
        cw = layer.post_mlp_convolution.conv1d.weight
        X, R, C = get_batch(li)
        lg = RMo.router_rows(X, moe.gate.weight)
        inds, wts = moe.gate(X[:, None, :])
        mx.eval(lg, inds, wts)
        u = len(set(inds.reshape(-1).tolist()))
        idx_u = inds.reshape(8, 8).astype(mx.uint32)
        mx.eval(idx_u)

        def gather_stage():
            g = RGa.gather_rows(X, gu["weight"], gu["scales"], None, idx_u,
                                mode=gu.mode, group_size=gu.group_size, bits=gu.bits, global_scale=gu.get("global_scale"))
            h = ex.act_fn(g[..., 0::2], g[..., 1::2])
            d = RGa.gather_rows(h, dn["weight"], dn["scales"], None, idx_u,
                                mode=dn.mode, group_size=dn.group_size, bits=dn.bits, global_scale=dn.get("global_scale"))
            return d

        def moe_kernel():
            return RMo.moe_rows(layer, X, R, C, route="kernel")["out"]

        def moe_launch():
            return RMo.moe_rows(layer, X, R, C, route="launch")["out"]

        tg = timed(gather_stage)
        tk = timed(moe_kernel)
        tl = timed(moe_launch)
        print(f"Layer {li:2d} ({gu.mode:5s}, uniq_experts={u:2d}/64): gather(gu+swiglu+dn)={tg:6.1f} us | moe(kernel)={tk:6.1f} us | moe(launch)={tl:6.1f} us", flush=True)


if __name__ == "__main__":
    print("=== Distinct-prompt C=8 LOOPLING_BD_FREEZE stage costs (pre,k1,post,kv) ===")
    b_cmp = run_bd("pre,k1,post,kv", "", "launch", "1")
    b_unc = run_bd("pre,k1,post,kv", "", "launch", "2")
    print(f"  base compiled(1): {b_cmp:.2f} ms/step | uncompiled(2): {b_unc:.2f} ms/step", flush=True)
    for st in ("k1", "qkv", "mid", "rel", "sdpa", "oproj", "k10", "gates", "moe"):
        v = run_bd("pre,k1,post,kv", st, "launch", "2")
        print(f"  FREEZE={st:6s}: {v:6.2f} ms/step  saved={b_unc - v:+6.2f} ms", flush=True)
    moe_micro()
