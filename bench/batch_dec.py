"""Batched decode (inkling2/batchdecode.py): exactness vs single-stream greedy + aggregate tok/s for C streams.

For every stream i the reference is generate_step (default fast decode, B=1) on prompt i. The batched run prefills
every stream exactly like generate_step (all but the last prompt token, 2048-token chunks), then advances all C
streams one token per forward. GATE: every stream's tokens identical to its reference."""
import argparse, json, os, statistics, sys, time

import mlx_lm  # first: applies LOOPLING_CMDBUF before mlx.core initializes the device
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, os.path.dirname(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
ap.add_argument("--passes", default="3")
ap.add_argument("--C", default="1,2,4,8")
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--warm", type=int, default=4)
ap.add_argument("--no-check", action="store_true")
ap.add_argument("--out", default="")
ap.add_argument("--ctx", type=int, default=0, help="prepend N synthetic filler tokens to every prompt (long-context runs)")
a = ap.parse_args()

PROMPTS = ["Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
           "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
           "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
           "Explain how a B-tree insertion works, step by step, with an example of a node split.",
           "Write a bash script that finds the ten largest files under a directory and prints their sizes in MB.",
           "Summarize the causes of the French Revolution in five bullet points, then give one counterargument.",
           "Implement binary search in Rust and explain the off-by-one pitfalls.",
           "Describe the water cycle to a ten-year-old, then list three common misconceptions."]
from mlx_lm.models.inkling2.batchdecode import BatchDecoder

greedy = lambda lp: mx.argmax(lp, axis=-1)
res = {"flags": {k: v for k, v in os.environ.items() if k.startswith("LOOPLING_")}, "runs": {}}


def prefill(model, ids):
    cache = make_prompt_cache(model)
    p = mx.array(ids)
    while p.size > 1:
        n = min(4096, p.size - 1)
        model(p[:n][None], cache=cache)
        mx.eval([c.state for c in cache])
        p = p[n:]
    return cache


for P in map(int, a.passes.split(",")):
    model, tok = load("%s/loopling-p%d" % (a.models, P))
    mx.eval(model.parameters())
    ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False)
                      + "<|message_model|><|content_text|>", add_special_tokens=False) for t in PROMPTS]
    if a.ctx:
        ids = [[(17 * j * j + 3 * j + 7 * i) % 40000 + 1000 for j in range(a.ctx)] + x for i, x in enumerate(ids)]
    Cs = list(map(int, a.C.split(",")))
    ref = {}
    if not a.no_check:
        for i in range(max(Cs)):
            toks = []
            t0 = None
            for t, _ in generate_step(mx.array(ids[i]), model, max_tokens=a.tokens, sampler=greedy):
                toks.append(int(t))
                if len(toks) == a.warm:
                    t0 = time.perf_counter()
            ref[i] = toks
            if i == 0:
                single = (len(toks) - a.warm) / (time.perf_counter() - t0)
                print("p%d single-stream reference %.1f tok/s" % (P, single), flush=True)
    for C in Cs:
        caches = [prefill(model, ids[i]) for i in range(C)]
        dec = BatchDecoder(model, caches)
        y = mx.array([ids[i][-1] for i in range(C)])
        out = []
        y = dec.step(y)
        mx.async_eval(y)
        t0 = None
        sync = os.environ.get("BD_SYNC_STEP", "") not in ("", "0")  # diagnostic: no CPU/GPU step overlap
        for n in range(a.tokens):
            ny = None
            if sync:
                mx.eval(y)
            if n + 1 < a.tokens:
                ny = dec.step(y)
                mx.async_eval(ny)
            out.append(y.tolist())
            if n + 1 == a.warm:
                t0 = time.perf_counter()
            y = ny
        dt = time.perf_counter() - t0
        agg = C * (a.tokens - a.warm) / dt
        streams = [[o[i] for o in out] for i in range(C)]
        ok = None
        if ref:
            bad = [i for i in range(C) if streams[i] != ref[i]]
            ok = not bad
            first = {i: next(j for j, (x, z) in enumerate(zip(streams[i], ref[i])) if x != z) for i in bad}
        print("p%d C=%d aggregate %.1f tok/s (%.1f per stream, %.2f ms/step) exact=%s %s" % (
            P, C, agg, agg / C, 1e3 * dt / (a.tokens - a.warm), ok, "" if ok in (None, True) else "first diff %s" % first), flush=True)
        res["runs"]["p%d/C%d" % (P, C)] = {"agg_tok_s": round(agg, 2), "ms_step": round(1e3 * dt / (a.tokens - a.warm), 3), "exact": ok}
        del dec, caches
        mx.clear_cache()
    del model
    mx.clear_cache()
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
