"""Loopling bench: greedy decode tok/s + token digests per (prompt, passes); prefill tok/s. Writes JSON."""
import argparse, hashlib, json, os, statistics, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
PROMPTS = {
 "code": "Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
 "prose": "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
 "math": "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
}
ap = argparse.ArgumentParser(); ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
ap.add_argument("--passes", default="1,3,5"); ap.add_argument("--tokens", type=int, default=256); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--prefill", default="2048,16384"); ap.add_argument("--logits", type=int, default=32); ap.add_argument("--out", required=True)
a = ap.parse_args(); out = {"flags": {k: v for k, v in os.environ.items() if k.startswith("LOOPLING_")}, "decode": {}, "prefill": {}}
greedy = make_sampler(0.0)
for p in map(int, a.passes.split(",")):
    model, tok = load("%s/loopling-p%d" % (a.models, p)); mx.eval(model.parameters())
    for name, text in PROMPTS.items():
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>"
        ids = tok.encode(prompt, add_special_tokens=False); runs = []
        for r in range(a.reps + 1):
            toks = []; t0 = None; gen_tps = None
            for resp in stream_generate(model, tok, ids, max_tokens=a.tokens, sampler=greedy):
                toks.append(resp.token); gen_tps = resp.generation_tps; pp = resp.prompt_tps
            if r: runs.append(gen_tps)
        out["decode"]["p%d/%s" % (p, name)] = {"tok_s_median": statistics.median(runs), "tok_s_runs": [round(x, 2) for x in runs],
            "n": len(toks), "digest": hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16], "tokens": toks}
        if getattr(model, "mtp_stats", None): out["decode"]["p%d/%s" % (p, name)]["mtp"] = dict(model.mtp_stats)
        print("p%d %-5s decode %.2f tok/s %s digest %s %s" % (p, name, statistics.median(runs), [round(x, 1) for x in runs], out["decode"]["p%d/%s" % (p, name)]["digest"], model.mtp_stats if getattr(model, "mtp_stats", None) else ""), flush=True)
    if p == 3:
        for n in map(int, filter(None, a.prefill.split(","))):
            ids = [(17 * i * i + 3 * i) % 40000 + 1000 for i in range(n)]; ts = []
            for r in range(3):
                t0 = time.perf_counter(); c = None
                from mlx_lm.models.cache import make_prompt_cache
                c = make_prompt_cache(model); step = 2048
                for s in range(0, n, step):
                    model(mx.array([ids[s:s + step]]), cache=c); mx.eval([x.state for x in c])
                ts.append(n / (time.perf_counter() - t0))
            out["prefill"]["p3/%d" % n] = {"tok_s_median": statistics.median(ts), "runs": [round(x, 1) for x in ts]}
            print("p3 prefill %d tokens: %.0f tok/s %s" % (n, statistics.median(ts), [round(x) for x in ts]), flush=True)
    del model; mx.clear_cache()
json.dump(out, open(a.out, "w"), indent=1); print("WROTE", a.out)
