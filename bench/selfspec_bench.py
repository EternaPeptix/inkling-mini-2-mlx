"""Lane B: LOOPLING_SELFSPEC sweep. Per target pass count: baseline vs spec configs, honest tok/s, acceptance, exact check.
tok/s here = (tokens after the first round) / (time after the first round), measured at stream_generate yield times,
because stream_generate's own generation_tps counts a whole first round against a clock started at its first token.
usage: selfspec_bench.py --targets 3,5 --drafts 1,2 --ks 2,3,4,6 --tokens 64 --reps 1 --out f.json [--window N]
"""
import argparse, json, os, statistics, time
import mlx.core as mx
from mlx_lm import load, selfspec
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

PROMPTS = {
 "code": "Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
 "prose": "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
 "math": "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
}
ap = argparse.ArgumentParser(); ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
ap.add_argument("--targets", default="3"); ap.add_argument("--drafts", default="1"); ap.add_argument("--ks", default="3")
ap.add_argument("--tokens", type=int, default=64); ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--prompts", default="code,prose,math"); ap.add_argument("--window", type=int, default=0)
ap.add_argument("--out", required=True); a = ap.parse_args()
greedy = make_sampler(0.0); out = {}


def run(model, tok, ids, spec):
    if spec: os.environ["LOOPLING_SELFSPEC"] = spec
    else: os.environ.pop("LOOPLING_SELFSPEC", None)
    toks, lps, ts = [], [], []
    for resp in stream_generate(model, tok, ids, max_tokens=a.tokens, sampler=greedy):
        toks.append(resp.token); ts.append(time.perf_counter()); lps.append(resp.logprobs)
    n1 = selfspec.STATS["rounds"][0][0] if spec and selfspec.STATS["rounds"] else 1
    n1 = min(n1, len(toks) - 1)
    tps = (len(toks) - n1) / (ts[-1] - ts[n1 - 1]) if len(toks) > 1 else 0.0
    return toks, lps, tps, list(selfspec.STATS["rounds"]) if spec else []


for T in map(int, a.targets.split(",")):
    model, tok = load("%s/loopling-p%d" % (a.models, T)); mx.eval(model.parameters())
    if a.window:
        model.args.text.sliding_window_size = a.window
        for l in model.model.language_model.layers:
            if hasattr(l.self_attn, "sliding_window"): l.self_attn.sliding_window = a.window
    for name in a.prompts.split(","):
        text = PROMPTS[name]
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>"
        ids = tok.encode(prompt, add_special_tokens=False)
        run(model, tok, ids, None)  # warmup
        base = [run(model, tok, ids, None) for _ in range(a.reps)]
        btoks, blps, _, _ = base[0]; btps = statistics.median(b[2] for b in base)
        out["p%d/%s/base" % (T, name)] = {"tok_s": btps, "n": len(btoks)}
        print("p%d %s base %.2f tok/s" % (T, name, btps), flush=True)
        for d in map(int, a.drafts.split(",")):
            for k in map(int, a.ks.split(",")):
                spec = "%d:%d" % (d, k)
                run(model, tok, ids, spec)  # warmup
                rs = [run(model, tok, ids, spec) for _ in range(a.reps)]
                toks, lps, _, rounds = rs[0]; tps = statistics.median(r[2] for r in rs)
                same = toks == btoks
                first = next((i for i, (x, y) in enumerate(zip(btoks, toks)) if x != y), None)
                rec = {"tok_s": tps, "speedup": tps / btps, "exact": same, "first_diff": first,
                       "accept_per_round": sum(r[1] for r in rounds) / max(1, len(rounds)),
                       "tokens_per_round": sum(r[0] for r in rounds) / max(1, len(rounds)), "rounds": len(rounds),
                       "tok_s_runs": [round(r[2], 2) for r in rs]}
                if first is not None:
                    def gap(lp):
                        v = mx.sort(lp)[-2:]; return (v[1] - v[0]).item()
                    rec["base_top2_gap_at_diff"] = gap(blps[first]); rec["spec_top2_gap_at_diff"] = gap(lps[first])
                    rec["base_tok"], rec["spec_tok"] = btoks[first], toks[first]
                out["p%d/%s/%s" % (T, name, spec)] = rec
                print("p%d %-5s spec %s: %.2f tok/s (x%.2f) accept/round %.2f tok/round %.2f exact=%s%s" % (
                    T, name, spec, tps, rec["speedup"], rec["accept_per_round"], rec["tokens_per_round"], same,
                    "" if same else " DIFF@%d gaps base %.4f spec %.4f" % (first, rec["base_top2_gap_at_diff"], rec["spec_top2_gap_at_diff"])), flush=True)
    del model; mx.clear_cache()
json.dump(out, open(a.out, "w"), indent=1); print("WROTE", a.out)
