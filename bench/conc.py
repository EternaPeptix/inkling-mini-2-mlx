"""Concurrency baseline: N independent processes, each B=1 greedy decode (current default fast path).
Every stream is computed exactly like a single stream, so this is exact by construction. Reports aggregate tok/s
= total tokens / (last end - first start) over the overlapped window."""
import argparse, json, os, subprocess, sys, time
ap = argparse.ArgumentParser(); ap.add_argument("--passes", type=int, default=3); ap.add_argument("--n", type=int, default=1)
ap.add_argument("--tokens", type=int, default=256); ap.add_argument("--worker", type=int, default=-1); ap.add_argument("--dir", default="/tmp/conc")
ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
a = ap.parse_args()
PROMPTS = ["Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
 "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
 "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
 "Explain how a B-tree insertion works, step by step, with an example of a node split.",
 "Write a bash script that finds the ten largest files under a directory and prints their sizes in MB.",
 "Summarize the causes of the French Revolution in five bullet points, then give one counterargument.",
 "Implement binary search in Rust and explain the off-by-one pitfalls.",
 "Describe the water cycle to a ten-year-old, then list three common misconceptions."]
if a.worker >= 0:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    model, tok = load("%s/loopling-p%d" % (a.models, a.passes)); mx.eval(model.parameters())
    text = PROMPTS[a.worker % len(PROMPTS)]
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>", add_special_tokens=False)
    greedy = lambda lp: mx.argmax(lp, axis=-1)
    for _ in zip(range(8), generate_step(mx.array(ids), model, max_tokens=8, sampler=greedy)): pass  # warm/compile
    open(f"{a.dir}/ready{a.worker}", "w").close()
    while len([f for f in os.listdir(a.dir) if f.startswith("ready")]) < a.n: time.sleep(0.005)
    gen = generate_step(mx.array(ids), model, max_tokens=a.tokens, sampler=greedy)
    toks = []; first = None
    for t, _ in gen:
        if first is None: first = time.time()  # after prefill
        toks.append(int(t))
    end = time.time()
    json.dump({"w": a.worker, "start": first, "end": end, "n": len(toks) - 1, "tokens": toks}, open(f"{a.dir}/res{a.worker}.json", "w"))
    sys.exit(0)
import shutil; shutil.rmtree(a.dir, ignore_errors=True); os.makedirs(a.dir)
ps = [subprocess.Popen([sys.executable, __file__, "--passes", str(a.passes), "--n", str(a.n), "--tokens", str(a.tokens), "--worker", str(i), "--dir", a.dir]) for i in range(a.n)]
for p in ps: p.wait()
rs = [json.load(open(f"{a.dir}/res{i}.json")) for i in range(a.n)]
t0 = min(r["start"] for r in rs); t1 = max(r["end"] for r in rs); tot = sum(r["n"] for r in rs)
per = [r["n"] / (r["end"] - r["start"]) for r in rs]
print(json.dumps({"passes": a.passes, "C": a.n, "aggregate_tok_s": round(tot / (t1 - t0), 1), "per_stream": [round(x, 1) for x in per]}))
