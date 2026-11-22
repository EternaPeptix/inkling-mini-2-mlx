"""Greedy self-consistency of an MTP run: teacher-force the MTP tokens through the 1-token decode path (the baseline's numerics)
and list every position where that path's argmax differs from the MTP token, with the top-2 logit gap there.
usage: mtp_consistency.py <bench json> [passes]"""
import json, os, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

PROMPTS = {
 "code": "Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
 "prose": "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
 "math": "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
}
passes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
res = json.load(open(os.path.expanduser(sys.argv[1])))["decode"]
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p%d" % passes))
for name, text in PROMPTS.items():
    key = "p%d/%s" % (passes, name)
    if key not in res:
        continue
    toks = res[key]["tokens"]
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>", add_special_tokens=False)
    c = make_prompt_cache(model)
    model(mx.array(ids[:-1])[None], cache=c)
    bad = []
    for i, t in enumerate([ids[-1]] + toks[:-1]):
        lg = model(mx.array([[t]]), cache=c)[0, -1]
        top = mx.argsort(lg)[-2:]
        a1, a2 = int(top[1]), int(top[0])
        if a1 != toks[i]:
            bad.append((i, toks[i], a1, a2, float(lg[a1] - lg[toks[i]]), float(lg[a1] - lg[a2])))
    print("%s: %d tokens, %d positions where the decode-path argmax != MTP token" % (name, len(toks), len(bad)), flush=True)
    for i, mt, a1, a2, d_tok, gap in bad:
        print("   pos %d: mtp tok %d, decode argmax %d (logit lead over mtp tok %.3f; top-2 gap %.3f)" % (i, mt, a1, d_tok, gap), flush=True)
