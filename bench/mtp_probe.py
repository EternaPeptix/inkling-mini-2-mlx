"""Teacher-forced drafter acceptance for candidate tap mappings. Prompt + baseline greedy tokens are the ground truth."""
import json, os, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2.mtp import Drafter

PROMPTS = {
 "code": "Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
 "prose": "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
 "math": "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
}
passes = int(os.environ.get("PASSES", "3"))
base = json.load(open(os.path.expanduser("~/loopling-opt/runs/baseline.json")))["decode"]
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p%d" % passes))
tm = model.model.language_model
nvis = 6 + 12 * passes + 6
tm.tap_visits = set(range(nvis + 1))
dr = Drafter(model)
LAST = nvis - 1
VARIANTS = {
    "A i5,i23,i41,F": ["i5", "i23", "i41", "F"],
    "B i5,i23,i41,o": ["i5", "i23", "i41", "o"],
    "C i5,i23,i41,f": ["i5", "i23", "i41", "f"],
    "D i6,i24,i42,F": ["i6", "i24", "i42", "F"],
    "E i4,i22,i40,F": ["i4", "i22", "i40", "F"],
    "F i5,i23,i41,i47": ["i5", "i23", "i41", "i47"],
}
if os.environ.get("VARIANTS"):
    VARIANTS = {v: v.split(",") for v in os.environ["VARIANTS"].split(";")}
res = {k: [] for k in VARIANTS}
for name, text in PROMPTS.items():
    key = "p%d/%s" % (passes, name)
    gen = base.get(key, base["p3/" + name])["tokens"]
    prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>"
    ids = tok.encode(prompt, add_special_tokens=False)
    seq = ids + gen
    cache = make_prompt_cache(model)
    lg = model(mx.array(seq)[None], cache=cache)
    am = mx.argmax(lg[0], axis=-1)
    mx.eval(am, list(tm.tap_buf.values()))
    ok = (am[len(ids) - 1 : len(seq) - 1].tolist() == gen)  # sanity: teacher forcing reproduces baseline
    print(name, "len", len(seq), "prompt", len(ids), "teacher-forced argmax == baseline tokens:", ok, flush=True)
    for vn, taps in VARIANTS.items():
        feats = mx.stack([tm.tap_buf[t if t in ("F", "f", "o") else int(t[1:])] for t in taps], axis=2)
        dr.reset(); dr.append_ctx(feats)
        accs = []
        for P in range(len(ids), len(seq) - 8, 3):
            dr.truncate(P)
            d = dr.draft(mx.array([seq[P]]), feats[:, P - 1 : P]).tolist()
            truth = seq[P + 1 : P + 8]
            a = 0
            while a < 7 and d[a] == truth[a]:
                a += 1
            accs.append(a)
        m = sum(accs) / len(accs)
        res[vn].append(m)
        print("  %-22s mean accepted drafts %.2f (accept length %.2f) over %d blocks; first-token acc %.2f" % (vn, m, m + 1, len(accs), sum(1 for x in accs if x > 0) / len(accs)), flush=True)
print("SUMMARY")
for vn, v in res.items():
    print("%-22s %s mean %.2f" % (vn, [round(x, 2) for x in v], sum(v) / len(v)))
