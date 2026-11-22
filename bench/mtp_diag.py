"""Exact-gate diagnosis for LOOPLING_MTP: (1) max |logit| diff of the first 32 steps, 8-token verify path vs 1-token decode path,
(2) at the first token flip between baseline and MTP, top-2 logit gap of the 1-token decode path."""
import json, os, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

PROMPTS = {
 "code": "Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
 "prose": "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
 "math": "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
}
passes = int(os.environ.get("PASSES", "3"))
base = json.load(open(os.path.expanduser("~/loopling-opt/runs/baseline.json")))["decode"]
mtpf = json.load(open(os.path.expanduser(sys.argv[1])))["decode"]
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p%d" % passes))


def decode_logits(ids, toks, nsteps):
    """1-token decode path (as generate_step): prefill ids[:-1], then feed ids[-1], toks[0], ... one at a time."""
    c = make_prompt_cache(model)
    model(mx.array(ids[:-1])[None], cache=c)
    out = []
    for t in [ids[-1]] + toks[: nsteps - 1]:
        out.append(model(mx.array([[t]]), cache=c)[0, -1]); mx.eval(out[-1])
    return mx.stack(out)


def verify_logits(ids, toks, nsteps, B=8):
    c = make_prompt_cache(model)
    model(mx.array(ids[:-1])[None], cache=c)
    seq = [ids[-1]] + toks[: nsteps - 1]
    out = []
    for s in range(0, nsteps, B):
        out.append(model(mx.array(seq[s : s + B])[None], cache=c)[0]); mx.eval(out[-1])
    return mx.concatenate(out)


def top2(lg):
    v = mx.sort(lg)[-2:]
    i = mx.argsort(lg)[-2:]
    return [int(i[1]), int(i[0])], float(v[1] - v[0])


for name, text in PROMPTS.items():
    prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>"
    ids = tok.encode(prompt, add_special_tokens=False)
    key = "p%d/%s" % (passes, name)
    b, m = base[key]["tokens"], mtpf[key]["tokens"]
    d = decode_logits(ids, b, 32); v = verify_logits(ids, b, 32)
    diff = mx.abs(d - v).max().item()
    for BB in (2, 4, 16, 32):
        vv = verify_logits(ids, b, 32, B=BB)
        print("  B=%d vs decode: max %.3f mean %.4f ; vs B=8 max %.3f ; argmax==baseline %s" % (BB, mx.abs(d - vv).max().item(), mx.abs(d - vv).mean().item(), mx.abs(v - vv).max().item(), mx.argmax(vv, -1).tolist() == b[:32]), flush=True)
    print("  B=8 mean|diff| %.4f" % mx.abs(d - v).mean().item())
    ad, av = mx.argmax(d, -1).tolist(), mx.argmax(v, -1).tolist()
    print("%s: max|logit diff| first 32 steps (8-token verify vs 1-token decode) = %.4f; argmax(decode)==baseline %s; argmax(verify)==baseline %s" % (
        name, diff, ad == b[:32], av == b[:32]), flush=True)
    flip = next((i for i, (x, y) in enumerate(zip(b, m)) if x != y), None)
    print("  baseline len %d, mtp len %d, first flip at token %s" % (len(b), len(m), flip), flush=True)
    if flip is not None:
        # decode-path logits at the flip position, teacher-forced on the common prefix
        c = make_prompt_cache(model)
        model(mx.array(ids[:-1])[None], cache=c)
        lg = None
        for t in [ids[-1]] + b[:flip]:
            lg = model(mx.array([[t]]), cache=c)[0, -1]; mx.eval(lg)
        (i1, i2), gap = top2(lg)
        print("  at flip: baseline tok %d, mtp tok %d; decode-path top-2 = (%d, %d) gap %.4f; logit[baseline]=%.4f logit[mtp]=%.4f" % (
            b[flip], m[flip], i1, i2, gap, lg[b[flip]].item(), lg[m[flip]].item()), flush=True)
