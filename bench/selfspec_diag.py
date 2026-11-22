"""Lane B diagnostics for LOOPLING_SELFSPEC.
(1) rounding: teacher-force the greedy baseline tokens; compare 1-token-step logits with (k+1)-token verify-forward logits
    (max |dlogit| over the first 32 steps, argmax flips, top-2 gaps at flips).
(2) rollback: after a verify forward with a deliberately wrong tail, roll back and compare the next-step logits against a
    reference cache that processed only the kept tokens in one forward (rounding-level diff expected; O(1) means a bug).
    Run with --window 48 --ctx 200 to exercise RotatingKVCache past saturation.
usage: selfspec_diag.py --target 3 [--window W] [--ctx N]
"""
import argparse, os
import mlx.core as mx
from mlx_lm import load, selfspec
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser(); ap.add_argument("--models", default=os.path.expanduser("~/loopling/models"))
ap.add_argument("--target", type=int, default=3); ap.add_argument("--window", type=int, default=0)
ap.add_argument("--ctx", type=int, default=0, help="extra synthetic context tokens before the prompt")
ap.add_argument("--steps", type=int, default=40); ap.add_argument("--draft", type=int, default=1); a = ap.parse_args()
model, tok = load("%s/loopling-p%d" % (a.models, a.target)); mx.eval(model.parameters())
cfg = model.model.language_model.args.loop_config
if a.window:
    model.args.text.sliding_window_size = a.window
    for l in model.model.language_model.layers:
        if hasattr(l.self_attn, "sliding_window"): l.self_attn.sliding_window = a.window
text = "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future."
prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>"
ids = tok.encode(prompt, add_special_tokens=False)
if a.ctx: ids = [(17 * i * i + 3 * i) % 40000 + 1000 for i in range(a.ctx)] + ids
ids = mx.array(ids, mx.uint32)


def fresh(p, upto):
    cfg["num_passes"] = p; c = make_prompt_cache(model)
    for s in range(0, upto, 512):
        model(ids[s:min(s + 512, upto)][None], cache=c); mx.eval([x.state for x in c])
    return c


def step(c, toks, p):
    cfg["num_passes"] = p
    return model(mx.array(toks, mx.uint32)[None], cache=c)[0]


def gap(lg):
    v = mx.sort(lg)[-2:]; return (v[1] - v[0]).item()


P = ids.size; T = a.target
# baseline greedy with 1-token steps
c1 = fresh(T, P - 1); toks = [ids[-1].item()]; base_lg = []
for _ in range(a.steps):
    lg = step(c1, [toks[-1]], T)[-1]; mx.eval(lg); base_lg.append(lg); toks.append(mx.argmax(lg).item())
print("window", a.window or "default", "ctx", P)
for k in (2, 3, 4, 6):
    c = fresh(T, P - 1); worst = 0.0; flips = []
    seq = toks[:-1]  # inputs for steps 0..steps-1
    for s in range(0, len(seq), k + 1):
        chunk = seq[s:s + k + 1]; lg = step(c, chunk, T); mx.eval(lg)
        for j in range(len(chunk)):
            i = s + j; d = mx.abs(lg[j] - base_lg[i]).max().item()
            if i < 32: worst = max(worst, d)
            if mx.argmax(lg[j]).item() != mx.argmax(base_lg[i]).item(): flips.append((i, round(gap(base_lg[i]), 4), round(gap(lg[j]), 4)))
    print("k=%d max|dlogit| first32 = %.4f ; argmax flips (step, base_gap, verify_gap): %s" % (k, worst, flips))
def seq1(c, tt, p, ent):
    if ent == "target": step(c, tt, p)
    else:
        for t in tt: step(c, [t], p)


# rollback test
for k, keep in ((4, 2), (6, 1), (3, 3)):
    for ent in ("target", "draft"):
        p = T if ent == "target" else a.draft
        cfg["num_passes"] = p
        n0 = P - 1 + 5
        ca = fresh(p, P - 1); seq1(ca, toks[:5], p, ent)  # context + 5 tokens
        chunk = toks[5:5 + keep] + [7, 11, 13, 17, 19, 23][: k + 1 - keep]; assert keep <= k
        cfg["num_passes"] = p
        for cc in ca: cc[1].record = True
        S = len(chunk)
        if ent == "target":
            step(ca, chunk, p); mx.eval([x.state for x in ca]); selfspec._rollback_target(ca, S - keep, keep, S)
        else:
            snaps = {}
            for i, t in enumerate(chunk):
                if i == keep: snaps[i] = selfspec._snap(ca)
                step(ca, [t], p)
            selfspec._restore(ca, snaps[keep])
        cb = fresh(p, P - 1); seq1(cb, toks[:5 + keep], p, ent)
        nxt = toks[5 + keep]
        la = step(ca, [nxt], p)[-1]; lb = step(cb, [nxt], p)[-1]; mx.eval(la, lb)
        print("rollback %s k=%d keep=%d: next-step max|dlogit| vs reference %.4f (argmax %s)" % (
            ent, k, keep, mx.abs(la - lb).max().item(), "same" if mx.argmax(la).item() == mx.argmax(lb).item() else "DIFFERENT"))
