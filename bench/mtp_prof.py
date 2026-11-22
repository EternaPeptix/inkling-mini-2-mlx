"""Time the pieces of one MTP cycle (ms, medians)."""
import os, statistics, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2.mtp import Drafter, default_taps, feats_from_buf, prepare_record, rollback

model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
tm = model.model.language_model
taps = default_taps(3).split(",")
tm.tap_visits = {int(t[1:]) for t in taps if t[0] == "i"}
dr = Drafter(model)
ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Prove that the sum of the first n odd numbers is n^2."}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>", add_special_tokens=False)
c = make_prompt_cache(model); prepare_record(c)
lg = model(mx.array(ids)[None], cache=c); f = feats_from_buf(tm, taps); dr.append_ctx(f); mx.eval(lg, f, dr.ck, dr.cv)
anchor = mx.argmax(lg[:, -1], -1); last = f[:, -1:]


def t(fn, n=8):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts[2:])


def draft():
    mx.eval(dr.draft(anchor, last))


def ver(L):
    def go():
        blk = mx.array([int(anchor.item())] * L)
        o = model(blk[None], cache=c); mx.eval(o, feats_from_buf(tm, taps))
        rollback(c, L, 1) if L > 1 else None
        if L == 1:
            for x in c: x[0].offset -= 1  # undo (timing only)
    return go

print("drafter forward+markov: %.1f ms" % t(draft))
for L in (1, 2, 4, 6, 8):
    print("target %d-token forward (+rollback): %.1f ms" % (L, t(ver(L))))
