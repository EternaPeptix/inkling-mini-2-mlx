"""Bit-exactness & speed test for rows/head.py (fused softcap -> logsumexp -> argmax) vs BatchDecoder reference."""
import os
import statistics
import time

import mlx_lm  # noqa: F401
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2.batchdecode import BatchDecoder
from mlx_lm.models.inkling2.rows import head as RH

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


def ref_step(logits_bf16, sc, V):
    B = logits_bf16.shape[0]
    x = logits_bf16.astype(mx.float32)
    if sc is not None and sc > 0:
        x = mx.tanh(x / sc) * sc
    if logits_bf16.ndim == 3:
        x = x[..., :V][:, -1, :]
    else:
        x = x[:, :V]
    lses, outs = [], []
    for i in range(B):
        r = x[i : i + 1]
        lse = mx.logsumexp(r, keepdims=True)
        lp = r - lse
        lses.append(lse.reshape(1))
        outs.append(mx.argmax(lp, axis=-1))
    return mx.concatenate(lses), mx.concatenate(outs)


def check(name, logits_bf16, sc, V):
    ref_lse, ref_tok = ref_step(logits_bf16, sc, V)
    mx.eval(ref_lse, ref_tok)
    for split in (1, 8):
        for tg_lse in (256, 512, 1024):
            got_lse = RH.softcap_lse_rows(logits_bf16, sc, V, tg=tg_lse)
            got_tok = RH.softcap_argmax_rows(logits_bf16, sc, V, tg_lse=tg_lse, split=split)
            mx.eval(got_lse, got_tok)
            lse_bad = int(mx.sum(ref_lse.view(mx.uint32) != got_lse.view(mx.uint32)))
            tok_bad = int(mx.sum(ref_tok.astype(mx.uint32) != got_tok))
            assert lse_bad == 0 and tok_bad == 0, f"{name} tg={tg_lse} sp={split}: lse_bad={lse_bad} tok_bad={tok_bad}"
    print(f"PASS {name}: B={logits_bf16.shape[0]} V={V} lse=0 tok=0")


def main():
    model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
    mx.eval(model.parameters())
    sc = model.args.text.final_logit_softcapping
    V = model.args.text.unpadded_vocab_size
    P = model.lm_head.weight.shape[0]

    # 1. Real hidden -> lm_head logits at B=1, 2, 4, 8
    ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False)
                      + "<|message_model|><|content_text|>", add_special_tokens=False) for t in PROMPTS]
    caches = []
    for i in range(8):
        c = make_prompt_cache(model)
        model(mx.array(ids[i][:-1])[None], cache=c)
        mx.eval([x.state for x in c])
        caches.append(c)
    dec = BatchDecoder(model, caches)
    h8 = dec.hidden(mx.array([ids[i][-1] for i in range(8)]))
    lg8 = model.lm_head(h8)
    mx.eval(lg8)
    for B in (1, 2, 4, 8):
        check(f"real_p3_B{B}", lg8[:B], sc, V)

    # 2. Adversarial cases (exact ties, +/-inf, near-equal softcap saturation, small/tail V)
    mx.random.seed(7)
    adv = mx.random.normal((8, 1, P)).astype(mx.bfloat16) * 10.0
    adv[0, 0, 5] = mx.array(25.0, mx.bfloat16)
    adv[0, 0, 100] = mx.array(25.0, mx.bfloat16)  # exact tie -> must pick 5
    adv[1, 0, :] = mx.array(-100.0, mx.bfloat16)
    adv[1, 0, V - 1] = mx.array(90.0, mx.bfloat16)
    adv[1, 0, V + 5] = mx.array(200.0, mx.bfloat16)  # beyond unpadded V -> must ignore!
    adv[2, 0, 12] = mx.array(float("inf"), mx.bfloat16)
    adv[2, 0, 4] = mx.array(float("inf"), mx.bfloat16)  # earliest +inf tie
    adv[3, 0, :] = mx.array(-float("inf"), mx.bfloat16)  # all -inf -> tie at idx 0
    adv[4, 0, :] = mx.array(0.0, mx.bfloat16)
    adv[5, 0, 50000:50010] = mx.array(60.0, mx.bfloat16)  # softcap saturation ties
    mx.eval(adv)
    check("adversarial_8", adv, sc, V)
    check("no_softcap", adv, None, V)
    check("small_V_3", adv, sc, 3)
    check("unaligned_V", adv, sc, 4099)

    # 3. Microbenchmark at B=1 and B=8
    def timed(fn, reps=40, nq=32):
        for _ in range(5):
            mx.eval([fn() for _ in range(nq)])
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            mx.eval([fn() for _ in range(nq)])
            ts.append((time.perf_counter() - t0) / nq * 1e6)
        return statistics.median(ts)

    for B in (1, 8):
        x = lg8[:B]
        t_ref = timed(lambda: ref_step(x, sc, V)[1])
        print(f"B={B} ref(queued)={t_ref:6.1f} us", end="")
        for sp in (1, 4, 8):
            for tg_lse in (256, 512, 1024):
                if sp != 1 and tg_lse != 512:
                    continue
                t = timed(lambda: RH.softcap_argmax_rows(x, sc, V, tg_lse=tg_lse, split=sp))
                print(f" | lse{tg_lse}_sp{sp}={t:5.1f} us", end="")
        print("", flush=True)


if __name__ == "__main__":
    main()
