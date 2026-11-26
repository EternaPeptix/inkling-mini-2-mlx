import time
import numpy as np
import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2.batchdecode import BatchDecoder

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

model, tok = mlx_lm.load("/Users/jeweled/loopling/models/loopling-p3")


def encode(t):
    chat = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False)
    return tok.encode(chat + "<|message_model|><|content_text|>", add_special_tokens=False)


def prefill(ids):
    c = make_prompt_cache(model)
    p = mx.array(ids, mx.int32)
    while p.size > 1:
        n = min(2048, int(p.size) - 1)
        model(p[:n][None], cache=c)
        mx.eval([x.state for x in c])
        p = p[n:]
    return c


base_ids = [encode(t) for t in PROMPTS]
# Also test heterogeneous / long (>=1024) + short (<1024) mix!
mixed_ids = []
for i, x in enumerate(base_ids):
    if i % 2 == 1:
        pad = [(17 * j * j + 3 * j + 7 * i) % 40000 + 1000 for j in range(1100)]
        mixed_ids.append(pad + x)
    else:
        mixed_ids.append(x)

N_TOK = 96
refs = {}
for label, id_list in [("std", base_ids), ("mixed", mixed_ids)]:
    refs[label] = []
    for i in range(8):
        toks = []
        for t, _ in generate_step(mx.array(id_list[i], mx.int32), model, max_tokens=N_TOK + 16):
            toks.append(int(t))
        refs[label].append(toks)
print("Generated greedy references for 8 streams x 2 regimes.")

for label, id_list in [("std", base_ids), ("mixed", mixed_ids)]:
    for C, W in [(2, 4), (2, 8), (4, 2), (4, 4), (8, 2)]:
        caches = [prefill(id_list[i]) for i in range(C)]
        dec = BatchDecoder(model, caches)
        anc = [id_list[i][-1] for i in range(C)]
        out = [[] for _ in range(C)]
        pos = [0] * C
        step_idx = 0
        while min(pos) < N_TOK:
            blk = []
            for c in range(C):
                # Mix exact oracle prefixes of length 1..W across streams & steps so keep_c varies dynamically!
                match_len = (step_idx + c) % W
                row = [anc[c]]
                for d in range(1, W):
                    if d <= match_len and pos[c] + d < len(refs[label][c]):
                        row.append(refs[label][c][pos[c] + d])
                    else:
                        row.append((anc[c] + 13 * d + c) % 1000 + 10)
                blk.extend(row)
            t_cw, kv_news, conv_ws, _ = dec.verify_batch_w(mx.array(blk, mx.int32), W, tap_visits=("F",))
            mx.eval(t_cw)
            tl = t_cw.tolist()
            keeps = []
            for c in range(C):
                row_b = blk[c * W : (c + 1) * W]
                k = 1
                while k < W and tl[c][k - 1] == row_b[k]:
                    out[c].append(row_b[k])
                    k += 1
                bonus = tl[c][k - 1]
                out[c].append(bonus)
                anc[c] = bonus
                pos[c] += k
                keeps.append(k)
            dec.commit_batch_w(kv_news, conv_ws, keeps, W)
            step_idx += 1
        # Also test flushing & resuming AR 4 steps to verify stage/flush consistency
        dec.flush()
        for c in range(C):
            got = out[c][:N_TOK]
            exp = refs[label][c][:N_TOK]
            mm = sum(1 for a, b in zip(got, exp) if a != b)
            assert mm == 0, f"MISMATCH {label} C={C} W={W} stream={c}: mm={mm}"
        print(f"PASS {label:5s} C={C} W={W}: 0 mismatches across {N_TOK} tok x {C} streams!")
        del dec, caches
        mx.clear_cache()

# Timing sweep on std prompts
print("\n=== Warmed verify_batch_w Latency (std prompts) ===")
for C, W in [(2, 4), (2, 6), (2, 8), (4, 2), (4, 4), (8, 2)]:
    caches = [prefill(base_ids[i]) for i in range(C)]
    dec = BatchDecoder(model, caches)
    blk = mx.array([42] * (C * W), mx.int32)
    for _ in range(3):
        t, _, _, _ = dec.verify_batch_w(blk, W, tap_visits=("F",))
        mx.eval(t)
    ts = []
    for _ in range(25):
        t0 = time.perf_counter()
        t, _, _, _ = dec.verify_batch_w(blk, W, tap_visits=("F",))
        mx.eval(t)
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"  C={C} W={W} (B={C*W:2d}): {np.median(ts):.2f}ms/cycle")
    del dec, caches
    mx.clear_cache()
