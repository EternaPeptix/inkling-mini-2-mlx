"""Step anatomy of the batched decoder: CPU graph-build time vs eval time; same-prompt (8 shared experts) vs distinct."""
import os, sys, time, statistics
import mlx.core as mx
from mlx_lm import load
sys.path.insert(0, os.path.dirname(__file__))
P = int(os.environ.get("P", "3"))
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p%d" % P)); mx.eval(model.parameters())
import importlib.util
spec = importlib.util.spec_from_file_location("bd", os.path.join(os.path.dirname(__file__), "batch_dec.py"))
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2.batchdecode import BatchDecoder
PROMPTS = ["Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
           "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
           "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
           "Explain how a B-tree insertion works, step by step, with an example of a node split.",
           "Write a bash script that finds the ten largest files under a directory and prints their sizes in MB.",
           "Summarize the causes of the French Revolution in five bullet points, then give one counterargument.",
           "Implement binary search in Rust and explain the off-by-one pitfalls.",
           "Describe the water cycle to a ten-year-old, then list three common misconceptions."]
ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) + "<|message_model|><|content_text|>", add_special_tokens=False) for t in PROMPTS]
def prefill(i):
    c = make_prompt_cache(model); p = mx.array(ids[i])
    model(p[:-1][None], cache=c); mx.eval([x.state for x in c]); return c
for C, same in [(1, False), (2, False), (4, False), (8, False), (8, True)]:
    sel = [0] * C if same else list(range(C))
    dec = BatchDecoder(model, [prefill(i) for i in sel])
    y = mx.array([ids[i][-1] for i in sel])
    for _ in range(4):
        y = dec.step(y); mx.eval(y)
    tb, te = [], []
    for _ in range(16):
        t0 = time.perf_counter(); ny = dec.step(y); t1 = time.perf_counter(); mx.eval(ny); t2 = time.perf_counter()
        tb.append(t1 - t0); te.append(t2 - t1); y = ny
    # GPU-only estimate: build graph first, then time eval alone (already done above), and sync per-step total
    print("C=%d same=%s build %.2f ms  eval %.2f ms  total %.2f ms" % (C, same, 1e3 * statistics.median(tb), 1e3 * statistics.median(te),
          1e3 * (statistics.median(tb) + statistics.median(te))), flush=True)
