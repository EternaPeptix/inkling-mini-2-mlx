"""Verify BatchDecoder integration in GenerationBatch / BatchGenerator (LOOPLING_SERVER_BATCHDECODE=1):
1. Static C=1,2,4,8 bit-exactness + throughput speedup vs LOOPLING_SERVER_BATCHDECODE=0 on loopling-p3.
2. Dynamic continuous batching (join mid-flight C=4->6, finish C=6->3->1, cache extract & resume) bit-exactness.

Usage:
    RHOST=512S2 scripts/r python3 bench/server_batch_test.py
"""
import os
import subprocess
import sys
import tempfile
import time

import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator


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


def _encode(tok, text):
    chat = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    return tok.encode(chat + "<|message_model|><|content_text|>", add_special_tokens=False)


def _prefill_b1(model, p):
    from mlx_lm.models.cache import make_prompt_cache
    c = make_prompt_cache(model)
    idx = mx.array(p[:-1])
    while idx.size > 0:
        n = min(2048, int(idx.size))
        model(idx[:n][None], cache=c)
        mx.eval([x.state for x in c])
        idx = idx[n:]
    return c


def run_static(model, encoded, C, max_tokens=64, *, b1_prefill=False):
    bg = BatchGenerator(model, prefill_step_size=2048, completion_batch_size=32, prefill_batch_size=8)
    if b1_prefill:
        caches = [_prefill_b1(model, p) for p in encoded[:C]]
        uids = bg.insert(
            [[p[-1]] for p in encoded[:C]],
            max_tokens=[max_tokens] * C,
            caches=caches,
            all_tokens=[p[:-1] for p in encoded[:C]],
            need_logprobs=[False] * C,
        )
    else:
        uids = bg.insert(encoded[:C], max_tokens=[max_tokens] * C, need_logprobs=[False] * C)
    toks = {u: [] for u in uids}
    done = 0
    t_dec0 = None
    while done < C:
        _, grs = bg.next()
        if grs and t_dec0 is None:
            t_dec0 = time.perf_counter()
        for r in grs:
            toks[r.uid].append(r.token)
            if r.finish_reason is not None:
                done += 1
    dt = time.perf_counter() - t_dec0
    bg.close()
    total_toks = C * max_tokens
    return (total_toks - C) / dt, [toks[u] for u in uids]


def run_dynamic(model, encoded):
    """Start C=4 (max_tokens 20,24,48,52), insert 2 more after 8 steps (C=4->6->4->3->1), resume from final cache."""
    bg = BatchGenerator(model, prefill_step_size=2048, completion_batch_size=32, prefill_batch_size=8)
    c0 = [_prefill_b1(model, p) for p in encoded[:4]]
    u0 = bg.insert([[p[-1]] for p in encoded[:4]], max_tokens=[20, 24, 48, 52], caches=c0, all_tokens=[p[:-1] for p in encoded[:4]], need_logprobs=[False] * 4)
    toks = {u: [] for u in u0}
    caches_out = {}
    steps = 0
    added = False
    active = 4
    while active > 0:
        _, grs = bg.next()
        if grs:
            steps += 1
        for r in grs:
            toks[r.uid].append(r.token)
            if r.finish_reason is not None:
                caches_out[r.uid] = (r.all_tokens[:], r.prompt_cache)
                active -= 1
        if steps == 8 and not added:
            added = True
            c1 = [_prefill_b1(model, p) for p in encoded[4:6]]
            u1 = bg.insert([[p[-1]] for p in encoded[4:6]], max_tokens=[18, 30], caches=c1, all_tokens=[p[:-1] for p in encoded[4:6]], need_logprobs=[False] * 2)
            for u in u1:
                toks[u] = []
            active += 2

    # Verify resuming stream u1[0] on a multi-turn continuation from its extracted cache is bit-exact
    prev_toks, prev_cache = caches_out[u1[0]]
    suffix = [1234, 5678, 9101]
    (u_res,) = bg.insert([suffix], max_tokens=[12], caches=[prev_cache], all_tokens=[prev_toks], need_logprobs=[False])
    res_toks = []
    while True:
        grs = bg.next_generated()
        for r in grs:
            res_toks.append(r.token)
            if r.finish_reason is not None:
                bg.close()
                # Cold reference of exact same full token sequence (prompt + turn1 + turn2 suffix)
                bg_cold = BatchGenerator(model, prefill_step_size=2048, completion_batch_size=32, prefill_batch_size=8)
                (u_c,) = bg_cold.insert([encoded[4] + toks[u1[0]] + suffix], max_tokens=[12], need_logprobs=[False])
                cold_toks = []
                while True:
                    cgrs = bg_cold.next_generated()
                    for cr in cgrs:
                        cold_toks.append(cr.token)
                        if cr.finish_reason is not None:
                            bg_cold.close()
                            return [toks[u] for u in (u0 + u1)], res_toks, cold_toks
    bg.close()


def worker(mode, out_path):
    os.environ["LOOPLING_SERVER_BATCHDECODE"] = mode
    model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
    mx.eval(model.parameters())
    encoded = [_encode(tok, p) for p in PROMPTS]

    # Reference single-stream greedy tokens for first 6 prompts (64 tokens)
    refs = []
    for p in encoded[:6]:
        _, ts = run_static(model, [p], 1, max_tokens=64)
        refs.append(ts[0])

    # Warmup C=1, 2, 4, 8
    for c in (1, 2, 4, 8):
        _ = run_static(model, encoded, c, max_tokens=16)

    results = {"mode": mode, "refs": refs, "static": {}}
    for c in (1, 2, 4, 8):
        runs = [run_static(model, encoded, c, max_tokens=64) for _ in range(2)]
        runs.sort(key=lambda r: r[0])
        tps, _ = runs[-1]
        _, exact_seqs = run_static(model, encoded, c, max_tokens=64, b1_prefill=True)
        mm = sum(a != b for i in range(min(c, 6)) for a, b in zip(exact_seqs[i], refs[i]))
        results["static"][str(c)] = {"tps": tps, "mm": mm}
        print(f"[mode={mode}] C={c}: {tps:7.2f} agg tok/s ({tps/c:6.2f}/stream), exact_mm_vs_c1={mm}", flush=True)

    dyn_seqs, res_toks, cold_toks = run_dynamic(model, encoded)
    # Compare against static 64-token single-stream runs
    dyn_mm = 0
    for i in range(6):
        dyn_mm += sum(a != b for a, b in zip(dyn_seqs[i], refs[i][: len(dyn_seqs[i])]))
    res_mm = sum(a != b for a, b in zip(res_toks, cold_toks))
    results["dyn_mm"] = dyn_mm
    results["res_mm"] = res_mm
    print(f"[mode={mode}] dynamic_join_leave mm={dyn_mm}, resumed_cache mm={res_mm}", flush=True)

    import json
    with open(out_path, "w") as f:
        json.dump(results, f)


def main():
    if "--worker" in sys.argv:
        idx = sys.argv.index("--worker")
        worker(sys.argv[idx + 1], sys.argv[idx + 2])
        return

    import json
    with tempfile.TemporaryDirectory() as td:
        p0 = os.path.join(td, "m0.json")
        p1 = os.path.join(td, "m1.json")
        subprocess.run(
            [sys.executable, __file__, "--worker", "0", p0],
            env=dict(os.environ, LOOPLING_SERVER_BATCHDECODE="0"),
            check=True,
        )
        subprocess.run(
            [sys.executable, __file__, "--worker", "1", p1],
            env=dict(os.environ, LOOPLING_SERVER_BATCHDECODE="1"),
            check=True,
        )
        r0 = json.load(open(p0))
        r1 = json.load(open(p1))

    print("\n=== BatchGenerator Throughput & Bit-Exact Comparison (loopling-p3) ===")
    for c in ("1", "2", "4", "8"):
        t0, t1 = r0["static"][c]["tps"], r1["static"][c]["tps"]
        mm0, mm1 = r0["static"][c]["mm"], r1["static"][c]["mm"]
        print(
            f"C={c}: orig={t0:7.2f} tok/s (mm={mm0:2d}) -> bd={t1:7.2f} tok/s ({t1/t0:.2f}x, exact_mm_vs_c1={mm1})",
            flush=True,
        )
        assert mm1 == 0, f"BatchDecoder C={c} mismatch vs C=1: {mm1}"
    assert r1["dyn_mm"] == 0 and r1["res_mm"] == 0, f"dynamic/resume mismatch: dyn={r1['dyn_mm']} res={r1['res_mm']}"
    print("PASS server_batch_test: BatchGenerator + BatchDecoder is 100% BIT-EXACT (0 mm) across C=1,2,4,8 and dynamic join/leave/resume!")


if __name__ == "__main__":
    main()
