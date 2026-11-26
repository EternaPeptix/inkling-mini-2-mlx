"""Verify BatchDecoder integration in GenerationBatch / BatchGenerator (LOOPLING_SERVER_BATCHDECODE=1):
1. Static C=1..32 bit-exactness + throughput speedup vs LOOPLING_SERVER_BATCHDECODE=0 on loopling-p3.
2. Dynamic continuous batching (C=4->6->4->3->1 + C=16->24->32->16->8->1, cache extract & resume) bit-exactness.

Usage:
    RHOST=512S2 scripts/r python3 bench/server_batch_test.py [--C 1,2,4,8,16,32]
"""
import argparse
import json
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
    "Explain the difference between TCP and UDP with three real-world protocol examples for each.",
    "Write a SQL query using a window function to find the top three earners in each department.",
    "Derive the closed-form formula for the geometric series and evaluate it for r = 1/3.",
    "Outline a five-day itinerary for a first-time visitor to Kyoto focused on food and architecture.",
    "Explain how garbage collection works in Go, focusing on the tri-color mark-and-sweep algorithm.",
    "Write a concise C function to reverse a singly linked list in place and analyze its complexity.",
    "Compare photosynthesis and cellular respiration in terms of inputs, outputs, and organelles.",
    "Explain why floating-point addition is not associative with a concrete IEEE-754 example.",
    "Draft a polite email declining a job offer while keeping the door open for future collaboration.",
    "Describe how RSA public-key encryption generates keys, encrypts, and decrypts a small message.",
    "List six practical tips for debugging memory leaks in a long-running Python web service.",
    "Explain the economic concept of comparative advantage using a two-country, two-good numerical table.",
    "Write a regex that validates IPv4 addresses without leading zeros and explain each group.",
    "Summarize how CRISPR-Cas9 edits DNA and name two current technical limitations.",
    "Explain the CAP theorem and classify PostgreSQL, Cassandra, and ZooKeeper with caveats.",
    "Write a short dialogue between two astronomers debating whether to investigate an anomalous radio signal.",
    "Solve the recurrence T(n) = 2T(n/2) + n log n using the Master Theorem or recursion tree.",
    "Describe how a turbocharger differs from a supercharger and the trade-offs in engine design.",
    "Write a TypeScript generic LRU cache with O(1) get and put operations.",
    "Explain how vaccines train adaptive immunity, distinguishing B cells, T cells, and memory cells.",
    "Give a step-by-step guide to normalizing a messy CSV dataset with pandas before modeling.",
    "Explain the Byzantine Generals Problem and why 3f + 1 nodes are needed under synchrony.",
    "Write a haiku sequence of four poems tracing the four seasons in a mountain valley.",
    "Compare quicksort, mergesort, and heapsort on worst-case time, space, stability, and cache locality.",
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


def run_dynamic_32(model, encoded):
    """Start C=16, join +8 at step 6 (C=24), join +8 at step 12 (C=32), finish waves 32->16->8->1, resume from extracted cache."""
    bg = BatchGenerator(model, prefill_step_size=2048, completion_batch_size=32, prefill_batch_size=8)
    # 16 initial streams: 8 finish at 18, 8 finish at 36
    mt0 = [18] * 8 + [36] * 8
    c0 = [_prefill_b1(model, p) for p in encoded[:16]]
    u0 = bg.insert(
        [[p[-1]] for p in encoded[:16]],
        max_tokens=mt0,
        caches=c0,
        all_tokens=[p[:-1] for p in encoded[:16]],
        need_logprobs=[False] * 16,
    )
    toks = {u: [] for u in u0}
    caches_out = {}
    steps = 0
    added1 = False
    added2 = False
    u1, u2 = [], []
    active = 16
    while active > 0:
        _, grs = bg.next()
        if grs:
            steps += 1
        for r in grs:
            toks[r.uid].append(r.token)
            if r.finish_reason is not None:
                caches_out[r.uid] = (r.all_tokens[:], r.prompt_cache)
                active -= 1
        if steps == 6 and not added1:
            added1 = True
            # +8 streams (C=16->24), finish at step 6+12=18 (together with first 8 of u0 -> 32->16)
            c1 = [_prefill_b1(model, p) for p in encoded[16:24]]
            u1 = bg.insert(
                [[p[-1]] for p in encoded[16:24]],
                max_tokens=[12] * 8,
                caches=c1,
                all_tokens=[p[:-1] for p in encoded[16:24]],
                need_logprobs=[False] * 8,
            )
            for u in u1:
                toks[u] = []
            active += 8
        if steps == 12 and not added2:
            added2 = True
            # +8 streams (C=24->32), 7 finish at 12+32=44 (leaving C=1), 1 finishes at 12+38=50
            c2 = [_prefill_b1(model, p) for p in encoded[24:32]]
            u2 = bg.insert(
                [[p[-1]] for p in encoded[24:32]],
                max_tokens=[32] * 7 + [38],
                caches=c2,
                all_tokens=[p[:-1] for p in encoded[24:32]],
                need_logprobs=[False] * 8,
            )
            for u in u2:
                toks[u] = []
            active += 8

    # Verify resuming stream u2[0] (prompt 24) from its extracted cache is bit-exact
    prev_toks, prev_cache = caches_out[u2[0]]
    suffix = [1234, 5678, 9101]
    (u_res,) = bg.insert([suffix], max_tokens=[12], caches=[prev_cache], all_tokens=[prev_toks], need_logprobs=[False])
    res_toks = []
    while True:
        grs = bg.next_generated()
        for r in grs:
            res_toks.append(r.token)
            if r.finish_reason is not None:
                bg.close()
                bg_cold = BatchGenerator(model, prefill_step_size=2048, completion_batch_size=32, prefill_batch_size=8)
                (u_c,) = bg_cold.insert([encoded[24] + toks[u2[0]] + suffix], max_tokens=[12], need_logprobs=[False])
                cold_toks = []
                while True:
                    cgrs = bg_cold.next_generated()
                    for cr in cgrs:
                        cold_toks.append(cr.token)
                        if cr.finish_reason is not None:
                            bg_cold.close()
                            return [toks[u] for u in (u0 + u1 + u2)], res_toks, cold_toks
    bg.close()


def worker(mode, out_path, cs):
    os.environ["LOOPLING_SERVER_BATCHDECODE"] = mode
    model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
    mx.eval(model.parameters())
    encoded = [_encode(tok, p) for p in PROMPTS]
    max_c = max(cs)
    n_ref = max(max_c, 6) if mode == "1" else 6

    # Reference single-stream greedy tokens (64 tokens)
    refs = []
    for p in encoded[:n_ref]:
        _, ts = run_static(model, [p], 1, max_tokens=64)
        refs.append(ts[0])

    # Warmup
    for c in cs:
        _ = run_static(model, encoded, c, max_tokens=12)

    results = {"mode": mode, "static": {}}
    for c in cs:
        runs = [run_static(model, encoded, c, max_tokens=64) for _ in range(2)]
        runs.sort(key=lambda r: r[0])
        tps, _ = runs[-1]
        _, exact_seqs = run_static(model, encoded, c, max_tokens=64, b1_prefill=True)
        chk_n = min(c, len(refs))
        mm = sum(a != b for i in range(chk_n) for a, b in zip(exact_seqs[i], refs[i]))
        results["static"][str(c)] = {"tps": tps, "mm": mm, "checked": chk_n}
        print(f"[mode={mode}] C={c:2d}: {tps:7.2f} agg tok/s ({tps/c:6.2f}/stream), exact_mm_vs_c1={mm} (checked={chk_n})", flush=True)

    dyn_seqs, res_toks, cold_toks = run_dynamic(model, encoded)
    dyn_mm = sum(a != b for i in range(6) for a, b in zip(dyn_seqs[i], refs[i][: len(dyn_seqs[i])]))
    res_mm = sum(a != b for a, b in zip(res_toks, cold_toks))
    results["dyn_mm"] = dyn_mm
    results["res_mm"] = res_mm
    print(f"[mode={mode}] dynamic_join_leave(4->6->1) mm={dyn_mm}, resumed_cache mm={res_mm}", flush=True)

    if mode == "1" and max_c >= 16:
        d32_seqs, d32_res, d32_cold = run_dynamic_32(model, encoded)
        d32_mm = sum(a != b for i in range(32) for a, b in zip(d32_seqs[i], refs[i][: len(d32_seqs[i])]))
        d32_res_mm = sum(a != b for a, b in zip(d32_res, d32_cold))
        results["dyn32_mm"] = d32_mm
        results["dyn32_res_mm"] = d32_res_mm
        print(f"[mode={mode}] dynamic_join_leave(16->24->32->16->8->1) mm={d32_mm}, resumed_cache mm={d32_res_mm}", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", default="1,2,4,8,16,32")
    ap.add_argument("--worker", nargs=2, metavar=("MODE", "OUT"))
    args = ap.parse_args()
    cs = [int(x) for x in args.C.split(",") if x.strip()]
    if args.worker:
        worker(args.worker[0], args.worker[1], cs)
        return

    with tempfile.TemporaryDirectory() as td:
        p0 = os.path.join(td, "m0.json")
        p1 = os.path.join(td, "m1.json")
        subprocess.run(
            [sys.executable, __file__, "--C", args.C, "--worker", "0", p0],
            env=dict(os.environ, LOOPLING_SERVER_BATCHDECODE="0"),
            check=True,
        )
        subprocess.run(
            [sys.executable, __file__, "--C", args.C, "--worker", "1", p1],
            env=dict(os.environ, LOOPLING_SERVER_BATCHDECODE="1"),
            check=True,
        )
        r0 = json.load(open(p0))
        r1 = json.load(open(p1))

    print("\n=== BatchGenerator Throughput & Bit-Exact Comparison (loopling-p3) ===")
    for ci in cs:
        c = str(ci)
        t0, t1 = r0["static"][c]["tps"], r1["static"][c]["tps"]
        mm0, mm1 = r0["static"][c]["mm"], r1["static"][c]["mm"]
        print(
            f"C={ci:2d}: orig={t0:7.2f} tok/s (mm={mm0:3d}) -> bd={t1:7.2f} tok/s ({t1/t0:.2f}x, exact_mm_vs_c1={mm1})",
            flush=True,
        )
        assert mm1 == 0, f"BatchDecoder C={c} mismatch vs C=1: {mm1}"
    assert r1["dyn_mm"] == 0 and r1["res_mm"] == 0, f"dynamic/resume mismatch: dyn={r1['dyn_mm']} res={r1['res_mm']}"
    if "dyn32_mm" in r1:
        assert r1["dyn32_mm"] == 0 and r1["dyn32_res_mm"] == 0, f"dynamic32 mismatch: dyn={r1['dyn32_mm']} res={r1['dyn32_res_mm']}"
    print(f"PASS server_batch_test: BatchGenerator + BatchDecoder is 100% BIT-EXACT (0 mm) across C={args.C} and dynamic join/leave/resume!")


if __name__ == "__main__":
    main()
