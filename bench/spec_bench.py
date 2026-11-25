"""Bit-exact greedy C=1 speculative decode benchmark & exactness gate (`mtp_exact_generate_step`)."""
import argparse
import os
import statistics
import sys
import time

import mlx_lm  # must precede `import mlx.core` so MLX_MAX_OPS_PER_BUFFER=32 applies
import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.inkling2.mtp import mtp_exact_generate_step
from mlx_lm.sample_utils import make_sampler

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


def encode_prompt(tok, text: str) -> mx.array:
    s = (
        tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False)
        + "<|message_model|><|content_text|>"
    )
    return mx.array(tok.encode(s, add_special_tokens=False), mx.int32)


def run_baseline(model, prompt: mx.array, max_tokens: int, sampler):
    t1 = None
    out = []
    for tok_id, _ in generate_step(prompt, model, max_tokens=max_tokens, sampler=sampler):
        now = time.perf_counter()
        if t1 is None:
            t1 = now
        out.append(int(tok_id))
    t2 = time.perf_counter()
    dec = (len(out) - 1) / (t2 - t1) if len(out) > 1 else 0.0
    return dec, out


def run_spec(model, prompt: mx.array, max_tokens: int):
    t1 = None
    out = []
    for tok_id, _, _ in mtp_exact_generate_step(prompt, model, max_tokens=max_tokens):
        now = time.perf_counter()
        if t1 is None:
            t1 = now
        out.append(int(tok_id))
    t2 = time.perf_counter()
    st = dict(model.mtp_stats)
    dec = (len(out) - 1) / (t2 - t1) if len(out) > 1 else 0.0
    tpc = st["tokens"] / max(1, st["cycles"])
    return dec, tpc, out


def count_mismatches(got, ref) -> int:
    return sum(a != b for a, b in zip(got, ref)) + abs(len(got) - len(ref))


def reset_peak_mem():
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def get_peak_mem_gb() -> float:
    b = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()
    return b / 1e9


def main():
    ap = argparse.ArgumentParser(description="Benchmark bit-exact speculative decoding on loopling-p1/p3.")
    ap.add_argument("--model", default=os.path.expanduser("~/loopling/models/loopling-p1"))
    ap.add_argument("--p3-model", default=os.path.expanduser("~/loopling/models/loopling-p3"))
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="Also verify S0=240, ctx=1020, and loopling-p3 (P0..P2, 128 tok)")
    args = ap.parse_args()

    sampler = make_sampler(0.0)
    print(f"Loading model: {args.model}")
    model, tok = mlx_lm.load(args.model)
    prompts = [encode_prompt(tok, p) for p in PROMPTS]

    # Warmup both baseline and MTP
    _ = run_baseline(model, prompts[0], 32, sampler)
    _ = run_spec(model, prompts[0], 64)

    total_mm = 0
    base_list, spec_list, tpc_list = [], [], []
    print(f"\n=== Canonical P0..P7 ({args.tokens} tokens, C=1 greedy, median of {args.reps} reps) ===")
    print(f"{'Prompt':<8} {'Base tok/s':>11} {'Spec tok/s':>11} {'Speedup':>9} {'tpc':>7} {'Mismatches':>11}")
    for i, p in enumerate(prompts):
        base_dec, ref = run_baseline(model, p, args.tokens, sampler)
        reps_res = []
        mm_prompt = 0
        for _ in range(args.reps):
            dec, tpc, out = run_spec(model, p, args.tokens)
            mm_prompt += count_mismatches(out, ref)
            reps_res.append((dec, tpc))
        reps_res.sort(key=lambda x: x[0])
        spec_dec, spec_tpc = reps_res[len(reps_res) // 2]
        total_mm += mm_prompt
        base_list.append(base_dec)
        spec_list.append(spec_dec)
        tpc_list.append(spec_tpc)
        print(
            f"P{i:<7} {base_dec:>11.2f} {spec_dec:>11.2f} {spec_dec / base_dec:>8.3f}x "
            f"{spec_tpc:>7.3f} {mm_prompt:>11d}"
        )

    m3_base = statistics.mean(base_list[:3])
    m8_base = statistics.mean(base_list)
    m3_spec = statistics.mean(spec_list[:3])
    m8_spec = statistics.mean(spec_list)
    m3_tpc = statistics.mean(tpc_list[:3])
    m8_tpc = statistics.mean(tpc_list)
    print("-" * 62)
    print(
        f"{'MEAN3':<8} {m3_base:>11.2f} {m3_spec:>11.2f} {m3_spec / m3_base:>8.3f}x "
        f"{m3_tpc:>7.3f} {total_mm:>11d}"
    )
    print(
        f"{'MEAN8':<8} {m8_base:>11.2f} {m8_spec:>11.2f} {m8_spec / m8_base:>8.3f}x "
        f"{m8_tpc:>7.3f} {total_mm:>11d}"
    )

    if args.full:
        print(f"\n=== Edge-Context Checks ({args.tokens} tokens, median of {args.reps} reps) ===")
        long_base = mx.concatenate([prompts[0]] * 16)
        for ctx_len in (240, 1020):
            pc = long_base[:ctx_len]
            base_dec, ref_c = run_baseline(model, pc, args.tokens, sampler)
            reset_peak_mem()
            reps_c = []
            mm_c = 0
            for _ in range(args.reps):
                dec_c, tpc_c, out_c = run_spec(model, pc, args.tokens)
                mm_c += count_mismatches(out_c, ref_c)
                reps_c.append((dec_c, tpc_c))
            reps_c.sort(key=lambda x: x[0])
            spec_c, tpc_med = reps_c[len(reps_c) // 2]
            peak_gb = get_peak_mem_gb()
            total_mm += mm_c
            print(
                f"ctx={ctx_len:<4d} base={base_dec:6.2f} tok/s  spec={spec_c:6.2f} tok/s  "
                f"speedup={spec_c / base_dec:.3f}x  tpc={tpc_med:.3f}  mm={mm_c}  peak_mem={peak_gb:.2f} GB"
            )

        print(f"\n=== loopling-p3 Verification (P0..P2, 128 tokens) ===")
        del model, tok
        mx.clear_cache()
        m3_model, tok3 = mlx_lm.load(args.p3_model)
        p3_prompts = [encode_prompt(tok3, p) for p in PROMPTS[:3]]
        _ = run_baseline(m3_model, p3_prompts[0], 16, sampler)
        _ = run_spec(m3_model, p3_prompts[0], 16)
        p3_mm = 0
        p3_base, p3_spec, p3_tpc = [], [], []
        for i, p in enumerate(p3_prompts):
            b3, ref3 = run_baseline(m3_model, p, 128, sampler)
            reps3 = []
            mm3 = 0
            for _ in range(args.reps):
                s3, t3, out3 = run_spec(m3_model, p, 128)
                mm3 += count_mismatches(out3, ref3)
                reps3.append((s3, t3))
            reps3.sort(key=lambda x: x[0])
            p3_mm += mm3
            p3_base.append(b3)
            p3_spec.append(reps3[len(reps3) // 2][0])
            p3_tpc.append(reps3[len(reps3) // 2][1])
            print(
                f"P{i:<7} base={b3:6.2f} tok/s  spec={p3_spec[-1]:6.2f} tok/s  "
                f"tpc={p3_tpc[-1]:.3f}  mm={mm3}"
            )
        total_mm += p3_mm
        print(
            f"P3_MEAN3 base={statistics.mean(p3_base):6.2f} tok/s  "
            f"spec={statistics.mean(p3_spec):6.2f} tok/s  "
            f"tpc={statistics.mean(p3_tpc):.3f}  mm={p3_mm}"
        )

    if total_mm != 0:
        print(f"\nFAIL: total_mismatches={total_mm} != 0", file=sys.stderr)
        sys.exit(1)
    print("\nALL CHECKS BIT-EXACT (0 mismatches).")


if __name__ == "__main__":
    main()
