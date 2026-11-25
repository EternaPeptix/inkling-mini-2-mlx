"""Benchmark & bit-exactness comparison on loopling-p3: AR baseline vs DSpark MTP vs Splash DFlash2."""
import argparse
import os
import statistics
import sys
import time

import mlx_lm  # must precede `import mlx.core`
import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.inkling2.dflash2 import dflash2_exact_generate_step
from mlx_lm.models.inkling2.mtp import mtp_exact_generate_step
from mlx_lm.sample_utils import make_sampler

from spec_bench import PROMPTS, count_mismatches, encode_prompt


def run_ar(model, prompt: mx.array, max_tokens: int, sampler):
    t1 = None
    out = []
    for tok_id, _ in generate_step(prompt, model, max_tokens=max_tokens, sampler=sampler):
        now = time.perf_counter()
        if t1 is None:
            t1 = now
        out.append(int(tok_id))
    dt = time.perf_counter() - t1
    return (len(out) - 1) / dt, out


def run_mtp(model, prompt: mx.array, max_tokens: int):
    t1 = None
    out = []
    for tok_id, _, _ in mtp_exact_generate_step(prompt, model, max_tokens=max_tokens):
        now = time.perf_counter()
        if t1 is None:
            t1 = now
        out.append(int(tok_id))
    dt = time.perf_counter() - t1
    st = dict(model.mtp_stats)
    tpc = st["tokens"] / max(1, st["cycles"])
    return (len(out) - 1) / dt, tpc, out


def run_df2(model, prompt: mx.array, max_tokens: int, quant: int, width: int, tree: bool, *, budget: bool = False, lookup: bool = False):
    os.environ["LOOPLING_DFLASH2_QUANT"] = str(quant)
    os.environ["LOOPLING_DFLASH2_W"] = str(width)
    os.environ["LOOPLING_DFLASH2_TREE"] = "1" if tree else "0"
    os.environ["LOOPLING_DFLASH2_BUDGET"] = "1" if budget else "0"
    os.environ["LOOPLING_DFLASH2_LOOKUP"] = "1" if lookup else "0"
    t1 = None
    out = []
    for tok_id, _, _ in dflash2_exact_generate_step(prompt, model, max_tokens=max_tokens):
        now = time.perf_counter()
        if t1 is None:
            t1 = now
        out.append(int(tok_id))
    dt = time.perf_counter() - t1
    st = dict(model.dflash2_stats)
    cyc = max(1, st["cycles"])
    tpc = st["tokens"] / cyc
    return (len(out) - 1) / dt, tpc, st["draft_ms"] / cyc, st["verify_ms"] / cyc, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/loopling/models/loopling-p3"))
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    sampler = make_sampler(0.0)
    print(f"Loading model: {args.model}")
    model, tok = mlx_lm.load(args.model)
    prompts_raw = list(PROMPTS) + [
        "Refactor this Python class by adding type hints, docstrings, and keeping all method names identical:\n"
        "```python\n"
        "class OrderBookProcessor:\n"
        "    def __init__(self, symbol, tick_size=0.01, max_levels=100, enable_audit=True):\n"
        "        self.symbol = symbol\n"
        "        self.tick_size = tick_size\n"
        "        self.max_levels = max_levels\n"
        "        self.enable_audit = enable_audit\n"
        "        self.bids = {}\n"
        "        self.asks = {}\n"
        "        self.sequence_number = 0\n\n"
        "    def apply_delta(self, side, price, quantity, sequence_number):\n"
        "        if sequence_number <= self.sequence_number:\n"
        "            return False\n"
        "        book = self.bids if side == 'bid' else self.asks\n"
        "        if quantity <= 0:\n"
        "            book.pop(price, None)\n"
        "        else:\n"
        "            book[price] = quantity\n"
        "        self.sequence_number = sequence_number\n"
        "        return True\n"
        "```\nReturn the full refactored class."
    ]
    prompts = [encode_prompt(tok, p) for p in prompts_raw]

    # Warmup
    _ = run_ar(model, prompts[0], 32, sampler)
    _ = run_df2(model, prompts[0], 32, quant=4, width=6, tree=False, budget=False, lookup=False)
    _ = run_df2(model, prompts[0], 32, quant=4, width=8, tree=True, budget=True, lookup=True)

    refs = []
    base_tps = []
    print(f"\n=== Autoregressive Baseline ({args.tokens} tok, P0..P7 + P8 code) ===")
    for i, p in enumerate(prompts):
        tps, out = run_ar(model, p, args.tokens, sampler)
        refs.append(out)
        base_tps.append(tps)
        print(f"P{i}: {tps:.2f} tok/s")
    print(f"AR MEAN8(P0..7)={statistics.mean(base_tps[:8]):.2f} tok/s | MEAN9={statistics.mean(base_tps):.2f} tok/s")

    total_mm = 0

    def bench_mode(label, quant, width, tree, *, budget=False, lookup=False):
        nonlocal total_mm
        print(f"\n=== DFlash2 [{label}] (P0..P8, {args.tokens} tok, median of {args.reps}) ===")
        tps_l, tpc_l, dm_l, vm_l = [], [], [], []
        mm_sum = 0
        for i, p in enumerate(prompts):
            runs = []
            mm_p = 0
            for _ in range(args.reps):
                tps, tpc, dm, vm, out = run_df2(model, p, args.tokens, quant, width, tree, budget=budget, lookup=lookup)
                mm = count_mismatches(out, refs[i])
                mm_p += mm
                mm_sum += mm
                runs.append((tps, tpc, dm, vm))
            runs.sort(key=lambda r: r[0])
            tps, tpc, dm, vm = runs[len(runs) // 2]
            tps_l.append(tps)
            tpc_l.append(tpc)
            dm_l.append(dm)
            vm_l.append(vm)
            print(
                f"P{i}: base={base_tps[i]:6.2f} df2={tps:6.2f} ({tps/base_tps[i]:.2f}x) "
                f"tpc={tpc:.2f} draft={dm:.2f}ms verify={vm:.2f}ms mm={mm_p}"
            )
        total_mm += mm_sum
        print(
            f"MEAN8 [{label}]: base={statistics.mean(base_tps[:8]):.2f} df2={statistics.mean(tps_l[:8]):.2f} "
            f"({statistics.mean(tps_l[:8])/statistics.mean(base_tps[:8]):.3f}x) MEAN9={statistics.mean(tps_l):.2f} "
            f"tpc={statistics.mean(tpc_l):.3f} draft={statistics.mean(dm_l):.2f}ms verify={statistics.mean(vm_l):.2f}ms mm={mm_sum}"
        )

    bench_mode("q4 fixed chain W=6", quant=4, width=6, tree=False, budget=False, lookup=False)
    bench_mode("q4 budget+tree W=8 (no lookup)", quant=4, width=8, tree=True, budget=True, lookup=False)
    bench_mode("q4 budget+tree+lookup W=8", quant=4, width=8, tree=True, budget=True, lookup=True)

    print(f"\n=== Built-in DSpark MTP (P0..P8, {args.tokens} tok) ===")
    _ = run_mtp(model, prompts[0], 32)
    mtp_tps, mtp_tpc = [], []
    mtp_mm = 0
    for i, p in enumerate(prompts):
        runs = []
        for _ in range(args.reps):
            tps, tpc, out = run_mtp(model, p, args.tokens)
            mtp_mm += count_mismatches(out, refs[i])
            runs.append((tps, tpc))
        runs.sort(key=lambda r: r[0])
        mtp_tps.append(runs[len(runs) // 2][0])
        mtp_tpc.append(runs[len(runs) // 2][1])
        print(f"P{i}: mtp={mtp_tps[-1]:6.2f} ({mtp_tps[-1]/base_tps[i]:.2f}x) tpc={mtp_tpc[-1]:.2f}")
    total_mm += mtp_mm
    print(f"MEAN8 [DSpark MTP]: mtp={statistics.mean(mtp_tps[:8]):.2f} ({statistics.mean(mtp_tps[:8])/statistics.mean(base_tps[:8]):.3f}x) MEAN9={statistics.mean(mtp_tps):.2f} tpc={statistics.mean(mtp_tpc):.3f} mm={mtp_mm}")

    if total_mm:
        print(f"\nFAIL: total_mismatches={total_mm}", file=sys.stderr)
        sys.exit(1)
    print("\nALL CHECKS BIT-EXACT (0 mismatches).")


if __name__ == "__main__":
    main()
