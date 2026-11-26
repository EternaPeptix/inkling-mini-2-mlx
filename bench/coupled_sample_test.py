"""Statistical distribution parity & throughput gate for coupled temp>0 DFlash2 + GPU sampling."""
import argparse
import collections
import os
import statistics
import sys
import time

import mlx_lm  # must precede `import mlx.core`
import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.inkling2.dflash2 import dflash2_exact_generate_step
from mlx_lm.sample_utils import make_sampler

from spec_bench import PROMPTS, encode_prompt


def check_parity(model, prompt_ids: mx.array, *, temp: float, top_p: float, trials: int = 300) -> float:
    sampler = make_sampler(temp=temp, top_p=top_p)
    ar_cnt = collections.Counter()
    cp_cnt = collections.Counter()
    os.environ["LOOPLING_DFLASH2_W"] = "8"
    os.environ["LOOPLING_DFLASH2_COUPLED_SAMPLE"] = "1"
    # Warmup
    _ = list(dflash2_exact_generate_step(prompt_ids, model, max_tokens=8, sampler=sampler))
    for t in range(trials):
        mx.random.seed(10000 + t)
        ar_toks = [int(x) for x, _ in generate_step(prompt_ids, model, max_tokens=2, sampler=sampler)]
        mx.random.seed(20000 + t)
        cp_toks = [int(x) for x, _, _ in dflash2_exact_generate_step(prompt_ids, model, max_tokens=2, sampler=sampler)]
        ar_cnt[ar_toks[1]] += 1
        cp_cnt[cp_toks[1]] += 1
    keys = set(ar_cnt) | set(cp_cnt)
    tv = 0.5 * sum(abs(ar_cnt[k] - cp_cnt[k]) for k in keys) / trials
    top5 = [k for k, _ in ar_cnt.most_common(5)]
    print(f"Parity (temp={temp}, top_p={top_p}, n={trials}): TV={tv:.3f}")
    print("  AR top5:", [(k, round(ar_cnt[k] / trials, 3)) for k in top5])
    print("  CP top5:", [(k, round(cp_cnt[k] / trials, 3)) for k in top5])
    assert tv < 0.15, f"Distribution TV too high: {tv}"
    return tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/loopling/models/loopling-p3"))
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    model, tok = mlx_lm.load(args.model)
    prompts = [encode_prompt(tok, p) for p in PROMPTS]
    g_samp = make_sampler(0.0)
    prefix = [int(t) for t, _ in generate_step(prompts[1], model, max_tokens=16, sampler=g_samp)]
    test_p = mx.concatenate([prompts[1], mx.array(prefix[:-1], mx.int32)])

    check_parity(model, test_p, temp=0.6, top_p=0.95, trials=args.trials)
    check_parity(model, test_p, temp=1.0, top_p=1.0, trials=args.trials)

    s_samp = make_sampler(temp=0.6, top_p=0.95)
    for mode in ("0", "1"):
        os.environ["LOOPLING_DFLASH2_COUPLED_SAMPLE"] = mode
        _ = list(dflash2_exact_generate_step(prompts[0], model, max_tokens=32, sampler=s_samp))
        tpss, tpcs = [], []
        for i, p in enumerate(prompts):
            mx.random.seed(100 + i)
            t1 = None
            toks = []
            for tid, _, _ in dflash2_exact_generate_step(p, model, max_tokens=args.tokens, sampler=s_samp):
                now = time.perf_counter()
                if t1 is None:
                    t1 = now
                toks.append(int(tid))
            dt = time.perf_counter() - t1
            st = dict(model.dflash2_stats)
            tpss.append((len(toks) - 1) / dt)
            tpcs.append(st["tokens"] / max(1, st["cycles"]))
        print(f"COUPLED={mode} MEAN8: {statistics.mean(tpss):.2f} tok/s | tpc={statistics.mean(tpcs):.3f}")


if __name__ == "__main__":
    main()
