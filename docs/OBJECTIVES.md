# Loopling serving optimization (512S1)

Model: TML-d24-varloop-research ("Loopling"), MLX build `Xylolabs/TML-d24-varloop-research-MLX`
(experts NVFP4 L0-19 / MXFP8 L20-23, attention/embed/head affine 8-bit), on 512S1 (M3 Ultra, 512 GB).
`512S1:~/loopling/models/loopling-p{1..5}` = the same weights with `num_passes` 1..5. Runtime = vendored mlx-lm
(`vendor/mlx_lm`, PR 2086 @ 0c4c99192, MIT) on mlx 0.32.4. Loopling will later be replaced by a post-trained checkpoint
with the same architecture, so work must not depend on these exact weights.

## Metrics (bench/bench.py; medians of >= 3 runs; numbers only from runs holding the GPU lock)
1. Single-stream decode tok/s, greedy, 256 tokens, at passes 1, 3, 5 (code / prose / math prompts, geometric mean).
2. Prefill tok/s at 2k and 16k tokens, passes 3.
3. Later: batched decode throughput (C = 4, 16), time to first token.

## Gates
* **Exact (default for anything on by default):** greedy tokens identical to the baseline on every prompt at every
  pass count (`bench/exact.py`), for 256 tokens. Report the max |logit| diff of the first 32 steps.
* **Lossy (opt-in only, never default, needs the user's OK):** anything that changes tokens. Must report OpenDecisions
  and perplexity deltas next to the speedup.

## Levers (one swarm lane each; see IMPLEMENTING.md)
A. decode overhead: graph compile, cache handling, Python per-step cost, sync points
B. self-speculative decoding: draft with fewer passes of the same weights, verify with the target pass count
C. MTP head: the official checkpoint ships `mtp.safetensors` (3-layer drafter, block 8) that the MLX build dropped
D. MoE expert matmul at decode sizes (custom Metal kernels; exact only if the summation order is kept)
E. prefill: chunking, attention kernels
