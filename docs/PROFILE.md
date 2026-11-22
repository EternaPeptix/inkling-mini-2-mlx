# Decode-step profile (lane A)

Measured on 512S1 (M3 Ultra), mlx 0.32.4, vendored mlx-lm, greedy, batch 1, 512-token context, models `loopling-p{1,3,5}`.
Tools: `bench/profile_a.py` (per-token wall/CPU, build vs eval, per-group timings), `bench/hist_a.py` (primitive histogram
of the decode graph), `bench/loop_a.py` + macOS `sample` (where the main thread spends time).

## Verdict

**Decode is CPU-bound on graph dispatch, not GPU-bound.** The GPU executes tiny kernels and idles waiting for the
encoder. The main thread spends ~85% of wall time inside `mx.async_eval` (encoding ~12,000 primitives per token at 3
passes, ~1.9 us each); the per-token CPU time (35 ms, multi-threaded) exceeds the wall time (27 ms) and the weights
stream (~60 MB per layer visit, ~0.08 ms at 800 GB/s) is 6-7x faster than the measured 0.45-0.55 ms per layer visit.
Therefore: **per-token time ~ number of MLX primitives**, not bytes moved. The levers are fewer primitives (fusion,
merging matmuls, removing no-op casts / aranges / gathers) and less Python.

## Per-token cost (reference path, 3 passes, ctx 512)

| quantity | value |
|---|---|
| tok/s (own loop, async) | 37.1 (26.98 ms/token); bench.py reports 38.8 |
| sync wall per token | 27.7 ms = 4.6 ms graph build (Python+C++ op construction) + 23.1 ms eval (encode + run) |
| CPU time per token | 34-36 ms (> wall: encode threads + completion handlers) |
| graph size | 12,190 nodes at p3; 4,490 at p1 (about 187 per layer visit, 250 at p3 incl. head) |
| empty `mx.eval` floor | 0.17 ms |

Eval time by layer group, synchronised after each group (ms; `build` = Python graph construction, `total` = build + eval):

| group | layers | visits | build | total | per layer visit |
|---|---|---|---|---|---|
| prelude | L0-5 | 6 | 0.49 | 3.05 | 0.51 |
| loop pass 0 | L6-17 | 12 | 0.94 | 5.87 | 0.49 |
| loop pass 1 | L6-17 | 12 | 1.33 | 7.58 | 0.63 |
| loop pass 2 | L6-17 | 12 | 1.32 | 7.56 | 0.63 |
| tail | L18-23 | 6 | 0.50 | 3.27 | 0.54 |
| final_input + norm | | | 0.01 | 0.24 | |
| lm_head + softcap | | | 0.01 | 0.90 | (201k x 2304, 8-bit, bandwidth bound) |
| logsumexp + argmax | | | 0.02 | 0.33 | |
| sum | | | | ~27 | |

Loop passes >= 1 are 25% slower per layer than pass 0 and the prelude: they additionally compute the `message_input`
depth-attention bank and four `MessageGate`s (each a concat + 3 matmuls + 2 tanh) per layer visit. Sampling
(`logsumexp`, `argmax`) is 0.3 ms and the lm_head 0.9 ms (~4% together); the 24-48-72 layer visits are 90+% of the step.
Per-layer-visit work is identical across passes, so decode time is ~linear in visits: 10.8 ms (p1, 24 visits),
25.8 ms (p3, 48), 41 ms (p5, 72) = ~0.45 / 0.54 / 0.57 ms per visit.

## Where the primitives are (p1 reference graph, 4,490 nodes)

AsType 922, Broadcast 582, Reshape 248, Add 208, Squeeze 187, QuantizedMatmul 177, Transpose 152, Concatenate 147,
Slice 141, RMSNorm 131, ExpandDims 112, Sum 104, Divide 101, **Arange 100**, Convolution 88, compiled elementwise ~220,
CustomKernel 72, Full 64, SliceUpdate 48, GatherQMM 48, ... Broadcast/Reshape/Squeeze/Transpose/Slice/ExpandDims are
views (cheap but still visited). The real kernels per layer visit: ~17 casts, 7 quantized matmuls, ~5 RMSNorms,
~5 concats (short-conv windows), 4 short convs (cast, concat, cast, conv, cast+add+div), 4 `take_along_axis` + 4
`arange` (conv-state update), ~6 `arange`/`full`/compare ops that rebuild the attention mask for every visit (identical
for all 48 visits), 3 depth-attention custom kernels, router + GatherQMM x2.

Custom attention kernels (`layers._use_kernels`): `relative_attention` is only used for L > 1 (prefill); decode (L == 1,
B == 1) goes through `mx.fast.scaled_dot_product_attention` with an explicit additive mask that is rebuilt per visit.
`depth_attention` and `expert_routing` metal kernels are used at decode.

## Sampling / generate loop

`generate_step` already double-buffers (`async_eval(next_y)` before `y.item()` of the previous token), so host syncs are
not the problem; the sampler is argmax of log-softmax (0.3 ms). `stream_generate` adds detokenizer work on the main
thread, which is on the critical path because the main thread is also the encoder (bench.py 38.8 vs 37.1 tok/s for a
bare loop shows it is small).

## Results of the lane-A levers

See the table at the bottom (filled in when the final bench completes).
