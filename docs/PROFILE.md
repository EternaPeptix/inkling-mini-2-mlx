# Decode-step profile (lane A)

Measured on 512S1 (M3 Ultra), mlx 0.32.4, vendored mlx-lm, greedy, batch 1, 512-token context, models `loopling-p{1,3,5}`.
Tools: `bench/profile_a.py` (per-token wall/CPU, build vs eval, per-group timings), `bench/hist_a.py` (primitive histogram
of the decode graph), `bench/loop_a.py` + macOS `sample` (where the main thread spends time).

## Verdict

**Decode is bound by the number of tiny kernels (dispatch/launch latency), not by bytes moved.** The weights stream
(~60 MB per layer visit, ~0.08 ms at 800 GB/s) is 6-7x faster than the measured 0.45-0.55 ms per layer visit. The main
thread spends ~85% of wall time inside `mx.async_eval` (encoding ~12,000 primitives per token at 3 passes, ~1.9 us
each; per-token CPU time 35 ms > wall 27 ms), and only ~15% waiting in `item()`, so CPU encode and GPU execution of
the small dependent kernels are nearly balanced: both scale with the primitive count. Per-token time ~ number of MLX
primitives. Python graph construction is only 4.6 of 27 ms. Levers: fewer / fused primitives and compiled (cached)
graph pieces. `MLX_MAX_OPS_PER_BUFFER` 20..1000 changes nothing (20.5-21.9 ms).

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

## Results of the lane-A levers (all exact: tokens identical to runs/baseline.json, 256 tokens x 3 prompts x passes 1/3/5)

Same session on 512S1, `bench/bench.py --prefill '' --reps 3`, medians over prompts of per-prompt medians
(tok/s; runs in `~/loopling-opt/runs/a/{off,c1,c1conv,c2}.json`). Max |logit| diff vs reference over 40-300 decode steps
(`bench/fastcheck_a.py`, incl. 1100- and 2100-token contexts that wrap the 1024 window / grow the full cache): 0.0.

| config (env) | p1 | p3 | p5 | exact gate |
|---|---|---|---|---|
| off (reference) | 91.7 | 38.0 | 24.1 | PASS |
| `LOOPLING_COMPILE_DECODE=2` (segmented path, hoisted masks, no compile) | 97.2 (+5.4%) | 39.6 (+2.4%) | 24.9 (+2.3%) | PASS |
| `LOOPLING_COMPILE_DECODE=1` (segments in mx.compile) | 116.8 (+26.7%) | 50.0 (+29.4%) | 31.5 (+30.0%) | PASS |
| `LOOPLING_COMPILE_DECODE=1 LOOPLING_FUSED_CONV=1` | 123.7 (+34.2%) | 53.7 (+38.9%) | 33.9 (+39.8%) | PASS |
| `LOOPLING_FUSE_QKV=1` (+COMPILE=1) | profile only: 26.1 ms/token vs 20.7 (slower; split adds slices/copies) | | | exact, not kept (code removed by lane A2: it froze concatenated weights) |

Graph nodes at p3: 12,190 -> 7,755 (compile) -> 7,019 (+fused conv).

What the flags do (`vendor/mlx_lm/models/inkling2/fastdecode.py`, hook in `DecoderLayer.__call__`, B=1, L=1, no padding):
the layer is split into S1 (norms, depth attention, projections, short convs, l2 clips) / cache update + attention core
(shapes vary with the cache, not compiled; mask pieces `clip/use_bias/fill` hoisted and shared by all visits of a step;
`tau` hoisted) / S2 (o_proj, MoE, convs, residual). S1/S2 are `mx.compile`d per layer (fixed shapes; one trace per
distinct input shape). `LOOPLING_FUSED_CONV` replaces cast+concat+cast+conv1d+cast/add/div of each decode
`ShortConvolution` by one Metal kernel, bit-identical to the reference (`bench/convtest_a.py`: 5,400 random cases on real
weights, 0 mismatches). Falls back to the reference path for batch > 1, prefill, padded/`lengths` caches.

Not done / ideas with measurable headroom: merge S2 of layer i with S1 of layer i+1 (one compile replay per layer and
fuse the residual cast into the next norm input); batch the four `MessageGate`s of looped passes (~35 kernels per
looped visit); fused kernels for `DynamicShortConvolution` (4 full layers); compile the lm_head softcap.
