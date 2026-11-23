# rows/qmv.py: exact multi-row affine 8-bit quantized matvec

`vendor/mlx_lm/models/inkling2/rows/qmv.py` (lane e1). Not wired into any decode path.

* `qmv_rows(x, w, scales, biases, group_size=64, bits=8)`: x (B, K) bf16, B = 1..16, returns (B, N) bf16.
* `qmv_rows_multi(x, [(w, s, b), ...])`: several weights sharing the same x in ONE launch; returns one (B, N_i) array per weight (the kernel has one output buffer per weight, so there is no slicing).

Row b is bitwise identical to `mx.quantized_matmul(x[b].reshape(1,1,K), w, s, b, transpose=True, group_size=64, bits=8, mode="affine")`.

## Design

At M=1 MLX 0.32.4 runs `affine_qmv_fast` (K % 256 == 0, K not in {64,128}; so not `qmv_quad`, and `qmv_wide` needs M >= 2) with 2 simdgroups per threadgroup, 4 output rows per simdgroup, `values_per_thread = 8`, `block_size = 256` (per `qmv_fast_impl`: each lane owns a fixed 8-wide K slice per block, accumulates `result[row] += qdot(...)` in float, then one reduction across the simdgroup).

The kernel inlines MLX's own `quantized.h` from the wheel (includes expanded recursively in Python; the part from `qouter` on is cut, since it needs `steel/` headers the custom-kernel preamble lacks) and calls MLX's `load_vector` unchanged. `qdot`'s 8-bit body is reproduced verbatim except that the weight bytes come from registers: for each (block, output row) the weight words are loaded once, then a loop over the B rows applies `scale * accum + sum * bias` with per-row `x_thread` / `sum` / `result[b][row]`. Same thread partition, same accumulation order, same `simd_sum`, same float types, so the arithmetic per output is the same code. B and K are template constants; the kernel is cached per number of weights. N % 8 != 0 uses the partial-rows tail (as `partial_rows` in MLX). Grid `(32, 2*tiles, 1)`, threadgroup `(32, 2, 1)`; multi launches concatenate the tile ranges of the weights along grid y.

Supported: affine, group_size 64, bits 8, bf16, K % 256 == 0, K not in {64, 128}; anything else raises ValueError (no silent fallback).

## Weights covered (loopling-p3, 177 affine 8-bit QuantizedLinear)

| (N, K) | count | modules |
|---|---|---|
| (24, 2304) | 24 | sink_proj |
| (32, 2304) | 8 | k_sconv / v_sconv weight_proj (full-attention layers) |
| (384, 2304) | 24 | r_proj |
| (1024, 2304) | 48 | k_proj, v_proj |
| (2304, 3072) | 24 | o_proj |
| (3072, 2304) | 48 | q_proj, gate_proj |
| (201024, 2304) | 1 | lm_head |

## Exactness (`bench/rows_qmv_test.py`, on 512S1)

Inputs per weight: N(0,1) bf16, rms-normalized x layernorm-like weights, and real activations captured from a 24-token decode (16 rows). B in {1,2,3,5,8,16}. Reference: per-row M=1 `mx.quantized_matmul`. Bit patterns (uint16) compared.

| case | cases | mismatching elements |
|---|---|---|
| (24,2304) | 432 | 0 |
| (32,2304) | 144 | 0 |
| (384,2304) | 432 | 0 |
| (1024,2304) | 864 | 0 |
| (2304,3072) | 432 | 0 |
| (3072,2304) | 864 | 0 |
| (201024,2304) | 18 | 0 |
| multi q+r+gate+sink, k+v, k/v sconv, all 24 layers | included in total | 0 |
| synthetic N in {36, 5, 1, 13} (partial rows) | included in total | 0 |
| **total** | **4152** | **0** |

Nothing was found that cannot be made exact.

## Speed (`bench/rows_qmv_speed.py`, 512S1, median of 60 reps; us per call)

"queued" = 32 launches per single eval (no sync latency). Reference at B=8 = 8 separate M=1 calls.

| weight | B | rows | reference | speedup |
|---|---|---|---|---|
| q_proj (3072x2304) | 1 | 17.2 | 17.8 | 1.04x |
| q_proj | 8 | 32.7 | 60.7 | 1.86x |
| k_proj (1024x2304) | 1 | 12.6 | 12.3 | 0.98x |
| k_proj | 8 | 18.6 | 56.1 | 3.02x |
| o_proj (2304x3072) | 1 | 15.3 | 14.1 | 0.92x |
| o_proj | 8 | 33.0 | 58.9 | 1.78x |
| sink_proj (24x2304) | 1 | 11.5 | 11.7 | 1.02x |
| sink_proj | 8 | 12.6 | 55.6 | 4.42x |
| lm_head | 1 | 697.5 | 684.8 | 0.98x |
| lm_head | 8 | 1696.7 | 5425.3 | 3.20x |
| multi q+r+gate+sink | 1 | 25.6 | 33.3 (4 calls) | 1.30x |
| multi q+r+gate+sink | 8 | 56.1 | 216.9 (32 calls) | 3.87x |

Dependent chain (output feeds next input), B=1, us per step: q_proj 26.1 vs 23.9, k_proj 24.2 vs 22.1, o_proj 26.7 vs 24.7, sink_proj 24.2 vs 22.5, lm_head 695.7 vs 694.2, multi q+r+gate+sink 34.7 vs 31.0.

Notes and caveats:
* B=1 device throughput (queued) is equal to MLX within +-4% noise (the 0.92x o_proj and 0.98x k_proj/lm_head figures are run-to-run noise: the first run measured o_proj 0.98x, q_proj 0.99x). B=1 is NOT faster; it is a tie. B=1 is not strictly "not slower" in the dependent chain: the custom-kernel path costs about 2-3 us more per call there, which is CPU-side dispatch/encode overhead of `mx.fast.metal_kernel` (not GPU time). It disappears when kernels are fused into larger chains (wave 2) or queued.
* B=8 gains are less than 8x for large weights (q/o_proj 1.8x) because at B=8 the kernel is no longer purely weight-bandwidth bound (8 rows of FMA per weight word); lm_head 3.2x.
* Weights are read once per launch regardless of B.
* Logs: `~/loopling-opt/runs/e1/qmv_test.log`, `qmv_speed.log` on 512S1.
