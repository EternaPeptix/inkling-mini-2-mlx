# rows/gemv.py: exact multi-row small dense matmuls (lane E4)

`vendor/mlx_lm/models/inkling2/rows/gemv.py`; tests `bench/rows_gemv_test.py`, `bench/rows_gemv_speed.py`. Not wired in anywhere.

## API
* `gemv_rows(x, w)`: x (B,K), w (N,K) as stored -> (B,N), output dtype = weight dtype.
* `gemv_rows_multi(x, [w1, ...])`: shared input (forget + message gate projections).
* `gemv_rows_group(xs, ws)`: independent (x_i, w_i) problems in one launch (the 4 or 3 gates of a layer: 4 down_proj, or 8 forget/message problems).
  Problems may differ in N (per-problem tile ranges are baked in); if their MLX block configs differ they are split into one launch per config.
* `rel_logits_rows(r, proj)`: r (B,24,16) bf16, proj (16,R) bf16 -> (B,24,R).
* x must be in the weight dtype (down: bf16 input; forget/message/router: f32 input; cast outside, as the reference does).
  Loaded dtypes (p3): down_proj bf16 (64,4608); forget/message f32 (2304,64); router f32 (128,2304); rel proj bf16 (16,1024) in all 24 layers (sliding and full: R = 1024 for both).

## Design
M == 1 `x @ w.T` in MLX 0.32.4 (`matmul.cpp::gemv_axbpy`, `kernels/gemv.h::GEMVKernel::run`, non-transposed variant because the weight is stored (N,K)).
`gemv_wide` does not apply at M == 1. Block config `(bm,bn,sm,sn,tm,tn)`: `bm = 8 if N >= 4096 else 4`, `sn = 32`; `K <= 64` -> `bm=1, sm=8, sn=4`;
else `K >= 16*N` -> `bm=1, bn=8`; `tm = 4, tn = 4`. For our shapes: down (64x4608) and router (128x2304) = bm1 bn8 sm1 sn32 tm4 tn4;
forget/message (2304x64) = bm1 bn1 sm8 sn4 tm4 tn4.

The row kernel is `GEMVKernel::run` transcribed with B accumulator sets per thread: same per-thread K partition (`bn = thread*TN + i*blockN`), same
`result[tm] += inter[tn]*v[tn]` order in float, same leftover handling, same `simd_shuffle_down` tree, same threadgroup reduction over simdgroups (BN > 1).
Each weight element is loaded once and applied to all B vectors; per-output arithmetic never depends on B. Kernels are JIT-compiled via `mx.fast.metal_kernel`
and cached per (dtype, B, problem list, K). No FMA tweaks were needed: the plain expression matched the offline-compiled MLX kernel bit for bit.

Relative logits: (1,1,24,16) @ (16,1024) bf16 is M=24, N=1024, K=16, nn -> `steel_matmul_regular_axpby` (no split-K since K/16 < 8, no NAX on M3),
kernel `steel_gemm_fused_nn_bf16_bf16_bm64_bn64_bk16_wm1_wn2` (device 'd', batch*M*N < 2^20, bf16). The same tile config is selected for any batch*M*N < 2^20 with
2*max(M,N) > K, and K=16 is one bk tile (two 8x8x8 MMA steps), so per-element arithmetic does not depend on M: one `(B*24,16) @ proj` GEMM is bitwise per row
equal to the M=24 call, and reads proj once. Asserted in code (B*24*R < 2^20, i.e. B <= 42). No custom kernel was needed.

## Exactness (bench/rows_gemv_test.py, 512S2, p3 weights, 0 mismatches required)
Per row bit compare vs the reference on that row alone (`(1,1,K) @ w.T`). "random" = realistic-scale random inputs (down input = concat of two rms-normalized rows; f32 gate input
N(0,4); router input bf16 N(0,9) cast f32; rel input bf16 N(0,1)); "real" = activations captured from an 11-step decode of the reference path. B in {1,2,3,5,8,16}.

| case | layers | cases per (src,B) | elements compared (B=16, per src) | mismatches (random, real, all B) |
|---|---|---|---|---|
| gate.down 64x4608 bf16 | 6..17 x q,k,v,mlp | 48 | 49152 | 0 / 0 |
| gate.forget 2304x64 f32 | same | 48 | 1769472 | 0 / 0 |
| gate.message 2304x64 f32 | same | 48 | 1769472 | 0 / 0 |
| forget+message one launch | same | 48 | 3538944 | 0 / 0 |
| 4 gates down one launch | 6..17 | 12 | | 0 / 0 |
| 4 gates forget+message one launch (8 problems) | 6..17 | 12 | | 0 / 0 |
| 3 gates (q,k,v) down / forget+message one launch | 6..17 | 12 | | 0 / 0 |
| router 128x2304 f32 | 0..23 | 24 | 49152 | 0 / 0 |
| rel_logits 24x16@16x1024 bf16 | 0..23 | 24 | 9437184 | 0 / 0 |

Total: 120 (case, source, B) groups, 0 failing, 0 mismatching elements. Negative control in the same script: the naive `x @ w.T` with M=B differs from the per-row reference
(24776 mismatching elements for B=5,8 on forget and down), so the comparison is sensitive.
Nothing is inexact.

## Speed (bench/rows_gemv_speed.py, 512S2, median of 60 reps, 48 launches queued per eval, per-call us)
"ref" = B separate reference single-row calls (what the decode path does today), including the Python graph-build cost of each call. Absolute numbers are therefore
dominated by per-op dispatch/encode overhead (~9-10 us); the weight-bandwidth part of each op is only ~1 us (<= 1.2 MB).

| case | B=1 ref | B=1 rows | B=8 ref (8 calls) | B=8 rows |
|---|---|---|---|---|
| gate.down | 10.8 | 9.2 | 59.9 | 10.6 |
| gate.forget | 10.8 | 9.7 | 59.6 | 11.5 |
| forget+message (1 launch vs 2 refs) | 18.2 | 12.3 | 115.6 | 12.5 |
| 4 gates down (1 launch vs 4 refs) | 33.0 | 16.8 | 232.6 | 20.8 |
| 4 gates forget+message (8 problems) | 65.2 | 28.2 | 478.0 | 30.3 |
| router | 10.9 | 9.4 | 60.3 | 10.5 |
| rel_logits (ref includes the `[None,None]` reshape) | 10.5 | 8.0 | 62.2 | 8.8 |

Dependent chain (router -> mix -> router ..., 24 weights x 8 rounds, per link incl. a small mix op): B=1 ref 22.5 us vs rows 19.5 us; B=8 ref 83.7 us vs rows 26.0 us.
B=1 is never slower than the reference.
