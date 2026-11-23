# GLUE: exact device functions between the row kernels (lane e5)

Code: `vendor/mlx_lm/models/inkling2/rows/glue.py` (`GLUE_HEADER` = Metal source, paste via `mx.fast.metal_kernel(header=...)`;
the python wrappers in the same file are the per-snippet test kernels). Test: `bench/glue_test.py` (on 512S2:
`LANE=e5 RHOST=512S2 scripts/r --gpu 'python bench/glue_test.py [rms|l2|unary|moe|ew|custom|head]'`). Op inventory: `docs/VISIT_OPS.md`.
Nothing is wired into the decode path.

**Result (full run, real p3 weights, activations captured from a 20-step decode, B in {1,2,3,5,8,16}): 648 required cases, 0
mismatching elements. 492 additional "probe" cases (alternatives expected to differ, used to establish which variant MLX uses)
print PASS/FAIL but never gate.**

Global facts (MLX 0.32.4 source and behaviour):
* Built-in kernels come from the metallib built with `-fno-fast-math` (kernels/CMakeLists.txt) but with the Metal compiler's default FP
  contraction: `acc += x*x` in `rms_norm.metal` IS an fma (verified: the unfused variant mismatches up to ~18% of rows).
* MLX compiled elementwise kernels and `mx.fast.metal_kernel` JIT kernels use math mode "safe" (no fast-math). The compiled
  elementwise kernels do not contract (each op is a separate expression, materialized in its own type; verified by the message gate:
  the plain mul/add variant matches, fma variants do not). Every function in the header starts with `GL_NOFMA`
  (`#pragma clang fp contract(off)`) and uses explicit `fma()` where MLX's metallib fuses (rms only).
* Unary ops (kernels/unary_ops.h): `Sigmoid = 1/(1+precise::exp(|x|))`, then `x<0 ? y : 1-y`; `Tanh=precise::tanh`; `Exp=precise::exp`;
  `Log=precise::log`; `Sqrt=precise::sqrt`. bf16<->f32 via `bfloat16_t(f)` / `float(b)` equals `astype` bitwise (4M random bit patterns,
  incl. NaN/inf/subnormal/ties).

## (a) mx.fast.rms_norm  (`glue_rms_inv_sg`, `glue_rms_inv_tg`, `glue_rms_apply[_nw]`)
* MLX `rms_single_row` (N <= 4096; here N = 2304 or 128): one threadgroup per row of `32*ceil(ceil(N/4)/32)` threads (N=2304: 576 threads =
  18 simdgroups; N=128: 32). Thread v reads `x[4v..4v+3]` (past N: 0), `acc = 0; acc = fma(x_i, x_i, acc)` for i = 0..3 in order;
  `simd_sum(acc)`; lane 0 writes `local_sums[sg]`; barrier; `acc = simd_sum(lane < NSG ? local_sums[lane] : 0)`;
  `inv = precise::rsqrt(acc / N + eps)`; `out[i] = w[i] * T(x[i] * inv)` (unweighted: `T(x*inv)`).
* Emulation: `glue_rms_inv_sg<N,RF>(x, eps, lane)`: one simdgroup computes the whole row, each lane plays NSG virtual lanes in a loop
  (no barrier, no scratch; works wherever a full simdgroup owns the row). `glue_rms_inv_tg<N,TG,RF>(x, eps, part, lid)`: TG threads
  (multiple of 32), virtual simdgroup g runs on real simdgroup g % (TG/32); needs `threadgroup float part[32]`, ends with a barrier so
  `part` can be reused. The 32-lane `simd_sum` is the same hardware reduction as MLX's, so identical lane groupings give identical bits.
  `RF` = fma (true = MLX).
* Output `glue_rms_apply<T,W>(xi, inv, w) = T(w) * T(xi*inv)`, T = output dtype = result_type(x, w) (f32 x with bf16 w gives f32;
  bf16 x with bf16 w gives bf16). Callers pass `float(x)`.
* Value-head rms (128, f32 in, `.astype(bf16)` after): sg mode, one simdgroup per head, then `bfloat16_t(...)`.
* Tested: unweighted f32 N=2304; weighted f32 x (input_layernorm, post_attention_layernorm, q_input_layernorm of a full layer, final
  norm); weighted bf16 x (embed_norm, layer-1 norm); N=128 value head; random (scale 3, offset 0.5) and real activations; TG in
  {32 (sg), 64, 256, 576}; all B. **0 mismatches** with RF=true. RF=false mismatches 2-18% of rows in several cases (probes).

## (b) `_l2_clip`  (`glue_l2_clip_lane128`, `glue_l2_clip_row16`)
Compiled graph: `states = f32(x)*scale` (Multiply), `states*states` (separate Multiply, so no fma into the sum), `Sum(axis=-1)`, Sqrt,
`+1e-6`, `thr / .`, `Minimum(., 1)`, `states * .`, `astype(bf16)`.
* Row of 128 (q: 24 rows with scale; k: 8 rows): `row_reduce_looped` with `threadgroup_size_from_row_size(128) = 32`: lane l reads
  `x[4l..4l+3]` and does `total = x_i + total` sequentially from 0, then `simd_sum`. Snippet: one simdgroup per row, each lane holds its
  4 scaled states. Any layout works if a full simdgroup owns the row.
* Row of 16 (rel: 24 rows): `row_reduce_small` (row_size <= 64), a strictly sequential sum per output: `acc = x_i + acc`, i = 0..15.
  Snippet: one thread per row, no simd ops.
* `min` is MLX's NaN-propagating `Minimum` (`glue_min`). Tested: q, k, rel on random (scale 20/8) and real layer-7 projections, plus wide
  range (scale 300 x 0.1/1/10). **0 mismatches.**

## (c) unary ops and conversions
`glue_sigmoid/tanh/exp/log/sqrt`, `glue_min/max` equal `mx.sigmoid/tanh/exp/log/sqrt` on 1M random values in [-40,40], [-6,6] and
specials (0, -0, +-inf, NaN, +-88, 1e38), also composite `tanh(x/4)*4`, `tanh(x/25)*25`, `tanh(x/30)*30`: **0 mismatches**.

## (d) MoE combine  (`glue_moe_combine8`)
`(es.astype(f32) * w[...,None]).sum(axis=-2).astype(bf16)` over (1,1,8,2304): Multiply is a separate materialized f32 op (no fma);
Sum is the strided `col_reduce_small` (reduction 8, stride 2304): per column `t = p0 + 0; t = p1 + t; ... t = p7 + t` (left fold).
Snippet: one thread per output column holding 8 products. Matches both the `mx.compile`d and the eager expression.
**0 mismatches** (random and real expert outputs / router weights, all B).

## (e) elementwise chains
* **Softcapped swiglu** `(tanh(g/4)*4 * sigmoid(g)) * (tanh(u/25)*25)` (f32 math, bf16 in/out; true divides, separate multiplies):
  `glue_swiglu`. **0 mismatches** vs `L._softcapped_swiglu` on random and real gate_up outputs.
* **Message-gate combine** `(1 - tanh(f)) * h + tanh(m) * msg`: compiled MLX does NOT contract. Variant MV=0 (plain mul/add) matches
  100% vs a compiled mini-function and vs the real `MessageGate` modules (layers 7 and 11; q/k/v/mlp gates; all B); fma variants
  MV=1/2 mismatch 15-40% (probes). The bf16 `down_proj` and two f32 projections are gemv lane business.
* `glue_gate(a, g) = a * sigmoid(g)` (then `.astype(bf16)`), `glue_tau(x, tau) = x * tau` (f32, then bf16; tau from the host's
  `_full_tau` per row), residual `bf16(residual_f32 + f32(hidden))`: **0 mismatches** vs compiled and eager.

## (f) existing custom kernels
* `short_conv_decode_rows` (fastdecode.py): grid (C, B), one thread per channel; explicit `fma`s and per-thread arithmetic, so it can be
  inlined unchanged as a device function. Rows kernel vs reference `ShortConvolution` and vs the B=1 kernel (post-attn, post-mlp,
  k_sconv; layers 1 and 7): 0 mismatches.
* `expert_routing`: one simdgroup per token row (grid rows*32, tg 256), E=128 -> 4 logits per lane, top-k via simd_max/simd_min; rows
  independent. Same `_ROUTING_SOURCE` compiled in a kernel with the glue header: identical (B=1, 8); B rows in one launch == each alone.
* `depth_attention`: one threadgroup of `row_threads(2304)=288` threads per row, `block_sum` (simd_sum then a second simd_sum over
  per-simdgroup partials), K+1 sources, Q in {1,3,4}. Rows independent; compiling its source with `GLUE_HEADER` prepended gives identical
  bits (B=1, 8). Caveat: the JIT default contracts `a*b+c` across statements; keep any pasted code in its own device function so the
  surrounding `GL_NOFMA` pragmas (scoped per function) do not change it.

## (g) head: softcap, logsumexp, argmax  (`glue_logsumexp<TG>`, `glue_argmax_sub<TG>`, `softcap_rows`)
* Softcap `tanh(x/30)*30` per element; the slice `[:200058]` is only a bound. **0 mismatches** on real raw logits.
* `mx.logsumexp(logits (1,200058), keepdims)`: row > 4096 so the `logsumexp_looped` kernel: 1024 threads, N_READS=4, round r reads
  `in[r*4096 + 4t .. +3]` (past n: -inf); running `maxval` (init -FLT_MAX), `normalizer *= fast::exp(prevmax - maxval)` then
  `+= fast::exp(v - maxval)`; then `simd_max`, rescale, `simd_sum` per simdgroup and across the 32 simdgroups through threadgroup
  arrays; result `isinf(max) ? max : log(norm) + max`. The snippet emulates the 1024 virtual threads with TG in {256, 1024} real
  threads; needs `threadgroup float lmax[32], lnrm[32]`.
* `mx.argmax` (`arg_reduce_general`): larger value wins, ties go to the lowest index, a NaN beats everything (lowest NaN index): a
  total order, so any partition gives the same answer; snippet = strided scan + shuffle-down reduce. It is applied to `logits - lse`
  as `generate_step` does (f32 subtraction per element before the compare, so near-ties may merge: reproduced). Tested on 20 real
  decode steps (all B) and adversarial rows (many exact maxima, all equal, NaNs, -inf stripes, all -inf, coarse values, large offset),
  and the raw-argmax variant: **0 mismatches**. Chosen-token logprob = `lg[idx] - lse` with the same expression.

## Results (full run)
| group | required cases | mismatches |
|---|---|---|
| (a) rms norm (variants x random/real x B x TG, + value head) | 342 | 0 |
| (b) l2 clip | 38 | 0 |
| (c) unary + conversions | 24 | 0 |
| (d) moe combine (vs compiled and eager) | 24 | 0 |
| (e) swiglu, gate, tau, residual | 42 (+156 probes: msg-combine plain variant 52/52 match, fma variants 0/104) | 0 |
| (f) conv rows kernel, routing, depth attention, inlined sources | 108 | 0 |
| (g) softcap, logsumexp, argmax (real + adversarial) | 70 | 0 |
| total | 648 | 0 |

Not covered / limits (explicit):
* `rms_looped` (N > 4096) is not implemented (unused: N is 2304 or 128).
* Matmul / QuantizedMatmul / GatherQMM / SDPA / Gather / SliceUpdate belong to other lanes.
* The 16-wide `rel` clip as a sequential per-thread sum relies on MLX choosing `row_reduce_small` for the decode shape (24,16); other
  leading shapes (L > 1) may select a different reduce kernel and are untested.
* DynamicShortConvolution (full-layer k/v conv) is not a custom kernel today (Slice/Multiply/Sum/Add/Divide ops, see VISIT_OPS.md) and has no glue snippet.
