# K5 MID: rows/mid.py (lane f2)

Code `vendor/mlx_lm/models/inkling2/rows/mid.py`, test `bench/fused_mid_test.py`, speed `bench/fused_mid_speed.py`. Not wired in.

## Interface
`mid_rows(layer, q_proj, r_proj, k_proj, v_proj, cs0, cs1, tau=None, k_wproj=None, v_wproj=None, debug=False)`
* `layer`: decoder layer (or its Attention). Inputs: q_proj (B,H*128), r_proj (B,H*16), k_proj/v_proj (B,HK*128) bf16; cs0/cs1 (B,3,HK*128) f32.
  Full layers (sliding_window == 0) also `tau` (B,) f32 (per row), `k_wproj`/`v_wproj` (B,HK*4) bf16 (= `weight_proj` outputs); sliding layers must not pass tau.
* Returns `(q (B,H,128), k (B,HK,128), v (B,HK,128), rel (B,H,16))` bf16, `w0, w1` (B,3,HK*128) f32 (+ `k_conv, v_conv` (B,HK*128) bf16 if `debug`).
  Layout = visit_ref's before the head transpose. One launch, grid 6 threadgroups x 256 threads per row (3 q, 1 k, 1 v, 1 rel).
* Layer parameters (conv weights, static_weight, clip scales, thresholds, eps) are read from the layer; constants (thresholds, eps) are baked per layer
  config (cached kernel); tau, wproj, states are arrays, so a whole visit can be `mx.compile`d. Wrong shapes/dtypes raise.

## Arithmetic (all verified bitwise)
* sliding conv: `fastdecode._CONV_SRC` per channel (state cast to bf16, explicit fma chain, `(conv + x)/f32(sqrt 5)`).
* full conv (DynamicShortConvolution): `w_k = f32(static[k][c]) + f32(wproj[b, head*4+k])`; products with lag order (x, s2, s1, s0); Sum over the kernel axis is MLX
  `row_reduce_small`: sequential from 0, `acc = p + acc`; then `(acc + x)/3.0f`, bf16. New state = [s1, s2, x] (both conv kinds). No fma anywhere in the full conv (GL_NOFMA).
* q: `f32(q)*scale` -> glue l2 clip lane128 -> bf16 -> (full) `bf16(f32 * tau)`. k: clip lane128 (scale 1). v: glue rms inv sg<128, fma> + `T(x*inv)`.
  rel: scale -> glue l2 clip row16 (one lane per head) -> bf16 -> tau.

## Exactness (512S2, real p3 weights; bits compared per row against visit_ref's stage M of that row alone; 0 required)
Real = rows built from different captured visits of the same layer (all 24 layers, message and non-message visits, 3 row selections per B);
random = every layer, input scale 1/6/40, random tau/state. Outputs compared: q, k, v, rel, w0, w1, k_conv, v_conv.
| source | B | cases | mismatches |
|---|---|---|---|
| real sliding / sliding+msg / full / full+msg | 1,2,3,5,8 | 60/30/12/6 per B | 0 |
| random sliding / full | 1,2,3,5,8 | 60/12 per B | 0 |
Total 900 cases, 46.8M elements compared, 0 mismatches. (The test also asserts the helper reference equals visit_ref's tensors.)

## Speed (median us per call, 60 reps x 48 queued launches, cycling layers; includes host graph-build cost per call, which dominates)
"ref" = the stage-M ops of the default path (fused short-conv kernel or the DSC ops, l2 clips, rms, tau, `mx.compile`d), B separate single-row calls.
| layer kind | B=1 ref | B=1 mid_rows | B=8 ref (8 calls) | B=8 mid_rows |
|---|---|---|---|---|
| sliding | 44.8 | 21.3 | 348.0 | 23.6 |
| full (wproj given) | 83.8 | 22.6 | 690.9 | 24.9 |

## position_bias (stretch)
Not done: `rel @ rel_logits_proj.proj` stays the exact MLX matmul (`rows/gemv.py rel_logits_rows`, one (B*24,16)@proj launch for all rows). Replicating the steel GEMM's
simdgroup_matrix MMA fragments inside this kernel was not attempted.
