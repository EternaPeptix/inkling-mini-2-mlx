# Fused visit kernels (lane F): ~13 Metal launches per layer visit, B rows, bitwise exact

Why: decode is bound by GPU-side latency of dependent kernels (~3-4 us each, docs/LANE_E.md). A layer visit is
~85 MLX kernels today (docs/VISIT_OPS.md); p3 = 48 visits/token -> ~4,000 dependent launches. Weight bytes alone are
~4.5 ms/token. Target: ~13 launches per visit (+6 on looped passes > 0) for B = 1..8 rows (streams), each row bitwise
equal to the current default B=1 fast path. The same kernels serve C=1 (B=1) and C=8 (B=8).

## Reference and test data

* `vendor/mlx_lm/models/inkling2/fused_ref.py: visit_ref(layer, h, bs, msg, cs, attn_out, tau)` is the staged B=1
  reference. It returns every intermediate tensor by name (list below), and is bitwise equal to the compiled fast
  path on 144/144 real visits (`bench/fused_capture.py --check`).
* Real visits: `~/loopling-opt/runs/visits_p3/` on BOTH Studios (`index.json` + `vNNNN.safetensors`; 432 visits =
  3 prompts x 3 decode steps x 48 visits of loopling-p3). Per file: `h` (1,1,D) bf16, `bs` (n,1,1,D) f32 (absent on layer 0),
  `msg` (1,1,4,D) f32 (looped passes > 0 only), `tau` (1,1,1) f32 (full-attention layers only), conv states `cs0..cs3`
  (1,3,C) f32, `attn_out` (1,1,24,128) bf16, outputs `o`, `bs_new`, `ns0..ns3`. index.json gives layer, prompt, step,
  full/boundary/has_msg/has_bs.
* B-row tests: build B rows from B different captured visits of the SAME layer and visit kind (different prompts /
  steps; when B > 9, repeat with other layers of the same kind), run `visit_ref` on each row alone, run the fused
  kernel once on all B rows, compare BIT PATTERNS (uint16/uint32 views) per row. 0 mismatching elements required,
  B in {1, 2, 3, 5, 8}, all 24 layers / all visit kinds that use the kernel. Also add a few random-input rows.

Visit kinds: S/F = sliding/full attention layer (full: layers 5, 11, 17, 23; 3 attention-input queries, 3 input
layernorms, DynamicShortConvolution for k/v, tau); b = block boundary (layer % 6 == 0: residual := conv output and
bs_new = concat(bs, normed_residual)); m = looped pass > 0 (layers 6..17: 4 MessageGates); layer 0 has no bs and no
attention_input.

Sizes: D = 2304, 24 q heads, 8 kv heads, head_dim 128, d_rel 16, R = 1024, 128 experts top-8, expert hidden 1152,
gate rank 64. Weights: q/k/v/r/o/gate/sink proj and k/v_sconv.weight_proj: affine 8-bit gs64 (QuantizedLinear);
experts: NVFP4 (layers 0-19), MXFP8 (20-23); router f32 (128,2304); message gate down_proj bf16 (64,4608),
forget/message proj f32 (2304,64); depth-attention queries f32; layernorm weights bf16.

## Stage tensors (names from visit_ref) and kernels

| kernel | inputs -> outputs | notes |
|---|---|---|
| K1 PRE (row-local) | h, bs -> normed_residual, attn_input, then q_in/k_in/v_in (bf16) **or** on m-visits `*_gate_in` (bf16 4608) | rms, depth attention (existing `depth_attention` kernel code), input layernorm(s) |
| K2 GATE-DOWN (m only) | q/k/v_gate_in -> *_gate_down (bf16 64) | 3 gemv problems, one launch (rows/gemv.py) |
| K3 GATE-MIX (m only) | *_gate_down, s, m -> *_mixed (f32) | forget/message gemv f32 (6 problems) + tanh + combine epilogue |
| K3b NORM (m only) | *_mixed -> q_in/k_in/v_in | input layernorm(s), row-local |
| K4 QKV | q_in,k_in,v_in -> q_proj,r_proj,gate,sinks,k_proj,v_proj[,k_wproj,v_wproj] | ONE qmv launch, a different x per weight |
| K5 MID | k_proj,v_proj[,wproj,k_in,v_in],q_proj,r_proj,cs0,cs1,tau -> q,k,v,rel (bf16), w0,w1 | short / dynamic conv, l2 clips, value rms, rel clip, tau |
| K6 REL | rel -> position_bias | (B*24,16)@(16,1024) MLX matmul is already exact for B <= 42 (rows/gemv.py rel_logits_rows) |
| K7 KV | k,v -> per-row caches | today 2 slice_updates per row; lane F5: one batched store, 1-2 launches total |
| K8 SDPA | rows/sdpa.py (exact) | 1-2 launches |
| K9 OPROJ | attn_out, gate -> o_proj | qmv with prologue x = bf16(f32(attn) * sigmoid(f32(gate))) |
| K10 POST (row-local) | o_proj, h, normed_residual, bs, cs2 -> post_attn_conv, w2, residual, bs_new, normed, mlp_src, then mlp_x (bf16) **or** on m-visits mlp_gate_in | conv, residual, rms, depth attention, post_attention_layernorm |
| K10b/c/d (m only) | mlp gate: down, mix, norm -> mlp_x | as K2/K3/K3b with 1 gate |
| K11 ROUTER | mlp_x -> router_logits (f32 128) | rows/gemv.py |
| K12 GATEUP | router_logits, mlp_x -> expert_inds, expert_w (routing recomputed per threadgroup from the 128 logits, exact `expert_routing`), gate_up, swiglu (bf16 8x1152) | gather qmv + routing prologue + softcapped-swiglu epilogue |
| K13 DOWN | swiglu, expert_w, residual, cs3 -> down, moe_out, post_mlp_conv, w3, out | gather qmv over the row's 8 experts per output tile, left-fold combine, post-mlp conv and residual epilogue (all per output channel) |

Intermediates that a kernel does not need to export may stay internal, but every kernel must be testable against the
named tensors (export them behind a debug flag if needed).

## Rules for every kernel

* Bitwise per row: replicate MLX's per-element arithmetic, per-thread K split, accumulation order, simd/threadgroup
  reduction tree, casts and math functions (MLX's `precise::` / own sigmoid etc.; see docs/GLUE.md and
  rows/glue.py, which already matches rms_norm, l2_clip, sigmoid/tanh/exp/log, MoE combine, swiglu, message-gate
  combine, argmax). Weights read ONCE for all B rows. FP contraction: check (`#pragma clang fp contract(off)` /
  explicit `fma` where MLX's own kernel fuses).
* Values that change per step (positions, lengths, ring indices) are ARRAY inputs, never template constants or
  Python ints baked into the source, so a whole visit can be `mx.compile`d once and reused every step.
* B = 1 must not be slower than today's ops for the same work, and must be faster where you merged launches; report
  per-kernel GPU time (median of many reps, launches queued) for B = 1 and 8 vs the current ops it replaces.
* No silent fallback: unsupported shapes / dtypes raise.
* Code: `vendor/mlx_lm/models/inkling2/rows/<your module>.py`, tests `bench/fused_<name>_test.py`, speed
  `bench/fused_<name>_speed.py`, notes `docs/FUSED_<NAME>.md`. Do not wire into the decode path (the lead does).
