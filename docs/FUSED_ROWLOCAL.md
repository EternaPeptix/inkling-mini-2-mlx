# FUSED_ROWLOCAL: K1 PRE, K3b NORM, K10 POST (lane f1)

Code `vendor/mlx_lm/models/inkling2/rows/rowlocal.py`; test `bench/fused_rowlocal_test.py`; speed `bench/fused_rowlocal_speed.py`.
Not wired into the decode path. Run: `LANE=f1 RHOST=512S1 scripts/r --gpu 'python bench/fused_rowlocal_test.py'`.

## Interface (B rows; every output row == fused_ref.visit_ref on that row alone)
- `k1_pre(h(B,D)bf16, bs(n,B,D)f32|None, msg(B,4,D)f32|None, query f32|None, ln_w, *, sliding, eps_ln=1e-6, debug=False) -> dict`
  `normed_residual (B,D)f32`; no msg: `q_in,k_in,v_in (B,D)bf16` (sliding: same array); msg: `s_q,s_k,s_v (B,D)f32`,
  `q_gate_in,k_gate_in,v_gate_in (B,4608)bf16`; debug adds `attn_input (B,G,D)f32`. `k1_pre_layer(layer,h,bs,msg)` takes weights from the layer.
- `k3b_norm(mixed (B,G,D)f32, ln_w, *, eps_ln) -> (B,G,D)bf16`; `k3b_norm_layer(layer, mixed, mlp=False)`.
- `k10_post(o_proj, h, nr, bs, cs2 (B,3,D)f32, conv_w, query, ln_w, msg, *, boundary, eps_ln) -> dict`
  `post_attn_conv bf16, w2 (B,3,D)f32, residual f32, normed f32, mlp_src f32`; no msg `mlp_x bf16`; msg `mlp_gate_in (B,4608)bf16`.
  bs_new is not materialized (boundary: bs plus the extra row nr). `k10_post_layer(layer, o_proj, h, nr, bs, cs2, msg)`.

## Layout and exactness
One threadgroup of 288 threads per row (K3b: per row and state, 576 threads). Arithmetic reuses glue.py device code (rms with
MLX virtual-thread layout and fma, as in GLUE.md), the repo depth_attention arithmetic (simd_sum, 9 simdgroup partials, softmax)
and the fastdecode short-conv fma sequence, all with fp contract off except where MLX fuses. Per-step values are array inputs.

## Results
Tests: all 24 layers, kinds S, Sb, F, Fm, Sm, Sbm, layer 0 (Sb0), B in {1,2,3,5,8}, real visits plus random rows at scales
0.1..30: 0 mismatching elements (PASS, ~5 s).

Speed on 512S1 (us per call, median, compiled N-call programs). "que" = independent calls queued; "chn" = dependent chain.
ref = the current MLX ops (compiled) for that stage, run B times for B rows (B=1 is the real comparison).

| stage | kind | B=1 ref que / fus que | B=1 ref chn / fus chn | B=8 fus que (ref 5-12x slower) |
|---|---|---|---|---|
| K1 | S/Sb | 11.9/8.4 | 30/18 | 9.0 |
| K1 | F | 18.5/10.2 | 34/26 | 12.0 |
| K1 | Sb0 | 9.6/8.2 | 13.5/14.5 | 8.3 |
| K1 | Sm/Sbm, Fm | 23/11.6, 27/14.4 | n/a | 13.3, 16.8 |
| K10 | S/Sb/F/Sb0 | 14.5-16.6/13.0-14.2 | 26/26 | 14.3-14.7 |
| K10 | Sm/Fm/Sbm | 18.5-19/13 | 26/28 | 14-15 |
| K3b | Sm/Fm/Sbm | 12-13.6/9 | n/a | 9.7 |

Notes: a trivial custom kernel costs ~7 us in a dependent chain (mx.add 4.8, mx.fast.rms_norm 7.5), so K1 S is near floor.
K10 on a dependent chain is on par with the MLX ops at B=1 (five internal barriers: conv, rms, depth attention, rms, norm), and
K1 layer 0 (a single rms) is on par. Gains at B=8 are large because the reference is B separate op chains.
