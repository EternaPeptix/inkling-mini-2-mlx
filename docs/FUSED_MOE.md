# FUSED_MOE: stage E (MoE half of a visit) in 3 launches

Code `vendor/mlx_lm/models/inkling2/rows/moe.py`; tests `bench/fused_moe_test.py`; speed `bench/fused_moe_speed.py`. Not wired into decode.

## Interface (B <= 8, x bf16)
- `router_rows(x (B,2304) bf16, w (128,2304) f32) -> logits (B,128) f32` (K11, gemv_rows arithmetic, bf16->f32 cast in the prologue).
- `gateup_rows(logits, bias, x, w, scales, global_scale, *, mode, group_size, bits, route_scale, gate_softcap, up_softcap, score_eps=1e-20, route="kernel"|"launch") -> (expert_inds (B,8) u32, expert_w (B,8) f32, swiglu (B,8,1152) bf16)` (K12). `route="kernel"` recomputes routing inside each threadgroup (1 launch); `"launch"` runs `expert_routing` first (2 launches). Shared experts across rows are deduped (weights read once).
- `down_rows(swiglu, inds, wts, residual (B,2304) f32, cs3 (B,3,2304) f32, conv_w, w, scales, global_scale, *, mode, group_size, bits) -> (out (B,2304) bf16, w3 (B,3,2304) f32)` (K13: down matvec, left-fold combine, short conv, residual).
- `moe_rows(layer, x, residual, cs3, route=, debug=)` convenience dict (router_logits, expert_inds, expert_w, swiglu, out, w3; debug adds gate_up, moe_out, post_mlp_conv, down).
- Env: `MOE_NSG_GU`, `MOE_NSG_DN` (simdgroups per threadgroup, default 8), `MOE_TMAX_CAP` (default 2). All per-step values are array inputs.

## Exactness
`bench/fused_moe_test.py`, all 24 layers, B in {1,2,3,5,8}, real captured visits plus random rows (independent, identical, correlated experts, scaled x), both routing modes: 2520 cases, 442,232,448 elements, 0 mismatches (bit patterns) on router_logits, expert_inds, expert_w, gate_up, swiglu, down, moe_out, post_mlp_conv, w3, out.

## Speed (512S2, median, queued, us per call of the whole stage)
See the table in the commit message / runs/f4/moe_speed_2.txt. Routing in-kernel vs separate launch is a wash at B=1 (NVFP4 72.7 vs 73.4; MXFP8 110.4 vs 119.6) and at B=8 it is noise-level mixed (NVFP4 362 vs 317, MXFP8 538 vs 541), so default is route="kernel" for B=1.
