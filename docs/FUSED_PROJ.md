# rows/proj.py: fused projection kernels K4 QKV, K9 OPROJ, K2 GATE-DOWN, K3 GATE-MIX (lane f3)

Code `vendor/mlx_lm/models/inkling2/rows/proj.py`; tests `bench/fused_proj_test.py`; speed `bench/fused_proj_speed.py`. Not wired in.
qmv.py / gemv.py are unchanged (proj.py builds on their headers by text patching).

## Interface (B = 1..16 rows)
* `qmv_rows_multi_x(xs, weights, flat=False)`: xs = one (B,K) bf16 per weight (the same array object may repeat; unique ones are
  bound once), weights = [(w, scales, biases)] affine 8-bit gs64. ONE launch.
* `qkv_rows(a, q_in, k_in, v_in, full=None, flat=False)` -> dict `q_proj (B,3072) r_proj (B,384) gate (B,3072) sinks (B,24)
  k_proj (B,1024) v_proj (B,1024)` [+ `k_wproj (B,32) v_wproj (B,32)` when `a.k_sconv` has `weight_proj`, from k_in / v_in].
  q/r/gate/sinks read q_in, k_proj reads k_in, v_proj reads v_in. `flat=True` returns `(buf, [(name, offset, N)])`.
* `oproj_rows(attn_out, gate, w, scales, biases)`: attn_out (B,3072) or (B,1,24,128) bf16, gate (B,3072) bf16 -> (B,2304) bf16
  = `o_proj(bf16(f32(attn_out) * sigmoid(f32(gate))))`.
* `gate_down_rows(x, ws)`: x (B,G,4608) bf16, ws = G weights (64,4608) bf16 -> (B,G,64) bf16.
* `gate_mix_rows(down, ss, msg, slots, wf, wm)`: down (B,G,64) bf16, ss = G arrays (B,2304) f32 (may alias; the hidden state
  mixed into: sliding = attention-input query 0 for all three, full = queries 0,1,2, mlp = mlp_src), msg (B,4,2304) f32,
  slots = message index per gate ((0,1,2) for q,k,v; (3,) for mlp), wf / wm = G forget / message weights (2304,64) f32
  -> (B,G,2304) f32 `(1 - tanh(wf d)) * s + tanh(wm d) * msg[slot]`, d = f32(down).
Everything is a runtime array input (no per-step Python constants), so a whole visit can be `mx.compile`d. Unsupported
shapes/dtypes raise.

## Design
* K4/K9: `qmv._TILE` (MLX `qmv_fast_impl` per-output arithmetic, weights read once for all B rows) with a per-weight x pointer.
  The kernel has ONE flat output buffer: Metal allows 31 buffers and the full-attention QKV launch needs 2 x + 24 weight +
  8 outputs = 34 otherwise. Output i is (B, N_i) row-major at offset `B * (N_0 + .. + N_{i-1})`; the dict form returns zero-copy
  slices. A consumer kernel should read the flat buffer (see the chain numbers: 6 slice views cost ~20 us of CPU encode).
* K9: the tile has a prologue: instead of `load_vector`, each lane computes its 8 x elements `bf16(a * glue_sigmoid(g))`
  (glue `glue_gate`, MLX's `Sigmoid` + separate multiply, no fma) and sums them like load_vector's 8-bit body. Each threadgroup
  recomputes the elements it needs (cheap: sigmoid of 8 values per lane per block).
* K2: `gemv_rows_run` with separate x / out row strides (header patched textually), x laid out (B,G,K), one threadgroup range
  per gate (bm1 bn8 sm1 sn32 tm4 tn4: MLX config for 64x4608). Weights of the G gates are separate inputs.
* K3: one 32-thread threadgroup owns 32 output rows of one gate; the 4 lanes of a quad run MLX's (2304,64) f32 gemv config
  (bm1 bn1 sm8 sn4 tm4 tn4, same per-thread K partition, same `result += inter*v` expression, shuffle_down 2,1) for the forget AND
  the message weight; the lane holding both sums for row d does the epilogue `glue_msg_combine<0>` (plain mul/add, tanh = precise::tanh).
  No fma tweaks were needed anywhere.

## Exactness (`bench/fused_proj_test.py`, 512S1, log `~/loopling-opt/runs/f3/proj_test.log`)
Real visits of loopling-p3 (`runs/visits_p3`, up to 9 visits per layer, all 24 layers; message visits for layers 6..17), B rows =
B different visits of the same layer, B in {1,2,3,5,8}, windows at offsets 0 and 1, reference tensors from `visit_ref` run on each
visit alone; plus random rows (random weights not needed: real weights, random activations at several scales); K2 -> K3 chained
too; sliding layers also with q_in = k_in = v_in aliased. Bit patterns compared.

| kernel / tensor | cases | elements | mismatching |
|---|---|---|---|
| K4 q_proj, r_proj, gate, sinks, k_proj, v_proj (real, 24 layers) | 240 each | 0.3M-2.8M each | 0 |
| K4 same, aliased input (sliding, no msg) | 100 each | | 0 |
| K4 same, random | 120 each | | 0 |
| K4 k_wproj, v_wproj (layers 5,11,17,23; real / random) | 40 / 20 each | | 0 |
| K9 o_proj (real / random) | 240 / 120 | 2.1M / 1.05M | 0 |
| K2 gate_down q,k,v / mlp (real) | 324 / 108 | | 0 |
| K2 random q,k,v / mlp | 180 / 60 | | 0 |
| K3 gate_mix q,k,v / mlp (real, ref gate_down input) | 324 / 108 | | 0 |
| K3 K2 -> K3 chained | 324 / 108 | | 0 |
| K3 random q,k,v / mlp | 180 / 60 | | 0 |
| **total** | **5016** | **27.1M** | **0** |

Everything requested is bitwise exact; nothing needed a caveat.

## Speed (`bench/fused_proj_speed.py`, 512S1, median of 60 reps, 32 launches queued per eval, us per call)
ref = the MLX ops it replaces, B separate single-row calls (what the decode path does today; includes Python graph-build cost).
"ref compiled" = the same elementwise ops under `mx.compile` (matmul calls stay separate).

| kernel | B | fused | ref | speedup | ref compiled |
|---|---|---|---|---|---|
| K4 sliding (6 weights) | 1 | 39.5 | 37.2 | 0.94x | |
| K4 full (8 weights) | 1 | 44.7 | 40.4 | 0.90x | |
| K9 OPROJ | 1 | 19.0 | 27.4 | 1.44x | 20.7 (1.09x) |
| K2 G=3 | 1 | 10.8 | 21.1 | 1.95x | |
| K2 G=1 (mlp) | 1 | 9.5 | 10.6 | 1.12x | |
| K3 G=3 | 1 | 15.6 | 85.6 | 5.5x | 62.6 (4.0x) |
| K3 G=1 | 1 | 12.6 | 37.0 | 2.9x | 27.9 (2.2x) |
| K4 sliding | 8 | 75.5 | 221.5 | 2.9x | |
| K4 full | 8 | 96.0 | 270.5 | 2.8x | |
| K9 OPROJ | 8 | 61.6 | 148.4 | 2.4x | 90.8 (1.5x) |
| K2 G=3 | 8 | 12.6 | 123.4 | 9.8x | |
| K2 G=1 | 8 | 10.3 | 42.2 | 4.1x | |
| K3 G=3 | 8 | 22.0 | 581.2 | 26x | 409.2 (18.6x) |
| K3 G=1 | 8 | 16.2 | 198.0 | 12x | 139.1 (8.6x) |

Dependent chains, B=1 (output feeds next input, includes one concatenate per step in both variants): K4 sliding fused flat
buffer 42.7 us vs 6 module calls 41.4 us (fused with 6 slice views 62.7: slicing costs CPU encode, so consume the flat buffer);
K9 33.2 vs 37.7; K2 -> K3 (G=3) 50.5 vs 94.9.

Caveat (honest): K4 at B=1 is a TIE, not a win. It is weight-bandwidth bound (~21 MB of 8-bit weights per sliding visit, ~450 GB/s
effective, the same efficiency MLX's own qmv has at this size; lm_head reaches ~665 GB/s); merging 6-8 launches into one only
removes launches that the GPU already overlaps. Tried and rejected: full / x3 unrolling of the K loop (no change), 64-bit weight
loads (no change), preloading all blocks' weights into registers (0.6x, register pressure). The launch-count saving still pays when
K4 sits in a chain of dependent kernels (fewer encodes), and B=8 is 2.8-2.9x. K9 / K2 / K3 win at B=1.
