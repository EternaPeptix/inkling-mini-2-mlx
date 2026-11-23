# Lane E: exact high-throughput decode (C = 1..8 streams)

Goal: maximize total decode tok/s for 1..8 concurrent streams with **bitwise-identical logits** to the current default
path (lane A2 fast decode, itself bitwise = the reference path). Prefill comes later.

## Where the time goes (512S1, p3)
* C=1: wall 16-19 ms/token, but GPU-only time is ~5.2 ms (one command buffer per step:
  `MLX_MAX_OPS_PER_BUFFER=100000 MLX_MAX_MB_PER_BUFFER=1000000`, bench/batch_cpu.py). The rest is CPU: Python graph build
  (~2 ms) + MLX encode (~3 us per primitive x ~85 primitives per layer visit x 48 visits).
* C=8 (inkling2/batchdecode.py, exact): 48 ms/step, GPU-only ~17 ms. Same story, plus 8x weight re-reads in the
  "broadcast" qmv formulation and 8 SDPA/mask/slice_update chains per visit.
* Weight bytes per visit ~66-73 MB (attention 8-bit ~28 MB, 8 experts NVFP4 ~36 MB, router 1.2 MB, message gates 7 MB)
  -> bandwidth floor ~4.5 ms/token at p3, C=1.
* `MLX_MAX_OPS_PER_BUFFER=200` alone: p3 52.3 -> 61.4 tok/s, p1 124.5 -> 128.3 (tokens identical).

## Plan
1. **Row kernels** (this wave): one launch computes B rows (B = 1..16), weights read once, and every output row is
   **bitwise identical** to what the current B=1 decode path computes for that row alone (the specific MLX kernel MLX
   picks at M=1). Bonus: a T-row verify forward built from them is bitwise = T one-token steps -> exact speculative
   decoding later.
2. **Fused chains**: norms / depth attention / message gates / clips / swiglu / residuals folded into prologues and
   epilogues of the row kernels -> ~10-15 primitives per layer visit instead of ~85.
3. Batched caches (one KV write per visit), whole-step static shapes; then exact spec decode; then prefill.

## Contract for every row kernel
* Python entry point in `vendor/mlx_lm/models/inkling2/rows/<name>.py` built on `mx.fast.metal_kernel` (cache the kernel
  object per template config). No change to existing code paths in this wave (nothing is wired in yet).
* **Exactness:** for every weight in the model that will use the kernel (all 24 layers, real Loopling weights from
  `~/loopling/models/loopling-p3`), random inputs at realistic scale AND real activations (capture them from a decode
  step), B in {1,2,3,5,8,16}: compare bit patterns (`a.view(mx.uint16)` / `mx.uint32`) row by row against the reference
  op called on that row alone with the B=1 shapes the decode path uses. Required result: 0 mismatching elements.
  Test script `bench/rows_<name>_test.py`, prints one PASS/FAIL line per case and a final summary.
* **How to be exact:** replicate the reference kernel's per-output arithmetic exactly: same per-thread partition of the
  reduction, same accumulation order, same `simd_sum` structure, same float types and casts, same math functions.
  Preferred: inline MLX's own headers from the installed wheel
  (`~/loopling-opt/.venv/lib/python3.14/site-packages/mlx/include/mlx/backend/metal/kernels/*.h`, recursively expand the
  `#include "mlx/..."` lines yourself and pass the text as `header=`) and call MLX's device functions (`qdot`,
  `load_vector`, ...) directly. MLX source with host dispatch logic: `~/loopling-opt/ref/mlx` (v0.32.4, same as the
  wheel), e.g. `mlx/backend/metal/quantized.cpp`, `scaled_dot_product_attention.cpp`, `matmul.cpp`, `normalization.cpp`.
  Built-in MLX kernels come from `mlx.metallib` compiled offline; custom kernels are JIT compiled. If a result differs in
  the last bit, suspect FP contraction (a*b+c fused or not): write `fma()` explicitly or split the expression to match.
* **Speed:** report median us (>= 50 timed reps, inside the GPU lock) for B=1 vs the reference single call, and for
  B=8 vs 8 separate reference calls. B=1 must not be slower than the reference (it will also serve C=1). Weights must be
  read once per launch regardless of B.
* Commit only your own new files (`rows/<name>.py`, `bench/rows_<name>*.py`, a short `docs/ROWS_<NAME>.md` with
  results). Commit message ends with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Model facts (config `~/loopling/models/loopling-p3/config.json`)
hidden 2304, 24 layers (sliding + full attention mix, `local_layer_ids`), loop core = layers 6..17 run `num_passes`
times, vocab 201024 (unpadded 200058), heads 24 / kv heads 8 / head_dim 128, d_rel 16, rel_extent 1024, sliding window
1024, 128 experts top-8, expert intermediate 1152 (`gate_up` rows alternate gate/up), final logit softcap 30.
Quantization: experts NVFP4 (gs 16) in layers 0..19 and MXFP8 (gs 32) in layers 20..23; attention projections
(q/k/v/r/o/gate/sink), dynamic-conv weight_proj, embed and lm_head affine 8-bit gs 64; router weight f32 (nn.Linear,
not quantized); message-gate down_proj bf16, forget/message gate projs f32. Activations bf16 between layers.

## Running (see docs/IMPLEMENTING.md)
`LANE=<lane> RHOST=<512S1|512S2> scripts/r [--gpu] '<cmd>'` from your worktree root (zsh script; never run Python or
Metal on this Mac). Both Studios have the same venv (`~/loopling-opt/.venv`, mlx 0.32.4) and models. Use `--gpu` for
anything touching Metal; the lock serializes GPU use per host; keep each GPU command to a few minutes.

## Finding 2026-11-23: decode is GPU-bound by dependent-dispatch latency (not CPU, not bandwidth)

bench/overlap.py (p3, C=1, 512S2): with one command buffer per step (`MLX_MAX_OPS_PER_BUFFER=100000
MLX_MAX_MB_PER_BUFFER=1000000`) an isolated step is build 2.3 + encode 7.9 ms CPU, then **16.5 ms of GPU** after the
commit. Pipelined steady state = 15.9 ms/step = the GPU time (CPU work hides under it). Default buffering: the GPU runs
while encoding (encode 16 ms, tail 1.7 ms) - same total. Weight bytes alone are ~4.5 ms/step, so ~11 ms is GPU-side
latency of ~4,000 dependent kernels (~3-4 us each, matches the native Metal microbench: serial/concurrent+barrier
2.2-4 us per dependent dispatch). The earlier "GPU-only 5.2 ms" reading (wait after encode) was wrong: wait = GPU minus
overlap.

Consequence: the only big lever is fewer kernels per layer visit (fusion), at every C. A native host (Zig) would not
fix this by itself - the latency is on the GPU side - so we stay on MLX custom kernels (same Metal source would port).

Row kernels wired into inkling2/batchdecode.py (`LOOPLING_BD_ROWS`, default `qmv,gemv,sdpa`), exact 4/4:
p3 C=1/2/4/8 = 57.5 / 92.5 / 135.9 / 173.6 tok/s aggregate (was 55.0 / 85.0 / 124.9 / 166.9). At C=8 the GPU time is
~25 ms, dominated by per-row expert reads (64 expert slots/visit, ~52 distinct) -> E2 dedups shared experts.
