# loopling-opt

Bit-exact MLX inference and serving optimizations for **TML-d24-varloop-research ("Loopling" / Inkling2 looped MoE)** on Apple Silicon (`mlx 0.32.4`, vendored `vendor/mlx_lm`).

All default optimizations — including fused `B=1` fastdecode, `BatchDecoder` `C=2..32` batching in `BatchGenerator`/`server.py`, lossless BF16 `MessageGate` cache (`LOOPLING_MSG_BF16=1`), short-prompt tiled MoE prefill (`L=8..63`) + left/right-padded batched prefill (`B=2..8`), long-context Split-K full attention (`S >= 32K`), and periodic prefix checkpointing — are **100% bit-exact** against the reference greedy decode path on every prompt and loop pass count while preserving live weight mutability (`setattr`, `module.update`, `load_weights`, adapters). Speculative decoding (`LOOPLING_MTP=1` / `dflash2`), Loop Contrastive Decoding (`LOOPLING_LOOPCD`, plus `LOOPLING_LOOPCD_EXCLUDE_THINKING=1`), INT8 full-attention KV cache (`LOOPLING_KV_INT8=1`), multi-user batched DFlash2 (`LOOPLING_DFLASH2_BATCH=1`), and tiled prefill MessageGate (`LOOPLING_SPLASH_MSGGATE=1`) are configurable via environment variables.

## Highlights & Verified Performance

Measured on **Apple M3 Ultra (80-core GPU, 512 GB unified memory)**:

### 1. Bit-Exact Single & Batched Decode (`C=1..32`, 0 mismatches)

Direct `BatchDecoder` (`bench/batch_dec.py --passes 3 --tokens 64`, `exact=True`, `loopling-p3`):

| Context Length | `C=1` | `C=2` agg | `C=4` agg | `C=8` agg | `C=16` agg | `C=24` agg | `C=32` agg (`ms/step`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ctx=0` | **107.1** (9.34 ms) | **162.1** (12.34 ms) | **233.6** (17.12 ms) | **333.8** (23.97 ms) | **449.9** (35.56 ms) | **546.4** (43.93 ms) | **620.0 tok/s** (**51.61 ms**) |
| `ctx=384` | **101.8** (9.82 ms) | — | **222.5** (17.98 ms) | **325.3** (24.60 ms) | **428.4** (37.35 ms) | — | **582.1 tok/s** (**54.98 ms**) |
| `ctx=2048` | **97.8** (10.23 ms) | — | — | **295.9** (27.03 ms) | **381.0** (42.00 ms) | — | **497.4 tok/s** (**64.33 ms**) |

- `loopling-p1` (`p1`, `C=1..8`, `ctx=0`): `C=1: 204.7 | C=2: 304.3 | C=4: 433.1 | C=6: 544.0 | C=8: 631.7 tok/s (12.66 ms)`.
- Post-Wave-12 headroom audit steady-state `C=32` (`loopling-p3`): **`620.8 tok/s`** (`ctx=0`), **`600.7 tok/s`** (`ctx=384`), **`513.2 tok/s`** (`ctx=2048`), with MoE decode (`gu4`) saturating **100% of M3 Ultra `668 GB/s` DRAM bandwidth** (`102%` at `C=16`) and `qmv_rows` at the scalar-FP32 bit-exact ALU ceiling.

Serving `BatchGenerator` (`bench/server_batch_test.py --C 1,2,4,8,16,32`, `loopling-p3`, `LOOPLING_SERVER_BATCHDECODE=1` default ON):

| Concurrency | Unpatched `BatchGenerator` | With `BatchDecoder` (`SERVER_BATCHDECODE=1`) | Pure Decode (`b1_prefill=True`) | Speedup | Mismatches vs `C=1` |
|---|---:|---:|---:|---:|---:|
| `C=1` | 102.05 tok/s | **106.42 tok/s** | **106.69 tok/s** | 1.04x | 0 |
| `C=2` | 57.73 tok/s | **160.27 tok/s** | — | **2.78x** | **0** (vs 81 unpatched) |
| `C=4` | 99.70 tok/s | **231.60 tok/s** | — | **2.33x** | **0** (vs 95 unpatched) |
| `C=8` | 164.47 tok/s | **334.28 tok/s** | — | **2.03x** | **0** (vs 95 unpatched) |
| `C=16` | 242.04 tok/s | **436.22 tok/s** | **444.62 tok/s** | **1.80x** | **0** (vs 99 unpatched) |
| `C=32` | 301.32 tok/s | **496.09 tok/s** | **607.37 tok/s** | **1.64x** | **0** (vs 106 unpatched) |

- Dynamic continuous batching (`C=4 -> 6 -> 3 -> 1` and `C=16 -> 24 -> 32 -> 16 -> 8 -> 1`) + multi-turn cache extract/resume pass with **0 mismatches** and **0.0** live weight mutability (`BatchDecoder(C=4, 24)` + `a2_tests all`).
- **Opt-in Multi-User Batched DFlash2 (`LOOPLING_DFLASH2_BATCH=1`, default `0`, `C=2..4`):** 100% bit-exact (`0` mismatches across `C=2..8` and dynamic join/leave/resume); `bench/dflash2_batch_bench.py` achieves `C=4 W=3/4: 237.0 tok/s`, while pipelined `BatchGenerator` AR (`231.60` / `334.28 tok/s`) remains competitive on mixed prose, so batched DFlash2 is kept opt-in.

### 2. Bit-Exact Prefill Silicon Optimization (`loopling-p3`, `0` mismatches)

Cache-only E2E (`model.model.language_model`, `bench/prefill_exact_test.py` `L=2..16384` + `step=2048/4096/8192` + batched `B=4,8` + live-weight mutability all **0 mismatches**):

| Sequence / Chunk (`loopling-p3`) | Cache-Only E2E Latency | Throughput | Notes |
|---|---:|---:|---|
| `L=16 / L=32 / L=39` | `40.24 / 57.23 / 65.03 ms` | `397.6 / 559.1 / 599.7 tok/s` | Adaptive `pf_moe_short_tiled` + branchless `nb` (`B*L=8..63`) |
| `L=63` | **`81.30 ms`** | **`774.9 tok/s`** | `FP8_E4M3_LUT` + adaptive `pf_moe_csort_short_tiles` |
| `L=64 / L=128 / L=512` | `68.04 / 104.79 / 265.95 ms` | `940.6 / 1,221.5 / 1,925.2 tok/s` | `pf_moe_gmm` + `(64,32,32,2,1)` `pf_aff8_qmm` |
| `L=2048 / step=2048` | **`911.41 ms`** | **`2,247.1 tok/s`** | `FP4_H2_LUT` (`BM>=32`) + `pf_unsort_combine8` |
| **`L=4096 / step=4096`** | **`1,784.30 ms`** (`1,778.5 ms` audit) | **`2,295.6 tok/s`** (**`2,303.0 tok/s`**) | `pf_moe_gmm` **21.16 TF (91.5% peak)**; `aff8_linear` **23.0–25.9 TF** |
| **`L=8192 / step=8192`** | **`3,609.22 ms`** | **`2,269.7 tok/s`** | Fused `pf_mgin_down_s` (`N>=8192`) + `SWAP_GRID` rel/fg |
| `L=16384 / step=2048` | `8,000.97 ms` | `2,047.8 tok/s` | `uint4` K/V `relative_attention` + fused RMS/DepthAttn |
| **`L=16384 / step=4096`** | **`7,718.80 ms`** | **`2,122.6 tok/s`** | `pf_urms_da_f32` + `pf_sc_attn_da` (`-24 GB` DRAM @ 16K) |
| **Batched Padded `B=8 L=512`** | **`1,674.74 ms`** (`L=64: 264.80`, `L=256: 869.55 ms`) | **`2,445.7 tok/s`** | `HAS_MASK`/`HAS_LENS` fused convs for `BatchGenerator` (`0` mm) |

### 3. Single-Stream Exact Speculative Decode (`LOOPLING_MTP`, 0 mismatches)

- **DSpark MTP (`LOOPLING_MTP=1`, `bench/spec_bench.py --full`):** 3-layer DSpark drafter + Markov head (`vendor/mlx_lm/models/inkling2/mtp.py` with `@mx.compile` on `FastDrafter.append_ctx_gpu` & `draft_and_pack_tree_dink`) and fused causal tree verification (`W = 2..8`):
  - **`loopling-p1`:** `239.4–241.5 tok/s` mean over `P0..P7` (`257.7–260.4 tok/s` over `P0..P2`, peaking at `322.3 tok/s` on `P2`).
- **DFlash 2 Block-Diffusion Drafter (`LOOPLING_MTP=dflash2`, `bench/dflash2_bench.py --tokens 256 --reps 2`, `loopling-p3`):** 5-layer non-causal Qwen3 drafter (`incoai/Inkling-2-Mini-DFlash2`, Q4 default) + pipe-ahead overlap (`LOOPLING_DFLASH2_PIPE_AHEAD=1`) + hysteresis `{4, 6, 8}` `VerifyBudget` + 2-point `MoEStepCosts` + `W=16` snapped `LookupIndex` hybrid verify + single-pass `_CAUSAL_2PFUSED_SRC` (`LOOPLING_CW_2PF=1`) causal verify (`rows/sdpa.py`):

| Mode (`loopling-p3`, 256 greedy tok) | `MEAN8` (`P0..P7`) | `MEAN9` (`P0..P8` + code) | Code Refactor (`P8`) | Mismatches |
|---|---:|---:|---:|---:|
| Autoregressive baseline | 106.12 tok/s | 105.77 tok/s | ~103 tok/s | 0 |
| DSpark MTP (`LOOPLING_MTP=1`) | 129.02 tok/s | 137.83 tok/s | — | 0 |
| DFlash2 Q4 fixed chain (`W=6`) | 129.10 tok/s | 139.56 tok/s | — | 0 |
| **DFlash2 Q4 `VerifyBudget` + Tree (`W=8`, no lookup)** | **136.56 tok/s** | **148.05 tok/s** | — | **0** |
| **DFlash2 Q4 `VerifyBudget` + Tree + `W=16` Lookup (`W=8`)** | **136.47 tok/s** | **156.25 tok/s** | **314.46 tok/s (3.05x)** | **0** |

- **Lossless Non-Greedy Coupled DFlash2 (`LOOPLING_DFLASH2_COUPLED_SAMPLE=1`, `bench/coupled_sample_test.py`):** Fused GPU Sparse-Nucleus Top-P Gumbel-max sampler (`rows/head.py` with 32-lane SIMD `_SAMP_SRAM_P2_SRC`) + Leviathan coupled draft/verify (`rows/qmv4.py` 8-lane epilogue, `batchdecode.py`): `temp=0.6, top_p=0.95` sampled `MEAN8` reaches **`121.86 tok/s`** (vs `114.65 tok/s` uncoupled, parity `TV=0.067`), greedy `0` mismatches.
- **Resident RAM Optimization:** Immediately freeing BF16 DFlash2 buffers after Q4 quantization reduces drafter resident footprint from `+0.929 GB` to `+0.805 GB` (**−124 MB**).

### 4. LoopCD Quality Boost (`LOOPLING_LOOPCD`, `loopling-p3`)

Implements **Loop Contrastive Decoding** ([arXiv:2610.02185](https://arxiv.org/abs/2610.02185)) across recurrent loop passes in [`vendor/mlx_lm/models/inkling2/loopcd.py`](vendor/mlx_lm/models/inkling2/loopcd.py), with optional `ThinkingTracker` (`LOOPLING_LOOPCD_EXCLUDE_THINKING=1` or `exclude_thinking=True`) to restrict contrastive subtraction cleanly outside thinking spans:

| Config (`loopling-p3`) | Compute Overhead | Gen (`n=40`) | Code (`n=12`) | `mean_rep4` | OpenDecisions Acc / NLL (`n=300`) |
|---|---|---:|---:|---:|---:|
| `off` (default, bit-exact) | 1.000x | 36/40 (90.0%) | 75.0% | 0.1388 | 49.3% / 1.345 |
| **`hidden:0.3`** | **1.000x (0 extra FLOPs)** | **37/40 (92.5%)** | **83.3%** | **0.1320** | 49.7% / 1.309 |
| `hidden:0.75` | 1.000x | 37/40 (92.5%) | 75.0% | 0.1251 | 52.0% / 1.272 |
| `hidden:1.0` | 1.000x | — | — | — | **54.0% (+4.7 pts)** / 1.273 |
| `logits_adaptive:1.0` | +1 coda/`lm_head` | **38/40 (95.0%)** | 83.3% | 0.1357 | 50.0% / 1.381 |

### 5. OpenAI HTTP Server, Prefix Checkpointing, MessageGate & KV Options

- **Bit-Exact Server Logprob, DFlash2 `C=1` Routing & `BatchDecoder` Serving (`C=1..32`):** Fixed unconditional per-token GPU logprob synchronization in `vendor/mlx_lm/server.py` (`_token_logprob`), wired `sdpa_rows(..., gate=gate)` + `qmv_rows` for `B=1` `BatchKVCache`/`BatchRotatingKVCache`, integrated `BatchDecoder` (`C=1..32`) directly into `GenerationBatch` with direct 1-token prefilled request admission (`generate.py`), and added direct `C=1` `LOOPLING_MTP=dflash2` serving with exact multi-turn cache sync, system-head checkpoint protection, and persistent SSD `LRUPromptCache` round-trips (`--persistent-cache`, `--persistent-cache-gb`, verified in `bench/server_dflash2_cache_test.py`).
- **Long-Context Split-K Full Attention (`S >= 32K`, `LOOPLING_LONG_SDPA=1`, `bench/long_sdpa_test.py`):** Split-K P1 (`SPLIT_K=256`, zero-barrier `TK=4`) + P2 (`1024`-thread `bf16` partials) in `rows/sdpa.py` & `batchdecode.py`: `64K` kernel `W=4` `1.979 -> 1.004 ms` / `W=8` `3.494 -> 1.618 ms`; `100K` kernel `W=4` `3.082 -> 1.343 ms` / `W=8` `2.304 ms` (`1422.0 GB/s eff`); E2E DFlash2 `32.8K` `201.89 -> 221.43 tok/s` and `65.6K` `43.18 -> 46.51 tok/s` (`mm_vs_AR = 0`).
- **Bit-Exact Periodic Prefix Checkpointing (`LOOPLING_PREFIX_CHECKPOINT=2048`):** Snapshots `(KV + ArraysCache)` states every 2048 tokens during long prefills/decodes into a dedicated lowest-priority `"checkpoint"` eviction tier in `LRUPromptCache` (with sliding-window `max_size` trim in `BatchRotatingKVCache.extract`). On tail-mutated 6400-token prompts (`bench/prefix_checkpoint_test.py`): **`6144/6400 (96.0%)` prefix reuse**, TTFT **`3.768s -> 0.255s (14.76x)`**, `32/32` tokens bit-exact.
- **MessageGate Decode & Prefill:** Default-ON bit-exact vectorized decode kernels in `rows/proj.py` (`bfloat4`/`uint2` loads + lossless BF16 weight cache `LOOPLING_MSG_BF16=1` adapted from oMLX v0.8.4) and `rows/rowlocal.py` (`k3b_norm` `TG=288` + `gpar=3` msg split), plus opt-in `simdgroup_matrix` MMA tiled prefill (`LOOPLING_SPLASH_MSGGATE=1`, `prefill.py`: `1.53x` isolated QKV at `L=2048`).
- **Opt-In INT8 Full-Attention KV Cache (`LOOPLING_KV_INT8=1`):** Symmetric per-token, per-head INT8 quantization for `FullDecoderLayer` (`Int8KVCache` / `BatchInt8KVCache` in `cache.py` + `quantize_kv_b1_int8` / `sdpa_b1_int8` in `rows/sdpa.py`). Cuts full-attention KV memory **`1.94x`** (`545.3 MB -> 281.1 MB` at `L=16384`) with **`64/64` token agreement** at `1K/8K/16K` (`bench/kv_int8_test.py`).

---

## Architecture & Repository Layout

```text
vendor/mlx_lm/
├── __init__.py                     # Sets MLX_MAX_OPS_PER_BUFFER=32 (when LOOPLING_CMDBUF != 0)
├── generate.py                     # generate_step, MTP/DFlash2 hook, BatchGenerator + BatchDecoder (C=1..32) / opt-in batched DFlash2
├── server.py                       # OpenAI HTTP server (C=1 DFlash2 routing, periodic prefix checkpoints, SSD cache)
└── models/
    ├── cache.py                    # Packed/Int8 KVCache, BatchKV B=1 flags, LRUPromptCache checkpoint & SSD tiers
    └── inkling2/
        ├── model.py                # TextModel, loop execution, LoopCD hooks, make_cache (INT8 option)
        ├── fastdecode.py           # Lane A compiled + fused B=1 decode path (default ON)
        ├── batchdecode.py          # Bit-exact batched C=1..32 decoder, shared compiled cache, stage pooling, causal verify
        ├── prefill.py              # pf_moe_gmm (FP4/FP8 LUT), pf_moe_short_tiled (B*L=8..63), HAS_MASK/HAS_LENS padded convs, pf_mgin_down_s
        ├── kernels.py              # Prefill relative_attention (uint4 K/V + SWAP_GRID) & depth_attention
        ├── mtp.py                  # DSpark drafter + Markov head + compiled drafter helpers + tree verify (LOOPLING_MTP=1)
        ├── dflash2.py              # DFlash 2 drafter, hysteresis PA, coupled sampling, VerifyBudget, W=16 LookupIndex
        ├── loopcd.py               # Opt-in LoopCD-Hidden, LoopCD-Logits & ThinkingTracker (LOOPLING_LOOPCD)
        └── rows/                   # Custom Metal row-invariant & B=1..32 fused kernels
            ├── glue.py             # Shared exact arithmetic / dequant primitives
            ├── qmv.py / qmv4.py    # 8-bit X-major qmv (MAX_B=32) & 4-bit row-grouped qmv / 8-lane top16 epilogue
            ├── gemv.py             # Multi-row small dense float32 matmuls (B=1..32 slabbed)
            ├── gather.py           # Weights-read-once NVFP4 / MXFP8 MoE expert matvecs
            ├── moe.py              # Fused MoE router+route_tile_rows (`route="tile"` B=9..32), MOE_FP8_LUT, gate_up+SwiGLU, down+combine+conv
            ├── proj.py             # K4 QKV multi-projection, vectorized K9 f3_oproj_gate_sx & MessageGate (`LOOPLING_MSG_BF16=1`)
            ├── mid.py              # K5 mid attention-pre kernel (ping-pong vbuf, vectorized norms/convs/RoPE/rel bias)
            ├── rowlocal.py         # Vectorized K1 pre / K3b norm / K10 post kernels (`gpar=3` msg split, B=1..32)
            ├── sdpa.py             # Zero-copy B=1..32 staged SDPA, TK=2/4 1-pass & 2-pass fast_clean, Split-K >=32K, single-pass _CAUSAL_2PFUSED_SRC
            ├── kvstore.py          # Staged / paged RowKV cache store
            ├── head.py             # Softcap + logprobs / argmax / Sparse-Nucleus top-p Gumbel-max row kernels
            └── visit.py            # Fused visit orchestrators
bench/                              # Verification suites, microbenchmarks, LoopCD eval, server/cache tests
docs/                               # Kernel design notes, profiling reports, exactness contracts
scripts/r                           # Remote Studio sync/run wrapper (holds per-host gpu.lock)
```

---

## Quick Start

Put `vendor/` ahead of `PYTHONPATH` (and always `import mlx_lm` before `import mlx.core` in custom scripts so command-buffer batching applies):

```bash
export PYTHONPATH="$PWD/vendor:$PWD"
export HF_HUB_OFFLINE=1
```

### Start the OpenAI-Compatible Server

```bash
LOOPLING_LOOPCD=hidden:0.3 \
python -m mlx_lm server \
  --model /path/to/loopling-p3 \
  --host 0.0.0.0 \
  --port 8080 \
  --decode-concurrency 8 \
  --prompt-concurrency 2 \
  --prefill-step-size 2048 \
  --prompt-cache-size 10
```

Query the endpoint:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/loopling-p3",
    "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
    "max_tokens": 128,
    "temperature": 0.0
  }'
```

### Run Verification & Benchmarks

```bash
# Fastdecode, live weight mutability, and BatchGenerator B=1/B=2 parity
python bench/a2_tests.py all

# Fused MoE kernels across all 24 layers (NVFP4 + MXFP8)
python bench/fused_moe_test.py

# LoopCD (hidden, logits & exclude_thinking) exactness across generate_step, prefill, and BatchDecoder
python bench/loopcd_test.py

# Causal tree speculative verification exactness
python bench/spec_exact_test.py

# Batched C=1..32 decode throughput & bit-exactness
python bench/batch_dec.py --passes 3 --C 1,2,4,8,16,32 --tokens 64

# Serving BatchGenerator (C=1..32 + dynamic join/leave + cache extract/resume)
python bench/server_batch_test.py --C 1,2,4,8,16,32

# C=1 server DFlash2 multi-turn cache sync, system-head checkpoint, SSD round-trip & coupled temp>0
LOOPLING_MTP=dflash2 python bench/server_dflash2_cache_test.py

# Periodic prefix checkpointing & cache eviction tier test
python bench/prefix_checkpoint_test.py

# Opt-in INT8 full-attention KV cache parity & memory test
python bench/kv_int8_test.py

# Long-context Split-K full-attention parity & E2E benchmark (32K..100K)
python bench/long_sdpa_test.py

# Coupled non-greedy DFlash2 parity & throughput benchmark
python bench/coupled_sample_test.py

# Batched DFlash2 verify & multi-stream C=2..8 sweep (exactness & throughput)
python bench/multi_verify_test.py
python bench/dflash2_batch_bench.py

# Full C=1 speculative decode benchmark (DSpark MTP or DFlash 2)
LOOPLING_MTP=1 python bench/spec_bench.py --full
python bench/dflash2_bench.py --model /path/to/loopling-p3 --tokens 256 --reps 2
```

---

## Environment Variables

| Variable | Default | Description |
|---|---:|---|
| `LOOPLING_COMPILE_DECODE` | `1` | `1`: compiled fused `B=1, L=1` decode path; `2`: uncompiled segmented; `0`: reference path |
| `LOOPLING_FUSED_CONV` | `1` | `1`: Metal kernel for decode `ShortConvolution` (`kernel_size=4`, `bf16`); `0`: reference conv |
| `LOOPLING_CMDBUF` | `1` | Sets `MLX_MAX_OPS_PER_BUFFER=32` on `import mlx_lm` before `mlx.core` |
| `LOOPLING_MSG_BF16` | `1` | `1`: lossless BF16 copy cache of FP32 `MessageGate` weights (identity-guarded, bit-exact) |
| `LOOPLING_SERVER_BATCHDECODE` | `1` | `1`: route `GenerationBatch`/`BatchGenerator` `C=1..32` through bit-exact `BatchDecoder` |
| `LOOPLING_LONG_SDPA` | `1` | `1`: enable Split-K (`SPLIT_K=256`) full-attention kernel for `S >= LOOPLING_LONG_SDPA_MIN_S` |
| `LOOPLING_LONG_SDPA_MIN_S` | `32768` | Minimum sequence length threshold for Split-K full-attention dispatch |
| `LOOPLING_PREFIX_CHECKPOINT` | `2048` | Stride in tokens for periodic `(KV + ArraysCache)` prefix checkpoints (`0` disables) |
| `LOOPLING_KV_INT8` | `0` | `1`: opt-in symmetric INT8 quantization for full-attention KV caches (~`1.94x` smaller KV memory) |
| `LOOPLING_SPLASH_MSGGATE` | `0` | `1`: opt-in tiled `simdgroup_matrix` MMA kernel for QKV `MessageGate` during prefill |
| `LOOPLING_MTP` | `0` | `1`: DSpark/Markov exact speculative decode; `dflash2`: DFlash 2 block-diffusion drafter |
| `LOOPLING_MTP_W` | `auto` | DSpark speculative tree verify width (`auto` or `2..8`) |
| `LOOPLING_DFLASH2_DIR` | `~/loopling/models/Inkling-2-Mini-DFlash2` | Path to local DFlash 2 drafter weights (or `incoai/Inkling-2-Mini-DFlash2` HF repo) |
| `LOOPLING_DFLASH2_QUANT` | `4` | DFlash 2 weight quantization bits (`4`, `8`, or `16`) |
| `LOOPLING_DFLASH2_W` | `6` | DFlash 2 chain/verify width (`VerifyBudget` dynamically selects `1..8` when tree enabled) |
| `LOOPLING_DFLASH2_BUDGET` | `1` | `1`: Splash `VerifyBudget` + `MoEStepCosts` width/AR-burst controller; `0`: fixed chain |
| `LOOPLING_DFLASH2_LOOKUP` | `1` | `1`: merge bounded suffix-automaton `LookupIndex` matches into DFlash 2 hybrid verify tree |
| `LOOPLING_DFLASH2_LOOKUP_W` | `16` | Prompt-copy / suffix-lookup verify chain width (`8..16`) |
| `LOOPLING_DFLASH2_PIPE_AHEAD` | `1` | `1`: overlap DFlash 2 draft with target KV verify via pipe-ahead execution (`C=1`) |
| `LOOPLING_DFLASH2_COUPLED_SAMPLE` | `1` | `1`: lossless GPU Sparse-Nucleus top-p Gumbel-max + Leviathan coupled draft/verify (`temp > 0`) |
| `LOOPLING_DFLASH2_BATCH` | `0` | `1`: opt-in multi-user batched DFlash 2 inside `BatchGenerator` (`C=2..LOOPLING_DFLASH2_BATCH_MAX_C`) |
| `LOOPLING_DFLASH2_BATCH_MAX_C` | `4` | Max concurrency for opt-in batched DFlash 2 (`1..8`; above this `BatchGenerator` uses AR) |
| `LOOPLING_DFLASH2_BATCH_W` | `auto` | Override batched DFlash 2 block width (`2..8`; default `3` for `C<=2`, `2` for `C>=3`) |
| `LOOPLING_LOOPCD` | `off` | Opt-in LoopCD: `hidden:0.3`, `logits:0.3`, `logits_adaptive:1.0`, or `off` |
| `LOOPLING_LOOPCD_BLEND_BS` | `0` | `1`: also extrapolate core block-residual attention states in `hidden` mode |
| `LOOPLING_LOOPCD_EXCLUDE_THINKING` | `0` | `1`: skip LoopCD contrastive subtraction inside thinking delimiter spans via `ThinkingTracker` |
| `LOOPLING_LOOPCD_REF` | `0` | Reference loop pass index for contrastive subtraction (`0` = pass 1, or negative index) |

---

## Acknowledgments

- Built on vendored [mlx-lm](https://github.com/ml-explore/mlx-examples) (MIT License, Apple Inc.) and [MLX](https://github.com/ml-explore/mlx).
- Loop Contrastive Decoding implemented from *Decoding Looped Transformers Better for (Almost) Free* ([arXiv:2610.02185](https://arxiv.org/abs/2610.02185)).
- DFlash 2 (`incoai/Inkling-2-Mini-DFlash2`), `VerifyBudget`/`MoEStepCosts`/`LookupProposals`, MessageGate tiling, and KVCache checkpointing/INT8 designs adapted from [incoai/splash](https://github.com/incoai/splash).
