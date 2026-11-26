# loopling-opt

Bit-exact MLX inference and serving optimizations for **TML-d24-varloop-research ("Loopling" / Inkling2 looped MoE)** on Apple Silicon (`mlx 0.32.4`, vendored `vendor/mlx_lm`).

All default optimizations — including fused `B=1` fastdecode, MessageGate decode vectorization, `BatchDecoder` `C=2..8` batching in `BatchGenerator`/`server.py`, long-context Split-K full attention (`S >= 32K`), and periodic prefix checkpointing — are **100% bit-exact** against the reference greedy decode path on every prompt and loop pass count while preserving live weight mutability (`setattr`, `module.update`, `load_weights`, adapters). Speculative decoding (`LOOPLING_MTP=1` / `dflash2`), Loop Contrastive Decoding (`LOOPLING_LOOPCD`), INT8 full-attention KV cache (`LOOPLING_KV_INT8=1`), multi-user batched DFlash2 (`LOOPLING_DFLASH2_BATCH=1`), and tiled prefill MessageGate (`LOOPLING_SPLASH_MSGGATE=1`) are opt-in via environment variables.

## Highlights & Verified Performance

Measured on **Apple M3 Ultra (80-core GPU, 512 GB unified memory)**:

### 1. Bit-Exact Single & Batched Decode (`C=1..8`, 0 mismatches)

Direct `BatchDecoder` (`bench/batch_dec.py`, 64 tokens, `exact=True`):

| Model (`num_passes`) | `C=1` (tok/s) | `C=2` agg | `C=4` agg | `C=6` agg | `C=8` agg | `C=8` ms/step |
|---|---:|---:|---:|---:|---:|---:|
| **`loopling-p1`** (`p1`) | 204.7 | 304.3 | 433.1 | 544.0 | **631.7** | 12.66 ms |
| **`loopling-p3`** (`p3`) | 100.8 | 150.6 | **219.6** | 278.5 | **314.6** | 25.43 ms |

Serving `BatchGenerator` (`bench/server_batch_test.py`, `loopling-p3`, `LOOPLING_SERVER_BATCHDECODE=1` default ON):

| Concurrency | Unpatched `BatchGenerator` | With `BatchDecoder` (`SERVER_BATCHDECODE=1`) | Speedup | Mismatches vs `C=1` |
|---|---:|---:|---:|---:|
| `C=1` | 99.58 tok/s | **104.03 tok/s** | 1.04x | 0 |
| `C=2` | 58.28 tok/s | **152.19 tok/s** | **2.61x** | **0** (vs 81 unpatched) |
| `C=4` | 101.36 tok/s | **215.54 tok/s** | **2.13x** | **0** (vs 95 unpatched) |
| `C=8` | 165.94 tok/s | **318.95 tok/s** | **1.92x** | **0** (vs 95 unpatched) |

- Dynamic continuous batching (`C=4 -> 6 -> 3 -> 1`) + multi-turn cache extract/resume pass with **0 mismatches** (`dyn_mm=0, res_mm=0`).
- **Opt-in Multi-User Batched DFlash2 (`LOOPLING_DFLASH2_BATCH=1`, default `0`, `C=2..4`):** 100% bit-exact (`0` mismatches across `C=1..8` and dynamic join/leave/resume); delivers **1.08x–1.14x** vs synchronous AR in isolated fixed-step benchmarks (`bench/dflash2_batch_bench.py`: `C=2 W=3` `151.3` vs `139.6`, `C=3 W=2` `193.5` vs `170.1`, `C=4 W=2` `226.7` vs `198.4`), while pipelined `BatchGenerator` AR (`151.54` / `215.57`) remains faster on general mixed prose, so batched DFlash2 is kept opt-in for high-acceptance/lookup workloads.

### 2. Single-Stream Exact Speculative Decode (`LOOPLING_MTP`, 0 mismatches)

- **DSpark MTP (`LOOPLING_MTP=1`, `bench/spec_bench.py --full`):** 3-layer DSpark drafter + Markov head (`vendor/mlx_lm/models/inkling2/mtp.py`) and fused causal tree verification (`W = 2..8`, auto width selection):
  - **`loopling-p1`:** `239.4–241.5 tok/s` mean over `P0..P7` (`257.7–260.4 tok/s` mean over `P0..P2`, peaking at `322.3 tok/s` on `P2`).
- **DFlash 2 Block-Diffusion Drafter (`LOOPLING_MTP=dflash2`, `bench/dflash2_bench.py --tokens 256 --reps 2`, `loopling-p3`):** 5-layer non-causal Qwen3 drafter (`incoai/Inkling-2-Mini-DFlash2`, Q4 default) + pipe-ahead overlap (`LOOPLING_DFLASH2_PIPE_AHEAD=1`) + Splash `VerifyBudget` online logistic regression + 2-point `MoEStepCosts` sunk-cost width selection + `W=16` bounded suffix-automaton `LookupIndex` hybrid verify (`vendor/mlx_lm/models/inkling2/dflash2.py`):

| Mode (`loopling-p3`, 256 greedy tok) | `MEAN8` (`P0..P7`) | `MEAN9` (`P0..P8` + code) | Code Refactor (`P8`) | Tokens/cycle | Mismatches |
|---|---:|---:|---:|---:|---:|
| Autoregressive baseline | 103.21 tok/s | 102.85 tok/s | 99.99 tok/s | 1.00 | 0 |
| DSpark MTP (`LOOPLING_MTP=1`) | 121.99 tok/s (1.18x) | 128.71 tok/s | 182.49 tok/s (1.83x) | 2.25 | 0 |
| DFlash2 Q4 fixed chain (`W=6`) | 127.55 tok/s (1.24x) | 137.99 tok/s | 221.50 tok/s (2.22x) | 3.07 | 0 |
| DFlash2 Q4 `VerifyBudget` + Tree (`W=8`, no lookup) | 130.97 tok/s (1.27x) | 142.11 tok/s | 231.17 tok/s (2.31x) | 3.31 | 0 |
| **DFlash2 Q4 `VerifyBudget` + Tree + `W=16` Lookup (`W=8`)** | **130.67 tok/s (1.27x)** | **146.24 tok/s** | **270.74 tok/s (2.71x)** | **3.66** | **0** |

- **Lossless Non-Greedy Coupled DFlash2 (`LOOPLING_DFLASH2_COUPLED_SAMPLE=1`, `bench/coupled_sample_test.py`):** Fused GPU Sparse-Nucleus Top-P Gumbel-max sampler (`rows/head.py`) + Leviathan coupled draft/verify (`rows/qmv4.py`, `batchdecode.py`): `temp=0.6, top_p=0.95` sampled `MEAN8` improves **`111.51 -> 116.15 tok/s`** (`tpc 2.448 -> 2.824`, parity `TV=0.044`), greedy `0` mismatches.
- **Resident RAM Optimization:** Immediately freeing BF16 DFlash2 buffers after Q4 quantization reduces drafter resident footprint from `+0.929 GB` to `+0.805 GB` (**−124 MB**).

### 3. LoopCD Quality Boost (`LOOPLING_LOOPCD`, `loopling-p3`)

Implements **Loop Contrastive Decoding** ([arXiv:2610.02185](https://arxiv.org/abs/2610.02185)) across recurrent loop passes in [`vendor/mlx_lm/models/inkling2/loopcd.py`](vendor/mlx_lm/models/inkling2/loopcd.py):

| Config (`loopling-p3`) | Compute Overhead | Gen (`n=40`) | Code (`n=12`) | `mean_rep4` | OpenDecisions Acc / NLL (`n=300`) |
|---|---|---:|---:|---:|---:|
| `off` (default, bit-exact) | 1.000x | 36/40 (90.0%) | 75.0% | 0.1388 | 49.3% / 1.345 |
| **`hidden:0.3`** | **1.000x (0 extra FLOPs)** | **37/40 (92.5%)** | **83.3%** | **0.1320** | 49.7% / 1.309 |
| `hidden:0.75` | 1.000x | 37/40 (92.5%) | 75.0% | 0.1251 | 52.0% / 1.272 |
| `hidden:1.0` | 1.000x | — | — | — | **54.0% (+4.7 pts)** / 1.273 |
| `logits_adaptive:1.0` | +1 coda/`lm_head` | **38/40 (95.0%)** | 83.3% | 0.1357 | 50.0% / 1.381 |

### 4. OpenAI HTTP Server, Prefix Checkpointing, MessageGate & KV Options

- **Bit-Exact Server Logprob, DFlash2 `C=1` Routing & `BatchDecoder` Serving:** Fixed unconditional per-token GPU logprob synchronization in `vendor/mlx_lm/server.py` (`_token_logprob`), wired `sdpa_rows(..., gate=gate)` + `qmv_rows` for `B=1` `BatchKVCache`/`BatchRotatingKVCache`, integrated `BatchDecoder` (`C=2..8`) directly into `GenerationBatch`, and added direct `C=1` `LOOPLING_MTP=dflash2` serving with exact multi-turn cache sync, system-head checkpoint protection, and persistent SSD `LRUPromptCache` round-trips (`--persistent-cache`, `--persistent-cache-gb`, verified in `bench/server_dflash2_cache_test.py`).
- **Long-Context Split-K Full Attention (`S >= 32K`, `LOOPLING_LONG_SDPA=1`, `bench/long_sdpa_test.py`):** Split-K P1 (`SPLIT_K=256`, zero-barrier `TK=4`) + P2 (`1024`-thread `bf16` partials) in `rows/sdpa.py` & `batchdecode.py`: `64K` kernel `W=4` `1.979 -> 1.004 ms` / `W=8` `3.494 -> 1.618 ms`; `100K` kernel `W=4` `3.082 -> 1.343 ms` / `W=8` `5.594 -> 2.300 ms`; E2E DFlash2 `32.8K` `201.89 -> 221.43 tok/s` and `65.6K` `43.18 -> 46.51 tok/s` (`mm_vs_AR = 0`).
- **Bit-Exact Periodic Prefix Checkpointing (`LOOPLING_PREFIX_CHECKPOINT=2048`):** Snapshots `(KV + ArraysCache)` states every 2048 tokens during long prefills/decodes into a dedicated lowest-priority `"checkpoint"` eviction tier in `LRUPromptCache` (with sliding-window `max_size` trim in `BatchRotatingKVCache.extract`). On tail-mutated 6400-token prompts (`bench/prefix_checkpoint_test.py`): **`6144/6400 (96.0%)` prefix reuse**, TTFT **`3.768s -> 0.255s (14.76x)`**, `32/32` tokens bit-exact.
- **MessageGate Decode & Prefill:** Default-ON bit-exact vectorized decode kernels in `rows/proj.py` (`bfloat4`/`uint2` loads) and `rows/rowlocal.py` (`k3b_norm` `TG=288` + vectorized norm), plus opt-in `simdgroup_matrix` MMA tiled prefill (`LOOPLING_SPLASH_MSGGATE=1`, `prefill.py`: `1.53x` isolated QKV at `L=2048`, `+0.92–0.98%` end-to-end prefill).
- **Opt-In INT8 Full-Attention KV Cache (`LOOPLING_KV_INT8=1`):** Symmetric per-token, per-head INT8 quantization for `FullDecoderLayer` (`Int8KVCache` / `BatchInt8KVCache` in `cache.py` + `quantize_kv_b1_int8` / `sdpa_b1_int8` in `rows/sdpa.py`). Cuts full-attention KV memory **`1.94x`** (`545.3 MB -> 281.1 MB` at `L=16384`) with **`64/64` token agreement** at `1K/8K/16K` (`bench/kv_int8_test.py`).

---

## Architecture & Repository Layout

```text
vendor/mlx_lm/
├── __init__.py                     # Sets MLX_MAX_OPS_PER_BUFFER=32 (when LOOPLING_CMDBUF != 0)
├── generate.py                     # generate_step, MTP/DFlash2 hook, BatchGenerator + BatchDecoder / opt-in batched DFlash2
├── server.py                       # OpenAI HTTP server (C=1 DFlash2 routing, periodic prefix checkpoints, SSD cache)
└── models/
    ├── cache.py                    # Packed/Int8 KVCache, BatchKV B=1 flags, LRUPromptCache checkpoint & SSD tiers
    └── inkling2/
        ├── model.py                # TextModel, loop execution, LoopCD hooks, make_cache (INT8 option)
        ├── fastdecode.py           # Lane A compiled + fused B=1 decode path (default ON)
        ├── batchdecode.py          # Bit-exact batched C=1..8 decoder, single & multi-stream causal tree verification
        ├── prefill.py              # Fused prefill path + opt-in MessageGate MMA (LOOPLING_SPLASH_MSGGATE=1)
        ├── mtp.py                  # DSpark drafter + Markov head + tree verification (LOOPLING_MTP=1)
        ├── dflash2.py              # DFlash 2 drafter, pipe-ahead, coupled sampling, VerifyBudget, LookupIndex (W=16), batched chains
        ├── loopcd.py               # Opt-in LoopCD-Hidden and LoopCD-Logits (LOOPLING_LOOPCD)
        └── rows/                   # Custom Metal row-invariant & B=1..8 fused kernels
            ├── glue.py             # Shared exact arithmetic / dequant primitives
            ├── qmv.py / qmv4.py    # 8-bit affine & 4-bit row-grouped qmv / argmax / coupled Gumbel / tree-walk
            ├── gemv.py             # Multi-row small dense float32 matmuls
            ├── gather.py           # Weights-read-once NVFP4 / MXFP8 MoE expert matvecs
            ├── moe.py              # Fused MoE router, gate_up+SwiGLU, down+combine+conv+residual
            ├── proj.py             # K4 QKV multi-projection & K9 O-proj+gate kernels (vectorized MessageGate)
            ├── mid.py              # K5 mid attention-pre kernel (norms, convs, RoPE, rel bias)
            ├── rowlocal.py         # K1 pre / K3b norm (`TG=288` vectorized) / K10 post kernels
            ├── sdpa.py             # Row-wise exact SDPA, packed/INT8 B=1 SDPA, Split-K >=32K, staged single/multi causal SDPA
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

# LoopCD (hidden & logits) exactness across generate_step, prefill, and BatchDecoder
python bench/loopcd_test.py

# Causal tree speculative verification exactness
python bench/spec_exact_test.py

# Batched C=1..8 decode throughput & bit-exactness
python bench/batch_dec.py --passes 1,3 --C 1,2,4,6,8 --tokens 64

# Serving BatchGenerator (C=1..8 + continuous batching + cache extract/resume)
python bench/server_batch_test.py

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
| `LOOPLING_SERVER_BATCHDECODE` | `1` | `1`: route `GenerationBatch`/`BatchGenerator` `C=2..8` through bit-exact `BatchDecoder` |
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
| `LOOPLING_LOOPCD_REF` | `0` | Reference loop pass index for contrastive subtraction (`0` = pass 1, or negative index) |

---

## Acknowledgments

- Built on vendored [mlx-lm](https://github.com/ml-explore/mlx-examples) (MIT License, Apple Inc.) and [MLX](https://github.com/ml-explore/mlx).
- Loop Contrastive Decoding implemented from *Decoding Looped Transformers Better for (Almost) Free* ([arXiv:2610.02185](https://arxiv.org/abs/2610.02185)).
- DFlash 2 (`incoai/Inkling-2-Mini-DFlash2`), `VerifyBudget`/`MoEStepCosts`/`LookupProposals`, MessageGate tiling, and KVCache checkpointing/INT8 designs adapted from [incoai/splash](https://github.com/incoai/splash).
