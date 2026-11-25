# loopling-opt

Bit-exact MLX inference and serving optimizations for **TML-d24-varloop-research ("Loopling" / Inkling2 looped MoE)** on Apple Silicon (`mlx 0.32.4`, vendored `vendor/mlx_lm`).

All default optimizations — including fused `B=1` fastdecode, MessageGate decode vectorization, `BatchDecoder` `C=2..8` batching in `BatchGenerator`/`server.py`, and periodic prefix checkpointing — are **100% bit-exact** against the reference greedy decode path on every prompt and loop pass count while preserving live weight mutability (`setattr`, `module.update`, `load_weights`, adapters). Speculative decoding (`LOOPLING_MTP=1` / `dflash2`), Loop Contrastive Decoding (`LOOPLING_LOOPCD`), INT8 full-attention KV cache (`LOOPLING_KV_INT8=1`), and tiled prefill MessageGate (`LOOPLING_SPLASH_MSGGATE=1`) are opt-in via environment variables.

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
| `C=1` | 81.57 tok/s | 80.33 tok/s | 0.98x | 0 |
| `C=2` | 56.40 tok/s | **145.09 tok/s** | **2.57x** | **0** (vs 81 unpatched) |
| `C=4` | 97.06 tok/s | **211.26 tok/s** | **2.18x** | **0** (vs 95 unpatched) |
| `C=8` | 158.27 tok/s | **305.30 tok/s** | **1.93x** | **0** (vs 95 unpatched) |

Dynamic continuous batching (`C=4 -> 6 -> 3 -> 1`) + multi-turn cache extract/resume also pass with **0 mismatches**.

### 2. Single-Stream Exact Speculative Decode (`LOOPLING_MTP`, 0 mismatches)

- **DSpark MTP (`LOOPLING_MTP=1`, `bench/spec_bench.py --full`):** 3-layer DSpark drafter + Markov head (`vendor/mlx_lm/models/inkling2/mtp.py`) and fused causal tree verification (`W = 2..8`, auto width selection):
  - **`loopling-p1`:** `239.4–241.5 tok/s` mean over `P0..P7` (`257.7–260.4 tok/s` mean over `P0..P2`, peaking at `322.3 tok/s` on `P2`).
- **DFlash 2 Block-Diffusion Drafter (`LOOPLING_MTP=dflash2`, `bench/dflash2_bench.py`, `loopling-p3`, 256 greedy tokens):** 5-layer non-causal Qwen3 drafter (`incoai/Inkling-2-Mini-DFlash2`, Q4 default) + Splash `VerifyBudget` online logistic regression + 2-point `MoEStepCosts` sunk-cost width selection + deferred-context AR bursts + bounded suffix-automaton `LookupIndex` hybrid verify tree (`vendor/mlx_lm/models/inkling2/dflash2.py`):

| Mode (`loopling-p3`) | `MEAN8` (`P0..P7`) | `MEAN9` (`P0..P8` + code) | Code Refactor (`P8`) | Tokens/cycle | Mismatches |
|---|---:|---:|---:|---:|---:|
| Autoregressive baseline | 97.91 tok/s | 97.71 tok/s | 96.12 tok/s | 1.00 | 0 |
| DSpark MTP (`LOOPLING_MTP=1`) | 109.40 tok/s (1.12x) | 117.44 tok/s | — | 2.15 | 0 |
| DFlash2 Q4 fixed chain (`W=6`) | 112.32 tok/s (1.15x) | 121.93 tok/s | — | 3.05 | 0 |
| **DFlash2 Q4 `VerifyBudget` + Tree + Lookup (`W=8`)** | **117.38 tok/s (1.20x)** | **128.53 tok/s (1.32x)** | **217.72 tok/s (2.27x)** | **3.43** | **0** |

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

- **Bit-Exact Server Logprob & `BatchDecoder` Serving:** Fixed unconditional per-token GPU logprob synchronization in `vendor/mlx_lm/server.py` (`_token_logprob`), wired `sdpa_rows(..., gate=gate)` + `qmv_rows` for `B=1` `BatchKVCache`/`BatchRotatingKVCache`, and integrated `BatchDecoder` (`C=2..8`) directly into `GenerationBatch`.
- **Bit-Exact Periodic Prefix Checkpointing (`LOOPLING_PREFIX_CHECKPOINT=2048`):** Snapshots `(KV + ArraysCache)` states every 2048 tokens during long prefills/decodes into a dedicated lowest-priority `"checkpoint"` eviction tier in `LRUPromptCache` (with sliding-window `max_size` trim in `BatchRotatingKVCache.extract`). On tail-mutated 6400-token prompts (`bench/prefix_checkpoint_test.py`): **`6144/6400 (96.0%)` prefix reuse**, TTFT **`3.768s -> 0.255s (14.76x)`**, `32/32` tokens bit-exact.
- **MessageGate Decode & Prefill:** Default-ON bit-exact vectorized decode kernels in `rows/proj.py` (`bfloat4`/`uint2` loads) and `rows/rowlocal.py` (`k3b_norm` `TG=288` + vectorized norm), plus opt-in `simdgroup_matrix` MMA tiled prefill (`LOOPLING_SPLASH_MSGGATE=1`, `prefill.py`: `1.53x` isolated QKV at `L=2048`, `+0.92–0.98%` end-to-end prefill).
- **Opt-In INT8 Full-Attention KV Cache (`LOOPLING_KV_INT8=1`):** Symmetric per-token, per-head INT8 quantization for `FullDecoderLayer` (`Int8KVCache` / `BatchInt8KVCache` in `cache.py` + `quantize_kv_b1_int8` / `sdpa_b1_int8` in `rows/sdpa.py`). Cuts full-attention KV memory **`1.94x`** (`545.3 MB -> 281.1 MB` at `L=16384`) with **`64/64` token agreement** at `1K/8K/16K` (`bench/kv_int8_test.py`).

---

## Architecture & Repository Layout

```text
vendor/mlx_lm/
├── __init__.py                     # Sets MLX_MAX_OPS_PER_BUFFER=32 (when LOOPLING_CMDBUF != 0)
├── generate.py                     # generate_step, MTP/DFlash2 hook, BatchGenerator + BatchDecoder C=2..8
├── server.py                       # OpenAI HTTP server (async logprob sync fix, periodic prefix checkpoints)
└── models/
    ├── cache.py                    # Packed/Int8 KVCache, BatchKV B=1 flags, LRUPromptCache checkpoint tier
    └── inkling2/
        ├── model.py                # TextModel, loop execution, LoopCD hooks, make_cache (INT8 option)
        ├── fastdecode.py           # Lane A compiled + fused B=1 decode path (default ON)
        ├── batchdecode.py          # Bit-exact batched C=1..8 decoder & causal tree verification
        ├── prefill.py              # Fused prefill path + opt-in MessageGate MMA (LOOPLING_SPLASH_MSGGATE=1)
        ├── mtp.py                  # DSpark drafter + Markov head + tree verification (LOOPLING_MTP=1)
        ├── dflash2.py              # DFlash 2 drafter, VerifyBudget, MoEStepCosts, LookupIndex (LOOPLING_MTP=dflash2)
        ├── loopcd.py               # Opt-in LoopCD-Hidden and LoopCD-Logits (LOOPLING_LOOPCD)
        └── rows/                   # Custom Metal row-invariant & B=1..8 fused kernels
            ├── glue.py             # Shared exact arithmetic / dequant primitives
            ├── qmv.py / qmv4.py    # 8-bit affine & 4-bit row-grouped qmv / argmax / tree-walk
            ├── gemv.py             # Multi-row small dense float32 matmuls
            ├── gather.py           # Weights-read-once NVFP4 / MXFP8 MoE expert matvecs
            ├── moe.py              # Fused MoE router, gate_up+SwiGLU, down+combine+conv+residual
            ├── proj.py             # K4 QKV multi-projection & K9 O-proj+gate kernels (vectorized MessageGate)
            ├── mid.py              # K5 mid attention-pre kernel (norms, convs, RoPE, rel bias)
            ├── rowlocal.py         # K1 pre / K3b norm (`TG=288` vectorized) / K10 post kernels
            ├── sdpa.py             # Row-wise exact SDPA, packed/INT8 B=1 SDPA, staged KV SDPA, causal tree SDPA
            ├── kvstore.py          # Staged / paged RowKV cache store
            ├── head.py             # Softcap + logprobs / argmax row kernels
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

# Periodic prefix checkpointing & cache eviction tier test
python bench/prefix_checkpoint_test.py

# Opt-in INT8 full-attention KV cache parity & memory test
python bench/kv_int8_test.py

# Full C=1 speculative decode benchmark (DSpark MTP or DFlash 2)
LOOPLING_MTP=1 python bench/spec_bench.py --full
python bench/dflash2_bench.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---:|---|
| `LOOPLING_COMPILE_DECODE` | `1` | `1`: compiled fused `B=1, L=1` decode path; `2`: uncompiled segmented; `0`: reference path |
| `LOOPLING_FUSED_CONV` | `1` | `1`: Metal kernel for decode `ShortConvolution` (`kernel_size=4`, `bf16`); `0`: reference conv |
| `LOOPLING_CMDBUF` | `1` | Sets `MLX_MAX_OPS_PER_BUFFER=32` on `import mlx_lm` before `mlx.core` |
| `LOOPLING_SERVER_BATCHDECODE` | `1` | `1`: route `GenerationBatch`/`BatchGenerator` `C=2..8` through bit-exact `BatchDecoder` |
| `LOOPLING_PREFIX_CHECKPOINT` | `2048` | Stride in tokens for periodic `(KV + ArraysCache)` prefix checkpoints (`0` disables) |
| `LOOPLING_KV_INT8` | `0` | `1`: opt-in symmetric INT8 quantization for full-attention KV caches (~`1.94x` smaller KV memory) |
| `LOOPLING_SPLASH_MSGGATE` | `0` | `1`: opt-in tiled `simdgroup_matrix` MMA kernel for QKV `MessageGate` during prefill |
| `LOOPLING_MTP` | `0` | `1`: DSpark/Markov exact speculative decode; `dflash2`: DFlash 2 block-diffusion drafter |
| `LOOPLING_MTP_W` | `auto` | DSpark speculative tree verify width (`auto` or `2..8`) |
| `LOOPLING_DFLASH2_DIR` | `~/loopling/models/Inkling-2-Mini-DFlash2` | Path to local DFlash 2 drafter weights (or `incoai/Inkling-2-Mini-DFlash2` HF repo) |
| `LOOPLING_DFLASH2_QUANT` | `4` | DFlash 2 weight quantization bits (`4` or `16`) |
| `LOOPLING_DFLASH2_W` | `8` | DFlash 2 max verify width (`VerifyBudget` dynamically selects `1..W`) |
| `LOOPLING_DFLASH2_BUDGET` | `1` | `1`: Splash `VerifyBudget` + `MoEStepCosts` width/AR-burst controller; `0`: fixed chain |
| `LOOPLING_DFLASH2_LOOKUP` | `1` | `1`: merge bounded suffix-automaton `LookupIndex` matches into DFlash 2 hybrid verify tree |
| `LOOPLING_LOOPCD` | `off` | Opt-in LoopCD: `hidden:0.3`, `logits:0.3`, `logits_adaptive:1.0`, or `off` |
| `LOOPLING_LOOPCD_BLEND_BS` | `0` | `1`: also extrapolate core block-residual attention states in `hidden` mode |
| `LOOPLING_LOOPCD_REF` | `0` | Reference loop pass index for contrastive subtraction (`0` = pass 1, or negative index) |

---

## Acknowledgments

- Built on vendored [mlx-lm](https://github.com/ml-explore/mlx-examples) (MIT License, Apple Inc.) and [MLX](https://github.com/ml-explore/mlx).
- Loop Contrastive Decoding implemented from *Decoding Looped Transformers Better for (Almost) Free* ([arXiv:2610.02185](https://arxiv.org/abs/2610.02185)).
- DFlash 2 (`incoai/Inkling-2-Mini-DFlash2`), `VerifyBudget`/`MoEStepCosts`/`LookupProposals`, MessageGate tiling, and KVCache checkpointing/INT8 designs adapted from [incoai/splash](https://github.com/incoai/splash).
