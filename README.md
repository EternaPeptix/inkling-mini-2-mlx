# loopling-opt

Bit-exact MLX inference and serving optimizations for **TML-d24-varloop-research ("Loopling" / Inkling2 looped MoE)** on Apple Silicon (`mlx 0.32.4`, vendored `vendor/mlx_lm`).

All default optimizations are **100% bit-exact** against the reference greedy decode path on every prompt and loop pass count while preserving live weight mutability (`setattr`, `module.update`, `load_weights`, adapters). Speculative decoding (`LOOPLING_MTP`) and Loop Contrastive Decoding (`LOOPLING_LOOPCD`) are opt-in via environment variables.

## Highlights & Verified Performance

Measured on **Apple M3 Ultra (80-core GPU, 512 GB unified memory)**:

### 1. Bit-Exact Single & Batched Decode (`bench/batch_dec.py`, 64 tokens, 0 mismatches)

| Model (`num_passes`) | `C=1` (tok/s) | `C=2` agg | `C=4` agg | `C=6` agg | `C=8` agg | `C=8` ms/step |
|---|---:|---:|---:|---:|---:|---:|
| **`loopling-p1`** (`p1`) | 204.7 | 304.3 | 433.1 | 544.0 | **631.7** | 12.66 ms |
| **`loopling-p3`** (`p3`) | 100.4 | 150.6 | 219.3 | 278.5 | **313.7** | 25.50 ms |

### 2. Single-Stream Exact Speculative Decode (`LOOPLING_MTP=1`, `bench/spec_bench.py --full`, 0 mismatches)

Ports the 3-layer DSpark drafter + Markov head (`vendor/mlx_lm/models/inkling2/mtp.py`) and fused causal tree verification (`W = 2..8`, auto width selection) with **0 mismatches**:

- **`loopling-p1`:** `239.4–241.5 tok/s` mean over `P0..P7` (`257.7–260.4 tok/s` mean over `P0..P2`, peaking at `322.3 tok/s` on `P2`; `275.7 tok/s` at `ctx=240`, `262.1 tok/s` at `ctx=1020`).
- **`loopling-p3`:** `110.2 tok/s` mean over `P0..P2`.

### 3. LoopCD Quality Boost (`LOOPLING_LOOPCD`, `loopling-p3`)

Implements **Loop Contrastive Decoding** ([arXiv:2610.02185](https://arxiv.org/abs/2610.02185)) across recurrent loop passes in [`vendor/mlx_lm/models/inkling2/loopcd.py`](vendor/mlx_lm/models/inkling2/loopcd.py):

| Config (`loopling-p3`) | Compute Overhead | Gen (`n=40`) | Code (`n=12`) | `mean_rep4` | OpenDecisions Acc / NLL (`n=300`) |
|---|---|---:|---:|---:|---:|
| `off` (default, bit-exact) | 1.000x | 36/40 (90.0%) | 75.0% | 0.1388 | 49.3% / 1.345 |
| **`hidden:0.3`** | **1.000x (0 extra FLOPs)** | **37/40 (92.5%)** | **83.3%** | **0.1320** | 49.7% / 1.309 |
| `hidden:0.75` | 1.000x | 37/40 (92.5%) | 75.0% | 0.1251 | 52.0% / 1.272 |
| `hidden:1.0` | 1.000x | — | — | — | **54.0% (+4.7 pts)** / 1.273 |
| `logits_adaptive:1.0` | +1 coda/`lm_head` | **38/40 (95.0%)** | 83.3% | 0.1357 | 50.0% / 1.381 |

### 4. OpenAI-Compatible HTTP Server (`mlx_lm server`)

- Fixed unconditional per-token GPU logprob synchronization in `vendor/mlx_lm/server.py` (`_token_logprob`) and wired `sdpa_rows(..., gate=gate)` + `qmv_rows` for `B=1` `BatchKVCache` / `BatchRotatingKVCache` in `fastdecode.py` / `cache.py`.
- **`loopling-p3` + `LOOPLING_LOOPCD=hidden:0.3`** live over HTTP (`max_tokens=128`, `temperature=0.0`): **`82.1 tok/s`** streaming decode (`ttft=0.181s`), `73.9–74.1 tok/s` end-to-end.

---

## Architecture & Repository Layout

```text
vendor/mlx_lm/
├── __init__.py                     # Sets MLX_MAX_OPS_PER_BUFFER=32 (when LOOPLING_CMDBUF != 0)
├── generate.py                     # generate_step / MTP speculative hook / BatchGenerator
├── server.py                       # OpenAI HTTP server (async logprob sync fix)
└── models/
    ├── cache.py                    # Packed KVCache/RotatingKVCache + BatchKV B=1 fastdecode flags
    └── inkling2/
        ├── model.py                # TextModel, loop execution, LoopCD hooks, decode_step_logprobs
        ├── fastdecode.py           # Lane A compiled + fused B=1 decode path (default ON)
        ├── batchdecode.py          # Bit-exact batched C=1..8 decoder & causal tree verification
        ├── prefill.py              # Fused prefill path
        ├── mtp.py                  # DSpark drafter + Markov head + tree verification (LOOPLING_MTP=1)
        ├── loopcd.py               # Opt-in LoopCD-Hidden and LoopCD-Logits (LOOPLING_LOOPCD)
        └── rows/                   # Custom Metal row-invariant & B=1..8 fused kernels
            ├── glue.py             # Shared exact arithmetic / dequant primitives
            ├── qmv.py / qmv4.py    # 8-bit affine & 4-bit row-grouped qmv / argmax / tree-walk
            ├── gemv.py             # Multi-row small dense float32 matmuls
            ├── gather.py           # Weights-read-once NVFP4 / MXFP8 MoE expert matvecs
            ├── moe.py              # Fused MoE router, gate_up+SwiGLU, down+combine+conv+residual
            ├── proj.py             # K4 QKV multi-projection & K9 O-proj+gate kernels
            ├── mid.py              # K5 mid attention-pre kernel (norms, convs, RoPE, rel bias)
            ├── rowlocal.py         # K1 pre / K3b norm / K10 post kernels
            ├── sdpa.py             # Row-wise exact SDPA, packed B=1 SDPA, staged KV SDPA, causal tree SDPA
            ├── kvstore.py          # Staged / paged RowKV cache store
            ├── head.py             # Softcap + logprobs / argmax row kernels
            └── visit.py            # Fused visit orchestrators
bench/                              # Verification suites, microbenchmarks, LoopCD eval, server tests
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

# Full C=1 speculative decode benchmark (requires mtp weights alongside model)
LOOPLING_MTP=1 python bench/spec_bench.py --full
```

---

## Environment Variables

| Variable | Default | Description |
|---|---:|---|
| `LOOPLING_COMPILE_DECODE` | `1` | `1`: compiled fused `B=1, L=1` decode path; `2`: uncompiled segmented; `0`: reference path |
| `LOOPLING_FUSED_CONV` | `1` | `1`: Metal kernel for decode `ShortConvolution` (`kernel_size=4`, `bf16`); `0`: reference conv |
| `LOOPLING_CMDBUF` | `1` | Sets `MLX_MAX_OPS_PER_BUFFER=32` on `import mlx_lm` before `mlx.core` |
| `LOOPLING_MTP` | `0` | `1`: enable exact DSpark/Markov multi-token speculative decoding (`generate_step`) |
| `LOOPLING_MTP_W` | `auto` | Speculative tree verify width (`auto` or `2..8`) |
| `LOOPLING_LOOPCD` | `off` | Opt-in LoopCD: `hidden:0.3`, `logits:0.3`, `logits_adaptive:1.0`, or `off` |
| `LOOPLING_LOOPCD_BLEND_BS` | `0` | `1`: also extrapolate core block-residual attention states in `hidden` mode |
| `LOOPLING_LOOPCD_REF` | `0` | Reference loop pass index for contrastive subtraction (`0` = pass 1, or negative index) |

---

## Acknowledgments

- Built on vendored [mlx-lm](https://github.com/ml-explore/mlx-examples) (MIT License, Apple Inc.) and [MLX](https://github.com/ml-explore/mlx).
- Loop Contrastive Decoding implemented from *Decoding Looped Transformers Better for (Almost) Free* ([arXiv:2610.02185](https://arxiv.org/abs/2610.02185)).
