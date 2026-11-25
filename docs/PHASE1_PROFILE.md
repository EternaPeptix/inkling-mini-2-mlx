# Phase 1 Profile (`512S2`, Apple M3 Ultra 80-core GPU, `loopling-p3`)

## 1. Decode Throughput Baselines (`bench/batch_dec.py`, `--passes 3 --tokens 128`, all `exact=True`)
Single-stream reference `generate_step`: `60.5–60.8 tok/s`.

| `LOOPLING_BD_FUSED` / flags | C=1 (tok/s) | C=2 (agg) | C=4 (agg) | C=8 (agg) | C=8 ms/step |
|---|---|---|---|---|---|
| `""` (`qmv,gemv,sdpa,gather`) | 64.1 | 96.9 | 156.2 | 228.2 | 35.06 |
| `"pre,k1"` | 67.0 | 106.2 | 167.1 | **237.5** | **33.69** |
| `"pre,k1,post"` (`ROUTE=kernel`) | **75.2** | **110.2** | 151.2 | 194.3 | 41.17 |
| `"pre,k1,post"` (`ROUTE=launch`) | 74.4 | 109.1 | 151.4 | 198.8 | 40.25 |
| `"pre,k1,post,kv"` (`ROWKV_DOUBLE=1`) | 74.7 | — | — | 200.7 | 39.86 |
| `"pre,k1,kv"` (`ROWKV_DOUBLE=1`) | 67.1 | — | — | 232.1 | 34.47 |
| `"pre,k1,kv"` (`ROWKV_DOUBLE=0`) | — | — | — | 216.4 | 36.97 |

## 2. Long-Context `C=8` & Buffer Donation (`--tokens 64`, `p3`)
`bench/donate_probe.py`:
- 64 MB store: single `183 us/step` vs double `195 us/step`
- 512 MB store: single `1530 us/step` (silent copy!) vs double `208 us/step`

| Context | Mode | Agg tok/s | ms/step | Peak GB | Notes |
|---|---|---|---|---|---|
| `ctx=2048` | `pre,k1` | 241.5 | 33.13 | 20.87 | per-row `KVCache` / `RotatingKVCache` |
| `ctx=2048` | `pre,k1,kv` (`ROWKV_DOUBLE=1`) | 239.2 | 33.44 | 23.98 | +3.11 GB full-store double-buffer |
| `ctx=2048` | `pre,k1,kv` (`ROWKV_DOUBLE=0`) | 217.1 | 36.85 | 22.69 | silent copies on scatter |
| `ctx=8192` | `pre,k1` | 232.7 | 34.38 | 22.60 | |
| `ctx=8192` | `pre,k1,kv` (`ROWKV_DOUBLE=1`) | 230.3 | 34.74 | 28.81 | +6.21 GB full-store double-buffer |
| `ctx=8192` | `pre,k1,kv` (`ROWKV_DOUBLE=0`) | 192.8 | 41.48 | 25.91 | -6.74 ms/step from un-donated KV copies |

## 3. Decode Micro-Timings (`B=8`, `p3`)
- **MoE on real distinct-prompt visits (`runs/visits_p3`, queued 16):**
  - Layer 0 (`nvfp4`, 31/64 unique experts): `gather(gu+swiglu+dn)` = **279.6 us** vs `moe(kernel)` = **361.4 us** vs `moe(launch)` = **316.7 us**
  - Layer 7 (`nvfp4`, 17/64 unique experts): `gather(gu+swiglu+dn)` = **238.1 us** vs `moe(kernel)` = **322.4 us** vs `moe(launch)` = **268.8 us**
  - Layer 20 (`mxfp8`, 37/64 unique experts): `gather(gu+swiglu+dn)` = **511.3 us** vs `moe(kernel)` = **539.2 us** vs `moe(launch)` = **542.6 us**
- **`qmv` / `proj` (`bench/qmv_b_speed.py`, B=8):**
  - `qkv_rows` sliding: `57.6 us` queued / `72.0 us` chain
  - `qkv_rows` full: `61.6 us` queued / `76.7 us` chain
  - `qmv_rows o` (plain): **`23.6 us`** queued vs `oproj_rows` (gated prologue): **`46.4 us`** queued (+1.1 ms/step!)
  - `lm_head` (201024 x 2304): `cfg=4,2,4` = **`1.446 ms`** -> `cfg=4,4,8` = **`0.789 ms`** / `4,2,8` = **`0.790 ms`** (-0.66 ms/step!)
- **Tail & loop overhead (`B=8`):**
  - Tail `softcap + 8x logsumexp + argmax`: `0.224 ms/step` queued (plus launch chain overhead)
  - `message_input` (12 calls/pass x 2 passes): `0.217 ms/pass` (`0.434 ms/step`)

## 4. Prefill Baselines & Op Breakdown (`p3`)
- Step-size sweep:
  - `L=2048`: `step=512` 1461.2 tok/s, `step=1024` 1597.1 tok/s, `step=2048` **1611.2 tok/s**
  - `L=16384`: `step=512` 1365.9 tok/s, `step=1024` 1516.5 tok/s, `step=2048` 1563.4 tok/s, `step=4096` **1571.7 tok/s**
- `C=8` prefill (`8 x 2048`):
  - Sequential: **1667.6 agg tok/s** (`9825.1 ms`), exact.
  - Batched (`B=8`): `1491.4 agg tok/s` (`10985.3 ms`), **NOT exact** (differs in K, V, conv).
- Synchronized 48-visit op sum (`L=2048` chunk):
  - At `ctx=0` (sum `1476.9 ms`): `moe_gate_up` 426.2 ms (28.9%), `moe_down` 222.6 ms (15.1%), `qkv_projs` 193.2 ms (13.1%), `o_proj_gate` 84.4 ms (5.7%), `rel_attn` 81.6 ms (5.5%), `msg_gates_qkv` 79.8 ms (5.4%), `moe_combine` 57.0 ms (3.9%), `kv_sconv` 49.2 ms (3.3%), `attend_pre_clips_rel` 41.0 ms (2.8%), `post_attn_conv`+`post_mlp_conv` 52.6 ms (3.6%), `moe_sort`+`moe_unsort` 41.1 ms (2.8%), `msg_gate_mlp` 23.9 ms (1.6%).
  - At `ctx=14336` (sum `1654.5 ms`): `rel_attn` jumps to **244.7 ms (14.8%)** (`21.3 ms/visit` per full-attention layer!), plus `kv_update` 18.7 ms (1.1%).
