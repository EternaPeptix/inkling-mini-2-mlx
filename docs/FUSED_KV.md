# FUSED_KV: RowKV batched KV store + SDPA (lane f5, kernels K7 + K8)

`vendor/mlx_lm/models/inkling2/rows/kvstore.py`. Not wired into the decode path.

## Interface
```python
store = RowKV(caches)                 # B KVCache (all) or B RotatingKVCache(1024, keep=0) (all); same step
p = store.plan()                      # host: advance every row's state by one token; uploads 2 tiny arrays
store.update(k, v, p)                 # k, v (B, Hkv, D) bf16 -> 2 scatter launches (K, V), in place
o = store.sdpa(q, pb, sinks, p, scale=..., window=...)   # q (B,Hq,1,D) -> (B,1,Hq,D); 1 (<1024 keys) or 2 launches
store.flush(i) / store.to_caches()    # write row state back into the original cache objects
```
`p` may be shared by every store whose rows are in the same cache state (all full visits share one plan, all sliding visits
another): 2 plans per token. Without a plan, `update` plans for itself and `sdpa` reuses the last plan.

## Layout
K, V: flat `(B*Hkv*Cap, D)` bf16; row b, head h, slot s at `(b*Hkv+h)*Cap+s`. Each row keeps its cache's exact physical slot
layout (same slot numbers; ring position kept). Per row table (int32, device): `[S, end, idx, rot]`; masks are built in
the kernel (rows/sdpa.py arithmetic, so bitwise equal to `_core`). The 11-row binding limit is gone (one K, one V buffer).
Nothing per-step is a template parameter or in the kernel source, so a whole visit can be `mx.compile`d; host-side plan
arrays are inputs. Growth past Cap (full rows only, every `step` tokens by the logical capacity rule of KVCache) reallocates.

## In-place update
`K[widx] = k` is MLX Scatter; `Scatter::eval_gpu` -> `copy_gpu` returns early when the input is donatable
(`use_count==1`), so the buffer is reused: update is 2 launches, ~35-70 us for 2 chained, independent of Cap
(8 to 495 MB per array measured). Caveat measured: if you `mx.eval` after EVERY update, the previous command buffer's
completion handler can still hold the buffer and the scatter silently copies the whole store (peak memory +1 store).
A decode step chains many visits in one graph (and chained async_eval keeps inputs unevaluated), where donation holds.
Do not hold other references to `store.K` / `store.V` across `update`.

## Memory
Cap = max logical capacity over rows, so one 70k row inflates every row's slab of that visit. Group rows by length if needed.

## Exactness (bench/fused_kv_test.py, 512S1)
T=48 decode steps, real KVCache/RotatingKVCache, prompts 10..5000 (incl. chunk-prefilled, 1010/1023/1024 so the ring wraps
and the 2-pass regime starts mid-test, 4200/5000 long rows), bf16 attention output compared every step plus flushed caches
(keys/values buffers incl. shape, offset, _idx) at the end: **0 mismatches** for full and sliding, B in {1,2,3,5,8,13},
f32 pb, all-rows>=1024 (narrow geometry), shared plan across two stores.

## Speed (bench/fused_kv_speed.py; us per visit, update+attention, visits chained)
See table in the commit message / run the script; B=8 RowKV is 1.2-1.7x faster than per-row update_and_fetch + sdpa_rows
for S<=2000 and equal at S=8000 (bandwidth bound); B=1 equals sdpa_rows (+/-10 us for plan upload).
