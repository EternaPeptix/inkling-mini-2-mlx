# rows/sdpa.py: decode attention for B rows with their own KV caches

`vendor/mlx_lm/models/inkling2/rows/sdpa.py`, `sdpa_rows(q, ks, vs, pb, sinks, meta, *, scale, window)`.
Not wired into the decode path. Output (B, 1, Hq, D) bf16, row b bitwise equal to `fastdecode._core` on that row alone
(`_core` returns the transposed (1, 1, Hq, D); the layouts coincide).

## Design
* Reference arithmetic (MLX 0.32.4, arch `applegpu_g15d`, bf16, D=128, GQA 3, float mask, sinks): key length < 1024 runs
  `sdpa_vector` (key i to simdgroup i % 32, online softmax per simdgroup, max/sum/rescale through threadgroup memory);
  >= 1024 runs `sdpa_vector_2pass_1`/`_2` with 128 blocks (n_simds = 3, so the block count rule gives 128), key i to
  block i % 128, 4-way prefetch loop plus tail loop, bf16 partials, f32 sums/maxs, then the reduce pass. The kernel text
  copies those device functions statement by statement (same types, same `simd_sum` and reduction order, `fma`
  kept where MLX has it). Only the work mapping differs, and each (row, head, simdgroup/block) is independent.
* One launch (stage 1) handles every row, each in its own regime (a row with S=300, a wrapped ring, a 70000 full cache
  in the same call). A second launch (stage 2, reduce) runs only if some row has S >= 1024 and passes 1-pass rows through.
* The mask is built in-kernel: slot i has position ((i - idx) mod S if rotating else i) + end - S; distance = end-1-pos;
  bias = pb[clip(distance, 0, R-1)]; visible = distance >= 0 and (window == 0 or distance < window) and distance < R for
  the bias, else 0 / -inf; cast to q dtype exactly as `_core` does (pb or sinks as f32 also verified).
* K/V are bound as the caches' own arrays (no copy): head stride = buffer capacity * D, passed in `meta`
  (`row_meta(kv)` gives `(end, idx, rot, kcap, vcap)`). Metal has 31 buffer slots, so at most 11 rows per kernel; larger
  B is chunked automatically. Kernels are cached per B.
* Stage-1 threadgroup width: 32 simdgroups if any short row is present, otherwise GQA simdgroups (MLX's own geometry,
  1024 groups of 96 threads). With 1024-thread groups the B=1, S=16384 case was 12% slower than `_core` (too few groups
  to balance 80 cores); the narrow geometry fixed it.
* Limits (asserted): bf16 q, D=128, query length 1, 3 <= GQA <= 5 (block count rule implemented for blocks=128, which is
  the case for the model, GQA 3). Other block counts (n_simds >= 6 or N > 8192 with n_simds <= 2) are not implemented.

## Exactness (`bench/rows_sdpa_test.py`, real KVCache / RotatingKVCache(1024) via their own update_and_fetch)
52 cases, 0 mismatching elements in all.

| Group | Cases | Mismatches |
|---|---|---|
| full attention S = 1, 2, 300, 1023, 1024, 1025, 1500, 5000, 20000, 70000 | 10 | 0 |
| sliding: S=300, 1023, full ring idx 1, wrapped ring idx 37 / 300 / 976 / 1023 / 1024, long prefill (70000) | 10 | 0 |
| mixed regimes, B = 2, 3, 5, 8 (3 random draws each, full and sliding) | 24 | 0 |
| explicit mixes (300 + ring + 5000; 70000 + 1 + 20000 + 1023; ring variants) | 3 | 0 |
| B = 13, 16 (chunked over 11-row launches) | 2 | 0 |
| window 700 on full caches (window narrower than cache) | 1 | 0 |
| pb and sinks in f32 | 2 | 0 |

## Speed (`bench/rows_sdpa_speed.py`, median of 200 reps under the GPU lock, 512S2; cache update excluded from both sides)
Reference = B x `_core` (mask build + sdpa), evaluated in one `mx.eval`.

| Case | B | `_core` us | rows us | speedup |
|---|---|---|---|---|
| full S=512 | 1 | 228 | 221 | 1.03x |
| full S=512 | 8 | 355 | 255 | 1.39x |
| sliding S=1024 wrapped | 1 | 232 | 218 | 1.06x |
| sliding S=1024 wrapped | 8 | 439 | 290 | 1.52x |
| full S=1024 | 1 | 235 | 226 | 1.04x |
| full S=1024 | 8 | 439 | 291 | 1.51x |
| full S=4096 | 1 | 246 | 239 | 1.03x |
| full S=4096 | 8 | 647 | 425 | 1.52x |
| full S=16384 | 1 | 302 | 304 | 1.00x (tie within noise) |
| full S=16384 | 8 | 1294 | 1038 | 1.25x |

About 200 us of every row is eval/dispatch latency of the host loop, so the kernel-only gain is larger than shown.
