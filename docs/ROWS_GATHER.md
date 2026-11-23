# rows/gather.py: exact weights-read-once MoE expert matvecs

`gather_rows(x, w, scales, biases, idx, *, mode, group_size, bits, global_scale=None)` -> (B, E, N) bf16.
x is (B, K) (gate_up: one row shared by a token's experts) or (B, E, K) (down: one row per pair). idx is (B, E) uint32.
Modes: nvfp4 (layers 0..19, gs16, global_scale), mxfp8 (layers 20..23, gs32). No biases exist on these modules.

## Design
* MLX's own `fp_quantized.h` is expanded from the wheel and passed as `header=` (cut before `qouter`, the steel GEMM part needs
  the .metal wrappers). `load_vector`, `dequantize_scale` and the same per-thread partition (fast: 2 packs/thread, 4 rows per
  simdgroup, 2 simdgroups, 8 rows per threadgroup; non-fast variant selected by the same N%8 / K-alignment test) are used, and
  the same `simd_sum` + `* inv_scale_enc` epilogue. Dequant (`dequant_vals`) and dot (`dot_vals`) are small local copies that
  mirror MLX's `qdot` operation for operation; weights are dequantized once and dotted against each token's x.
* Grouping with no host sync and no extra primitive: one threadgroup per (token, expert) pair (grid z = B*E). Each threadgroup
  scans idx (<=256 pairs) in threadgroup memory; the first pair that picked expert e is the owner, later duplicates return
  immediately. The owner collects all pairs with that expert, streams the weight tile once and keeps per-pair partial sums in
  registers up to `TMAX` pairs per pass (`ROWS_GATHER_TMAX_CAP`, default 2; more pairs than TMAX just re-read the tile).
  Results are stored bf16 after the epilogue scalar, so a swiglu / routing-weight epilogue can be added at that point.
* At B=1 all 8 experts are distinct, TMAX=1, and it is the same work as MLX's gather_qmv.
* TMAX_CAP trade-off (L0/L20, B=8 queued, us gate_up/down): cap1 187/129, 312/177; cap2 176/127, 239/141; cap4 201/183, 233/149;
  cap16 about 445/390 (register spills). Default 2.

## Exactness
`bench/rows_gather_test.py`: real weights from loopling-p3, all 24 layers (20 nvfp4, 4 mxfp8), both projections, B in
{1,2,3,5,8,16}, random inputs and real captured activations, routing random / all-same / all-distinct / partial overlap /
duplicates inside a token. Bit patterns per (token, expert) vs `gather_qmm` for that token alone (B=1 decode shapes).

| scope | cases | elements | mismatches |
|---|---|---|---|
| layers 0..23, gate_up + down, all B and routings | 1728 | 185,794,560 | 0 |

Log: host `~/loopling-opt/runs/e2/gather_test_all.txt`. No inexact case known.

## Speed (median us over 60 reps, 16 launches queued then one eval; GPU lock held; M3 Ultra 512S1)
ratio = ref/rows (>1 means rows is faster).

| layer | proj | B | case | ref us | rows us | ratio |
|---|---|---|---|---|---|---|
| L0 nvfp4 | gate_up | 1 | queued | 50.5 | 43.1 | 1.17 |
| L0 nvfp4 | gate_up | 1 | dependent chain | 53.7 | 45.0 | 1.19 |
| L0 nvfp4 | gate_up | 8 | vs 8 ref calls (38 uniq experts) | 443.0 | 174.5 | 2.54 |
| L0 nvfp4 | gate_up | 8 | vs one sorted 64-pair call (incl sort+unsort) | 198.1 | 174.5 | 1.14 |
| L0 nvfp4 | gate_up | 8 | dependent chain vs 8 ref | 343.5 | 202.0 | 1.70 |
| L0 nvfp4 | down | 1 | queued | 45.1 | 39.0 | 1.16 |
| L0 nvfp4 | down | 8 | vs 8 ref calls | 381.8 | 125.1 | 3.05 |
| L0 nvfp4 | down | 8 | vs sorted (gather_qmm only) | 195.8 | 125.1 | 1.57 |
| L20 mxfp8 | gate_up | 1 | queued | 59.0 | 56.0 | 1.05 |
| L20 mxfp8 | gate_up | 1 | dependent chain | 66.9 | 65.3 | 1.02 |
| L20 mxfp8 | gate_up | 8 | vs 8 ref calls (24 uniq) | 521.7 | 238.7 | 2.19 |
| L20 mxfp8 | gate_up | 8 | vs sorted (incl sort+unsort) | 304.9 | 238.7 | 1.28 |
| L20 mxfp8 | gate_up | 8 | dependent chain vs 8 ref | 456.7 | 251.2 | 1.82 |
| L20 mxfp8 | down | 1 | queued | 50.1 | 47.0 | 1.07 |
| L20 mxfp8 | down | 8 | vs 8 ref calls | 409.0 | 143.2 | 2.86 |
| L20 mxfp8 | down | 8 | vs sorted (gather_qmm only) | 151.9 | 143.2 | 1.06 |

B=8 uses 8 real captured tokens routed by the real router. Log: `~/loopling-opt/runs/e2/gather_speed_final.txt`.
Not wired into the decode path.
