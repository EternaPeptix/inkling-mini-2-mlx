# qmv-family speed at larger B (lane g1)

Kernel tile `rows_qmv_tile<B,K,PR,RPS,NSG,RB,STG>` (qmv.py, shared by proj.py): grid z splits the B rows into
groups of RB rows per threadgroup (weights re-read from L2/SLC), weights of a K block are loaded once into
registers and reused for the RB rows, loops fully unrolled. Per-row arithmetic (per-thread K partition,
accumulation order, simd_sum) unchanged: exactness tests 0 mismatches.
Default (RPS,NSG,RB,STG) = (4,2,4,0). `LOOPLING_QMV_CFG="rps,nsg,rb[,stg]"` overrides for tuning
(`bench/qmv_b_speed.py --cfg ...`). STG (threadgroup staging of the K9 gate prologue) was no faster; off.
Tried and rejected: software prefetch of next block (slower), RB=8/16 (register pressure at B>=8), RPS=8, NSG=4.
