#!/bin/zsh
for F in 0 1; do
  echo "== FAST=$F timing p3"; LOOPLING_COMPILE_DECODE=$F LOOPLING_FUSED_CONV=$F LOOPLING_BATCH_INVARIANT=1 python bench/bi_time.py --passes 3 --Ts 2,3,4,5 2>&1 | grep -v transformers
done
for F in 0 1; do
  echo "== FAST=$F selfspec"; LOOPLING_COMPILE_DECODE=$F LOOPLING_FUSED_CONV=$F LOOPLING_BATCH_INVARIANT=1 python bench/selfspec_bench.py --targets 3,5 --drafts 1 --ks 3,4 --tokens 128 --reps 1 --out /tmp/ss_d_$F.json 2>&1 | grep -v transformers
done
