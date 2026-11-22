#!/bin/zsh
# lane D: official exact gate for selfspec under LOOPLING_BATCH_INVARIANT (256 tokens, 3 prompts, passes 1,3,5)
mkdir -p ~/loopling-opt/runs/d
for cfg in "1:3" "1:4"; do
  for F in 0 1; do
    out=~/loopling-opt/runs/d/gate_${cfg/:/_}_f$F.json
    echo "== selfspec $cfg FAST=$F"
    LOOPLING_SELFSPEC=$cfg LOOPLING_BATCH_INVARIANT=1 LOOPLING_COMPILE_DECODE=$F LOOPLING_FUSED_CONV=$F python bench/bench.py --passes 1,3,5 --reps 1 --prefill "" --out $out 2>&1 | grep -v transformers
    python bench/exact.py ~/loopling-opt/runs/baseline.json $out
  done
done
