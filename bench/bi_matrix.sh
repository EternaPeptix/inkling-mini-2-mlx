#!/bin/zsh
# lane D: exactness matrix for LOOPLING_BATCH_INVARIANT (fast path off/on x configs)
for F in 0 1; do
  for cfg in "-T 4 --passes 3" "-T 8 --passes 5 --window 48 --ctx 200" "-T 5 --passes 3 --ctx 3000" "-T 2 --passes 1" "-T 3 --passes 3 --window 64 --ctx 500"; do
    echo "FAST=$F $cfg"
    LOOPLING_COMPILE_DECODE=$F LOOPLING_FUSED_CONV=$F LOOPLING_BATCH_INVARIANT=1 python bench/bi_compare.py ${=cfg} 2>&1 | grep -v transformers | cut -c1-230
  done
done
