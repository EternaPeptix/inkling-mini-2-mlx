import os, collections
import mlx.nn as nn
from mlx_lm import load
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
shapes = collections.defaultdict(list)
for name, m in model.named_modules():
    if isinstance(m, nn.QuantizedLinear):
        if m.bits == 8 and m.mode == "affine" and "layers" in name or "lm_head" in name:
            N, Kp = m.weight.shape
            K = Kp * 32 // m.bits
            shapes[(N, K, m.group_size, m.bits, m.mode)].append(name)
for k, v in sorted(shapes.items()):
    print(k, len(v), v[:4])
