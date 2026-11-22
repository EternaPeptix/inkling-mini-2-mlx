"""Lane A: fused decode short-conv kernel vs reference ShortConvolution, bit-exact over random inputs and real weights."""
import os
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import fastdecode as F
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p1")); mx.eval(model.parameters())
tm = model.model.language_model; bad = 0; n = 0
mx.random.seed(0)
for li in (0, 1, 7, 13, 23):
    layer = tm.layers[li]
    mods = [layer.post_attention_convolution, layer.post_mlp_convolution]
    if hasattr(layer.self_attn.k_sconv, "conv1d"): mods += [layer.self_attn.k_sconv, layer.self_attn.v_sconv]
    for m in mods:
        C = m.conv1d.weight.shape[0]; print(li, type(m).__name__, "C", C, "w", m.conv1d.weight.dtype)
        for t in range(300):
            sc = [1.0, 4.0, 50.0][t % 3]
            x = (mx.random.normal((1, 1, C)) * sc).astype(mx.bfloat16)
            st = (mx.random.normal((1, 3, C)) * sc).astype(mx.bfloat16).astype(mx.float32)
            ro, rw = m(x, st, None); rs = rw[:, 1:, :]
            fo, fs = F.short_conv_decode(m, x, st); mx.eval(ro, rs, fo, fs)
            d1 = int((ro != fo).sum().item()); d2 = int((rs != fs).sum().item()); n += 1
            if d1 or d2: bad += 1; print("MISMATCH", li, t, d1, d2)
print("cases", n, "mismatching", bad)
