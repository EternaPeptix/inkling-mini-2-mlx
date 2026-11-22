"""Lane D micro-tests: which formulation of a T-row op is bitwise equal to T separate 1-row calls."""
import os
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load
model, tok = load(os.path.expanduser("~/loopling/models/loopling-p1")); mx.eval(model.parameters())
T = 4
def same(a, b): return bool(mx.array_equal(a, b).item())
mods = {}
for n, m in model.named_modules():
    if isinstance(m, (nn.QuantizedLinear, nn.Linear, nn.Conv1d)):
        key = type(m).__name__ + str(tuple(m.weight.shape))
        mods.setdefault(key, (n, m))
print("distinct modules", len(mods))
mx.random.seed(0)
for key, (n, m) in mods.items():
    if isinstance(m, nn.Conv1d):
        K, C = m.weight.shape[1], m.weight.shape[0]
        win = (mx.random.normal((1, K - 1 + T, C)) * 2).astype(m.weight.dtype)
        ref = mx.concatenate([m(win[:, j:j + K]) for j in range(T)], axis=1)
        v = {"batchB": m(mx.concatenate([win[:, j:j + K] for j in range(T)], 0)).reshape(1, T, C), "whole": m(win)}
        w = m.weight.astype(mx.float32)[:, :, 0]
        acc = None
        for t in range(K):
            term = win[:, t:t + T].astype(mx.float32) * w[:, t]
            acc = term if acc is None else acc + term
        v["manual_f32_seq"] = acc.astype(win.dtype)
        print(n, key, {k: same(x, ref) for k, x in v.items()}, flush=True)
        continue
    quant = isinstance(m, nn.QuantizedLinear)
    K = m.weight.shape[1] * 32 // m.bits if quant else m.weight.shape[1]
    dt = mx.float32 if (not quant and m.weight.dtype == mx.float32) else mx.bfloat16
    x = mx.random.normal((1, T, K)).astype(dt)
    ref = mx.concatenate([m(x[:, j:j + 1]) for j in range(T)], axis=1); mx.eval(ref)
    v = {"whole": m(x)}
    if quant:
        w, s, b = m["weight"], m["scales"], m.get("biases")
        kw = dict(transpose=True, group_size=m.group_size, bits=m.bits, mode=m.mode)
        v["batchx"] = mx.quantized_matmul(x.reshape(T, 1, K), w, s, b, **kw).reshape(1, T, -1)
        bc = lambda a: None if a is None else mx.broadcast_to(a[None], (T,) + a.shape)
        v["bcast_w"] = mx.quantized_matmul(x.reshape(T, 1, K), bc(w), bc(s), bc(b), **kw).reshape(1, T, -1)
        z = mx.zeros((T,), mx.uint32)
        v["gather_qmm"] = mx.gather_qmm(x.reshape(T, 1, 1, K), w[None], s[None], None if b is None else b[None], rhs_indices=z, **kw).reshape(1, T, -1)
    else:
        wt = m["weight"].T
        v["batchx"] = (x.reshape(T, 1, K) @ wt).reshape(1, T, -1)
        v["bcast_w"] = (x.reshape(T, 1, K) @ mx.broadcast_to(wt[None], (T,) + wt.shape)).reshape(1, T, -1)
    print(n, key, getattr(m, "mode", ""), getattr(m, "bits", ""), {k: same(x_, ref) for k, x_ in v.items()}, flush=True)
