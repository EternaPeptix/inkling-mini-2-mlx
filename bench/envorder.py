"""Does MLX_MAX_OPS_PER_BUFFER set AFTER `import mlx.core` still take effect? (ops=1 makes a long chain slow)."""
import os, sys, time
if sys.argv[1] == "before":
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = "1"
import mlx.core as mx
if sys.argv[1] == "after":
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = "1"
x = mx.ones((256,))
for _ in range(3):
    y = x
    t = time.perf_counter()
    for i in range(3000):
        y = y * 1.0001 + 0.5
    mx.eval(y)
    dt = time.perf_counter() - t
print(sys.argv[1], "%.2f us/op" % (dt / 3000 * 1e6))
