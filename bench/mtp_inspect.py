import mlx.core as mx
w = mx.load("/Users/jeweled/loopling/models/official-mtp/mtp.safetensors")
for k in sorted(w):
    if ".layers.1." in k or ".layers.2." in k:
        continue
    print(k, w[k].shape, w[k].dtype)
k = [k for k in w if k.endswith("fuse_weight")][0]
mx.set_printoptions(precision=3)
print(mx.softmax(w[k].astype(mx.float32), axis=-1))
print(w[k])
