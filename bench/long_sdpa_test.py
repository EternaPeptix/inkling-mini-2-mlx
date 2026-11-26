import time
import numpy as np
import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm.models.inkling2.rows import sdpa as RS
from mlx_lm.models.inkling2.batchdecode import build_tree_tables

mx.random.seed(0)
Hq, Hkv, D, R = 24, 8, 128, 128
scale = D ** -0.5

for S in (32776, 65536, 100000):
    cap = S + 32
    kv_buf = mx.random.normal((2, Hkv, cap, D)).astype(mx.bfloat16)
    pb_b1 = mx.random.normal((1, Hq, R)).astype(mx.bfloat16)
    qkv_b1 = mx.random.normal((1, Hq * D + 64)).astype(mx.bfloat16)
    q_b1 = mx.random.normal((1, Hq, 1, D)).astype(mx.bfloat16)

    W = 8
    S0 = S - W
    q_w = mx.random.normal((W, Hq, 1, D)).astype(mx.bfloat16)
    kv_new = mx.random.normal((2, Hkv, W, D)).astype(mx.bfloat16)
    pb_w = mx.random.normal((W, Hq, R)).astype(mx.bfloat16)
    qkv_w = mx.random.normal((W, Hq * D + 64)).astype(mx.bfloat16)
    # Build matching kv_buf with kv_new copied into [S0 : S0 + W] so B=1 == causal_w bit-for-bit
    kv_full_b1 = mx.concatenate([kv_buf[:, :, :S0, :], kv_new, kv_buf[:, :, S0 + W :, :]], axis=2)
    mx.eval(kv_buf, pb_b1, qkv_b1, q_b1, q_w, kv_new, pb_w, qkv_w, kv_full_b1)

    qkv_stride = qkv_w.shape[1]
    # 1. Accuracy vs LOOPLING_LONG_SDPA=0 and bitwise row-exactness
    RS._LONG_SDPA = False
    ref_b1 = RS.sdpa_b1_packed(q_b1, kv_full_b1, pb_b1, qkv_b1, 0, Hq * D, S, S, 0, 0, cap, scale=scale, window=0, Hq=Hq, Hkv=Hkv)
    ref_w8 = RS.sdpa_causal_w_packed(q_w, kv_buf, kv_new, pb_w, qkv_w, 0, Hq * D, S0, S0, 0, 0, cap, cap, W, scale=scale, window=0, Hq=Hq, Hkv=Hkv, gate_stride=qkv_stride, sink_stride=qkv_stride)

    RS._LONG_SDPA = True
    out_b1 = RS.sdpa_b1_packed(q_b1, kv_full_b1, pb_b1, qkv_b1, 0, Hq * D, S, S, 0, 0, cap, scale=scale, window=0, Hq=Hq, Hkv=Hkv)
    out_w8 = RS.sdpa_causal_w_packed(q_w, kv_buf, kv_new, pb_w, qkv_w, 0, Hq * D, S0, S0, 0, 0, cap, cap, W, scale=scale, window=0, Hq=Hq, Hkv=Hkv, gate_stride=qkv_stride, sink_stride=qkv_stride)
    mx.eval(ref_b1, ref_w8, out_b1, out_w8)

    b1_diff = float(mx.max(mx.abs(ref_b1.astype(mx.float32) - out_b1.astype(mx.float32))).item())
    w8_diff = float(mx.max(mx.abs(ref_w8.astype(mx.float32) - out_w8.astype(mx.float32))).item())

    # Check B=1 (causal_w W=1 when S0 < 32768 else sdpa_b1_packed) at S0+b+1 == causal_w row b BITWISE!
    row_mm = 0
    for b in range(W):
        sb = S0 + b + 1
        qb = q_w[b : b + 1]
        pbb = pb_w[b : b + 1]
        qkvb = qkv_w[b : b + 1]
        if sb >= 32768:
            ob_b1 = RS.sdpa_b1_packed(qb, kv_full_b1, pbb, qkvb, 0, Hq * D, sb, sb, 0, 0, cap, scale=scale, window=0, Hq=Hq, Hkv=Hkv)
            mx.eval(ob_b1)
            row_mm += int(mx.sum(ob_b1.reshape(-1).view(mx.uint16) != out_w8[b].reshape(-1).view(mx.uint16)).item())

    # Check tree topology row-exactness + HAS_STAGE=True
    parents = (-1, 0, 0, 1, 2, 1, 3, 4)
    depths = (0, 1, 1, 2, 2, 2, 3, 3)
    d_t, _, tm_tree = build_tree_tables(parents, depths)
    out_tree = RS.sdpa_causal_w_packed(
        q_w, kv_buf, kv_new, pb_w, qkv_w, 0, Hq * D, S0, S0, 0, 0, cap, cap, W,
        scale=scale, window=0, tree_meta=tm_tree, depths=d_t, Hq=Hq, Hkv=Hkv, gate_stride=qkv_stride, sink_stride=qkv_stride,
    )
    mx.eval(out_tree)
    # Row 6 has path [0, 1, 3, 6]
    kv_path6 = mx.concatenate([kv_buf[:, :, :S0, :], kv_new[:, :, mx.array([0, 1, 3, 6], mx.int32), :], kv_buf[:, :, S0 + 4 :, :]], axis=2)
    ob_tree6 = RS.sdpa_b1_packed(q_w[6:7], kv_path6, pb_w[6:7], qkv_w[6:7], 0, Hq * D, S0 + 4, S0 + 4, 0, 0, cap, scale=scale, window=0, Hq=Hq, Hkv=Hkv)
    mx.eval(ob_tree6)
    tree_mm = int(mx.sum(ob_tree6.reshape(-1).view(mx.uint16) != out_tree[6].reshape(-1).view(mx.uint16)).item())

    # Check HAS_STAGE=True
    st_len = 16
    slot_main = S0 - st_len
    kv_buf5 = kv_buf.reshape(1, 2, Hkv, cap, D)
    kv_st5 = kv_buf5[:, :, :, slot_main : slot_main + st_len, :] * 1.0
    kv_buf5_zero_tail = mx.slice_update(kv_buf5, mx.zeros((1, 2, Hkv, st_len, D), mx.bfloat16), mx.array([slot_main], mx.int32), axes=(3,))
    meta_st = mx.array([S0, S0, 0, cap, 0, 0, slot_main, st_len], mx.int32)
    mx.eval(kv_st5, kv_buf5_zero_tail, meta_st)
    out_st = RS.sdpa_causal_w_packed(
        q_w, kv_buf5_zero_tail, kv_new, pb_w, qkv_w, 0, Hq * D, S0, S0, 0, 0, cap, cap, W,
        scale=scale, window=0, depths=tuple(range(W)), meta_arr=meta_st, s_bounds=(S0 + 1, S0 + W), kv_stage=kv_st5, Hq=Hq, Hkv=Hkv,
        gate_stride=qkv_stride, sink_stride=qkv_stride,
    )
    mx.eval(out_st)
    stage_mm = int(mx.sum(out_st.reshape(-1).view(mx.uint16) != out_w8.reshape(-1).view(mx.uint16)).item())

    print(f"S={S:6d}: max_abs_diff(b1={b1_diff:.5f}, w8={w8_diff:.5f}) | bitwise_mm(b1==w8:{row_mm}, tree:{tree_mm}, stage:{stage_mm})")

    # 2. Speed benchmark A/B
    kv_bytes = 2 * Hkv * S * D * 2
    for flag in (False, True):
        RS._LONG_SDPA = flag
        lbl = "LONG=1" if flag else "LONG=0"
        for _ in range(5):
            mx.eval(RS.sdpa_b1_packed(q_b1, kv_full_b1, pb_b1, qkv_b1, 0, Hq * D, S, S, 0, 0, cap, scale=scale, window=0, Hq=Hq, Hkv=Hkv))
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            mx.eval(RS.sdpa_b1_packed(q_b1, kv_full_b1, pb_b1, qkv_b1, 0, Hq * D, S, S, 0, 0, cap, scale=scale, window=0, Hq=Hq, Hkv=Hkv))
            ts.append((time.perf_counter() - t0) * 1000)
        ms = float(np.median(ts))
        print(f"  {lbl} B=1: {ms:.3f}ms ({kv_bytes/(ms*1e6):.1f} GB/s)")

        for Wc in (1, 4, 8):
            qw = mx.random.normal((Wc, Hq, 1, D)).astype(mx.bfloat16)
            kvn = mx.random.normal((2, Hkv, Wc, D)).astype(mx.bfloat16)
            pbw = mx.random.normal((Wc, Hq, R)).astype(mx.bfloat16)
            qkvw = mx.random.normal((Wc, Hq * D + 64)).astype(mx.bfloat16)
            mx.eval(qw, kvn, pbw, qkvw)
            qs = qkvw.shape[1]
            for _ in range(5):
                mx.eval(RS.sdpa_causal_w_packed(qw, kv_buf, kvn, pbw, qkvw, 0, Hq * D, S - Wc, S - Wc, 0, 0, cap, cap, Wc, scale=scale, window=0, Hq=Hq, Hkv=Hkv, gate_stride=qs, sink_stride=qs))
            ts = []
            for _ in range(30):
                t0 = time.perf_counter()
                mx.eval(RS.sdpa_causal_w_packed(qw, kv_buf, kvn, pbw, qkvw, 0, Hq * D, S - Wc, S - Wc, 0, 0, cap, cap, Wc, scale=scale, window=0, Hq=Hq, Hkv=Hkv, gate_stride=qs, sink_stride=qs))
                ts.append((time.perf_counter() - t0) * 1000)
            ms = float(np.median(ts))
            print(f"  {lbl} W={Wc}: {ms:.3f}ms ({Wc*kv_bytes/(ms*1e6):.1f} GB/s eff, {kv_bytes/(ms*1e6):.1f} GB/s unique)")
