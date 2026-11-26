import numpy as np
import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm.models.inkling2.rows import sdpa as RS

mx.random.seed(7)
Hq, Hkv, D, R = 24, 8, 128, 128
scale = D ** -0.5

for C, W in [(2, 4), (2, 8), (4, 2), (4, 4), (8, 2)]:
    B = C * W
    scap = 32
    nv = 4
    lidx = 1
    for mode_name, end_ms, win in [
        ("short_full", [120 + 17 * c for c in range(C)], 0),
        ("long_full", [1100 + 83 * c for c in range(C)], 0),
        ("mixed_full", [150 if c % 2 == 0 else 1200 + 30 * c for c in range(C)], 0),
        ("ring_short", [200 + 15 * c for c in range(C)], 1024),
        ("ring_wrap", [1024 + 90 * c for c in range(C)], 1024),
        ("mixed_ring", [300 if c % 2 == 0 else 1300 for c in range(C)], 1024),
    ]:
        st_lens = [3 + (c % 5) for c in range(C)]
        ks_main, vs_main = [], []
        kv_main_packed_refs = []
        stage = mx.random.normal((nv, C, 2, Hkv, scap, D)).astype(mx.bfloat16)
        q = mx.random.normal((B, Hq, 1, D)).astype(mx.bfloat16)
        kv_new_multi = mx.random.normal((2, Hkv, B, D)).astype(mx.bfloat16)
        pb = mx.random.normal((B, Hq, R)).astype(mx.bfloat16)
        qkv = mx.random.normal((B, Hq * D + 64)).astype(mx.bfloat16)

        mt_rows = []
        Ss = []
        for c in range(C):
            em = end_ms[c]
            stl = st_lens[c]
            max_s = win if win > 0 else em + 64
            cap = win if (win > 0 and em >= win) else em + 64
            rot = 1 if win > 0 else 0
            rot_w = 1 if (rot and em >= win) else 0
            slot_m = ((17 * c + 5) % win) if rot_w else em
            k_c = mx.random.normal((1, Hkv, cap, D)).astype(mx.bfloat16)
            v_c = mx.random.normal((1, Hkv, cap, D)).astype(mx.bfloat16)
            ks_main.append(k_c)
            vs_main.append(v_c)
            mt_rows.append((em, slot_m, rot, rot_w, max_s, cap, cap, stl))
            base_c = em + stl
            for d in range(W):
                Ss.append(min(base_c + d + 1, win) if rot else (base_c + d + 1))

        mta = mx.array(np.array(mt_rows, np.int32))
        mx.eval(stage, q, kv_new_multi, pb, qkv, mta, *ks_main, *vs_main)

        out = RS.sdpa_multi_causal_w_staged(
            q, ks_main, vs_main, stage, kv_new_multi, pb, qkv,
            0, qkv.shape[1], Hq * D, qkv.shape[1], mta, tuple(Ss), C, W,
            scale=scale, window=win, kv_layer_idx=lidx, scap=scap, Hq=Hq, Hkv=Hkv,
        )
        mx.eval(out)

        # Compare bitwise against single-stream sdpa_causal_w_packed per stream c
        mm_total = 0
        for c in range(C):
            em, slot_m, rot, rot_w, max_s, cap, _, stl = mt_rows[c]
            # Reconstruct exact single-stream KV buffer by writing stage tokens on top of main KV
            km_buf = ks_main[c][0]
            vm_buf = vs_main[c][0]
            buf_c = mx.stack([km_buf, vm_buf], axis=0)
            if stl > 0:
                kst = stage[lidx, c, 0, :, :stl, :]
                vst = stage[lidx, c, 1, :, :stl, :]
                for s in range(stl):
                    pos = ((slot_m + s) % max_s) if rot_w else (slot_m + s)
                    buf_c = mx.slice_update(buf_c, mx.stack([kst[:, s : s + 1, :], vst[:, s : s + 1, :]], axis=0), mx.array([pos], mx.int32), axes=(2,))
            end0_c = em + stl
            S0_c = min(end0_c, win) if rot else end0_c
            slot0_c = (((slot_m + stl) % win) if rot_w else end0_c) if rot else end0_c
            qc = q[c * W : (c + 1) * W]
            kvnc = kv_new_multi[:, :, c * W : (c + 1) * W, :]
            pbc = pb[c * W : (c + 1) * W]
            qkvc = qkv[c * W : (c + 1) * W]
            ref_c = RS.sdpa_causal_w_packed(
                qc, buf_c, kvnc, pbc, qkvc, 0, Hq * D, S0_c, end0_c, slot0_c, rot, cap, max_s if rot else 0, W,
                scale=scale, window=win, Hq=Hq, Hkv=Hkv, gate_stride=qkvc.shape[1], sink_stride=qkvc.shape[1],
            )
            mx.eval(ref_c)
            got_c = out[c * W : (c + 1) * W]
            mm_c = int(mx.sum(got_c.reshape(-1).view(mx.uint16) != ref_c.reshape(-1).view(mx.uint16)).item())
            mm_total += mm_c
        assert mm_total == 0, f"Mismatch for C={C} W={W} {mode_name}: mm={mm_total}"
    print(f"PASS C={C} W={W}: 0 bitwise mismatches across all 6 KV/stage regimes!")
