"""Prefill profiler & step-size / C=8 batched prefill check (Phase 1).

Usage: LANE=opt RHOST=512S2 zsh scripts/r --gpu 'python bench/prefill_prof.py'
"""
import os
import statistics
import time

import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import layers as L
from mlx_lm.layers.moe import _gather_sort, _scatter_unsort

model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
mx.eval(model.parameters())
tm = model.model.language_model


def make_ids(n, seed=0):
    return [(17 * i * i + 3 * i + 7 * seed) % 40000 + 1000 for i in range(n)]


def bench_prefill_steps():
    print("=== Prefill step-size sweep (p3) ===")
    for total in (2048, 16384):
        ids = make_ids(total)
        for step in (512, 1024, 2048, 4096):
            if step > total:
                continue
            ts = []
            for r in range(4):
                c = make_prompt_cache(model)
                t0 = time.perf_counter()
                for s in range(0, total, step):
                    model(mx.array([ids[s : s + step]]), cache=c)
                    mx.eval([x.state for x in c])
                dt = time.perf_counter() - t0
                if r > 0:
                    ts.append(total / dt)
                del c
                mx.clear_cache()
            print(f"L={total:5d} step={step:4d}: {statistics.median(ts):7.1f} tok/s  runs={[round(x, 1) for x in ts]}", flush=True)


def bench_c8_prefill():
    print("\n=== C=8 prefill (8 x 2048 tokens): sequential vs batched + bit-exact check ===")
    B, Len = 8, 2048
    ids = [make_ids(Len, seed=i) for i in range(B)]

    # Sequential reference
    t_seq = []
    ref_caches = None
    for r in range(3):
        cs = [make_prompt_cache(model) for _ in range(B)]
        t0 = time.perf_counter()
        for i in range(B):
            model(mx.array([ids[i]]), cache=cs[i])
            mx.eval([x.state for x in cs[i]])
        dt = time.perf_counter() - t0
        if r > 0:
            t_seq.append(B * Len / dt)
        if r == 2:
            ref_caches = cs
        else:
            del cs
            mx.clear_cache()
    print(f"Sequential 8x2048: {statistics.median(t_seq):7.1f} agg tok/s ({1e3 * B * Len / statistics.median(t_seq):.1f} ms)", flush=True)

    # Batched B=8
    t_bat = []
    bat_cache = None
    x_bat = mx.array(ids)
    for r in range(3):
        c = make_prompt_cache(model)
        t0 = time.perf_counter()
        model(x_bat, cache=c)
        mx.eval([x.state for x in c])
        dt = time.perf_counter() - t0
        if r > 0:
            t_bat.append(B * Len / dt)
        if r == 2:
            bat_cache = c
        else:
            del c
            mx.clear_cache()
    print(f"Batched    8x2048: {statistics.median(t_bat):7.1f} agg tok/s ({1e3 * B * Len / statistics.median(t_bat):.1f} ms)", flush=True)

    # Bitwise comparison of KV and conv states between sequential and batched
    k_bad = v_bad = c_bad = 0
    k_tot = v_tot = c_tot = 0
    for vi in range(len(ref_caches[0])):
        bk = bat_cache[vi][0].keys
        bv = bat_cache[vi][0].values
        rk = mx.concatenate([ref_caches[i][vi][0].keys for i in range(B)], axis=0)
        rv = mx.concatenate([ref_caches[i][vi][0].values for i in range(B)], axis=0)
        # RotatingKVCache after _update_concat(2048) trims to 1024; compare
        k_tot += rk.size
        v_tot += rv.size
        k_bad += int(mx.sum(bk.view(mx.uint16) != rk.view(mx.uint16)).item())
        v_bad += int(mx.sum(bv.view(mx.uint16) != rv.view(mx.uint16)).item())
        for j in range(4):
            rc = mx.concatenate([ref_caches[i][vi][1][j] for i in range(B)], axis=0)
            bc = bat_cache[vi][1][j]
            c_tot += rc.size
            c_bad += int(mx.sum(bc.view(mx.uint32) != rc.view(mx.uint32)).item())
    print(f"Batched vs sequential bit mismatches: K {k_bad}/{k_tot}, V {v_bad}/{v_tot}, conv {c_bad}/{c_tot}", flush=True)
    del ref_caches, bat_cache
    mx.clear_cache()


def time_op(fn, warm=2, reps=6):
    for _ in range(warm):
        out = fn()
        mx.eval(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def profile_ops_at(ctx_len):
    print(f"\n=== Op-level breakdown of L=2048 prefill chunk (prior ctx={ctx_len}) ===")
    B, seq = 1, 2048
    ids = mx.array([make_ids(seq, seed=9)])
    h = tm.embed(ids)
    mx.eval(h)
    dtype = h.dtype

    # Representative layers:
    # - Sliding nvfp4 (L7: 35 visits without msg? Wait: prelude 5S, p0 10S, tail 3S = 18 S without msg; p1/p2 20S with msg)
    # - Full nvfp4 (L5 prelude, L11 loop)
    # - Sliding mxfp8 (L20..22 = 3 visits)
    # - Full mxfp8 (L23 = 1 visit)
    bs_3 = mx.random.normal((3, B, seq, 2304)).astype(mx.float32)
    bank = (mx.random.normal((2, B, seq, 2304)).astype(mx.float32), mx.random.normal((B, seq, 2304)).astype(mx.float32))
    mx.eval(bs_3, bank)

    def prof_layer(li, has_msg, n_visits):
        layer = tm.layers[li]
        sliding = isinstance(layer, L.SlidingDecoderLayer)
        a = layer.self_attn
        c = model.make_cache()[0 if sliding else 5]
        if ctx_len > 0:
            c[0].update_and_fetch(
                mx.random.normal((1, a.num_key_value_heads, ctx_len, 128)).astype(mx.bfloat16),
                mx.random.normal((1, a.num_key_value_heads, ctx_len, 128)).astype(mx.bfloat16),
            )
            mx.eval(c[0].state)
        cs = [mx.zeros((1, 3, a.num_key_value_heads * 128 if j < 2 else 2304), mx.float32) for j in range(4)]
        mx.eval(cs)
        msg = tm.message_input(bank, 0) if has_msg else None
        if msg is not None:
            mx.eval(msg)

        stats = {}
        stats["unw_rms1"] = time_op(lambda: L.unweighted_rms_norm(h.astype(mx.float32)))
        nr = L.unweighted_rms_norm(h.astype(mx.float32)); mx.eval(nr)
        stats["attn_input"] = time_op(lambda: layer.attention_input(bs_3, nr))
        ai = layer.attention_input(bs_3, nr); mx.eval(ai)
        if has_msg:
            stats["msg_input"] = time_op(lambda: tm.message_input(bank, 0))
            if sliding:
                hs = ai[..., 0, :]
                stats["msg_gates_qkv"] = time_op(lambda: layer._mix_messages((hs,) * 3, msg))
                st = layer._mix_messages((hs,) * 3, msg); mx.eval(st)
                stats["in_ln"] = time_op(lambda: [layer.input_layernorm(s).astype(dtype) for s in st])
                q_in, k_in, v_in = [layer.input_layernorm(s).astype(dtype) for s in st]
            else:
                qs, ks, vs = (ai[..., i, :] for i in range(3))
                stats["msg_gates_qkv"] = time_op(lambda: layer._mix_messages((qs, ks, vs), msg))
                qs, ks, vs = layer._mix_messages((qs, ks, vs), msg); mx.eval(qs, ks, vs)
                stats["in_ln"] = time_op(lambda: (layer.q_input_layernorm(qs).astype(dtype), layer.k_input_layernorm(ks).astype(dtype), layer.v_input_layernorm(vs).astype(dtype)))
                q_in = layer.q_input_layernorm(qs).astype(dtype); k_in = layer.k_input_layernorm(ks).astype(dtype); v_in = layer.v_input_layernorm(vs).astype(dtype)
        else:
            stats["msg_input"] = 0.0
            stats["msg_gates_qkv"] = 0.0
            if sliding:
                hs = ai[..., 0, :]
                stats["in_ln"] = time_op(lambda: layer.input_layernorm(hs).astype(dtype))
                q_in = k_in = v_in = layer.input_layernorm(hs).astype(dtype)
            else:
                qs, ks, vs = (ai[..., i, :] for i in range(3))
                stats["in_ln"] = time_op(lambda: (layer.q_input_layernorm(qs).astype(dtype), layer.k_input_layernorm(ks).astype(dtype), layer.v_input_layernorm(vs).astype(dtype)))
                q_in = layer.q_input_layernorm(qs).astype(dtype); k_in = layer.k_input_layernorm(ks).astype(dtype); v_in = layer.v_input_layernorm(vs).astype(dtype)
        mx.eval(q_in, k_in, v_in)

        stats["qkv_projs"] = time_op(lambda: (a.q_proj(q_in), a.r_proj(q_in), a.gate_proj(q_in), a.sink_proj(q_in), a.k_proj(k_in), a.v_proj(v_in)))
        qp, rp, gate, sinks, kp, vp = a.q_proj(q_in), a.r_proj(q_in), a.gate_proj(q_in), a.sink_proj(q_in), a.k_proj(k_in), a.v_proj(v_in)
        mx.eval(qp, rp, gate, sinks, kp, vp)

        if sliding:
            stats["kv_sconv"] = time_op(lambda: (a.k_sconv(kp, cs[0], None), a.v_sconv(vp, cs[1], None)))
            (k_c, _), (v_c, _) = a.k_sconv(kp, cs[0], None), a.v_sconv(vp, cs[1], None)
            tau = None
        else:
            stats["kv_sconv"] = time_op(lambda: (a.k_sconv(kp, k_in, cs[0], None), a.v_sconv(vp, v_in, cs[1], None)))
            (k_c, _), (v_c, _) = a.k_sconv(kp, k_in, cs[0], None), a.v_sconv(vp, v_in, cs[1], None)
            pos = L._positions(ctx_len, seq).astype(mx.float32)
            tau = (1.0 + a.log_scaling_alpha * mx.log(mx.maximum((pos + 1) / a.log_scaling_n_floor, 1.0)))[..., None, None]
            mx.eval(tau)
        mx.eval(k_c, v_c)

        stats["attend_pre_clips_rel"] = time_op(lambda: a._attend_pre(q_in, qp, k_c, v_c, rp, tau, gate=gate, sinks=sinks))
        qs_t, ks_t, vs_t, pb, sinks, gate = a._attend_pre(q_in, qp, k_c, v_c, rp, tau, gate=gate, sinks=sinks)
        mx.eval(qs_t, ks_t, vs_t, pb)

        # kv update cost (without mutating cache across reps: clone keys/values or measure once on fresh copy)
        if ctx_len > 0:
            def upd():
                ck = mx.array(c[0].keys); cv = mx.array(c[0].values); mx.eval(ck, cv)
                from mlx_lm.models.cache import KVCache, RotatingKVCache
                tmp = KVCache() if not sliding else RotatingKVCache(1024)
                tmp.keys, tmp.values, tmp.offset = ck, cv, c[0].offset
                if sliding: tmp._idx = c[0]._idx
                t0 = time.perf_counter()
                k_all, v_all = tmp.update_and_fetch(ks_t, vs_t)
                mx.eval(k_all, v_all)
                return time.perf_counter() - t0
            stats["kv_update"] = statistics.median([upd() for _ in range(4)]) * 1e3
        else:
            stats["kv_update"] = 0.0
        k_all, v_all = c[0].update_and_fetch(ks_t, vs_t)
        mx.eval(k_all, v_all)

        stats["rel_attn"] = time_op(lambda: a._attention(qs_t, k_all, v_all, pb, sinks, c[0]))
        attn_out = a._attention(qs_t, k_all, v_all, pb, sinks, c[0]); mx.eval(attn_out)

        stats["o_proj_gate"] = time_op(lambda: a._attend_post(attn_out, gate))
        att = a._attend_post(attn_out, gate); mx.eval(att)

        stats["post_attn_conv"] = time_op(lambda: layer.post_attention_convolution(att, cs[2], None))
        hid, _ = layer.post_attention_convolution(att, cs[2], None); mx.eval(hid)
        res = h.astype(mx.float32) + hid.astype(mx.float32); mx.eval(res)
        stats["post_attn_rms_mlp_in"] = time_op(lambda: layer.mlp_input(bs_3, L.unweighted_rms_norm(res))[..., 0, :])
        mh = layer.mlp_input(bs_3, L.unweighted_rms_norm(res))[..., 0, :]; mx.eval(mh)

        if has_msg:
            stats["msg_gate_mlp"] = time_op(lambda: layer.mlp_message_gate(mh, msg[..., 3, :]))
            mh = layer.mlp_message_gate(mh, msg[..., 3, :]); mx.eval(mh)
        else:
            stats["msg_gate_mlp"] = 0.0

        stats["post_attn_ln"] = time_op(lambda: layer.post_attention_layernorm(mh).astype(dtype))
        mx_in = layer.post_attention_layernorm(mh).astype(dtype); mx.eval(mx_in)

        moe = layer.mlp; ex = moe.experts
        stats["router"] = time_op(lambda: moe.gate(mx_in))
        inds, wts = moe.gate(mx_in); mx.eval(inds, wts)
        xe = mx.expand_dims(mx_in, (-2, -3))
        stats["moe_sort"] = time_op(lambda: _gather_sort(xe, inds))
        xs, idx_s, inv = _gather_sort(xe, inds); mx.eval(xs, idx_s, inv)
        stats["moe_gate_up"] = time_op(lambda: ex.gate_up_proj(xs, idx_s, sorted_indices=True))
        gu = ex.gate_up_proj(xs, idx_s, sorted_indices=True); mx.eval(gu)
        stats["moe_swiglu"] = time_op(lambda: ex.act_fn(gu[..., 0::2], gu[..., 1::2]))
        act = ex.act_fn(gu[..., 0::2], gu[..., 1::2]); mx.eval(act)
        stats["moe_down"] = time_op(lambda: ex.down_proj(act, idx_s, sorted_indices=True))
        dn = ex.down_proj(act, idx_s, sorted_indices=True); mx.eval(dn)
        stats["moe_unsort"] = time_op(lambda: _scatter_unsort(dn, inv, inds.shape).squeeze(-2))
        es = _scatter_unsort(dn, inv, inds.shape).squeeze(-2); mx.eval(es)
        stats["moe_combine"] = time_op(lambda: (es.astype(mx.float32) * wts[..., None]).sum(axis=-2).astype(dtype))
        mo = (es.astype(mx.float32) * wts[..., None]).sum(axis=-2).astype(dtype); mx.eval(mo)
        stats["post_mlp_conv"] = time_op(lambda: layer.post_mlp_convolution(mo, cs[3], None))
        _, w3 = layer.post_mlp_convolution(mo, cs[3], None); mx.eval(w3)
        stats["conv_state_take"] = time_op(lambda: L._conv_state(w3, None, 3))

        tot = sum(stats.values())
        print(f"Layer {li:2d} ({'S' if sliding else 'F'}, msg={int(has_msg)}, {ex.gate_up_proj.mode}, visits={n_visits}): sum={tot:.2f} ms/visit -> {tot*n_visits:.1f} ms")
        for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
            if v >= 0.05:
                print(f"  {k:22s}: {v:6.2f} ms/visit  ({100*v/tot:4.1f}%)  x{n_visits}={v*n_visits:6.1f} ms")
        return stats, n_visits

    groups = [
        prof_layer(7, False, 17),   # sliding nvfp4 no-msg (L0..4, L6..16 p0, L18..19)
        prof_layer(7, True, 20),    # sliding nvfp4 msg (p1, p2)
        prof_layer(5, False, 3),    # full nvfp4 no-msg (L5, L11 p0, L17 p0)
        prof_layer(11, True, 4),    # full nvfp4 msg (L11, L17 p1/p2)
        prof_layer(20, False, 3),   # sliding mxfp8 (L20..22)
        prof_layer(23, False, 1),   # full mxfp8 (L23)
    ]
    agg = {}
    for st, nv in groups:
        for k, v in st.items():
            agg[k] = agg.get(k, 0.0) + v * nv
    grand = sum(agg.values())
    print(f"\nEstimated 48-visit chunk breakdown (sum={grand:.1f} ms = {2048/(grand/1e3):.0f} tok/s):")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s}: {v:6.1f} ms  ({100*v/grand:5.1f}%)")


if __name__ == "__main__":
    bench_prefill_steps()
    bench_c8_prefill()
    profile_ops_at(0)
    profile_ops_at(14336)
