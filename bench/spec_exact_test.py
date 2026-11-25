"""Bit-exactness & latency gate for causal W-token verify_w + commit_w in BatchDecoder (p1/p3)."""
import os
import statistics
import sys
import time

import mlx_lm  # first
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2.batchdecode import BatchDecoder


def prefill_ids(model, ids):
    cache = make_prompt_cache(model)
    p = mx.array(ids)
    while p.size > 0:
        m = min(2048, p.size)
        model(p[:m][None], cache=cache)
        mx.eval([c.state for c in cache])
        p = p[m:]
    return cache


def snapshot_b1(dec):
    kvs = [mx.array(dec.kv_packed(v)) for v in range(dec.nv)]
    offs = [int(c[0].offset) for c in dec.caches[0]]
    idxs = [int(getattr(c[0], "_idx", 0)) for c in dec.caches[0]]
    convs = [[mx.array(x) for x in cv] for cv in dec.conv]
    mx.eval(kvs, convs)
    return kvs, offs, idxs, convs


def restore_b1(dec, snap):
    dec.unstack_kv()
    kvs, offs, idxs, convs = snap
    for v, (kv_arr, off, idx, cs) in enumerate(zip(kvs, offs, idxs, convs)):
        c = dec.caches[0][v][0]
        c._packed = mx.array(kv_arr)
        c.offset = off
        if hasattr(c, "_idx"):
            c._idx = idx
        dec.conv[v] = [mx.array(x) for x in cs]
        mx.eval(c._packed, dec.conv[v])


def check_case(model, ids, cand_tokens, W, keeps, tap_visits):
    c_ref = prefill_ids(model, ids)
    dec_ref = BatchDecoder(model, [c_ref])
    assert dec_ref._b1_fast
    snap0 = snapshot_b1(dec_ref)

    # Sequential ground truth for W steps + tap F at each step + intermediate snapshots at each keep
    ref_toks = []
    ref_Fs = []
    ref_snaps = {}
    for b in range(W):
        tok_b, tap_b = dec_ref.step_with_taps(mx.array([cand_tokens[b]], mx.uint32), tap_visits=tap_visits)
        mx.eval(tok_b, list(tap_b.values()))
        ref_toks.append(int(tok_b.item()))
        ref_Fs.append(tap_b["F"])
        if (b + 1) in keeps:
            ref_snaps[b + 1] = snapshot_b1(dec_ref)

    # Verify W in one causal forward starting from identical snap0
    c_test = prefill_ids(model, ids)
    dec_test = BatchDecoder(model, [c_test])
    restore_b1(dec_test, snap0)
    toks_w, kv_news, conv_ws, tap_w = dec_test.verify_w(mx.array(cand_tokens[:W], mx.uint32), tap_visits=tap_visits)
    mx.eval(toks_w, kv_news, conv_ws, list(tap_w.values()))
    got_toks = toks_w.tolist()
    if got_toks != ref_toks:
        raise AssertionError(f"W={W} S0={len(ids)} toks mismatch: {got_toks} != {ref_toks}")

    ref_F_cat = mx.concatenate(ref_Fs, axis=0)
    max_F_diff = float(mx.max(mx.abs(tap_w["F"].astype(mx.float32) - ref_F_cat.astype(mx.float32))).item())
    if max_F_diff != 0.0:
        raise AssertionError(f"W={W} S0={len(ids)} tap F max_abs_diff={max_F_diff:.3e} != 0")

    snap_verify = snapshot_b1(dec_test)

    # Check commit_w(keep) and post-commit decode parity
    for keep in sorted(keeps):
        restore_b1(dec_test, snap_verify)
        dec_test.commit_w(kv_news, conv_ws, keep)
        mx.eval(dec_test.kv_eval_targets(), dec_test.conv)

        # Compare cache offset, _idx, valid KV entries, and conv states against ref_snaps[keep]
        r_kvs, r_offs, r_idxs, r_convs = ref_snaps[keep]
        for v in range(dec_test.nv):
            kv_t = dec_test.caches[0][v][0]
            assert kv_t.offset == r_offs[v], f"v={v} offset {kv_t.offset} != {r_offs[v]}"
            if hasattr(kv_t, "_idx"):
                assert kv_t._idx == r_idxs[v], f"v={v} _idx {kv_t._idx} != {r_idxs[v]}"
                valid = min(r_offs[v], kv_t.max_size)
            else:
                valid = r_offs[v]
            k_diff = float(mx.max(mx.abs(dec_test.kv_packed(v)[:, :, :valid, :].astype(mx.float32) - r_kvs[v][:, :, :valid, :].astype(mx.float32))).item())
            if k_diff != 0.0:
                raise AssertionError(f"W={W} keep={keep} v={v} kv_buf max_abs_diff={k_diff:.3e}")
            for j in range(4):
                c_diff = float(mx.max(mx.abs(dec_test.conv[v][j] - r_convs[v][j])).item())
                if c_diff != 0.0:
                    raise AssertionError(f"W={W} keep={keep} v={v} conv[{j}] max_abs_diff={c_diff:.3e}")

        # Run 8 follow-up steps on both dec_test and dec_ref
        restore_b1(dec_ref, ref_snaps[keep])
        seed = cand_tokens[keep - 1]
        cur_r = mx.array([seed], mx.uint32)
        cur_t = mx.array([seed], mx.uint32)
        for s in range(8):
            cur_r = dec_ref.step(cur_r)
            cur_t = dec_test.step(cur_t)
            mx.eval(cur_r, cur_t)
            if int(cur_r.item()) != int(cur_t.item()):
                raise AssertionError(f"W={W} keep={keep} followup step {s}: {int(cur_t.item())} != {int(cur_r.item())}")

    print(f"PASS S0={len(ids):4d} W={W} keeps={sorted(keeps)} max_F_diff={max_F_diff:.2e} follow8=exact", flush=True)


def _all_tree_paths(parents):
    W = len(parents)
    children = {i: [] for i in range(W)}
    for i in range(1, W):
        children[parents[i]].append(i)
    paths = []

    def dfs(u, cur):
        if not children[u]:
            paths.append(tuple(cur))
            return
        for c in children[u]:
            dfs(c, cur + [c])

    dfs(0, [0])
    return paths


def check_tree_case(model, ids, cand_tokens, label, parents, depths, tap_visits):
    W = len(parents)
    assert len(cand_tokens) >= W
    c_test = prefill_ids(model, ids)
    dec_test = BatchDecoder(model, [c_test])
    assert dec_test._b1_fast
    snap0 = snapshot_b1(dec_test)

    # Run causal tree verify_w once from snap0
    toks_w, kv_news, conv_ws, tap_w = dec_test.verify_w(
        mx.array(cand_tokens[:W], mx.uint32), tap_visits=tap_visits, parents=parents, depths=depths,
    )
    mx.eval(toks_w, kv_news, conv_ws, list(tap_w.values()))
    got_toks = toks_w.tolist()
    snap_verify = snapshot_b1(dec_test)

    c_ref = prefill_ids(model, ids)
    dec_ref = BatchDecoder(model, [c_ref])
    leaf_paths = _all_tree_paths(parents)
    max_f_diff = 0.0
    checked_commits = []

    # Verify every root-to-leaf path matches sequential B=1 step_with_taps
    for p_idx, path in enumerate(leaf_paths):
        restore_b1(dec_ref, snap0)
        for pos, node in enumerate(path):
            tok_b, tap_b = dec_ref.step_with_taps(mx.array([cand_tokens[node]], mx.uint32), tap_visits=tap_visits)
            mx.eval(tok_b, list(tap_b.values()))
            ref_tok = int(tok_b.item())
            if got_toks[node] != ref_tok:
                raise AssertionError(f"TREE {label} S0={len(ids)} path={path} node={node}: {got_toks[node]} != {ref_tok}")
            fd = float(mx.max(mx.abs(tap_w["F"][node].astype(mx.float32) - tap_b["F"][0].astype(mx.float32))).item())
            if fd > max_f_diff:
                max_f_diff = fd
            if fd != 0.0:
                raise AssertionError(f"TREE {label} S0={len(ids)} path={path} node={node} F_diff={fd:.3e} != 0")

        # Also test commit_w(path=...) on non-contiguous paths + follow-up 8 B=1 steps
        # Pick full leaf path on even index, and a proper prefix on non-contiguous paths
        test_paths = [list(path)]
        if len(path) >= 2 and path != tuple(range(len(path))):
            test_paths.append(list(path[:-1]))
        for cp in test_paths:
            # Ground-truth sequential state after cp
            restore_b1(dec_ref, snap0)
            for node in cp:
                cur_r = dec_ref.step(mx.array([cand_tokens[node]], mx.uint32))
                mx.eval(cur_r)
            ref_snap = snapshot_b1(dec_ref)

            restore_b1(dec_test, snap_verify)
            dec_test.commit_w(kv_news, conv_ws, len(cp), path=cp)
            mx.eval(dec_test.kv_eval_targets(), dec_test.conv)

            r_kvs, r_offs, r_idxs, r_convs = ref_snap
            for v in range(dec_test.nv):
                kv_t = dec_test.caches[0][v][0]
                assert kv_t.offset == r_offs[v], f"{label} cp={cp} v={v} offset {kv_t.offset} != {r_offs[v]}"
                if hasattr(kv_t, "_idx"):
                    assert kv_t._idx == r_idxs[v], f"{label} cp={cp} v={v} _idx {kv_t._idx} != {r_idxs[v]}"
                    valid = min(r_offs[v], kv_t.max_size)
                else:
                    valid = r_offs[v]
                kd = float(mx.max(mx.abs(dec_test.kv_packed(v)[:, :, :valid, :].astype(mx.float32) - r_kvs[v][:, :, :valid, :].astype(mx.float32))).item())
                if kd != 0.0:
                    raise AssertionError(f"TREE {label} S0={len(ids)} cp={cp} v={v} kv_diff={kd:.3e}")
                for j in range(4):
                    cd = float(mx.max(mx.abs(dec_test.conv[v][j] - r_convs[v][j])).item())
                    if cd != 0.0:
                        raise AssertionError(f"TREE {label} S0={len(ids)} cp={cp} v={v} conv[{j}]={cd:.3e}")

            restore_b1(dec_ref, ref_snap)
            seed = cand_tokens[cp[-1]]
            cur_r = mx.array([seed], mx.uint32)
            cur_t = mx.array([seed], mx.uint32)
            for s in range(8):
                cur_r = dec_ref.step(cur_r)
                cur_t = dec_test.step(cur_t)
                mx.eval(cur_r, cur_t)
                if int(cur_r.item()) != int(cur_t.item()):
                    raise AssertionError(f"TREE {label} S0={len(ids)} cp={cp} follow {s}: {int(cur_t.item())} != {int(cur_r.item())}")
            checked_commits.append(cp)

    print(
        f"PASS TREE {label:10s} S0={len(ids):4d} W={W} paths={len(leaf_paths)} commits={len(checked_commits)} "
        f"max_F_diff={max_f_diff:.2e} follow8=exact",
        flush=True,
    )


def main():
    passes = int(os.environ.get("PASSES", "1"))
    model, tok = load(os.path.expanduser(f"~/loopling/models/loopling-p{passes}"))
    mx.eval(model.parameters())
    prompt_text = "Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests."
    base_ids = tok.encode(
        tok.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False)
        + "<|message_model|><|content_text|>",
        add_special_tokens=False,
    )
    cand = [40, 149951, 150572, 149899, 149890, 220, 198, 42]
    taps = {5, 14, 17} if passes == 1 else {5, 23, 41}

    # 1. Short context
    for W in (2, 3, 4, 6, 8):
        keeps = sorted({1, 2, max(1, W // 2), W})
        check_case(model, base_ids, cand, W, keeps, taps)

    # 2. Exact 1024 boundary crossing (S0 = 1020, W = 8 -> spans 1021..1028)
    ids_1020 = [(17 * j * j + 3 * j + 7) % 40000 + 1000 for j in range(1020 - len(base_ids))] + base_ids
    check_case(model, ids_1020, cand, 8, {1, 3, 5, 8}, taps)

    # 3. Wrapped long rotating context (S0 = 1500, W = 8)
    ids_1500 = [(17 * j * j + 3 * j + 7) % 40000 + 1000 for j in range(1500 - len(base_ids))] + base_ids
    check_case(model, ids_1500, cand, 8, {1, 4, 8}, taps)

    # 3b. Arbitrary-topology speculative trees (S0 = 45, S0 = 1020 crossing, S0 = 1500 wrapped)
    # Depth-sorted topologies (node 0 = anchor, parent[b] < b, depths non-decreasing):
    #   wide_fan4:   0 -> {1, 2, 3}
    #   bin_tree6:   0 -> {1, 2}, 1 -> {3, 4}, 2 -> {5}
    #   beam_tree8:  0 -> {1, 2}, 1 -> {3, 4}, 2 -> {5}, 3 -> {6}, 4 -> {7} (depths 0..3)
    #   deep_asym8:  0 -> {1, 2}, 1 -> {3}, 3 -> {4, 5}, 4 -> {6}, 6 -> {7} (depths 0..4, spans >= 4-tap conv!)
    tree_specs = [
        ("wide_fan4",  (-1, 0, 0, 0),            (0, 1, 1, 1)),
        ("bin_tree6",  (-1, 0, 0, 1, 1, 2),      (0, 1, 1, 2, 2, 2)),
        ("beam_tree8", (-1, 0, 0, 1, 1, 2, 3, 4), (0, 1, 1, 2, 2, 2, 3, 3)),
        ("deep_asym8", (-1, 0, 0, 1, 3, 3, 4, 6), (0, 1, 1, 2, 3, 3, 4, 5)),
    ]
    for S_ids in (base_ids, ids_1020, ids_1500):
        for nm, pars, deps in tree_specs:
            check_tree_case(model, S_ids, cand, nm, pars, deps, taps)

    # 4. Latency benchmark W = 1, 2, 3, 4, 5, 6, 8
    print(f"\n=== CAUSAL VERIFY LATENCY p{passes} (S0={len(base_ids)} and S0=1500, median of 15) ===", flush=True)
    for S_lbl, p_ids in [("short", base_ids), ("long1500", ids_1500)]:
        c_b = prefill_ids(model, p_ids)
        dec_b = BatchDecoder(model, [c_b])
        snap = snapshot_b1(dec_b)
        row_out = []
        # W=1 via step_with_taps
        t1_arr = mx.array([cand[0]], mx.uint32)
        def run1():
            restore_b1(dec_b, snap)
            o, tp = dec_b.step_with_taps(t1_arr, tap_visits=taps)
            return (o, list(tp.values()))
        # Better: time without snapshot restore overhead! On verify_w caches don't mutate so we can time verify_w directly + commit_w separately!
        # For W=1, advance sequentially over 15 tokens after 5 warmup without restore overhead:
        ts1 = []
        for i in range(20):
            mx.synchronize()
            t0 = time.perf_counter()
            o, tp = dec_b.step_with_taps(t1_arr, tap_visits=taps)
            mx.eval(o, list(tp.values()))
            mx.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= 5:
                ts1.append(dt)
        row_out.append(f"W=1:{statistics.median(ts1):.2f}ms")
        restore_b1(dec_b, snap)
        for W in (2, 3, 4, 5, 6, 8):
            blk = mx.array(cand[:W], mx.uint32)
            mx.eval(blk)
            ts = []
            for i in range(20):
                mx.synchronize()
                t0 = time.perf_counter()
                nt, kn, cw, tp = dec_b.verify_w(blk, tap_visits=taps)
                mx.eval(nt, kn, cw, list(tp.values()))
                mx.synchronize()
                dt = (time.perf_counter() - t0) * 1000.0
                if i >= 5:
                    ts.append(dt)
            row_out.append(f"W={W}:{statistics.median(ts):.2f}ms")
        # Time commit_w(keep=3) for W=6
        blk6 = mx.array(cand[:6], mx.uint32)
        nt, kn, cw, tp = dec_b.verify_w(blk6, tap_visits=taps)
        mx.eval(nt, kn, cw, list(tp.values()))
        ts_c = []
        for i in range(15):
            restore_b1(dec_b, snap)
            mx.synchronize()
            t0 = time.perf_counter()
            dec_b.commit_w(kn, cw, 3)
            mx.eval(dec_b.kv_eval_targets(), dec_b.conv)
            mx.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= 3:
                ts_c.append(dt)
        row_out.append(f"commit(3):{statistics.median(ts_c):.2f}ms")
        print(f"  {S_lbl:8s}: " + " | ".join(row_out), flush=True)
    print("ALL SPEC EXACT TESTS PASS", flush=True)


if __name__ == "__main__":
    main()
