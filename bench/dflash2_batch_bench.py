import os
import time
import numpy as np
import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2.batchdecode import BatchDecoder
from mlx_lm.models.inkling2 import dflash2 as DF2
from mlx_lm.models.inkling2.mtp import default_taps, feats_from_buf, parse_taps

PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
    "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
    "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
    "Explain how a B-tree insertion works, step by step, with an example of a node split.",
    "Write a bash script that finds the ten largest files under a directory and prints their sizes in MB.",
    "Summarize the causes of the French Revolution in five bullet points, then give one counterargument.",
    "Implement binary search in Rust and explain the off-by-one pitfalls.",
    "Describe the water cycle to a ten-year-old, then list three common misconceptions.",
]

model, tok = mlx_lm.load("/Users/jeweled/loopling/models/loopling-p3")
tm = model.model.language_model
passes = tm.args.loop_config["num_passes"]
taps = parse_taps(os.environ.get("LOOPLING_MTP_TAPS", default_taps(passes)), passes)
tap_keys = [t if t in ("F", "f", "o") else int(t[1:]) for t in taps]
tap_set = {k for k in tap_keys if isinstance(k, int)}
dr = DF2._get_drafter(model, DF2.DFLASH2_DIR)


def _feats(tb):
    return mx.concatenate([tb[k].reshape(-1, 1, 2304) for k in tap_keys], axis=1)[None]


def encode(t):
    chat = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False)
    return tok.encode(chat + "<|message_model|><|content_text|>", add_special_tokens=False)


def prefill_with_state(ids):
    c = make_prompt_cache(model)
    st = dr.new_stream_state()
    st.lookup.append(ids)
    p = mx.array(ids, mx.int32)
    pos = 0
    n_p = int(p.size)
    tm.tap_visits = tap_set
    while n_p - pos > 1:
        tm.tap_buf = {}
        m = min(2048, n_p - 1 - pos)
        _ = model(p[pos : pos + m][None], cache=c)
        f = feats_from_buf(tm, taps)
        dr.append_ctx_batch([st], [f])
        mx.eval([x.state for x in c], *st.ck, *st.cv)
        pos += m
    tm.tap_visits = None
    tm.tap_buf = {}
    return c, st


ids_all = [encode(t) for t in PROMPTS]
N_TOK = 128

refs = []
for i in range(8):
    refs.append([int(t) for t, _ in generate_step(mx.array(ids_all[i], mx.int32), model, max_tokens=N_TOK)])


def run_ar_batch(C):
    caches = [make_prompt_cache(model) for _ in range(C)]
    for i in range(C):
        p = mx.array(ids_all[i], mx.int32)
        _ = model(p[:-1][None], cache=caches[i])
        mx.eval([x.state for x in caches[i]])
    dec = BatchDecoder(model, caches)
    y = mx.array([ids_all[i][-1] for i in range(C)], mx.int32)
    # Warmup 4 tokens
    for _ in range(4):
        y = dec.step(y)
        mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(N_TOK - 4):
        y = dec.step(y)
        mx.eval(y)
    dt = time.perf_counter() - t0
    agg = C * (N_TOK - 4) / dt
    del dec, caches
    mx.clear_cache()
    return agg, agg / C, 1e3 * dt / (N_TOK - 4)


def run_df2_batch(C, W):
    caches, states = [], []
    for i in range(C):
        c, st = prefill_with_state(ids_all[i])
        caches.append(c)
        states.append(st)
    dec = BatchDecoder(model, caches)
    y0 = mx.array([ids_all[i][-1] for i in range(C)], mx.int32)
    # First 1 token per stream via verify_batch_w or step_with_taps? Run AR warmup step to get anchor + tap_0!
    # Actually use dec.step_taps or 1 dummy verify_batch_w of W tokens starting at prompt[-1]!
    out = [[] for _ in range(C)]
    anc = [ids_all[i][-1] for i in range(C)]
    for i, a in enumerate(anc):
        states[i].lookup.append_one(a)

    # Warmup 1 cycle
    blk = dr.draft_batch_chains(states, anc, W)
    t_cw, kv_news, conv_ws, tap_buf = dec.verify_batch_w(blk, W, tap_visits=tap_set)
    mx.eval(t_cw)
    tl = t_cw.tolist()
    bl = blk.tolist()
    keeps, f_list = [], []
    fw = _feats(tap_buf)[0]  # (C*W, 4, H)
    for c in range(C):
        row = bl[c]
        k = 1
        while k < W and tl[c][k - 1] == row[k]:
            out[c].append(row[k])
            states[c].lookup.append_one(row[k])
            k += 1
        bonus = tl[c][k - 1]
        out[c].append(bonus)
        states[c].lookup.append_one(bonus)
        anc[c] = bonus
        keeps.append(k)
        f_list.append(fw[c * W : c * W + k][None])
    dec.commit_batch_w(kv_news, conv_ws, keeps, W)
    dr.append_ctx_batch(states, f_list)
    mx.eval(*[st.ck for st in states], *[st.cv for st in states])

    t0 = time.perf_counter()
    cycles = 0
    draft_ms = 0.0
    verify_ms = 0.0
    while min(len(o) for o in out) < N_TOK:
        td0 = time.perf_counter()
        blk = dr.draft_batch_chains(states, anc, W)
        mx.eval(blk)
        draft_ms += (time.perf_counter() - td0) * 1000
        tv0 = time.perf_counter()
        t_cw, kv_news, conv_ws, tap_buf = dec.verify_batch_w(blk, W, tap_visits=tap_set)
        mx.eval(t_cw)
        verify_ms += (time.perf_counter() - tv0) * 1000
        tl = t_cw.tolist()
        bl = blk.tolist()
        keeps, f_list = [], []
        fw = _feats(tap_buf)[0]
        for c in range(C):
            row = bl[c]
            k = 1
            while k < W and tl[c][k - 1] == row[k]:
                out[c].append(row[k])
                states[c].lookup.append_one(row[k])
                k += 1
            bonus = tl[c][k - 1]
            out[c].append(bonus)
            states[c].lookup.append_one(bonus)
            anc[c] = bonus
            keeps.append(k)
            f_list.append(fw[c * W : c * W + k][None])
        dec.commit_batch_w(kv_news, conv_ws, keeps, W)
        dr.append_ctx_batch(states, f_list)
        cycles += 1

    mx.synchronize()
    dt = time.perf_counter() - t0
    tot_new = sum(len(o) for o in out) - sum(k for k in keeps if False)
    # Count exact tokens generated after warmup cycle
    # Re-track tokens generated inside timed loop cleanly:
    mm = 0
    for c in range(C):
        mm += sum(1 for a, b in zip(out[c][:N_TOK], refs[c][:N_TOK]) if a != b)
    dec.flush()
    del dec, caches, states
    mx.clear_cache()
    return cycles, draft_ms / cycles, verify_ms / cycles, mm, out, dt


for C, ws in [(2, [2, 3, 4]), (3, [2, 3, 4]), (4, [2, 3, 4]), (6, [2]), (8, [2])]:
    ar_agg, ar_ps, ar_ms = run_ar_batch(C)
    print(f"\n=== C={C} | BatchDecoder AR: {ar_agg:.1f} agg tok/s ({ar_ps:.1f}/stream, {ar_ms:.2f}ms/step) ===")
    for W in ws:
        # Count generated tokens accurately inside timed loop
        caches, states = [], []
        for i in range(C):
            c, st = prefill_with_state(ids_all[i])
            caches.append(c)
            states.append(st)
        dec = BatchDecoder(model, caches)
        out = [[] for _ in range(C)]
        anc = [ids_all[i][-1] for i in range(C)]
        for i, a in enumerate(anc):
            states[i].lookup.append_one(a)
        # Warmup 1 cycle
        blk = dr.draft_batch_chains(states, anc, W)
        t_cw, kv_news, conv_ws, tap_buf = dec.verify_batch_w(blk, W, tap_visits=tap_set)
        mx.eval(t_cw)
        tl, bl = t_cw.tolist(), blk.tolist()
        fw = _feats(tap_buf)[0]
        keeps, f_list = [], []
        for c in range(C):
            row = bl[c]
            k = 1
            while k < W and tl[c][k - 1] == row[k]:
                out[c].append(row[k]); states[c].lookup.append_one(row[k]); k += 1
            bonus = tl[c][k - 1]; out[c].append(bonus); states[c].lookup.append_one(bonus); anc[c] = bonus; keeps.append(k)
            f_list.append(fw[c * W : c * W + k][None])
        dec.commit_batch_w(kv_news, conv_ws, keeps, W)
        dr.append_ctx_batch(states, f_list)
        mx.eval([x for s in states for x in s.ck + s.cv])

        warm_toks = sum(len(o) for o in out)
        t0 = time.perf_counter()
        cycles = 0
        d_ms = v_ms = 0.0
        while min(len(o) for o in out) < N_TOK:
            td0 = time.perf_counter()
            blk = dr.draft_batch_chains(states, anc, W)
            mx.eval(blk)
            d_ms += (time.perf_counter() - td0) * 1000
            tv0 = time.perf_counter()
            t_cw, kv_news, conv_ws, tap_buf = dec.verify_batch_w(blk, W, tap_visits=tap_set)
            mx.eval(t_cw)
            v_ms += (time.perf_counter() - tv0) * 1000
            tl, bl = t_cw.tolist(), blk.tolist()
            fw = _feats(tap_buf)[0]
            keeps, f_list = [], []
            for c in range(C):
                row = bl[c]
                k = 1
                while k < W and tl[c][k - 1] == row[k]:
                    out[c].append(row[k]); states[c].lookup.append_one(row[k]); k += 1
                bonus = tl[c][k - 1]; out[c].append(bonus); states[c].lookup.append_one(bonus); anc[c] = bonus; keeps.append(k)
                f_list.append(fw[c * W : c * W + k][None])
            dec.commit_batch_w(kv_news, conv_ws, keeps, W)
            dr.append_ctx_batch(states, f_list)
            cycles += 1
        mx.synchronize()
        dt = time.perf_counter() - t0
        gen_toks = sum(len(o) for o in out) - warm_toks
        agg = gen_toks / dt
        tpc = gen_toks / (cycles * C)
        mm = sum(sum(1 for a, b in zip(out[c][:N_TOK], refs[c][:N_TOK]) if a != b) for c in range(C))
        print(f"  DF2 W={W}: {agg:6.1f} agg tok/s ({agg/ar_agg:.2f}x, {agg/C:.1f}/stream) tpc={tpc:.2f} cyc={1e3*dt/cycles:.2f}ms (draft={d_ms/cycles:.2f}ms verify={v_ms/cycles:.2f}ms) mm={mm}")
        del dec, caches, states
        mx.clear_cache()
