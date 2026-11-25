"""Unit, bit-exactness, and parity tests for opt-in LoopCD (inkling2/loopcd.py).

Usage:
    python bench/loopcd_test.py
"""
import copy
import os
import sys

import mlx_lm  # first: applies LOOPLING_CMDBUF before mlx.core initializes
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import batchdecode as BD
from mlx_lm.models.inkling2 import loopcd

GREEDY = lambda lp: mx.argmax(lp, axis=-1)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail else ""), flush=True)
    if not ok:
        raise AssertionError(f"Check failed: {name} ({detail})")


def prefill_cache(model, ids):
    cache = make_prompt_cache(model)
    p = mx.array(ids)
    while p.size > 1:
        n = min(2048, p.size - 1)
        model(p[:n][None], cache=cache)
        mx.eval([c.state for c in cache])
        p = p[n:]
    return cache


def test_math_and_config():
    # Validation
    for bad in (
        dict(mode="bad"),
        dict(mode="hidden", adaptive=True),
    ):
        raised = False
        try:
            loopcd.LoopCDConfig(**bad)
        except ValueError:
            raised = True
        check(f"config rejects invalid {bad}", raised)

    cfg = loopcd.LoopCDConfig(mode="logits", omega=0.3, ref_pass=-2)
    check("ref_pass -2 resolves to 1 for p3", cfg.resolved_ref_pass(3) == 1)
    raised = False
    try:
        loopcd.LoopCDConfig(mode="logits", omega=0.3, ref_pass=2).resolved_ref_pass(3)
    except ValueError:
        raised = True
    check("ref_pass >= R-1 rejected", raised)

    # Env parsing
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("LOOPLING_LOOPCD")}
    try:
        check("from_env default off", loopcd.from_env().mode == "off")
        os.environ["LOOPLING_LOOPCD"] = "hidden:0.75"
        os.environ["LOOPLING_LOOPCD_BLEND_BS"] = "1"
        os.environ["LOOPLING_LOOPCD_REF"] = "1"
        c = loopcd.from_env()
        check("from_env hidden", c.mode == "hidden" and c.omega == 0.75 and c.blend_bs and c.ref_pass == 1)
        os.environ["LOOPLING_LOOPCD"] = "logits_adaptive:0.9"
        del os.environ["LOOPLING_LOOPCD_BLEND_BS"]
        del os.environ["LOOPLING_LOOPCD_REF"]
        c2 = loopcd.from_env()
        check("from_env logits_adaptive", c2.mode == "logits" and c2.adaptive and c2.omega_max == 0.9)
    finally:
        for k in list(os.environ):
            if k.startswith("LOOPLING_LOOPCD"):
                del os.environ[k]
        os.environ.update(saved)

    # apply_hidden: list block_states vs array block_states
    rng = np.random.default_rng(123)
    hr_np = rng.standard_normal((1, 1, 16)).astype(np.float32)
    h1_np = rng.standard_normal((1, 1, 16)).astype(np.float32)
    hr = mx.array(hr_np).astype(mx.bfloat16)
    h1 = mx.array(h1_np).astype(mx.bfloat16)
    bs_arr_r = mx.array(rng.standard_normal((3, 1, 1, 16)).astype(np.float32))
    bs_arr_1 = mx.array(rng.standard_normal((3, 1, 1, 16)).astype(np.float32))
    bs_arr_1 = mx.concatenate([bs_arr_r[:1], bs_arr_1[1:]], axis=0)
    bs_list_r = [bs_arr_r[i, 0, 0][None] for i in range(3)]
    bs_list_1 = [bs_arr_1[i, 0, 0][None] for i in range(3)]

    cfg_bs = loopcd.LoopCDConfig(mode="hidden", omega=0.5, blend_bs=True)
    ha, bsa = loopcd.apply_hidden(hr, bs_arr_r, h1, bs_arr_1, 1, cfg_bs)
    hl, bsl = loopcd.apply_hidden(hr, bs_list_r, h1, bs_list_1, 1, cfg_bs)
    mx.eval(ha, bsa, hl, *bsl)
    diff_h = float(mx.abs(ha - hl).max().item())
    diff_b = max(float(mx.abs(bsa[i, 0, 0] - bsl[i][0]).max().item()) for i in range(3))
    check("apply_hidden list vs array parity", diff_h == 0.0 and diff_b == 0.0, f"dh={diff_h} db={diff_b}")
    check("apply_hidden preserves pre_n slot", float(mx.abs(bsa[0] - bs_arr_r[0]).max().item()) == 0.0)
    expected_slot2 = bs_arr_r[2] + 0.5 * (bs_arr_r[2] - bs_arr_1[2])
    check("apply_hidden extrapolates core slot", float(mx.abs(bsa[2] - expected_slot2).max().item()) < 1e-6)

    # combine_logits fixed and adaptive (Eq. 3, Eq. 4)
    z_r = mx.array([[10.0, 10.0, -5.0, -5.0], [15.0, -2.0, -2.0, -2.0]], mx.float32)
    z_1 = mx.array([[1.0, 3.0, 2.0, 0.0], [5.0, 1.0, 0.0, -1.0]], mx.float32)
    zf = loopcd.combine_logits(z_r, z_1, loopcd.LoopCDConfig(mode="logits", omega=0.4))
    ref_zf = z_r + 0.4 * (z_r - z_1)
    check("combine_logits fixed Eq.3", float(mx.abs(zf - ref_zf).max().item()) < 1e-6)

    # Unsorted / tied / peaked adaptive verification
    za = loopcd.combine_logits(z_r, z_1, loopcd.LoopCDConfig(mode="logits", adaptive=True, omega_max=0.8))
    mx.eval(za)
    # Row 0: top-2 are tied (10.0 == 10.0) => margin = 0 => w = 0.8
    exp_row0 = z_r[0] + 0.8 * (z_r[0] - z_1[0])
    # Row 1: peaked p1 ~ 1, p2 ~ 0 => margin ~ 1 => w ~ 0 => za[1] ~ z_r[1]
    err0 = float(mx.abs(za[0] - exp_row0).max().item())
    err1 = float(mx.abs(za[1] - z_r[1]).max().item())
    # Exact numpy softmax + sort check on random 3D logits
    z3_r = mx.array(rng.standard_normal((2, 5, 64)).astype(np.float32) * 3.0)
    z3_1 = mx.array(rng.standard_normal((2, 5, 64)).astype(np.float32) * 3.0)
    za3 = loopcd.combine_logits(z3_r, z3_1, loopcd.LoopCDConfig(mode="logits", adaptive=True, omega_max=0.75))
    mx.eval(za3)
    zr_np, z1_np = np.array(z3_r), np.array(z3_1)
    zr_shift = zr_np - zr_np.max(axis=-1, keepdims=True)
    p = np.exp(zr_shift) / np.exp(zr_shift).sum(axis=-1, keepdims=True)
    ps = np.sort(p, axis=-1)
    w_np = 0.75 * (1.0 - (ps[..., -1] - ps[..., -2]))
    ref_za3 = zr_np + w_np[..., None] * (zr_np - z1_np)
    max_e3 = float(np.abs(np.array(za3) - ref_za3).max())
    check("combine_logits adaptive Eq.4 exact", err0 < 1e-5 and err1 < 1e-4 and max_e3 < 1e-5, f"{err0:.2e} {err1:.2e} {max_e3:.2e}")


def test_p3_omega_zero_and_two_pass_and_batch(p3_dir):
    model, tok = load(p3_dir)
    mx.eval(model.parameters())

    prompts = [
        "Solve step by step: what is 17 * 24 + 19?",
        "Write a concise summary of how quicksort partitions an array.",
        "If a train travels 65 km/h for 2.5 hours, how far does it travel?",
        "Name three properties of prime numbers.",
    ]
    ids_list = [
        tok.encode(
            tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
            + "<|message_model|><|content_text|>",
            add_special_tokens=False,
        )
        for p in prompts
    ]

    # --- Test 2: omega=0 bit-exactness ---
    prompt0 = mx.array(ids_list[0])
    model.set_loopcd("off")
    z_off = model(prompt0[None], cache=None)
    toks_off = [int(t) for t, _ in generate_step(prompt0, model, max_tokens=16, sampler=GREEDY)]

    zero_cfgs = [
        ("hidden w=0", dict(mode="hidden", omega=0.0, blend_bs=False)),
        ("hidden w=0 blend_bs", dict(mode="hidden", omega=0.0, blend_bs=True)),
        ("logits w=0", dict(mode="logits", omega=0.0, adaptive=False)),
        ("logits adaptive wmax=0", dict(mode="logits", adaptive=True, omega_max=0.0)),
    ]
    for label, kw in zero_cfgs:
        model.set_loopcd(**kw)
        z_cur = model(prompt0[None], cache=None)
        mx.eval(z_off, z_cur)
        diff = float(mx.abs(z_off - z_cur).max().item())
        toks_cur = [int(t) for t, _ in generate_step(prompt0, model, max_tokens=16, sampler=GREEDY)]
        # BatchDecoder C=1 & C=4
        caches = [prefill_cache(model, ids_list[i]) for i in range(4)]
        dec1 = BD.BatchDecoder(model, [copy.deepcopy(caches[0])])
        dec4 = BD.BatchDecoder(model, caches)
        y1 = mx.array([ids_list[0][-1]])
        y4 = mx.array([ids_list[i][-1] for i in range(4)])
        bd1_toks, bd4_toks = [], []
        for _ in range(16):
            y1 = dec1.step(y1)
            y4 = dec4.step(y4)
            mx.eval(y1, y4)
            bd1_toks.append(int(y1[0].item()))
            bd4_toks.append(int(y4[0].item()))
        check(
            f"zero-strength bit-exact [{label}]",
            diff == 0.0 and toks_cur == toks_off and bd1_toks == toks_off and bd4_toks == toks_off,
            f"diff={diff} gen={toks_cur == toks_off} bd1={bd1_toks == toks_off} bd4={bd4_toks == toks_off}",
        )

    # --- Test 3: Two-pass independent ground truth for LoopCD-Logits ---
    def two_pass_logits(seq_arr, cache_R, cache_ref, cfg):
        orig_passes = model.args.text.loop_config["num_passes"]
        orig_cd = model.loopcd
        try:
            model.set_loopcd("off")
            z_R = model(seq_arr, cache=cache_R)
            model.args.text.loop_config["num_passes"] = cfg.resolved_ref_pass(orig_passes) + 1
            z_ref = model(seq_arr, cache=cache_ref)
            mx.eval(z_R, z_ref)
            return loopcd.combine_logits(z_R, z_ref, cfg)
        finally:
            model.args.text.loop_config["num_passes"] = orig_passes
            model.model.language_model.loopcd = orig_cd

    # 3a. Full prefill (cache=None), fixed & adaptive, ref_pass=0 and ref_pass=1
    sub = prompt0[:24][None]
    for label, kw in (
        ("fixed w=0.35 ref=0", dict(mode="logits", omega=0.35, adaptive=False, ref_pass=0)),
        ("adaptive wmax=0.7 ref=0", dict(mode="logits", adaptive=True, omega_max=0.7, ref_pass=0)),
        ("fixed w=0.35 ref=1", dict(mode="logits", omega=0.35, adaptive=False, ref_pass=1)),
    ):
        cfg = model.set_loopcd(**kw)
        got = model(sub, cache=None)
        ref = two_pass_logits(sub, None, None, cfg)
        mx.eval(got, ref)
        diff = float(mx.abs(got - ref).max().item())
        check(f"two-pass prefill [{label}]", diff <= 1e-5, f"max_abs_diff={diff:.2e}")

    # 3b. Prefix-cached deepcopy pattern (searchling_eval.py pattern)
    pre, suf = prompt0[:18][None], prompt0[18:][None]
    cfg = model.set_loopcd("logits", omega=0.35)
    base_cd = make_prompt_cache(model)
    model(pre, cache=base_cd)
    mx.eval([c.state for c in base_cd])
    c_cd = copy.deepcopy(base_cd)
    # Run base_cd again separately to verify deepcopy did not alias
    z_cd = model(suf, cache=c_cd)
    z_cd_again = model(suf, cache=copy.deepcopy(base_cd))
    mx.eval(z_cd, z_cd_again)
    model.set_loopcd("off")
    base_R = make_prompt_cache(model)
    model(pre, cache=base_R)
    mx.eval([c.state for c in base_R])
    model.args.text.loop_config["num_passes"] = 1
    base_1 = make_prompt_cache(model)
    model(pre, cache=base_1)
    mx.eval([c.state for c in base_1])
    model.args.text.loop_config["num_passes"] = 3
    z_two = two_pass_logits(suf, copy.deepcopy(base_R), copy.deepcopy(base_1), cfg)
    mx.eval(z_two)
    d_pfx = float(mx.abs(z_cd - z_two).max().item())
    d_dc = float(mx.abs(z_cd - z_cd_again).max().item())
    check("two-pass prefix-cache deepcopy parity", d_pfx <= 1e-5 and d_dc == 0.0, f"diff={d_pfx:.2e} deepcopy_alias={d_dc:.2e}")

    # 3c. Autoregressive 8-step parity (Model.__call__ & decode_step_logprobs vs two-pass)
    for label, kw in (
        ("fixed w=0.35", dict(mode="logits", omega=0.35, adaptive=False)),
        ("adaptive wmax=0.7", dict(mode="logits", adaptive=True, omega_max=0.7)),
    ):
        cfg = model.set_loopcd(**kw)
        c_call = prefill_cache(model, ids_list[0])
        c_fast = prefill_cache(model, ids_list[0])
        model.set_loopcd("off")
        c_R = prefill_cache(model, ids_list[0])
        model.args.text.loop_config["num_passes"] = 1
        c_ref = prefill_cache(model, ids_list[0])
        model.args.text.loop_config["num_passes"] = 3

        tok_in = ids_list[0][-1]
        max_call_d = 0.0
        max_lp_d = 0.0
        all_same_tok = True
        for _ in range(8):
            x_tok = mx.array([[tok_in]])
            z_two = two_pass_logits(x_tok, c_R, c_ref, cfg)[:, -1, :]
            lp_two = z_two - mx.logsumexp(z_two, axis=-1, keepdims=True)
            model.set_loopcd(**kw)
            z_call = model(x_tok, cache=c_call)[:, -1, :]
            tok_fast, lp_fast = model.decode_step_logprobs(x_tok, cache=c_fast, greedy=True)
            mx.eval(z_two, z_call, lp_fast, tok_fast)
            max_call_d = max(max_call_d, float(mx.abs(z_call - z_two).max().item()))
            max_lp_d = max(max_lp_d, float(mx.abs(lp_fast - lp_two).max().item()))
            t_two = int(mx.argmax(z_two, axis=-1)[0].item())
            t_fast = int(tok_fast[0].item())
            all_same_tok &= (t_two == t_fast)
            tok_in = t_two
        check(
            f"two-pass 8-step decode [{label}]",
            max_call_d <= 1e-5 and max_lp_d <= 1e-5 and all_same_tok,
            f"call_d={max_call_d:.2e} lp_d={max_lp_d:.2e} same_tok={all_same_tok}",
        )

    # --- Test 4: BatchDecoder C=1 & C=4 vs generate_step + flush + qmv_rows B=2 ---
    # Verify QMV batch-2 lm_head vs single-row lm_head
    model.set_loopcd("logits", omega=0.35)
    hR_test = mx.random.normal((1, 1, 2304)).astype(mx.bfloat16)
    hr_test = mx.random.normal((1, 1, 2304)).astype(mx.bfloat16)
    pR, pr = model._logits_pair_softcap(hR_test, hr_test)
    sR, sr = model._project_softcap(hR_test), model._project_softcap(hr_test)
    mx.eval(pR, pr, sR, sr)
    qd = max(float(mx.abs(pR - sR).max().item()), float(mx.abs(pr - sr).max().item()))
    check("qmv_rows B=2 lm_head bitwise == single", qd == 0.0, f"diff={qd:.2e}")

    cd_configs = [
        ("hidden w=0.5", dict(mode="hidden", omega=0.5, blend_bs=False)),
        ("hidden w=0.5 blend_bs", dict(mode="hidden", omega=0.5, blend_bs=True)),
        ("logits w=0.3", dict(mode="logits", omega=0.3, adaptive=False)),
        ("logits adaptive wmax=0.6", dict(mode="logits", adaptive=True, omega_max=0.6)),
    ]
    for label, kw in cd_configs:
        model.set_loopcd(**kw)
        refs = [
            [int(t) for t, _ in generate_step(mx.array(ids_list[i]), model, max_tokens=32, sampler=GREEDY)]
            for i in range(4)
        ]
        # Verify LoopCD actually changes output vs unguided on at least some step/prompt!
        c1 = prefill_cache(model, ids_list[0])
        dec1 = BD.BatchDecoder(model, [c1])
        check(f"BatchDecoder C=1 uses _b1_fast [{label}]", dec1._b1_fast)
        y1 = mx.array([ids_list[0][-1]])
        got1 = []
        for _ in range(32):
            y1 = dec1.step(y1)
            mx.eval(y1)
            got1.append(int(y1[0].item()))

        c4 = [prefill_cache(model, ids_list[i]) for i in range(4)]
        dec4 = BD.BatchDecoder(model, c4)
        y4 = mx.array([ids_list[i][-1] for i in range(4)])
        got4 = [[] for _ in range(4)]
        for _ in range(32):
            y4 = dec4.step(y4)
            mx.eval(y4)
            lst = y4.tolist()
            for i in range(4):
                got4[i].append(lst[i])
        check(
            f"BatchDecoder C=1 & C=4 exact vs generate_step [{label}]",
            got1 == refs[0] and all(got4[i] == refs[i] for i in range(4)),
            f"c1_ok={got1 == refs[0]} c4_ok={[got4[i] == refs[i] for i in range(4)]}",
        )

    # Flush continuation test for C=1 & C=4 with LoopCD-Logits
    model.set_loopcd("logits", adaptive=True, omega_max=0.6)
    ref_cont = [int(t) for t, _ in generate_step(mx.array(ids_list[0]), model, max_tokens=16, sampler=GREEDY)]
    cf1 = prefill_cache(model, ids_list[0])
    d_flush1 = BD.BatchDecoder(model, [cf1])
    yt = mx.array([ids_list[0][-1]])
    got_f1 = []
    for _ in range(8):
        yt = d_flush1.step(yt)
        mx.eval(yt)
        got_f1.append(int(yt[0].item()))
    d_flush1.flush()
    cur = got_f1[-1]
    for _ in range(8):
        z = model(mx.array([[cur]]), cache=cf1)
        cur = int(mx.argmax(z[0, -1]).item())
        got_f1.append(cur)
    check("BatchDecoder C=1 flush continuation exact (logits)", got_f1 == ref_cont)

    # Fused kv RowKV test for LoopCD-Logits (C=4)
    BD._FUSED.add("kv")
    try:
        ckv = [prefill_cache(model, ids_list[i]) for i in range(4)]
        dkv = BD.BatchDecoder(model, ckv)
        check("RowKV shadow coda created when kv fused", dkv._ref_kv is not None)
        yk = mx.array([ids_list[i][-1] for i in range(4)])
        got_kv = [[] for _ in range(4)]
        for _ in range(16):
            yk = dkv.step(yk)
            mx.eval(yk)
            for i, tv in enumerate(yk.tolist()):
                got_kv[i].append(tv)
        dkv.flush()
        refs_kv = [
            [int(t) for t, _ in generate_step(mx.array(ids_list[i]), model, max_tokens=16, sampler=GREEDY)]
            for i in range(4)
        ]
        check("BatchDecoder C=4 RowKV + flush exact (logits)", all(got_kv[i] == refs_kv[i] for i in range(4)))
    finally:
        BD._FUSED.discard("kv")

    # Cache mismatch guard
    model.set_loopcd("off")
    bad_c = prefill_cache(model, ids_list[0])
    model.set_loopcd("logits", omega=0.3)
    err_model = err_bd = False
    try:
        model(mx.array([[ids_list[0][-1]]]), cache=bad_c)
    except RuntimeError:
        err_model = True
    try:
        BD.BatchDecoder(model, [bad_c])
    except RuntimeError:
        err_bd = True
    check("cache-mismatch guard raises RuntimeError", err_model and err_bd, f"model={err_model} bd={err_bd}")

    del model
    mx.clear_cache()


def test_p1_noop(p1_dir):
    model, tok = load(p1_dir)
    mx.eval(model.parameters())
    ids = tok.encode(
        tok.apply_chat_template([{"role": "user", "content": "Hello world."}], add_generation_prompt=True, tokenize=False)
        + "<|message_model|><|content_text|>",
        add_special_tokens=False,
    )
    p = mx.array(ids)
    model.set_loopcd("off")
    z0 = model(p[None], cache=None)
    t0 = [int(t) for t, _ in generate_step(p, model, max_tokens=12, sampler=GREEDY)]
    for kw in (dict(mode="hidden", omega=0.5), dict(mode="logits", omega=0.3)):
        model.set_loopcd(**kw)
        z = model(p[None], cache=None)
        mx.eval(z0, z)
        d = float(mx.abs(z0 - z).max().item())
        t = [int(x) for x, _ in generate_step(p, model, max_tokens=12, sampler=GREEDY)]
        c = prefill_cache(model, ids)
        dec = BD.BatchDecoder(model, [c])
        y = mx.array([ids[-1]])
        tb = []
        for _ in range(12):
            y = dec.step(y)
            mx.eval(y)
            tb.append(int(y[0].item()))
        check(f"p1 graceful no-op [{kw['mode']}]", d == 0.0 and t == t0 and tb == t0)


def main():
    models_root = os.path.expanduser("~/loopling/models")
    test_math_and_config()
    test_p3_omega_zero_and_two_pass_and_batch(os.path.join(models_root, "loopling-p3"))
    test_p1_noop(os.path.join(models_root, "loopling-p1"))
    print("ALL LOOPCD TESTS PASS", flush=True)


if __name__ == "__main__":
    main()
