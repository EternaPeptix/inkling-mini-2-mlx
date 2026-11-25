"""Lane A2 tests for the fast decode path (vendor/mlx_lm/models/inkling2/fastdecode.py). usage: python bench/a2_tests.py conv|tau|weights|server|all
Each check prints PASS/FAIL; exit code 1 on any FAIL. The reference is L._FAST_ON = False in the same process."""
import copy, math, os, sys, types
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.inkling2 import layers as L, fastdecode as F

FAILS = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra, flush=True)
    if not ok:
        FAILS.append(name)


def t_conv():
    rng = np.random.default_rng(0)
    C = 4096
    mod = L.ShortConvolution(C, 4)
    mod.conv1d.weight = mx.array(rng.standard_normal((C, 4, 1)).astype(np.float32)).astype(mx.bfloat16)
    bad = 0
    nonbf = 0
    for rep in range(8):
        st = mx.array(rng.standard_normal((1, 3, C)).astype(np.float32) * 3)  # f32 values that are NOT bf16-representable
        nonbf += int((st != st.astype(mx.bfloat16).astype(mx.float32)).sum().item())
        x = mx.array(rng.standard_normal((1, 1, C)).astype(np.float32)).astype(mx.bfloat16)
        ro, rw = mod(x, st, None)
        fo, fs = F.short_conv_decode(mod, x, st)
        mx.eval(ro, rw, fo, fs)
        bad += int((ro != fo).sum().item()) + int((rw[:, 1:, :] != fs).sum().item())
    check("conv: non-bf16 f32 state, fused == reference bitwise", bad == 0 and nonbf > 0, f"mismatches {bad} (non-bf16 state elements {nonbf})")
    # kernel_size 3: must fall back, not assert
    m3 = L.ShortConvolution(C, 3)
    m3.conv1d.weight = mx.array(rng.standard_normal((C, 3, 1)).astype(np.float32)).astype(mx.bfloat16)
    st = mx.array(rng.standard_normal((1, 3, C)).astype(np.float32))
    x = mx.array(rng.standard_normal((1, 1, C)).astype(np.float32)).astype(mx.bfloat16)
    try:
        o, w = F._sc(m3, x, st)
        ro, rw = m3(x, st, None)
        mx.eval(o, w, ro, rw)
        check("conv: kernel_size=3 falls back to reference", (not F._conv_ok(m3, x)) and mx.array_equal(o, ro).item() and mx.array_equal(w, rw[:, rw.shape[1] - 3:, :]).item())
    except AssertionError:
        check("conv: kernel_size=3 falls back to reference", False, "AssertionError")


def t_tau():
    def lay(alpha, nf):
        return types.SimpleNamespace(self_attn=types.SimpleNamespace(log_scaling_alpha=alpha, log_scaling_n_floor=nf))
    F._tau.clear()
    A, B = lay(0.5, 100), lay(2.0, 300)
    ta, tb = F._full_tau(A, 5000), F._full_tau(B, 5000)
    ea = 1.0 + 0.5 * math.log(max(5001 / 100, 1.0)); eb = 1.0 + 2.0 * math.log(max(5001 / 300, 1.0))
    ok = abs(ta.item() - ea) < 1e-5 and abs(tb.item() - eb) < 1e-5 and abs(ta.item() - tb.item()) > 0.1
    check("tau: cache keyed by (offset, alpha, n_floor)", ok, f"{ta.item():.4f} vs {ea:.4f}, {tb.item():.4f} vs {eb:.4f}")


def load(p):
    from mlx_lm import load as ld
    model, tok = ld(os.path.expanduser("~/loopling/models/loopling-p%d" % p))
    mx.eval(model.parameters())
    return model, tok


def t_weights():
    from mlx_lm.models.cache import make_prompt_cache
    model, tok = load(3)
    tm = model.model.language_model
    ids = [(17 * i * i + 3 * i) % 40000 + 1000 for i in range(120)]
    c = make_prompt_cache(model)
    L._FAST_ON = False
    lg = model(mx.array([ids]), cache=c)
    mx.eval([x.state for x in c])
    t = int(mx.argmax(lg[0, -1]).item())
    cr, cf = copy.deepcopy(c), copy.deepcopy(c)

    def step(cache, tok_, fast):
        L._FAST_ON = fast
        try:
            out = model(mx.array([[tok_]]), cache=cache)[0, -1]
            mx.eval(out)
        finally:
            L._FAST_ON = True
        return out

    def both(tag, n=3):
        nonlocal t
        worst = 0.0
        eq = True
        tt = t
        for _ in range(n):
            lr, lf = step(cr, tt, False), step(cf, tt, True)
            eq &= mx.array_equal(lr, lf).item()
            worst = max(worst, mx.abs(lr - lf).max().item())
            tt = int(mx.argmax(lr).item())
        t = tt
        check(tag, eq, f"max|dlogit| {worst}")
        return lr

    L._FAST_ON = True
    base = both("weights: baseline fast == reference (first decode traces the closures)")
    calls = {"n": 0}
    orig = F.decode_layer
    F.decode_layer = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
    # (a) direct attribute assignment
    for l in tm.layers:
        l.post_attention_layernorm.weight = l.post_attention_layernorm.weight * 1.5
    la = both("weights: direct attribute assignment (norm weight x1.5)")
    # (b) update() on a quantized projection + conv weights (adapter-fuse style) and (c) load_weights
    for l in tm.layers:
        a = l.self_attn
        l.self_attn.q_proj.update({"scales": a.q_proj.scales * 1.25})
        l.update({"post_mlp_convolution": {"conv1d": {"weight": (l.post_mlp_convolution.conv1d.weight.astype(mx.float32) * 1.1).astype(mx.bfloat16)}}})
    lb = both("weights: module.update (q_proj.scales x1.25, conv weight x1.1)")
    items = [(f"model.language_model.layers.{i}.self_attn.o_proj.scales", l.self_attn.o_proj.scales * 0.8) for i, l in enumerate(tm.layers)]
    try:
        model.load_weights(items, strict=False)
    except Exception as e:
        print("load_weights path raised", type(e).__name__, e)
        items = None
    lc = both("weights: model.load_weights (o_proj.scales x0.8)")
    check("weights: each reassignment changed the output (not vacuous)", not mx.array_equal(base, la).item() and not mx.array_equal(la, lb).item() and not mx.array_equal(lb, lc).item())
    F.decode_layer = orig
    check("weights: fast path was used", calls["n"] > 0, f"decode_layer calls {calls['n']}")


def t_server():
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.inkling2.batchdecode import BatchDecoder
    model, tok = load(3)
    orig = F.decode_layer
    orig_b1 = BatchDecoder._layer_b1
    calls = {"n": 0, "bd": 0}
    F.decode_layer = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
    BatchDecoder._layer_b1 = lambda self, *a, **k: (calls.__setitem__("bd", calls["bd"] + 1), orig_b1(self, *a, **k))[1]

    def run(prompt, fast, caches=None, n=24):
        L._FAST_ON = fast
        calls["n"] = 0
        calls["bd"] = 0
        g = BatchGenerator(model, max_tokens=n, prefill_step_size=512)
        g.insert([prompt], max_tokens=[n], caches=caches)
        toks, lps = [], []
        while True:
            rs = g.next_generated()
            for r in rs:
                toks.append(r.token); lps.append(r.logprobs); mx.eval(r.logprobs)
            if rs and rs[0].finish_reason is not None:
                break
        g.close()
        L._FAST_ON = True
        return toks, lps, calls["n"] + calls["bd"]

    for ctx, tag in ((200, "fresh B=1 request, ctx 200"), (1100, "fresh B=1 request, ctx 1100 (rotating window wraps)")):
        ids = [(17 * i * i + 3 * i) % 40000 + 1000 for i in range(ctx)]
        a, la, na = run(ids, False)
        b, lb, nb = run(ids, True)
        same = a == b and all(mx.array_equal(x, y).item() for x, y in zip(la, lb))
        check("server: " + tag + ": tokens + logprobs identical", same, f"n={len(a)}")
        check("server: " + tag + ": fast path taken", na == 0 and nb > 0, f"fast calls ref {na} fast {nb}")
    # prompt-cache hit: merge of non-empty caches
    ids = [(17 * i * i + 3 * i) % 40000 + 1000 for i in range(1100)]

    def cached():
        c = make_prompt_cache(model)
        L._FAST_ON = False
        model(mx.array([ids[:1000]]), cache=c)
        mx.eval([x.state for x in c])
        L._FAST_ON = True
        return c

    a, la, na = run(ids[1000:], False, caches=[cached()])
    b, lb, nb = run(ids[1000:], True, caches=[cached()])
    same = a == b and all(mx.array_equal(x, y).item() for x, y in zip(la, lb))
    check("server: prompt-cache hit B=1: tokens + logprobs identical", same, f"n={len(a)}")
    check("server: prompt-cache hit B=1: fast path taken", na == 0 and nb > 0, f"calls ref {na} fast {nb}")
    # two concurrent requests must NOT take the FD.decode_layer B=1 fast path (and stay correct)
    L._FAST_ON = True
    calls["n"] = 0
    calls["bd"] = 0
    g = BatchGenerator(model, max_tokens=8, prefill_step_size=512)
    g.insert([ids[:100], ids[:90]], max_tokens=[8, 8])
    out = []
    while len(out) < 16:
        out += g.next_generated()
    g.close()
    check("server: B=2 never takes the fast path", calls["n"] == 0, f"calls {calls['n']}")
    F.decode_layer = orig
    BatchDecoder._layer_b1 = orig_b1


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("conv", t_conv), ("tau", t_tau), ("weights", t_weights), ("server", t_server)):
        if which in ("all", name):
            fn()
    print("FAILS:", FAILS)
    sys.exit(1 if FAILS else 0)
