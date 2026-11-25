"""Prefill bit-exactness gate (Step 6.1).

Compares reference prefill (LOOPLING_FAST_PREFILL=0, LOOPLING_FAST_REL_ATTN=0) against
optimized prefill across all 48 visits' K (uint16), V (uint16), 4 conv states (uint32),
and final chunk logits (uint32) on loopling-p3. 0 bit mismatches required.

Usage:
    python bench/prefill_exact_test.py [--lens 17,129,513,2048,4096,16384] [--step 2048]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time


def _make_ids(n, seed=7):
    return [(17 * i * i + 3 * i + 11 * seed) % 40000 + 1000 for i in range(n)]


def _dump_or_check(mode, lens, step, tmpdir, model_path):
    import mlx_lm  # noqa: F401  (before mlx.core)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = load(os.path.expanduser(model_path))
    mx.eval(model.parameters())

    def collect(n):
        ids = _make_ids(n)
        c = make_prompt_cache(model)
        logits = None
        for s in range(0, n, step):
            logits = model(mx.array([ids[s : s + step]]), cache=c)
            mx.eval([x.state for x in c], logits)
        tensors = {"logits": logits}
        for vi, cl in enumerate(c):
            k, v = cl[0].keys_and_values()
            tensors[f"k_{vi}"] = k
            tensors[f"v_{vi}"] = v
            tensors[f"c_{vi}"] = mx.concatenate([cl[1][j] for j in range(4)], axis=-1)
        mx.eval(list(tensors.values()))
        return tensors, len(c)

    if mode == "ref":
        for L in lens:
            tensors, nv = collect(L)
            p = os.path.join(tmpdir, f"ref_{L}.safetensors")
            mx.save_safetensors(p, tensors)
            print(f"ref L={L:5d} saved ({nv} visits)", flush=True)
        return 0

    tot_bad = 0
    for L in lens:
        t0 = time.perf_counter()
        got, nv = collect(L)
        dt = time.perf_counter() - t0
        p = os.path.join(tmpdir, f"ref_{L}.safetensors")
        ref = mx.load(p)
        k_bad = v_bad = c_bad = l_bad = 0
        k_tot = v_tot = c_tot = l_tot = 0
        for vi in range(nv):
            for tag, dt_view, kind in (("k", mx.uint16, "K"), ("v", mx.uint16, "V"), ("c", mx.uint32, "conv")):
                key = f"{tag}_{vi}"
                g, r = got[key], ref[key]
                assert g.shape == r.shape and g.dtype == r.dtype, f"L={L} {key}: {g.shape}/{g.dtype} vs {r.shape}/{r.dtype}"
                m = int(mx.sum(g.view(dt_view) != r.view(dt_view)).item())
                if tag == "k":
                    k_tot += g.size
                    k_bad += m
                elif tag == "v":
                    v_tot += g.size
                    v_bad += m
                else:
                    c_tot += g.size
                    c_bad += m
        lg, lr = got["logits"], ref["logits"]
        assert lg.shape == lr.shape and lg.dtype == lr.dtype, f"L={L} logits: {lg.shape}/{lg.dtype} vs {lr.shape}/{lr.dtype}"
        l_tot = lg.size
        l_bad = int(mx.sum(lg.view(mx.uint32) != lr.view(mx.uint32)).item())
        bad = k_bad + v_bad + c_bad + l_bad
        tot_bad += bad
        status = "PASS" if bad == 0 else "FAIL"
        print(
            f"{status} L={L:5d} ({L / dt:6.1f} tok/s): K {k_bad}/{k_tot}, V {v_bad}/{v_tot}, "
            f"conv {c_bad}/{c_tot}, logits {l_bad}/{l_tot}",
            flush=True,
        )
        os.remove(p)
    return 0 if tot_bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=("ref", "opt"), default=None)
    ap.add_argument("--tmpdir", default=None)
    ap.add_argument("--lens", default="17,129,513,2048,4096,16384")
    ap.add_argument("--step", type=int, default=2048)
    ap.add_argument("--model", default="~/loopling/models/loopling-p3")
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",") if x]

    if a.worker is not None:
        sys.exit(_dump_or_check(a.worker, lens, a.step, a.tmpdir, a.model))

    with tempfile.TemporaryDirectory(prefix="loopling_prefill_exact_") as td:
        cmd_base = [sys.executable, __file__, "--tmpdir", td, "--lens", a.lens, "--step", str(a.step), "--model", a.model]
        ref_env = dict(os.environ, LOOPLING_FAST_PREFILL="0", LOOPLING_FAST_REL_ATTN="0")
        opt_env = dict(os.environ)
        opt_env.setdefault("LOOPLING_FAST_PREFILL", "1")
        opt_env.setdefault("LOOPLING_FAST_REL_ATTN", "1")
        subprocess.run(cmd_base + ["--worker", "ref"], env=ref_env, check=True)
        res = subprocess.run(cmd_base + ["--worker", "opt"], env=opt_env)
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
