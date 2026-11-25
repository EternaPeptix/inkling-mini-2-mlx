"""Test opt-in INT8 KV cache (LOOPLING_KV_INT8=1): kernel exactness, memory reduction, quality, and long-context decode speed.

Usage:
    RHOST=512S2 scripts/r python3 bench/kv_int8_test.py
"""
import os
import statistics
import time

import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import Int8KVCache, KVCache
from mlx_lm.models.inkling2.rows import sdpa as RS


def test_sdpa_int8_kernel():
    mx.random.seed(11)
    for S in (17, 1024, 4096):
        q = mx.random.normal((1, 24, 128), mx.bfloat16)
        k = mx.random.normal((1, 8, S, 128), mx.bfloat16)
        v = mx.random.normal((1, 8, S, 128), mx.bfloat16)
        pb = mx.random.normal((1, 24, 1024), mx.float32) * 0.1
        qkv = mx.random.normal((4096,), mx.bfloat16)
        gate_off, sink_off = 0, 3072
        kv_bf = KVCache()
        kv_bf.update_and_fetch(k, v)
        kv_bf.pack_b1()

        kv_q8 = Int8KVCache()
        kv_q8.update_and_fetch(k, v)
        mx.eval(kv_bf._packed, kv_q8.k_q, kv_q8.v_q, kv_q8.k_s, kv_q8.v_s)

        out_bf = RS.sdpa_b1_packed(
            q, kv_bf._packed, pb, qkv, gate_off, sink_off, S, S, 0, 0, kv_bf._packed.shape[2], scale=128**-0.5, window=0
        )
        out_q8 = RS.sdpa_b1_int8(
            q, kv_q8.k_q, kv_q8.v_q, kv_q8.k_s, kv_q8.v_s, pb, qkv, gate_off, sink_off, S, S, kv_q8.k_q.shape[2], scale=128**-0.5
        )
        mx.eval(out_bf, out_q8)
        diff = mx.abs(out_bf.astype(mx.float32) - out_q8.astype(mx.float32))
        print(f"kernel sdpa_b1_int8 S={S:4d}: max_diff={diff.max().item():.5f}, mean_diff={diff.mean().item():.6f}", flush=True)
        assert diff.max().item() < 0.08 and diff.mean().item() < 0.01, f"diff too large at S={S}"

    # Verify quantize_kv_b1_int8 matches Batch/Prefill quantizer
    k1 = mx.random.normal((1, 8, 128), mx.bfloat16)
    v1 = mx.random.normal((1, 8, 128), mx.bfloat16)
    kq, vq, ks, vs = RS.quantize_kv_b1_int8(k1, v1)
    kv_q8_1 = Int8KVCache()
    kv_q8_1.update_and_fetch(k1[:, :, None, :], v1[:, :, None, :])
    mx.eval(kq, vq, ks, vs, kv_q8_1.k_q, kv_q8_1.v_q, kv_q8_1.k_s, kv_q8_1.v_s)
    assert mx.max(mx.abs(kq.astype(mx.int32) - kv_q8_1.k_q[:, :, :1, :].astype(mx.int32))).item() <= 1
    assert mx.max(mx.abs(vq.astype(mx.int32) - kv_q8_1.v_q[:, :, :1, :].astype(mx.int32))).item() <= 1
    print("PASS test_sdpa_int8_kernel", flush=True)


def run_prompt(model, ids, *, int8: bool, max_tokens: int):
    os.environ["LOOPLING_KV_INT8"] = "1" if int8 else "0"
    cache = model.make_cache()
    p = mx.array(ids)
    # Chunked prefill
    t0 = time.perf_counter()
    prompt_cache = cache
    toks = []
    logits_list = []
    # Keep first logit too via generate_step
    for step, (tok_id, logprobs) in zip(
        range(max_tokens),
        generate_step(p, model, max_tokens=max_tokens, prompt_cache=prompt_cache, prefill_step_size=2048),
    ):
        if step == 0:
            ttft = time.perf_counter() - t0
            t_dec0 = time.perf_counter()
        toks.append(int(tok_id))
        if step == 0:
            logits_list.append(logprobs)
    dt_dec = time.perf_counter() - t_dec0
    full_bytes = sum(c[0].nbytes for c in prompt_cache if isinstance(c[0], (KVCache, Int8KVCache)))
    del prompt_cache
    mx.clear_cache()
    return toks, full_bytes, ttft, (max_tokens - 1) / dt_dec


def main():
    test_sdpa_int8_kernel()
    model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
    mx.eval(model.parameters())

    PROMPTS = [
        "Write a Python function that parses an ISO-8601 duration like P3DT4H12M into total seconds, with tests.",
        "Write a short story, about 300 words, about a lighthouse keeper who receives a letter from the future.",
        "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the odd numbers below 1000.",
        "Explain how a B-tree insertion works, step by step, with an example of a node split.",
    ]

    print("\n=== Quality Check (128 greedy tokens on P0..P3: INT8 vs BF16) ===", flush=True)
    exact_matches = 0
    top1_total = 0
    for i, text in enumerate(PROMPTS):
        chat = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        ids = tok.encode(chat + "<|message_model|><|content_text|>", add_special_tokens=False)
        t_bf, b_bf, _, _ = run_prompt(model, ids, int8=False, max_tokens=128)
        t_q8, b_q8, _, _ = run_prompt(model, ids, int8=True, max_tokens=128)
        if t_bf[0] == t_q8[0]:
            top1_total += 1
        eq = sum(a == b for a, b in zip(t_bf, t_q8))
        exact_matches += (t_bf == t_q8)
        print(f"P{i}: next_tok={t_bf[0]}vs{t_q8[0]}, prefix_match={eq}/128, full_kv_bytes bf16={b_bf/1e6:.2f}MB vs int8={b_q8/1e6:.2f}MB", flush=True)
    print(f"Summary P0..P3: step0_top1={top1_total}/{len(PROMPTS)}, bit-exact_128tok={exact_matches}/{len(PROMPTS)}", flush=True)

    print("\n=== Long-Context Memory & Decode Benchmark (p3, 64 tokens) ===", flush=True)
    for L in (1024, 8192, 16384):
        ids = [(17 * j * j + 3 * j + 91) % 40000 + 1000 for j in range(L)]
        _, b_bf, tt_bf, tps_bf = run_prompt(model, ids, int8=False, max_tokens=64)
        _, b_q8, tt_q8, tps_q8 = run_prompt(model, ids, int8=True, max_tokens=64)
        # Second warmup run for stable decode speed
        t_bf, _, _, tps_bf2 = run_prompt(model, ids, int8=False, max_tokens=64)
        t_q8, _, _, tps_q82 = run_prompt(model, ids, int8=True, max_tokens=64)
        match = sum(a == b for a, b in zip(t_bf, t_q8))
        print(
            f"L={L:5d}: full_kv bf16={b_bf/1e6:6.1f}MB vs int8={b_q8/1e6:6.1f}MB ({b_bf/b_q8:.2f}x smaller) | "
            f"decode bf16={tps_bf2:6.2f} tok/s vs int8={tps_q82:6.2f} tok/s ({(tps_q82/tps_bf2-1)*100:+.1f}%) | "
            f"agree={match}/64, next_tok={t_bf[0]}=={t_q8[0]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
