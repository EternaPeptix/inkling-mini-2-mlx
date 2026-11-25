"""Test periodic (KVCache + ArraysCache) prefix checkpointing: eviction priority, bit-exactness, and TTFT speedup.

Usage:
    RHOST=512S2 scripts/r python3 bench/prefix_checkpoint_test.py
"""
import os
import time

import mlx_lm  # noqa: F401
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, LRUPromptCache, RotatingKVCache


def test_lru_checkpoint_eviction():
    pc = LRUPromptCache(max_size=2, max_bytes=1 << 60, max_checkpoints=3)
    # Non-trimmable Inkling2-like CacheList
    def make_dummy(n):
        kv = KVCache()
        kv.keys = mx.zeros((1, 8, n, 128), mx.bfloat16)
        kv.values = mx.zeros((1, 8, n, 128), mx.bfloat16)
        kv.offset = n
        ac = ArraysCache(4)
        ac[0] = mx.zeros((1, 3, 2304), mx.float32)
        mx.eval(kv.keys, kv.values, ac[0])
        return [CacheList(kv, ac)]

    pc.insert_cache("m", [1, 2, 3], make_dummy(3), cache_type="system")
    pc.insert_cache("m", [1, 2, 3, 4], make_dummy(4), cache_type="user")
    for k in (5, 6, 7, 8):
        pc.insert_cache("m", list(range(1, k + 1)), make_dummy(k), cache_type="checkpoint")

    st = pc.stats_by_type()
    assert st["system"]["n_sequences"] == 1, st
    assert st["user"]["n_sequences"] == 1, st
    assert st["checkpoint"]["n_sequences"] == 3, st  # max_checkpoints=3 evicted only oldest checkpoint (k=5)

    # Byte pressure must evict checkpoints before system/user entries
    sys_bytes = st["system"]["n_bytes"] + st["user"]["n_bytes"]
    keep_bytes = st["checkpoint"]["n_bytes"] // 2
    pc.trim_to(n_bytes=sys_bytes + keep_bytes)
    st2 = pc.stats_by_type()
    assert st2["system"]["n_sequences"] == 1 and st2["user"]["n_sequences"] == 1, st2
    assert st2["checkpoint"]["n_sequences"] == 1, st2

    # Longest common prefix hit across checkpoints on non-trimmable cache
    hit_cache, rest = pc.fetch_nearest_cache("m", list(range(1, 9)) + [999, 998])
    assert hit_cache is not None and rest == [999, 998], f"got rest={rest}"
    print("PASS test_lru_checkpoint_eviction", flush=True)


def run_batch_with_checkpoints(model, prompt_cache, model_key, prompt, max_tokens=32, ckpt_stride=2048):
    bg = BatchGenerator(model, prefill_step_size=2048)
    cache, rest = prompt_cache.fetch_nearest_cache(model_key, prompt)
    hit_len = len(prompt) - len(rest)
    (uid,) = bg.insert_segments(
        segments=[[rest]] if rest else [[[prompt[-1]]]],
        max_tokens=[max_tokens],
        caches=[cache],
        all_tokens=[prompt[:hit_len]] if hit_len > 0 else [[]],
    )
    last_ckpt = hit_len
    gen_tokens = []
    t0 = time.perf_counter()
    ttft = None
    while True:
        prs, grs = bg.next()
        if not prs and not grs:
            break
        ckpt_ids = []
        for r in prs:
            if not r.end_of_prompt:
                processed, total = r.progress
                cur = hit_len + processed
                if ckpt_stride > 0 and cur - last_ckpt >= ckpt_stride and (total - processed) >= 256:
                    ckpt_ids.append(r.uid)
        if ckpt_ids:
            for u, (c, ck) in bg.extract_cache(ckpt_ids).items():
                prompt_cache.insert_cache(model_key, ck[:], c, cache_type="checkpoint")
                last_ckpt = len(ck)
        for r in grs:
            if ttft is None:
                ttft = time.perf_counter() - t0
            gen_tokens.append(r.token)
            if r.finish_reason is not None:
                prompt_cache.insert_cache(model_key, r.all_tokens[:], r.prompt_cache, cache_type="assistant")
                bg.close()
                return hit_len, ttft, gen_tokens
    bg.close()
    return hit_len, ttft, gen_tokens


def main():
    test_lru_checkpoint_eviction()
    from mlx_lm.models.inkling2 import fastdecode as FD
    model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
    mx.eval(model.parameters())

    # 6400-token prompt: cold run saves checkpoints at 2048, 4096, 6144
    prompt_a = [(17 * i * i + 3 * i + 77) % 40000 + 1000 for i in range(6400)]
    # Mutate tail at token 6200..6400 (first 6200 identical -> hits checkpoint at 6144!)
    prompt_b = prompt_a[:6200] + [(31 * i + 123) % 40000 + 1000 for i in range(200)]

    pc_warm = LRUPromptCache(max_size=10, max_bytes=1 << 60, max_checkpoints=32)
    hit_a, ttft_a, toks_a = run_batch_with_checkpoints(model, pc_warm, "p3", prompt_a, max_tokens=32, ckpt_stride=2048)
    print(f"Prompt A (cold 6400): hit={hit_a}, ttft={ttft_a:.3f}s, cache_stats={pc_warm.stats_by_type()}", flush=True)
    assert hit_a == 0

    fd_calls = [0]
    orig_fd = FD.decode_layer
    def counted_fd(*a, **k):
        fd_calls[0] += 1
        return orig_fd(*a, **k)
    FD.decode_layer = counted_fd
    try:
        hit_b, ttft_b, toks_b = run_batch_with_checkpoints(model, pc_warm, "p3", prompt_b, max_tokens=32, ckpt_stride=2048)
    finally:
        FD.decode_layer = orig_fd
    print(f"Prompt B (mutated tail @6200): hit={hit_b}/6400 ({hit_b/6400*100:.1f}%), ttft={ttft_b:.3f}s ({ttft_a/ttft_b:.1f}x faster), fastdecode_calls={fd_calls[0]}", flush=True)
    assert hit_b == 6144, f"expected checkpoint hit at 6144, got {hit_b}"
    assert fd_calls[0] > 0, "expected fastdecode to remain active after checkpoint resume!"

    # Reference cold run of Prompt B without any checkpoints
    pc_cold = LRUPromptCache(max_size=10, max_bytes=1 << 60, max_checkpoints=32)
    hit_ref, ttft_ref, toks_ref = run_batch_with_checkpoints(model, pc_cold, "p3", prompt_b, max_tokens=32, ckpt_stride=0)
    print(f"Prompt B (cold ref): ttft={ttft_ref:.3f}s vs warm={ttft_b:.3f}s ({ttft_ref/ttft_b:.1f}x speedup)", flush=True)
    assert toks_b == toks_ref, f"TOKEN MISMATCH:\n  warm={toks_b}\n  ref ={toks_ref}"
    print(f"PASS prefix_checkpoint bit-exact 32/32 tokens, TTFT speedup={ttft_ref/ttft_b:.2f}x", flush=True)


if __name__ == "__main__":
    main()
