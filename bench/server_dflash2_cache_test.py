"""Verify Splash 1.7.2 adaptations on loopling-p3:
1. Multi-turn ResponseGenerator (C=1, LOOPLING_MTP=dflash2) cache sync & bit-exactness across:
   - Turn 1 (>1024 tokens, crossing RotatingKVCache window, max_tokens stop)
   - Turn 2 (resuming Turn 1 reply, stop_words early exit)
   - Turn 3 (resuming Turn 2 reply, EOS exit)
   - Turn 4 (mutated system tail hitting protected cache_type='system' checkpoint)
2. Persistent SSD LRUPromptCache round-trip (flush_disk -> fresh cache reload -> Turn 2 bit-exactness).

Usage:
    RHOST=512S2 LANE=opt ./scripts/r --gpu 'LOOPLING_MTP=dflash2 python3 bench/server_dflash2_cache_test.py'
"""
import os
import shutil
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

import mlx_lm  # noqa: F401
import mlx.core as mx

from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.server import (
    CompletionRequest,
    GenerationArguments,
    LogitsProcessorArguments,
    ModelDescription,
    ModelProvider,
    ResponseGenerator,
    SamplingArguments,
)


def make_cli_args(model_path: str, persistent_cache: str = ""):
    return SimpleNamespace(
        model=model_path,
        adapter_path=None,
        draft_model=None,
        trust_remote_code=False,
        chat_template=None,
        chat_template_args={},
        use_default_chat_template=False,
        pipeline=False,
        decode_concurrency=8,
        prompt_concurrency=8,
        prefill_step_size=2048,
        prompt_cache_size=16,
        prompt_cache_bytes=16 << 30,
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=0,
        persistent_cache=persistent_cache,
        persistent_cache_gb=8.0,
    )


def make_gen_args(max_tokens: int, stop_words=None):
    return GenerationArguments(
        model=ModelDescription(model="default_model", draft=None, adapter=None),
        sampling=SamplingArguments(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            xtc_probability=0.0,
            xtc_threshold=0.0,
        ),
        logits=LogitsProcessorArguments(
            logit_bias=None,
            repetition_penalty=0.0,
            repetition_context_size=20,
            presence_penalty=0.0,
            presence_context_size=20,
            frequency_penalty=0.0,
            frequency_context_size=20,
        ),
        stop_words=stop_words or [],
        max_tokens=max_tokens,
        num_draft_tokens=3,
        logprobs=False,
        top_logprobs=0,
        seed=None,
        chat_template_kwargs=None,
    )


def run_chat(rg: ResponseGenerator, messages, max_tokens: int, stop_words=None):
    req = CompletionRequest(request_type="chat", prompt="", messages=messages, tools=None, role_mapping=None)
    args = make_gen_args(max_tokens=max_tokens, stop_words=stop_words)
    t0 = time.perf_counter()
    t_first = None
    tokens = []
    texts = []
    finish_reason = None
    prompt_toks = 0
    ctx, it = rg.generate(req, args)
    for r in it:
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        if r.token is not None:
            tokens.append(int(r.token))
        if r.text:
            texts.append(r.text)
        if r.finish_reason is not None:
            finish_reason = r.finish_reason
    dt_total = time.perf_counter() - t0
    dt_dec = max(1e-6, time.perf_counter() - (t_first or t0))
    prompt_toks = len(getattr(ctx, "prompt", []) or [])
    text = "".join(texts)
    return {
        "tokens": tokens,
        "text": text,
        "finish_reason": finish_reason,
        "ttft_ms": ((t_first or time.perf_counter()) - t0) * 1000.0,
        "tps": max(0, len(tokens) - 1) / dt_dec if len(tokens) > 1 else len(tokens) / dt_total,
        "prompt_toks": prompt_toks,
        "cached_toks": max(0, getattr(ctx, "prompt_cache_count", 0) or 0),
    }


def count_mismatches(a, b):
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def build_long_system_prompt(suffix: str) -> str:
    rules = "\n".join(
        f"Rule {i:02d}: Always validate inputs cleanly, avoid unnecessary allocations, prefer clear naming, "
        f"document boundary conditions, and verify complexity bounds when analyzing algorithms ({i * 7 % 97})."
        for i in range(1, 75)
    )
    return (
        "You are a precise systems engineering assistant specialized in Python and Apple Silicon performance.\n"
        + rules
        + f"\nSession Tag: {suffix}\n"
    )


def main():
    model_path = os.path.expanduser("~/loopling/models/loopling-p3")
    sys_a = build_long_system_prompt("ALPHA-001")
    sys_b = build_long_system_prompt("BETA-999")

    t1_msgs = [
        {"role": "system", "content": sys_a},
        {"role": "user", "content": "Write a concise Python binary search function with edge case comments, then explain time complexity."},
    ]
    t2_msgs_tpl = lambda r1: t1_msgs + [
        {"role": "assistant", "content": r1},
        {"role": "user", "content": "Output immediately without preamble: Step 1, Step 2, Step 3, Step 4."},
    ]
    t3_msgs_tpl = lambda r1, r2: t2_msgs_tpl(r1) + [
        {"role": "assistant", "content": r2},
        {"role": "user", "content": "Say OK and stop."},
    ]
    t4_sys_mutated = [
        {"role": "system", "content": sys_b},
        {"role": "user", "content": "Write a concise two-line Python helper `clamp(x, lo, hi)`."},
    ]

    cli_args = make_cli_args(model_path)
    mp = ModelProvider(cli_args)

    os.environ["LOOPLING_MTP"] = "0"
    os.environ["LOOPLING_SERVER_BATCHDECODE"] = "0"
    ar_cache = LRUPromptCache(max_size=16, max_bytes=16 << 30)
    rg = ResponseGenerator(mp, ar_cache)
    tmp_disk = tempfile.mkdtemp(prefix="loopling_ssd_test_")
    mm_ssd = 0
    try:
        # 1. Cold AR baseline (LOOPLING_MTP=0)
        print("=== Running AR Baseline (LOOPLING_MTP=0) ===")
        ar1 = run_chat(rg, t1_msgs, max_tokens=140)
        ar2 = run_chat(rg, t2_msgs_tpl(ar1["text"]), max_tokens=120, stop_words=["Step 3"])
        ar3 = run_chat(rg, t3_msgs_tpl(ar1["text"], ar2["text"]), max_tokens=64)
        ar4 = run_chat(rg, t4_sys_mutated, max_tokens=48)
        print(f"AR T1: tok={len(ar1['tokens'])} finish={ar1['finish_reason']} tps={ar1['tps']:.2f} ttft={ar1['ttft_ms']:.1f}ms prompt={ar1['prompt_toks']}")
        print(f"AR T2: tok={len(ar2['tokens'])} finish={ar2['finish_reason']} tps={ar2['tps']:.2f} ttft={ar2['ttft_ms']:.1f}ms cached={ar2['cached_toks']}")
        print(f"AR T3: tok={len(ar3['tokens'])} finish={ar3['finish_reason']} tps={ar3['tps']:.2f} ttft={ar3['ttft_ms']:.1f}ms cached={ar3['cached_toks']}")
        print(f"AR T4: tok={len(ar4['tokens'])} finish={ar4['finish_reason']} tps={ar4['tps']:.2f} ttft={ar4['ttft_ms']:.1f}ms cached={ar4['cached_toks']}")

        # 2. DFlash2 multi-turn + protected system-head checkpoint test
        os.environ["LOOPLING_MTP"] = "dflash2"
        os.environ["LOOPLING_DFLASH2_QUANT"] = "4"
        os.environ["LOOPLING_DFLASH2_W"] = "8"
        os.environ["LOOPLING_DFLASH2_TREE"] = "1"
        os.environ["LOOPLING_DFLASH2_BUDGET"] = "1"
        os.environ["LOOPLING_DFLASH2_LOOKUP"] = "1"

        df2_cache = LRUPromptCache(max_size=16, max_bytes=16 << 30)
        rg.prompt_cache = df2_cache
        print("\n=== Running DFlash2 C=1 Multi-Turn + System-Head Checkpoint ===")
        _ = run_chat(rg, [{"role": "user", "content": "Write a 3-line Python quicksort."}], max_tokens=48)
        df1 = run_chat(rg, t1_msgs, max_tokens=140)
        st1 = dict(getattr(mp.model, "dflash2_stats", {}))
        df2 = run_chat(rg, t2_msgs_tpl(df1["text"]), max_tokens=120, stop_words=["Step 3"])
        st2 = dict(getattr(mp.model, "dflash2_stats", {}))
        df3 = run_chat(rg, t3_msgs_tpl(df1["text"], df2["text"]), max_tokens=64)
        st3 = dict(getattr(mp.model, "dflash2_stats", {}))
        df4 = run_chat(rg, t4_sys_mutated, max_tokens=48)
        st4 = dict(getattr(mp.model, "dflash2_stats", {}))

        mm1 = count_mismatches(df1["tokens"], ar1["tokens"])
        mm2 = count_mismatches(df2["tokens"], ar2["tokens"])
        mm3 = count_mismatches(df3["tokens"], ar3["tokens"])
        mm4 = count_mismatches(df4["tokens"], ar4["tokens"])
        sys_stats = df2_cache.stats_by_type()
        print(f"DF2 T1 (max_tokens): tok={len(df1['tokens'])} mm={mm1} tps={df1['tps']:.2f} ttft={df1['ttft_ms']:.1f}ms prompt={df1['prompt_toks']} stats={sys_stats} df2={st1}")
        print(f"DF2 T2 (stop_word):  tok={len(df2['tokens'])} finish={df2['finish_reason']} mm={mm2} tps={df2['tps']:.2f} ttft={df2['ttft_ms']:.1f}ms cached={df2['cached_toks']}/{df2['prompt_toks']} df2={st2}")
        print(f"DF2 T3 (EOS resume): tok={len(df3['tokens'])} finish={df3['finish_reason']} mm={mm3} tps={df3['tps']:.2f} ttft={df3['ttft_ms']:.1f}ms cached={df3['cached_toks']}/{df3['prompt_toks']} df2={st3}")
        print(f"DF2 T4 (sys-head):   tok={len(df4['tokens'])} mm={mm4} tps={df4['tps']:.2f} ttft={df4['ttft_ms']:.1f}ms cached={df4['cached_toks']}/{df4['prompt_toks']} df2={st4}")

        # 3. Persistent SSD LRUPromptCache round-trip
        print(f"\n=== Running Persistent SSD Cache Round-Trip ({tmp_disk}) ===")
        ssd_cache1 = LRUPromptCache(max_size=16, max_bytes=16 << 30, disk_dir=tmp_disk, disk_max_bytes=4 << 30)
        rg.prompt_cache = ssd_cache1
        s1 = run_chat(rg, t1_msgs, max_tokens=140)
        mm_s1 = count_mismatches(s1["tokens"], ar1["tokens"])
        ssd_cache1.flush_disk()

        disk_files = os.listdir(tmp_disk)
        print(f"Disk files persisted: {len(disk_files)} files (e.g. {disk_files[:4]})")

        ssd_cache2 = LRUPromptCache(max_size=16, max_bytes=16 << 30, disk_dir=tmp_disk, disk_max_bytes=4 << 30)
        rg.prompt_cache = ssd_cache2
        s2 = run_chat(rg, t2_msgs_tpl(s1["text"]), max_tokens=120, stop_words=["Step 3"])
        mm_s2 = count_mismatches(s2["tokens"], ar2["tokens"])
        print(f"SSD Reloaded T2: mm_s1={mm_s1} mm_s2={mm_s2} tps={s2['tps']:.2f} ttft={s2['ttft_ms']:.1f}ms cached={s2['cached_toks']}/{s2['prompt_toks']}")
        mm_ssd = mm_s1 + mm_s2
        if not disk_files:
            print("FAIL: no SSD cache files written", file=sys.stderr)
            sys.exit(1)
        if s2["cached_toks"] <= 0:
            print(f"FAIL: expected SSD cache hit (cached_toks > 0), got {s2['cached_toks']}", file=sys.stderr)
            sys.exit(1)
        if df4["cached_toks"] <= 0:
            print(f"FAIL: expected system-head cache hit on DF2 T4 (cached_toks > 0), got {df4['cached_toks']}", file=sys.stderr)
            sys.exit(1)
    finally:
        rg.stop_and_join()
        shutil.rmtree(tmp_disk, ignore_errors=True)

    total_mm = mm1 + mm2 + mm3 + mm4 + mm_ssd
    if sys_stats.get("system", {}).get("n_sequences", 0) < 1:
        print(f"FAIL: expected protected cache_type='system' entries >= 1, got {sys_stats}", file=sys.stderr)
        sys.exit(1)
    if total_mm:
        print(f"\nFAIL: total_mismatches={total_mm}", file=sys.stderr)
        sys.exit(1)
    print("\nALL SERVER + DFLASH2 + SYSTEM-HEAD + SSD CACHE CHECKS PASSED (0 mismatches).")


if __name__ == "__main__":
    main()
