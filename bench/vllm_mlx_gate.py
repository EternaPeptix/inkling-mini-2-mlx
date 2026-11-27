#!/usr/bin/env python3
"""Bit-exactness verification & benchmark gate harness for vllm-mlx + loopling-opt (Inkling 2).

Gates:
  t1  Pure-CPU unit & parity tests (reasoning parser, tool parser, chat template safety,
      optional FastByteLevelDetokenizer). Runs fast without GPU.
  t2  In-process vllm-mlx Scheduler + CacheList + SSDCacheTier bit-exactness on 512S2 (--gpu):
      greedy & seeded sampling parity, RAM prefix-cache HIT & 1-tok immutability,
      SSD spill/reload bitwise CacheList (K/V uint16, 4x conv uint32, rotating _idx/offset)
      + continuation parity, need_logprobs=[False] parity, C=1 vs C=4 (L>=64) parity,
      T2.6 concurrent prefill-vs-decode fairness, T2.7 multi-branch prefix retention,
      and T2.8 128K-132K DFlash2 boundary crossing (_KV_STACK_MAX_CAP=131072).
  t3  Core loopling-opt regressions on 512S2 (--gpu): a2_tests (live-weight mutability),
      prefill_exact_test (--step 4096), server_batch_test (--C 1,2,4,8), optional spec_exact_test.
  t4  Live HTTP /v1/chat/completions golden & perf gate: streaming vs non-streaming exact parity,
      zero leaked <|...|> control tokens, multi-turn >1024-tok rotating-window prefix HIT,
      C=4 (short + L>=64) vs C=1 parity, SHA-256 verification against golden baseline,
      TTFT / dec_tps / p50 / p95 / max_gap_ms / C=4 agg tok/s delta report, plus optional
      --cold-start (Req #1 warm-tier + zero Task-destroyed leak check), --fairness,
      --multibranch, and --disconnect-check.

Usage:
  RHOST=512S2 LANE=verif scripts/r 'python3 bench/vllm_mlx_gate.py t1'
  RHOST=512S2 LANE=verif scripts/r --gpu 'python3 bench/vllm_mlx_gate.py t2'
  RHOST=512S2 LANE=verif scripts/r --gpu 'python3 bench/vllm_mlx_gate.py t3'
  python3 bench/vllm_mlx_gate.py t4 --url http://512s2.tail91699.ts.net:8080 --capture golden.json
  python3 bench/vllm_mlx_gate.py t4 --url http://512s2.tail91699.ts.net:8080 --verify golden.json --fairness --multibranch
  RHOST=512S2 LANE=verif scripts/r --gpu 'python3 bench/vllm_mlx_gate.py t4 --cold-start --verify scratch/golden_baseline.json'
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

LEAK_RE = re.compile(r"<\|[a-zA-Z0-9_]+\|>")
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Gate T1: Pure-CPU parser, chat-template patch, and detokenizer parity
# ---------------------------------------------------------------------------

def _chunk_splits(text: str) -> list[tuple[str, list[str]]]:
    splits: list[tuple[str, list[str]]] = [
        ("1char", [c for c in text]),
        ("3char", [text[i : i + 3] for i in range(0, len(text), 3)]),
        ("7char", [text[i : i + 7] for i in range(0, len(text), 7)]),
        ("full", [text]),
    ]
    rng = random.Random(42)
    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        n = rng.randint(1, 11)
        chunks.append(text[pos : pos + n])
        pos += n
    splits.append(("rand42", chunks))
    # Adversarial split right inside every `<|` tag
    adv: list[str] = []
    pos = 0
    for m in re.finditer(r"<\|[^|]+\|>", text):
        if m.start() > pos:
            adv.append(text[pos : m.start()])
        tag = m.group(0)
        mid = len(tag) // 2
        adv.extend([tag[:1], tag[1:mid], tag[mid:-1], tag[-1:]])
        pos = m.end()
    if pos < len(text):
        adv.append(text[pos:])
    splits.append(("mid_tag", adv))
    return splits


def _check_no_leaks(label: str, *fields: str | None) -> None:
    for f in fields:
        if not f:
            continue
        m = LEAK_RE.findall(f)
        if m:
            raise AssertionError(f"{label}: leaked control tokens {m} in {f!r}")


def _norm_tcs(tcs: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    if not tcs:
        return []
    out = []
    for tc in tcs:
        if "function" in tc:
            fn = tc["function"]
            out.append((fn["name"], fn["arguments"]))
        else:
            out.append((tc["name"], tc["arguments"]))
    return out


def run_t1(model_path: str) -> int:
    print("=== GATE T1: Pure-CPU Inkling 2 Parser / Patch / Detokenizer Parity ===", flush=True)
    from vllm_mlx_inkling2.reasoning_parser import Inkling2ReasoningParser
    from vllm_mlx_inkling2.tool_parser import Inkling2ToolParser

    # 1. Reasoning parser streaming vs non-streaming across edge cases
    reasoning_cases = [
        (
            "basic_think_and_text",
            "<|message_model|><|content_thinking|>Step 1: compute 2+2=4.<|end_message|>"
            "<|message_model|><|content_text|>The answer is 4.<|end_message|><|content_model_end_sampling|>",
        ),
        (
            "recipient_header_metadata",
            "<|message_model|>recipient=user\nchannel=final<|content_thinking|>Check edge cases.<|end_message|>"
            "<|message_model|>to=user<|content_text|>Done!<|content_model_end_sampling|>",
        ),
        (
            "multi_think_and_tool",
            "<|message_model|><|content_thinking|>Need weather.<|end_message|>"
            '<|message_model|><|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Tokyo"}}<|end_message|>'
            "<|content_model_end_sampling|>",
        ),
        (
            "truncated_mid_tag",
            "<|message_model|><|content_thinking|>Halfway through<|end_mes",
        ),
        (
            "thinking_only_max_tokens",
            "<|message_model|><|content_thinking|>Still thinking when hitting length cap",
        ),
        (
            "plain_text_no_markers_short",
            "Hello world without structural markers.",
        ),
        (
            "plain_text_no_markers_long",
            "Plain prose " * 80,
        ),
        (
            "text_with_angle_brackets",
            "<|message_model|><|content_text|>Use a < b and x <|not_a_tag|> y and <|pipe<|end_message|>",
        ),
    ]

    for cname, raw in reasoning_cases:
        rp_ref = Inkling2ReasoningParser()
        ref_r, ref_c = rp_ref.extract_reasoning(raw)
        ref_r_s = ref_r or ""
        ref_c_s = ref_c or ""
        for sname, chunks in _chunk_splits(raw):
            rp = Inkling2ReasoningParser()
            acc = ""
            r_parts: list[str] = []
            c_parts: list[str] = []
            for ch in chunks:
                prev = acc
                acc += ch
                dm = rp.extract_reasoning_streaming(prev, acc, ch)
                if dm is not None:
                    if dm.reasoning:
                        r_parts.append(dm.reasoning)
                    if dm.content:
                        c_parts.append(dm.content)
            fin = rp.finalize_stream()
            if fin is not None:
                if fin.reasoning:
                    r_parts.append(fin.reasoning)
                if fin.content:
                    c_parts.append(fin.content)
            got_r = "".join(r_parts)
            got_c = "".join(c_parts)
            if got_r != ref_r_s or got_c != ref_c_s:
                raise AssertionError(
                    f"ReasoningParser mismatch on {cname}/{sname}:\n"
                    f"  ref_r={ref_r_s!r} vs got_r={got_r!r}\n"
                    f"  ref_c={ref_c_s!r} vs got_c={got_c!r}"
                )
            if cname not in ("multi_think_and_tool", "text_with_angle_brackets"):
                _check_no_leaks(f"reasoning:{cname}/{sname}", got_r, got_c)
    print(f"PASS T1.1 ReasoningParser: {len(reasoning_cases)} cases x 6 chunkings = 0 mismatches", flush=True)

    # 2. Tool parser streaming vs non-streaming across edge cases
    tool_cases = [
        (
            "single_call",
            '<|content_invoke_tool_json|>{"name": "get_weather", "args": {"city": "Tokyo"}}<|end_message|>',
        ),
        (
            "array_multi_call",
            '<|content_invoke_tool_json|>[{"name": "get_weather", "args": {"city": "Tokyo"}}, {"name": "ping", "args": {}}]<|end_message|>',
        ),
        (
            "sequential_blocks",
            'Prefix text.<|content_invoke_tool_json|>{"name": "a", "arguments": "{\\"x\\": 1}"}<|end_message|>'
            '<|content_invoke_tool_json|>{"name": "b", "args": ""}<|end_message|>'
            '<|content_invoke_tool_json|>{"name": "c", "args": null}<|end_message|>',
        ),
        (
            "tool_text_suppressed",
            'Before.<|content_invoke_tool_text|>raw tool text payload<|end_message|>After.',
        ),
        (
            "unclosed_at_eos",
            '<|content_invoke_tool_json|>{"name": "ping", "args": {}}',
        ),
        (
            "malformed_json_graceful",
            '<|content_invoke_tool_json|>{"name": "broken", "args": {<|end_message|>Fallback text.',
        ),
        (
            "large_2k_tok_payload",
            '<|content_invoke_tool_json|>{"name": "write_file", "args": {"content": "'
            + ("line_of_code_0123456789 " * 350)
            + '"}}<|end_message|>',
        ),
    ]

    for cname, raw in tool_cases:
        tp_ref = Inkling2ToolParser()
        ref = tp_ref.extract_tool_calls(raw)
        ref_tcs = _norm_tcs(ref.tool_calls)
        ref_content = (ref.content or "").strip()
        for sname, chunks in _chunk_splits(raw):
            if cname == "large_2k_tok_payload" and sname not in ("7char", "mid_tag"):
                continue
            tp = Inkling2ToolParser()
            acc = ""
            c_parts: list[str] = []
            got_tcs_raw: list[dict[str, Any]] = []
            t0 = time.perf_counter()
            for ch in chunks:
                prev = acc
                acc += ch
                res = tp.extract_tool_calls_streaming(prev, acc, ch)
                if res:
                    if res.get("content"):
                        c_parts.append(res["content"])
                    if res.get("tool_calls"):
                        got_tcs_raw.extend(res["tool_calls"])
            fin = tp.finalize_streaming(acc)
            if fin:
                if fin.get("content"):
                    c_parts.append(fin["content"])
                if fin.get("tool_calls"):
                    got_tcs_raw.extend(fin["tool_calls"])
            dt_ms = (time.perf_counter() - t0) * 1000.0
            got_tcs = _norm_tcs(got_tcs_raw)
            got_content = "".join(c_parts).strip()
            if got_tcs != ref_tcs or got_content != ref_content:
                raise AssertionError(
                    f"ToolParser mismatch on {cname}/{sname}:\n"
                    f"  ref_tcs={ref_tcs!r} vs got_tcs={got_tcs!r}\n"
                    f"  ref_c={ref_content!r} vs got_c={got_content!r}"
                )
            _check_no_leaks(f"tool:{cname}/{sname}", got_content, *[a for _, a in got_tcs])
            if cname == "large_2k_tok_payload" and sname == "7char":
                print(f"  [perf] ToolParser large_2k_tok_payload (7char, {len(chunks)} chunks): {dt_ms:.2f}ms", flush=True)
    print(f"PASS T1.2 ToolParser: {len(tool_cases)} cases streaming == non-streaming, 0 leaks", flush=True)

    # 3. Patches: _patch_chat_template_safety
    from vllm_mlx_inkling2 import patches as ipatches
    ipatches.install()
    from vllm_mlx.engine import chat_template_safety as cts

    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "thought",
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "a", "arguments": ""}},
                {"id": "2", "type": "function", "function": {"name": "b", "arguments": None}},
                {"id": "3", "type": "function", "function": {"name": "c", "arguments": '{"k": 1}'}},
                {"id": "4", "type": "function", "function": {"name": "d", "arguments": "[1,2]"}},
                {"id": "5", "type": "function", "function": {"name": "e", "arguments": {"value": ""}}},
            ],
        },
    ]
    normed = cts.normalize_messages_for_chat_template(msgs)
    am = normed[1]
    assert am.get("reasoning_content") == "thought", am
    expected_args = [{}, {}, {"k": 1}, {"value": [1, 2]}, {}]
    got_args = [tc["function"]["arguments"] for tc in am["tool_calls"]]
    assert got_args == expected_args, f"got {got_args} != {expected_args}"
    print("PASS T1.3 _patch_chat_template_safety: reasoning + tool_calls args normalization", flush=True)

    # 4. Optional FastByteLevelDetokenizer parity (if stream_opt is installed)
    stream_opt_mod = None
    try:
        import vllm_mlx_inkling2.stream_opt as stream_opt_mod
        if hasattr(stream_opt_mod, "install"):
            stream_opt_mod.install()
    except ImportError:
        pass

    expanded_model = os.path.expanduser(model_path)
    if os.path.isdir(expanded_model):
        from transformers import AutoTokenizer
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer, TokenizerWrapper
        from vllm_mlx.scheduler import Scheduler

        raw_tok = AutoTokenizer.from_pretrained(expanded_model)
        tw = TokenizerWrapper(raw_tok)
        dummy_sched = object.__new__(Scheduler)
        dummy_sched.tokenizer = tw
        detok = dummy_sched._get_detokenizer()
        detok_cls = type(detok).__name__
        if detok_cls != "NaiveStreamingDetokenizer":
            # Full single-token sample + tricky multi-token UTF-8 / space / tag parity check
            vocab_size = len(raw_tok)
            step = max(1, vocab_size // 2560)
            sample_ids = list(range(0, vocab_size, step)) + list(raw_tok.added_tokens_decoder.keys())
            bad = 0
            for tid in sample_ids:
                ref_s = raw_tok.decode([tid], skip_special_tokens=False)
                d = dummy_sched._get_detokenizer()
                d.reset()
                d.add_token(tid)
                d.finalize()
                if d.text != ref_s:
                    if bad < 5:
                        print(f"  DETOK MISMATCH id={tid}: got={d.text!r} ref={ref_s!r}", flush=True)
                    bad += 1
            assert bad == 0, f"{detok_cls} single-token mismatches: {bad}/{len(sample_ids)}"

            test_strings = [
                "  leading two spaces and   multiple   spaces\n\tand tabs!",
                "UTF-8 test: 日本語, 한국어, café, naïve, 🚀🔥🧪, —–…",
                "<|message_model|><|content_thinking|>Think 1+1=2<|end_message|><|message_model|><|content_text|>Answer: 2<|content_model_end_sampling|>",
                '{"name": "get_weather", "args": {"city": "München 🌍"}}',
            ]
            for s in test_strings:
                ids = raw_tok.encode(s, add_special_tokens=False)
                d_fast = dummy_sched._get_detokenizer()
                d_ref = NaiveStreamingDetokenizer(tw)
                d_fast.reset()
                d_ref.reset()
                fast_chunks, ref_chunks = [], []
                for tid in ids:
                    d_fast.add_token(tid)
                    d_ref.add_token(tid)
                    fast_chunks.append(d_fast.last_segment)
                    ref_chunks.append(d_ref.last_segment)
                d_fast.finalize()
                d_ref.finalize()
                fast_chunks.append(d_fast.last_segment)
                ref_chunks.append(d_ref.last_segment)
                assert "".join(fast_chunks) == "".join(ref_chunks) == d_ref.text, (
                    f"{detok_cls} stream mismatch on {s!r}: {''.join(fast_chunks)!r} vs {d_ref.text!r}"
                )
            print(f"PASS T1.4 {detok_cls}: 0/{len(sample_ids)} single-tok mm + 4/4 UTF-8/tag/space stream parity", flush=True)
        else:
            print("INFO T1.4 Detokenizer is baseline NaiveStreamingDetokenizer (skipping fast-detok parity)", flush=True)

    print("ALL GATE T1 CHECKS PASSED", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Gate T2: In-process vllm-mlx Scheduler, Prefix Cache, SSD Cache, Seeded, C=4
# ---------------------------------------------------------------------------

def _assert_cachelist_bitwise_equal(label: str, got: list[Any], ref: list[Any]) -> None:
    import mlx.core as mx

    assert len(got) == len(ref), f"{label}: layer count {len(got)} != {len(ref)}"
    k_bad = v_bad = c_bad = 0
    for li, (gl, rl) in enumerate(zip(got, ref)):
        assert type(gl).__name__ == type(rl).__name__, f"{label} L{li}: type {type(gl)} != {type(rl)}"
        assert len(gl.caches) == len(rl.caches), f"{label} L{li}: sub_count {len(gl.caches)} != {len(rl.caches)}"
        for si, (gs, rs) in enumerate(zip(gl.caches, rl.caches)):
            assert type(gs).__name__ == type(rs).__name__, f"{label} L{li}.{si}: {type(gs)} != {type(rs)}"
            if hasattr(rs, "keys") and hasattr(rs, "values") and hasattr(rs, "offset"):
                assert gs.offset == rs.offset, f"{label} L{li}.{si}: offset {gs.offset} != {rs.offset}"
                for attr in ("max_size", "keep", "step", "_idx"):
                    if hasattr(rs, attr):
                        assert getattr(gs, attr, None) == getattr(rs, attr), (
                            f"{label} L{li}.{si}: {attr} {getattr(gs, attr, None)} != {getattr(rs, attr)}"
                        )
                gk, gv = gs.keys, gs.values
                rk, rv = rs.keys, rs.values
                # Compare valid region (some caches trim pre-allocated tail padding to valid length)
                valid_len = min(rs.offset, rs.max_size) if hasattr(rs, "max_size") else rs.offset
                gk_v = gk[..., :valid_len, :]
                rk_v = rk[..., :valid_len, :]
                gv_v = gv[..., :valid_len, :]
                rv_v = rv[..., :valid_len, :]
                assert gk_v.dtype == rk_v.dtype and gk_v.shape == rk_v.shape, (
                    f"{label} L{li}.{si} K: {gk_v.shape}/{gk_v.dtype} vs {rk_v.shape}/{rk_v.dtype}"
                )
                assert gv_v.dtype == rv_v.dtype and gv_v.shape == rv_v.shape, (
                    f"{label} L{li}.{si} V: {gv_v.shape}/{gv_v.dtype} vs {rv_v.shape}/{rv_v.dtype}"
                )
                dt_v = mx.uint16 if rk_v.dtype in (mx.bfloat16, mx.float16) else mx.uint32
                k_bad += int(mx.sum(gk_v.view(dt_v) != rk_v.view(dt_v)).item())
                v_bad += int(mx.sum(gv_v.view(dt_v) != rv_v.view(dt_v)).item())
            else:
                g_arrs = gs.cache if hasattr(gs, "cache") else gs.state
                r_arrs = rs.cache if hasattr(rs, "cache") else rs.state
                assert len(g_arrs) == len(r_arrs), f"{label} L{li}.{si}: arrays len {len(g_arrs)} != {len(r_arrs)}"
                for ai, (ga, ra) in enumerate(zip(g_arrs, r_arrs)):
                    assert ga.dtype == ra.dtype and ga.shape == ra.shape, (
                        f"{label} L{li}.{si}[{ai}]: {ga.shape}/{ga.dtype} vs {ra.shape}/{ra.dtype}"
                    )
                    dt_v = mx.uint16 if ra.dtype in (mx.bfloat16, mx.float16) else mx.uint32
                    c_bad += int(mx.sum(ga.view(dt_v) != ra.view(dt_v)).item())
    if k_bad or v_bad or c_bad:
        raise AssertionError(f"{label}: bitwise CacheList mismatch K={k_bad} V={v_bad} conv={c_bad}")


def _run_scheduler_reqs(sched: Any, requests: list[Any]) -> dict[str, list[int]]:
    for r in requests:
        sched.add_request(r)
    outs: dict[str, list[int]] = {r.request_id: [] for r in requests}
    done: set[str] = set()
    while len(done) < len(requests):
        so = sched.step()
        for ro in so.outputs:
            if ro.new_token_ids:
                outs[ro.request_id].extend(int(t) for t in ro.new_token_ids)
            if ro.finished:
                done.add(ro.request_id)
    return outs


def _encode_chat(tok: Any, messages: list[dict[str, str]]) -> list[int]:
    s = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tok.encode(s, add_special_tokens=False)


def run_t2(model_path: str, skip_131k: bool = False) -> int:
    print("=== GATE T2: In-Process vllm-mlx Scheduler / Prefix-Cache / SSD / Seeded / C=4 ===", flush=True)
    os.environ["LOOPLING_SERVER_BATCHDECODE"] = "1"
    import mlx_lm  # noqa: F401  # MUST precede mlx.core!
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator, generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    import vllm_mlx_inkling2.reasoning_parser  # noqa: F401
    import vllm_mlx_inkling2.tool_parser  # noqa: F401
    from vllm_mlx_inkling2 import patches as ipatches
    ipatches.install()
    try:
        import vllm_mlx_inkling2.stream_opt as stream_opt
        if hasattr(stream_opt, "install"):
            stream_opt.install()
    except ImportError:
        pass
    try:
        import vllm_mlx_inkling2.sched_opt as sched_opt
        if hasattr(sched_opt, "install"):
            sched_opt.install()
    except ImportError:
        pass

    from vllm_mlx.request import Request, SamplingParams
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.ssd_cache import SSDCacheConfig, SSDCacheTier

    model, tok = load(os.path.expanduser(model_path))
    mx.eval(model.parameters())

    greedy_sampler = make_sampler(temp=0.0)

    def ref_generate(prompt_ids: list[int], max_tokens: int, sampler: Any = None, seed: int | None = None) -> list[int]:
        if seed is not None:
            mx.random.seed(seed)
        samp = sampler or greedy_sampler
        out = []
        for tid, _ in generate_step(mx.array(prompt_ids), model, max_tokens=max_tokens, sampler=samp):
            out.append(int(tid))
        return out

    # --- T2.1: Greedy & Seeded Sampling Parity ---
    p_short = _encode_chat(tok, [{"role": "user", "content": "Explain why 0.1 + 0.2 != 0.3 in IEEE-754 in 2 sentences."}])
    long_sys = "\n".join(
        f"Rule {i:02d}: Validate bounds, avoid allocations, preserve bitwise determinism ({i * 13 % 97})."
        for i in range(1, 72)
    )
    p_long = _encode_chat(
        tok,
        [
            {"role": "system", "content": long_sys},
            {"role": "user", "content": "Summarize Rule 01 and Rule 71 concisely."},
        ],
    )
    assert len(p_long) > 1050, f"expected >1050 prompt tokens to cross rotating window, got {len(p_long)}"

    sched_nocache = Scheduler(
        model,
        tok,
        SchedulerConfig(enable_prefix_cache=False, use_memory_aware_cache=False, prefill_step_size=4096),
    )
    for tag, pids in (("short", p_short), ("long_rot", p_long)):
        ref_ids = ref_generate(pids, max_tokens=32)
        req = Request(
            request_id=f"greedy_{tag}",
            prompt=pids,
            prompt_token_ids=list(pids),
            sampling_params=SamplingParams(max_tokens=32, temperature=0.0),
        )
        got_ids = _run_scheduler_reqs(sched_nocache, [req])[req.request_id]
        assert got_ids == ref_ids, f"T2.1 greedy {tag} mismatch:\n  got={got_ids}\n  ref={ref_ids}"
    print(f"PASS T2.1a Greedy Scheduler == generate_step (short L={len(p_short)}, long_rot L={len(p_long)}, 0 mm)", flush=True)

    # Seeded non-greedy parity: need_logprobs=[False] vs [True], and BatchDecoder vs ref
    seed_sp = make_sampler(temp=0.7, top_p=0.95)
    saved_fs = os.environ.get("LOOPLING_FAST_SAMPLE")
    os.environ["LOOPLING_FAST_SAMPLE"] = "0"
    try:
        mx.random.seed(12345)
        ref_seeded = ref_generate(p_short, max_tokens=24, sampler=seed_sp, seed=12345)

        bg_lp_true = BatchGenerator(model, prefill_step_size=4096, sampler=seed_sp)
        mx.random.seed(12345)
        bg_lp_true.insert([p_short], max_tokens=[24], need_logprobs=[True])
        toks_lpt = []
        while True:
            grs = bg_lp_true.next_generated()
            for r in grs:
                toks_lpt.append(int(r.token))
                if r.finish_reason is not None:
                    break
            if grs and grs[-1].finish_reason is not None:
                break
        bg_lp_true.close()

        bg_lp_false = BatchGenerator(model, prefill_step_size=4096, sampler=seed_sp)
        mx.random.seed(12345)
        bg_lp_false.insert([p_short], max_tokens=[24], need_logprobs=[False])
        toks_lpf = []
        while True:
            grs = bg_lp_false.next_generated()
            for r in grs:
                toks_lpf.append(int(r.token))
                if r.finish_reason is not None:
                    break
            if grs and grs[-1].finish_reason is not None:
                break
        bg_lp_false.close()
    finally:
        if saved_fs is None:
            os.environ.pop("LOOPLING_FAST_SAMPLE", None)
        else:
            os.environ["LOOPLING_FAST_SAMPLE"] = saved_fs

    assert toks_lpt == toks_lpf, f"T2.1b need_logprobs False!=True: {toks_lpf} vs {toks_lpt}"
    assert toks_lpf == ref_seeded, f"T2.1b seeded BatchGenerator!=generate_step: {toks_lpf} vs {ref_seeded}"
    print("PASS T2.1b Seeded sampling (temp=0.7, top_p=0.95, seed=12345, FAST_SAMPLE=0): need_logprobs=[False]==[True]==generate_step (0 mm)", flush=True)

    # --- T2.2: Prefix Cache RAM HIT + 1-tok immutability vs Uncached Full Prefill ---
    sched_cache = Scheduler(
        model,
        tok,
        SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            cache_memory_percent=0.35,
            prefill_step_size=4096,
        ),
    )
    ref_t1 = ref_generate(p_long, max_tokens=24)
    r_t1 = Request("mt_t1", p_long, SamplingParams(max_tokens=24, temperature=0.0), prompt_token_ids=list(p_long))
    got_t1 = _run_scheduler_reqs(sched_cache, [r_t1])["mt_t1"]
    assert got_t1 == ref_t1, f"T2.2 T1 mismatch: {got_t1} vs {ref_t1}"

    # Repeat T1 twice (tests exact repeat HIT + 1-tok direct-admit does NOT mutate stored CacheList!)
    r_t1_r1 = Request("mt_t1_r1", p_long, SamplingParams(max_tokens=24, temperature=0.0), prompt_token_ids=list(p_long))
    got_t1_r1 = _run_scheduler_reqs(sched_cache, [r_t1_r1])["mt_t1_r1"]
    r_t1_r2 = Request("mt_t1_r2", p_long, SamplingParams(max_tokens=24, temperature=0.0), prompt_token_ids=list(p_long))
    got_t1_r2 = _run_scheduler_reqs(sched_cache, [r_t1_r2])["mt_t1_r2"]
    assert got_t1_r1 == ref_t1, f"T2.2 T1 repeat #1 mismatch: {got_t1_r1} vs {ref_t1}"
    assert got_t1_r2 == ref_t1, f"T2.2 T1 repeat #2 (immutability) mismatch: {got_t1_r2} vs {ref_t1}"

    # Turn 2 continuation (prompt = p_long + ref_t1 + suffix)
    t2_suffix = tok.encode("<|end_message|><|message_user|><|content_text|>Say OK.<|end_message|>", add_special_tokens=False)
    p_t2 = p_long + ref_t1 + t2_suffix
    ref_t2 = ref_generate(p_t2, max_tokens=24)
    r_t2 = Request("mt_t2", p_t2, SamplingParams(max_tokens=24, temperature=0.0), prompt_token_ids=list(p_t2))
    got_t2 = _run_scheduler_reqs(sched_cache, [r_t2])["mt_t2"]
    assert got_t2 == ref_t2, f"T2.2 T2 continuation mismatch: {got_t2} vs {ref_t2}"
    cstats = sched_cache.memory_aware_cache.get_stats() if sched_cache.memory_aware_cache else {}
    print(
        f"PASS T2.2 Prefix-cache RAM HIT & 1-tok immutability (L={len(p_long)}, T2 L={len(p_t2)}, 0 mm) "
        f"r1_cached={r_t1_r1.cached_tokens} r2_cached={r_t1_r2.cached_tokens} t2_cached={r_t2.cached_tokens} stats={cstats}",
        flush=True,
    )

    # --- T2.3: SSD Spill / Reload Bitwise CacheList + Continuation Parity ---
    tmp_ssd = tempfile.mkdtemp(prefix="vllm_mlx_ssd_gate_")
    try:
        ssd = SSDCacheTier(SSDCacheConfig(cache_dir=tmp_ssd, max_size_gb=8.0))
        ssd.start_writer()
        for tag, pids in (("unwrapped_160", p_long[:160]), ("wrapped_1200", p_long[:1200])):
            # Prefill N-1 tokens into fresh CacheList
            prefix = pids[:-1]
            c_orig = make_prompt_cache(model)
            model(mx.array([prefix]), cache=c_orig)
            mx.eval([x.state for x in c_orig])
            # Deep reference snapshot before spill
            c_ref = copy.deepcopy(c_orig)
            mx.eval([x.state for x in c_ref])

            t_sp0 = time.perf_counter()
            ok = ssd.enqueue_spill(tuple(prefix), c_orig, memory_bytes=1)
            assert ok, f"T2.3 enqueue_spill returned False for {tag}"
            # Wait for background writer to finish entry
            entry = None
            deadline = time.perf_counter() + 60.0
            while time.perf_counter() < deadline:
                entry = ssd.lookup_ssd(tuple(prefix))
                if entry is not None:
                    break
                time.sleep(0.02)
            spill_ms = (time.perf_counter() - t_sp0) * 1000.0
            assert entry is not None, f"T2.3 SSD entry not written within timeout for {tag}"

            t_rl0 = time.perf_counter()
            layer_dicts = ssd._read_entry(tuple(prefix), entry["file_path"])
            assert layer_dicts is not None, f"T2.3 _read_entry returned None for {tag}"
            c_reloaded = sched_cache._reconstruct_ssd_layers(layer_dicts)
            assert c_reloaded is not None, f"T2.3 _reconstruct_ssd_layers returned None for {tag}"
            mx.eval([x.state for x in c_reloaded])
            reload_ms = (time.perf_counter() - t_rl0) * 1000.0

            # Bitwise verify every layer's K/V uint16 + ArraysCache conv uint32 + rotating attrs
            _assert_cachelist_bitwise_equal(f"T2.3:{tag}", c_reloaded, c_ref)

            # Verify 24-token greedy continuation from reloaded cache == original cache
            def cont_from(cache_obj: list[Any], last_tok: int, n: int = 24) -> list[int]:
                out = []
                for tid, _ in generate_step(mx.array([last_tok]), model, max_tokens=n, sampler=greedy_sampler, prompt_cache=cache_obj):
                    out.append(int(tid))
                return out

            toks_reloaded = cont_from(c_reloaded, pids[-1], 24)
            toks_orig = cont_from(c_ref, pids[-1], 24)
            assert toks_reloaded == toks_orig, f"T2.3 {tag} continuation mismatch:\n  reloaded={toks_reloaded}\n  orig={toks_orig}"

            # Compute on-disk size for this entry
            entry_dir = os.path.join(tmp_ssd, "data", entry["file_path"])
            if not os.path.isdir(entry_dir):
                entry_dir = os.path.join(tmp_ssd, entry["file_path"])
            disk_mb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(entry_dir)
                for f in fns
            ) / (1024 * 1024)
            print(
                f"PASS T2.3 SSD round-trip ({tag}, L={len(prefix)}): 0 bit mm across 48 visits' K/V+4xConv, "
                f"24/24 continuation exact, spill={spill_ms:.1f}ms reload={reload_ms:.1f}ms disk={disk_mb:.1f}MB",
                flush=True,
            )
        ssd.close()
    finally:
        shutil.rmtree(tmp_ssd, ignore_errors=True)

    # --- T2.4: C=1 vs C=4 Batching Parity in Scheduler (prefix-cached B=1 prefill -> C=4 BatchDecoder) ---
    # Note: Cold batched prefill of 4 prompts (M=B*L=336) crosses MLX GEMM tile boundaries vs B=1 prefill (M=84);
    # prefilling each prompt individually into MemoryAwarePrefixCache first isolates C=4 continuous-batching
    # prefix-hit + BatchDecoder decode parity against C=1 (0 mm required!).
    c4_pad = (
        "Context: We are evaluating deterministic, bit-exact batched inference on Apple Silicon M3 Ultra "
        "using Inkling 2 with sliding-window attention and short convolutions. Answer precisely and concisely.\n\n"
    )
    c4_texts = [
        c4_pad + "Task 1: Prove that the sum of the first n odd positive integers equals n^2 in 4 lines.",
        c4_pad + "Task 2: Write a bash command to list the 5 largest files in /var/log and explain each flag.",
        c4_pad + "Task 3: State 3 concrete differences between TCP and UDP with one protocol example each.",
        c4_pad + "Task 4: Write a Python binary search returning the leftmost index of target or -1.",
    ]
    c4_pids = [_encode_chat(tok, [{"role": "user", "content": t}]) for t in c4_texts]
    assert all(len(p) >= 64 for p in c4_pids), f"expected L>=64 for C=4 parity, got {[len(p) for p in c4_pids]}"

    c1_refs = [ref_generate(p, max_tokens=32) for p in c4_pids]
    # Populate N-1 prefix cache via C=1 runs on sched_cache, verifying C=1 parity first
    for i, p in enumerate(c4_pids):
        r1 = Request(f"c4_warm_{i}", p, SamplingParams(max_tokens=4, temperature=0.0), prompt_token_ids=list(p))
        g1 = _run_scheduler_reqs(sched_cache, [r1])[r1.request_id]
        assert g1 == c1_refs[i][:4], f"T2.4 warm C=1 stream {i} mismatch: {g1} vs {c1_refs[i][:4]}"
    # Now fire all 4 concurrently on sched_cache -> all 4 hit N-1 prefix cache & decode together at C=4!
    c4_reqs = [
        Request(f"c4_{i}", p, SamplingParams(max_tokens=32, temperature=0.0), prompt_token_ids=list(p))
        for i, p in enumerate(c4_pids)
    ]
    c4_got = _run_scheduler_reqs(sched_cache, c4_reqs)
    for i in range(4):
        assert c4_got[f"c4_{i}"] == c1_refs[i], (
            f"T2.4 C=4 vs C=1 stream {i} (L={len(c4_pids[i])}, cached={c4_reqs[i].cached_tokens}) mismatch:\n"
            f"  c4={c4_got[f'c4_{i}']}\n  c1={c1_refs[i]}"
        )
    print(
        f"PASS T2.4 C=4 vs C=1 Scheduler parity (L={[len(p) for p in c4_pids]}, "
        f"cached={[r.cached_tokens for r in c4_reqs]}, 4x32=128 tokens, 0 mm)",
        flush=True,
    )

    # --- T2.5: Dynamic C=1 -> C=2 -> C=1 handoff & mixed-temp Scheduler parity ---
    # Tests continuous join/leave handoff (crucial when LOOPLING_MTP=dflash2 runs at C=1 and hands off at C>=2)
    r_a = Request("handoff_a", c4_pids[0], SamplingParams(max_tokens=32, temperature=0.0), prompt_token_ids=list(c4_pids[0]))
    r_b = Request("handoff_b", c4_pids[1], SamplingParams(max_tokens=16, temperature=0.0), prompt_token_ids=list(c4_pids[1]))
    sched_nocache.add_request(r_a)
    got_a: list[int] = []
    got_b: list[int] = []
    added_b = False
    done_h: set[str] = set()
    while len(done_h) < 2:
        so = sched_nocache.step()
        for ro in so.outputs:
            if ro.request_id == "handoff_a":
                got_a.extend(int(t) for t in ro.new_token_ids)
                if len(got_a) >= 6 and not added_b:
                    added_b = True
                    sched_nocache.add_request(r_b)
            elif ro.request_id == "handoff_b":
                got_b.extend(int(t) for t in ro.new_token_ids)
            if ro.finished:
                done_h.add(ro.request_id)
    assert got_a == c1_refs[0][:32], f"T2.5 handoff_a mismatch:\n  got={got_a}\n  ref={c1_refs[0][:32]}"
    assert got_b == c1_refs[1][:16], f"T2.5 handoff_b mismatch:\n  got={got_b}\n  ref={c1_refs[1][:16]}"
    print(f"PASS T2.5 Dynamic C=1->C=2->C=1 handoff parity (MTP={os.environ.get('LOOPLING_MTP', 'off')}, 0 mm)", flush=True)

    # --- T2.6: Concurrent Prefill-vs-Decode Fairness + Bit-Exactness ---
    # Ensures a large (~16K) uncached prefill does not starve an interactive short request (max_tokens=64).
    p_16k_body = "\n".join(
        f"Log line {i:04d}: checksum={(i * 2654435761) & 0xFFFF:04x} status=OK latency_ms={(i % 17) + 1}."
        for i in range(1, 700)
    )
    p_16k = _encode_chat(
        tok,
        [{"role": "user", "content": "Count the OK entries and summarize briefly:\n" + p_16k_body}],
    )
    if len(p_16k) < 16384:
        pad_block = tok.encode("Verify invariant alpha-beta-gamma-delta 0123456789.\n", add_special_tokens=False)
        need = 16384 - len(p_16k)
        reps = (need + len(pad_block) - 1) // len(pad_block)
        p_16k = (pad_block * reps)[:need] + p_16k
    p_16k = p_16k[:16384]
    ref_16k = ref_generate(p_16k, max_tokens=16)
    ref_fast = ref_generate(p_short, max_tokens=64)

    r_prefill = Request("fair_16k", p_16k, SamplingParams(max_tokens=16, temperature=0.0), prompt_token_ids=list(p_16k))
    r_fast = Request("fair_short", p_short, SamplingParams(max_tokens=64, temperature=0.0), prompt_token_ids=list(p_short))
    sched_nocache.add_request(r_prefill)
    got_16k: list[int] = []
    got_fast: list[int] = []
    fast_times: list[float] = []
    fast_submit_t = 0.0
    added_fast = False
    done_f: set[str] = set()
    step_i = 0
    while len(done_f) < 2:
        so = sched_nocache.step()
        now = time.perf_counter()
        step_i += 1
        for ro in so.outputs:
            if ro.request_id == "fair_16k" and ro.new_token_ids:
                got_16k.extend(int(t) for t in ro.new_token_ids)
            elif ro.request_id == "fair_short" and ro.new_token_ids:
                got_fast.extend(int(t) for t in ro.new_token_ids)
                fast_times.append(now)
            if ro.finished:
                done_f.add(ro.request_id)
        if not added_fast and step_i >= 1:
            added_fast = True
            fast_submit_t = time.perf_counter()
            sched_nocache.add_request(r_fast)

    assert got_16k == ref_16k, f"T2.6 fair_16k mismatch:\n  got={got_16k}\n  ref={ref_16k}"
    assert got_fast == ref_fast, f"T2.6 fair_short mismatch:\n  got={got_fast}\n  ref={ref_fast}"
    assert fast_times, "T2.6 fair_short produced no timestamps"
    fast_ttft_ms = (fast_times[0] - fast_submit_t) * 1000.0
    fast_gaps = [(fast_times[i] - fast_times[i - 1]) * 1000.0 for i in range(1, len(fast_times))]
    fast_max_gap_ms = max(fast_gaps) if fast_gaps else 0.0
    assert fast_ttft_ms <= 4000.0 and fast_max_gap_ms <= 2500.0, (
        f"T2.6 prefill starved interactive decode: ttft={fast_ttft_ms:.1f}ms max_gap={fast_max_gap_ms:.1f}ms"
    )
    print(
        f"PASS T2.6 Concurrent prefill-vs-decode fairness (L_prefill={len(p_16k)}, "
        f"fast_ttft={fast_ttft_ms:.1f}ms fast_max_gap={fast_max_gap_ms:.1f}ms, 0 mm)",
        flush=True,
    )

    # --- T2.7: Multi-branch Prefix Cache Retention ---
    # Branching Session A (L~800 -> L~1200) must NOT evict Session B's shared ancestor (L~800).
    mb_base_txt = "\n".join(
        f"BranchBase {i:03d}: deterministic cache retention check invariant ({i * 19 % 101})."
        for i in range(1, 55)
    )
    p_mb_base = _encode_chat(tok, [{"role": "system", "content": mb_base_txt}, {"role": "user", "content": "State Base ID."}])
    assert len(p_mb_base) >= 750, f"expected >=750 tokens for T2.7 base, got {len(p_mb_base)}"
    suf_a = tok.encode(" ".join(f"BranchA_token_{i}" for i in range(120)), add_special_tokens=False)
    suf_b = tok.encode(" ".join(f"BranchB_token_{i}" for i in range(60)), add_special_tokens=False)
    p_mb_a = p_mb_base + suf_a
    p_mb_b = p_mb_base + suf_b
    assert len(p_mb_a) >= 1150, f"expected >=1150 tokens for T2.7 branch A, got {len(p_mb_a)}"

    ref_mb_base = ref_generate(p_mb_base, max_tokens=16)
    ref_mb_a = ref_generate(p_mb_a, max_tokens=16)
    ref_mb_b = ref_generate(p_mb_b, max_tokens=16)

    r_mb_base = Request("mb_base", p_mb_base, SamplingParams(max_tokens=16, temperature=0.0), prompt_token_ids=list(p_mb_base))
    got_mb_base = _run_scheduler_reqs(sched_cache, [r_mb_base])["mb_base"]
    assert got_mb_base == ref_mb_base, f"T2.7 mb_base mismatch: {got_mb_base} vs {ref_mb_base}"

    r_mb_a = Request("mb_a", p_mb_a, SamplingParams(max_tokens=16, temperature=0.0), prompt_token_ids=list(p_mb_a))
    got_mb_a = _run_scheduler_reqs(sched_cache, [r_mb_a])["mb_a"]
    assert got_mb_a == ref_mb_a, f"T2.7 mb_a mismatch: {got_mb_a} vs {ref_mb_a}"
    assert r_mb_a.cached_tokens >= len(p_mb_base) - 1, (
        f"T2.7 mb_a expected ancestor hit >= {len(p_mb_base) - 1}, got {r_mb_a.cached_tokens}"
    )

    r_mb_b = Request("mb_b", p_mb_b, SamplingParams(max_tokens=16, temperature=0.0), prompt_token_ids=list(p_mb_b))
    got_mb_b = _run_scheduler_reqs(sched_cache, [r_mb_b])["mb_b"]
    assert got_mb_b == ref_mb_b, f"T2.7 mb_b mismatch: {got_mb_b} vs {ref_mb_b}"
    assert r_mb_b.cached_tokens >= len(p_mb_base) - 1, (
        f"T2.7 mb_b shared ancestor destroyed! cached_tokens={r_mb_b.cached_tokens} < {len(p_mb_base) - 1}"
    )
    mb_stats = sched_cache.memory_aware_cache.get_stats() if sched_cache.memory_aware_cache else {}
    assert mb_stats.get("store_rejections", 0) == 0, f"T2.7 unexpected store_rejections={mb_stats}"
    assert tuple(p_mb_base[:-1]) in sched_cache.memory_aware_cache._entries, "T2.7 base key missing from RAM cache"
    print(
        f"PASS T2.7 Multi-branch prefix retention (base={len(p_mb_base)} -> A={len(p_mb_a)}, "
        f"B_cached={r_mb_b.cached_tokens}, store_rejections=0, 0 mm)",
        flush=True,
    )

    # --- T2.8: 128K-132K DFlash2 Boundary Check (_KV_STACK_MAX_CAP = 131,072) ---
    if not skip_131k:
        seed_blk = tok.encode(
            "Verify deterministic KV state transition across the 131072 token stack boundary without drift.\n",
            add_special_tokens=False,
        )
        tail_chat = _encode_chat(
            tok,
            [
                {
                    "role": "user",
                    "content": "Write the integers from 1000 to 1300 separated by newlines, with no extra commentary.",
                }
            ],
        )
        n_over = 131_200
        n_cross = 130_800  # 130,799 rounds to 130,816 (< 131,072 cap) and crosses 131,072 during 320-tok decode
        p_131k = (seed_blk * ((n_cross + len(seed_blk) - 1) // len(seed_blk)))[: n_cross - len(tail_chat)] + tail_chat
        assert len(p_131k) == n_cross
        p_131k_over = p_131k + (seed_blk * ((n_over - n_cross + len(seed_blk) - 1) // len(seed_blk)))[: n_over - n_cross]
        assert len(p_131k_over) == n_over

        t_pf0 = time.perf_counter()
        c_130k = make_prompt_cache(model)
        idx_arr = mx.array(p_131k[: n_cross - 1])
        pos = 0
        while pos < idx_arr.size:
            step_n = min(4096, int(idx_arr.size) - pos)
            model(idx_arr[pos : pos + step_n][None], cache=c_130k)
            mx.eval([x.state for x in c_130k])
            mx.clear_cache()
            pos += step_n
        prefill_130k_s = time.perf_counter() - t_pf0

        c_130k_ref = copy.deepcopy(c_130k)
        mx.eval([x.state for x in c_130k_ref])
        sched_cache.memory_aware_cache.store(p_131k[: n_cross - 1], copy.deepcopy(c_130k), evict_prefixes=False)

        ref_130k_cross = [
            int(tid)
            for tid, _ in generate_step(
                mx.array([p_131k[n_cross - 1]]),
                model,
                max_tokens=320,
                sampler=greedy_sampler,
                prompt_cache=c_130k_ref,
            )
        ]
        r_cross = Request(
            "df2_130k_cross",
            p_131k[:n_cross],
            SamplingParams(max_tokens=320, temperature=0.0),
            prompt_token_ids=list(p_131k[:n_cross]),
        )
        got_130k_cross = _run_scheduler_reqs(sched_cache, [r_cross])["df2_130k_cross"]
        assert r_cross.cached_tokens == n_cross - 1, (
            f"T2.8 expected full cache hit {n_cross - 1}, got {r_cross.cached_tokens}"
        )
        assert len(got_130k_cross) >= 280, (
            f"T2.8 stopped before crossing 131072! len={len(got_130k_cross)} (<280)"
        )
        assert got_130k_cross == ref_130k_cross[: len(got_130k_cross)], (
            f"T2.8 crossing 131072 mismatch (cached={r_cross.cached_tokens}, len={len(got_130k_cross)})"
        )

        # Case B: start >= 131,072 (_b1_unpacked_full=True)
        model(mx.array(p_131k_over[n_cross - 1 : n_over - 1])[None], cache=c_130k)
        mx.eval([x.state for x in c_130k])
        mx.clear_cache()
        c_over_ref = copy.deepcopy(c_130k)
        mx.eval([x.state for x in c_over_ref])
        sched_cache.memory_aware_cache.store(p_131k_over[: n_over - 1], c_130k, evict_prefixes=False)

        ref_over = [
            int(tid)
            for tid, _ in generate_step(
                mx.array([p_131k_over[n_over - 1]]),
                model,
                max_tokens=32,
                sampler=greedy_sampler,
                prompt_cache=c_over_ref,
            )
        ]
        r_over = Request(
            "df2_131k_over",
            p_131k_over[:n_over],
            SamplingParams(max_tokens=32, temperature=0.0),
            prompt_token_ids=list(p_131k_over[:n_over]),
        )
        got_over = _run_scheduler_reqs(sched_cache, [r_over])["df2_131k_over"]
        assert r_over.cached_tokens == n_over - 1, (
            f"T2.8 expected full cache hit {n_over - 1}, got {r_over.cached_tokens}"
        )
        assert got_over == ref_over[: len(got_over)], f"T2.8 >=131072 mismatch:\n  got={got_over}\n  ref={ref_over}"
        b_stats = sched_cache.memory_aware_cache.get_stats()
        assert b_stats.get("store_rejections", 0) == 0, f"T2.8 store_rejections={b_stats}"
        del c_130k_ref, c_over_ref, c_130k
        mx.clear_cache()
        print(
            f"PASS T2.8 128K-132K DFlash2 boundary (prefill_130k={prefill_130k_s:.1f}s, "
            f"cross {n_cross}->+320=0 mm, over {n_over}->+32=0 mm, store_rejections=0)",
            flush=True,
        )

    print("ALL GATE T2 CHECKS PASSED", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Gate T3: Core loopling-opt regression suites
# ---------------------------------------------------------------------------

def run_t3(with_spec: bool = False) -> int:
    print("=== GATE T3: Core loopling-opt Regressions ===", flush=True)
    cmds = [
        ("a2_tests", [sys.executable, str(REPO_ROOT / "bench/a2_tests.py"), "all"]),
        (
            "prefill_exact_test",
            [
                sys.executable,
                str(REPO_ROOT / "bench/prefill_exact_test.py"),
                "--step",
                "4096",
                "--lens",
                "17,43,63,64,129,521,2048,4096",
            ],
        ),
        ("server_batch_test", [sys.executable, str(REPO_ROOT / "bench/server_batch_test.py"), "--C", "1,2,4,8"]),
    ]
    if with_spec:
        cmds.append(("spec_exact_test", [sys.executable, str(REPO_ROOT / "bench/spec_exact_test.py")]))
    for name, cmd in cmds:
        print(f"\n--- Running T3: {name} ---", flush=True)
        t0 = time.perf_counter()
        res = subprocess.run(cmd, cwd=str(REPO_ROOT))
        dt = time.perf_counter() - t0
        if res.returncode != 0:
            print(f"FAIL T3.{name} (exit={res.returncode}, {dt:.1f}s)", flush=True)
            return res.returncode
        print(f"PASS T3.{name} ({dt:.1f}s)", flush=True)
    print("\nALL GATE T3 CHECKS PASSED", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Gate T4: Live HTTP Golden Fingerprint & Perf Gate
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_http_tcs(tcs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tcs:
        return []
    out = []
    for i, tc in enumerate(tcs):
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments", "")
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else raw_args
            canon = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        except Exception:
            canon = str(raw_args)
        out.append({"pos": i, "type": tc.get("type", "function"), "name": fn.get("name", ""), "arguments_raw": raw_args, "arguments_canon": canon})
    return out


def _http_chat_nonstream(base_url: str, payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    body = dict(payload)
    body["stream"] = False
    body.pop("stream_options", None)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    dt_ms = (time.perf_counter() - t0) * 1000.0
    ch = raw["choices"][0]
    msg = ch["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
        "tool_calls": _normalize_http_tcs(msg.get("tool_calls")),
        "finish_reason": ch.get("finish_reason"),
        "usage": raw.get("usage") or {},
        "total_ms": round(dt_ms, 2),
    }


def _http_chat_stream(base_url: str, payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    body = dict(payload)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tc_map: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    payload_times: list[float] = []
    n_sse = 0
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            ps = line[5:].strip()
            if ps == "[DONE]":
                break
            now = time.perf_counter()
            n_sse += 1
            ev = json.loads(ps)
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if ch.get("finish_reason") is not None:
                finish_reason = ch["finish_reason"]
            delta = ch.get("delta") or {}
            c = delta.get("content")
            r = delta.get("reasoning_content")
            if r is None:
                r = delta.get("reasoning")
            tcs = delta.get("tool_calls")
            has_payload = False
            if c:
                content_parts.append(c)
                has_payload = True
            if r:
                reasoning_parts.append(r)
                has_payload = True
            if tcs:
                has_payload = True
                for tc in tcs:
                    idx = tc.get("index", 0)
                    entry = tc_map.setdefault(
                        idx,
                        {"index": idx, "id": "", "type": tc.get("type") or "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        entry["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]
            if has_payload:
                payload_times.append(now)

    t_end = time.perf_counter()
    ttft_ms = ((payload_times[0] if payload_times else t_end) - t0) * 1000.0
    total_ms = (t_end - t0) * 1000.0
    gaps = [(payload_times[i] - payload_times[i - 1]) * 1000.0 for i in range(1, len(payload_times))]
    comp_toks = int(usage.get("completion_tokens") or 0)
    dec_s = (t_end - payload_times[0]) if payload_times else max(1e-6, t_end - t0)
    dec_tps = (comp_toks - 1) / dec_s if (comp_toks > 1 and dec_s > 0) else comp_toks / max(1e-6, t_end - t0)
    raw_tcs = [tc_map[i] for i in sorted(tc_map)] if tc_map else None
    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_calls": _normalize_http_tcs(raw_tcs),
        "finish_reason": finish_reason,
        "usage": usage,
        "ttft_ms": round(ttft_ms, 2),
        "total_ms": round(total_ms, 2),
        "dec_tps": round(dec_tps, 2),
        "p50_gap_ms": round(statistics.median(gaps), 2) if gaps else 0.0,
        "p95_gap_ms": round(sorted(gaps)[int(0.95 * (len(gaps) - 1))], 2) if gaps else 0.0,
        "max_gap_ms": round(max(gaps), 2) if gaps else 0.0,
        "n_sse_events": n_sse,
        "n_payload_chunks": len(payload_times),
    }


def _http_fp(rec: dict[str, Any]) -> str:
    canonical = {
        "content": rec["content"],
        "reasoning_content": rec["reasoning_content"],
        "tool_calls": [
            {"pos": t["pos"], "type": t["type"], "name": t["name"], "arguments_raw": t["arguments_raw"]}
            for t in rec["tool_calls"]
        ],
        "finish_reason": rec["finish_reason"],
        "prompt_tokens": (rec.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (rec.get("usage") or {}).get("completion_tokens"),
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _http_leaks(rec: dict[str, Any]) -> list[str]:
    leaks = []
    for k in ("content", "reasoning_content"):
        m = LEAK_RE.findall(rec.get(k) or "")
        if m:
            leaks.append(f"{k}:{m}")
    for tc in rec.get("tool_calls") or []:
        m = LEAK_RE.findall(tc.get("arguments_raw") or "")
        if m:
            leaks.append(f"tool_args:{m}")
    return leaks


def _build_t4_suite(model: str) -> dict[str, Any]:
    medium_code = """class OrderBookProcessor:
    def __init__(self, symbol, tick_size=0.01, max_levels=100, enable_audit=True):
        self.symbol = symbol
        self.tick_size = tick_size
        self.max_levels = max_levels
        self.enable_audit = enable_audit
        self.bids = {}
        self.asks = {}
        self.sequence_number = 0
        self.audit_log = []

    def apply_delta(self, side, price, quantity, sequence_number):
        if sequence_number <= self.sequence_number:
            return False
        book = self.bids if side == "bid" else self.asks
        if quantity <= 0:
            book.pop(price, None)
        else:
            book[price] = quantity
        self.sequence_number = sequence_number
        if self.enable_audit:
            self.audit_log.append((sequence_number, side, price, quantity))
        return True

    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def spread(self):
        bb, ba = self.best_bid(), self.best_ask()
        return None if (bb is None or ba is None) else round(ba - bb, 8)
"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        },
        {
            "type": "function",
            "function": {"name": "ping", "description": "Return pong to check liveness.", "parameters": {"type": "object", "properties": {}}},
        },
    ]
    long_rules = "\n".join(
        f"Rule {i:02d}: Always validate inputs cleanly, avoid unnecessary allocations, prefer clear naming, "
        f"document boundary conditions, and verify complexity bounds when analyzing algorithms ({i * 7 % 97})."
        for i in range(1, 52)
    )
    mt_sys = (
        "You are a precise systems engineering assistant specialized in Python and Apple Silicon performance.\n"
        + long_rules
        + "\nSession Tag: VERIF-GOLDEN-001\n"
    )
    c4_ge64_pad = (
        "Context: We are evaluating deterministic, bit-exact batched inference on Apple Silicon M3 Ultra "
        "using Inkling 2 with sliding-window attention and short convolutions. Answer precisely and concisely.\n\n"
    )
    return {
        "short_greedy": {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 160,
            "messages": [{"role": "user", "content": "Explain in 3 concise sentences why IEEE-754 floating-point addition is not associative, with one concrete numerical example."}],
        },
        "medium_greedy": {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 180,
            "messages": [
                {"role": "system", "content": "You are a senior Python performance engineer. Keep responses structured, precise, and concise."},
                {"role": "user", "content": "Review the following Python class and list 4 concrete correctness or performance edge cases with a one-line fix for each:\n```python\n" + medium_code + "```"},
            ],
        },
        "tool_call_greedy": {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 256,
            "tools": tools,
            "tool_choice": "auto",
            "messages": [{"role": "user", "content": "Call both get_weather(city='Tokyo') and ping() right now, and do not explain until tool results arrive."}],
        },
        "multiturn_sys": mt_sys,
        "c4_short_prompts": [
            "Prove that the sum of the first n odd positive integers equals n^2 in 4 short lines.",
            "Write a bash one-liner to find the 5 largest files in /var/log and explain each flag briefly.",
            "List 3 key differences between TCP and UDP with one concrete protocol example for each.",
            "Write a concise Python binary search function returning the leftmost index of target or -1.",
        ],
        "c4_ge64_prompts": [
            c4_ge64_pad + "Task 1: Prove that the sum of the first n odd positive integers equals n^2 in 4 short lines.",
            c4_ge64_pad + "Task 2: Write a bash one-liner to find the 5 largest files in /var/log and explain each flag briefly.",
            c4_ge64_pad + "Task 3: List 3 key differences between TCP and UDP with one concrete protocol example for each.",
            c4_ge64_pad + "Task 4: Write a concise Python binary search function returning the leftmost index of target or -1.",
        ],
    }


def _capture_t4(base_url: str) -> dict[str, Any]:
    health = _http_get_json(f"{base_url.rstrip('/')}/health")
    model = health["model_name"]
    suite = _build_t4_suite(model)
    res: dict[str, Any] = {
        "meta": {"base_url": base_url, "model": model, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "cases": {},
        "parity_checks": {},
    }

    for name in ("short_greedy", "medium_greedy", "tool_call_greedy"):
        pl = suite[name]
        st = _http_chat_stream(base_url, pl)
        ns = _http_chat_nonstream(base_url, pl)
        st["sha256"] = _http_fp(st)
        ns["sha256"] = _http_fp(ns)
        st["leaks"] = _http_leaks(st)
        ns["leaks"] = _http_leaks(ns)
        eq = st["sha256"] == ns["sha256"]
        res["cases"][name] = {"stream": st, "nonstream": ns}
        res["parity_checks"][f"{name}_stream_eq_nonstream"] = eq
        print(
            f"[{name:16s}] ptok={st['usage'].get('prompt_tokens'):4} ctok={st['usage'].get('completion_tokens'):3} "
            f"ttft={st['ttft_ms']:7.1f}ms dec={st['dec_tps']:6.2f}t/s p50/p95/max={st['p50_gap_ms']:.1f}/{st['p95_gap_ms']:.1f}/{st['max_gap_ms']:.1f}ms "
            f"eq={eq} leaks={len(st['leaks'])+len(ns['leaks'])} sha={st['sha256'][:12]}",
            flush=True,
        )

    # Multi-turn >1024-tok rotating window
    mt_t1_pl = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": suite["multiturn_sys"]},
            {"role": "user", "content": "Summarize Rule 01 and Rule 51 in one sentence each."},
        ],
    }
    c0 = _http_get_json(f"{base_url.rstrip('/')}/v1/cache/stats").get("engine_cache", {})
    mt1 = _http_chat_stream(base_url, mt_t1_pl)
    mt1["sha256"], mt1["leaks"] = _http_fp(mt1), _http_leaks(mt1)
    c1 = _http_get_json(f"{base_url.rstrip('/')}/v1/cache/stats").get("engine_cache", {})
    mt1r = _http_chat_stream(base_url, mt_t1_pl)
    mt1r["sha256"], mt1r["leaks"] = _http_fp(mt1r), _http_leaks(mt1r)
    c1r = _http_get_json(f"{base_url.rstrip('/')}/v1/cache/stats").get("engine_cache", {})
    asst: dict[str, Any] = {"role": "assistant", "content": mt1["content"]}
    if mt1["reasoning_content"]:
        asst["reasoning_content"] = mt1["reasoning_content"]
    mt_t2_pl = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 128,
        "messages": mt_t1_pl["messages"] + [asst, {"role": "user", "content": "Now write a 2-line Python helper `is_pow2(n: int) -> bool`."}],
    }
    mt2 = _http_chat_stream(base_url, mt_t2_pl)
    mt2["sha256"], mt2["leaks"] = _http_fp(mt2), _http_leaks(mt2)
    c2 = _http_get_json(f"{base_url.rstrip('/')}/v1/cache/stats").get("engine_cache", {})
    res["cases"]["multiturn_t1"] = mt1
    res["cases"]["multiturn_t1_repeat"] = mt1r
    res["cases"]["multiturn_t2"] = mt2
    res["cases"]["multiturn_cache_stats"] = {"before": c0, "after_t1": c1, "after_t1_repeat": c1r, "after_t2": c2}
    res["parity_checks"]["multiturn_t1_repeat_exact"] = mt1["sha256"] == mt1r["sha256"]
    print(
        f"[multiturn       ] T1 ttft={mt1['ttft_ms']:6.1f}ms sha={mt1['sha256'][:12]} | "
        f"T1_rep ttft={mt1r['ttft_ms']:6.1f}ms eq={mt1['sha256']==mt1r['sha256']} (hits {c0.get('hits')}->{c1r.get('hits')}) | "
        f"T2 ttft={mt2['ttft_ms']:6.1f}ms sha={mt2['sha256'][:12]} (hits->{c2.get('hits')})",
        flush=True,
    )

    # C=4 short & C=4 L>=64 (bit-exact regime)
    for tag, pkey in (("c4_short", "c4_short_prompts"), ("c4_ge64", "c4_ge64_prompts")):
        pls = [{"model": model, "temperature": 0.0, "max_tokens": 128, "messages": [{"role": "user", "content": p}]} for p in suite[pkey]]
        c1_refs = []
        for pl in pls:
            r = _http_chat_stream(base_url, pl)
            r["sha256"], r["leaks"] = _http_fp(r), _http_leaks(r)
            c1_refs.append(r)
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            c4_outs = [f.result() for f in [ex.submit(_http_chat_stream, base_url, pl) for pl in pls]]
        wall = time.perf_counter() - t0
        eqs = []
        tot = 0
        for i, r in enumerate(c4_outs):
            r["sha256"], r["leaks"] = _http_fp(r), _http_leaks(r)
            tot += int((r.get("usage") or {}).get("completion_tokens") or 0)
            eqs.append(r["sha256"] == c1_refs[i]["sha256"])
        agg = round(tot / max(1e-6, wall), 2)
        res["cases"][f"{tag}_c1_refs"] = c1_refs
        res["cases"][f"{tag}_concurrent"] = {"wall_s": round(wall, 3), "total_completion_tokens": tot, "agg_tps": agg, "streams": c4_outs}
        res["parity_checks"][f"{tag}_vs_c1_exact"] = all(eqs)
        print(
            f"[{tag:16s}] wall={wall:.2f}s agg={agg:6.1f}t/s max_gap={max(r['max_gap_ms'] for r in c4_outs):5.1f}ms "
            f"c4==c1:{eqs} shas={[r['sha256'][:8] for r in c4_outs]}",
            flush=True,
        )
    return res


def _raw_http_abort(
    base_url: str,
    payload: dict[str, Any],
    *,
    stop_after_chunks: int | None = None,
    abort_after_s: float | None = None,
) -> None:
    """Open a raw TCP HTTP stream and close it abruptly mid-prefill or mid-stream."""
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    body = dict(payload)
    body["stream"] = True
    raw_body = json.dumps(body).encode("utf-8")
    req_bytes = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(raw_body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("utf-8") + raw_body

    sock = socket.create_connection((host, port), timeout=15.0)
    try:
        sock.sendall(req_bytes)
        if abort_after_s is not None:
            time.sleep(abort_after_s)
            return
        buf = b""
        seen = 0
        target = stop_after_chunks or 4
        while seen < target:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            seen = buf.count(b"data:")
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _run_t4_fairness(base_url: str, model: str, failures: list[str]) -> None:
    nonce = f"FAIR-{int(time.time() * 1000)}"
    short_pl = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": f"[{nonce}] Explain in 2 sentences why 0.1 + 0.2 != 0.3 in IEEE-754."}],
    }
    long_lines = "\n".join(
        f"Record {i:04d} [{nonce}]: val={(i * 2654435761) & 0xFFFF:04x} status=OK."
        for i in range(1, 950)
    )
    long_pl = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": f"Summarize status for {nonce}:\n{long_lines}"}],
    }

    short_ref = _http_chat_stream(base_url, short_pl)
    short_ref["sha256"] = _http_fp(short_ref)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_long = ex.submit(_http_chat_stream, base_url, long_pl, 300.0)
        time.sleep(0.25)
        short_conc = _http_chat_stream(base_url, short_pl)
        long_conc = fut_long.result()

    short_conc["sha256"], short_conc["leaks"] = _http_fp(short_conc), _http_leaks(short_conc)
    long_conc["sha256"], long_conc["leaks"] = _http_fp(long_conc), _http_leaks(long_conc)
    long_ref = _http_chat_stream(base_url, long_pl)
    long_ref["sha256"] = _http_fp(long_ref)

    eq_short = short_conc["sha256"] == short_ref["sha256"]
    eq_long = long_conc["sha256"] == long_ref["sha256"]
    print(
        f"[fairness        ] short ttft={short_conc['ttft_ms']:6.1f}ms dec={short_conc['dec_tps']:5.1f}t/s "
        f"max_gap={short_conc['max_gap_ms']:6.1f}ms eq={eq_short} | long eq={eq_long} leaks={len(short_conc['leaks'])+len(long_conc['leaks'])}",
        flush=True,
    )
    if not eq_short:
        failures.append(f"fairness short SHA mismatch ({short_conc['sha256'][:12]} != {short_ref['sha256'][:12]})")
    if not eq_long:
        failures.append(f"fairness long SHA mismatch ({long_conc['sha256'][:12]} != {long_ref['sha256'][:12]})")
    if short_conc["leaks"] or long_conc["leaks"]:
        failures.append(f"fairness leaked control tokens: {short_conc['leaks'] + long_conc['leaks']}")
    if short_conc["ttft_ms"] > 4000.0 or short_conc["max_gap_ms"] > 2500.0:
        failures.append(
            f"fairness interactive decode starved: ttft={short_conc['ttft_ms']:.1f}ms max_gap={short_conc['max_gap_ms']:.1f}ms"
        )


def _run_t4_multibranch(base_url: str, model: str, failures: list[str]) -> None:
    nonce = f"MB-{int(time.time() * 1000)}"
    rules = "\n".join(
        f"Rule {i:02d} [{nonce}]: Preserve deterministic prefix trees and zero-copy KV retention ({i * 11 % 97})."
        for i in range(1, 55)
    )
    sys_msg = {"role": "system", "content": rules}
    pl_base = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 32,
        "messages": [sys_msg, {"role": "user", "content": "Acknowledge the rules in one short line."}],
    }
    r_base = _http_chat_stream(base_url, pl_base)
    c_after_base = _http_get_json(f"{base_url.rstrip('/')}/v1/cache/stats").get("engine_cache", {})
    asst_base: dict[str, Any] = {"role": "assistant", "content": r_base["content"]}
    if r_base.get("reasoning_content"):
        asst_base["reasoning_content"] = r_base["reasoning_content"]

    pl_a = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 32,
        "messages": pl_base["messages"]
        + [asst_base, {"role": "user", "content": "Branch A: " + ("alpha " * 260) + "Summarize Rule 01."}],
    }
    pl_b = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 32,
        "messages": pl_base["messages"]
        + [asst_base, {"role": "user", "content": "Branch B: State Rule 02 concisely."}],
    }

    r_a = _http_chat_stream(base_url, pl_a)
    c_after_a = _http_get_json(f"{base_url.rstrip('/')}/v1/cache/stats").get("engine_cache", {})
    r_b = _http_chat_stream(base_url, pl_b)
    c_after_b = _http_get_json(f"{base_url.rstrip('/')}/v1/cache/stats").get("engine_cache", {})
    r_b_rep = _http_chat_stream(base_url, pl_b)

    sha_b, sha_b_rep = _http_fp(r_b), _http_fp(r_b_rep)
    a_hits_delta = int(c_after_a.get("hits", 0)) - int(c_after_base.get("hits", 0))
    saved_delta = int(c_after_b.get("tokens_saved", 0)) - int(c_after_a.get("tokens_saved", 0))
    hits_delta = int(c_after_b.get("hits", 0)) - int(c_after_a.get("hits", 0))
    rej = int(c_after_b.get("store_rejections", 0))
    print(
        f"[multibranch     ] base_ptok={r_base['usage'].get('prompt_tokens')} A_ptok={r_a['usage'].get('prompt_tokens')} "
        f"A_hits_delta={a_hits_delta} B_hits_delta={hits_delta} B_saved_delta={saved_delta} B_eq={sha_b==sha_b_rep} store_rej={rej}",
        flush=True,
    )
    if a_hits_delta < 1 or hits_delta < 1 or saved_delta < 700:
        failures.append(
            f"multibranch shared ancestor lost! A_hits={a_hits_delta}, B_hits={hits_delta}, B_saved_delta={saved_delta}"
        )
    if sha_b != sha_b_rep:
        failures.append(f"multibranch Branch B parity mismatch ({sha_b[:12]} != {sha_b_rep[:12]})")
    if rej != 0:
        failures.append(f"multibranch store_rejections={rej} != 0")


def _run_t4_disconnect_check(
    base_url: str,
    model: str,
    failures: list[str],
    *,
    log_path: str = "",
    log_offset: int = 0,
    golden: dict[str, Any] | None = None,
) -> None:
    suite = _build_t4_suite(model)
    mid_stream_pl = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Write a detailed 20-step explanation of MergeSort with Python code."}],
    }
    nonce = f"ABORT-{int(time.time() * 1000)}"
    mid_prefill_pl = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": f"[{nonce}] " + ("token_pad_0123456789 " * 1400)}],
    }

    _raw_http_abort(base_url, mid_stream_pl, stop_after_chunks=4)
    _raw_http_abort(base_url, mid_prefill_pl, abort_after_s=0.15)

    st = _http_chat_stream(base_url, suite["short_greedy"])
    ns = _http_chat_nonstream(base_url, suite["short_greedy"])
    st["sha256"], ns["sha256"] = _http_fp(st), _http_fp(ns)
    if st["sha256"] != ns["sha256"]:
        failures.append(f"disconnect follow-up stream != nonstream ({st['sha256'][:12]} != {ns['sha256'][:12]})")
    if golden is not None:
        g_sha = golden["cases"]["short_greedy"]["stream"]["sha256"]
        if st["sha256"] != g_sha:
            failures.append(f"disconnect follow-up SHA mismatch vs golden ({st['sha256'][:12]} != {g_sha[:12]})")

    time.sleep(1.5)
    status = _http_get_json(f"{base_url.rstrip('/')}/v1/status")
    if status.get("num_running", -1) != 0 or status.get("num_waiting", -1) != 0:
        failures.append(f"disconnect leaked active requests in /v1/status: {status}")

    bad_lines: list[str] = []
    if log_path and os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(log_offset)
            tail = f.read()
        for pat in ("Task was destroyed but it is pending", "ERROR:asyncio", "Traceback (most recent call last)"):
            if pat in tail:
                bad_lines.append(pat)
        if bad_lines:
            failures.append(f"disconnect server log errors: {bad_lines}")
    print(
        f"[disconnect_check] follow_up_eq={st['sha256']==ns['sha256']} running={status.get('num_running')} "
        f"waiting={status.get('num_waiting')} log_errors={bad_lines}",
        flush=True,
    )


def _run_t4_cold_start(cold_model: str, cold_port: int, verify_path: str) -> int:
    print(f"=== GATE T4 COLD-START + DISCONNECT CHECK (port={cold_port}, model={cold_model}) ===", flush=True)
    expanded_model = os.path.expanduser(cold_model)
    served_name = os.path.expanduser("~/loopling/models/loopling-p3")
    base_url = f"http://127.0.0.1:{cold_port}"
    golden = json.load(open(verify_path)) if verify_path else None
    failures: list[str] = []

    with tempfile.NamedTemporaryFile(prefix="vllm_gate_cold_", suffix=".log", delete=False) as tf:
        log_path = tf.name

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'vendor'}:{REPO_ROOT}"
    env["HF_HUB_OFFLINE"] = "1"
    env.setdefault("LOOPLING_SERVER_BATCHDECODE", "1")
    env.setdefault("LOOPLING_MTP", "dflash2")

    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx_inkling2.serve",
        "serve",
        expanded_model,
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(cold_port),
        "--continuous-batching",
        "--max-tokens",
        "32768",
        "--max-request-tokens",
        "1048576",
        "--timeout",
        "7200",
        "--default-temperature",
        "0.7",
        "--default-top-p",
        "0.95",
        "--prefill-step-size",
        "4096",
        "--cache-memory-percent",
        "0.70",
        "--reasoning-parser",
        "inkling2",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "inkling2",
    ]
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        deadline = time.perf_counter() + 180.0
        health: dict[str, Any] | None = None
        while time.perf_counter() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Cold server exited early (code={proc.returncode}), log={log_path}")
            try:
                health = _http_get_json(f"{base_url}/health", timeout=2.0)
                if health.get("status") == "healthy" or health.get("model_name"):
                    break
            except Exception:
                time.sleep(0.5)
        if health is None:
            raise TimeoutError(f"Cold server did not become healthy on port {cold_port}, log={log_path}")

        model_id = health.get("model_name") or served_name
        suite = _build_t4_suite(model_id)
        mt_t1_pl = {
            "model": model_id,
            "temperature": 0.0,
            "max_tokens": 128,
            "messages": [
                {"role": "system", "content": suite["multiturn_sys"]},
                {"role": "user", "content": "Summarize Rule 01 and Rule 51 in one sentence each."},
            ],
        }

        # Cold Request #1 immediately after boot!
        req1 = _http_chat_stream(base_url, suite["short_greedy"])
        req1["sha256"], req1["leaks"] = _http_fp(req1), _http_leaks(req1)
        req2 = _http_chat_stream(base_url, mt_t1_pl)
        req2["sha256"], req2["leaks"] = _http_fp(req2), _http_leaks(req2)
        req1_warm = _http_chat_stream(base_url, suite["short_greedy"])
        req1_warm["sha256"] = _http_fp(req1_warm)
        req2_warm = _http_chat_stream(base_url, mt_t1_pl)
        req2_warm["sha256"] = _http_fp(req2_warm)

        status = _http_get_json(f"{base_url}/v1/status")
        if env.get("LOOPLING_MTP") == "dflash2" and not (status.get("mtp") or {}).get("enabled"):
            failures.append(f"cold-start /v1/status mtp.enabled is False under LOOPLING_MTP=dflash2: {status.get('mtp')}")

        if req1["leaks"] or req2["leaks"]:
            failures.append(f"cold-start leaked tags: {req1['leaks'] + req2['leaks']}")
        if req1["sha256"] != req1_warm["sha256"]:
            failures.append("cold Req #1 != warm Req #1 repeat")
        if req2["sha256"] != req2_warm["sha256"]:
            failures.append("cold Req #2 != warm Req #2 repeat")
        if golden is not None:
            if req1["sha256"] != golden["cases"]["short_greedy"]["stream"]["sha256"]:
                failures.append("cold Req #1 SHA mismatch vs golden")
            if req2["sha256"] != golden["cases"]["multiturn_t1"]["sha256"]:
                failures.append("cold Req #2 SHA mismatch vs golden")

        if req1["ttft_ms"] > 450.0 or req1["max_gap_ms"] > 120.0 or req1["dec_tps"] < req1_warm["dec_tps"] * 0.85:
            failures.append(
                f"cold Req #1 warmup tax detected: ttft={req1['ttft_ms']:.1f}ms max_gap={req1['max_gap_ms']:.1f}ms "
                f"dec={req1['dec_tps']:.1f}t/s (warm={req1_warm['dec_tps']:.1f}t/s)"
            )

        log_f.flush()
        with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
            boot_log = lf.read()
        pos_start = boot_log.find("Starting server at")
        pos_detok = boot_log.find("FastByteLevelDetokenizer verified")
        if pos_detok != -1 and pos_start != -1 and pos_detok > pos_start:
            failures.append("FastByteLevelDetokenizer verified lazily AFTER server started!")
        if "warmup skipped:" in boot_log:
            failures.append("Startup warmup skipped (check server log for warmup warning)!")

        print(
            f"[cold_req1       ] ttft={req1['ttft_ms']:6.1f}ms dec={req1['dec_tps']:5.1f}t/s (warm={req1_warm['dec_tps']:5.1f}) "
            f"max_gap={req1['max_gap_ms']:5.1f}ms sha={req1['sha256'][:12]} mtp={(status.get('mtp') or {}).get('enabled')}",
            flush=True,
        )
        print(
            f"[cold_req2_l1897 ] ttft={req2['ttft_ms']:6.1f}ms dec={req2['dec_tps']:5.1f}t/s (warm={req2_warm['dec_tps']:5.1f}) "
            f"max_gap={req2['max_gap_ms']:5.1f}ms sha={req2['sha256'][:12]}",
            flush=True,
        )

        _run_t4_disconnect_check(base_url, model_id, failures, log_path=log_path, log_offset=0, golden=golden)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        log_f.close()
        if not failures:
            try:
                os.remove(log_path)
            except OSError:
                pass
        else:
            print(f"Preserved failing cold-start server log at {log_path}", flush=True)

    if failures:
        print("\n=== GATE T4 COLD-START FAILED ===", flush=True)
        for f in failures:
            print("  FAIL:", f, flush=True)
        return 1
    print("\nALL GATE T4 COLD-START & DISCONNECT CHECKS PASSED", flush=True)
    return 0


def run_t4(
    base_url: str,
    capture_path: str,
    verify_path: str,
    *,
    fairness: bool = False,
    multibranch: bool = False,
    disconnect_check: bool = False,
    server_log: str = "",
) -> int:
    print(f"=== GATE T4: Live HTTP Golden & Perf Gate ({base_url}) ===", flush=True)
    log_offset = os.path.getsize(server_log) if (disconnect_check and server_log and os.path.isfile(server_log)) else 0
    cur = _capture_t4(base_url)
    if capture_path:
        with open(capture_path, "w") as f:
            json.dump(cur, f, indent=2, ensure_ascii=False)
        print(f"Saved T4 golden baseline to {capture_path}", flush=True)

    failures: list[str] = []
    # Intrinsic checks (always enforced)
    for name in ("short_greedy", "medium_greedy", "tool_call_greedy"):
        st, ns = cur["cases"][name]["stream"], cur["cases"][name]["nonstream"]
        if st["leaks"] or ns["leaks"]:
            failures.append(f"{name}: leaked tags stream={st['leaks']} nonstream={ns['leaks']}")
        if st["sha256"] != ns["sha256"]:
            failures.append(f"{name}: stream != nonstream ({st['sha256'][:12]} vs {ns['sha256'][:12]})")
    if cur["cases"]["multiturn_t1"]["sha256"] != cur["cases"]["multiturn_t1_repeat"]["sha256"]:
        failures.append("multiturn_t1_repeat != multiturn_t1")
    if not cur["parity_checks"]["c4_ge64_vs_c1_exact"]:
        failures.append("c4_ge64 concurrent != c1 refs (L>=64 regime must be 0-mm bit-exact!)")
    t2_rej = (cur["cases"].get("multiturn_cache_stats") or {}).get("after_t2", {}).get("store_rejections", 0)
    if t2_rej != 0:
        failures.append(f"multiturn_t2 store_rejections={t2_rej} != 0")

    golden = json.load(open(verify_path)) if verify_path else None
    # Golden comparison if --verify provided
    if golden is not None:
        print("\n--- Golden Comparison (baseline -> current) ---", flush=True)
        for name in ("short_greedy", "medium_greedy", "tool_call_greedy"):
            g_st = golden["cases"][name]["stream"]
            c_st = cur["cases"][name]["stream"]
            c_ns = cur["cases"][name]["nonstream"]
            ok = c_st["sha256"] == g_st["sha256"] and c_ns["sha256"] == golden["cases"][name]["nonstream"]["sha256"]
            print(
                f"  {name:16s}: TTFT {g_st['ttft_ms']:6.1f} -> {c_st['ttft_ms']:6.1f}ms | "
                f"dec {g_st['dec_tps']:5.1f} -> {c_st['dec_tps']:5.1f} t/s | "
                f"max_gap {g_st['max_gap_ms']:5.1f} -> {c_st['max_gap_ms']:5.1f}ms | exact={ok}",
                flush=True,
            )
            if not ok:
                failures.append(f"{name}: SHA mismatch vs golden ({c_st['sha256'][:12]} != {g_st['sha256'][:12]})")

        for name in ("multiturn_t1", "multiturn_t1_repeat", "multiturn_t2"):
            g = golden["cases"][name]
            c = cur["cases"][name]
            ok = c["sha256"] == g["sha256"]
            print(
                f"  {name:16s}: TTFT {g['ttft_ms']:6.1f} -> {c['ttft_ms']:6.1f}ms | "
                f"dec {g['dec_tps']:5.1f} -> {c['dec_tps']:5.1f} t/s | exact={ok}",
                flush=True,
            )
            if not ok:
                failures.append(f"{name}: SHA mismatch vs golden ({c['sha256'][:12]} != {g['sha256'][:12]})")

        # Compare C=1 refs for c4_short (and c4_ge64 if present in golden)
        g_c1_key = "c4_short_c1_refs" if "c4_short_c1_refs" in golden["cases"] else "c4_c1_refs"
        if g_c1_key in golden["cases"]:
            for i in range(4):
                g = golden["cases"][g_c1_key][i]
                c = cur["cases"]["c4_short_c1_refs"][i]
                if c["sha256"] != g["sha256"]:
                    failures.append(f"c4_short_c1_ref[{i}]: SHA mismatch vs golden ({c['sha256'][:12]} != {g['sha256'][:12]})")
        if "c4_ge64_c1_refs" in golden["cases"]:
            for i in range(4):
                g = golden["cases"]["c4_ge64_c1_refs"][i]
                c1 = cur["cases"]["c4_ge64_c1_refs"][i]
                c4 = cur["cases"]["c4_ge64_concurrent"]["streams"][i]
                if c1["sha256"] != g["sha256"] or c4["sha256"] != g["sha256"]:
                    failures.append(f"c4_ge64[{i}]: SHA mismatch vs golden")

    model = cur["meta"]["model"]
    if fairness:
        _run_t4_fairness(base_url, model, failures)
    if multibranch:
        _run_t4_multibranch(base_url, model, failures)
    if disconnect_check:
        _run_t4_disconnect_check(
            base_url,
            model,
            failures,
            log_path=server_log,
            log_offset=log_offset,
            golden=golden,
        )

    if failures:
        print("\n=== GATE T4 FAILED ===", flush=True)
        for f in failures:
            print("  FAIL:", f, flush=True)
        return 1
    print("\nALL GATE T4 CHECKS PASSED (100% BIT-EXACT)", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Bit-exactness verification gate for vllm-mlx + loopling-opt")
    ap.add_argument("gate", choices=("t1", "t2", "t3", "t4"))
    ap.add_argument("--model", default=os.environ.get("INKLING2_MODEL", "~/loopling/models/loopling-p3-tess"))
    ap.add_argument("--url", default="http://512s2.tail91699.ts.net:8080")
    ap.add_argument("--capture", default="")
    ap.add_argument("--verify", default="")
    ap.add_argument("--with-spec", action="store_true")
    ap.add_argument("--skip-131k", action="store_true")
    ap.add_argument("--fairness", action="store_true")
    ap.add_argument("--multibranch", action="store_true")
    ap.add_argument("--disconnect-check", action="store_true")
    ap.add_argument("--server-log", default="")
    ap.add_argument("--cold-start", action="store_true")
    ap.add_argument("--cold-port", type=int, default=18080)
    ap.add_argument("--cold-model", default="~/loopling/models/loopling-p3-tess")
    args = ap.parse_args()

    if args.gate == "t1":
        sys.exit(run_t1(args.model))
    elif args.gate == "t2":
        sys.exit(run_t2(args.model, skip_131k=args.skip_131k))
    elif args.gate == "t3":
        sys.exit(run_t3(with_spec=args.with_spec))
    elif args.gate == "t4":
        if args.cold_start:
            sys.exit(_run_t4_cold_start(args.cold_model, args.cold_port, args.verify))
        sys.exit(
            run_t4(
                args.url,
                args.capture,
                args.verify,
                fairness=args.fairness,
                multibranch=args.multibranch,
                disconnect_check=args.disconnect_check,
                server_log=args.server_log,
            )
        )


if __name__ == "__main__":
    main()
