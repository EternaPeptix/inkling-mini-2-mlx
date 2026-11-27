"""Bit-exact HTTP/SSE, detokenizer, tokenization, and runtime optimizations for vllm-mlx."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import socket
import time
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

_INSTALLED = False

# Threshold above which prompt tokenization runs on a worker thread so the
# asyncio event loop stays responsive to concurrent SSE streams.
_ASYNC_TOKENIZE_CHAR_THRESHOLD = 32768


def install() -> None:
    """Apply idempotent bit-exact streaming, detokenizer, and runtime hooks."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _patch_tcp_nodelay()
    _patch_fast_bytelevel_detokenizer()
    _patch_stream_chat_tokenization()
    _patch_wired_memory_and_gc_freeze()
    _patch_metrics_middleware_passthrough()
    _patch_lightweight_disconnect_guard()
    _patch_quiet_hot_logs()
    logger.info("[vllm_mlx_inkling2] stream_opt hooks installed")


# ---------------------------------------------------------------------------
# P1: TCP_NODELAY = 1 on accepted uvicorn HTTP sockets
# ---------------------------------------------------------------------------


def _set_nodelay(transport: Any) -> None:
    sock = transport.get_extra_info("socket") if transport is not None else None
    if sock is None:
        return
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


def _patch_tcp_nodelay() -> None:
    """Disable Nagle's algorithm on accepted uvicorn client sockets."""
    patched: list[str] = []
    for mod_path, cls_name in (
        ("uvicorn.protocols.http.h11_impl", "H11Protocol"),
        ("uvicorn.protocols.http.httptools_impl", "HttpToolsProtocol"),
    ):
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            proto_cls = getattr(mod, cls_name, None)
            if proto_cls is None or getattr(proto_cls, "_inkling2_nodelay", False):
                continue
            orig_connection_made = proto_cls.connection_made

            def _connection_made_nodelay(self, transport, _orig=orig_connection_made):
                _set_nodelay(transport)
                return _orig(self, transport)

            proto_cls.connection_made = _connection_made_nodelay
            proto_cls._inkling2_nodelay = True
            patched.append(cls_name)
        except Exception:
            continue
    if patched:
        logger.info("[stream_opt] TCP_NODELAY enabled on %s", ", ".join(patched))


# ---------------------------------------------------------------------------
# P2: 100% Bit-Exact O(1)/tok FastByteLevelDetokenizer
# ---------------------------------------------------------------------------


class FastByteLevelDetokenizer:
    """O(1)-per-token ByteLevel streaming detokenizer matching `tokenizer.decode` bit-for-bit."""

    __slots__ = ("_table", "_pending", "_parts", "_seg_buf", "tokens")

    def __init__(self, byte_table: tuple[bytes, ...]) -> None:
        self._table = byte_table
        self._pending: bytes = b""
        self._parts: list[str] = []
        self._seg_buf: list[str] = []
        self.tokens: list[int] = []

    def reset(self) -> None:
        self._pending = b""
        self._parts.clear()
        self._seg_buf.clear()
        self.tokens.clear()

    def add_token(self, token: int) -> None:
        self.tokens.append(token)
        table = self._table
        b = table[token] if 0 <= token < len(table) else b"!"
        if self._pending:
            b = self._pending + b
            self._pending = b""
        while b:
            try:
                s = b.decode("utf-8")
                self._parts.append(s)
                self._seg_buf.append(s)
                return
            except UnicodeDecodeError as e:
                if e.start > 0:
                    s = b[: e.start].decode("utf-8")
                    self._parts.append(s)
                    self._seg_buf.append(s)
                    b = b[e.start :]
                    continue
                if e.reason == "unexpected end of data":
                    self._pending = b
                    return
                self._parts.append("\ufffd")
                self._seg_buf.append("\ufffd")
                b = b[e.end :]

    @property
    def last_segment(self) -> str:
        buf = self._seg_buf
        if not buf:
            return ""
        if len(buf) == 1:
            seg = buf[0]
            buf.clear()
            return seg
        seg = "".join(buf)
        buf.clear()
        return seg

    @property
    def text(self) -> str:
        parts = self._parts
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return "".join(parts)

    def finalize(self) -> None:
        if self._pending:
            s = self._pending.decode("utf-8", "replace")
            self._parts.append(s)
            self._seg_buf.append(s)
            self._pending = b""


def _is_bytelevel_decoder(tokenizer: Any) -> bool:
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        return False
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        return False
    try:
        cfg = json.loads(to_str())
    except Exception:
        return False
    dec = cfg.get("decoder")
    return isinstance(dec, dict) and dec.get("type") == "ByteLevel"


def build_verified_byte_table(tokenizer: Any) -> tuple[bytes, ...] | None:
    """Build and verify a 100%-exact `id -> bytes` table for a ByteLevel tokenizer."""
    if not _is_bytelevel_decoder(tokenizer):
        return None
    try:
        from mlx_lm.tokenizer_utils import _byte_decoder

        bd = _byte_decoder()
        n_vocab = len(tokenizer)
        if n_vocab <= 0:
            return None
        added_vocab = tokenizer.get_added_vocab()
        added_ids = set(added_vocab.values())
        raw_tokens = tokenizer.convert_ids_to_tokens(list(range(n_vocab)))

        table_list: list[bytes] = []
        for tid, tok_str in enumerate(raw_tokens):
            if tok_str is None:
                table_list.append(b"")
            elif tid in added_ids:
                table_list.append(tok_str.encode("utf-8"))
            else:
                ba = bytearray()
                for c in tok_str:
                    v = bd.get(c)
                    if v is not None:
                        ba.append(v)
                    else:
                        ba.extend(c.encode("utf-8"))
                table_list.append(bytes(ba))
        table = tuple(table_list)

        # Full-vocab parity verification (takes ~0.15s once at startup on 200K vocab).
        batch_decode = getattr(tokenizer, "batch_decode", None)
        if callable(batch_decode):
            refs = batch_decode([[i] for i in range(n_vocab)])
            for tid, ref in enumerate(refs):
                if table[tid].decode("utf-8", "replace") != ref:
                    logger.warning(
                        "[stream_opt] ByteLevel parity mismatch at tid=%d (%r != %r); "
                        "falling back to NaiveStreamingDetokenizer",
                        tid,
                        table[tid].decode("utf-8", "replace"),
                        ref,
                    )
                    return None
        else:
            probe_ids = list(range(min(512, n_vocab))) + list(added_ids)
            for tid in probe_ids:
                ref = tokenizer.decode([tid])
                if table[tid].decode("utf-8", "replace") != ref:
                    return None

        return table
    except Exception as exc:
        logger.warning("[stream_opt] Failed to build FastByteLevelDetokenizer table: %s", exc)
        return None


def _unwrap_hf_tokenizer(tokenizer: Any) -> Any:
    """Unwrap TokenizerWrapper or HF Processor to the underlying HF tokenizer."""
    tok = tokenizer
    for _ in range(3):
        if _is_bytelevel_decoder(tok):
            return tok
        inner = getattr(tok, "_tokenizer", None)
        if inner is not None and _is_bytelevel_decoder(inner):
            return inner
        inner_t = getattr(tok, "tokenizer", None)
        if inner_t is not None and inner_t is not tok:
            tok = inner_t
            continue
        break
    return tok


def _patch_fast_bytelevel_detokenizer() -> None:
    """Replace O(T^2) NaiveStreamingDetokenizer in Scheduler with FastByteLevelDetokenizer."""
    from vllm_mlx import scheduler as vsched

    if getattr(vsched.Scheduler, "_inkling2_fast_detok", False):
        return

    orig_get_detok = vsched.Scheduler._get_detokenizer
    _table_cache: dict[int, tuple[bytes, ...] | None] = {}

    def _fast_get_detokenizer(self, request_id: str | None = None) -> Any:
        pool = getattr(self, "_detokenizer_pool", None)
        if request_id is not None and pool is not None:
            detok = pool.get(request_id)
            if detok is not None:
                return detok

        raw = getattr(self, "_actual_tokenizer", None)
        if raw is None:
            raw = getattr(self, "tokenizer", None)
        if raw is None:
            return orig_get_detok(self, request_id) if request_id is not None else None

        tok_key = id(raw)
        if tok_key not in _table_cache:
            actual_tok = _unwrap_hf_tokenizer(raw)
            actual_key = id(actual_tok)
            if actual_key in _table_cache:
                table = _table_cache[actual_key]
            else:
                table = build_verified_byte_table(actual_tok)
                _table_cache[actual_key] = table
                if table is not None:
                    logger.info(
                        "[stream_opt] FastByteLevelDetokenizer verified for %d tokens",
                        len(table),
                    )
            _table_cache[tok_key] = table
        table = _table_cache[tok_key]
        if table is None:
            if request_id is None:
                from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

                return NaiveStreamingDetokenizer(raw)
            return orig_get_detok(self, request_id)

        detok = FastByteLevelDetokenizer(table)
        if request_id is not None and pool is not None:
            pool[request_id] = detok
        return detok

    vsched.Scheduler._get_detokenizer = _fast_get_detokenizer
    vsched.Scheduler._inkling2_fast_detok = True


# ---------------------------------------------------------------------------
# P3: Eliminate 3x redundant chat-template render & tokenization in stream_chat
# ---------------------------------------------------------------------------


def _str_lcp_len(a: str, b: str) -> int:
    """Fast binary-search longest common prefix length for large strings."""
    lo = 0
    hi = min(len(a), len(b))
    if hi == 0 or a[0] != b[0]:
        return 0
    if a[:hi] == b[:hi]:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _token_lcp(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _fast_prefix_boundary_from_prompts(
    tokenizer: Any,
    real_prompt: str,
    real_tokens: list[int],
    dummy_prompt: str,
) -> int:
    """Compute exact token LCP between real_prompt and dummy_prompt with O(tail) work."""
    char_lcp = _str_lcp_len(real_prompt, dummy_prompt)
    if char_lcp == 0:
        return 0

    # Find an added/special token ending at or before char_lcp: HuggingFace tokenizers
    # split strictly at added tokens before BPE, guaranteeing zero cross-boundary merges
    # for both real_prompt and dummy_prompt.
    get_added = getattr(tokenizer, "get_added_vocab", None)
    if not callable(get_added):
        inner = getattr(tokenizer, "_tokenizer", None)
        get_added = getattr(inner, "get_added_vocab", None) if inner is not None else None
    if callable(get_added):
        try:
            added_vocab = get_added()
            search_start = max(0, char_lcp - 512)
            end_pos = real_prompt.rfind("|>", search_start, char_lcp)
            while end_pos >= 0:
                start_pos = real_prompt.rfind("<|", max(0, end_pos - 64), end_pos)
                if start_pos >= 0:
                    candidate = real_prompt[start_pos : end_pos + 2]
                    if candidate in added_vocab:
                        split_char = end_pos + 2
                        real_tail = real_prompt[split_char:]
                        dummy_tail = dummy_prompt[split_char:]
                        real_tail_ids = tokenizer.encode(
                            real_tail, add_special_tokens=False
                        )
                        tail_len = len(real_tail_ids)
                        if (
                            tail_len <= len(real_tokens)
                            and real_tokens[len(real_tokens) - tail_len :]
                            == real_tail_ids
                        ):
                            split_tok_idx = len(real_tokens) - tail_len
                            dummy_tail_ids = tokenizer.encode(
                                dummy_tail, add_special_tokens=False
                            )
                            return split_tok_idx + _token_lcp(
                                real_tail_ids, dummy_tail_ids
                            )
                        break
                end_pos = real_prompt.rfind("|>", search_start, end_pos)
        except Exception:
            pass

    # Fallback: exact full encode of dummy_prompt.
    dummy_tokens = tokenizer.encode(dummy_prompt)
    return _token_lcp(real_tokens, dummy_tokens)


def _patch_stream_chat_tokenization() -> None:
    """Eliminate 3x template render + 3x tokenization on multi-turn stream_chat requests."""
    from vllm_mlx.api.utils import extract_multimodal_content
    from vllm_mlx.engine import batched as engine_batched

    convert_tools_for_template = engine_batched.convert_tools_for_template

    BatchedEngine = engine_batched.BatchedEngine
    if getattr(BatchedEngine, "_inkling2_fast_stream_chat", False):
        return

    orig_stream_chat = BatchedEngine.stream_chat

    async def _fast_stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ):
        if not self._loaded:
            await self.start()

        # Delegate MLLM or multimodal requests to the untouched original implementation.
        if self._is_mllm or images or videos:
            async for output in orig_stream_chat(
                self,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                **kwargs,
            ):
                yield output
            return

        _, extracted_images, extracted_videos, extracted_audios = (
            extract_multimodal_content(messages)
        )
        if extracted_images or extracted_videos or extracted_audios:
            async for output in orig_stream_chat(
                self,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                **kwargs,
            ):
                yield output
            return

        template_tools = convert_tools_for_template(tools) if tools else None
        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})
        enable_thinking = kwargs.pop("enable_thinking", None)

        # Render the real prompt ONCE.
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            num_images=0,
            num_audios=0,
            chat_template_kwargs=chat_template_kwargs,
            enable_thinking=enable_thinking,
        )

        # Encode ONCE (offloading large prompts so concurrent SSE streams never stall).
        tok = self.tokenizer
        if hasattr(tok, "tokenizer") and not hasattr(tok, "encode"):
            tok = tok.tokenizer
        enc_tok = self.tokenizer if hasattr(self.tokenizer, "encode") else tok

        # Determine if multi-turn prefix_boundary is needed.
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        def _tokenize_and_boundary() -> tuple[list[int], int]:
            real_tokens = enc_tok.encode(prompt)
            if last_user_idx is None or last_user_idx == 0:
                return real_tokens, 0
            try:
                dummy_messages = list(messages)
                dummy_messages[last_user_idx] = {
                    **messages[last_user_idx],
                    "content": "XXXXXXXXXX",
                }
                dummy_prompt = self._apply_chat_template(
                    dummy_messages,
                    template_tools,
                    num_images=0,
                    num_audios=0,
                    chat_template_kwargs=chat_template_kwargs,
                    enable_thinking=enable_thinking,
                )
                lcp = _fast_prefix_boundary_from_prompts(
                    enc_tok, prompt, real_tokens, dummy_prompt
                )
                return real_tokens, lcp
            except Exception:
                return real_tokens, 0

        if len(prompt) >= _ASYNC_TOKENIZE_CHAR_THRESHOLD:
            prompt_tokens, prefix_boundary = await asyncio.to_thread(_tokenize_and_boundary)
        else:
            prompt_tokens, prefix_boundary = _tokenize_and_boundary()

        if prefix_boundary > 0:
            kwargs["prefix_boundary"] = prefix_boundary

        async for output in self.stream_generate(
            prompt=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            images=None,
            videos=None,
            audio=None,
            **kwargs,
        ):
            yield output

    BatchedEngine.stream_chat = _fast_stream_chat
    BatchedEngine._inkling2_fast_stream_chat = True


# ---------------------------------------------------------------------------
# P5 (non-scheduler): mx.set_wired_limit + gc.freeze() after model load
# ---------------------------------------------------------------------------


def _patch_wired_memory_and_gc_freeze() -> None:
    """Pin Metal wired memory limit and freeze permanent Python objects after load."""
    from vllm_mlx import server as vserver
    from vllm_mlx.engine import batched as engine_batched

    BatchedEngine = engine_batched.BatchedEngine
    if not getattr(BatchedEngine, "_inkling2_wired_limit", False):
        orig_configure = BatchedEngine._configure_metal_memory_limits

        def _configure_with_wired_limit(self) -> None:
            orig_configure(self)
            try:
                import mlx.core as mx

                if mx.metal.is_available() and hasattr(mx, "set_wired_limit"):
                    device_info = mx.device_info()
                    max_rec = int(
                        device_info.get(
                            "max_recommended_working_set_size",
                            device_info.get("memory_size", 0),
                        )
                    )
                    if max_rec > 0:
                        mx.set_wired_limit(max_rec)
                        logger.info(
                            "[stream_opt] Metal wired_limit set to %.1fGB",
                            max_rec / 1e9,
                        )
            except Exception as exc:
                logger.warning("[stream_opt] Failed to set Metal wired_limit: %s", exc)

        BatchedEngine._configure_metal_memory_limits = _configure_with_wired_limit
        BatchedEngine._inkling2_wired_limit = True

    if not getattr(vserver, "_inkling2_gc_freeze_hook", False):
        orig_load_model = vserver.load_model

        def _load_model_and_freeze(*args, **kwargs):
            result = orig_load_model(*args, **kwargs)
            try:
                gc.collect()
                gc.freeze()
                gc.set_threshold(50_000, 20, 20)
                logger.info(
                    "[stream_opt] gc.freeze() complete (frozen=%d, threshold=%s)",
                    gc.get_freeze_count(),
                    gc.get_threshold(),
                )
            except Exception as exc:
                logger.debug("[stream_opt] gc.freeze skipped: %s", exc)
            return result

        vserver.load_model = _load_model_and_freeze
        vserver._inkling2_gc_freeze_hook = True


# ---------------------------------------------------------------------------
# P6a: Pure-ASGI passthrough for _metrics_middleware when metrics disabled
# ---------------------------------------------------------------------------


class _FastMetricsASGIMiddleware:
    """Bypass Starlette BaseHTTPMiddleware memory-stream channel when metrics are disabled."""

    def __init__(self, app, orig_dispatch) -> None:
        from starlette.middleware.base import BaseHTTPMiddleware

        self.app = app
        self._fallback = BaseHTTPMiddleware(app, dispatch=orig_dispatch)

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        from vllm_mlx import server as vserver

        if not vserver._metrics.enabled:
            await self.app(scope, receive, send)
            return
        await self._fallback(scope, receive, send)


def _patch_metrics_middleware_passthrough() -> None:
    """Replace BaseHTTPMiddleware(_metrics_middleware) with pure-ASGI fast passthrough."""
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from vllm_mlx import server as vserver

    if getattr(vserver.app, "_inkling2_fast_metrics_mw", False):
        return

    new_mw = []
    replaced = False
    for mw in vserver.app.user_middleware:
        if (
            mw.cls is BaseHTTPMiddleware
            and mw.kwargs.get("dispatch") is vserver._metrics_middleware
        ):
            new_mw.append(
                Middleware(_FastMetricsASGIMiddleware, orig_dispatch=vserver._metrics_middleware)
            )
            replaced = True
        else:
            new_mw.append(mw)

    if replaced:
        vserver.app.user_middleware = new_mw
        vserver.app.middleware_stack = None
        vserver.app._inkling2_fast_metrics_mw = True
        logger.info("[stream_opt] Pure-ASGI passthrough installed for _metrics_middleware")


# ---------------------------------------------------------------------------
# P6b: Lightweight _disconnect_guard (zero per-chunk Task/Timer during fast streaming)
# ---------------------------------------------------------------------------


def _patch_lightweight_disconnect_guard() -> None:
    """Avoid per-chunk asyncio.Task + asyncio.wait overhead during active streaming.

    Preserves all semantics of vllm_mlx.server._disconnect_guard:
    - 0.5s disconnect polling + 5.0s SSE `: heartbeat\\n\\n` during slow phases (prefill/stalls)
    - output inactivity timeout (`timeout` / `_default_timeout`)
    - deferred `aclose()` (0.5s) for Metal thread safety
    - `cleanup` callback execution
    """
    from vllm_mlx import server as vserver

    if getattr(vserver, "_inkling2_fast_disconnect_guard", False):
        return

    async def _fast_disconnect_guard(
        generator: AsyncIterator[str],
        raw_request: Any,
        poll_interval: float = 0.5,
        heartbeat_interval: float = 5.0,
        cleanup=None,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        chunk_timeout = timeout or vserver._default_timeout

        chunk_count = 0
        heartbeat_count = 0
        last_chunk_at = t0
        last_hb_at = t0

        aiter = generator.__aiter__()
        # Single persistent pump task pulls from `aiter` into a 1-slot Future handoff,
        # avoiding per-chunk Task creation & asyncio.wait() set/timer churn.
        item_ready: asyncio.Future[tuple[str, Any]] | None = None
        pending_item: tuple[str, Any] | None = None
        consumed_event = asyncio.Event()

        async def _pump() -> None:
            nonlocal item_ready, pending_item
            try:
                while True:
                    chunk = await aiter.__anext__()
                    item = ("chunk", chunk)
                    if item_ready is not None and not item_ready.done():
                        fut = item_ready
                        item_ready = None
                        fut.set_result(item)
                    else:
                        pending_item = item
                    await consumed_event.wait()
                    consumed_event.clear()
            except StopAsyncIteration:
                end_item = ("eof", None)
                if item_ready is not None and not item_ready.done():
                    fut = item_ready
                    item_ready = None
                    fut.set_result(end_item)
                else:
                    pending_item = end_item
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                err_item = ("err", exc)
                if item_ready is not None and not item_ready.done():
                    fut = item_ready
                    item_ready = None
                    fut.set_result(err_item)
                else:
                    pending_item = err_item

        pump_task = asyncio.create_task(_pump())
        poll_handle: asyncio.TimerHandle | None = None
        disconnected = False

        def _poll_tick() -> None:
            nonlocal poll_handle, disconnected, item_ready
            if vserver._is_client_disconnected(raw_request):
                disconnected = True
                if item_ready is not None and not item_ready.done():
                    fut = item_ready
                    item_ready = None
                    fut.set_result(("disc", None))
                return
            now = time.monotonic()
            idle = now - last_chunk_at
            if idle >= chunk_timeout:
                if item_ready is not None and not item_ready.done():
                    fut = item_ready
                    item_ready = None
                    fut.set_result(("timeout", idle))
                return
            if (
                now - last_hb_at >= heartbeat_interval
                and item_ready is not None
                and not item_ready.done()
            ):
                fut = item_ready
                item_ready = None
                fut.set_result(("hb", None))
            poll_handle = loop.call_later(poll_interval, _poll_tick)

        poll_handle = loop.call_later(poll_interval, _poll_tick)

        try:
            while True:
                if disconnected or vserver._is_client_disconnected(raw_request):
                    logger.info(
                        "[disconnect_guard] CLIENT DISCONNECTED after %d chunks, elapsed=%.1fs",
                        chunk_count,
                        time.monotonic() - t0,
                    )
                    break

                if pending_item is not None:
                    kind, payload = pending_item
                    pending_item = None
                else:
                    fut = loop.create_future()
                    item_ready = fut
                    kind, payload = await fut

                if kind == "chunk":
                    chunk_count += 1
                    now = time.monotonic()
                    last_chunk_at = now
                    last_hb_at = now
                    consumed_event.set()
                    yield payload
                    continue

                if kind == "eof":
                    break

                if kind == "err":
                    logger.error(
                        "[disconnect_guard] generator raised %s: %s, after %d chunks",
                        type(payload).__name__,
                        payload,
                        chunk_count,
                    )
                    break

                if kind == "disc":
                    logger.info(
                        "[disconnect_guard] CLIENT DISCONNECTED after %d chunks, elapsed=%.1fs",
                        chunk_count,
                        time.monotonic() - t0,
                    )
                    break

                if kind == "timeout":
                    logger.warning(
                        "[disconnect_guard] OUTPUT INACTIVITY TIMEOUT after %.1fs, %d chunks",
                        payload,
                        chunk_count,
                    )
                    break

                if kind == "hb":
                    now = time.monotonic()
                    if now - last_hb_at >= heartbeat_interval:
                        heartbeat_count += 1
                        last_hb_at = now
                        yield ": heartbeat\n\n"
                    continue
        except GeneratorExit:
            pass
        finally:
            if poll_handle is not None:
                poll_handle.cancel()
            if not pump_task.done():
                pump_task.cancel()
            _gen_to_close = aiter

            async def _deferred_generator_close() -> None:
                await asyncio.sleep(0.5)
                try:
                    await _gen_to_close.aclose()
                except Exception:
                    pass

            asyncio.create_task(_deferred_generator_close())
            if cleanup is not None:
                res = cleanup()
                if asyncio.iscoroutine(res):
                    await res

    vserver._disconnect_guard = _fast_disconnect_guard
    vserver._inkling2_fast_disconnect_guard = True
    logger.info("[stream_opt] Lightweight _disconnect_guard installed")


# ---------------------------------------------------------------------------
# P5b: Quiet verbose per-request INFO log spam (keeps warnings/errors/startup)
# ---------------------------------------------------------------------------


class _QuietHotLogFilter(logging.Filter):
    _SUPPRESS_PREFIXES = (
        "[REQUEST] last user message preview:",
        "[disconnect_guard] START",
        "[disconnect_guard] first chunk arrived",
        "[disconnect_guard] poll #",
        "[disconnect_guard] generator exhausted normally",
        "[disconnect_guard] CLEANUP done",
        "[stream_outputs]",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        if msg.startswith("[stream_outputs]"):
            # Keep aborts, cancellations, and timeouts visible.
            return "ABORTING" in msg or "TIMEOUT" in msg or "CancelledError" in msg
        return not msg.startswith(self._SUPPRESS_PREFIXES)


def _patch_quiet_hot_logs() -> None:
    """Suppress noisy per-request INFO chatter unless LOOPLING_VERBOSE_HTTP_LOGS=1."""
    if os.environ.get("LOOPLING_VERBOSE_HTTP_LOGS", "0") == "1":
        return
    flt = _QuietHotLogFilter()
    for name in ("vllm_mlx.server", "vllm_mlx.engine_core"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _QuietHotLogFilter) for f in lg.filters):
            lg.addFilter(flt)
