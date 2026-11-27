"""Tool call parser for Inkling 2 (`<|content_invoke_tool_json|>`)."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from vllm_mlx.tool_parsers.abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

_TAG_TOOL_JSON = "<|content_invoke_tool_json|>"
_TAG_TOOL_TEXT = "<|content_invoke_tool_text|>"
_TAG_END = "<|end_message|>"
_TAG_EOS = "<|content_model_end_sampling|>"

_TOOL_JSON_BLOCK_RE = re.compile(
    r"<\|content_invoke_tool_json\|>(.*?)(?:<\|end_message\|>|<\|content_model_end_sampling\|>|(?=<\|content_invoke_tool_|<\|content_text\|>|<\|content_thinking\|>|<\|message_model\|>)|\Z)",
    re.DOTALL,
)
_TOOL_TEXT_BLOCK_RE = re.compile(
    r"<\|content_invoke_tool_text\|>.*?(?:<\|end_message\|>|<\|content_model_end_sampling\|>|(?=<\|content_invoke_tool_|<\|content_text\|>|<\|content_thinking\|>|<\|message_model\|>)|\Z)",
    re.DOTALL,
)
_THINK_FALLBACK_RE = re.compile(
    r"(?:<\|message_model\|>[^<]*)?<\|content_thinking\|>.*?(?:<\|end_message\|>|<\|content_model_end_sampling\|>|\Z)",
    re.DOTALL,
)
_HEADER_RECIPIENT_RE = re.compile(
    r"<\|message_model\|>(?:(?!(?:<\|content_|<\|end_message\|>)).)*(?=<\|content_)",
    re.DOTALL,
)
_STRUCTURAL_TAG_RE = re.compile(
    r"<\|message_model\|>|<\|content_text\|>|<\|content_thinking\|>|<\|end_message\|>|<\|content_model_end_sampling\|>"
)

_OPEN_TAGS = (_TAG_TOOL_JSON, _TAG_TOOL_TEXT)
_HOLD_TAGS = (
    _TAG_TOOL_JSON,
    _TAG_TOOL_TEXT,
    _TAG_END,
    _TAG_EOS,
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
)
_MAX_HOLD_LEN = max(len(t) for t in _HOLD_TAGS)
_CLOSE_MARKERS = (
    (_TAG_END, len(_TAG_END)),
    (_TAG_EOS, len(_TAG_EOS)),
    (_TAG_TOOL_JSON, 0),
    (_TAG_TOOL_TEXT, 0),
)
_MAX_CLOSE_LEN = max(len(m) for m, _ in _CLOSE_MARKERS)


def _strip_structural_tags(text: str) -> str:
    return _STRUCTURAL_TAG_RE.sub("", text) if "<|" in text else text


def _gen_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _normalize_call_item(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    if not isinstance(name, str) or not name:
        return None
    if "args" in item:
        raw_args = item["args"]
    elif "arguments" in item:
        raw_args = item["arguments"]
    else:
        raw_args = {}
    if raw_args is None or raw_args == "":
        args_obj: Any = {}
    elif isinstance(raw_args, str):
        stripped = raw_args.strip()
        if not stripped:
            args_obj = {}
        else:
            try:
                args_obj = json.loads(stripped)
            except json.JSONDecodeError:
                args_obj = {"value": raw_args}
    else:
        args_obj = raw_args
    args_str = json.dumps(args_obj, ensure_ascii=False) if isinstance(args_obj, (dict, list)) else str(args_obj)
    call_id = str(item["id"]) if item.get("id") is not None else _gen_tool_id()
    return {"id": call_id, "name": name, "arguments": args_str}


def _parse_json_payload(raw_payload: str) -> list[dict[str, str]]:
    text = raw_payload.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = parsed if isinstance(parsed, list) else [parsed]
    out: list[dict[str, str]] = []
    for item in items:
        norm = _normalize_call_item(item)
        if norm is not None:
            out.append(norm)
    return out


def _partial_open_len(text: str) -> int:
    idx = text.rfind("<|")
    if idx >= 0:
        tail = text[idx:]
        if len(tail) < _MAX_HOLD_LEN and any(
            t.startswith(tail) and t != tail for t in _HOLD_TAGS
        ):
            return len(tail)
    if text.endswith("<"):
        return 1
    return 0


@ToolParserManager.register_module(["inkling2"])
class Inkling2ToolParser(ToolParser):
    """Tool parser for Inkling 2 (`<|content_invoke_tool_json|>` blocks)."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    REQUIRES_EAGER_STREAMING = False
    STREAMING_MARKERS = (_TAG_TOOL_JSON, _TAG_TOOL_TEXT)

    def __init__(self, tokenizer: Any | None = None) -> None:
        super().__init__(tokenizer)
        self._fed_len: int = 0
        self._pending_text: str = ""
        self._in_tool_kind: str | None = None
        self._tool_chunks: list[str] = []
        self._tool_tail: str = ""

    def reset(self) -> None:
        super().reset()
        self._fed_len = 0
        self._pending_text = ""
        self._in_tool_kind = None
        self._tool_chunks = []
        self._tool_tail = ""

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        cleaned = _THINK_FALLBACK_RE.sub("", model_output)
        cleaned = _HEADER_RECIPIENT_RE.sub("", cleaned)

        tool_calls: list[dict[str, str]] = []
        for match in _TOOL_JSON_BLOCK_RE.finditer(cleaned):
            tool_calls.extend(_parse_json_payload(match.group(1)))

        cleaned = _TOOL_JSON_BLOCK_RE.sub("", cleaned)
        cleaned = _TOOL_TEXT_BLOCK_RE.sub("", cleaned)
        cleaned = _STRUCTURAL_TAG_RE.sub("", cleaned).strip()

        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=cleaned or None,
        )

    def _emit_tool_payload(self, payload: str, new_calls: list[dict[str, Any]]) -> None:
        parsed = _parse_json_payload(payload)
        for tc in parsed:
            idx = len(self.prev_tool_call_arr)
            self.prev_tool_call_arr.append(tc)
            new_calls.append(
                {
                    "index": idx,
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
            )

    def _scan_stream(self, current_text: str, *, allow_unclosed: bool) -> dict[str, Any] | None:
        if len(current_text) < self._fed_len:
            # Guard against callers resetting current_text externally.
            self._fed_len = 0
            self._pending_text = ""
            self._in_tool_kind = None
            self._tool_chunks = []
            self._tool_tail = ""

        new_slice = current_text[self._fed_len :]
        self._fed_len = len(current_text)

        # Ultra-fast path: outside tool block with no buffered prefix and no '<' in delta.
        if (
            self._in_tool_kind is None
            and not self._pending_text
            and not allow_unclosed
            and new_slice
            and "<" not in new_slice
        ):
            return {"content": new_slice}

        content_parts: list[str] = []
        new_calls: list[dict[str, Any]] = []

        if self._in_tool_kind is not None:
            window = self._tool_tail + new_slice
            self._tool_tail = ""
        else:
            if new_slice:
                self._pending_text += new_slice
            window = ""

        while True:
            if self._in_tool_kind is not None:
                close_pos = -1
                close_len = 0
                if "<|" in window:
                    for marker, mlen in _CLOSE_MARKERS:
                        mp = window.find(marker)
                        if mp >= 0 and (close_pos < 0 or mp < close_pos):
                            close_pos = mp
                            close_len = mlen

                if close_pos >= 0:
                    if close_pos > 0:
                        self._tool_chunks.append(window[:close_pos])
                    if self._in_tool_kind == "json":
                        payload = (
                            self._tool_chunks[0]
                            if len(self._tool_chunks) == 1
                            else "".join(self._tool_chunks)
                        )
                        self._emit_tool_payload(payload, new_calls)
                    self._in_tool_kind = None
                    self._tool_chunks = []
                    self._tool_tail = ""
                    rem = window[close_pos + close_len :]
                    window = ""
                    if rem:
                        self._pending_text += rem
                    if not self._pending_text and not allow_unclosed:
                        break
                    continue

                if allow_unclosed:
                    if window:
                        self._tool_chunks.append(window)
                    if self._in_tool_kind == "json":
                        payload = (
                            self._tool_chunks[0]
                            if len(self._tool_chunks) == 1
                            else "".join(self._tool_chunks)
                        )
                        self._emit_tool_payload(payload, new_calls)
                    self._in_tool_kind = None
                    self._tool_chunks = []
                    self._tool_tail = ""
                    window = ""
                    break

                keep = _MAX_CLOSE_LEN - 1
                if len(window) > keep:
                    self._tool_chunks.append(window[:-keep])
                    self._tool_tail = window[-keep:]
                else:
                    self._tool_tail = window
                break

            buf = self._pending_text
            if not buf:
                break

            j_pos = buf.find(_TAG_TOOL_JSON) if "<|" in buf else -1
            t_pos = buf.find(_TAG_TOOL_TEXT) if "<|" in buf else -1
            if j_pos < 0 and t_pos < 0:
                if allow_unclosed:
                    cleaned = _strip_structural_tags(buf)
                    if cleaned:
                        content_parts.append(cleaned)
                    self._pending_text = ""
                else:
                    hold = _partial_open_len(buf)
                    emit_end = len(buf) - hold
                    if emit_end > 0:
                        cleaned = _strip_structural_tags(buf[:emit_end])
                        if cleaned:
                            content_parts.append(cleaned)
                        self._pending_text = buf[emit_end:]
                break

            if j_pos >= 0 and (t_pos < 0 or j_pos < t_pos):
                tag_pos, tag_len, kind = j_pos, len(_TAG_TOOL_JSON), "json"
            else:
                tag_pos, tag_len, kind = t_pos, len(_TAG_TOOL_TEXT), "text"

            if tag_pos > 0:
                cleaned = _strip_structural_tags(buf[:tag_pos])
                if cleaned:
                    content_parts.append(cleaned)

            self._in_tool_kind = kind
            self._tool_chunks = []
            self._tool_tail = ""
            window = buf[tag_pos + tag_len :]
            self._pending_text = ""

        out: dict[str, Any] = {}
        joined = (
            content_parts[0]
            if len(content_parts) == 1
            else ("".join(content_parts) if content_parts else "")
        )
        if joined:
            out["content"] = joined
        if new_calls:
            out["tool_calls"] = new_calls
        return out or None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return self._scan_stream(current_text, allow_unclosed=False)

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        return self._scan_stream(current_text, allow_unclosed=True)
