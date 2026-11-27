"""Reasoning parser for Inkling 2 (`inkling2_loop`) structural token format."""

from __future__ import annotations

from typing import Any

from vllm_mlx.reasoning import register_parser
from vllm_mlx.reasoning.base import DeltaMessage, ReasoningParser

_HEADER = "header"
_REASONING = "reasoning"
_NORMAL = "normal"
_TOOL = "tool"

_TAG_MODEL = "<|message_model|>"
_TAG_THINK = "<|content_thinking|>"
_TAG_TEXT = "<|content_text|>"
_TAG_TOOL_JSON = "<|content_invoke_tool_json|>"
_TAG_TOOL_TEXT = "<|content_invoke_tool_text|>"
_TAG_END = "<|end_message|>"
_TAG_EOS = "<|content_model_end_sampling|>"

_ALL_MARKERS = (
    _TAG_MODEL,
    _TAG_THINK,
    _TAG_TEXT,
    _TAG_TOOL_JSON,
    _TAG_TOOL_TEXT,
    _TAG_END,
    _TAG_EOS,
)
_MAX_MARKER_LEN = max(len(m) for m in _ALL_MARKERS)


def _partial_suffix_len(text: str, markers: tuple[str, ...] = _ALL_MARKERS) -> int:
    """Return length of longest trailing substring that prefixes any structural marker."""
    idx = text.rfind("<|")
    if idx < 0:
        return 0
    tail = text[idx:]
    if len(tail) >= _MAX_MARKER_LEN:
        return 0
    if any(m.startswith(tail) and m != tail for m in markers):
        return len(tail)
    return 0


class _Inkling2StreamState:
    """Incremental state machine matching vendor/mlx_lm/chat_templates/inkling2.py."""

    def __init__(self, implicit_mode: bool = False) -> None:
        self.state: str = _REASONING if implicit_mode else _HEADER
        self.buf: str = ""
        self.saw_marker: bool = implicit_mode

    def _find_next_marker(self) -> tuple[int, str | None]:
        best_idx = -1
        best_tag: str | None = None
        for tag in _ALL_MARKERS:
            pos = self.buf.find(tag)
            if pos >= 0 and (best_idx < 0 or pos < best_idx):
                best_idx = pos
                best_tag = tag
        return best_idx, best_tag

    def feed(self, delta_text: str) -> tuple[str, str]:
        """Consume `delta_text` and return `(reasoning_delta, content_delta)`."""
        self.buf += delta_text
        r_parts: list[str] = []
        c_parts: list[str] = []

        while self.buf:
            idx, tag = self._find_next_marker()
            if tag is None:
                if self.state == _HEADER:
                    # Wait in header while recipient/prefix text accumulates before `<|content_*|>`.
                    if len(self.buf) > 512 and "<|" not in self.buf and not self.saw_marker:
                        c_parts.append(self.buf)
                        self.buf = ""
                    break
                hold = _partial_suffix_len(self.buf)
                emit_len = len(self.buf) - hold
                if emit_len > 0:
                    chunk = self.buf[:emit_len]
                    self.buf = self.buf[emit_len:]
                    if self.state == _REASONING:
                        r_parts.append(chunk)
                    elif self.state in (_NORMAL, _TOOL):
                        c_parts.append(chunk)
                break

            self.saw_marker = True
            before = self.buf[:idx]
            self.buf = self.buf[idx + len(tag) :]

            if before:
                if self.state == _REASONING:
                    r_parts.append(before)
                elif self.state in (_NORMAL, _TOOL):
                    c_parts.append(before)
                # In _HEADER, text before `<|content_*|>` is structural recipient metadata; discard.

            if tag == _TAG_THINK:
                self.state = _REASONING
            elif tag == _TAG_TEXT:
                self.state = _NORMAL
            elif tag in (_TAG_TOOL_JSON, _TAG_TOOL_TEXT):
                self.state = _TOOL
                c_parts.append(tag)
            elif tag == _TAG_END:
                if self.state == _TOOL:
                    c_parts.append(tag)
                self.state = _HEADER
            elif tag in (_TAG_MODEL, _TAG_EOS):
                self.state = _HEADER

        return "".join(r_parts), "".join(c_parts)

    def finalize(self) -> tuple[str, str]:
        """Flush any buffered text on end-of-stream without leaking structural tags."""
        rem = self.buf
        self.buf = ""
        if not rem:
            return "", ""
        hold = _partial_suffix_len(rem)
        if hold:
            rem = rem[:-hold]
        if not rem:
            return "", ""
        if self.state == _REASONING:
            return rem, ""
        if self.state in (_NORMAL, _TOOL):
            return "", rem
        if self.state == _HEADER and not self.saw_marker:
            return "", rem
        return "", ""


class Inkling2ReasoningParser(ReasoningParser):
    """Extract Inkling 2 `<|content_thinking|>` vs `<|content_text|>` / tool blocks."""

    start_token = _TAG_THINK

    def __init__(self, tokenizer: Any | None = None) -> None:
        super().__init__(tokenizer)
        self._stream = _Inkling2StreamState(implicit_mode=False)

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        st = _Inkling2StreamState(implicit_mode=False)
        r0, c0 = st.feed(model_output)
        r1, c1 = st.finalize()
        reasoning = r0 + r1
        content = c0 + c1
        if reasoning and not content:
            return reasoning, ""
        return (reasoning or None), (content or None)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        if not delta_text:
            return None
        r, c = self._stream.feed(delta_text)
        if not r and not c:
            return None
        return DeltaMessage(reasoning=r or None, content=c or None)

    def reset_state(self, implicit_mode: bool = False) -> None:
        self._stream = _Inkling2StreamState(implicit_mode=implicit_mode)

    def finalize_stream(self) -> DeltaMessage | None:
        r, c = self._stream.finalize()
        if not r and not c:
            return None
        return DeltaMessage(reasoning=r or None, content=c or None)


register_parser("inkling2", Inkling2ReasoningParser)
