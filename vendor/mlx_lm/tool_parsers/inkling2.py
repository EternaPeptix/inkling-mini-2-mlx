# Copyright © 2026 Apple Inc.

"""Tool call parser for Inkling 2 (`<|content_invoke_tool_json|>`)."""

import json
from typing import Any, Optional

tool_call_start = "<|content_invoke_tool_json|>"
tool_call_end = "<|end_message|>"


def _normalize_call(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Tool call payload must be a JSON object.")
    name = item.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Tool call object missing non-empty 'name'.")
    if "args" in item:
        arguments = item["args"]
    elif "arguments" in item:
        arguments = item["arguments"]
    else:
        arguments = {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    elif arguments is None:
        arguments = {}
    out: dict[str, Any] = {"name": name, "arguments": arguments}
    if item.get("id") is not None:
        out["id"] = str(item["id"])
    return out


def parse_tool_call(
    model_output: str, tools: Optional[Any] = None
) -> dict[str, Any] | list[dict[str, Any]]:
    raw = model_output.strip()
    if not raw:
        raise ValueError("Empty tool call payload.")
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("Empty tool call list.")
        calls = [_normalize_call(item) for item in parsed]
        return calls[0] if len(calls) == 1 else calls
    return _normalize_call(parsed)
