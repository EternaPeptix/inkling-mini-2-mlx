# Copyright © 2026 Apple Inc.

"""Response format of Inkling 2 (`inkling2_loop`)."""

RESPONSE_FORMAT = {
    "initial_state": "header",
    "transitions": {
        "header": [
            ("<|content_thinking|>", "reasoning"),
            ("<|content_text|>", "normal"),
            ("<|content_invoke_tool_json|>", "tool"),
            ("<|content_invoke_tool_text|>", "tool"),
            ("<|content_model_end_sampling|>", "stop"),
        ],
        "reasoning": [
            ("<|end_message|>", "header"),
            ("<|content_model_end_sampling|>", "stop"),
        ],
        "normal": [
            ("<|end_message|>", "header"),
            ("<|content_model_end_sampling|>", "stop"),
        ],
        "tool": [
            ("<|content_model_end_sampling|>", "stop"),
        ],
    },
}
