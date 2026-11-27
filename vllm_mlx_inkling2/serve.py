"""Launch vllm-mlx with Inkling 2 parsers, cache patches, and loopling-opt kernels."""

from __future__ import annotations

import mlx_lm  # noqa: F401  # must precede mlx.core

import vllm_mlx_inkling2.reasoning_parser  # noqa: F401
import vllm_mlx_inkling2.tool_parser  # noqa: F401
from vllm_mlx_inkling2.patches import install
from vllm_mlx.cli import main

if __name__ == "__main__":
    install()
    main()
