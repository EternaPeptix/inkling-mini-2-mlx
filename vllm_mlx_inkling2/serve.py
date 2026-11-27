"""Launch vllm-mlx with Inkling 2 parsers, cache patches, and loopling-opt kernels."""

from __future__ import annotations

import mlx_lm  # noqa: F401  # must precede mlx.core

import vllm_mlx_inkling2.reasoning_parser  # noqa: F401
import vllm_mlx_inkling2.tool_parser  # noqa: F401
from vllm_mlx_inkling2.patches import install
from vllm_mlx_inkling2.stream_opt import install as install_stream_opt
from vllm_mlx.cli import main


def _install_all() -> None:
    install()
    install_stream_opt()
    try:
        from vllm_mlx_inkling2.sched_opt import install as install_sched_opt
    except ImportError:
        return
    install_sched_opt()


if __name__ == "__main__":
    _install_all()
    main()
