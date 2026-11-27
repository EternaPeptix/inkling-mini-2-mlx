"""Inkling 2 adapter package for vllm-mlx backed by loopling-opt kernels."""

# Must import vendored mlx_lm before mlx.core so MLX_MAX_OPS_PER_BUFFER=32 applies.
import mlx_lm  # noqa: F401
