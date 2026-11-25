# Copyright © 2023 Apple Inc.

import os
import sys

from ._version import __version__

# Loopling decode is bound by dependent-kernel latency: fewer, larger Metal command buffers are faster and change
# no bits (same kernels, same order; docs/LANE_E.md).  MLX reads these once, when mlx.core initializes the device,
# so they only apply if mlx.core was not imported before mlx_lm.  LOOPLING_CMDBUF=0 disables; explicit env wins.
if os.environ.get("LOOPLING_CMDBUF", "1") not in ("0", ""):
    if "mlx.core" in sys.modules:
        print("[mlx_lm] note: mlx.core imported before mlx_lm; LOOPLING_CMDBUF buffer sizes not applied",
              file=sys.stderr)
    else:
        os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "64")
        os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from .convert import convert
from .generate import batch_generate, generate, stream_generate
from .utils import load

__all__ = [
    "__version__",
    "convert",
    "batch_generate",
    "generate",
    "stream_generate",
    "load",
]
