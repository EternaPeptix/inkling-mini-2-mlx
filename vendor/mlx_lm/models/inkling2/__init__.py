# Copyright © 2026 Apple Inc.

"""Inkling 2.

https://huggingface.co/thinkingmachines/Inkling-2-Small
"""

from mlx_lm.models.inkling2.config import ModelArgs
from mlx_lm.models.inkling2.model import Model

__all__ = [
    "Model",
    "ModelArgs",
]

import os as _os

if _os.environ.get("LOOPLING_BATCH_INVARIANT") == "1":  # lane D, default off
    from mlx_lm.models.inkling2 import batch_invariant as _bi

    _bi.install()
