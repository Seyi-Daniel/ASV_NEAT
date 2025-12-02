"""Lightweight accelerator that opportunistically offloads math to the GPU.

The simulator and training loops are primarily scalar math. To keep the codebase
portable while still benefiting from GPU hardware when available, we centralise
the numeric backend here. The module prefers ``cupy`` (CUDA-enabled) and falls
back to ``numpy`` on CPU without changing any call sites. All helpers return
plain Python floats so existing consumers remain unchanged.
"""
from __future__ import annotations

from typing import Any

BACKEND_NAME = "cpu"

try:  # pragma: no cover - optional dependency
    import cupy as cp

    xp = cp  # type: ignore[assignment]
    BACKEND_NAME = "gpu (cupy)"
except Exception:  # pragma: no cover - optional dependency
    import numpy as np

    xp = np  # type: ignore[assignment]
    BACKEND_NAME = "cpu (numpy)"


def to_scalar(value: Any) -> float:
    """Convert ``value`` from the chosen backend to a plain ``float``.

    NumPy/CuPy return 0-D arrays for scalar operations; ``item()`` handles both.
    When the backend already yields a builtin ``float`` this is a no-op.
    """

    try:
        return float(value.item())  # type: ignore[call-arg]
    except Exception:
        return float(value)

