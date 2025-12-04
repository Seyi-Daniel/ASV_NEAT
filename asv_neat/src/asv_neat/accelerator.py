"""Lightweight accelerator that opportunistically offloads math to the GPU.

The simulator and training loops are primarily scalar math. To keep the codebase
portable while still benefiting from GPU hardware when available, we centralise
the numeric backend here. The module prefers ``cupy`` (CUDA-enabled) and falls
back to ``numpy`` on CPU without changing any call sites. All helpers return
plain Python floats so existing consumers remain unchanged.
"""
from __future__ import annotations

from typing import Any
import warnings

BACKEND_NAME = "cpu"


def _select_backend() -> tuple[str, Any]:
    """Prefer CuPy when usable, otherwise fall back to NumPy.

    Some environments have ``cupy`` installed without the required CUDA
    runtime/driver DLLs (e.g., ``nvrtc64_120_0.dll``). In that case, attempting
    to compile a kernel will raise at runtime. We proactively test a simple
    elementwise operation to validate the installation; if it fails we emit a
    warning and revert to CPU safely.
    """

    try:  # pragma: no cover - optional dependency
        import cupy as cp

        try:  # sanity-check kernel compilation
            _ = cp.add(cp.ones(1), cp.ones(1))
            _.item()  # force computation
        except Exception as exc:  # pragma: no cover - optional dependency
            warnings.warn(
                f"CuPy is installed but unusable; falling back to NumPy ({exc})."
            )
            raise

        return "gpu (cupy)", cp
    except Exception:  # pragma: no cover - optional dependency
        import numpy as np

        return "cpu (numpy)", np


BACKEND_NAME, xp = _select_backend()


def to_scalar(value: Any) -> float:
    """Convert ``value`` from the chosen backend to a plain ``float``.

    NumPy/CuPy return 0-D arrays for scalar operations; ``item()`` handles both.
    When the backend already yields a builtin ``float`` this is a no-op.
    """

    try:
        return float(value.item())  # type: ignore[call-arg]
    except Exception:
        return float(value)

