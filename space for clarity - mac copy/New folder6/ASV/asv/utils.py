"""Utility helpers for the simplified crossing scenario package."""
from __future__ import annotations

import math


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the inclusive range ``[lo, hi]``."""

    return max(lo, min(hi, value))


def wrap_pi(angle: float) -> float:
    """Wrap an angle to the ``[-π, π]`` interval."""

    return math.atan2(math.sin(angle), math.cos(angle))


def angle_deg(angle: float) -> float:
    """Convert radians to a canonical degree representation in ``[0, 360)``."""

    return math.degrees(angle) % 360.0
