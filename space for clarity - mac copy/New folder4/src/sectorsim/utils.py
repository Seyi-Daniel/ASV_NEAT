from __future__ import annotations
import math
from typing import Tuple

def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def angle_deg(a: float) -> float:
    return a * 180.0 / math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x

def tcpa_dcpa(ax, ay, ah, aspd, bx, by, bh, bspd) -> Tuple[float, float]:
    """
    Constant-velocity CPA metrics:
    - tcpa (s): time to closest approach (>=0 if future).
    - dcpa (m): distance at closest approach.
    """
    rvx = math.cos(bh) * bspd - math.cos(ah) * aspd
    rvy = math.sin(bh) * bspd - math.sin(ah) * aspd
    rx, ry = (bx - ax), (by - ay)
    rv2 = rvx*rvx + rvy*rvy
    if rv2 < 1e-9:
        return 0.0, math.hypot(rx, ry)
    tcpa = - (rx*rvx + ry*rvy) / rv2
    if tcpa < 0.0:
        tcpa = 0.0
    cx = rx + rvx * tcpa
    cy = ry + rvy * tcpa
    return tcpa, math.hypot(cx, cy)