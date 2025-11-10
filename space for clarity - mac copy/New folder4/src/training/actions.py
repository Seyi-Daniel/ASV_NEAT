from __future__ import annotations

def decode_action_idx(a: int) -> tuple[int, int]:
    """
    Returns (steer, throttle) with:
      steer ∈ {0: straight, 1: right, 2: left}
      throttle ∈ {0: coast,   1: accel, 2: decel}
    Only used for logging readability.
    """
    return a // 3, a % 3