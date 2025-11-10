from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TurnSessionConfig:
    # Chunked turn control:
    turn_deg: float = 15.0               # chunk size in degrees per command
    turn_rate_degps: float = 45.0        # constant rate (deg/s) to play out each chunk
    allow_cancel: bool = False           # if True: new non-zero helm retargets mid-chunk
    hysteresis_deg: float = 1.5          # small deadband to finish early (0..~2° is typical)

    # Throttle behavior:
    passthrough_throttle: bool = True     # throttle accepted every step, independent of turn session
    hold_throttle_while_turning: bool = False  # if True: freeze last_thr while a session is active