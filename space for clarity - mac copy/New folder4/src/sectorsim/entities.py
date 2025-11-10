from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

from .utils import wrap_pi, clamp
from .control import TurnSessionConfig

@dataclass
class BoatParams:
    # Geometry (meters) used for rendering and collision radius.
    length: float = 6.0
    width:  float = 2.2

    # Speed limits and linear accel model (m/s and m/s^2).
    max_speed:  float = 18.0
    min_speed:  float = 0.0
    accel_rate: float = 1.6
    decel_rate: float = 1.2

class Boat:
    def __init__(self, boat_id: int, x: float, y: float, heading: float,
                 speed: float, kin: BoatParams, tcfg: TurnSessionConfig):
        self.id = boat_id
        self.x = float(x)
        self.y = float(y)
        self.h = float(heading)          # radians
        self.u = float(speed)            # m/s
        self.kin = kin
        self.tcfg = tcfg

        # Controls (signed)
        self.last_thr  = 0               # -1 decel, 0 coast, +1 accel
        self.last_helm = 0               # -1 starboard/right, 0 straight, +1 port/left

        # Turn-session state (latched chunk being executed over time)
        self.session_active = False
        self.session_target = 0.0
        self.session_dir    = 0          # -1 right, +1 left

    @staticmethod
    def decode_action(a: int) -> Tuple[int, int]:
        """
        Map action index to signed (helm, thr).
        steer ∈ {0: straight, 1: right, 2: left}
        throttle ∈ {0: coast, 1: accel, 2: decel}
        """
        steer    = a // 3
        throttle = a % 3
        helm = (-1 if steer == 1 else +1 if steer == 2 else 0)
        thr  = (+1 if throttle == 1 else -1 if throttle == 2 else 0)
        return helm, thr

    def _start_session(self, direction: int):
        """Latch a new chunk target ±turn_deg from current heading."""
        dpsi = math.radians(self.tcfg.turn_deg) * direction
        self.session_dir    = direction
        self.session_target = wrap_pi(self.h + dpsi)
        self.session_active = True

    def apply_action(self, a: int):
        """Decode and apply control intent for this step."""
        helm, thr = self.decode_action(a)
        self.last_helm = helm

        # Throttle passthrough (optionally hold while turning)
        if self.tcfg.passthrough_throttle:
            if not (self.tcfg.hold_throttle_while_turning and self.session_active):
                self.last_thr = thr

        # Steering: start/retarget a chunk (only if non-zero helm)
        if not self.session_active:
            if helm != 0:
                self._start_session(helm)
        else:
            if self.tcfg.allow_cancel and (helm != 0):
                self._start_session(helm)
            # else: ignore helm until current chunk finishes

    def integrate(self, dt: float):
        """
        Advance boat state by dt using:
        1) Turn session playback at constant rate with hysteresis + no-overshoot guard.
        2) Linear accel/decel for speed with clamp and no passive drag on coast.
        3) Straight-line kinematics in world frame.
        """
        # 1) Heading update (turn session)
        if self.session_active:
            rate = math.radians(self.tcfg.turn_rate_degps)
            err = wrap_pi(self.session_target - self.h)
            # hysteresis: early finish if within deadband and no overshoot
            if abs(err) <= math.radians(self.tcfg.hysteresis_deg):
                self.h = self.session_target
                self.session_active = False
                self.session_dir = 0
            else:
                step = math.copysign(rate * dt, err)
                # avoid overshoot in this step
                if abs(step) >= abs(err):
                    self.h = self.session_target
                    self.session_active = False
                    self.session_dir = 0
                else:
                    self.h = wrap_pi(self.h + step)

        # 2) Speed: linear accel/decel + clamp (no passive drag on coast)
        if self.last_thr > 0:
            self.u += self.kin.accel_rate * dt
        elif self.last_thr < 0:
            self.u -= self.kin.decel_rate * dt
        self.u = clamp(self.u, self.kin.min_speed, self.kin.max_speed)

        # 3) Position: simple kinematics
        self.x += math.cos(self.h) * self.u * dt
        self.y += math.sin(self.h) * self.u * dt