"""Boat dynamics used by the crossing scenario environment."""
from __future__ import annotations

import math
from typing import Optional, Tuple

from .config import BoatParams, TurnSessionConfig
from .utils import clamp, wrap_pi


class Boat:
    """Vessel model with simple chunked turn sessions."""

    def __init__(
        self,
        boat_id: int,
        x: float,
        y: float,
        heading: float,
        speed: float,
        kin: BoatParams,
        tcfg: TurnSessionConfig,
        goal: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.id = boat_id
        self.x = float(x)
        self.y = float(y)
        self.h = float(heading)
        self.u = float(speed)
        self.kin = kin
        self.tcfg = tcfg
        if goal is not None:
            gx, gy = goal
            self.goal_x = float(gx)
            self.goal_y = float(gy)
        else:
            self.goal_x = None
            self.goal_y = None

        self.last_thr = 0
        self.last_helm = 0

        self.session_active = False
        self.session_dir = 0
        self.session_target = 0.0

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Return the observable state for this boat."""

        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "heading": self.h,
            "speed": self.u,
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
        }

    @staticmethod
    def decode_action(action: int) -> Tuple[int, int]:
        """Map an action index to signed ``(helm, throttle)``."""

        steer = action // 3
        throttle = action % 3
        helm = -1 if steer == 1 else 1 if steer == 2 else 0
        thr = 1 if throttle == 1 else -1 if throttle == 2 else 0
        return helm, thr

    def _start_session(self, direction: int) -> None:
        delta = math.radians(self.tcfg.turn_deg) * direction
        self.session_dir = direction
        self.session_target = wrap_pi(self.h + delta)
        self.session_active = True

    def apply_action(self, action: int) -> None:
        helm, thr = self.decode_action(action)
        self.last_helm = helm

        if self.tcfg.passthrough_throttle and not (
            self.tcfg.hold_throttle_while_turning and self.session_active
        ):
            self.last_thr = thr

        if not self.session_active:
            if helm != 0:
                self._start_session(helm)
        else:
            if self.tcfg.allow_cancel and helm != 0:
                self._start_session(helm)

    def integrate(self, dt: float) -> None:
        if self.session_active and self.u > 0.0:
            rate = math.radians(self.tcfg.turn_rate_degps)
            err = wrap_pi(self.session_target - self.h)
            hysteresis = math.radians(self.tcfg.hysteresis_deg)

            if abs(err) <= hysteresis:
                self.h = self.session_target
                self.session_active = False
                self.session_dir = 0
            else:
                step = math.copysign(rate * dt, err)
                if abs(step) >= abs(err):
                    self.h = self.session_target
                    self.session_active = False
                    self.session_dir = 0
                else:
                    self.h = wrap_pi(self.h + step)

        if self.last_thr > 0:
            self.u += self.kin.accel_rate * dt
        elif self.last_thr < 0:
            self.u -= self.kin.decel_rate * dt
        self.u = clamp(self.u, self.kin.min_speed, self.kin.max_speed)

        self.x += math.cos(self.h) * self.u * dt
        self.y += math.sin(self.h) * self.u * dt
