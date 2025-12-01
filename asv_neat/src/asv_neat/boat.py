"""Boat dynamics used by the crossing scenario environment."""
from __future__ import annotations

import math
from typing import Optional, Tuple

from .config import BoatParams, TurnSessionConfig
from .utils import clamp, wrap_pi


class Boat:
    """Vessel model with simple per-step helm inputs."""

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
        """Map an action index to discrete ``(steer, throttle)`` selections."""

        steer = action // 3  # 0: straight, 1: port, 2: starboard
        throttle = action % 3  # 0: hold, 1: accelerate, 2: decelerate
        return steer, throttle

    def apply_action(self, action: int) -> None:
        steer, throttle = self.decode_action(action)
        self.last_helm = steer
        self.last_thr = throttle

    def integrate(self, dt: float) -> None:
        if self.u > 0.0:
            turn_rate = 0.26  # rad/s
            direction = -1 if self.last_helm == 1 else 1 if self.last_helm == 2 else 0
            if direction:
                self.h = wrap_pi(self.h + direction * turn_rate * dt)

        if self.last_thr == 1:
            self.u += self.kin.accel_rate * dt
        elif self.last_thr == 2:
            self.u -= self.kin.decel_rate * dt
        self.u = clamp(self.u, self.kin.min_speed, self.kin.max_speed)

        self.x += math.cos(self.h) * self.u * dt
        self.y += math.sin(self.h) * self.u * dt
