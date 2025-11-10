"""Configuration dataclasses for the crossing scenario environment."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoatParams:
    """Geometry and kinematic limits for a vessel."""

    length: float = 6.0
    width: float = 2.2

    max_speed: float = 18.0
    min_speed: float = 0.0
    accel_rate: float = 1.6
    decel_rate: float = 1.2


@dataclass
class TurnSessionConfig:
    """Chunked turning behaviour parameters."""

    turn_deg: float = 15.0
    turn_rate_degps: float = 45.0
    allow_cancel: bool = False
    hysteresis_deg: float = 1.5

    passthrough_throttle: bool = True
    hold_throttle_while_turning: bool = False


@dataclass
class EnvConfig:
    """Rendering and integration settings for the simplified environment."""

    world_w: float = 520.0
    world_h: float = 520.0
    dt: float = 0.05
    substeps: int = 1

    render: bool = False
    pixels_per_meter: float = 1.5
    show_grid: bool = True
    show_trails: bool = True
    show_hud: bool = True
