"""Geometry helpers for the deterministic crossing scenarios."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence, Tuple


STAND_ON_BEARINGS_DEG: Tuple[float, ...] = (
    5.0,
    (5.0 + 112.5) / 2.0,
    112.5,
    5.0 + 0.25 * (112.5 - 5.0),
    5.0 + 0.75 * (112.5 - 5.0),
)


@dataclass(frozen=True)
class VesselState:
    """Description of an individual vessel participating in the encounter."""

    name: str
    x: float
    y: float
    heading_deg: float
    speed: float
    goal: Optional[Tuple[float, float]] = None

    def bearing_to(self, other: "VesselState") -> float:
        """Clockwise (starboard) relative bearing to ``other`` in degrees."""

        dx = other.x - self.x
        dy = other.y - self.y
        ch = math.cos(math.radians(self.heading_deg))
        sh = math.sin(math.radians(self.heading_deg))
        x_rel = ch * dx + sh * dy
        y_rel = -sh * dx + ch * dy
        rel_port = math.degrees(math.atan2(y_rel, x_rel))
        rel_port = (rel_port + 360.0) % 360.0
        return (360.0 - rel_port) % 360.0


@dataclass(frozen=True)
class CrossingScenario:
    """Container for the initial geometry of a crossing encounter."""

    agent: VesselState
    stand_on: VesselState
    crossing_point: Tuple[float, float]
    requested_bearing: float

    def describe(self) -> str:
        bearing = self.agent.bearing_to(self.stand_on)
        return (
            f"Stand-on bearing (requested) : {self.requested_bearing:6.2f}°\n"
            f"Stand-on bearing (realised)  : {bearing:6.2f}°\n"
            f"Agent position               : ({self.agent.x:7.2f}, {self.agent.y:7.2f}) m\n"
            f"Stand-on position            : ({self.stand_on.x:7.2f}, {self.stand_on.y:7.2f}) m\n"
            f"Agent heading                : {self.agent.heading_deg:6.2f}°\n"
            f"Stand-on heading             : {self.stand_on.heading_deg:6.2f}°\n"
            f"Agent speed                  : {self.agent.speed:6.2f} m/s\n"
            f"Stand-on speed               : {self.stand_on.speed:6.2f} m/s"
        )


@dataclass(frozen=True)
class ScenarioRequest:
    """User-controllable parameters for the crossing scenario."""

    crossing_distance: float = 220.0
    agent_speed: float = 7.0
    stand_on_speed: float = 7.0


def compute_crossing_geometry(angle_deg: float, request: ScenarioRequest) -> CrossingScenario:
    """Create the crossing encounter for a single bearing value."""

    crossing_point = (0.0, 0.0)
    approach = request.crossing_distance

    agent = VesselState(
        name="give_way",
        x=-approach,
        y=0.0,
        heading_deg=0.0,
        speed=request.agent_speed,
        goal=(crossing_point[0] + approach, crossing_point[1]),
    )

    port_angle_rad = math.radians(360.0 - angle_deg)
    stand_x = agent.x + approach * math.cos(port_angle_rad)
    stand_y = agent.y + approach * math.sin(port_angle_rad)
    heading_rad = math.atan2(crossing_point[1] - stand_y, crossing_point[0] - stand_x)
    heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0
    goal_x = crossing_point[0] + approach * math.cos(heading_rad)
    goal_y = crossing_point[1] + approach * math.sin(heading_rad)

    stand_on = VesselState(
        name="stand_on",
        x=stand_x,
        y=stand_y,
        heading_deg=heading_deg,
        speed=request.stand_on_speed,
        goal=(goal_x, goal_y),
    )

    return CrossingScenario(
        agent=agent,
        stand_on=stand_on,
        crossing_point=crossing_point,
        requested_bearing=angle_deg,
    )


def iter_scenarios(angles: Iterable[float], request: ScenarioRequest) -> Iterator[CrossingScenario]:
    """Yield scenarios for each provided bearing value."""

    for ang in angles:
        yield compute_crossing_geometry(ang, request)


def scenario_states_for_env(
    env, scenario: CrossingScenario
) -> Tuple[Sequence[dict], dict]:
    """Convert the dataclass description into environment-specific state dictionaries."""

    cx = env.world_w / 2.0
    cy = env.world_h / 2.0
    cross_x = cx + scenario.crossing_point[0]
    cross_y = cy + scenario.crossing_point[1]

    def convert(vessel: VesselState) -> dict:
        data = {
            "x": cross_x + vessel.x,
            "y": cross_y + vessel.y,
            "heading": math.radians(vessel.heading_deg),
            "speed": vessel.speed,
        }
        if vessel.goal is not None:
            gx, gy = vessel.goal
            data["goal_x"] = cross_x + gx
            data["goal_y"] = cross_y + gy
        return data

    states: Sequence[dict] = [convert(scenario.agent), convert(scenario.stand_on)]
    meta = {
        "bearing": scenario.requested_bearing,
        "cross_x": cross_x,
        "cross_y": cross_y,
    }
    return states, meta


def raw_agent_inputs(state: dict) -> Tuple[float, float, float, float, float, float]:
    """Return the six raw values expected by the NEAT controller interface."""

    goal_x = float(state.get("goal_x", 0.0))
    goal_y = float(state.get("goal_y", 0.0))
    return (
        float(state["x"]),
        float(state["y"]),
        float(state.get("speed", 0.0)),
        math.degrees(float(state["heading"])),
        goal_x,
        goal_y,
    )


__all__ = [
    "STAND_ON_BEARINGS_DEG",
    "VesselState",
    "CrossingScenario",
    "ScenarioRequest",
    "compute_crossing_geometry",
    "iter_scenarios",
    "scenario_states_for_env",
    "raw_agent_inputs",
]

