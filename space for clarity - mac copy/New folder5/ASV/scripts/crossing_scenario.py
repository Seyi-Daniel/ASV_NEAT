"""Deterministic crossing scenario generator with optional rendering."""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asv import BoatParams, CrossingScenarioEnv, EnvConfig, TurnSessionConfig

# Ordered bearings matching the sequence requested by the user: first 5°, then
# the midpoint, the maximum, followed by the quarter and three-quarter points.
STAND_ON_BEARINGS_DEG: Tuple[float, ...] = (
    5.0,
    (5.0 + 112.5) / 2.0,
    112.5,
    5.0 + 0.25 * (112.5 - 5.0),
    5.0 + 0.75 * (112.5 - 5.0),
)


@dataclass(frozen=True)
class VesselState:
    """Description of a vessel participating in the encounter."""

    name: str
    x: float
    y: float
    heading_deg: float
    speed: float
    goal: Optional[Tuple[float, float]] = None

    def bearing_to(self, other: "VesselState") -> float:
        """Return clockwise (starboard) relative bearing to ``other`` in degrees."""

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
        """Create a human-readable summary of the encounter."""

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

    # Give-way vessel approaches from the west toward the crossing point.
    agent = VesselState(
        name="give_way",
        x=-approach,
        y=0.0,
        heading_deg=0.0,
        speed=request.agent_speed,
        goal=(crossing_point[0] + approach, crossing_point[1]),
    )

    # Place the stand-on vessel by rotating around the give-way bow to achieve
    # the requested starboard bearing. Its heading is set to drive through the
    # crossing point, and the goal lies further along that course for future
    # controllers (e.g. NEAT) to utilise.
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


def iter_scenarios(
    angles: Iterable[float], request: ScenarioRequest
) -> Iterable[CrossingScenario]:
    """Yield scenarios for each provided bearing value."""

    for ang in angles:
        yield compute_crossing_geometry(ang, request)


def build_env(args: argparse.Namespace) -> CrossingScenarioEnv:
    cfg = EnvConfig(
        world_w=args.world_width,
        world_h=args.world_height,
        dt=args.dt,
        substeps=args.substeps,
        render=args.render,
        pixels_per_meter=args.pixels_per_meter,
        show_grid=not args.hide_grid,
        show_trails=not args.hide_trails,
        show_hud=not args.hide_hud,
    )
    return CrossingScenarioEnv(cfg=cfg, kin=BoatParams(), tcfg=TurnSessionConfig())


def scenario_states_for_env(env: CrossingScenarioEnv, scenario: CrossingScenario) -> Tuple[list, dict]:
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

    states = [convert(scenario.agent), convert(scenario.stand_on)]
    meta = {
        "bearing": scenario.requested_bearing,
        "cross_x": cross_x,
        "cross_y": cross_y,
    }
    return states, meta


class RandomGiveWayPolicy:
    """Placeholder controller emitting random actions for the give-way boat."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def choose_action(self, inputs: Tuple[float, float, float, float, float, float]) -> int:
        """Select one of the nine discrete helm/throttle combinations."""

        return self._rng.randrange(9)


def agent_inputs_from_state(state: dict) -> Tuple[float, float, float, float, float, float]:
    """Build the six-element observation vector expected by future NEAT logic."""

    goal_x = float(state.get("goal_x", 0.0))
    goal_y = float(state.get("goal_y", 0.0))
    return (
        float(state["x"]),
        float(state["y"]),
        float(state["speed"]),
        math.degrees(float(state["heading"])),
        goal_x,
        goal_y,
    )


def run_render_loop(
    env: CrossingScenarioEnv,
    scenario: CrossingScenario,
    duration: float,
    seed: Optional[int] = None,
) -> None:
    states, meta = scenario_states_for_env(env, scenario)
    env.reset_from_states(states, meta=meta)
    env.enable_render()
    controller = RandomGiveWayPolicy(seed=seed)

    steps = max(1, int(round(duration / env.cfg.dt)))
    env.render()
    for _ in range(steps):
        snapshot = env.snapshot()
        agent_state = snapshot[0] if snapshot else {}
        inputs = agent_inputs_from_state(agent_state) if agent_state else (0,) * 6
        action = controller.choose_action(inputs)
        env.step([action, None])
        env.render()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-distance",
        type=float,
        default=ScenarioRequest.crossing_distance,
        help="Longitudinal distance between the agent and the crossing point in metres.",
    )
    parser.add_argument(
        "--agent-speed",
        type=float,
        default=ScenarioRequest.agent_speed,
        help="Initial speed for the give-way vessel in m/s.",
    )
    parser.add_argument(
        "--stand-on-speed",
        type=float,
        default=ScenarioRequest.stand_on_speed,
        help="Initial speed for the stand-on vessel in m/s.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame rendering of each deterministic scenario.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=35.0,
        help="Simulation time per scenario when rendering (seconds).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=EnvConfig.dt,
        help="Simulation step in seconds.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=EnvConfig.substeps,
        help="Number of integration substeps per frame.",
    )
    parser.add_argument(
        "--world-width",
        type=float,
        default=EnvConfig.world_w,
        help="World width in metres for the renderer.",
    )
    parser.add_argument(
        "--world-height",
        type=float,
        default=EnvConfig.world_h,
        help="World height in metres for the renderer.",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=EnvConfig.pixels_per_meter,
        help="Display scaling factor for rendering.",
    )
    parser.add_argument(
        "--hide-grid",
        action="store_true",
        help="Disable the background grid overlay.",
    )
    parser.add_argument(
        "--hide-trails",
        action="store_true",
        help="Disable vessel motion trails.",
    )
    parser.add_argument(
        "--hide-hud",
        action="store_true",
        help="Disable the on-screen HUD panel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the give-way controller (enables reproducibility).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = ScenarioRequest(
        crossing_distance=args.cross_distance,
        agent_speed=args.agent_speed,
        stand_on_speed=args.stand_on_speed,
    )

    scenarios = list(iter_scenarios(STAND_ON_BEARINGS_DEG, request))

    print("Deterministic crossing scenarios (give-way vs stand-on)")
    print("=" * 56)
    for idx, scenario in enumerate(scenarios, start=1):
        print(f"\nScenario {idx}:")
        print(scenario.describe())

    env = build_env(args)

    if args.render and env._screen is None:
        # Rendering was requested but pygame is unavailable or initialisation failed.
        raise RuntimeError("pygame could not be initialised; rendering is unavailable.")

    try:
        if args.render:
            print("\nRendering scenarios — close the window or press ESC to exit early.")
            for idx, scenario in enumerate(scenarios, start=1):
                print(
                    f"  • Scenario {idx}: bearing {scenario.requested_bearing:6.2f}°"
                )
                seed_value = None if args.seed is None else args.seed + idx - 1
                run_render_loop(env, scenario, args.duration, seed=seed_value)
        else:
            print("\nRendering disabled; use --render to open the pygame visualisation.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
