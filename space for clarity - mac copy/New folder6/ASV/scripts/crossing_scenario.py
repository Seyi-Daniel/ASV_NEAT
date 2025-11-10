"""Deterministic crossing scenario generator with optional rendering."""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asv import BoatParams, CrossingScenarioEnv, EnvConfig, TurnSessionConfig
from asv.scenario import (
    STAND_ON_BEARINGS_DEG,
    CrossingScenario,
    ScenarioRequest,
    iter_scenarios,
    raw_agent_inputs,
    scenario_states_for_env,
)


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


class RandomGiveWayPolicy:
    """Placeholder controller emitting random actions for the give-way boat."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def choose_action(self, inputs: Tuple[float, float, float, float, float, float]) -> int:
        """Select one of the nine discrete helm/throttle combinations."""

        return self._rng.randrange(9)


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
        inputs = raw_agent_inputs(agent_state) if agent_state else (0.0,) * 6
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
