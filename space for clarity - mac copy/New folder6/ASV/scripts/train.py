#!/usr/bin/env python3
"""Train a give-way controller for the crossing scenario with NEAT-Python."""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Optional

try:
    import neat
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("neat-python must be installed to run the training script.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asv import (  # noqa: E402
    BoatParams,
    CrossingScenarioEnv,
    EnvConfig,
    TurnSessionConfig,
)
from asv.neat_training import (  # noqa: E402
    FitnessParameters,
    build_scenarios,
    episode_cost,
    run_episode,
    train_population,
)
from asv.scenario import ScenarioRequest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "neat_crossing.cfg",
        help="Path to the NEAT configuration file.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations to run evolution for.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for Python's random module to make runs reproducible.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory in which to store NEAT checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Number of generations between checkpoint saves.",
    )
    parser.add_argument(
        "--save-winner",
        type=Path,
        default=None,
        help="File to pickle the winning genome to after training.",
    )

    # Scenario shaping -------------------------------------------------
    parser.add_argument(
        "--cross-distance",
        type=float,
        default=ScenarioRequest.crossing_distance,
        help="Longitudinal distance between the agent and the crossing point.",
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

    # Environment ------------------------------------------------------
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame rendering during winner evaluation.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=EnvConfig.dt,
        help="Simulation time step in seconds.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=EnvConfig.substeps,
        help="Integration substeps per frame.",
    )
    parser.add_argument(
        "--world-width",
        type=float,
        default=EnvConfig.world_w,
        help="Environment width in metres (used for coordinate transforms).",
    )
    parser.add_argument(
        "--world-height",
        type=float,
        default=EnvConfig.world_h,
        help="Environment height in metres.",
    )

    # Fitness shaping --------------------------------------------------
    parser.add_argument(
        "--max-steps",
        type=int,
        default=FitnessParameters.max_steps,
        help="Maximum number of simulation steps per episode.",
    )
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        default=FitnessParameters.goal_tolerance,
        help="Distance in metres considered a successful arrival at the goal.",
    )
    parser.add_argument(
        "--collision-distance",
        type=float,
        default=FitnessParameters.collision_distance,
        help="Minimum separation in metres before a collision is recorded.",
    )
    parser.add_argument(
        "--timeout-penalty",
        type=float,
        default=FitnessParameters.timeout_penalty,
        help="Cost added when the agent fails to reach the goal in time.",
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=FitnessParameters.collision_penalty,
        help="Cost added when a collision occurs.",
    )
    parser.add_argument(
        "--distance-penalty",
        type=float,
        default=FitnessParameters.distance_penalty,
        help="Multiplier for the remaining goal distance when time expires.",
    )
    parser.add_argument(
        "--separation-threshold",
        type=float,
        default=FitnessParameters.separation_threshold,
        help="Desired closest-approach distance to the stand-on vessel in metres.",
    )
    parser.add_argument(
        "--separation-penalty",
        type=float,
        default=FitnessParameters.separation_penalty,
        help="Penalty applied when the minimum separation falls below the threshold.",
    )
    parser.add_argument(
        "--distance-normaliser",
        type=float,
        default=FitnessParameters.distance_normaliser,
        help="Scale factor used for normalising distances in the cost function.",
    )

    return parser


def build_fitness_parameters(args: argparse.Namespace) -> FitnessParameters:
    return FitnessParameters(
        max_steps=args.max_steps,
        goal_tolerance=args.goal_tolerance,
        collision_distance=args.collision_distance,
        timeout_penalty=args.timeout_penalty,
        collision_penalty=args.collision_penalty,
        distance_penalty=args.distance_penalty,
        separation_threshold=args.separation_threshold,
        separation_penalty=args.separation_penalty,
        distance_normaliser=args.distance_normaliser,
    )


def build_env(
    args: argparse.Namespace, boat_params: BoatParams, *, render: bool = False
) -> CrossingScenarioEnv:
    cfg = EnvConfig(
        world_w=args.world_width,
        world_h=args.world_height,
        dt=args.dt,
        substeps=args.substeps,
        render=render,
        pixels_per_meter=EnvConfig.pixels_per_meter,
        show_grid=False,
        show_trails=False,
        show_hud=False,
    )
    return CrossingScenarioEnv(cfg=cfg, kin=boat_params, tcfg=TurnSessionConfig())


def summarise_winner(
    result,
    env: CrossingScenarioEnv,
    scenarios,
    params: FitnessParameters,
    feature_scale: float,
    boat_params: BoatParams,
    *,
    render: bool = False,
) -> None:
    """Print a small summary of the winning genome's behaviour."""

    if render:
        env.enable_render()

    network = neat.nn.FeedForwardNetwork.create(result.winner, result.config)
    total_cost = 0.0
    print("\nWinner evaluation summary:")
    for idx, scenario in enumerate(scenarios, start=1):
        metrics = run_episode(
            env,
            scenario,
            network,
            boat_params,
            params,
            feature_scale,
            render=render,
        )
        cost = episode_cost(metrics, params)
        total_cost += cost
        status = "reached goal" if metrics.reached_goal else ("collision" if metrics.collided else "timeout")
        print(
            f"  Scenario {idx}: steps={metrics.steps:4d} status={status:9s} "
            f"min_sep={metrics.min_separation:6.2f}m cost={cost:8.2f}"
        )
    print(f"Average cost: {total_cost / len(scenarios):.2f}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    fitness_params = build_fitness_parameters(args)
    scenario_request = ScenarioRequest(
        crossing_distance=args.cross_distance,
        agent_speed=args.agent_speed,
        stand_on_speed=args.stand_on_speed,
    )

    scenarios = build_scenarios(scenario_request)
    feature_scale = max(1.0, args.distance_normaliser)

    boat_params = BoatParams()
    env = build_env(args, boat_params, render=False)

    try:
        result = train_population(
            config_path=args.config,
            env=env,
            scenarios=scenarios,
            kin=boat_params,
            params=fitness_params,
            feature_scale=feature_scale,
            generations=args.generations,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
        )

        summarise_winner(
            result,
            env,
            scenarios,
            fitness_params,
            feature_scale,
            boat_params,
            render=args.render,
        )

        if args.save_winner is not None:
            with args.save_winner.open("wb") as fh:
                pickle.dump(result.winner, fh)
            print(f"Saved winning genome to {args.save_winner}")
    finally:
        env.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

