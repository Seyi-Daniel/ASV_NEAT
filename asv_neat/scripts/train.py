#!/usr/bin/env python3
"""Train the give-way vessel controller for the COLREGs crossing scenario."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable, Optional

try:  # pragma: no cover - optional dependency
    import neat
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("neat-python must be installed to run the training script.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asv_neat import (  # noqa: E402
    BoatParams,
    CrossingScenario,
    CrossingScenarioEnv,
    EnvConfig,
    HyperParameters,
    ScenarioRequest,
    TurnSessionConfig,
    apply_cli_overrides,
    build_scenarios,
    episode_cost,
    simulate_episode,
    train_population,
)


def build_parser(hparams: HyperParameters) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "neat_crossing.cfg",
        help="Path to the neat-python configuration file.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of evolutionary generations to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory in which neat-python checkpoints should be stored.",
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
        help="Optional path for pickling the winning genome after training.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame visualisation while summarising the winning genome.",
    )
    parser.add_argument(
        "--list-hyperparameters",
        action="store_true",
        help="List available hyperparameters and exit without training.",
    )
    parser.add_argument(
        "--hp",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a hyperparameter (repeatable). See --list-hyperparameters for names.",
    )
    return parser


def print_hyperparameters(hparams: HyperParameters) -> None:
    print("Available hyperparameters (NAME = default | description):")
    for name, value, help_text in hparams.iter_documentation():
        description = help_text or ""
        print(f"  {name} = {value!r}\n      {description}")


def build_boat_params(hparams: HyperParameters) -> BoatParams:
    return BoatParams(
        length=hparams.boat_length,
        width=hparams.boat_width,
        max_speed=hparams.boat_max_speed,
        min_speed=hparams.boat_min_speed,
        accel_rate=hparams.boat_accel_rate,
        decel_rate=hparams.boat_decel_rate,
    )


def build_turn_config(hparams: HyperParameters) -> TurnSessionConfig:
    return TurnSessionConfig(
        turn_deg=hparams.turn_chunk_deg,
        turn_rate_degps=hparams.turn_rate_degps,
        hysteresis_deg=hparams.turn_hysteresis_deg,
    )


def build_env_config(hparams: HyperParameters, *, render: bool) -> EnvConfig:
    return EnvConfig(
        world_w=hparams.env_world_w,
        world_h=hparams.env_world_h,
        dt=hparams.env_dt,
        substeps=hparams.env_substeps,
        render=render,
        pixels_per_meter=hparams.env_pixels_per_meter,
        show_grid=False,
        show_trails=False,
        show_hud=False,
    )


def build_scenario_request(hparams: HyperParameters) -> ScenarioRequest:
    return ScenarioRequest(
        crossing_distance=hparams.scenario_crossing_distance,
        agent_speed=hparams.scenario_agent_speed,
        stand_on_speed=hparams.scenario_stand_on_speed,
        goal_extension=hparams.scenario_goal_extension,
    )


def summarise_winner(
    result,
    scenarios: Iterable[CrossingScenario],
    hparams: HyperParameters,
    boat_params: BoatParams,
    turn_cfg: TurnSessionConfig,
    env_cfg: EnvConfig,
    *,
    render: bool = False,
) -> None:
    scenario_list = list(scenarios)
    env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params, tcfg=turn_cfg)
    try:
        if render:
            env.enable_render()
        network = neat.nn.FeedForwardNetwork.create(result.winner, result.config)
        total_cost = 0.0
        print("\nWinner evaluation summary:")
        for idx, scenario in enumerate(scenario_list, start=1):
            metrics = simulate_episode(env, scenario, network, hparams, render=render)
            cost = episode_cost(metrics, hparams)
            total_cost += cost
            status = (
                "goal"
                if metrics.reached_goal
                else "collision" if metrics.collided else "timeout"
            )
            print(
                f"  Scenario {idx}: steps={metrics.steps:4d} status={status:8s} "
                f"min_sep={metrics.min_separation:6.2f}m colregs={metrics.wrong_action_cost:6.2f} "
                f"cost={cost:7.2f}"
            )
        print(f"Average cost: {total_cost / len(scenario_list):.2f}")
    finally:
        env.close()


def main(argv: Optional[list[str]] = None) -> None:
    hparams = HyperParameters()
    parser = build_parser(hparams)
    args = parser.parse_args(argv)

    if args.list_hyperparameters:
        print_hyperparameters(hparams)
        return

    try:
        apply_cli_overrides(hparams, args.hp)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    scenario_request = build_scenario_request(hparams)
    scenarios = build_scenarios(scenario_request)

    boat_params = build_boat_params(hparams)
    turn_cfg = build_turn_config(hparams)
    env_cfg = build_env_config(hparams, render=False)

    result = train_population(
        config_path=args.config,
        scenarios=scenarios,
        env_cfg=env_cfg,
        boat_params=boat_params,
        turn_cfg=turn_cfg,
        params=hparams,
        generations=args.generations,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    if args.save_winner is not None:
        args.save_winner.parent.mkdir(parents=True, exist_ok=True)
        with args.save_winner.open("wb") as fh:
            pickle.dump(result.winner, fh)
        print(f"Saved winning genome to {args.save_winner}")

    render_cfg = build_env_config(hparams, render=args.render)
    summarise_winner(
        result,
        scenarios,
        hparams,
        boat_params,
        turn_cfg,
        render_cfg,
        render=args.render,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

