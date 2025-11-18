#!/usr/bin/env python3
"""Train the give-way vessel controller across the deterministic COLREGs encounters."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Optional

try:  # pragma: no cover - optional dependency
    import neat  # noqa: F401
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("neat-python must be installed to run the training script.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asv_neat import (  # noqa: E402
    BoatParams,
    EnvConfig,
    HyperParameters,
    ScenarioKind,
    ScenarioRequest,
    TurnSessionConfig,
    apply_cli_overrides,
    build_scenarios,
    summarise_genome,
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
        "--save-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "winners",
        help="Directory used for automatically saving winners when --save-winner is not provided.",
    )
    parser.add_argument(
        "--disable-auto-save",
        action="store_true",
        help="Skip the default automatic saving of the winning genome.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame visualisation while summarising the winning genome.",
    )
    scenario_choices = ["all"] + [kind.value for kind in ScenarioKind]
    parser.add_argument(
        "--scenario",
        choices=scenario_choices,
        default="all",
        help=(
            "Encounter family to train on. Use 'all' to evaluate every crossing, head-on and "
            "overtaking scenario together."
        ),
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
    print(
        "\nScenario-specific overrides can be provided by prefixing the name with "
        "'crossing.', 'head_on.' or 'overtaking.' (e.g. --hp crossing.goal_bonus=-55)."
    )


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
        goal_extension=hparams.scenario_goal_extension,
        crossing_agent_speed=hparams.scenario_crossing_agent_speed,
        crossing_stand_on_speed=hparams.scenario_crossing_stand_on_speed,
        head_on_agent_speed=hparams.scenario_head_on_agent_speed,
        head_on_stand_on_speed=hparams.scenario_head_on_stand_on_speed,
        overtaking_agent_speed=hparams.scenario_overtaking_agent_speed,
        overtaking_stand_on_speed=hparams.scenario_overtaking_stand_on_speed,
    )


def _scenario_selection(choice: str) -> Optional[List[ScenarioKind]]:
    if choice == "all":
        return None
    return [ScenarioKind(choice)]


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
    selected_kinds = _scenario_selection(args.scenario)
    scenarios = build_scenarios(scenario_request, kinds=selected_kinds)
    if not scenarios:
        parser.error("No scenarios were generated for the selected encounter type.")

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
        winner_path = args.save_winner
    elif args.disable_auto_save:
        winner_path = None
    else:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        winner_filename = f"{args.scenario}_winner.pkl"
        winner_path = args.save_dir / winner_filename

    if winner_path is not None:
        winner_path.parent.mkdir(parents=True, exist_ok=True)
        with winner_path.open("wb") as fh:
            pickle.dump(result.winner, fh)
        print(f"Saved winning genome to {winner_path}")

    render_cfg = build_env_config(hparams, render=args.render)
    summarise_genome(
        result.winner,
        result.config,
        scenarios,
        hparams,
        boat_params,
        turn_cfg,
        render_cfg,
        render=args.render,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

