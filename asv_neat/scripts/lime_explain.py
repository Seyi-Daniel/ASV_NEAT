#!/usr/bin/env python3
"""Generate LIME explanations for a saved NEAT controller."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from lime import lime_tabular
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'lime' package is required to run the explanation script.") from exc

try:  # pragma: no cover - optional dependency
    import neat
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("neat-python must be installed to run the explanation script.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asv_neat import (  # noqa: E402
    HyperParameters,
    apply_cli_overrides,
    build_scenarios,
    episode_cost,
    simulate_episode,
)
from asv_neat.boat import Boat  # noqa: E402
from asv_neat.cli_helpers import (  # noqa: E402
    SCENARIO_KIND_CHOICES,
    build_boat_params,
    build_env_config,
    build_scenario_request,
    build_turn_config,
    filter_scenarios_by_kind,
)
from asv_neat.config import BoatParams, EnvConfig, TurnSessionConfig  # noqa: E402
from asv_neat.env import CrossingScenarioEnv  # noqa: E402
from asv_neat.neat_training import TraceCallback  # noqa: E402
from asv_neat.scenario import EncounterScenario  # noqa: E402

FEATURE_NAMES: List[str] = [
    "agent_x",
    "agent_y",
    "agent_heading",
    "agent_speed",
    "agent_goal_x",
    "agent_goal_y",
    "stand_on_x",
    "stand_on_y",
    "stand_on_heading",
    "stand_on_speed",
    "stand_on_goal_x",
    "stand_on_goal_y",
]


def _action_label(action: int) -> str:
    helm, thr = Boat.decode_action(action)
    helm_map = { -1: "turn_port", 0: "hold_course", 1: "turn_starboard" }
    thr_map = { -1: "decelerate", 0: "hold_speed", 1: "accelerate" }
    return f"{helm_map[helm]}|{thr_map[thr]}"


ACTION_LABELS: List[str] = [_action_label(idx) for idx in range(9)]


class LimeNetworkWrapper:
    """Adapter that exposes the NEAT network with a scikit-learn style API."""

    def __init__(self, network: neat.nn.FeedForwardNetwork) -> None:
        self._network = network

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        predictions = []
        for row in data:
            outputs = np.asarray(self._network.activate(row.tolist()), dtype=float)
            if outputs.ndim == 0:
                outputs = np.asarray([outputs], dtype=float)
            predictions.append(outputs)
        logits = np.asarray(predictions, dtype=float)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        denom = exp.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return exp / denom


def _serialise_vessel(state) -> dict:
    data = asdict(state)
    goal = data.get("goal")
    if goal is not None:
        data["goal"] = list(goal)
    return data


def _build_parser(hparams: HyperParameters) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "neat_crossing.cfg",
        help="Path to the neat-python configuration file that matches the saved genome.",
    )
    parser.add_argument(
        "--winner",
        type=Path,
        default=None,
        help="Path to the pickled winning genome. Defaults to winners/<scenario>_winner.pkl.",
    )
    parser.add_argument(
        "--scenario-kind",
        choices=SCENARIO_KIND_CHOICES,
        default="all",
        help="Select which encounter family should be explained.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "lime_reports",
        help="Directory where explanation artefacts will be written.",
    )
    parser.add_argument(
        "--hp",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a hyperparameter (repeatable).",
    )
    return parser


def _trace_recorder(container: List[dict]) -> TraceCallback:
    def _record(
        step_idx: int,
        features: List[float],
        outputs,
        action: int,
        agent_state: dict,
        stand_on_state: Optional[dict],
    ) -> None:
        container.append(
            {
                "step": step_idx,
                "features": features,
                "outputs": list(outputs),
                "predicted_action": action,
                "agent_state": agent_state,
                "stand_on_state": stand_on_state,
            }
        )

    return _record


def _scenario_metadata(
    scenario: EncounterScenario,
    metrics,
    cost: float,
) -> dict:
    metadata = {
        "scenario_kind": scenario.kind.value,
        "requested_bearing_deg": scenario.requested_bearing,
        "bearing_frame": scenario.bearing_frame,
        "agent": _serialise_vessel(scenario.agent),
        "stand_on": _serialise_vessel(scenario.stand_on),
        "metrics": asdict(metrics),
        "episode_cost": cost,
    }
    return metadata


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _generate_lime(
    scenario_dir: Path,
    trace: List[dict],
    network: neat.nn.FeedForwardNetwork,
) -> List[dict]:
    if not trace:
        return []

    features = np.asarray([item["features"] for item in trace], dtype=float)
    wrapper = LimeNetworkWrapper(network)
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=features,
        feature_names=FEATURE_NAMES,
        class_names=ACTION_LABELS,
        discretize_continuous=False,
    )

    explanations: List[dict] = []
    for item in trace:
        probs = wrapper.predict_proba(np.asarray([item["features"]], dtype=float))[0]
        exp = explainer.explain_instance(
            np.asarray(item["features"], dtype=float),
            wrapper.predict_proba,
            num_features=len(FEATURE_NAMES),
            top_labels=1,
        )
        label = item["predicted_action"]
        attribution = [
            {
                "feature": FEATURE_NAMES[idx],
                "weight": float(weight),
                "value": float(item["features"][idx]),
            }
            for idx, weight in exp.as_map()[label]
        ]
        explanation = {
            "step": item["step"],
            "predicted_action": label,
            "action_label": ACTION_LABELS[label],
            "input_features": item["features"],
            "network_outputs": item["outputs"],
            "predicted_probabilities": probs.tolist(),
            "feature_attributions": attribution,
        }
        explanations.append(explanation)
        _write_json(scenario_dir / f"lime_step_{item['step']:03d}.json", explanation)

    _write_json(scenario_dir / "lime_summary.json", explanations)
    return explanations


def _load_winner(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def explain_scenarios(
    *,
    winner_path: Path,
    config_path: Path,
    scenarios: Iterable[EncounterScenario],
    hparams: HyperParameters,
    boat_params: BoatParams,
    turn_cfg: TurnSessionConfig,
    env_cfg: EnvConfig,
    output_dir: Path,
) -> None:
    winner = _load_winner(winner_path)
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    network = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    for idx, scenario in enumerate(scenarios, start=1):
        scenario_dir = output_dir / f"{idx:02d}_{scenario.kind.value}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params, tcfg=turn_cfg)
        trace: List[dict] = []
        try:
            recorder = _trace_recorder(trace)
            metrics = simulate_episode(
                env,
                scenario,
                network,
                hparams,
                render=False,
                trace_callback=recorder,
            )
        finally:
            env.close()
        cost = episode_cost(metrics, hparams)
        metadata = _scenario_metadata(scenario, metrics, cost)
        _write_json(scenario_dir / "metadata.json", metadata)
        _write_json(scenario_dir / "trace.json", trace)
        _generate_lime(scenario_dir, trace, network)
        print(
            f"Scenario {idx:02d} [{scenario.kind.value}] steps={metrics.steps:4d} "
            f"cost={cost:7.2f} outputs saved to {scenario_dir}"
        )


def main(argv: Optional[list[str]] = None) -> None:
    hparams = HyperParameters()
    parser = _build_parser(hparams)
    args = parser.parse_args(argv)

    try:
        apply_cli_overrides(hparams, args.hp)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    scenario_request = build_scenario_request(hparams)
    scenarios = filter_scenarios_by_kind(
        build_scenarios(scenario_request), args.scenario_kind
    )
    if not scenarios:
        parser.error("No scenarios available for the requested encounter kind.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    winner_path = args.winner
    if winner_path is None:
        winner_path = PROJECT_ROOT / "winners" / f"{args.scenario_kind}_winner.pkl"
    if not winner_path.exists():
        parser.error(f"Winner file '{winner_path}' does not exist.")

    boat_params = build_boat_params(hparams)
    turn_cfg = build_turn_config(hparams)
    env_cfg = build_env_config(hparams, render=False)

    explain_scenarios(
        winner_path=winner_path,
        config_path=args.config,
        scenarios=scenarios,
        hparams=hparams,
        boat_params=boat_params,
        turn_cfg=turn_cfg,
        env_cfg=env_cfg,
        output_dir=output_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
