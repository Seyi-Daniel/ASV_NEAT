#!/usr/bin/env python3
"""Generate SHAP explanations and visualisations for a saved NEAT controller."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'imageio' package is required to generate videos.") from exc

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'matplotlib' package is required to plot explanations.") from exc

try:  # pragma: no cover - optional dependency
    import shap
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'shap' package is required to run the explanation script.") from exc

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'Pillow' package is required to compose visualisations.") from exc

try:  # pragma: no cover - optional dependency
    import pygame
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'pygame' package is required to capture render frames.") from exc

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
from asv_neat.cli_helpers import (  # noqa: E402
    SCENARIO_KIND_CHOICES,
    build_boat_params,
    build_env_config,
    build_scenario_request,
    build_rudder_config,
    filter_scenarios_by_kind,
)
from asv_neat.config import BoatParams, EnvConfig, RudderParams  # noqa: E402
from asv_neat.env import CrossingScenarioEnv  # noqa: E402
from asv_neat.neat_training import TraceCallback  # noqa: E402
from asv_neat.scenario import EncounterScenario  # noqa: E402
from asv_neat.paths import default_winner_path  # noqa: E402
from asv_neat.utils import helm_label_from_rudder_cmd  # noqa: E402

FEATURE_NAMES: List[str] = [
    "x_goal_TV",
    "y_goal_TV",
    "speed_TV",
    "heading_TV",
    "x_TV",
    "y_TV",
    "x_goal_ASV",
    "y_goal_ASV",
    "speed_ASV",
    "heading_ASV",
    "x_ASV",
    "y_ASV",
]

FRAME_DIRNAME = "frames"
PLOT_DIRNAME = "plots"
COMBINED_DIRNAME = "combined_frames"
RUDDER_ANIMATION_FILENAME = "explanation_rudder_animation.gif"
THROTTLE_ANIMATION_FILENAME = "explanation_throttle_animation.gif"


THROTTLE_LABELS: List[str] = ["hold speed", "accelerate", "decelerate"]


def _format_rudder_angle(angle_rad: float) -> str:
    angle_deg = math.degrees(angle_rad)
    if abs(angle_deg) < 1e-6:
        return "0.0"
    return f"{angle_deg:+.1f}"


def _throttle_distribution(throttle_val: float) -> float:
    return max(0.0, min(1.0, throttle_val)) * 2.0


class ShapNetworkWrapper:
    """Adapter exposing rudder and throttle outputs for SHAP."""

    def __init__(self, network: neat.nn.FeedForwardNetwork) -> None:
        self._network = network

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        predictions = []
        for row in data:
            outputs = np.asarray(self._network.activate(row.tolist()), dtype=float)
            rudder = float(outputs[0]) if outputs.ndim > 0 and outputs.size > 0 else 0.0
            throttle_val = float(outputs[1]) if outputs.ndim > 0 and outputs.size > 1 else 0.0
            predictions.append([rudder, _throttle_distribution(throttle_val)])
        return np.asarray(predictions, dtype=float)


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
        default=PROJECT_ROOT / "shap_reports",
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
    def _record(payload: dict) -> None:
        if "features" not in payload and "obs" in payload:
            payload = {**payload, "features": list(payload["obs"])}
        container.append(payload)

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


def _save_frame(path: Path, surface) -> None:
    if surface is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))


def _plot_explanation(step_data: dict, output_path: Path, *, title: str) -> None:
    weights = {item["feature"]: item["shap_value"] for item in step_data["feature_attributions"]}
    values = [weights.get(name, 0.0) for name in FEATURE_NAMES]
    colors = ["#3CB371" if weight >= 0 else "#D95F02" for weight in values]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(FEATURE_NAMES, values, color=colors)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("SHAP value", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_explanations(explanations: List[dict], plot_dir: Path) -> List[tuple[int, Path, Path]]:
    plot_paths: List[tuple[int, Path, Path]] = []
    for item in explanations:
        step = item["step"]
        rudder_path = plot_dir / f"explanation_rudder_{step:03d}.png"
        throttle_path = plot_dir / f"explanation_throttle_{step:03d}.png"
        rudder_info = item["rudder"]
        actual_rudder = float(rudder_info.get("actual_rudder", 0.0))
        helm_label = rudder_info.get("helm_label") or "keep_straight"
        _plot_explanation(
            rudder_info,
            rudder_path,
            title=(
                f"Step: {step:03d}  Rudder angle: {_format_rudder_angle(actual_rudder)}° "
                f"Turn action: {helm_label.replace('_', ' ')}"
            ),
        )
        _plot_explanation(
            item["throttle"],
            throttle_path,
            title=(
                f"Step: {step:03d}  Throttle action: "
                f"{THROTTLE_LABELS[item['throttle']['prediction']]}"
            ),
        )
        plot_paths.append((step, rudder_path, throttle_path))
    return plot_paths


def _combine_images(scene_path: Path, plot_path: Path, output_path: Path) -> None:
    scene = Image.open(scene_path).convert("RGB")
    plot = Image.open(plot_path).convert("RGB")
    height = max(scene.height, plot.height)
    canvas = Image.new("RGB", (scene.width + plot.width, height), color=(0, 0, 0))
    canvas.paste(scene, (0, 0))
    canvas.paste(plot, (scene.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _combine_frames(
    frame_dir: Path, plot_dir: Path, combined_dir: Path, *, kind: str
) -> List[Path]:
    combined: List[Path] = []
    # Sort frames numerically to avoid lexicographic ordering (e.g., 1000 before
    # 200) when step indices exceed the padding width.
    for frame_path in sorted(frame_dir.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[-1])):
        step = frame_path.stem.split("_")[-1]
        plot_path = plot_dir / f"explanation_{kind}_{step}.png"
        if not plot_path.exists():
            continue
        output_path = combined_dir / f"combined_{kind}_{step}.png"
        _combine_images(frame_path, plot_path, output_path)
        combined.append(output_path)
    return combined


def _write_animation(frame_paths: List[Path], output_path: Path, fps: int = 8) -> None:
    if not frame_paths:
        return
    images = [imageio.imread(path) for path in frame_paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, images, fps=fps)


def _select_shap_values(shap_values, label: int) -> np.ndarray:
    values = shap_values[label] if isinstance(shap_values, list) else shap_values
    arr = np.asarray(values, dtype=float)

    # ``shap.KernelExplainer`` can return a 3D array for multi-output models where
    # the final dimension holds a SHAP value per output class. Extract the column
    # for the predicted label while preserving the feature dimension.
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape[1] > 1:
            col = min(label, arr.shape[1] - 1)
            arr = arr[:, col]

    # Handle 2D outputs (nsamples x nfeatures) and select the label column if
    # present.
    if arr.ndim == 2:
        if arr.shape[1] > 1:
            col = min(label, arr.shape[1] - 1)
            arr = arr[:, col]
        elif arr.shape[0] == 1:
            arr = arr[0]

    # ``arr`` should now be 1D where each entry corresponds to a feature.
    arr = np.squeeze(arr)
    return np.asarray(arr, dtype=float).reshape(-1)


def _expected_value_for_label(expected_value, label: int) -> float:
    values = expected_value[label] if isinstance(expected_value, list) else expected_value
    arr = np.asarray(values, dtype=float)

    if arr.ndim == 0:
        return float(arr)

    if arr.ndim == 1:
        idx = min(label, arr.shape[0] - 1)
        return float(arr[idx])

    if arr.ndim == 2:
        row = 0 if arr.shape[0] == 1 else min(label, arr.shape[0] - 1)
        col = min(label, arr.shape[1] - 1)
        return float(arr[row, col])

    return float(arr.flat[0])


def _generate_shap(
    scenario_dir: Path,
    trace: List[dict],
    network: neat.nn.FeedForwardNetwork,
) -> List[dict]:
    if not trace:
        return []

    features = np.asarray([item.get("features") or item.get("obs") or [] for item in trace], dtype=float)
    wrapper = ShapNetworkWrapper(network)
    explainer = shap.KernelExplainer(wrapper.predict_proba, features)

    explanations: List[dict] = []
    for item in trace:
        features = item.get("features") or item.get("obs") or []
        instance = np.asarray([features], dtype=float)
        outputs = wrapper.predict_proba(instance)[0]
        throttle_command = min(2, max(0, int(round(outputs[1]))))
        shap_values = explainer.shap_values(instance, nsamples="auto")

        rudder_expected = _expected_value_for_label(explainer.expected_value, 0)
        rudder_vector = _select_shap_values(shap_values, 0)
        rudder_attr = [
            {
                "feature": FEATURE_NAMES[idx],
                "shap_value": float(value),
                "value": float(features[idx]),
            }
            for idx, value in enumerate(rudder_vector)
        ]

        throttle_expected = _expected_value_for_label(explainer.expected_value, 1)
        throttle_vector = _select_shap_values(shap_values, 1)
        throttle_attr = [
            {
                "feature": FEATURE_NAMES[idx],
                "shap_value": float(value),
                "value": float(features[idx]),
            }
            for idx, value in enumerate(throttle_vector)
        ]

        rudder_cmd = float(item.get("rudder_cmd", outputs[0]))
        helm_label = item.get("helm_label") or helm_label_from_rudder_cmd(rudder_cmd)
        actual_rudder = float(item.get("agent_state", {}).get("rudder", 0.0))

        explanation = {
            "step": item["step"],
            "rudder": {
                "prediction": float(outputs[0]),
                "rudder_cmd": rudder_cmd,
                "helm_label": helm_label,
                "actual_rudder": actual_rudder,
                "expected_value": rudder_expected,
                "feature_attributions": rudder_attr,
            },
            "throttle": {
                "prediction": throttle_command,
                "raw_output": float(outputs[1]),
                "expected_value": throttle_expected,
                "feature_attributions": throttle_attr,
            },
        }
        explanations.append(explanation)
        _write_json(scenario_dir / f"shap_step_{item['step']:03d}.json", explanation)

    _write_json(scenario_dir / "shap_summary.json", explanations)
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
    rudder_cfg: RudderParams,
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
        frame_dir = scenario_dir / FRAME_DIRNAME
        plot_dir = scenario_dir / PLOT_DIRNAME
        combined_dir = scenario_dir / COMBINED_DIRNAME
        env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params, rudder_cfg=rudder_cfg)
        trace: List[dict] = []
        try:
            recorder = _trace_recorder(trace)

            def frame_recorder(step: int, surf) -> None:
                _save_frame(frame_dir / f"frame_{step:03d}.png", surf)

            metrics = simulate_episode(
                env,
                scenario,
                network,
                hparams,
                render=True,
                trace_callback=recorder,
                frame_callback=frame_recorder,
            )
        finally:
            env.close()
        cost = episode_cost(metrics, hparams)
        metadata = _scenario_metadata(scenario, metrics, cost)
        _write_json(scenario_dir / "metadata.json", metadata)
        _write_json(scenario_dir / "trace.json", trace)
        explanations = _generate_shap(scenario_dir, trace, network)
        _plot_explanations(explanations, plot_dir)
        combined_rudder_frames = _combine_frames(
            frame_dir, plot_dir, combined_dir / "rudder", kind="rudder"
        )
        combined_throttle_frames = _combine_frames(
            frame_dir, plot_dir, combined_dir / "throttle", kind="throttle"
        )
        _write_animation(combined_rudder_frames, scenario_dir / RUDDER_ANIMATION_FILENAME)
        _write_animation(
            combined_throttle_frames, scenario_dir / THROTTLE_ANIMATION_FILENAME
        )
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
        winner_path = default_winner_path(args.scenario_kind)
    if not winner_path.exists():
        parser.error(f"Winner file '{winner_path}' does not exist.")

    boat_params = build_boat_params(hparams)
    rudder_cfg = build_rudder_config(hparams)
    env_cfg = build_env_config(hparams, render=True)

    explain_scenarios(
        winner_path=winner_path,
        config_path=args.config,
        scenarios=scenarios,
        hparams=hparams,
        boat_params=boat_params,
        rudder_cfg=rudder_cfg,
        env_cfg=env_cfg,
        output_dir=output_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
