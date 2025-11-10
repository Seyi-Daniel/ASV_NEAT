"""Utilities for training a give-way controller with NEAT-Python."""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import neat
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "The neat-python package is required to use the NEAT training helpers."
    ) from exc

from .config import BoatParams
from .env import CrossingScenarioEnv
from .scenario import (
    STAND_ON_BEARINGS_DEG,
    CrossingScenario,
    ScenarioRequest,
    iter_scenarios,
    scenario_states_for_env,
)
from .utils import clamp


@dataclass
class EpisodeMetrics:
    """Summary of a single NEAT-controlled roll-out."""

    steps: int
    reached_goal: bool
    collided: bool
    final_distance: float
    min_separation: float


@dataclass
class FitnessParameters:
    """Cost-shaping weights for the minimisation-based fitness."""

    max_steps: int = 1000
    goal_tolerance: float = 10.0
    collision_distance: float = 8.0
    timeout_penalty: float = 300.0
    collision_penalty: float = 1200.0
    distance_penalty: float = 1.5
    separation_threshold: float = 15.0
    separation_penalty: float = 8.0
    distance_normaliser: float = 250.0


def _argmax_index(values: Sequence[float]) -> int:
    best_idx = 0
    best_val = float("-inf")
    for idx, val in enumerate(values):
        if val > best_val:
            best_idx = idx
            best_val = val
    return best_idx


def _goal_distance(state: dict) -> float:
    gx = float(state.get("goal_x", state["x"]))
    gy = float(state.get("goal_y", state["y"]))
    return math.hypot(gx - float(state["x"]), gy - float(state["y"]))


def _separation(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def agent_feature_vector(state: dict, kin: BoatParams, scale: float) -> List[float]:
    """Six agent-centric, normalised inputs for the NEAT controller."""

    x = float(state["x"])
    y = float(state["y"])
    heading = float(state["heading"])
    speed = float(state.get("speed", 0.0))
    gx = float(state.get("goal_x", x))
    gy = float(state.get("goal_y", y))

    dx = gx - x
    dy = gy - y
    distance = math.hypot(dx, dy)
    scale = max(1.0, float(scale))
    max_speed = max(1.0, float(kin.max_speed))

    return [
        clamp(dx / scale, -1.0, 1.0),
        clamp(dy / scale, -1.0, 1.0),
        math.cos(heading),
        math.sin(heading),
        clamp(speed / max_speed, -1.0, 1.0),
        clamp(distance / scale, -1.0, 1.0),
    ]


def episode_cost(metrics: EpisodeMetrics, params: FitnessParameters) -> float:
    """Compute the scalar cost associated with ``metrics``."""

    cost = float(metrics.steps)

    if not metrics.reached_goal:
        cost += params.timeout_penalty
        normaliser = max(1.0, params.distance_normaliser)
        cost += params.distance_penalty * (metrics.final_distance / normaliser)

    if metrics.collided:
        cost += params.collision_penalty

    if math.isfinite(metrics.min_separation):
        deficit = max(0.0, params.separation_threshold - metrics.min_separation)
        if params.separation_threshold > 0.0:
            deficit /= params.separation_threshold
        cost += params.separation_penalty * deficit

    return cost


def run_episode(
    env: CrossingScenarioEnv,
    scenario: CrossingScenario,
    network,
    kin: BoatParams,
    params: FitnessParameters,
    feature_scale: float,
    *,
    render: bool = False,
) -> EpisodeMetrics:
    states, meta = scenario_states_for_env(env, scenario)
    env.reset_from_states(states, meta=meta)

    if render:
        env.render()

    min_sep = float("inf")
    steps = 0

    for step in range(params.max_steps):
        snapshot = env.snapshot()
        if not snapshot:
            break
        agent_state = snapshot[0]
        stand_on_state = snapshot[1] if len(snapshot) > 1 else None

        features = agent_feature_vector(agent_state, kin, feature_scale)
        outputs = network.activate(features)
        action = _argmax_index(outputs)

        env.step([action, None])
        steps = step + 1

        if render:
            env.render()

        snapshot = env.snapshot()
        if not snapshot:
            break
        agent_state = snapshot[0]
        stand_on_state = snapshot[1] if len(snapshot) > 1 else None
        distance = _goal_distance(agent_state)

        if stand_on_state is not None:
            sep = _separation(agent_state, stand_on_state)
            min_sep = min(min_sep, sep)
            if sep <= params.collision_distance:
                return EpisodeMetrics(
                    steps=steps,
                    reached_goal=False,
                    collided=True,
                    final_distance=distance,
                    min_separation=min_sep,
                )

        if distance <= params.goal_tolerance:
            return EpisodeMetrics(
                steps=steps,
                reached_goal=True,
                collided=False,
                final_distance=distance,
                min_separation=min_sep,
            )

    snapshot = env.snapshot()
    if snapshot:
        agent_state = snapshot[0]
        distance = _goal_distance(agent_state)
        stand_on_state = snapshot[1] if len(snapshot) > 1 else None
        if stand_on_state is not None:
            min_sep = min(min_sep, _separation(agent_state, stand_on_state))
    else:
        distance = 0.0

    if render:
        env.render()

    return EpisodeMetrics(
        steps=max(steps, params.max_steps),
        reached_goal=False,
        collided=False,
        final_distance=distance,
        min_separation=min_sep,
    )


def evaluate_population(
    genomes,
    config,
    env: CrossingScenarioEnv,
    scenarios: Sequence[CrossingScenario],
    kin: BoatParams,
    params: FitnessParameters,
    feature_scale: float,
) -> None:
    """Assign fitness values to ``genomes`` using the supplied ``env``."""

    for genome_id, genome in genomes:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        total_cost = 0.0
        for scenario in scenarios:
            metrics = run_episode(env, scenario, network, kin, params, feature_scale)
            total_cost += episode_cost(metrics, params)
        genome.fitness = total_cost / len(scenarios)


@dataclass
class TrainingResult:
    """Outcome of a NEAT training run."""

    winner: neat.DefaultGenome
    config: neat.Config
    statistics: neat.StatisticsReporter


def train_population(
    config_path: Path,
    env: CrossingScenarioEnv,
    scenarios: Sequence[CrossingScenario],
    kin: BoatParams,
    params: FitnessParameters,
    feature_scale: float,
    generations: int,
    seed: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 5,
) -> TrainingResult:
    """Run NEAT evolution with the provided environment and scenarios."""

    if seed is not None:
        random.seed(seed)

    cfg_path = Path(config_path)
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(cfg_path),
    )

    population = neat.Population(neat_config)
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prefix = os.path.join(str(checkpoint_dir), "neat-checkpoint-")
        population.add_reporter(
            neat.Checkpointer(
                generation_interval=max(1, checkpoint_interval),
                filename_prefix=prefix,
            )
        )

    def _eval(genomes, config):
        evaluate_population(genomes, config, env, scenarios, kin, params, feature_scale)

    winner = population.run(_eval, generations)

    return TrainingResult(winner=winner, config=neat_config, statistics=stats)


def build_scenarios(request: ScenarioRequest) -> List[CrossingScenario]:
    """Construct the five deterministic scenarios for the given request."""

    return list(iter_scenarios(STAND_ON_BEARINGS_DEG, request))


__all__ = [
    "EpisodeMetrics",
    "FitnessParameters",
    "TrainingResult",
    "agent_feature_vector",
    "episode_cost",
    "run_episode",
    "evaluate_population",
    "train_population",
    "build_scenarios",
]

