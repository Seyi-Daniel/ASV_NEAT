"""NEAT integration for the deterministic COLREGs crossing scenarios."""
from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import neat

from .boat import Boat
from .config import BoatParams, EnvConfig, TurnSessionConfig
from .env import CrossingScenarioEnv
from .hyperparameters import HyperParameters
from .scenario import (
    EncounterScenario,
    ScenarioRequest,
    default_scenarios,
    scenario_states_for_env,
)
from .utils import (
    euclidean_distance,
    goal_distance,
    heading_error_deg,
    relative_bearing_deg,
    tcpa_dcpa,
)


@dataclass
class EpisodeMetrics:
    """Summary statistics produced by a single simulated episode."""

    steps: int
    reached_goal: bool
    collided: bool
    final_distance: float
    min_separation: float
    wrong_action_cost: float
    goal_progress_bonus: float


def _argmax(values: Sequence[float]) -> int:
    best_idx = 0
    best_val = float("-inf")
    for idx, val in enumerate(values):
        if val > best_val:
            best_idx = idx
            best_val = val
    return best_idx


def _normalise(value: float, scale: float) -> float:
    if scale <= 0.0:
        return value
    return max(-1.0, min(1.0, value / scale))


def observation_vector(
    agent: dict,
    stand_on: dict,
    params: HyperParameters,
) -> List[float]:
    """Return the 12-element feature vector consumed by the controller."""

    def pack(state: dict) -> List[float]:
        x = _normalise(float(state["x"]), params.feature_position_scale)
        y = _normalise(float(state["y"]), params.feature_position_scale)
        heading = _normalise(float(state["heading"]), params.feature_heading_scale)
        speed = _normalise(float(state.get("speed", 0.0)), params.feature_speed_scale)
        goal_x = _normalise(float(state.get("goal_x", state["x"])), params.feature_position_scale)
        goal_y = _normalise(float(state.get("goal_y", state["y"])), params.feature_position_scale)
        return [x, y, heading, speed, goal_x, goal_y]

    return pack(agent) + pack(stand_on)


def simulate_episode(
    env: CrossingScenarioEnv,
    scenario: EncounterScenario,
    network: neat.nn.FeedForwardNetwork,
    params: HyperParameters,
    *,
    render: bool = False,
) -> EpisodeMetrics:
    """Roll a network-controlled episode within ``env`` for ``scenario``."""

    states, meta = scenario_states_for_env(env, scenario)
    env.reset_from_states(states, meta=meta)

    if render:
        env.enable_render()

    min_sep = float("inf")
    wrong_action_cost = 0.0
    goal_progress_bonus = 0.0
    steps = 0

    for step_idx in range(params.max_steps):
        snapshot = env.snapshot()
        if not snapshot:
            break

        agent_state = snapshot[0]
        stand_on_state = snapshot[1] if len(snapshot) > 1 else snapshot[0]
        previous_distance = goal_distance(agent_state)
        previous_heading_error = heading_error_deg(agent_state)

        features = observation_vector(agent_state, stand_on_state, params)
        outputs = network.activate(features)
        action = _argmax(outputs)
        helm, _ = Boat.decode_action(action)

        env.step([action, None])
        steps = step_idx + 1

        if render:
            env.render()

        snapshot = env.snapshot()
        if not snapshot:
            break
        agent_state = snapshot[0]
        stand_on_state = snapshot[1] if len(snapshot) > 1 else None

        distance = goal_distance(agent_state)
        heading_error = heading_error_deg(agent_state)

        distance_improved = distance < (previous_distance - 1e-6)
        distance_regressed = distance > (previous_distance + 1e-6)

        prev_outside = previous_heading_error > params.heading_alignment_threshold_deg
        new_outside = heading_error > params.heading_alignment_threshold_deg
        heading_delta = heading_error - previous_heading_error
        heading_improved = heading_delta < 0.0 and (prev_outside or new_outside)
        heading_regressed = heading_delta > 0.0 and new_outside

        reward_step = False
        penalty_step = False

        if distance_improved and (heading_delta <= 0.0 or not new_outside or heading_improved):
            reward_step = True
        elif distance_regressed and (heading_regressed or new_outside):
            penalty_step = True

        if not reward_step and heading_improved:
            reward_step = True
        if not penalty_step and heading_regressed and not distance_improved:
            penalty_step = True

        if reward_step:
            goal_progress_bonus += params.goal_progress_bonus
        elif penalty_step:
            goal_progress_bonus -= params.goal_progress_bonus

        if stand_on_state is not None:
            sep = euclidean_distance(
                float(agent_state["x"]),
                float(agent_state["y"]),
                float(stand_on_state["x"]),
                float(stand_on_state["y"]),
            )
            min_sep = min(min_sep, sep)

            if sep <= params.collision_distance:
                return EpisodeMetrics(
                    steps=steps,
                    reached_goal=False,
                    collided=True,
                    final_distance=distance,
                    min_separation=min_sep,
                    wrong_action_cost=wrong_action_cost,
                    goal_progress_bonus=goal_progress_bonus,
                )

            tcpa, dcpa = tcpa_dcpa(agent_state, stand_on_state)
            bearing = relative_bearing_deg(agent_state, stand_on_state)
            if (
                0.0 <= tcpa <= params.tcpa_threshold
                and dcpa <= params.dcpa_threshold
                and bearing <= params.angle_threshold_deg
            ):
                if helm != 1:
                    wrong_action_cost += params.wrong_action_penalty

        if distance <= params.goal_tolerance:
            return EpisodeMetrics(
                steps=steps,
                reached_goal=True,
                collided=False,
                final_distance=distance,
                min_separation=min_sep,
                wrong_action_cost=wrong_action_cost,
                goal_progress_bonus=goal_progress_bonus,
            )

    snapshot = env.snapshot()
    if snapshot:
        agent_state = snapshot[0]
        distance = goal_distance(agent_state)
        if len(snapshot) > 1:
            min_sep = min(
                min_sep,
                euclidean_distance(
                    float(agent_state["x"]),
                    float(agent_state["y"]),
                    float(snapshot[1]["x"]),
                    float(snapshot[1]["y"]),
                ),
            )
    else:
        distance = 0.0

    return EpisodeMetrics(
        steps=max(steps, params.max_steps),
        reached_goal=False,
        collided=False,
        final_distance=distance,
        min_separation=min_sep,
        wrong_action_cost=wrong_action_cost,
        goal_progress_bonus=goal_progress_bonus,
    )


def episode_cost(metrics: EpisodeMetrics, params: HyperParameters) -> float:
    """Convert ``metrics`` into a scalar cost value (lower is better)."""

    cost = params.step_cost * metrics.steps

    if metrics.reached_goal:
        cost += params.goal_bonus
    else:
        cost += params.timeout_penalty
        normaliser = max(1.0, params.distance_normaliser)
        cost += params.distance_cost * (metrics.final_distance / normaliser)

    if metrics.collided:
        cost += params.collision_penalty

    cost += metrics.wrong_action_cost
    cost += metrics.goal_progress_bonus
    return cost


def _make_env(cfg: EnvConfig, kin: BoatParams, turn: TurnSessionConfig) -> CrossingScenarioEnv:
    return CrossingScenarioEnv(cfg=cfg, kin=kin, tcfg=turn)


def evaluate_individual(
    genome,
    config,
    scenarios: Sequence[EncounterScenario],
    env_cfg: EnvConfig,
    boat_params: BoatParams,
    turn_cfg: TurnSessionConfig,
    params: HyperParameters,
) -> float:
    """Return the average cost accrued by ``genome`` over all scenarios."""

    def run_single(scenario: EncounterScenario) -> EpisodeMetrics:
        local_network = neat.nn.FeedForwardNetwork.create(genome, config)
        env = _make_env(env_cfg, boat_params, turn_cfg)
        try:
            return simulate_episode(env, scenario, local_network, params)
        finally:
            env.close()

    with ThreadPoolExecutor(max_workers=len(scenarios)) as executor:
        metrics = list(executor.map(run_single, scenarios))

    total_cost = sum(episode_cost(item, params) for item in metrics)
    return total_cost / len(metrics)


def evaluate_population(
    genomes,
    config,
    scenarios: Sequence[EncounterScenario],
    env_cfg: EnvConfig,
    boat_params: BoatParams,
    turn_cfg: TurnSessionConfig,
    params: HyperParameters,
) -> None:
    """Assign NEAT fitness to each genome in ``genomes`` using the minimisation cost."""

    for _, genome in genomes:
        average_cost = evaluate_individual(
            genome,
            config,
            scenarios,
            env_cfg,
            boat_params,
            turn_cfg,
            params,
        )
        genome.fitness = -average_cost


@dataclass
class TrainingResult:
    """Return value from :func:`train_population`."""

    winner: neat.DefaultGenome
    config: neat.Config
    statistics: neat.StatisticsReporter


def train_population(
    config_path: Path,
    scenarios: Sequence[EncounterScenario],
    env_cfg: EnvConfig,
    boat_params: BoatParams,
    turn_cfg: TurnSessionConfig,
    params: HyperParameters,
    generations: int,
    seed: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 5,
) -> TrainingResult:
    """Run NEAT evolution configured for the COLREGs crossing experiments."""

    if seed is not None:
        random.seed(seed)

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(Path(config_path)),
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

    def _eval(genomes, neat_config):
        evaluate_population(genomes, neat_config, scenarios, env_cfg, boat_params, turn_cfg, params)

    winner = population.run(_eval, generations)

    return TrainingResult(winner=winner, config=neat_config, statistics=stats)


def build_scenarios(request: ScenarioRequest) -> List[EncounterScenario]:
    """Generate the fifteen deterministic scenarios (crossing, head-on, overtaking)."""

    return list(default_scenarios(request))


__all__ = [
    "EpisodeMetrics",
    "TrainingResult",
    "build_scenarios",
    "episode_cost",
    "evaluate_population",
    "simulate_episode",
    "train_population",
]

