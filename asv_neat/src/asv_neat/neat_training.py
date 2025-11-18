"""NEAT integration for the deterministic COLREGs crossing scenarios."""
from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import neat

from .boat import Boat
from .config import BoatParams, EnvConfig, TurnSessionConfig
from .env import CrossingScenarioEnv
from .hyperparameters import HyperParameters
from .scenario import (
    EncounterScenario,
    ScenarioKind,
    ScenarioRequest,
    default_scenarios,
    scenario_states_for_env,
)
from .utils import euclidean_distance, goal_distance, relative_bearing_deg, tcpa_dcpa


@dataclass
class EpisodeMetrics:
    """Summary statistics produced by a single simulated episode."""

    steps: int
    reached_goal: bool
    collided: bool
    final_distance: float
    min_separation: float
    wrong_action_cost: float


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
    steps = 0

    for step_idx in range(params.max_steps):
        snapshot = env.snapshot()
        if not snapshot:
            break

        agent_state = snapshot[0]
        stand_on_state = snapshot[1] if len(snapshot) > 1 else snapshot[0]

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


def build_scenarios(
    request: ScenarioRequest, *, kinds: Sequence[ScenarioKind] | None = None
) -> List[EncounterScenario]:
    """Generate the deterministic scenarios for the requested encounter types."""

    if kinds is None:
        return list(default_scenarios(request))
    return list(default_scenarios(request, kinds=kinds))


def summarise_genome(
    genome,
    config,
    scenarios: Iterable[EncounterScenario],
    params: HyperParameters,
    boat_params: BoatParams,
    turn_cfg: TurnSessionConfig,
    env_cfg: EnvConfig,
    *,
    render: bool = False,
) -> None:
    """Replay ``genome`` across ``scenarios`` and print per-episode metrics."""

    scenario_list = list(scenarios)
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    print("\nWinner evaluation summary:")

    def _scenario_prefix(idx: int, scenario: EncounterScenario) -> str:
        frame = "stand-on" if scenario.bearing_frame == "agent" else "agent (stand-on frame)"
        return (
            f"Scenario {idx} [{scenario.kind.value}, "
            f"bearing {scenario.requested_bearing:6.2f}° ({frame})]"
        )

    if render:
        env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params, tcfg=turn_cfg)
        try:
            env.enable_render()
            total_cost = 0.0
            for idx, scenario in enumerate(scenario_list, start=1):
                metrics = simulate_episode(env, scenario, network, params, render=True)
                cost = episode_cost(metrics, params)
                total_cost += cost
                status = (
                    "goal"
                    if metrics.reached_goal
                    else "collision" if metrics.collided else "timeout"
                )
                prefix = _scenario_prefix(idx, scenario)
                print(
                    f"  {prefix}: steps={metrics.steps:4d} status={status:8s} "
                    f"min_sep={metrics.min_separation:6.2f}m colregs={metrics.wrong_action_cost:6.2f} "
                    f"cost={cost:7.2f}"
                )
            if scenario_list:
                print(f"Average cost: {total_cost / len(scenario_list):.2f}")
        finally:
            env.close()
        return

    def _evaluate(idx_scenario: int, scenario: EncounterScenario):
        local_env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params, tcfg=turn_cfg)
        try:
            metrics = simulate_episode(local_env, scenario, network, params, render=False)
        finally:
            local_env.close()
        cost = episode_cost(metrics, params)
        status = (
            "goal"
            if metrics.reached_goal
            else "collision" if metrics.collided else "timeout"
        )
        return idx_scenario, metrics, cost, status

    results = []
    with ThreadPoolExecutor(max_workers=max(1, len(scenario_list))) as executor:
        futures = [
            executor.submit(_evaluate, idx, scenario)
            for idx, scenario in enumerate(scenario_list, start=1)
        ]
        for fut in futures:
            results.append(fut.result())

    results.sort(key=lambda item: item[0])
    total_cost = 0.0
    for idx, metrics, cost, status in results:
        total_cost += cost
        scenario = scenario_list[idx - 1]
        prefix = _scenario_prefix(idx, scenario)
        print(
            f"  {prefix}: steps={metrics.steps:4d} status={status:8s} "
            f"min_sep={metrics.min_separation:6.2f}m colregs={metrics.wrong_action_cost:6.2f} "
            f"cost={cost:7.2f}"
        )
    if results:
        print(f"Average cost: {total_cost / len(results):.2f}")


__all__ = [
    "EpisodeMetrics",
    "TrainingResult",
    "build_scenarios",
    "episode_cost",
    "evaluate_population",
    "summarise_genome",
    "simulate_episode",
    "train_population",
]

