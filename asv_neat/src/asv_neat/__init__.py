"""Convenience exports for the NEAT-controlled COLREGs crossing project."""

from .boat import Boat
from .config import BoatParams, EnvConfig, TurnSessionConfig
from .env import CrossingScenarioEnv
from .hyperparameters import HyperParameters, apply_cli_overrides
from .neat_training import (
    EpisodeMetrics,
    TrainingResult,
    build_scenarios,
    episode_cost,
    evaluate_population,
    simulate_episode,
    train_population,
)
from .scenario import (
    STAND_ON_BEARINGS_DEG,
    CrossingScenario,
    ScenarioRequest,
    compute_crossing_geometry,
    iter_scenarios,
    scenario_states_for_env,
)

__all__ = [
    "Boat",
    "BoatParams",
    "EnvConfig",
    "TurnSessionConfig",
    "CrossingScenarioEnv",
    "HyperParameters",
    "apply_cli_overrides",
    "EpisodeMetrics",
    "TrainingResult",
    "build_scenarios",
    "episode_cost",
    "evaluate_population",
    "simulate_episode",
    "train_population",
    "STAND_ON_BEARINGS_DEG",
    "CrossingScenario",
    "ScenarioRequest",
    "compute_crossing_geometry",
    "iter_scenarios",
    "scenario_states_for_env",
]
