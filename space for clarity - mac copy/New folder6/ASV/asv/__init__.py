"""Public interface for the simplified crossing scenario package."""
from .boat import Boat
from .config import BoatParams, EnvConfig, TurnSessionConfig
from .env import CrossingScenarioEnv
from .scenario import (
    STAND_ON_BEARINGS_DEG,
    CrossingScenario,
    ScenarioRequest,
    compute_crossing_geometry,
    iter_scenarios,
    raw_agent_inputs,
    scenario_states_for_env,
)

__all__ = [
    "Boat",
    "BoatParams",
    "EnvConfig",
    "TurnSessionConfig",
    "CrossingScenarioEnv",
    "STAND_ON_BEARINGS_DEG",
    "CrossingScenario",
    "ScenarioRequest",
    "compute_crossing_geometry",
    "iter_scenarios",
    "raw_agent_inputs",
    "scenario_states_for_env",
]
