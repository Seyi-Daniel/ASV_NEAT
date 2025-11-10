from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class SpawnConfig:
    # Boats & initial state
    n_boats: int = 2
    start_speed: float = 0.0

    # Placement constraints
    margin: float = 80.0                 # keep initial spawns this far from world borders
    min_sep_factor: float = 2.0          # separation ≥ min_sep_factor × collision_radius()

    # Goals (per-scenario knobs)
    goal_ahead_distance: float = 450.0   # place goal this far along the initial heading
    goal_radius: float = 10.0            # arrival radius (m)
    goal_edge_clearance: float = 20.0    # clamp goals away from edges by this many meters

    # Robustness
    max_spawn_attempts: int = 1000       # bound the rejection sampling

class EnvConfig:
    # World & time
    world_w: float = 100.0
    world_h: float = 100.0
    dt: float = 0.05
    substeps: int = 1
    sensor_range: Optional[float] = None
    seed: Optional[int] = None

    # Rendering
    render: bool = False
    pixels_per_meter: float = 10.0
    show_grid: bool = True
    show_sectors: bool = False
    show_trails: bool = True
    show_hud: bool = True

    # Reset / borders
    border_margin: float = 10.0
    collision_margin: float = 0.0

    # Step limits
    max_steps: int = 3000

    # Reward shaping
    goal_reward: float = 1.0
    max_steps_penalty: float = -0.1
    oob_penalty: float = -1.0
    collision_penalty: float = -2.0

    living_penalty: float = -0.01
    goal_bonus: float = 10

    # Progress shaping (per meter toward goal)
    progress_weight: float = 0.002

    # CPA shaping knobs
    cpa_horizon: float = 60.0
    tcpa_decay: float = 20.0
    dcpa_scale: float = 120.0
    risk_weight: float = 0.10
    colregs_penalty: float = 0.05  # extra nudge against porting into starboard-forward contact