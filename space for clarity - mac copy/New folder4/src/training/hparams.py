from __future__ import annotations
from dataclasses import dataclass

@dataclass
class HParams:
    # training schedule
    episodes: int = 1500
    steps_per_episode: int = 3000   # align with EnvConfig.max_steps
    gamma: float = 0.995

    # optimization
    lr: float = 2e-4
    batch_size: int = 256
    replay_size: int = 200_000
    min_replay: int = 10_000        # warm-up threshold before updates
    target_update: int = 4000       # sync online -> target every N transitions

    # epsilon-greedy (linear decay over global steps)
    eps_start: float = 1.0
    eps_end: float = 0.10               # your requested minimum
    eps_decay: float = 0.0              # if 0, use eps_decay_episodes instead
    eps_decay_episodes: int = 250       # multiplicative decay episodes -> eps_end

    # logging & checkpoints
    out_dir: str = "runs/latest"
    save_every: int = 10000
    save_best: bool = True
    ema_beta: float = 0.98
    best_filename: str = "q_best.pth"

    # env/render
    render: bool = False
    print_action_log: bool = False
    log_actions: bool = True
    actions_filename: str = "actions_all.csv"

    # fleet
    n_boats: int | None = None

    # v3 control & integration knobs
    turn_deg: float = 15.0
    turn_rate_degps: float = 45.0
    hysteresis_deg: float = 1.5
    allow_cancel: bool = False
    passthrough_throttle: bool = True
    hold_thr_while_turning: bool = False
    substeps: int = 1

    goal_reward: float = 1.0
    progress_weight: float = 0.002

    # CPA shaping knobs
    cpa_horizon: float = 60.0
    tcpa_decay: float = 20.0
    dcpa_scale: float = 120.0
    risk_weight: float = 0.10
    colregs_penalty: float = 0.05