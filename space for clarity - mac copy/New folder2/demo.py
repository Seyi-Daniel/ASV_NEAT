#!/usr/bin/env python3
"""
Demo runner for the Multi-Boat Sectors Env v3 using a saved Double-DQN Q-network.

Examples:
  python demo_v3.py --checkpoint results/20250101_120000/q_best.pth --episodes 3 --render
  python demo_v3.py --checkpoint results/20250101_120000/q_ep1500.pth --steps_per_episode 3000
  python demo_v3.py --checkpoint results/20250101_120000/q_best.pth --print_action_log --log_actions

Notes:
- By default actions are taken greedily (epsilon = 0) — no exploration.
- If your env uses a different observation length or action count, pass --in_dim / --n_actions.
- Env knobs are aligned with your training script defaults so behavior matches training.
"""

from __future__ import annotations
import argparse, csv, math, os, sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

# Use your v3 env module. Change the import name if your file differs.
from env3 import (
    MultiBoatSectorsEnv, EnvConfig, BoatParams, TurnSessionConfig, SpawnConfig
)

# --------------- Model (same arch & init as training) ---------------
class QNet(nn.Module):
    def __init__(self, in_dim: int = 100, n_actions: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256),    nn.ReLU(inplace=True),
            nn.Linear(256, 128),    nn.ReLU(inplace=True),
            nn.Linear(128, 128),    nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------- Utils ---------------
def decode_action_idx(a: int) -> tuple[int, int]:
    """(steer, throttle) with steer: 0=straight,1=right,2=left; throttle: 0=coast,1=accel,2=decel"""
    return a // 3, a % 3

def infer_env_dims(env: MultiBoatSectorsEnv) -> tuple[int, Optional[int]]:
    """Returns (in_dim, n_actions?). Falls back to None if cannot infer n_actions."""
    obs_list = env.reset()  # type: ignore
    in_dim = int(len(obs_list[0]))
    n_actions = None
    # Try common attributes in case your env exposes them
    for cand in ("n_actions",):
        if hasattr(env, cand):
            try:
                n_actions = int(getattr(env, cand))
                break
            except Exception:
                pass
    # Gym-like fallback
    if n_actions is None and hasattr(env, "action_space"):
        try:
            n_actions = int(env.action_space.n)  # type: ignore
        except Exception:
            pass
    return in_dim, n_actions

def maybe_set_overlay(env, payload: dict):
    """Don't crash if overlay isn't implemented."""
    try:
        if hasattr(env, "set_overlay"):
            env.set_overlay(payload)
    except Exception:
        pass

# --------------- Demo args aligned with your training defaults ---------------
@dataclass
class DemoArgs:
    episodes: int = 3
    steps_per_episode: int = 3000  # align with EnvConfig.max_steps
    n_boats: int = 2
    seed: int = 0
    render: bool = False
    print_action_log: bool = False
    log_actions: bool = False
    actions_filename: str = "demo_actions.csv"

    # v3 control & CPA shaping (match training defaults)
    turn_deg: float = 15.0
    turn_rate_degps: float = 45.0
    hysteresis_deg: float = 1.5
    allow_cancel: bool = False
    passthrough_throttle: bool = True
    hold_thr_while_turning: bool = False
    substeps: int = 1
    cpa_horizon: float = 60.0
    tcpa_decay: float = 20.0
    dcpa_scale: float = 120.0
    risk_weight: float = 0.10
    colregs_penalty: float = 0.05

# --------------- Main demo runner ---------------
def run_demo(checkpoint: str,
             in_dim_opt: Optional[int],
             n_actions_opt: Optional[int],
             args: DemoArgs):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build env with your v3 knobs
    cfg = EnvConfig(
        render=args.render,
        seed=args.seed,
        substeps=args.substeps,
        max_steps=args.steps_per_episode,
        cpa_horizon=args.cpa_horizon,
        tcpa_decay=args.tcpa_decay,
        dcpa_scale=args.dcpa_scale,
        risk_weight=args.risk_weight,
        colregs_penalty=args.colregs_penalty,
    )
    kin = BoatParams()
    tcfg = TurnSessionConfig(
        turn_deg=args.turn_deg,
        turn_rate_degps=args.turn_rate_degps,
        hysteresis_deg=args.hysteresis_deg,
        allow_cancel=args.allow_cancel,
        passthrough_throttle=args.passthrough_throttle,
        hold_throttle_while_turning=args.hold_thr_while_turning,
    )
    spawn = SpawnConfig(n_boats=args.n_boats, start_speed=0.0)
    env = MultiBoatSectorsEnv(cfg, kin, tcfg, spawn)

    # Infer input/action sizes unless overridden
    inferred_in_dim, inferred_n_actions = infer_env_dims(env)
    in_dim = in_dim_opt if in_dim_opt is not None else inferred_in_dim
    n_actions = n_actions_opt if n_actions_opt is not None else (inferred_n_actions or 9)

    # Model
    q = QNet(in_dim=in_dim, n_actions=n_actions).to(device)
    try:
        state = torch.load(checkpoint, map_location=device)
        # If checkpoint was saved as state_dict() of QNet only, this will work.
        # If you saved a dict with extra keys, try common patterns:
        if isinstance(state, dict) and not any(k.startswith("net.") or k.endswith(".weight") for k in state.keys()):
            # Allow { "model": state_dict, ... } style
            for key in ("model", "state_dict", "q_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        q.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"[!] Failed to load checkpoint '{checkpoint}': {e}", file=sys.stderr)
        print("    Tip: ensure in_dim and n_actions match training (defaults 100 & 9).", file=sys.stderr)
        sys.exit(1)
    q.eval()

    # Optional CSV logging
    writer = None
    fhandle = None
    if args.log_actions:
        out_dir = os.path.dirname(os.path.abspath(checkpoint)) or "."
        path = os.path.join(out_dir, args.actions_filename)
        fhandle = open(path, "w", newline="")
        writer = csv.writer(fhandle)
        writer.writerow([
            "episode","step","boat_id",
            "action","steer","throttle",
            "reward",
            "ship_x","ship_y","ship_speed","ship_heading_deg",
            "goal_x","goal_y","goal_dist","reached","reason"
        ])

    returns: List[float] = []

    for ep in range(1, args.episodes + 1):
        obs_list = env.reset(seed=args.seed + ep)  # reshuffle per-episode start
        ep_ret = 0.0
        reason = ""
        steps = 0

        while steps < args.steps_per_episode:
            if args.render:
                maybe_set_overlay(env, {
                    "mode": "demo",
                    "episode": ep,
                    "steps": steps,
                    "return_so_far": ep_ret,
                })
                env.render()

            actions = []
            with torch.no_grad():
                for obs in obs_list:
                    s = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(device)
                    qv = q(s)[0]
                    a = int(torch.argmax(qv).item())
                    actions.append(a)

            next_obs_list, rewards, done, info = env.step(actions)
            reason = info.get("reason", "")
            steps += 1
            ep_ret += float(sum(rewards))

            # Optional console + CSV logging
            if args.print_action_log or writer is not None:
                for i in range(args.n_boats):
                    sh = env.ships[i]
                    gx, gy = env.goals[i]
                    dist = math.hypot(gx - sh.x, gy - sh.y)
                    s_dec, th_dec = decode_action_idx(actions[i])
                    if args.print_action_log:
                        print(f"EP{ep} STEP{steps} B{i} A={actions[i]} "
                              f"(steer={s_dec},thr={th_dec}) r={rewards[i]:+.3f}")
                    if writer is not None:
                        writer.writerow([
                            ep, steps, i,
                            actions[i], s_dec, th_dec,
                            float(rewards[i]),
                            float(sh.x), float(sh.y), float(sh.u), float(sh.h*180.0/math.pi),
                            float(gx), float(gy), float(dist), bool(env.reached[i]), reason
                        ])

            obs_list = next_obs_list
            if done:
                if args.render:
                    env.render()
                break

        returns.append(ep_ret)
        print(f"Demo Ep {ep:04d} | steps {steps:4d} | return {ep_ret:8.3f} | reason={reason}")

        if fhandle is not None and (steps % 200 == 0):
            fhandle.flush()

    if fhandle is not None:
        fhandle.close()
        print(f"Action log saved to: {os.path.join(os.path.dirname(checkpoint), args.actions_filename)}")

    print("Returns:", ", ".join(f"{r:.1f}" for r in returns))
    print(f"Avg return over {len(returns)} ep(s): {np.mean(returns):.3f}")

# --------------- CLI ---------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved .pth (e.g., results/.../q_best.pth)")
    parser.add_argument("--episodes", type=int, default=DemoArgs.episodes)
    parser.add_argument("--steps_per_episode", type=int, default=DemoArgs.steps_per_episode)
    parser.add_argument("--n_boats", type=int, default=DemoArgs.n_boats)
    parser.add_argument("--seed", type=int, default=DemoArgs.seed)
    parser.add_argument("--render", action="store_true", default=DemoArgs.render)
    parser.add_argument("--print_action_log", action="store_true", default=DemoArgs.print_action_log)
    parser.add_argument("--log_actions", action="store_true", default=DemoArgs.log_actions)
    parser.add_argument("--actions_filename", type=str, default=DemoArgs.actions_filename)

    # Optional overrides if your env doesn't reveal dims:
    parser.add_argument("--in_dim", type=int, default=None,
                        help="Obs length override (defaults to env inference).")
    parser.add_argument("--n_actions", type=int, default=None,
                        help="Action count override (defaults to env inference or 9).")

    # Control & shaping knobs (kept for parity with training defaults)
    parser.add_argument("--turn_deg", type=float, default=DemoArgs.turn_deg)
    parser.add_argument("--turn_rate_degps", type=float, default=DemoArgs.turn_rate_degps)
    parser.add_argument("--hysteresis_deg", type=float, default=DemoArgs.hysteresis_deg)
    parser.add_argument("--allow_cancel", action="store_true", default=DemoArgs.allow_cancel)
    parser.add_argument("--passthrough_throttle", action="store_true", default=DemoArgs.passthrough_throttle)
    parser.add_argument("--hold_thr_while_turning", action="store_true", default=DemoArgs.hold_thr_while_turning)
    parser.add_argument("--substeps", type=int, default=DemoArgs.substeps)
    parser.add_argument("--cpa_horizon", type=float, default=DemoArgs.cpa_horizon)
    parser.add_argument("--tcpa_decay", type=float, default=DemoArgs.tcpa_decay)
    parser.add_argument("--dcpa_scale", type=float, default=DemoArgs.dcpa_scale)
    parser.add_argument("--risk_weight", type=float, default=DemoArgs.risk_weight)
    parser.add_argument("--colregs_penalty", type=float, default=DemoArgs.colregs_penalty)

    args_ns = parser.parse_args()
    demo_args = DemoArgs(
        episodes=args_ns.episodes,
        steps_per_episode=args_ns.steps_per_episode,
        n_boats=args_ns.n_boats,
        seed=args_ns.seed,
        render=args_ns.render,
        print_action_log=args_ns.print_action_log,
        log_actions=args_ns.log_actions,
        actions_filename=args_ns.actions_filename,
        turn_deg=args_ns.turn_deg,
        turn_rate_degps=args_ns.turn_rate_degps,
        hysteresis_deg=args_ns.hysteresis_deg,
        allow_cancel=args_ns.allow_cancel,
        passthrough_throttle=args_ns.passthrough_throttle,
        hold_thr_while_turning=args_ns.hold_thr_while_turning,
        substeps=args_ns.substeps,
        cpa_horizon=args_ns.cpa_horizon,
        tcpa_decay=args_ns.tcpa_decay,
        dcpa_scale=args_ns.dcpa_scale,
        risk_weight=args_ns.risk_weight,
        colregs_penalty=args_ns.colregs_penalty,
    )

    run_demo(
        checkpoint=args_ns.checkpoint,
        in_dim_opt=args_ns.in_dim,
        n_actions_opt=args_ns.n_actions,
        args=demo_args,
    )

if __name__ == "__main__":
    main()
