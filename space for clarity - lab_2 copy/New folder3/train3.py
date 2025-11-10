#!/usr/bin/env python3
"""
Double-DQN trainer for Multi-Boat Sectors Env v3 (chunked sessions + CPA shaping)
===============================================================================

What you get:
- Compatible with the v3 env you built: chunked turn sessions (with hysteresis + cancel toggle),
  signed action decoding, passthrough throttle (and optional hold-while-turning), CPA shaping +
  COLREGs nudge, substeps, max_steps w/ penalty.
- Replay warm-up gating (stable training).
- EMA-based "best so far" checkpoint.
- CSV logging of EVERY action for EVERY boat at EVERY step.

Usage (common examples):
  python train_v3.py --episodes 1500 --steps_per_episode 3000 --n_boats 2
  python train_v3.py --render --print_action_log --steps_per_episode 800

Tips:
- Keep --render off for speed; use it to debug.
- Align --steps_per_episode with EnvConfig.max_steps (this file does that for you).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import types
from collections import deque, namedtuple
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import List, Optional, Union, get_args, get_origin, get_type_hints

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# NOTE: change this import to match your env filename if needed.
# Example: if you saved the v3 env as "multi_boat_sectors_v3.py", keep this as-is.
from env3 import (BoatParams, EnvConfig, MultiBoatSectorsEnv, SpawnConfig,
                  TurnSessionConfig)


# ---------------- Hyperparameters (CLI-exposed via argparse) ----------------
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
    eps_decay_episodes: int = 3000      # reach ~eps_end after ~3000 episodes
    eps_decay: float = 0.0              # if 0, we compute it from the two lines above

    # i/o
    save_every: int = 50
    seed: int = 0
    render: bool = False

    # action logging
    log_actions: bool = True
    actions_filename: str = "actions_all.csv"
    print_action_log: bool = False   # per-step debug print (slow)

    # fleet
    n_boats: int | None = None

    # ---- v3 control & integration knobs ----
    turn_deg: float = 15.0
    turn_rate_degps: float = 45.0
    hysteresis_deg: float = 1.5
    allow_cancel: bool = False
    passthrough_throttle: bool = True
    hold_thr_while_turning: bool = False
    substeps: int = 1

    # ---- CPA shaping knobs (surface for sweeps) ----
    cpa_horizon: float = 60.0
    tcpa_decay: float = 20.0
    dcpa_scale: float = 120.0
    risk_weight: float = 0.10
    colregs_penalty: float = 0.05

    # ---- best-checkpoint via EMA of returns ----
    save_best: bool = True
    ema_beta: float = 0.98           # smoothing factor for EMA of returns
    best_filename: str = "q_best.pth"


# ---------------- Replay Buffer ----------------
Transition = namedtuple("Transition", ("state","action","reward","next_state","done"))

class Replay:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, bs: int) -> Transition:
        batch = random.sample(self.buf, bs)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buf)


# ---------------- Q-Network ----------------
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
        # He init (Kaiming) for ReLU layers, zero biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------- Agent (Double-DQN) ----------------
class Agent:
    def __init__(self, in_dim: int, n_actions: int, hp: HParams, device: torch.device):
        self.q = QNet(in_dim, n_actions).to(device)
        self.t = QNet(in_dim, n_actions).to(device)
        self.t.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=hp.lr)

        self.n_actions = n_actions
        self.gamma = hp.gamma
        self.device = device

        self.eps_end   = hp.eps_end
        # per-episode epsilon value + multiplicative decay
        self.eps_value = hp.eps_start
        self.eps_decay = (hp.eps_decay if hp.eps_decay > 0.0
                          else (hp.eps_end / hp.eps_start) ** (1.0 / max(1, hp.eps_decay_episodes)))
        self.global_step = 0  # still used for target updates etc.

    def epsilon(self) -> float:
        # per-episode schedule: no within-episode decay
        return self.eps_value

    def decay_epsilon_episode(self):
        self.eps_value = max(self.eps_end, self.eps_value * self.eps_decay)

    def act(self, state: np.ndarray) -> tuple[int, bool, float]:
        eps = self.epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions), True, eps
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            qv = self.q(s)[0]
            return int(torch.argmax(qv).item()), False, eps

    def update(self, replay: Replay, batch_size: int):
        # (we also guard updates externally with min_replay)
        if len(replay) < max(batch_size, 1):
            return None

        batch = replay.sample(batch_size)
        s  = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        a  = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        r  = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        d  = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            next_best = torch.argmax(self.q(ns), dim=1, keepdim=True)  # action selection by online net
            t_q = self.t(ns).gather(1, next_best)                      # evaluation by target net
            target = r + (1.0 - d) * self.gamma * t_q

        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optim.step()
        return float(loss.item())

    def hard_update(self):
        self.t.load_state_dict(self.q.state_dict())


# ---------------- Helpers ----------------
def decode_action_idx(a: int) -> tuple[int, int]:
    """
    Returns (steer, throttle) with:
      steer ∈ {0: straight, 1: right, 2: left}
      throttle ∈ {0: coast,   1: accel, 2: decel}
    Only used for logging readability.
    """
    return a // 3, a % 3


# ---------------- Training Loop ----------------
def train(hp: HParams):
    # seeds
    random.seed(hp.seed); np.random.seed(hp.seed); torch.manual_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env config (align steps_per_episode with env.max_steps for clarity)
    cfg = EnvConfig(
        render=hp.render,
        seed=hp.seed,
        substeps=hp.substeps,
        max_steps=hp.steps_per_episode,
        # CPA shaping knobs (surfaced for sweeps)
        cpa_horizon=hp.cpa_horizon,
        tcpa_decay=hp.tcpa_decay,
        dcpa_scale=hp.dcpa_scale,
        risk_weight=hp.risk_weight,
        colregs_penalty=hp.colregs_penalty,
    )
    kin = BoatParams()
    tcfg = TurnSessionConfig(
        turn_deg=hp.turn_deg,
        turn_rate_degps=hp.turn_rate_degps,   # v3 rename (was yaw_rate_degps)
        hysteresis_deg=hp.hysteresis_deg,
        allow_cancel=hp.allow_cancel,
        passthrough_throttle=hp.passthrough_throttle,
        hold_throttle_while_turning=hp.hold_thr_while_turning,
    )
    default_nb = SpawnConfig().n_boats #Just a record of what the default no. of ships in the environment it
    spawn = SpawnConfig(
        n_boats=(hp.n_boats if hp.n_boats is not None else default_nb),
        start_speed=0.0
        )
    env = MultiBoatSectorsEnv(cfg, kin, tcfg, spawn)

    replay = Replay(hp.replay_size)
    agent  = Agent(in_dim=100, n_actions=9, hp=hp, device=device)

    # outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", ts); os.makedirs(out_dir, exist_ok=True)

    writer = None; fhandle = None
    if hp.log_actions:
        path = os.path.join(out_dir, hp.actions_filename)
        fhandle = open(path, "w", newline="")
        writer = csv.writer(fhandle)
        writer.writerow([
            "episode","step","boat_id",
            "action","steer","throttle","was_random","epsilon",
            "reward",
            "ship_x","ship_y","ship_speed","ship_heading_deg",
            "goal_x","goal_y","goal_dist","reached","reason"
        ])

    returns: List[float] = []
    ema_loss = None
    ema_ret: Optional[float] = None
    best_metric = float("-inf")

    for ep in range(iceil(1), hp.episodes + 1) if False else range(1, hp.episodes + 1):
        # reshuffle starts per episode; keep seed stable-ish
        obs_list = env.reset(seed=hp.seed + ep)
        num_boats = len(env.ships)
        ep_ret = 0.0
        reason = ""
        steps  = 0

        for t in range(hp.steps_per_episode):
            if hp.render:
                # optional HUD overlay in v3
                env.set_overlay({
                    "episode": ep,
                    "epsilon": agent.epsilon(),
                    "global_steps": agent.global_step,
                    "loss_ema": ema_loss,
                })
                env.render()

            # epsilon-greedy actions for each boat
            actions, was_rand, eps_used = [], [], []
            for obs in obs_list:
                a, rnd, eps = agent.act(obs)
                actions.append(a); was_rand.append(rnd); eps_used.append(eps)

            next_obs_list, rewards, done, info = env.step(actions)
            reason = info.get("reason","")

            # push N transitions + (optional) learn
            for i in range(len(env.ships)):
                agent.global_step += 1
                replay.push(obs_list[i], actions[i], rewards[i], next_obs_list[i], float(done))
                if len(replay) >= hp.min_replay:
                    loss = agent.update(replay, hp.batch_size)
                    if loss is not None:
                        ema_loss = loss if ema_loss is None else (0.99 * ema_loss + 0.01 * loss)

            ep_ret += float(sum(rewards))
            steps += 1

            # target sync
            if agent.global_step % hp.target_update == 0:
                agent.hard_update()

            # logging per boat
            if writer is not None:
                for i in range(len(env.ships)):
                    sh = env.ships[i]; gx, gy = env.goals[i]
                    dist = math.hypot(gx - sh.x, gy - sh.y)
                    s, th = decode_action_idx(actions[i])
                    writer.writerow([
                        ep, steps, i, actions[i], s, th, bool(was_rand[i]), float(eps_used[i]), float(rewards[i]),
                        float(sh.x), float(sh.y), float(sh.u), float(sh.h*180.0/math.pi),
                        float(gx), float(gy), float(dist), bool(env.reached[i]), reason
                    ])
                    if hp.print_action_log:
                        print(f"EP{ep} STEP{steps} B{i} A={actions[i]} "
                              f"rnd={was_rand[i]} eps={eps_used[i]:.3f} r={rewards[i]:+.3f}")
            obs_list = next_obs_list

            if done:
                if hp.render: env.render()
                break

            if fhandle is not None and (steps % 200 == 0):
                fhandle.flush()

        # episode accounting
        returns.append(ep_ret)
        # EMA of returns (best-checkpoint metric)
        if ema_ret is None:
            ema_ret = ep_ret
        else:
            ema_ret = hp.ema_beta * ema_ret + (1.0 - hp.ema_beta) * ep_ret

        loss_str = f"{ema_loss:.5f}" if isinstance(ema_loss, (int, float)) else "-"

        print(f"Ep {ep:04d} | steps {steps:4d} | return {ep_ret:8.3f} | ema_ret {ema_ret:8.3f} | "
              f"eps {agent.epsilon():.3f} | replay {len(replay):6d} | loss_ema {loss_str} | reason={reason}")

        # periodic snapshots
        if ep % hp.save_every == 0 or ep == hp.episodes:
            torch.save(agent.q.state_dict(), os.path.join(out_dir, f"q_ep{ep}.pth"))
            np.save(os.path.join(out_dir, "returns.npy"), np.array(returns, dtype=np.float32))
            if fhandle is not None: fhandle.flush()

        # decay epsilon ONCE per episode
        agent.decay_epsilon_episode()

        # best-so-far checkpoint (using EMA)
        if hp.save_best and ema_ret > best_metric:
            best_metric = ema_ret
            torch.save(agent.q.state_dict(), os.path.join(out_dir, hp.best_filename))

    if fhandle is not None:
        fhandle.close()
    print("Training finished. Results in:", out_dir)
    if hp.log_actions:
        print("Action log:", os.path.join(out_dir, hp.actions_filename))


def _cli_type_from_annotation(ftype, default):
    # Resolve Optional[T] (Union[T, NoneType]) and the 3.10 "T | None" form
    origin = get_origin(ftype)
    if origin in (Union, getattr(types, "UnionType", None)):
        args = [a for a in get_args(ftype)]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            ftype = non_none[0]  # Optional[T] -> T

    # Map common primitives; fall back to type(default) or str
    if ftype in (int, float, str, bool):
        return ftype
    if default is not None:
        return type(default)
    return str


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    defaults = HParams()
    # Resolve forward-referenced (string) annotations to real types
    hints = get_type_hints(HParams)

    for f in fields(HParams):
        name = f.name
        default = getattr(defaults, name)
        anno = hints.get(name, f.type)  # use resolved type if available

        # ---- Handle booleans as flags ----
        if anno is bool:
            if default:  # default True -> provide --no-flag to turn off
                parser.add_argument(f"--no-{name}", dest=name, action="store_false")
            else:        # default False -> provide --flag to turn on
                parser.add_argument(f"--{name}", dest=name, action="store_true")
            continue

        # ---- Unwrap Optional[T] / Union[T, None] ----
        origin = get_origin(anno)
        is_union = origin in (Union, getattr(types, "UnionType", None))
        if is_union:
            args = [a for a in get_args(anno) if a is not type(None)]
            if len(args) == 1:
                anno = args[0]  # Optional[T] -> T

        # ---- Decide argparse 'type' for the remaining primitives ----
        if anno in (int, float, str):
            parser.add_argument(f"--{name}", type=anno, default=default)
        else:
            # Fallback: use type(default) if present; else accept string
            atype = type(default) if default is not None else str
            # If the fallback resolves to bool here, it means annotation resolution failed;
            # still treat non-flag booleans as explicit values (rare in your HParams).
            parser.add_argument(f"--{name}", type=atype, default=default)

    args = parser.parse_args()
    hp = HParams(**vars(args))
    train(hp)