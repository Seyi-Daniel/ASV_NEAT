from __future__ import annotations
import argparse, csv, os, math
from datetime import datetime
from dataclasses import asdict, fields
from typing import get_args, get_origin, get_type_hints, Union

import numpy as np
import torch

from sectorsim.env import MultiBoatSectorsEnv
from sectorsim.entities import BoatParams
from sectorsim.control import TurnSessionConfig
from sectorsim.env_config import EnvConfig, SpawnConfig

from .hparams import HParams
from .replay import Replay, Transition
from .agent import Agent
from .actions import decode_action_idx

def train(hp: HParams):
    # === directories ===
    os.makedirs(hp.out_dir, exist_ok=True)

    # === device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === env ===
    n_boats = hp.n_boats if hp.n_boats is not None else 2
    cfg  = EnvConfig()
    cfg.render = hp.render
    cfg.max_steps = hp.steps_per_episode
    cfg.substeps  = hp.substeps
    cfg.cpa_horizon   = hp.cpa_horizon
    cfg.tcpa_decay    = hp.tcpa_decay
    cfg.dcpa_scale    = hp.dcpa_scale
    cfg.risk_weight   = hp.risk_weight
    cfg.colregs_penalty = hp.colregs_penalty
    cfg.progress_weight = hp.progress_weight 

    kin  = BoatParams()
    tcfg = TurnSessionConfig(
        turn_deg=hp.turn_deg,
        turn_rate_degps=hp.turn_rate_degps,
        allow_cancel=hp.allow_cancel,
        hysteresis_deg=hp.hysteresis_deg,
        passthrough_throttle=hp.passthrough_throttle,
        hold_throttle_while_turning=hp.hold_thr_while_turning,
    )
    spawn = SpawnConfig(n_boats=n_boats)

    env = MultiBoatSectorsEnv(cfg=cfg, kin=kin, tcfg=tcfg, spawn=spawn)

    obs_dim = 100  # your env observation size per boat
    n_actions = 9  # (3 steer) x (3 throttle)

    # === agent & replay ===
    agent = Agent(obs_dim, n_actions, hp, device)
    replay = Replay(hp.replay_size)

    # === logging ===
    log_fp = None
    writer = None
    if hp.log_actions:
        log_path = os.path.join(hp.out_dir, hp.actions_filename)
        log_fp = open(log_path, "w", newline="")
        writer = csv.writer(log_fp)
        writer.writerow(["episode","step","boat","action","steer","thr","reward","eps"])

    # === training ===
    best_ema = None
    returns_ema = None
    for ep in range(hp.episodes):
        obs_all = env.reset()
        ep_returns = np.zeros(n_boats, dtype=np.float32)

        for t in range(hp.steps_per_episode):
            actions = []
            for i in range(n_boats):
                a, rand, eps = agent.act(np.array(obs_all[i], dtype=np.float32))
                actions.append(a)

            next_obs_all, rewards, done, info = env.step(actions)

            for i in range(n_boats):
                s  = np.array(obs_all[i], dtype=np.float32)
                ns = np.array(next_obs_all[i], dtype=np.float32)
                r  = rewards[i]
                d  = done
                replay.push(s, actions[i], r, ns, d)
                ep_returns[i] += r

                if writer is not None:
                    st, th = decode_action_idx(actions[i])
                    writer.writerow([ep, t, i, actions[i], st, th, r, eps])

            obs_all = next_obs_all

            # Update Q network if warm-up complete
            if len(replay) >= hp.min_replay:
                batch = replay.sample(hp.batch_size)
                s  = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=device)
                a  = torch.tensor(batch.action, dtype=torch.int64, device=device)
                r  = torch.tensor(batch.reward, dtype=torch.float32, device=device)
                ns = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=device)
                d  = torch.tensor(batch.done, dtype=torch.float32, device=device)

                q_sa = agent.q(s).gather(1, a.view(-1,1)).squeeze(1)
                with torch.no_grad():
                    a_star = agent.q(ns).argmax(dim=1)
                    tgt = r + hp.gamma * (1.0 - d) * agent.t(ns).gather(1, a_star.view(-1,1)).squeeze(1)
                loss = torch.nn.functional.smooth_l1_loss(q_sa, tgt)

                agent.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.q.parameters(), 5.0)
                agent.optim.step()

                # target update periodically
                if agent.global_step % hp.target_update == 0:
                    agent.t.load_state_dict(agent.q.state_dict())
                agent.global_step += 1

            if done:
                break

        # episode end
        agent.decay_epsilon_episode()

        # keep simple EMA of returns across boats
        ep_ret = float(np.mean(ep_returns))
        if returns_ema is None:
            returns_ema = ep_ret
        else:
            returns_ema = hp.ema_beta * returns_ema + (1 - hp.ema_beta) * ep_ret

        # save best
        if hp.save_best:
            if (best_ema is None) or (returns_ema > best_ema):
                best_ema = returns_ema
                torch.save(agent.q.state_dict(), os.path.join(hp.out_dir, hp.best_filename))

    if writer is not None:
        log_fp.close()

def _cli_type_from_annotation(anno):
    """
    Convert a type annotation into an argparse-compatible callable.
    - Optional[X] / X|None -> returns type for X
    - List[X] -> returns type for X (argparse will parse a single value; extend if you later add list CLI)
    - Fallback: str
    """
    origin = get_origin(anno)
    if origin is None:
        # Plain types like int, float, str are fine as callables
        return anno

    # Handle Optional/Union (including PEP-604 X | Y)
    args = get_args(anno)
    if args:
        # If it's Optional[X] or Union[X, NoneType], pick the non-None arg's type
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if non_none:
            return _cli_type_from_annotation(non_none[0])  # recurse in case it's nested

    # Handle List[X]
    if origin in (list,):
        inner = get_args(anno)
        return _cli_type_from_annotation(inner[0]) if inner else str

    # Fallback
    return str