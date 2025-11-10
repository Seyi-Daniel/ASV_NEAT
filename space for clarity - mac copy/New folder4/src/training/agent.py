from __future__ import annotations
import random
import numpy as np
import torch
import torch.optim as optim

from .hparams import HParams
from .networks import QNet

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