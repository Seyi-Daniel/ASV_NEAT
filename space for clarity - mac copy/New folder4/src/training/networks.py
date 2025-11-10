from __future__ import annotations
import torch
import torch.nn as nn

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