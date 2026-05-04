import torch
import torch.nn as nn


class PropertyPredictor(nn.Module):
    def __init__(self, hidden_dim=196):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        # z: (batch, hidden_dim) -> (batch, 1)
        return self.net(z).squeeze(-1)
