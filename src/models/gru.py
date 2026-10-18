import torch
from torch import nn


class GRUNet(nn.Module):
    """GRU over the feature sequence (each feature is one time step)."""

    def __init__(self, n_features: int, hidden: int = 32, n_classes: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x.unsqueeze(-1))
        return self.head(h[-1])
