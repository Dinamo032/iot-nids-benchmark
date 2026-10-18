import torch
from torch import nn


class CNN(nn.Module):
    """1D-CNN over the feature vector (features treated as a 1-channel signal)."""

    def __init__(self, n_features: int, hidden: int = 32, n_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1)).squeeze(-1)
        return self.head(h)
