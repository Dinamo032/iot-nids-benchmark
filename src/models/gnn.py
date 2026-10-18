import torch
from torch import nn


class SimpleGNN(nn.Module):
    """Skeleton GNN: flows in a batch form a kNN graph (dense, pure PyTorch).

    Placeholder for the final graph model (e.g. GraphSAGE via PyTorch Geometric,
    to be added with pinned versions before Week 7). Keeps the same interface as CNN/GRU.
    """

    def __init__(self, n_features: int, hidden: int = 32, n_classes: int = 2, k: int = 3):
        super().__init__()
        self.k = k
        self.lin1 = nn.Linear(n_features, hidden)
        self.lin2 = nn.Linear(hidden, n_classes)

    def _adjacency(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        k = min(self.k, n - 1) if n > 1 else 0
        adj = torch.eye(n, device=x.device)
        if k > 0:
            dist = torch.cdist(x, x)
            idx = dist.topk(k + 1, largest=False).indices  # includes self
            adj.scatter_(1, idx, 1.0)
        return adj / adj.sum(1, keepdim=True)  # row-normalised

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adj = self._adjacency(x)
        h = torch.relu(self.lin1(adj @ x))
        return self.lin2(adj @ h)


def build_model(kind: str, n_features: int, hidden: int):
    from .cnn import CNN
    from .gru import GRUNet

    models = {"cnn": CNN, "gru": GRUNet, "gnn": SimpleGNN}
    if kind not in models:
        raise ValueError(f"Unknown model type '{kind}'. Choose from {list(models)}")
    return models[kind](n_features, hidden)
