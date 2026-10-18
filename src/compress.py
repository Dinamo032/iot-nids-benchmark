"""Model compression: dynamic INT8 quantization and L1 pruning (RQ2)."""
import torch
from torch import nn
from torch.nn.utils import prune


def apply_compression(model: nn.Module, mode: str, prune_amount: float = 0.5) -> nn.Module:
    if mode == "none":
        return model
    if mode == "int8":
        # Dynamic quantization supports Linear / GRU / LSTM layers (not Conv1d).
        return torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear, nn.GRU}, dtype=torch.qint8
        )
    if mode == "prune":
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                prune.l1_unstructured(module, name="weight", amount=prune_amount)
                prune.remove(module, "weight")  # make pruning permanent
        return model
    raise ValueError(f"Unknown compression mode '{mode}' (use none | int8 | prune)")
