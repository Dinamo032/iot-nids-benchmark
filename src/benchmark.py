"""Training, evaluation and latency measurement."""
import time
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch import nn

from .compress import apply_compression
from .models.gnn import build_model
from .utils import peak_ram_mb


def train(model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: Dict) -> None:
    m = cfg["model"]
    opt = torch.optim.Adam(model.parameters(), lr=m["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    model.train()
    for _ in range(m["epochs"]):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), m["batch_size"]):
            idx = perm[i:i + m["batch_size"]]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(x)).argmax(1).numpy()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"macro_f1": float("nan"), "fpr": float("nan"), "recall": float("nan")}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


@torch.no_grad()
def measure_latency_ms(model: nn.Module, sample: np.ndarray, warmup: int, runs: int) -> Dict[str, float]:
    """Latency of a single-flow inference (batch size 1), in milliseconds."""
    model.eval()
    x = torch.from_numpy(sample[:1])
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return {"latency_mean_ms": float(np.mean(times)), "latency_p95_ms": float(np.percentile(times, 95))}


def run_once(cfg: Dict, data, seed: int) -> Dict[str, float]:
    x_tr, y_tr, x_te, y_te, x_zd, y_zd = data
    model = build_model(cfg["model"]["type"], x_tr.shape[1], cfg["model"]["hidden_size"])
    train(model, x_tr, y_tr, cfg)
    model = apply_compression(model, cfg["compression"]["mode"], cfg["compression"]["prune_amount"])

    result = {"seed": seed, **classification_metrics(y_te, predict(model, x_te))}
    if len(y_zd):
        result["recall_zero_day"] = classification_metrics(y_zd, predict(model, x_zd))["recall"]
    else:
        result["recall_zero_day"] = float("nan")
    b = cfg["benchmark"]
    result.update(measure_latency_ms(model, x_te, b["latency_warmup"], b["latency_runs"]))
    result["peak_ram_mb"] = peak_ram_mb()
    return result
