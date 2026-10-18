"""Smoke test: the full pipeline runs end-to-end on the bundled sample."""
import copy
import subprocess
import sys
from pathlib import Path

import yaml

from src.benchmark import run_once
from src.data import load_dataset
from src.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def test_all_models_and_compressions_run():
    for model in ("cnn", "gru", "gnn"):
        for comp in ("none", "int8", "prune"):
            cfg = copy.deepcopy(_cfg())
            cfg["data"]["path"] = str(ROOT / cfg["data"]["path"])
            cfg["model"].update(type=model, epochs=1)
            cfg["compression"]["mode"] = comp
            cfg["benchmark"].update(latency_warmup=1, latency_runs=3)
            set_seed(0)
            res = run_once(cfg, load_dataset(cfg, 0), 0)
            assert res["latency_mean_ms"] > 0


def test_cli_writes_outputs(tmp_path):
    cfg = _cfg()
    cfg["data"]["path"] = str(ROOT / cfg["data"]["path"])
    cfg["experiment"]["output_dir"] = str(tmp_path)
    cfg["experiment"]["seeds"] = [0]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, "-m", "src.main", "--config", str(cfg_path)], cwd=ROOT, check=True)
    assert (tmp_path / "sample_metrics.csv").exists()
    assert (tmp_path / "sample_run.log").exists()
