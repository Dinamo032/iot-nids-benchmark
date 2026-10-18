"""Entry point: python -m src.main --config configs/config.yaml"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from .benchmark import run_once
from .data import load_dataset
from .utils import get_logger, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="IoT NIDS benchmark")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["experiment"]["output_dir"])
    name = cfg["experiment"]["name"]
    log = get_logger(out_dir / f"{name}_run.log")
    log.info("Config: model=%s compression=%s data=%s",
             cfg["model"]["type"], cfg["compression"]["mode"], cfg["data"]["path"])

    rows = []
    for seed in cfg["experiment"]["seeds"]:
        set_seed(seed)
        data = load_dataset(cfg, seed, log)
        res = run_once(cfg, data, seed)
        log.info("seed=%d | %s", seed, {k: round(v, 4) for k, v in res.items() if k != "seed"})
        rows.append({"model": cfg["model"]["type"], "compression": cfg["compression"]["mode"], **res})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_metrics.csv"
    df.to_csv(csv_path, index=False)
    mean = df.drop(columns=["model", "compression", "seed"]).mean()
    log.info("MEAN over %d seeds: %s", len(df), mean.round(4).to_dict())
    log.info("Saved metrics to %s", csv_path)


if __name__ == "__main__":
    main()
