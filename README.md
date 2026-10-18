# Lightweight Deep Learning Architectures for Network Intrusion Detection in Resource-Constrained IoT Environments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange.svg)

**Authors:** Amir Khaziyev, Asylbek Seisembay
**Repository:** `https://github.com/Dinamo032/iot-nids-benchmark`

---

## 1. Project Overview

Deep-learning-based Network Intrusion Detection Systems (NIDS) reach very high accuracy on closed-world IoT benchmarks, yet existing studies typically evaluate only one of three dimensions in isolation: (i) detection quality of the architecture, (ii) the effect of model compression on inference latency on real edge hardware, or (iii) generalization to previously unseen (zero-day) attacks. This repository provides a unified, reproducible benchmarking pipeline that evaluates all three dimensions under identical experimental conditions.

### Research Aim

To empirically compare lightweight CNN, GRU and GNN architectures for intrusion detection in IoT networks in terms of accuracy, computational efficiency, and generalization to unseen attack classes.

### Research Questions

| ID | Research Question |
|----|-------------------|
| **RQ1** (primary) | How does the model architecture (CNN vs. GRU vs. GNN) affect macro-F1 and false positive rate (FPR) on benchmark IoT datasets (TON-IoT, CIC-IoT2023)? |
| RQ2 (extension) | How does model compression (quantization, pruning) affect the trade-off between F1-score and inference latency on resource-constrained hardware? |
| RQ3 (extension) | Does graph-based traffic representation or self-supervised pre-training improve recall on held-out (zero-day) attack classes compared with fully supervised training? |

### Core Hypothesis (RQ1)

- **H0:** There is no statistically significant difference in macro-F1 and FPR between CNN, GRU and GNN architectures under identical preprocessing and training conditions.
- **H1:** The GNN architecture achieves a statistically significantly higher macro-F1 and lower FPR than CNN and GRU (paired test over multiple random seeds, α = 0.05).

The full Methodology Card (variables, metrics, baseline) is in [`docs/methodology_card.md`](docs/methodology_card.md).

---

## 2. Repository Structure

```
iot-nids-benchmark/
├── README.md                 # this file
├── LICENSE                   # MIT
├── Dockerfile                # pinned runtime environment
├── .dockerignore
├── .gitignore
├── requirements.txt          # exact version pins
├── Makefile                  # build / run / test shortcuts
├── configs/
│   └── config.yaml           # experiment parameters (independent variables, paths, seed)
├── data/
│   ├── README.md             # download instructions for full datasets
│   └── sample/               # tiny sample (5-10 rows) for smoke testing
├── src/
│   ├── main.py               # entry point: reads config, runs the benchmark
│   ├── benchmark.py          # metric and latency measurement
│   ├── data.py               # loading and preprocessing
│   ├── compress.py           # quantization and pruning (RQ2)
│   ├── utils.py              # seeding, logging
│   └── models/               # cnn.py, gru.py, gnn.py
├── scripts/
│   └── run.sh                # one-command wrapper
├── notebooks/                # exploratory analysis (outputs cleared)
├── results/                  # generated logs and tables (git-ignored)
├── tests/
│   └── test_smoke.py
└── docs/
    └── methodology_card.md
```

---

## 3. System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Linux, macOS or Windows 10/11 (with WSL2) |
| Software | Docker >= 24.0 (only requirement for the Quickstart) |
| CPU / RAM | x86-64 or ARM64; >= 4 GB RAM for the sample run, >= 16 GB for full datasets |
| GPU | **Not required.** All reported latency measurements use CPU inference |
| Disk | ~3 GB for the image; additional space for full datasets (see `data/README.md`) |
| Edge target (RQ2) | Raspberry Pi 4/5 (ARM64) or Jetson Nano |

---

## 4. Quickstart

Reproduce the smoke-test experiment on the bundled sample data in three commands:

```bash
git clone https://github.com/Dinamo032/iot-nids-benchmark.git && cd iot-nids-benchmark
docker build -t iot-nids-benchmark .
docker run --rm -v "$(pwd)/results:/app/results" iot-nids-benchmark
```

The container runs `src/main.py` with `configs/config.yaml` on `data/sample/` and writes metrics and a log to `results/`. Expected output: a `results/sample_run.log` file and a `results/sample_metrics.csv` file.

### Running the full experiment

1. Download the datasets following [`data/README.md`](data/README.md) and place them in `data/raw/`.
2. Edit `configs/config.yaml` (set `data.path`, `model.type`, `compression.mode`, `seed`).
3. Run:

```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" \
  iot-nids-benchmark python src/main.py --config configs/config.yaml
```

### Reproducibility notes

- All random generators (`random`, `numpy`, `torch`) are seeded from `config.yaml`; deterministic algorithms are enabled in PyTorch.
- All dependencies are pinned to exact versions in `requirements.txt` and the base image is pinned in the `Dockerfile`.
- Each result is averaged over multiple seeds (see `config.yaml: seeds`).

---

## 5. Planned Benchmarks / Expected Results

The table below is a **placeholder**; values will be filled after the data collection phase (Week 7).

| Model | Compression | Macro-F1 | FPR | Recall (zero-day) | Latency (ms) | RAM (MB) |
|-------|-------------|----------|-----|-------------------|--------------|----------|
| CNN (baseline) | FP32 | TBD | TBD | TBD | TBD | TBD |
| GRU | FP32 | TBD | TBD | TBD | TBD | TBD |
| GNN | FP32 | TBD | TBD | TBD | TBD | TBD |
| Best model | INT8 | TBD | TBD | TBD | TBD | TBD |
| Best model | Pruned 50% | TBD | TBD | TBD | TBD | TBD |

**Primary metric:** macro-F1. **Guardrail metrics:** FPR, CPU utilization <= 80%, no crashes or out-of-memory errors during inference.

---

## 6. Citation

If you use this repository, please cite:

```bibtex
@misc{khaziyev_seisembay_2026_iotnids,
  author       = {Khaziyev, Amir and Seisembay, Asylbek},
  title        = {Lightweight Deep Learning Architectures for Network Intrusion
                  Detection in Resource-Constrained IoT Environments},
  year         = {2026},
  howpublished = {\url{https://github.com/Dinamo032/iot-nids-benchmark}},
  note         = {Research repository, work in progress}
}
```

## 7. License

This project is released under the [MIT License](LICENSE). The datasets (TON-IoT, CIC-IoT2023) are subject to their own licenses; see `data/README.md`.
