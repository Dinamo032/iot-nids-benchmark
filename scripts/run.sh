#!/usr/bin/env bash
# One-command wrapper: run the benchmark with the default config.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.main --config "${1:-configs/config.yaml}"
