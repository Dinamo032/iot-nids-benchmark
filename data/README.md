# Data

Only `data/sample/` is tracked in Git (10 synthetic rows for smoke testing; **not** real traffic).

## Full datasets (not tracked)

Download manually and place under `data/raw/`:

| Dataset | Source | License |
|---------|--------|---------|
| TON-IoT | UNSW Canberra Cyber, https://research.unsw.edu.au/projects/toniot-datasets | see source page |
| CIC-IoT2023 | Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/iotdataset-2023.html | see source page |

> Verify the current download links on the dataset pages; they may change.

Expected CSV format (after preprocessing): numeric feature columns, a binary `label` column (0 = benign, 1 = attack) and an `attack_type` column. Column names are set in `configs/config.yaml`.

## Feature notes- f1-f8: normalized flow-level features (packet rate, byte rate, duration, etc.).  Replace with the real TON-IoT / CIC-IoT2023 feature list after preprocessing.
