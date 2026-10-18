"""Data loading and preprocessing."""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(cfg: Dict, seed: int, logger=None) -> Tuple[np.ndarray, ...]:
    """Return X_train, y_train, X_test, y_test, X_zero_day, y_zero_day (float32/int64).

    - Rows whose attack type is in `holdout_attack_types` are removed from training
      and used only for the zero-day evaluation (RQ3).
    - Tiny datasets (< min_rows_for_split) are used for both training and testing:
      this is a smoke test only, metrics are NOT meaningful.
    """
    d = cfg["data"]
    df = pd.read_csv(d["path"])
    label, atype = d["label_column"], d["attack_type_column"]
    feature_cols = [c for c in df.columns if c not in (label, atype)]

    holdout = d.get("holdout_attack_types") or []
    zero_day = df[df[atype].isin(holdout)] if holdout else df.iloc[0:0]
    known = df[~df[atype].isin(holdout)] if holdout else df

    if len(known) < d["min_rows_for_split"]:
        if logger:
            logger.warning("Dataset has %d rows: train == test (smoke test only).", len(known))
        train, test = known, known
    else:
        train, test = train_test_split(
            known, test_size=d["test_size"], random_state=seed, stratify=known[label]
        )

    scaler = StandardScaler().fit(train[feature_cols])  # fit on train only (no leakage)

    def to_xy(frame):
        x = scaler.transform(frame[feature_cols]).astype(np.float32) if len(frame) else np.zeros((0, len(feature_cols)), np.float32)
        y = frame[label].to_numpy().astype(np.int64)
        return x, y

    return (*to_xy(train), *to_xy(test), *to_xy(zero_day))
