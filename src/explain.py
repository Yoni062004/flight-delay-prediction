"""Explain the model's predictions with SHAP.

For one flight, SHAP splits the model's prediction into one number per feature:
how much that feature pushed the risk up (positive) or down (negative), compared
with an average flight. Adding all the pushes to the starting point gives back
exactly the model's prediction.

The pushes are in "log-odds", the scale the model works in. Two handy readings:
  - bigger push (either direction) = the feature mattered more for that flight
  - exp(push) = how many times the odds of a late arrival were multiplied
    (1.0 = no effect, 2.0 = doubled, 0.5 = halved)

This file explains the all-features LightGBM model on two samples of test flights
(July - August 2026): 50,000 Delta flights and 50,000 flights from all airlines.

Run (from the project folder), after src\\train.py:
    .venv\\Scripts\\python src\\explain.py
"""

from __future__ import annotations

import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import ID_COLUMNS, TARGET, load_features
from load import PROCESSED_DIR
from train import MODELS_DIR, TEST_START

MODEL_FILE = MODELS_DIR / "lightgbm_all_features.txt"
SAMPLE_SIZE = 50_000
SHAP_PREFIX = "push_"          # the SHAP value of feature X is stored in column push_X


def shap_file(name: str):
    return PROCESSED_DIR / f"shap_{name}.parquet"


def explain(model: lgb.Booster, flights: pd.DataFrame) -> pd.DataFrame:
    """Return the flights with their predicted risk and one SHAP column per feature."""
    import shap   # imported here so that loading saved results does not need it

    features = model.feature_name()
    inputs = flights[features]
    explainer = shap.TreeExplainer(model)
    pushes = np.asarray(explainer.shap_values(inputs))
    start = float(np.ravel(explainer.expected_value)[-1])

    # Safety check: starting point + all pushes must equal the model's own output.
    raw = model.predict(inputs, raw_score=True)
    gap = np.abs(start + pushes.sum(axis=1) - raw).max()
    if gap > 1e-3:
        raise RuntimeError(f"SHAP values do not add up to the model output (largest gap {gap:.4f})")

    result = flights[ID_COLUMNS + features + [TARGET]].copy()
    result["risk"] = (1 / (1 + np.exp(-raw))).astype("float32")
    result["starting_point"] = np.float32(start)
    for position, feature in enumerate(features):
        result[SHAP_PREFIX + feature] = pushes[:, position].astype("float32")
    return result.reset_index(drop=True)


def load_shap(name: str) -> pd.DataFrame:
    """Load a saved SHAP table: 'delta' or 'all'."""
    path = shap_file(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/explain.py first.")
    return pd.read_parquet(path)


def main() -> int:
    start = time.time()
    model = lgb.Booster(model_file=str(MODEL_FILE))
    data = load_features()
    test = data[data["flight_date"] >= TEST_START]
    samples = {
        "delta": test[test["carrier"] == "DL"].sample(SAMPLE_SIZE, random_state=42),
        "all": test.sample(SAMPLE_SIZE, random_state=42),
    }
    for name, flights in samples.items():
        table = explain(model, flights)
        table.to_parquet(shap_file(name), index=False)
        print(f"{name}: {len(table):,} flights explained, average risk {table['risk'].mean():.1%}, "
              f"actually late {table[TARGET].mean():.1%}  ({time.time() - start:.0f}s)", flush=True)
    print(f"Saved {shap_file('delta')} and {shap_file('all')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
