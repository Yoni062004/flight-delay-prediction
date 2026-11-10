"""Train the models and score them fairly.

The split is by date, the way the model would be used in real life:
    train     Sep 2025 - May 2026   the model learns from these flights
    validate  Jun 2026              decides when LightGBM stops adding trees
    test      Jul - Aug 2026        used once, at the end, for the reported scores

Four models are compared:
    1. "Always on time"            predicts the same low risk for every flight
    2. Logistic regression         a simple model on the schedule features
    3. LightGBM, schedule only     the features that need only the timetable
    4. LightGBM, all features      adds the aircraft rotation and aircraft details

Model 3 is the conservative result. Model 4 uses the aircraft that actually flew,
which can differ from the plan (see features.py).

Scores:
    ROC-AUC   how well the model ranks late flights above on-time ones
              (0.5 = guessing, 1.0 = perfect)
    PR-AUC    the same idea, focused on the late flights; compare it with the
              share of late flights, which is what guessing would score
    Late rate in the riskiest 10%   of the flights the model is most worried
              about, how many really arrived late

Run (from the project folder):
    .venv\\Scripts\\python src\\train.py              full run, saves everything
    .venv\\Scripts\\python src\\train.py --sample 0.2 quick try on 20% of the training flights
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import FEATURE_COLUMNS, ID_COLUMNS, SCHEDULE_ONLY_FEATURES, TARGET, load_features
from load import PROCESSED_DIR, ROOT

VALIDATE_START = "2026-06-01"
TEST_START = "2026-07-01"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PREDICTIONS_PARQUET = PROCESSED_DIR / "predictions.parquet"
METRICS_CSV = REPORTS_DIR / "metrics.csv"
IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"

# Three calendar features are left out of the models, because the data covers only one year:
#   - `month`: the test months (July, August) never appear in training, so the model
#     has no way to learn what "month 7" means.
#   - `day_of_month` and `days_from_holiday`: together with the weekday they pin down an
#     exact date. The model then memorizes single bad days (a storm on one Sunday in
#     January) instead of patterns that repeat. Each holiday also appears only once, so
#     its effect can't be told apart from that day's weather.
# Measured on the validation month with 20% of the training flights: with them LightGBM
# stopped after 13 trees (ROC-AUC 0.696); without them it grew 200+ trees (0.707).
# With several years of data these features would be worth adding back.
NOT_USED = ["month", "day_of_month", "days_from_holiday"]
ALL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NOT_USED]
SCHEDULE_FEATURES = [c for c in SCHEDULE_ONLY_FEATURES if c not in NOT_USED]

LOGISTIC_CATEGORIES = ["carrier", "departure_hour", "day_of_week"]
LOGISTIC_NUMBERS = [
    "distance_miles", "scheduled_elapsed_minutes",
    "origin_departures_that_hour", "destination_arrivals_that_hour",
    "origin_late_rate", "destination_late_rate",
]

# Deliberately cautious settings: small trees, large leaves, and each tree sees only part
# of the features. Chosen on the validation month; the test months were not looked at.
LIGHTGBM_SETTINGS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 5000,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 50.0,
    "cat_smooth": 50,
    "cat_l2": 10.0,
    "seed": 42,
    "verbose": -1,
}
MAX_TREES = 3000
STOP_AFTER_NO_GAIN = 50


def split_labels(dates: pd.Series) -> pd.Series:
    """Label every flight as train, validate or test, by its date."""
    labels = np.where(dates >= TEST_START, "test", np.where(dates >= VALIDATE_START, "validate", "train"))
    return pd.Series(labels, index=dates.index, name="split")


def score(actual, predicted) -> dict:
    """ROC-AUC, PR-AUC, and the late rate among the 10% of flights with the highest predicted risk."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    riskiest = predicted >= np.quantile(predicted, 0.9)
    return {
        "flights": int(len(actual)),
        "late_rate": float(actual.mean()),
        "roc_auc": float(roc_auc_score(actual, predicted)),
        "pr_auc": float(average_precision_score(actual, predicted)),
        "late_rate_in_riskiest_10pct": float(actual[riskiest].mean()),
    }


# ---------- logistic regression (the simple baseline) ----------

def airport_late_rates(train: pd.DataFrame, column: str, smoothing: int = 200) -> pd.Series:
    """Late rate per airport, measured on the TRAINING flights only.

    Small airports are pulled toward the overall rate, so a handful of flights
    cannot produce an extreme value.
    """
    overall = train[TARGET].mean()
    stats = train.groupby(column, observed=True)[TARGET].agg(["sum", "count"])
    return (stats["sum"] + smoothing * overall) / (stats["count"] + smoothing)


def logistic_inputs(df: pd.DataFrame, rates: dict[str, pd.Series], overall: float) -> pd.DataFrame:
    """The columns the logistic regression uses, with airports replaced by their training-period late rate."""
    inputs = df[LOGISTIC_CATEGORIES + [c for c in LOGISTIC_NUMBERS if c in df.columns]].copy()
    for column in ("origin", "destination"):
        by_category = rates[column].reindex(df[column].cat.categories).fillna(overall).to_numpy()
        inputs[f"{column}_late_rate"] = by_category[df[column].cat.codes]
    return inputs


def fit_logistic(train: pd.DataFrame):
    """Fit the baseline and return (model, function that prepares its inputs)."""
    overall = float(train[TARGET].mean())
    rates = {column: airport_late_rates(train, column) for column in ("origin", "destination")}
    prepare = lambda df: logistic_inputs(df, rates, overall)
    model = make_pipeline(
        ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), LOGISTIC_CATEGORIES),
            ("numbers", StandardScaler(), LOGISTIC_NUMBERS),
        ]),
        LogisticRegression(max_iter=300),
    )
    model.fit(prepare(train), train[TARGET])
    return model, prepare


# ---------- LightGBM (the main model) ----------

def fit_lightgbm(train: pd.DataFrame, validate: pd.DataFrame, features: list[str]) -> lgb.Booster:
    """Train LightGBM, adding trees until the validation month stops improving."""
    train_set = lgb.Dataset(train[features], label=train[TARGET])
    validate_set = lgb.Dataset(validate[features], label=validate[TARGET], reference=train_set)
    return lgb.train(
        LIGHTGBM_SETTINGS, train_set, num_boost_round=MAX_TREES,
        valid_sets=[validate_set], valid_names=["validate"],
        callbacks=[lgb.early_stopping(STOP_AFTER_NO_GAIN, verbose=False), lgb.log_evaluation(100)],
    )


def importance_table(model: lgb.Booster, name: str) -> pd.DataFrame:
    """How much each feature improved the model's splits, as a share of the total."""
    gain = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    share = (100 * gain / gain.sum()).sort_values(ascending=False).rename("gain_share_%")
    return share.rename_axis("feature").reset_index().assign(model=name)


# ---------- the full run ----------

def run(sample: float = 1.0) -> pd.DataFrame:
    start = time.time()
    data = load_features()
    data["split"] = split_labels(data["flight_date"])
    train = data[data["split"] == "train"]
    if sample < 1:
        train = train.sample(frac=sample, random_state=42)
    validate = data[data["split"] == "validate"]
    holdout = data[data["split"] != "train"].copy()
    print(f"train {len(train):,} | validate {len(validate):,} | test {(data['split'] == 'test').sum():,}", flush=True)

    predictions = {}
    predictions["Always on time"] = np.full(len(holdout), train[TARGET].mean())

    print("Fitting logistic regression ...", flush=True)
    logistic, prepare = fit_logistic(train)
    predictions["Logistic regression"] = logistic.predict_proba(prepare(holdout))[:, 1]
    print(f"  done ({time.time() - start:.0f}s)", flush=True)

    models = {}
    for name, features in [("LightGBM, schedule only", SCHEDULE_FEATURES), ("LightGBM, all features", ALL_FEATURES)]:
        print(f"Fitting {name} ({len(features)} features) ...", flush=True)
        models[name] = fit_lightgbm(train, validate, features)
        predictions[name] = models[name].predict(holdout[features], num_iteration=models[name].best_iteration)
        print(f"  {models[name].best_iteration} trees ({time.time() - start:.0f}s)", flush=True)

    groups = {
        "All flights": pd.Series(True, index=holdout.index),
        "Delta": holdout["carrier"] == "DL",
        "Recorded aircraft fits the schedule": holdout["aircraft_plan_known"] == 1,
        "Aircraft swapped in on the day": holdout["aircraft_plan_known"] == 0,
    }
    rows = []
    for split in ("validate", "test"):
        for group, in_group in groups.items():
            mask = ((holdout["split"] == split) & in_group).to_numpy()
            for name, predicted in predictions.items():
                rows.append({"split": split, "group": group, "model": name,
                             **score(holdout.loc[mask, TARGET], predicted[mask]),
                             "average_predicted_risk": float(predicted[mask].mean())})
    metrics = pd.DataFrame(rows)

    if sample == 1:
        MODELS_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)
        info = {}
        for name, model in models.items():
            key = "schedule_only" if "schedule" in name else "all_features"
            model.save_model(MODELS_DIR / f"lightgbm_{key}.txt", num_iteration=model.best_iteration)
            info[key] = {"name": name, "trees": model.best_iteration, "features": model.feature_name()}
        (MODELS_DIR / "model_info.json").write_text(json.dumps(
            {"validate_start": VALIDATE_START, "test_start": TEST_START,
             "lightgbm_settings": LIGHTGBM_SETTINGS, "models": info}, indent=2))

        metrics.round(4).to_csv(METRICS_CSV, index=False)
        pd.concat([importance_table(m, n) for n, m in models.items()]).round(2).to_csv(IMPORTANCE_CSV, index=False)

        saved = holdout[ID_COLUMNS + ["carrier", "origin", "destination", "departure_hour",
                                      "aircraft_plan_known", TARGET, "split"]].copy()
        saved["risk_logistic"] = predictions["Logistic regression"].astype("float32")
        saved["risk_schedule_only"] = predictions["LightGBM, schedule only"].astype("float32")
        saved["risk_all_features"] = predictions["LightGBM, all features"].astype("float32")
        saved.to_parquet(PREDICTIONS_PARQUET, index=False)
        print(f"Saved models to {MODELS_DIR}, scores to {METRICS_CSV}, predictions to {PREDICTIONS_PARQUET}")
    else:
        print("Sample run: nothing saved.")

    print(f"Done in {time.time() - start:.0f}s")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and score the flight delay models.")
    parser.add_argument("--sample", type=float, default=1.0, help="share of the training flights to use (default: all)")
    args = parser.parse_args()
    metrics = run(args.sample)
    pd.set_option("display.width", 200)
    main_groups = metrics[metrics["group"].isin(["All flights", "Delta"])]
    print(main_groups.drop(columns="average_predicted_risk").round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
