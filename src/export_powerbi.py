"""Export small summary tables for the Power BI report.

Power BI does not need all 7 million flights for one page. This file writes a
handful of small CSV files into powerbi/data/, arranged as a star schema:

  Lookup tables (one row per thing)
    dim_airline.csv    airline code and name
    dim_airport.csv    airport code, name, city, state, map position
    dim_hour.csv       departure hour and a readable label

  Fact tables (numbers to add up)
    fact_flights_by_airport_hour.csv   full year: flights and late flights by airline, origin, hour
    fact_flights_by_month.csv          full year: flights and late flights by airline and month
    fact_risk_groups.csv               test months: flights in each of the model's 10 risk groups
    fact_model_scores.csv              test months: ROC-AUC and PR-AUC per airline
    fact_model_scores_overall.csv      test months: the headline scores for all flights
    fact_shap_groups.csv               what drives the predictions, per airline

Run (from the project folder), after train.py, explain.py and airports.py:
    .venv\\Scripts\\python src\\export_powerbi.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from airports import load_airports
from explain import SHAP_PREFIX, load_shap
from features import FEATURE_GROUPS, TARGET, load_features
from load import ROOT
from train import METRICS_CSV, PREDICTIONS_PARQUET

POWERBI_DATA = ROOT / "powerbi" / "data"

AIRLINE_NAMES = {
    "AA": "American", "AS": "Alaska", "B6": "JetBlue", "DL": "Delta", "F9": "Frontier",
    "G4": "Allegiant", "HA": "Hawaiian", "MQ": "Envoy", "NK": "Spirit", "OH": "PSA",
    "OO": "SkyWest", "UA": "United", "WN": "Southwest", "YX": "Republic",
}
MODELS = {"risk_schedule_only": "Timetable only", "risk_all_features": "Timetable + aircraft"}
GROUP_NAMES = {
    "time": "Time of day and weekday",
    "flight": "Route and airline",
    "rotation": "Aircraft rotation",
    "aircraft": "Aircraft type and age",
    "crowding": "Airport crowding",
}


def hour_label(hour: int) -> str:
    return f"{(hour - 1) % 12 + 1} {'AM' if hour < 12 else 'PM'}"


def save(table: pd.DataFrame, name: str) -> None:
    POWERBI_DATA.mkdir(parents=True, exist_ok=True)
    path = POWERBI_DATA / name
    table.to_csv(path, index=False, encoding="utf-8")
    print(f"  {name:36s} {len(table):>7,} rows  {path.stat().st_size / 1e3:>8,.0f} KB")


def count_late(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Flights and late flights for each combination of `keys`."""
    table = frame.groupby(keys, observed=True)[TARGET].agg(flights="size", late_flights="sum").reset_index()
    return table[table["flights"] > 0]


def main() -> int:
    print(f"Writing Power BI tables to {POWERBI_DATA}")

    # ---------- lookup tables ----------
    features = load_features(columns=["flight_date", "carrier", "origin", "departure_hour", TARGET])
    features["carrier"] = features["carrier"].astype(str)
    features["origin"] = features["origin"].astype(str)

    carriers = sorted(features["carrier"].unique())
    save(pd.DataFrame({"carrier": carriers,
                       "airline": [AIRLINE_NAMES.get(c, c) for c in carriers]}), "dim_airline.csv")

    airports = load_airports()
    airports["airport_label"] = airports["city"] + " (" + airports["airport"] + ")"
    # The map shows the contiguous U.S. only; Alaska, Hawaii and the territories would zoom it out to the whole Pacific.
    far_away = airports["state"].isin(["AK", "HI", "PR", "VI", "GU", "MP", "AS"])
    airports["map_region"] = np.where(far_away, "Alaska, Hawaii and territories", "Contiguous U.S.")
    save(airports[["airport", "airport_name", "airport_label", "city", "state", "latitude", "longitude", "size",
                   "map_region"]], "dim_airport.csv")

    save(pd.DataFrame({"departure_hour": range(24), "hour_label": [hour_label(h) for h in range(24)]}), "dim_hour.csv")

    # ---------- full-year facts ----------
    save(count_late(features, ["carrier", "origin", "departure_hour"]), "fact_flights_by_airport_hour.csv")

    features["month_start"] = features["flight_date"].dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
    save(count_late(features, ["carrier", "month_start"]), "fact_flights_by_month.csv")

    # ---------- model results on the test months ----------
    predictions = pd.read_parquet(PREDICTIONS_PARQUET)
    test = predictions[predictions["split"] == "test"].copy()
    test["carrier"] = test["carrier"].astype(str)

    risk_groups, scores = [], []
    for column, model in MODELS.items():
        # Ten equal groups over ALL test flights, so a group means the same thing for every airline.
        test["risk_group"] = pd.qcut(test[column].rank(method="first"), 10, labels=range(1, 11)).astype(int)
        grouped = test.groupby(["carrier", "risk_group"]).agg(
            flights=(TARGET, "size"), late_flights=(TARGET, "sum"), predicted_late_flights=(column, "sum")).reset_index()
        grouped["predicted_late_flights"] = grouped["predicted_late_flights"].round(1)
        risk_groups.append(grouped.assign(model=model))

        for carrier, flights in test.groupby("carrier"):
            scores.append({
                "carrier": carrier, "model": model, "flights": len(flights),
                "late_rate": round(float(flights[TARGET].mean()), 4),
                "roc_auc": round(float(roc_auc_score(flights[TARGET], flights[column])), 4),
                "pr_auc": round(float(average_precision_score(flights[TARGET], flights[column])), 4),
            })
    save(pd.concat(risk_groups, ignore_index=True), "fact_risk_groups.csv")
    save(pd.DataFrame(scores), "fact_model_scores.csv")

    overall = pd.read_csv(METRICS_CSV)
    overall = overall[(overall["split"] == "test") & (overall["group"] == "All flights")]
    overall = overall.replace({"model": {"LightGBM, schedule only": "Timetable only",
                                         "LightGBM, all features": "Timetable + aircraft"}})
    save(overall[["model", "flights", "late_rate", "roc_auc", "pr_auc", "late_rate_in_riskiest_10pct"]],
         "fact_model_scores_overall.csv")

    # ---------- what drives the predictions (SHAP) ----------
    # Delta comes from its own 50,000-flight sample; the other airlines from the all-airline sample.
    everyone = load_shap("all")
    everyone["carrier"] = everyone["carrier"].astype(str)
    samples = {carrier: flights for carrier, flights in everyone.groupby("carrier")}
    samples["DL"] = load_shap("delta")
    rows = []
    for carrier, flights in sorted(samples.items()):
        sizes = {}
        for group, columns in FEATURE_GROUPS.items():
            pushes = [SHAP_PREFIX + c for c in columns if SHAP_PREFIX + c in flights.columns]
            sizes[GROUP_NAMES[group]] = float(np.abs(flights[pushes].sum(axis=1)).mean())
        total = sum(sizes.values())
        for group, size in sizes.items():
            rows.append({"carrier": carrier, "feature_group": group, "share": round(size / total, 4),
                         "flights_in_sample": len(flights)})
    save(pd.DataFrame(rows), "fact_shap_groups.csv")

    total_kb = sum(f.stat().st_size for f in POWERBI_DATA.glob("*.csv")) / 1e3
    print(f"Done. {len(list(POWERBI_DATA.glob('*.csv')))} files, {total_kb:,.0f} KB in total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
