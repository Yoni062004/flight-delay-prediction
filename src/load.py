"""Clean the raw BTS files and save them as one Parquet file.

This is the ONLY file in the project that knows BTS column names. It renames them
to general names (carrier, tail_number, origin, ...). Features, training and SHAP
use only the general names, so another airline's data needs only a new version of
this file.

What it does:
  1. Reads each data/raw/ontime_YYYY_MM.csv, loading only the columns we need.
  2. Renames the columns and stores them in compact types to save memory.
  3. Turns the date text ("10/1/2025 12:00:00 AM") into a real date.
  4. Saves everything as data/processed/flights.parquet.

Cancelled and diverted flights stay in the file, marked by the `cancelled` and
`diverted` columns, because the aircraft-rotation features need the full schedule.
`load_flights()` removes them by default, so analysis and training never see them.

Build the Parquet file (from the project folder):
    .venv\\Scripts\\python src\\load.py

Use it afterwards:
    from load import load_flights
    flights = load_flights()                      # completed flights only
    schedule = load_flights(completed_only=False)  # every scheduled flight
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FLIGHTS_PARQUET = PROCESSED_DIR / "flights.parquet"

# BTS column name -> general name. Only these columns are loaded.
BTS_COLUMNS = {
    # Known before departure (allowed as features)
    "FL_DATE": "flight_date",
    "OP_UNIQUE_CARRIER": "carrier",
    "TAIL_NUM": "tail_number",
    "OP_CARRIER_FL_NUM": "flight_number",
    "ORIGIN": "origin",
    "DEST": "destination",
    "CRS_DEP_TIME": "scheduled_departure_hhmm",   # local time, e.g. 1435 = 2:35 PM
    "CRS_ARR_TIME": "scheduled_arrival_hhmm",     # local time at the destination
    "CRS_ELAPSED_TIME": "scheduled_elapsed_minutes",
    "DISTANCE": "distance_miles",
    # Known only after the flight (never features; see OUTCOME_COLUMNS)
    "CANCELLED": "cancelled",
    "DIVERTED": "diverted",
    "ARR_DEL15": "arrived_late",                  # the target: 1 = arrived 15+ minutes late
    "ARR_DELAY": "arrival_delay_minutes",
    "DEP_DELAY": "departure_delay_minutes",
    "CARRIER_DELAY": "carrier_delay_minutes",
    "WEATHER_DELAY": "weather_delay_minutes",
    "NAS_DELAY": "nas_delay_minutes",
    "SECURITY_DELAY": "security_delay_minutes",
    "LATE_AIRCRAFT_DELAY": "late_aircraft_delay_minutes",
}

# Columns that are only known after the flight. `arrived_late` is the target;
# the rest are for charts and analysis. None of them may be used as a feature.
OUTCOME_COLUMNS = [
    "cancelled", "diverted", "arrived_late", "arrival_delay_minutes", "departure_delay_minutes",
    "carrier_delay_minutes", "weather_delay_minutes", "nas_delay_minutes",
    "security_delay_minutes", "late_aircraft_delay_minutes",
]

_TEXT_COLUMNS = ["carrier", "tail_number", "origin", "destination"]
_SMALL_INT_COLUMNS = {
    "flight_number": "int16",
    "scheduled_departure_hhmm": "int16",
    "scheduled_arrival_hhmm": "int16",
    "distance_miles": "int16",
    "cancelled": "int8",
    "diverted": "int8",
}
_NUMERIC_BTS = [bts for bts, name in BTS_COLUMNS.items() if name not in _TEXT_COLUMNS and name != "flight_date"]


def read_bts_month(path: Path) -> pd.DataFrame:
    """Read one raw BTS month and return it with general column names and compact types."""
    df = pd.read_csv(path, usecols=list(BTS_COLUMNS), dtype={c: "float32" for c in _NUMERIC_BTS})
    df = df.rename(columns=BTS_COLUMNS)[list(BTS_COLUMNS.values())]

    # The date is text with a fake midnight time. Parse each distinct value once (about 30 per month).
    distinct = df["flight_date"].unique()
    parsed = pd.to_datetime(distinct, format="%m/%d/%Y %I:%M:%S %p")
    df["flight_date"] = df["flight_date"].map(dict(zip(distinct, parsed)))

    match = re.search(r"(\d{4})_(\d{2})", path.stem)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        wrong = (df["flight_date"].dt.year != year) | (df["flight_date"].dt.month != month)
        if wrong.any():
            raise ValueError(f"{path.name}: {wrong.sum()} rows are not from {year}-{month:02d}")

    for column, dtype in _SMALL_INT_COLUMNS.items():
        if df[column].isna().any():
            raise ValueError(f"{path.name}: {column} has missing values")
        df[column] = df[column].astype(dtype)
    return df


def build_flights_parquet() -> pd.DataFrame:
    """Read every raw month, combine them, and save data/processed/flights.parquet."""
    files = sorted(RAW_DIR.glob("ontime_*.csv"))
    if not files:
        raise FileNotFoundError(f"No ontime_*.csv files in {RAW_DIR}. Run src/download_bts.py first.")

    frames = []
    for path in files:
        start = time.time()
        frames.append(read_bts_month(path))
        print(f"  read {path.name}: {len(frames[-1]):,} flights in {time.time() - start:.0f}s", flush=True)

    df = pd.concat(frames, ignore_index=True)
    del frames
    for column in _TEXT_COLUMNS:
        df[column] = df[column].astype("category")
    df = df.sort_values(
        ["flight_date", "scheduled_departure_hhmm", "carrier", "flight_number", "origin"],
        ignore_index=True,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FLIGHTS_PARQUET, index=False)
    return df


def load_flights(completed_only: bool = True, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the cleaned flights.

    completed_only=True (default) removes cancelled and diverted flights, which have
    no arrival delay, and returns `arrived_late` as 0/1 whole numbers.
    completed_only=False returns every scheduled flight, for building schedule features.
    """
    if not FLIGHTS_PARQUET.exists():
        raise FileNotFoundError(f"{FLIGHTS_PARQUET} not found. Run src/load.py first.")
    if columns is not None and completed_only:
        needed = {"cancelled", "diverted", "arrived_late"}
        read_columns = list(dict.fromkeys(list(columns) + sorted(needed)))
    else:
        read_columns = columns
    df = pd.read_parquet(FLIGHTS_PARQUET, columns=read_columns)
    if not completed_only:
        return df

    keep = (df["cancelled"] == 0) & (df["diverted"] == 0) & df["arrived_late"].notna()
    df = df[keep].drop(columns=["cancelled", "diverted"]).reset_index(drop=True)
    df["arrived_late"] = df["arrived_late"].astype("int8")
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    return df


def _print_summary(df: pd.DataFrame) -> None:
    completed = (df["cancelled"] == 0) & (df["diverted"] == 0) & df["arrived_late"].notna()
    by_month = pd.DataFrame({
        "scheduled": df.groupby(df["flight_date"].dt.to_period("M")).size(),
        "cancelled": df.groupby(df["flight_date"].dt.to_period("M"))["cancelled"].sum(),
        "diverted": df.groupby(df["flight_date"].dt.to_period("M"))["diverted"].sum(),
        "completed": completed.groupby(df["flight_date"].dt.to_period("M")).sum(),
    })
    late = df[completed].groupby(df.loc[completed, "flight_date"].dt.to_period("M"))["arrived_late"].mean()
    by_month["late_%"] = (100 * late).round(1)
    print()
    print(by_month.to_string())
    print()
    print(f"Scheduled flights: {len(df):,}")
    print(f"Cancelled:         {int(df['cancelled'].sum()):,}")
    print(f"Diverted:          {int(df['diverted'].sum()):,}")
    print(f"Completed flights: {int(completed.sum()):,}  (late: {100 * df.loc[completed, 'arrived_late'].mean():.1f}%)")
    print(f"Memory in pandas:  {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(f"Parquet on disk:   {FLIGHTS_PARQUET.stat().st_size / 1e6:.0f} MB  ({FLIGHTS_PARQUET})")


def main() -> int:
    start = time.time()
    print(f"Reading raw files from {RAW_DIR}")
    df = build_flights_parquet()
    _print_summary(df)
    print(f"Done in {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
