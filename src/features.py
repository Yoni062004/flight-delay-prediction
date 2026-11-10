"""Turn the flight schedule into model inputs (features).

Every feature here comes from the SCHEDULE: who flies, where, when, and which
aircraft is planned for the flight. Nothing uses what happened on the day
(actual times, delays, cancellations).

One trap is handled on purpose. BTS records the aircraft that ACTUALLY flew, and
airlines swap aircraft on the day when a plane is running late. A swapped-in
aircraft often shows an impossible schedule (it was due to land after this
flight's departure, or at another airport), and 9 out of 10 of those flights are
late. Using that would be leakage: the swap is a result of the delay. So for
those flights the rotation features are replaced with the flight's usual pattern
from other days. See add_rotation_features().

Feature groups:
  1. Time:      hour, weekday, month, day of month, days from a U.S. federal holiday.
  2. Flight:    airline, origin, destination, distance, scheduled flight time.
  3. Rotation:  the aircraft's leg number that day, its scheduled turnaround time,
                where it comes from, how long it has been flying that day.
  4. Crowding:  how many flights are scheduled at the origin and destination
                in the same hour.

This file uses only the general column names from load.py, so it works for any
airline's data once that data has been translated.

Build the feature table (from the project folder):
    .venv\\Scripts\\python src\\features.py

Use it afterwards:
    from features import load_features, FEATURE_COLUMNS, CATEGORICAL_FEATURES, TARGET
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from load import PROCESSED_DIR, load_flights

FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"

TARGET = "arrived_late"

# Columns that identify a flight. They are kept for joins and charts, not used as model inputs.
ID_COLUMNS = ["flight_date", "flight_number", "tail_number"]

# For analysis only, NEVER a model input: 1 if the recorded aircraft fits the schedule,
# 0 if it was probably swapped in on the day (its rotation features are then usual values).
DIAGNOSTIC_COLUMNS = ["aircraft_plan_known"]

CATEGORICAL_FEATURES = ["carrier", "origin", "destination", "previous_leg_origin"]
NUMERIC_FEATURES = [
    # time
    "departure_hour", "arrival_hour", "day_of_week", "month", "day_of_month", "days_from_holiday",
    # flight
    "distance_miles", "scheduled_elapsed_minutes",
    # aircraft rotation
    "aircraft_leg_number", "aircraft_legs_that_day", "turnaround_minutes",
    "aircraft_minutes_since_first_departure",
    # crowding
    "origin_departures_that_hour", "destination_arrivals_that_hour",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

MAX_TURNAROUND_MINUTES = 24 * 60

# Shortest turnaround an airline really plans. In the data, turnarounds of 30+ minutes
# are common (270,000 flights at 30-34 minutes) and anything shorter is rare and
# mostly late (20,000 flights at 25-29 minutes, 43% late; negative ones, 92% late).
# So a turnaround under 30 minutes means the aircraft was swapped in on the day.
MIN_PLANNED_TURNAROUND_MINUTES = 30

_ROTATION_NUMBERS = [
    "turnaround_minutes", "aircraft_leg_number", "aircraft_legs_that_day",
    "aircraft_minutes_since_first_departure",
]

# Airports on the other side of the International Date Line (Guam, Saipan).
# Their local clock is almost a full day ahead of the rest of the U.S.
DATE_LINE_AIRPORTS = {"GUM", "SPN"}


def _hhmm_to_minutes(hhmm: pd.Series) -> pd.Series:
    """1435 (2:35 PM) -> 875 minutes after midnight."""
    hhmm = hhmm.astype("int32")
    return (hhmm // 100) * 60 + hhmm % 100


def add_schedule_times(df: pd.DataFrame) -> pd.DataFrame:
    """Add real date-times for the scheduled departure and arrival, and repair bad flight times.

    Departure and arrival times are local, so a flight from New York to Chicago
    "gains" an hour. The time-zone difference is worked out from the schedule
    itself: arrival time - departure time - flight time. It must be a whole
    number of hours. In the few rows where it is not, the scheduled flight time
    is wrong, and it is replaced with the usual value for that route.
    """
    df = df.copy()
    origin = df["origin"].astype(str)
    destination = df["destination"].astype(str)
    dep_minutes = _hhmm_to_minutes(df["scheduled_departure_hhmm"])
    arr_minutes = _hhmm_to_minutes(df["scheduled_arrival_hhmm"]) % 1440
    elapsed = df["scheduled_elapsed_minutes"].astype("float64")

    raw = (arr_minutes - dep_minutes - elapsed) % 1440
    zone_shift = raw.where(raw < 720, raw - 1440)          # between -12 and +12 hours
    too_long = elapsed > 270 + df["distance_miles"] / 8     # hours longer than the distance allows
    consistent = elapsed.gt(0) & ~too_long & (zone_shift % 60 == 0) & (zone_shift.abs() <= 360)

    if not consistent.all():
        good = pd.DataFrame({"origin": origin, "destination": destination,
                             "elapsed": elapsed, "zone_shift": zone_shift})[consistent]
        by_route = good.groupby(["origin", "destination"])
        usual_elapsed = by_route["elapsed"].median()
        usual_shift = by_route["zone_shift"].agg(lambda s: s.mode().iloc[0])

        bad = ~consistent
        routes = pd.MultiIndex.from_arrays([origin[bad], destination[bad]])
        fallback_elapsed = 30 + df.loc[bad, "distance_miles"].astype("float64") / 8   # about 480 mph plus taxi time
        elapsed[bad] = pd.Series(usual_elapsed.reindex(routes).to_numpy(), index=elapsed[bad].index).fillna(fallback_elapsed)
        zone_shift[bad] = pd.Series(usual_shift.reindex(routes).to_numpy(), index=zone_shift[bad].index).fillna(0)

    # Flights across the date line: the true shift is about 20 hours, not 4.
    far_origin = origin.isin(DATE_LINE_AIRPORTS)
    far_destination = destination.isin(DATE_LINE_AIRPORTS)
    zone_shift = zone_shift.mask(far_destination & ~far_origin & (zone_shift < 0), zone_shift + 1440)
    zone_shift = zone_shift.mask(far_origin & ~far_destination & (zone_shift > 0), zone_shift - 1440)

    df["scheduled_elapsed_minutes"] = elapsed.astype("float32")
    df["scheduled_departure"] = df["flight_date"] + pd.to_timedelta(dep_minutes, unit="m")
    df["scheduled_arrival"] = df["scheduled_departure"] + pd.to_timedelta(elapsed + zone_shift, unit="m")
    df.attrs["repaired_flight_times"] = int((~consistent).sum())
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, weekday, month, day of month, and distance in days from the nearest federal holiday."""
    df = df.copy()
    date = df["flight_date"]
    df["departure_hour"] = df["scheduled_departure"].dt.hour.astype("int8")
    df["arrival_hour"] = df["scheduled_arrival"].dt.hour.astype("int8")
    df["day_of_week"] = date.dt.day_of_week.astype("int8")      # 0 = Monday
    df["month"] = date.dt.month.astype("int8")
    df["day_of_month"] = date.dt.day.astype("int8")

    holidays = USFederalHolidayCalendar().holidays(
        start=date.min() - pd.Timedelta(days=30), end=date.max() + pd.Timedelta(days=30)
    )
    days = pd.Series(date.unique()).sort_values()
    holiday_days = holidays.to_numpy().astype("datetime64[D]").astype("int64")
    day_numbers = days.to_numpy().astype("datetime64[D]").astype("int64")
    nearest = np.abs(day_numbers[:, None] - holiday_days[None, :]).min(axis=1)
    df["days_from_holiday"] = date.map(dict(zip(days, np.minimum(nearest, 14)))).astype("int8")
    return df


def _most_common(frame: pd.DataFrame, keys: list[str], column: str) -> pd.Series:
    """The most common value of `column` for each combination of `keys`."""
    counts = frame.groupby(keys + [column]).size().reset_index(name="n")
    top = counts.sort_values("n", ascending=False).drop_duplicates(keys)
    return top.set_index(keys)[column]


def add_rotation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Follow each aircraft through its scheduled day.

    For every flight, look at the same aircraft's previous scheduled flight:
      - aircraft_leg_number:  1 for the plane's first flight of the day, 2 for the next, ...
      - aircraft_legs_that_day:  how many flights the plane is scheduled for that day
      - previous_leg_origin:  where the previous flight came from
      - turnaround_minutes:  scheduled minutes on the ground between the previous
        flight's arrival and this departure (at most 24 hours)
      - aircraft_minutes_since_first_departure:  how long the plane has been in
        service that day

    Guard against leakage: the recorded aircraft is the one that actually flew. If its
    previous flight was scheduled to land somewhere else, or less than 30 minutes before
    this departure (or after it), the aircraft was swapped in on the day, usually because
    of a delay. We don't know the original plan for those flights, so their rotation
    features are replaced with the usual values for the same flight (same airline,
    flight number and route) on days when the plan is known. `aircraft_plan_known`
    marks which flights these are; it is for analysis only and must not be a model input.

    Needs the columns from add_schedule_times() and add_time_features().
    """
    df = df.sort_values(["tail_number", "scheduled_departure"], kind="stable").copy()
    tail = df["tail_number"].cat.codes                      # -1 when the tail number is missing
    has_tail = tail >= 0
    has_previous = has_tail & (tail == tail.shift())

    df["_carrier"] = df["carrier"].astype(str)
    df["_origin"] = df["origin"].astype(str)
    df["_destination"] = df["destination"].astype(str)
    previous_destination = df["_destination"].shift()
    turnaround = (df["scheduled_departure"] - df["scheduled_arrival"].shift()).dt.total_seconds() / 60
    plan_known = (
        has_previous
        & (previous_destination == df["_origin"])
        & (turnaround >= MIN_PLANNED_TURNAROUND_MINUTES)
    )

    day = [tail, df["flight_date"]]
    first_departure = df.groupby(day)["scheduled_departure"].transform("min")
    df["turnaround_minutes"] = turnaround.clip(upper=MAX_TURNAROUND_MINUTES)
    df["aircraft_leg_number"] = (df.groupby(day).cumcount() + 1).astype("float64")
    df["aircraft_legs_that_day"] = df.groupby(day)["flight_date"].transform("size").astype("float64")
    df["aircraft_minutes_since_first_departure"] = (df["scheduled_departure"] - first_departure).dt.total_seconds() / 60
    df["_previous_origin"] = df["_origin"].shift().astype(object)

    # Replace the rotation features of swapped-in aircraft with the flight's usual pattern.
    # Try the same flight first, then the same airline, airport and hour, then the overall usual value.
    known = df[plan_known]
    unknown = ~plan_known
    df.loc[unknown, _ROTATION_NUMBERS] = np.nan
    df.loc[unknown, "_previous_origin"] = None
    for keys in (["_carrier", "flight_number", "_origin", "_destination"],
                 ["_carrier", "_origin", "departure_hour"]):
        todo = unknown & df["turnaround_minutes"].isna()
        index = pd.MultiIndex.from_frame(df.loc[todo, keys])
        usual_numbers = known.groupby(keys)[_ROTATION_NUMBERS].median()
        df.loc[todo, _ROTATION_NUMBERS] = usual_numbers.reindex(index).to_numpy()
        usual_origin = _most_common(known, keys, "_previous_origin")
        df.loc[todo, "_previous_origin"] = usual_origin.reindex(index).to_numpy()
    df[_ROTATION_NUMBERS] = df[_ROTATION_NUMBERS].fillna(known[_ROTATION_NUMBERS].median())
    df["_previous_origin"] = df["_previous_origin"].fillna(df["_destination"])   # assume an out-and-back trip

    # Usual values are medians; round them so they look like ordinary schedule values.
    df[_ROTATION_NUMBERS] = df[_ROTATION_NUMBERS].round()
    df["aircraft_legs_that_day"] = df[["aircraft_legs_that_day", "aircraft_leg_number"]].max(axis=1)

    df["previous_leg_origin"] = pd.Categorical(df["_previous_origin"], categories=sorted(df["_origin"].unique()))
    df["aircraft_plan_known"] = plan_known.astype("int8")
    df["turnaround_minutes"] = df["turnaround_minutes"].astype("float32")
    df["aircraft_minutes_since_first_departure"] = df["aircraft_minutes_since_first_departure"].astype("float32")
    df["aircraft_leg_number"] = df["aircraft_leg_number"].astype("int8")
    df["aircraft_legs_that_day"] = df["aircraft_legs_that_day"].astype("int8")
    return df.drop(columns=[c for c in df.columns if c.startswith("_")]).sort_index()


def add_crowding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count the flights scheduled at the same airport in the same hour."""
    df = df.copy()
    departure_slot = df["scheduled_departure"].dt.floor("h")
    arrival_slot = df["scheduled_arrival"].dt.floor("h")
    df["origin_departures_that_hour"] = (
        df.groupby([df["origin"], departure_slot], observed=True)["origin"].transform("size").astype("int16")
    )
    df["destination_arrivals_that_hour"] = (
        df.groupby([df["destination"], arrival_slot], observed=True)["destination"].transform("size").astype("int16")
    )
    return df


def build_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Build the feature table from every scheduled flight; return one row per completed flight.

    `schedule` must include cancelled and diverted flights (load_flights(completed_only=False)),
    because they were part of each aircraft's planned day.
    """
    df = add_schedule_times(schedule)
    repaired = df.attrs["repaired_flight_times"]
    df = add_time_features(df)
    df = add_rotation_features(df)
    df = add_crowding_features(df)

    completed = (df["cancelled"] == 0) & (df["diverted"] == 0) & df[TARGET].notna()
    df = df.loc[completed, ID_COLUMNS + FEATURE_COLUMNS + DIAGNOSTIC_COLUMNS + [TARGET]].reset_index(drop=True)
    df[TARGET] = df[TARGET].astype("int8")
    df.attrs["repaired_flight_times"] = repaired
    return df


def load_features(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the saved feature table (one row per completed flight)."""
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(f"{FEATURES_PARQUET} not found. Run src/features.py first.")
    return pd.read_parquet(FEATURES_PARQUET, columns=columns)


def main() -> int:
    start = time.time()
    schedule = load_flights(completed_only=False)
    print(f"Loaded {len(schedule):,} scheduled flights")
    features = build_features(schedule)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PARQUET, index=False)

    swapped = 1 - features["aircraft_plan_known"].mean()
    print(f"Repaired scheduled flight times: {features.attrs['repaired_flight_times']}")
    print(f"Flights with a swapped-in aircraft (rotation features set to usual values): {swapped:.1%}")
    print(f"Feature table: {len(features):,} completed flights, {len(FEATURE_COLUMNS)} features")
    print(f"Memory in pandas: {features.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(f"Parquet on disk:  {FEATURES_PARQUET.stat().st_size / 1e6:.0f} MB  ({FEATURES_PARQUET})")
    print(f"Done in {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
