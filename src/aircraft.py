"""Look up every aircraft in the FAA aircraft registry: what model it is and when it was built.

The FAA publishes its registry as a free download. Each U.S. aircraft has a
registration (its "N-number"), which is the tail number in the flight data.

Privacy: the registry also holds owners' names and addresses. This file never
reads those columns. It loads only the registration, the model code and the
year the aircraft was built.

What it does:
  1. Downloads ReleasableAircraft.zip into data/raw/faa_registry/ (once).
  2. Matches each tail number in the flight data to a registry record.
  3. Groups the many model codes into families (737-800, A321neo, E175, ...).
  4. Saves data/processed/aircraft.parquet, one row per tail number.

Build the aircraft table (from the project folder):
    .venv\\Scripts\\python src\\aircraft.py
"""

from __future__ import annotations

import re
import sys
import zipfile

import pandas as pd
import requests

from load import FLIGHTS_PARQUET, PROCESSED_DIR, RAW_DIR

REGISTRY_URL = "https://registry.faa.gov/database/ReleasableAircraft.zip"
REGISTRY_ZIP = RAW_DIR / "faa_registry" / "ReleasableAircraft.zip"
AIRCRAFT_PARQUET = PROCESSED_DIR / "aircraft.parquet"

UNKNOWN_FAMILY = "Unknown"
MIN_AIRLINER_SEATS = 30

# Aircraft removed from the registry before the flight data starts cannot have flown these flights.
EARLIEST_DEREGISTRATION = "20250901"

# The registry lists hundreds of model codes (Boeing gives each airline its own code,
# such as 737-7H4 for Southwest). These rules group them into the families people know.
# The first matching rule wins.
_FAMILY_RULES = [
    (r"^737-(7|8|9|10)$", r"737 MAX \1"),
    (r"^737-8200$", "737 MAX 8"),
    (r"^737-9\w*ER$", "737-900ER"),
    (r"^737-([6-9])\w+$", r"737-\g<1>00"),
    (r"^717-", "717"),
    (r"^787-(\d+)", r"787-\1"),
    (r"^7([5-7])7-(\d)", r"7\g<1>7-\g<2>00"),
    (r"^A(318|319|320|321)-\d+NX?$", r"A\1neo"),
    (r"^A(318|319|320|321)-", r"A\1"),
    (r"^A(330|350)-", r"A\1"),
    (r"^BD-500-1A10", "A220-100"),
    (r"^BD-500-1A11", "A220-300"),
    (r"^CL-600-2B19", "CRJ200"),
    (r"^CL-600-2C10", "CRJ700"),
    (r"^CL-600-2C11", "CRJ550"),
    (r"^CL-600-2D", "CRJ900"),
    (r"^ERJ 170-1", "E170"),
    (r"^ERJ 170-2", "E175"),
    (r"^ERJ 190-1", "E190"),
    (r"^ERJ 190-2", "E195"),
    (r"^EMB-145", "ERJ145"),
    (r"^EMB-135", "ERJ135"),
    (r"^ATR", "ATR"),
    (r"^DHC-8", "Dash 8"),
]


def aircraft_family(model: str) -> str:
    """'737-7H4' -> '737-700', 'A321-271NX' -> 'A321neo', 'ERJ 170-200 LR' -> 'E175'."""
    model = model.strip().upper()
    for pattern, family in _FAMILY_RULES:
        if re.search(pattern, model):
            return re.sub(pattern + ".*", family, model)
    return model.title()


def download_registry(force: bool = False) -> None:
    """Download the FAA registry ZIP (about 73 MB) unless it is already on disk."""
    if REGISTRY_ZIP.exists() and not force:
        return
    REGISTRY_ZIP.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REGISTRY_URL} ...", flush=True)
    headers = {"User-Agent": "Mozilla/5.0 (flight-delay-prediction portfolio project)"}
    with requests.get(REGISTRY_URL, headers=headers, stream=True, timeout=300) as response:
        response.raise_for_status()
        partial = REGISTRY_ZIP.with_suffix(".zip.part")
        with partial.open("wb") as out:
            for chunk in response.iter_content(1 << 20):
                out.write(chunk)
    if not zipfile.is_zipfile(partial):
        raise RuntimeError("The FAA download is not a valid ZIP file")
    partial.replace(REGISTRY_ZIP)


def _read(archive: zipfile.ZipFile, name: str, columns: dict[str, str]) -> pd.DataFrame:
    """Read only the listed columns of one registry file, with spaces trimmed."""
    with archive.open(name) as file:
        frame = pd.read_csv(file, usecols=list(columns), dtype=str, encoding="utf-8-sig")
    return frame.rename(columns=columns).apply(lambda s: s.str.strip())


def read_registry() -> pd.DataFrame:
    """Return the airliners in the registry: registration, model, year built, and where the record is from.

    Current registrations come first; aircraft deregistered since the flight data
    starts are included too, because they may have flown early in the period.
    """
    with zipfile.ZipFile(REGISTRY_ZIP) as archive:
        current = _read(archive, "MASTER.txt", {
            "N-NUMBER": "n_number", "MFR MDL CODE": "model_code", "YEAR MFR": "year_built"})
        removed = _read(archive, "DEREG.txt", {
            "N-NUMBER": "n_number", "MFR-MDL-CODE": "model_code", "YEAR-MFR": "year_built",
            "CANCEL-DATE": "cancel_date"})
        models = _read(archive, "ACFTREF.txt", {
            "CODE": "model_code", "MFR": "manufacturer", "MODEL": "model",
            "TYPE-ACFT": "aircraft_type", "NO-SEATS": "seats"})

    current["registry_source"] = "current"
    current["cancel_date"] = ""
    removed = removed[removed["cancel_date"] >= EARLIEST_DEREGISTRATION].copy()
    removed["registry_source"] = "deregistered"

    records = pd.concat([current, removed], ignore_index=True).merge(models, on="model_code", how="inner")
    records["registration"] = "N" + records["n_number"]
    seats = pd.to_numeric(records["seats"], errors="coerce")
    # Type 5 is "fixed wing, multi-engine". This drops small planes, helicopters and drones
    # that now carry a registration an airliner used to have.
    airliners = records[(records["aircraft_type"] == "5") & (seats >= MIN_AIRLINER_SEATS)].copy()
    airliners["year_built"] = pd.to_numeric(airliners["year_built"], errors="coerce")
    return airliners[["registration", "manufacturer", "model", "year_built", "registry_source", "cancel_date"]]


def build_aircraft_table() -> pd.DataFrame:
    """Match every tail number in the flight data to the registry; save data/processed/aircraft.parquet."""
    download_registry()
    flights = pd.read_parquet(FLIGHTS_PARQUET, columns=["tail_number", "carrier"])
    flights = flights[flights["tail_number"].notna()]
    tails = (
        flights.astype(str).groupby("tail_number")["carrier"]
        .agg(flights="size", carrier=lambda s: s.mode().iloc[0]).reset_index()
    )

    # Two quirks in how airlines report tail numbers:
    #   - Allegiant leaves off the leading "N" (816NV is N816NV).
    #   - From January to April 2026, Hawaiian's aircraft appear under Alaska with an "AS"
    #     ending instead of "HA" (N202AS is N202HA). The real N202AS is a small private plane.
    # So each tail gets up to two candidate registrations, tried in order.
    plain = tails["tail_number"].where(tails["tail_number"].str.startswith("N"), "N" + tails["tail_number"])
    candidates = pd.concat([
        pd.DataFrame({"tail_number": tails["tail_number"], "registration": plain, "attempt": 0}),
        pd.DataFrame({"tail_number": tails["tail_number"],
                      "registration": plain.str.replace(r"AS$", "HA", regex=True), "attempt": 1}),
    ]).drop_duplicates(["tail_number", "registration"])

    registry = read_registry()
    matches = candidates.merge(registry, on="registration", how="inner")
    matches["is_removed"] = (matches["registry_source"] == "deregistered").astype(int)
    best = (
        matches.sort_values(["tail_number", "attempt", "is_removed", "cancel_date"],
                            ascending=[True, True, True, False])
        .drop_duplicates("tail_number")
    )

    aircraft = tails.merge(
        best[["tail_number", "registration", "manufacturer", "model", "year_built", "registry_source"]],
        on="tail_number", how="left",
    )
    aircraft["registry_source"] = aircraft["registry_source"].fillna("not found")
    aircraft["aircraft_family"] = aircraft["model"].map(aircraft_family, na_action="ignore").fillna(UNKNOWN_FAMILY)

    # Some records have no build year. Use the usual year for the same airline and family,
    # then for the family, then overall.
    aircraft["year_built_estimated"] = aircraft["year_built"].isna()
    for keys in (["carrier", "aircraft_family"], ["aircraft_family"]):
        usual = aircraft.groupby(keys)["year_built"].transform("median")
        aircraft["year_built"] = aircraft["year_built"].fillna(usual)
    aircraft["year_built"] = aircraft["year_built"].fillna(aircraft["year_built"].median()).round().astype("int16")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    aircraft.to_parquet(AIRCRAFT_PARQUET, index=False)
    return aircraft


def load_aircraft() -> pd.DataFrame:
    """Load the aircraft table (one row per tail number in the flight data)."""
    if not AIRCRAFT_PARQUET.exists():
        raise FileNotFoundError(f"{AIRCRAFT_PARQUET} not found. Run src/aircraft.py first.")
    return pd.read_parquet(AIRCRAFT_PARQUET)


def main() -> int:
    aircraft = build_aircraft_table()
    found = aircraft["registry_source"] != "not found"
    print(f"Aircraft in the flight data: {len(aircraft):,}")
    print(f"Matched to the FAA registry: {found.sum():,} ({found.mean():.1%} of aircraft, "
          f"{aircraft.loc[found, 'flights'].sum() / aircraft['flights'].sum():.2%} of flights)")
    print(aircraft["registry_source"].value_counts().to_string())
    print(f"Build year estimated: {int(aircraft['year_built_estimated'].sum()):,} aircraft")
    print(f"Aircraft families: {aircraft['aircraft_family'].nunique()}")
    print(f"Saved {AIRCRAFT_PARQUET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
