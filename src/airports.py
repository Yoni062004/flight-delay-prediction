"""Look up every airport in the flight data: its name, city, state and position on the map.

The flight data only has three-letter airport codes (ATL, LGA, ...). OurAirports
publishes a free, public list of the world's airports. This file matches each
code in the flight data to that list, so charts and the Power BI map can show
real names and locations.

These details are for reporting. They are not model inputs: the model already
has the airport code itself.

What it does:
  1. Downloads airports.csv into data/raw/ourairports/ (once, about 13 MB).
  2. Matches each airport code in the flight data by its IATA code.
  3. Saves data/processed/airports.parquet, one row per airport.

Build the airport table (from the project folder):
    .venv\\Scripts\\python src\\airports.py
"""

from __future__ import annotations

import sys

import pandas as pd
import requests

from load import FLIGHTS_PARQUET, PROCESSED_DIR, RAW_DIR

AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
AIRPORTS_CSV = RAW_DIR / "ourairports" / "airports.csv"
AIRPORTS_PARQUET = PROCESSED_DIR / "airports.parquet"

# When several records share a code, prefer the bigger, active airport.
_SIZE_RANK = {"large_airport": 0, "medium_airport": 1, "small_airport": 2, "seaplane_base": 3, "heliport": 4, "closed": 9}
_SIZE_NAMES = {"large_airport": "Large", "medium_airport": "Medium", "small_airport": "Small"}


def download_airports(force: bool = False) -> None:
    """Download the OurAirports list unless it is already on disk."""
    if AIRPORTS_CSV.exists() and not force:
        return
    AIRPORTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {AIRPORTS_URL} ...", flush=True)
    headers = {"User-Agent": "Mozilla/5.0 (flight-delay-prediction portfolio project)"}
    response = requests.get(AIRPORTS_URL, headers=headers, timeout=120)
    response.raise_for_status()
    if not response.text.startswith('"id","ident"'):
        raise RuntimeError("The OurAirports download does not look like the airport list")
    AIRPORTS_CSV.write_bytes(response.content)


def build_airport_table() -> pd.DataFrame:
    """Match every airport code in the flight data to OurAirports; save data/processed/airports.parquet."""
    download_airports()
    flights = pd.read_parquet(FLIGHTS_PARQUET, columns=["origin", "destination"])
    departures = flights["origin"].astype(str).value_counts().rename("departures")
    arrivals = flights["destination"].astype(str).value_counts().rename("arrivals")
    airports = pd.concat([departures, arrivals], axis=1).fillna(0).astype("int32")
    airports = airports.rename_axis("airport").reset_index()

    listing = pd.read_csv(
        AIRPORTS_CSV,
        usecols=["ident", "type", "name", "latitude_deg", "longitude_deg", "iso_country", "iso_region",
                 "municipality", "scheduled_service", "iata_code"],
        keep_default_na=False,
    )
    listing["rank"] = listing["type"].map(_SIZE_RANK).fillna(5)
    listing["no_service"] = (listing["scheduled_service"] != "yes").astype(int)
    listing = listing.sort_values(["rank", "no_service"])

    # First match on the IATA code. An airport that changed its code keeps its old
    # four-letter identifier ("K" + the old code in the U.S.), so try that second.
    wanted = ["type", "name", "latitude_deg", "longitude_deg", "iso_country", "iso_region", "municipality"]
    by_iata = listing[listing["iata_code"] != ""].drop_duplicates("iata_code").set_index("iata_code")[wanted]
    by_ident = listing.drop_duplicates("ident").set_index("ident")[wanted]
    details = by_iata.reindex(airports["airport"])
    missing = details["name"].isna().to_numpy()
    details.iloc[missing] = by_ident.reindex("K" + airports.loc[missing, "airport"]).to_numpy()

    table = pd.concat([airports, details.reset_index(drop=True)], axis=1)
    table["found"] = table["name"].notna()
    # U.S. states come from the region code (US-GA -> GA); territories use their own code (PR, GU, ...).
    table["state"] = table["iso_region"].str.split("-").str[1].where(table["iso_country"] == "US", table["iso_country"])
    table["size"] = table["type"].map(_SIZE_NAMES).fillna("Other")
    table = table.rename(columns={
        "name": "airport_name", "municipality": "city", "iso_country": "country",
        "latitude_deg": "latitude", "longitude_deg": "longitude",
    })
    columns = ["airport", "airport_name", "city", "state", "country", "latitude", "longitude", "size",
               "departures", "arrivals", "found"]
    table = table[columns].sort_values("departures", ascending=False).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(AIRPORTS_PARQUET, index=False)
    return table


def load_airports() -> pd.DataFrame:
    """Load the airport table (one row per airport in the flight data)."""
    if not AIRPORTS_PARQUET.exists():
        raise FileNotFoundError(f"{AIRPORTS_PARQUET} not found. Run src/airports.py first.")
    return pd.read_parquet(AIRPORTS_PARQUET)


def main() -> int:
    table = build_airport_table()
    found = table["found"]
    print(f"Airports in the flight data: {len(table)}")
    print(f"Matched to OurAirports: {found.sum()} ({found.mean():.1%} of airports, "
          f"{table.loc[found, 'departures'].sum() / table['departures'].sum():.2%} of departures)")
    if not found.all():
        print("Not matched:", ", ".join(table.loc[~found, "airport"]))
    print(table["size"].value_counts().to_string())
    print(f"Saved {AIRPORTS_PARQUET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
