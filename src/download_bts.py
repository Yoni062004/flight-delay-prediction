"""Download monthly BTS "Reporting Carrier On-Time Performance" files.

What it does, for each requested month:
  1. Opens the BTS download page (transtats.bts.gov) to get a fresh form.
  2. Ticks the same 57 fields as the October 2025 file, so every month has identical columns.
  3. Submits the form, receives a ZIP, and saves the CSV inside it as data/raw/ontime_YYYY_MM.csv.
  4. Checks the CSV: correct header, and every row has 57 values (catches cut-off downloads).

Usage (from the project folder):
    .venv\\Scripts\\python src\\download_bts.py 2025-09
    .venv\\Scripts\\python src\\download_bts.py 2025-09 2025-11 2025-12 ...
Months that already have a valid file in data/raw/ are skipped. Add --force to re-download them.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import zipfile
from html import unescape
from pathlib import Path

import requests

PAGE_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Same fields, same order, as the October 2025 file (T_ONTIME_REPORTING.csv).
FIELDS = [
    "YEAR", "QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "FL_DATE",
    "OP_UNIQUE_CARRIER", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM",
    "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_CITY_MARKET_ID", "ORIGIN",
    "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "ORIGIN_STATE_NM",
    "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_CITY_MARKET_ID", "DEST",
    "DEST_CITY_NAME", "DEST_STATE_ABR", "DEST_STATE_NM",
    "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY", "DEP_DELAY_NEW", "DEP_DEL15", "DEP_DELAY_GROUP",
    "DEP_TIME_BLK", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY", "ARR_DELAY_NEW", "ARR_DEL15", "ARR_DELAY_GROUP",
    "ARR_TIME_BLK", "CANCELLED", "CANCELLATION_CODE", "DIVERTED",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "FLIGHTS", "DISTANCE",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "DIV_AIRPORT_LANDINGS",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (flight-delay-prediction portfolio project)"}


def hidden_fields(html: str) -> dict[str, str]:
    """Read the ASP.NET hidden form values (__VIEWSTATE etc.) the server expects back."""
    fields = {}
    for tag in re.findall(r"<input[^>]*type=\"hidden\"[^>]*>", html):
        name = re.search(r'name="([^"]*)"', tag)
        value = re.search(r'value="([^"]*)"', tag)
        if name:
            fields[name.group(1)] = unescape(value.group(1)) if value else ""
    return fields


def check_csv(path: Path) -> int:
    """Return the number of data rows; raise if the header or any row is wrong."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        # BTS sometimes adds an empty column at the end because of a trailing comma.
        if header and header[-1] == "":
            header = header[:-1]
        if header != FIELDS:
            raise ValueError(f"{path.name}: header doesn't match the October columns: {header}")
        rows = 0
        for row in reader:
            if row and row[-1] == "" and len(row) == len(FIELDS) + 1:
                row = row[:-1]
            if len(row) != len(FIELDS):
                raise ValueError(f"{path.name}: row {rows + 2} has {len(row)} values, expected {len(FIELDS)}")
            rows += 1
    return rows


def download_month(year: int, month: int, force: bool = False) -> Path:
    out = RAW_DIR / f"ontime_{year}_{month:02d}.csv"
    if out.exists() and not force:
        try:
            rows = check_csv(out)
            print(f"{out.name}: already downloaded and valid ({rows:,} rows), skipping")
            return out
        except ValueError as e:
            print(f"{out.name}: existing file is bad ({e}); downloading again")

    session = requests.Session()
    session.headers.update(HEADERS)
    page = session.get(PAGE_URL, timeout=60)
    page.raise_for_status()

    form = hidden_fields(page.text)
    form.update({
        "cboGeography": "All",
        "cboYear": str(year),
        "cboPeriod": str(month),
        "btnDownload": "Download",
    })
    for field in FIELDS:
        form[field] = "on"

    print(f"{out.name}: requesting {year}-{month:02d} from BTS ...", flush=True)
    start = time.time()
    resp = session.post(PAGE_URL, data=form, timeout=900)
    resp.raise_for_status()
    if not resp.content.startswith(b"PK"):
        raise RuntimeError(
            f"BTS didn't send a ZIP for {year}-{month:02d} "
            f"(content-type {resp.headers.get('Content-Type')}, {len(resp.content):,} bytes)"
        )

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv") and "/" not in n and "\\" not in n]
        if len(csv_names) != 1:
            raise RuntimeError(f"Expected one CSV in the ZIP, found: {zf.namelist()}")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".csv.part")
        with zf.open(csv_names[0]) as src, tmp.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)

    rows = check_csv(tmp)
    tmp.replace(out)
    size_mb = out.stat().st_size / 1e6
    print(f"{out.name}: saved {rows:,} rows, {size_mb:.0f} MB, in {time.time() - start:.0f}s")
    return out


def parse_month(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", text)
    if not m or not 1 <= int(m.group(2)) <= 12:
        raise argparse.ArgumentTypeError(f"Use YYYY-MM, e.g. 2025-09 (got {text!r})")
    return int(m.group(1)), int(m.group(2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("months", nargs="+", type=parse_month, help="months as YYYY-MM")
    parser.add_argument("--force", action="store_true", help="re-download even if a valid file exists")
    args = parser.parse_args()

    failed = []
    for year, month in args.months:
        try:
            download_month(year, month, force=args.force)
        except Exception as e:  # keep going so one bad month doesn't stop the rest
            print(f"{year}-{month:02d}: FAILED: {e}", file=sys.stderr)
            failed.append(f"{year}-{month:02d}")
    if failed:
        print("Failed months: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
