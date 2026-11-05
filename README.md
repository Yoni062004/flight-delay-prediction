# Predicting U.S. Flight Delays

Can we tell, before a flight leaves, whether it will arrive 15 or more minutes late, and explain why?

This project trains a LightGBM model on a year of U.S. Bureau of Transportation Statistics on-time data. It uses only information that's known in advance: the schedule, airline, route, and the aircraft's rotation that day. SHAP explains which factors drive each prediction.

> Work in progress. The results section will be filled in as the project develops.

## Data
- **Source:** [BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/), U.S. domestic flights
- **Period:** 12 consecutive months (to be confirmed)
- **Target:** `ARR_DEL15` (1 = arrived 15+ minutes late). Cancelled and diverted flights are excluded.

## Approach
1. Clean the monthly files and convert them to Parquet.
2. Explore delay patterns by airline, airport, time of day, and day of week.
3. Engineer features using only pre-departure information, including aircraft-rotation features from the tail number.
4. Split by date: train on earlier months and test on the most recent months.
5. Compare against baselines (always "on time", then logistic regression), and train LightGBM.
6. Evaluate with ROC-AUC and PR-AUC, not accuracy, because about 80% of flights are on time.
7. Explain the model with SHAP.

## Results
_To be added._

## Project structure
```
data/raw/           monthly BTS downloads (not tracked)
data/processed/     cleaned Parquet (not tracked)
notebooks/          analysis notebooks, in order
src/                reusable Python code
models/             saved models (not tracked)
reports/figures/    charts
```

## How to run
```bash
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
Download the monthly CSVs from BTS into `data/raw/`, then run the notebooks in order.
