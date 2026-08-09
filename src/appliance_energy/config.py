"""
config.py
=========
Central configuration for the appliance-energy-forecasting project.

Keeping paths and constants in one place means every module and script
(pipeline.py, notebooks, tests) refers to the same values, so changing the
test window, the seasonal periods, or a directory only has to happen once.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "model_objects"

for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
             FORECAST_DIR, METRICS_DIR, FIGURE_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Data source
# ------------------------------------------------------------------

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)

RAW_FILENAME = "energydata_complete.csv"
HOURLY_FILENAME = "appliance_hourly.csv"

# ------------------------------------------------------------------
# Modelling constants
# ------------------------------------------------------------------

TARGET = "Appliances"

# Hourly data -> 24 obs/day, 168 obs/week
DAILY_PERIOD = 24
WEEKLY_PERIOD = 168

# Final 14 days held out as the test set, forecast horizon = 24h
TEST_DAYS = 14
TEST_STEPS = TEST_DAYS * DAILY_PERIOD
HORIZON = DAILY_PERIOD

RANDOM_STATE = 0

# Candidate exogenous / covariate columns available in the raw dataset
CANDIDATE_EXOG_COLS = [
    "T_out",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
    "Press_mm_hg",
]

# SARIMAX grid-search ranges (assignment: p in [0,6], d in [0,2], q in [0,6])
SARIMAX_P_RANGE = range(0, 7)
SARIMAX_D_RANGE = range(0, 3)
SARIMAX_Q_RANGE = range(0, 7)

# Seasonal order kept fixed (daily seasonality) to keep the grid search
# tractable; students can extend this to loop over (P, D, Q) too.
SARIMAX_SEASONAL_ORDER = (1, 1, 1, DAILY_PERIOD)

# A reduced grid used by default (full grid is 7*3*7=147 fits and is slow -
# see models/sarimax.py:grid_search_sarimax(quick=False) for the full grid)
QUICK_P_RANGE = range(0, 3)
QUICK_D_RANGE = range(0, 2)
QUICK_Q_RANGE = range(0, 3)
