
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from . import config


def download_raw_data(force: bool = False) -> pd.DataFrame:
   
    raw_path = config.RAW_DIR / config.RAW_FILENAME

    if raw_path.exists() and not force:
        print(f"Using cached raw data: {raw_path}")
        return pd.read_csv(raw_path)

    print(f"Downloading data from {config.DATA_URL} ...")
    df = pd.read_csv(config.DATA_URL)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path} (shape={df.shape})")
    return df


def clean_and_resample(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_missing_target = df[config.TARGET].isna().sum()
    if n_missing_target:
        print(f"Dropping {n_missing_target} rows with missing target")
    df = df.dropna(subset=[config.TARGET])

    resampled = df.resample(freq).mean()

    n_gap_rows = resampled[config.TARGET].isna().sum()
    if n_gap_rows:
        print(f"Interpolating {n_gap_rows} hourly gaps created by resampling")

    resampled = resampled.interpolate("time").dropna()

    return resampled


def load_appliance_data(force_download: bool = False) -> pd.DataFrame:
    
    processed_path = config.PROCESSED_DIR / config.HOURLY_FILENAME

    if processed_path.exists() and not force_download:
        print(f"Loading cached processed data: {processed_path}")
        hourly = pd.read_csv(processed_path, index_col=0, parse_dates=True)
        return hourly

    raw = download_raw_data(force=force_download)
    hourly = clean_and_resample(raw, freq="h")

    hourly.to_csv(processed_path)
    print(f"Saved processed hourly data: {processed_path} (shape={hourly.shape})")

    return hourly

# Stationarity testing

def adf_test(series: pd.Series, name: str = "") -> dict:
    
    result = adfuller(series.dropna(), autolag="AIC")
    out = {
        "series": name,
        "test": "ADF",
        "statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "stationary_at_5pct": result[1] < 0.05,
    }
    return out


def kpss_test(series: pd.Series, name: str = "", regression: str = "c") -> dict:
  
    statistic, p_value, n_lags, crit = kpss(
        series.dropna(), regression=regression, nlags="auto"
    )
    out = {
        "series": name,
        "test": "KPSS",
        "statistic": statistic,
        "p_value": p_value,
        "n_lags": n_lags,
        "n_obs": len(series.dropna()),
        "stationary_at_5pct": p_value > 0.05,
    }
    return out


def stationarity_report(series: pd.Series, name: str = "Appliances") -> pd.DataFrame:
  
    rows = [adf_test(series, name), kpss_test(series, name)]

    diffed = series.diff().dropna()
    rows.append(adf_test(diffed, f"{name} (1st diff)"))
    rows.append(kpss_test(diffed, f"{name} (1st diff)"))

    return pd.DataFrame(rows)
