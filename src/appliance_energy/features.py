"""
features.py
============
Feature engineering for the feature-based model and for SARIMAX exogenous
variables (Part 5 of the assignment).

Rule followed throughout: any feature derived from the target variable is
built on `.shift(1)` first, so a feature for time t never uses the value of
Appliances at time t or later. This avoids the data-leakage failure mode
called out explicitly in the README ("Using future values of Appliances in
lag or rolling features").
"""

import numpy as np
import pandas as pd

from . import config

DEFAULT_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
DEFAULT_WINDOWS = [3, 6, 12, 24, 168]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical time-of-day and day-of-week features. These are known at the
    forecast origin for any future timestamp, so they never leak information
    from the target.
    """
    out = df.copy()

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    return out


def add_lag_features(df: pd.DataFrame, target: str = config.TARGET,
                      lags=None) -> pd.DataFrame:
    out = df.copy()
    lags = lags or DEFAULT_LAGS
    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, target: str = config.TARGET,
                          windows=None) -> pd.DataFrame:
    out = df.copy()
    windows = windows or DEFAULT_WINDOWS
    shifted = out[target].shift(1)  # never include the current observation
    for window in windows:
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()
    return out


def make_ml_table(df: pd.DataFrame, target: str = config.TARGET,
                   lags=None, windows=None, dropna_target: bool = True) -> pd.DataFrame:
    """
    Build the full supervised-learning feature table used by the
    feature-based model (Part 5 + Part 6): original sensor/weather columns,
    time features, lag features and rolling features, with rows containing
    NaNs (from the longest lag/window) dropped.
    """
    out = add_time_features(df)
    out = add_lag_features(out, target=target, lags=lags)
    out = add_rolling_features(out, target=target, windows=windows)
    feature_cols = [c for c in out.columns if c != target]
    if dropna_target:
        return out.dropna()
    return out.dropna(subset=feature_cols)


def get_feature_columns(ml_table: pd.DataFrame, target: str = config.TARGET) -> list:
    return [c for c in ml_table.columns if c != target]


def get_exog_columns(df: pd.DataFrame) -> list:
    """
    Build the exogenous feature set for SARIMAX: candidate weather columns
    that are actually present in the data, plus cyclical time features.
    """
    exog_df = add_time_features(df)
    weather_cols = [c for c in config.CANDIDATE_EXOG_COLS if c in df.columns]
    time_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    return exog_df[weather_cols + time_cols]
