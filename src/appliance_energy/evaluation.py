"""
evaluation.py
=============
Common accuracy metrics used to compare every model on the same test
period (Part 8): MAE, RMSE, MASE and Bias.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def bias(y_true, y_pred) -> float:
    """Mean signed error. Positive => model over-forecasts on average."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def mase(y_true, y_pred, y_train, seasonality: int = 24) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Scales MAE by the in-sample mean absolute error of a seasonal naive
    forecast, so a value < 1 means the model beats seasonal naive on the
    training period and a value > 1 means it does worse.
    """
    y_train = pd.Series(y_train).astype(float)

    seasonal_errors = np.abs(
        y_train.iloc[seasonality:].values - y_train.iloc[:-seasonality].values
    )
    scale = seasonal_errors.mean()

    if scale == 0 or np.isnan(scale):
        return np.nan

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_forecast(name: str, y_true: pd.Series, y_pred: pd.Series,
                       y_train: pd.Series, seasonality: int = 24) -> dict:
    """
    Evaluate a single forecast, aligning y_pred to y_true's index and
    dropping any points where either is missing (e.g. feature-model
    forecasts that need warm-up lags).
    """
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).reindex(y_true.index).astype(float)

    valid = y_true.notna() & y_pred.notna()
    y_true_v = y_true.loc[valid]
    y_pred_v = y_pred.loc[valid]

    return {
        "model": name,
        "n_points": int(valid.sum()),
        "MAE": mae(y_true_v, y_pred_v),
        "RMSE": rmse(y_true_v, y_pred_v),
        "MASE": mase(y_true_v, y_pred_v, y_train, seasonality=seasonality),
        "Bias": bias(y_true_v, y_pred_v),
    }


def evaluate_all(forecasts: dict, y_true: pd.Series, y_train: pd.Series,
                  seasonality: int = 24) -> pd.DataFrame:
    """
    Evaluate a dict of {model_name: forecast_series} and return a results
    table sorted by MASE (ascending = better), with the strongest benchmark
    flagged for comparison in the report (Part 9, Q1/Q2).
    """
    results = [
        evaluate_forecast(name, y_true, pred, y_train, seasonality=seasonality)
        for name, pred in forecasts.items()
    ]
    df = pd.DataFrame(results).sort_values("MASE").reset_index(drop=True)
    return df
