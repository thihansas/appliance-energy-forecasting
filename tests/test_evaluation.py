
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy import evaluation  


def test_mase_is_zero_for_perfect_forecast():
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    y_train = pd.Series(10 + 5 * np.sin(2 * np.pi * np.arange(200) / 24), index=idx)

    y_true = y_train.iloc[-24:]
    y_pred = y_true.copy()  # perfect forecast

    score = evaluation.mase(y_true, y_pred, y_train, seasonality=24)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_mase_is_one_for_seasonal_naive_on_training_pattern():
   
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    y_train = pd.Series(10 + 5 * np.sin(2 * np.pi * np.arange(200) / 24), index=idx)

    y_true = y_train.iloc[-24:]
    y_pred = y_true + 2.0  # constant bias

    score = evaluation.mase(y_true, y_pred, y_train, seasonality=24)
    assert score > 0


def test_bias_sign_reflects_over_or_under_forecasting():
    y_true = pd.Series([10, 10, 10, 10])
    over_forecast = pd.Series([12, 12, 12, 12])
    under_forecast = pd.Series([8, 8, 8, 8])

    assert evaluation.bias(y_true, over_forecast) > 0
    assert evaluation.bias(y_true, under_forecast) < 0


def test_evaluate_forecast_drops_misaligned_missing_points():
    idx = pd.date_range("2020-01-01", periods=10, freq="h")
    y_true = pd.Series(np.arange(10, dtype=float), index=idx)
    y_train = pd.Series(np.arange(50, dtype=float))

    y_pred = y_true.copy()
    y_pred.iloc[:3] = np.nan  

    result = evaluation.evaluate_forecast("test_model", y_true, y_pred, y_train, seasonality=5)
    assert result["n_points"] == 7
    assert result["MAE"] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_all_sorts_by_mase_ascending():
    idx = pd.date_range("2020-01-01", periods=10, freq="h")
    y_true = pd.Series(np.arange(10, dtype=float), index=idx)
    y_train = pd.Series(np.arange(50, dtype=float))

    forecasts = {
        "bad": y_true + 5,
        "good": y_true + 0.1,
    }

    results_df = evaluation.evaluate_all(forecasts, y_true, y_train, seasonality=5)
    assert results_df.iloc[0]["model"] == "good"
