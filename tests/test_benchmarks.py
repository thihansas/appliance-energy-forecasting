"""
tests/test_benchmarks.py
==========================
Tests for the benchmark forecasting functions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.models import benchmarks  # noqa: E402


@pytest.fixture
def toy_series():
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    # Simple deterministic daily seasonal pattern for predictable tests.
    values = 10 + 5 * np.sin(2 * np.pi * np.arange(200) / 24)
    return pd.Series(values, index=idx, name="Appliances")


def test_forecast_lengths_match_horizon(toy_series):
    horizon = 24
    future_index = pd.date_range(
        toy_series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )

    for forecast in [
        benchmarks.mean_forecast(toy_series, horizon, future_index),
        benchmarks.naive_forecast(toy_series, horizon, future_index),
        benchmarks.seasonal_naive_forecast(toy_series, horizon, future_index, seasonality=24),
        benchmarks.drift_forecast(toy_series, horizon, future_index),
    ]:
        assert len(forecast) == horizon


def test_naive_forecast_equals_last_value(toy_series):
    horizon = 5
    future_index = pd.date_range(
        toy_series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )
    forecast = benchmarks.naive_forecast(toy_series, horizon, future_index)
    assert (forecast == toy_series.iloc[-1]).all()


def test_seasonal_naive_matches_expected_lag(toy_series):
    horizon = 24
    future_index = pd.date_range(
        toy_series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )
    forecast = benchmarks.seasonal_naive_forecast(
        toy_series, horizon, future_index, seasonality=24
    )
    # First forecast step should equal the value 24 steps before the end of train.
    assert forecast.iloc[0] == pytest.approx(toy_series.iloc[-24])


def test_drift_forecast_extrapolates_slope(toy_series):
    # A perfectly linear series should be forecast exactly by drift.
    idx = pd.date_range("2020-01-01", periods=50, freq="h")
    linear = pd.Series(np.arange(50, dtype=float), index=idx)

    horizon = 5
    future_index = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    forecast = benchmarks.drift_forecast(linear, horizon, future_index)

    expected = linear.iloc[-1] + np.arange(1, horizon + 1)
    np.testing.assert_allclose(forecast.values, expected)
