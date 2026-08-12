
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy import features  


@pytest.fixture
def toy_df():
    idx = pd.date_range("2020-01-01", periods=300, freq="h")
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "Appliances": rng.uniform(10, 100, size=300),
        "T_out": rng.uniform(0, 20, size=300),
        "RH_out": rng.uniform(30, 90, size=300),
    }, index=idx)


def test_time_features_known_at_forecast_origin(toy_df):
    out = features.add_time_features(toy_df)
    assert set(["hour", "dayofweek", "is_weekend",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos"]).issubset(out.columns)
    
    assert out[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].isna().sum().sum() == 0


def test_lag_features_do_not_use_future_target_values(toy_df):
    out = features.add_lag_features(toy_df, target="Appliances", lags=[1, 24])
    
    shifted = toy_df["Appliances"].shift(1)
    pd.testing.assert_series_equal(out["lag_1"], shifted, check_names=False)


def test_rolling_features_are_shifted_before_rolling(toy_df):
    out = features.add_rolling_features(toy_df, target="Appliances", windows=[24])
    
    manual = toy_df["Appliances"].shift(1).rolling(24).mean()
    pd.testing.assert_series_equal(out["roll_mean_24"], manual, check_names=False)


def test_ml_table_has_no_missing_target_after_dropna(toy_df):
    ml_table = features.make_ml_table(toy_df, target="Appliances")
    assert ml_table["Appliances"].isna().sum() == 0
    
    assert len(ml_table) < len(toy_df)


def test_get_exog_columns_only_includes_available_columns(toy_df):
    exog = features.get_exog_columns(toy_df)
    
    assert "Windspeed" not in exog.columns
    assert "T_out" in exog.columns
