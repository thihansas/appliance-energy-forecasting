
import pandas as pd


def mean_forecast(y_train: pd.Series, horizon: int, index) -> pd.Series:
    return pd.Series(y_train.mean(), index=index, name="mean")


def naive_forecast(y_train: pd.Series, horizon: int, index) -> pd.Series:
    return pd.Series(y_train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(y_train: pd.Series, horizon: int, index,
                             seasonality: int) -> pd.Series:
 
    history = list(y_train.values)
    values = []

    for _ in range(horizon):
        values.append(history[-seasonality])
        history.append(values[-1])

    name = "seasonal_naive_daily" if seasonality == 24 else "seasonal_naive_weekly"
    return pd.Series(values, index=index, name=name)


def drift_forecast(y_train: pd.Series, horizon: int, index) -> pd.Series:
    
    slope = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)
    values = [y_train.iloc[-1] + slope * step for step in range(1, horizon + 1)]
    return pd.Series(values, index=index, name="drift")


def all_benchmark_forecasts(y_train: pd.Series, horizon: int, index,
                             daily_period: int = 24, weekly_period: int = 168) -> dict:
    
    return {
        "mean": mean_forecast(y_train, horizon, index),
        "naive": naive_forecast(y_train, horizon, index),
        "seasonal_naive_daily": seasonal_naive_forecast(
            y_train, horizon, index, seasonality=daily_period),
        "seasonal_naive_weekly": seasonal_naive_forecast(
            y_train, horizon, index, seasonality=weekly_period),
        "drift": drift_forecast(y_train, horizon, index),
    }
