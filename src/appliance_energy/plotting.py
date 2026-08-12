
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from . import config


def plot_series_overview(series: pd.Series, title: str = "Appliance energy use"):
    
    fig, ax = plt.subplots(figsize=(14, 5))
    series.plot(ax=ax, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    fig.tight_layout()
    return fig


def plot_seasonal_snapshot(series: pd.Series, days: int = 14,
                            title: str = "Recent appliance energy use"):
    
    fig, ax = plt.subplots(figsize=(14, 5))
    series.tail(days * config.DAILY_PERIOD).plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    fig.tight_layout()
    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 72,
                   title_prefix: str = "Appliances"):
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"{title_prefix}: ACF")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"{title_prefix}: PACF")
    fig.tight_layout()
    return fig


def plot_decomposition(decomposition, title: str = "Seasonal decomposition"):
    
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_forecasts(train: pd.Series, test: pd.Series, forecast_df: pd.DataFrame,
                    context_days: int = 14, title: str = "Appliance energy forecasting"):
    
    fig, ax = plt.subplots(figsize=(14, 7))

    train.tail(context_days * config.DAILY_PERIOD).plot(
        ax=ax, label="Training data", linewidth=1.2, color="tab:gray"
    )
    test.plot(ax=ax, label="Actual (test)", linewidth=2.2, color="black")

    for col in forecast_df.columns:
        if col == "actual":
            continue
        forecast_df[col].plot(ax=ax, label=col, alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_forecast_with_ci(test: pd.Series, mean_forecast: pd.Series,
                           lower: pd.Series, upper: pd.Series,
                           title: str = "SARIMAX forecast with 95% CI"):
    
    fig, ax = plt.subplots(figsize=(14, 5))
    test.plot(ax=ax, label="Actual", color="black", linewidth=2)
    mean_forecast.plot(ax=ax, label="SARIMAX forecast", color="tab:red")
    ax.fill_between(test.index, lower, upper, color="tab:red", alpha=0.2,
                     label="95% CI")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residual_diagnostics(residuals: pd.Series, lags: int = 48,
                               title_prefix: str = "SARIMAX residuals"):
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(residuals.index, residuals.values, linewidth=0.8)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"{title_prefix}: over time")

    plot_acf(residuals.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title(f"{title_prefix}: ACF")

    axes[2].hist(residuals.dropna(), bins=40)
    axes[2].set_title(f"{title_prefix}: distribution")

    fig.tight_layout()
    return fig


def plot_error_diagnostics(test: pd.Series, forecast_df: pd.DataFrame,
                            title: str = "Forecast error by model"):
    
    error_cols = [c for c in forecast_df.columns if c != "actual"]
    errors = pd.DataFrame({
        col: (forecast_df[col] - test).abs() for col in error_cols
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    errors.boxplot(ax=ax, rot=45)
    ax.set_title(title)
    ax.set_ylabel("Absolute error")
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: pd.Series, top_n: int = 20,
                             title: str = "Feature importance"):
    
    top = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * top_n)))
    ax.barh(top.index, top.values)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def plot_model_comparison_bar(results_df: pd.DataFrame, metric: str = "RMSE",
                               title: str = None):
    
    title = title or f"Model comparison ({metric})"
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = results_df.sort_values(metric)
    ax.bar(ordered["model"], ordered[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig
