#!/usr/bin/env python

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy import config, data, evaluation, plotting  # noqa: E402


def main():
    forecast_path = config.FORECAST_DIR / "all_forecasts.csv"
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"{forecast_path} not found - run scripts/run_pipeline.py first."
        )

    forecast_df = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
    test = forecast_df["actual"]

    df = data.load_appliance_data()
    y = df[config.TARGET]
    train = y.loc[y.index < test.index.min()]

    model_cols = [c for c in forecast_df.columns
                  if c not in ("actual", "sarimax_lower", "sarimax_upper")]
    forecasts = {c: forecast_df[c] for c in model_cols}

    results_df = evaluation.evaluate_all(
        forecasts=forecasts, y_true=test, y_train=train, seasonality=config.DAILY_PERIOD,
    )
    results_df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    print(results_df.round(3))

    fig = plotting.plot_error_diagnostics(test=test, forecast_df=forecast_df)
    fig.savefig(config.FIGURE_DIR / "error_diagnostics.png", dpi=200, bbox_inches="tight")
    
        # 24-hour error comparison
    fig = plotting.plot_error_diagnostics(
        test=test_24,
        forecast_df=forecast_df_24,
        title="Forecast error by model - first 24 hours"
    )

    fig.savefig(
        config.FIGURE_DIR / "error_diagnostics_24h.png",
        dpi=300,
        bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
