
import json

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from . import config, data, features, evaluation, plotting
from .models import benchmarks, sarimax as sarimax_model, feature_models, foundation


def run_pipeline(run_sarimax_grid_search: bool = False, quick_grid: bool = True,
                  feature_algorithm: str = "xgboost",
                  foundation_backend: str = "auto") -> dict:

    # Part 1: load data

    df = data.load_appliance_data()
    y = df[config.TARGET]

    stationarity_df = data.stationarity_report(y, name=config.TARGET)
    stationarity_df.to_csv(config.METRICS_DIR / "stationarity_tests.csv", index=False)
    print("\nStationarity tests:")
    print(stationarity_df.round(4))

    fig = plotting.plot_series_overview(y)
    fig.savefig(config.FIGURE_DIR / "series_overview.png", dpi=200, bbox_inches="tight")

    fig = plotting.plot_seasonal_snapshot(y)
    fig.savefig(config.FIGURE_DIR / "series_recent.png", dpi=200, bbox_inches="tight")

    fig = plotting.plot_acf_pacf(y)
    fig.savefig(config.FIGURE_DIR / "acf_pacf.png", dpi=200, bbox_inches="tight")

    try:
        decomposition = seasonal_decompose(y, model="additive", period=config.DAILY_PERIOD)
        fig = plotting.plot_decomposition(decomposition)
        fig.savefig(config.FIGURE_DIR / "seasonal_decomposition.png", dpi=200, bbox_inches="tight")
    except Exception as exc:
        print(f"Seasonal decomposition skipped: {exc}")

    
    # Part 2: forecasting problem definition (train/test split)
    
    train = y.iloc[:-config.TEST_STEPS]
    test = y.iloc[-config.TEST_STEPS:]
    horizon = config.HORIZON  

    print(f"\nTrain: {train.index.min()} -> {train.index.max()} ({len(train)} obs)")
    print(f"Test:  {test.index.min()} -> {test.index.max()} ({len(test)} obs)")

    forecasts = {}

    
    # Part 3: benchmark models
    
    forecasts.update(benchmarks.all_benchmark_forecasts(
        y_train=train, horizon=len(test), index=test.index,
        daily_period=config.DAILY_PERIOD, weekly_period=config.WEEKLY_PERIOD,
    ))

    
    # Part 4/5: SARIMAX with selected exogenous variables
    
    exog_full = features.get_exog_columns(df)
    X_train = exog_full.iloc[:-config.TEST_STEPS]
    X_test = exog_full.iloc[-config.TEST_STEPS:]

    best_order = (1, 0, 1)
    if run_sarimax_grid_search:
        grid = sarimax_model.grid_search_sarimax(
            y_train=train, X_train=X_train, quick=quick_grid,
        )
        grid.to_csv(config.METRICS_DIR / "sarimax_grid_search.csv", index=False)
        best_row = grid.dropna(subset=["aic"]).iloc[0]
        best_order = (int(best_row.p), int(best_row.d), int(best_row.q))
        print(f"\nBest SARIMAX order from grid search: {best_order} "
              f"(AIC={best_row.aic:.1f})")

    sarimax_fit = sarimax_model.fit_sarimax(y_train=train, order=best_order, X_train=X_train)

    sarimax_mean, sarimax_lower, sarimax_upper = sarimax_model.forecast_sarimax(
        fit=sarimax_fit, horizon=len(test), index=test.index, X_test=X_test,
    )
    forecasts["sarimax"] = sarimax_mean

    fig = plotting.plot_forecast_with_ci(test, sarimax_mean, sarimax_lower, sarimax_upper)
    fig.savefig(config.FIGURE_DIR / "sarimax_forecast_ci.png", dpi=200, bbox_inches="tight")

    fig = plotting.plot_residual_diagnostics(sarimax_fit.resid)
    fig.savefig(config.FIGURE_DIR / "residual_acf.png", dpi=200, bbox_inches="tight")

    resid_summary = sarimax_model.residual_diagnostics_summary(sarimax_fit)
    with open(config.METRICS_DIR / "sarimax_residual_diagnostics.json", "w") as f:
        json.dump(resid_summary, f, indent=2)
    print("\nSARIMAX residual diagnostics:", resid_summary)

    
    # Part 5/6: feature-based model
    
    ml_table = features.make_ml_table(df, target=config.TARGET)
    feature_cols = features.get_feature_columns(ml_table, target=config.TARGET)
    
    ml_train = ml_table[ml_table.index < test.index[0]]

    fm = feature_models.fit_feature_model(
        X_train=ml_train[feature_cols], y_train=ml_train[config.TARGET],
        algorithm=feature_algorithm,
    )
    feature_pred = feature_models.forecast_feature_model_recursive(
        model=fm,
        history=df.loc[df.index < test.index[0]],
        future=df.loc[test.index],
        target=config.TARGET,
        feature_columns=feature_cols,
    )
    forecasts["feature_model"] = feature_pred.reindex(test.index)

    try:
        importances = feature_models.get_feature_importance(
            fm, feature_cols, X=ml_train[feature_cols], y=ml_train[config.TARGET],
        )
        importances.to_csv(config.METRICS_DIR / "feature_importance.csv", header=["importance"])

        fig = plotting.plot_feature_importance(importances)
        fig.savefig(config.FIGURE_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")

        group_importance = feature_models.summarise_feature_group_importance(importances)
        group_importance.to_csv(config.METRICS_DIR / "feature_group_importance.csv",
                                 header=["importance"])
        print("\nFeature group importance:")
        print(group_importance)
    except AttributeError as exc:
        print(f"Feature importance not available for this model: {exc}")

    # Part 7: foundation model

    forecasts["foundation_model"] = foundation.forecast_foundation_model(
        y_train=train, horizon=len(test), index=test.index, backend=foundation_backend,
    )


    # Part 8: evaluate everything
   
    results_df = evaluation.evaluate_all(
        forecasts=forecasts, y_true=test, y_train=train, seasonality=config.DAILY_PERIOD,
    )
    print("\nModel comparison:")
    print(results_df.round(3))

    strongest_benchmark = (
        results_df[results_df["model"].isin(
            ["mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly", "drift"]
        )]
        .sort_values("MASE")
        .iloc[0]["model"]
    )
    print(f"\nStrongest benchmark model: {strongest_benchmark}")

  
    # Save outputs
    
    forecast_df = pd.DataFrame({"actual": test})
    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test.index)
    forecast_df["sarimax_lower"] = sarimax_lower
    forecast_df["sarimax_upper"] = sarimax_upper

    forecast_df.to_csv(config.FORECAST_DIR / "all_forecasts.csv")
    results_df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)

    fig = plotting.plot_forecasts(train=train, test=test, forecast_df=forecast_df)
    fig.savefig(config.FIGURE_DIR / "forecast_comparison.png", dpi=300, bbox_inches="tight")

    fig = plotting.plot_error_diagnostics(test=test, forecast_df=forecast_df)
    fig.savefig(config.FIGURE_DIR / "error_diagnostics.png", dpi=200, bbox_inches="tight")

    fig = plotting.plot_model_comparison_bar(results_df, metric="RMSE")
    fig.savefig(config.FIGURE_DIR / "model_comparison_rmse.png", dpi=200, bbox_inches="tight")

    print("\nSaved outputs to:", config.OUTPUT_DIR)

    return {
        "results_df": results_df,
        "forecast_df": forecast_df,
        "stationarity_df": stationarity_df,
        "strongest_benchmark": strongest_benchmark,
        "sarimax_order": best_order,
        "sarimax_residual_diagnostics": resid_summary,
    }
