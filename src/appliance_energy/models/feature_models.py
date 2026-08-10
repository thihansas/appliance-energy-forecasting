"""
models/feature_models.py
=========================
Feature-based machine-learning model (Part 6). Tries XGBoost first (the
assignment's suggested default), falls back to
sklearn's HistGradientBoostingRegressor if xgboost isn't installed, so the
pipeline always runs. Swap ALGORITHM below or pass algorithm="lightgbm" /
"random_forest" / "hist_gbr" / "xgboost" explicitly.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .. import config

ALGORITHM = "xgboost"  # default; falls back automatically if unavailable


def _make_model(algorithm: str, random_state: int = config.RANDOM_STATE):
    if algorithm == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=600, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=random_state, n_jobs=-1,
            )
        except ImportError:
            print("xgboost not installed - falling back to HistGradientBoostingRegressor")
            algorithm = "hist_gbr"

    if algorithm == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=600, learning_rate=0.03, num_leaves=31,
                random_state=random_state, n_jobs=-1,
            )
        except ImportError:
            print("lightgbm not installed - falling back to HistGradientBoostingRegressor")
            algorithm = "hist_gbr"

    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=500, max_depth=None, n_jobs=-1, random_state=random_state,
        )

    # default / fallback: always available in sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
        random_state=random_state,
    )


def fit_feature_model(X_train: pd.DataFrame, y_train: pd.Series,
                       algorithm: str = ALGORITHM):
    model = _make_model(algorithm)
    model.fit(X_train, y_train)
    return model


def forecast_feature_model(model, X_test: pd.DataFrame, index) -> pd.Series:
    pred = model.predict(X_test)
    return pd.Series(pred, index=index, name="feature_model")


def forecast_feature_model_recursive(
    model,
    history: pd.DataFrame,
    future: pd.DataFrame,
    target: str = config.TARGET,
    feature_columns=None,
) -> pd.Series:
    """Forecast a future block without using its realised target values.

    The ordinary ``forecast_feature_model`` function is useful when a feature
    matrix has already been constructed, but target lags in a test matrix can
    accidentally contain realised test observations.  This routine appends
    each prediction to a copy of the history and rebuilds the feature row for
    the next timestamp, making lag and rolling features genuinely recursive.

    Future sensor/weather columns in ``future`` are retained as supplied.  If
    they are realised observations, the resulting forecast is conditional on
    those covariates; this assumption is documented in the report.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(make_ml_table(history, target=target), target)

    work = history.copy()
    predictions = []
    for timestamp, row in future.iterrows():
        # Keep covariates and timestamp information for the current forecast
        # step, but leave the target unknown until after prediction.
        next_row = row.copy()
        next_row[target] = np.nan
        work = pd.concat([work, pd.DataFrame([next_row], index=[timestamp])])
        candidate = make_ml_table(work, target=target).loc[[timestamp]]
        prediction = float(model.predict(candidate[feature_columns])[0])
        predictions.append(prediction)
        work.loc[timestamp, target] = prediction

    return pd.Series(predictions, index=future.index, name="feature_model")


def get_feature_importance(model, feature_names, X=None, y=None) -> pd.Series:
    """
    Return a Series of feature importances. For tree ensembles that expose
    feature_importances_, use that directly. For models such as
    HistGradientBoostingRegressor, estimate importance via permutation on the
    training data when X and y are provided.
    """
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names)

    if X is not None and y is not None:
        result = permutation_importance(
            model, X, y, n_repeats=20, random_state=config.RANDOM_STATE, n_jobs=-1,
        )
        return pd.Series(result.importances_mean, index=feature_names)

    raise AttributeError(
        f"{type(model).__name__} has no feature_importances_ and no X/y were provided"
    )


def summarise_feature_group_importance(importances: pd.Series) -> pd.Series:
    """
    Aggregate individual feature importances into interpretable groups
    (lag, rolling, time, indoor sensor, outdoor weather) to directly answer
    "which feature groups appear most useful?".
    """
    def group_of(name: str) -> str:
        if name.startswith("lag_"):
            return "lag"
        if name.startswith("roll_"):
            return "rolling"
        if name in {"hour", "dayofweek", "is_weekend", "hour_sin", "hour_cos",
                     "dow_sin", "dow_cos"}:
            return "time"
        if name.startswith("T") or name.startswith("RH_"):
            return "indoor_sensor"
        if name in {"T_out", "RH_out", "Windspeed", "Visibility",
                     "Tdewpoint", "Press_mm_hg"}:
            return "outdoor_weather"
        return "other"

    grouped = importances.groupby(importances.index.map(group_of)).sum()
    return grouped.sort_values(ascending=False)
