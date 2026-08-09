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


def get_feature_importance(model, feature_names) -> pd.Series:
    """
    Return a Series of feature importances, handling both tree-ensemble
    APIs (feature_importances_) uniformly. Used for Part 9 Q3 ("which
    feature groups appear most useful?").
    """
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names)
    raise AttributeError(f"{type(model).__name__} has no feature_importances_")


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
