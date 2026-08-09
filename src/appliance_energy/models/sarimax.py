"""
models/sarimax.py
==================
SARIMAX model (Part 4): grid search over (p,d,q) via AIC, fitting with or
without exogenous variables, forecasting with confidence intervals, and
residual diagnostics.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .. import config


def grid_search_sarimax(y_train: pd.Series, X_train: pd.DataFrame = None,
                         p_range=None, d_range=None, q_range=None,
                         seasonal_order=None, quick: bool = True,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Loop over all (p, d, q) combinations and record the AIC of each fitted
    model, as required by the assignment ("loop over all possible parameter
    combinations for p=[0,6], d=[0,2] and q=[0,6]").

    Parameters
    ----------
    quick : bool
        If True (default), use the reduced ranges in config (QUICK_*_RANGE)
        for a fast sanity-check search. Set quick=False and/or pass explicit
        p_range/d_range/q_range to run the full assignment-specified grid
        (p in 0-6, d in 0-2, q in 0-6 = 147 fits); this is slow (potentially
        tens of minutes) because each fit involves numerical optimisation.

    Returns
    -------
    DataFrame with columns [p, d, q, aic, converged], sorted by AIC ascending.
    Failed/non-converged fits are kept with aic = NaN so students can see
    which regions of the grid are unstable.
    """
    if p_range is None:
        p_range = config.QUICK_P_RANGE if quick else config.SARIMAX_P_RANGE
    if d_range is None:
        d_range = config.QUICK_D_RANGE if quick else config.SARIMAX_D_RANGE
    if q_range is None:
        q_range = config.QUICK_Q_RANGE if quick else config.SARIMAX_Q_RANGE

    seasonal_order = seasonal_order or config.SARIMAX_SEASONAL_ORDER

    results = []
    combos = list(itertools.product(p_range, d_range, q_range))

    for i, (p, d, q) in enumerate(combos):
        if verbose:
            print(f"[{i + 1}/{len(combos)}] fitting SARIMAX(p={p}, d={d}, q={q}) "
                  f"x {seasonal_order} ...", end="\r")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_train, exog=X_train,
                    order=(p, d, q), seasonal_order=seasonal_order,
                    trend="c" if d == 0 else None,
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                fit = model.fit(disp=False)
            results.append({"p": p, "d": d, "q": q, "aic": fit.aic, "converged": True})
        except Exception:
            results.append({"p": p, "d": d, "q": q, "aic": np.nan, "converged": False})

    if verbose:
        print()

    df = pd.DataFrame(results).sort_values("aic", na_position="last").reset_index(drop=True)
    return df


def fit_sarimax(y_train: pd.Series, order=(1, 0, 1), seasonal_order=None,
                 X_train: pd.DataFrame = None):
    """Fit a SARIMAX model with a chosen (p,d,q) order and daily seasonality."""
    seasonal_order = seasonal_order or config.SARIMAX_SEASONAL_ORDER

    model = SARIMAX(
        y_train, exog=X_train,
        order=order, seasonal_order=seasonal_order,
        trend="c" if order[1] == 0 else None,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarimax(fit, horizon: int, index, X_test: pd.DataFrame = None,
                      alpha: float = 0.05):
    """
    Forecast `horizon` steps ahead, returning the point forecast plus lower
    and upper confidence bounds (Part 4: "Add confidence intervals").
    """
    fc = fit.get_forecast(steps=horizon, exog=X_test)

    mean = fc.predicted_mean
    mean.index = index
    mean.name = "sarimax"

    ci = fc.conf_int(alpha=alpha)
    ci.index = index
    lower = ci.iloc[:, 0].rename("sarimax_lower")
    upper = ci.iloc[:, 1].rename("sarimax_upper")

    return mean, lower, upper


def residual_diagnostics_summary(fit) -> dict:
    """
    Numeric summary to accompany the residual ACF/histogram plots: the
    Ljung-Box p-value at lag 24 tests whether residual autocorrelation
    remains (low p-value => model has NOT captured all the structure).
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    resid = fit.resid.dropna()
    lb = acorr_ljungbox(resid, lags=[config.DAILY_PERIOD], return_df=True)

    return {
        "resid_mean": float(resid.mean()),
        "resid_std": float(resid.std()),
        "ljung_box_stat_lag24": float(lb["lb_stat"].iloc[0]),
        "ljung_box_pvalue_lag24": float(lb["lb_pvalue"].iloc[0]),
        "residuals_look_white_noise": bool(lb["lb_pvalue"].iloc[0] > 0.05),
    }
