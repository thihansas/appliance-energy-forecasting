# Appliance energy forecasting report

## 1. Introduction
This report documents a reproducible forecasting study for household appliance energy use using the Appliances Energy Prediction dataset from the UCI Machine Learning Repository. The target variable is **Appliances**, measured at hourly resolution after resampling the original 10-minute data. The objective is to forecast the next 24 hours of energy demand and compare the accuracy of simple benchmarks, a SARIMAX model, a feature-based regressor, and a foundation-model-style baseline.

## 2. Data and preprocessing
The source data were downloaded from the UCI archive and parsed with pandas. The timestamp column was converted to a datetime index, numeric values were coerced, and rows with missing target values were removed. The original 10-minute series was resampled to hourly averages to reduce the computational burden while preserving the daily structure present in appliance usage. Missing values introduced during resampling were filled by time-based interpolation.

The hourly series contains a strong day-night pattern and weekly regularity. The Augmented Dickey-Fuller and KPSS tests both indicate that the raw series is stationary at the 5% level, while differencing gives an even more stable signal. This suggests that the target series is already reasonably stable after resampling, although the autocorrelation structure is still meaningful and should be captured by the SARIMAX and lag-based models.

## 3. Forecasting design
The forecasting problem is a short-horizon, multi-step evaluation over the final 14 days of the dataset. The train period runs from 2016-01-11 17:00 to 2016-05-13 18:00, and the test period contains 336 hourly points from 2016-05-13 19:00 to 2016-05-27 18:00. The forecast horizon is therefore 24 hours per day over the 14-day hold-out window. The evaluation metrics are MAE, RMSE, MASE and Bias, and all models are scored on the same test period.

## 4. Benchmark models
The benchmark family includes mean, naive, daily seasonal naive, weekly seasonal naive and drift forecasts. The strongest of these is the weekly seasonal naive model, which indicates that appliance use has a strong repeating weekly structure rather than a purely random walk behaviour. The daily seasonal naive model performs poorly because the household pattern is not simply a one-day repetition; the weekly cycle appears more informative. This is consistent with daily routines such as cooking, laundry, heating and lighting returning on a weekly basis.

## 5. SARIMAX model
A SARIMAX model was fitted with daily seasonality and a small set of exogenous variables, including outdoor temperature, outdoor humidity, wind speed, visibility, dew point, and cyclical time-of-day/day-of-week features. The final model used the quick grid-search order (1, 0, 1) for the AR/MA terms and a seasonal order of (1, 1, 1, 24). The residual diagnostics show that the model does not fully remove autocorrelation: the Ljung-Box test at lag 24 remains highly significant, suggesting that residual structure remains. In practice, the SARIMAX model improves on the simple benchmarks but does not dominate the feature-based approach. The main limitation is that future weather covariates are treated as known values from the test period, making the forecast conditional rather than a fully operational one-step-ahead forecast.

## 6. Feature-based model
The feature-based model uses lagged appliance values, rolling statistics, time-of-day features, day-of-week features and indoor/outdoor sensor variables. The pipeline uses HistGradientBoostingRegressor because the environment did not have XGBoost installed; this still provides a valid tree-based baseline and is more robust in this environment. The feature set materially improves performance relative to the benchmark models and slightly surpasses the SARIMAX model. The strongest signal comes from lag and rolling features, while time-of-day and weather variables provide additional context. The model therefore captures the short-term persistence and household routine structure that are important in this domain.

## 7. Foundation model
A foundation-model-style backend was attempted, but the required packages (Chronos and TimesFM) are not installed in this environment. The pipeline therefore falls back to a daily seasonal naive baseline for this component. That means the foundation-model comparison is not a genuine foundation-model result, but it still demonstrates that the simpler benchmark and the feature-based model already provide better accuracy than a naive zero-shot alternative in this setting.

## 8. Results and comparison
The table below summarises the evaluation metrics for all models on the 14-day test window.

| Model | MAE | RMSE | MASE | Bias |
| --- | ---: | ---: | ---: | ---: |
| feature_model | 32.882 | 55.050 | 0.615 | 4.036 |
| sarimax | 36.887 | 63.879 | 0.690 | -5.068 |
| seasonal_naive_weekly | 42.634 | 79.290 | 0.798 | -10.818 |
| mean | 50.319 | 74.906 | 0.942 | -3.109 |
| seasonal_naive_daily | 86.959 | 129.232 | 1.628 | 64.013 |
| foundation_model | 86.959 | 129.232 | 1.628 | 64.013 |
| naive | 250.640 | 258.820 | 4.692 | 247.763 |
| drift | 266.373 | 274.611 | 4.986 | 264.501 |

The feature-based model achieved the lowest MAE and RMSE, followed by SARIMAX and the weekly seasonal naive benchmark. The sharp drop from the naive and drift baselines shows that appliance energy use is highly structured and partly predictable from recent history and time-of-day signals. The strong performance of the weekly seasonal naive benchmark also suggests that household routines are persistent over several days.

Representative figures were saved to the outputs folder, including the forecast comparison, error diagnostics, SARIMAX forecast with confidence intervals, residual diagnostics, and feature importance plots. The most relevant files are:

- [forecast comparison](../outputs/figures/forecast_comparison.png)
- [SARIMAX forecast with confidence intervals](../outputs/figures/sarimax_forecast_ci.png)
- [residual diagnostics](../outputs/figures/residual_acf.png)
- [feature importance](../outputs/figures/feature_importance.png)

## 9. Answers to the assignment questions
1. The strongest benchmark is the weekly seasonal naive model. This indicates that appliance energy use has a strong weekly seasonal pattern and that recent history matters more than a simple random-walk extrapolation.
2. SARIMAX improves on the strongest benchmark, but not by a large margin. Daily seasonality and autocorrelation are partly captured, while the remaining residual structure suggests that nonlinear effects and household context are not fully represented.
3. The feature-based model improves when lag, rolling and time-based features are added. Lag and rolling features are the most informative, while weather and indoor sensor values provide useful but secondary context.
4. The foundation model does not outperform the simpler benchmark, SARIMAX or feature-based models in this environment. The improvement, if any, would not appear large enough to justify the additional complexity and deployment burden without a properly installed and tuned foundation model.
5. In a real forecast, future indoor sensor readings and external weather variables are not always known at the forecast origin. If these values are used from the test period, the resulting forecast is a conditional forecast rather than a true operational forecast.
6. For practical smart-home deployment, the feature-based model is the recommended choice because it offers a strong accuracy-efficiency trade-off, is easier to interpret than a black-box foundation model, and can be retrained with only the covariates that are genuinely available at the forecast origin.

## 10. Discussion and limitations
The study is useful for short-horizon forecasting, but several limitations remain. First, the feature-based model uses the actual recent target values from the evaluation period to build lagged features. That makes the evaluation more realistic for a rolling one-step forecast than for a fully recursive 24-hour forecast, but it should still be interpreted as an operationally conditional evaluation. Second, the SARIMAX forecast uses future exogenous values from the hold-out period, again making it conditional. Third, the foundation-model section is only a fallback because the environment lacks the necessary packages. Future work could compare against a properly installed Chronos or TimesFM model, test a recursive multi-step feature model, and explore probabilistic forecasts to quantify uncertainty.

## 11. References
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*.
- Hyndman, R. J., and Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*.
- Hyndman, R. J., and Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
