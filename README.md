# Appliance Energy Forecasting

This repository contains a reproducible time-series forecasting pipeline for modelling and forecasting household appliance energy use.

The project uses the **Appliances Energy Prediction** dataset, which contains appliance energy consumption, indoor temperature and humidity sensor measurements, outdoor weather variables, and timestamp information. The aim is to compare simple benchmark models, a SARIMAX model, a feature-based machine-learning model, and a time-series foundation model.

## Project aim

The aim of this assignment is to forecast short-term household appliance energy use and evaluate whether increasingly complex models improve on simple benchmark methods.

The main questions are:

1. How well do simple benchmark models forecast appliance energy use?
2. Does a SARIMAX model improve on the benchmark forecasts?
3. Do sensor, weather, and time-based covariates improve forecast accuracy?
4. Does a feature-based machine-learning model such as XGBoost improve performance?
5. Does a time-series foundation model such as Chronos, TimesFM, or TimeGPT provide any additional benefit?
6. Which model would be most suitable for a practical smart-home energy forecasting system?

## Current implementation status

The repository now contains a runnable end-to-end pipeline for the assignment. Running `python scripts/run_pipeline.py` downloads or loads the UCI dataset, resamples it to hourly frequency, fits benchmark forecasts, fits a SARIMAX model, fits a feature-based regressor, and evaluates all models on the final 14-day hold-out window. 

A typical run produces:

- `outputs/forecasts/all_forecasts.csv`
- `outputs/metrics/model_comparison.csv`
- `outputs/figures/forecast_comparison.png`
- `outputs/figures/sarimax_forecast_ci.png`
- `outputs/figures/feature_importance.png`

## Dataset

The dataset used in this project is the **Appliances Energy Prediction** dataset.

The target variable is:

```text
Appliances
```

This represents household appliance energy use for each time interval.

The original dataset is sampled every 10 minutes and contains variables including:

```text
date
Appliances
lights
T1, RH_1
T2, RH_2
T3, RH_3
T4, RH_4
T5, RH_5
T6, RH_6
T7, RH_7
T8, RH_8
T9, RH_9
T_out
Press_mm_hg
RH_out
Windspeed
Visibility
Tdewpoint
```

The `T` variables are indoor temperature measurements from different rooms or sensor locations. The `RH` variables are indoor relative humidity measurements. The outdoor weather variables include outdoor temperature, pressure, outdoor humidity, wind speed, visibility, and dew point.


## Installation

Create a Python environment.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```


If using XGBoost:

```text
xgboost
```

If using TimeGPT:

```text
nixtla
```

If using Chronos or TimesFM, additional packages such as `torch`, `transformers`, or model-specific dependencies may be required.

## Running the pipeline


The main pipeline entry point is:

```bash
python scripts/run_pipeline.py
```

The pipeline :

1. Load or download the dataset.
2. Clean and prepare the time series.
3. Create time, lag, rolling, sensor, and weather features.
4. Split the data into training and test sets.
5. Fit benchmark models.
6. Fit the SARIMAX model.
7. Fit the feature-based model.
8. Fit or call the foundation model.
9. Evaluate all forecasts.
10. Save forecasts, metrics, and plots.

## Outputs

The pipeline saves forecasts to:

```text
outputs/forecasts/all_forecasts.csv
```

This file contains the actual values and model forecasts:

```text
actual
mean
naive
seasonal_naive_daily
seasonal_naive_weekly
drift
sarimax
feature_model
foundation_model
```

The pipeline saves model comparison metrics to:

```text
outputs/metrics/model_comparison.csv
```

This file contains:

```text
model
MAE
RMSE
MASE
Bias
```

The pipeline saves figures to:

```text
outputs/figures/
```

Suggested figures include:

```text
forecast_comparison.png
error_diagnostics.png
residual_acf.png
feature_importance.png
```

## Evaluation metrics

All models evaluates on the same test period.

Required metrics:

```text
MAE
RMSE
MASE
Bias
```

## Data leakage

Examples of leakage include:

```text
Using future values of Appliances in lag or rolling features
Creating rolling features without shifting the target first
Scaling the full dataset before the train-test split
Using future sensor or weather values without discussing forecast realism
Choosing the final model based only on test-set performance
```

Important point:

```text
Future time-of-day and day-of-week variables are known in advance. Future indoor sensor and weather variables may not be known in a real operational forecast. If realised future sensor or weather values are used from the test set, the result are described as a conditional forecast.
```


## Tests


Examples:

```text
test that forecast lengths match the test period
test that MASE is zero for a perfect forecast
test that lag features do not use future target values
test that the processed dataset has no missing target values
```

Run tests using:

```bash
pytest
```

