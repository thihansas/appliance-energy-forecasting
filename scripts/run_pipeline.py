#!/usr/bin/env python
"""
scripts/run_pipeline.py
========================
Command-line entry point that runs the full reproducible pipeline
(Parts 1-9 of the assignment) and writes forecasts, metrics and figures
to outputs/.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --grid-search            # SARIMAX (p,d,q) AIC search
    python scripts/run_pipeline.py --grid-search --full-grid
    python scripts/run_pipeline.py --feature-model lightgbm
    python scripts/run_pipeline.py --foundation-backend chronos
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.pipeline import run_pipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run the appliance energy forecasting pipeline.")
    parser.add_argument("--grid-search", action="store_true",
                         help="Run the SARIMAX (p,d,q) AIC grid search before final fit.")
    parser.add_argument("--full-grid", action="store_true",
                         help="Use the full assignment grid (p:0-6, d:0-2, q:0-6) instead of the quick default.")
    parser.add_argument("--feature-model", default="xgboost",
                         choices=["xgboost", "lightgbm", "random_forest", "hist_gbr"])
    parser.add_argument("--foundation-backend", default="auto",
                         choices=["auto", "chronos", "timesfm"])
    return parser.parse_args()


def main():
    args = parse_args()

    results = run_pipeline(
        run_sarimax_grid_search=args.grid_search,
        quick_grid=not args.full_grid,
        feature_algorithm=args.feature_model,
        foundation_backend=args.foundation_backend,
    )

    print("\n=== Pipeline complete ===")
    print("Strongest benchmark:", results["strongest_benchmark"])
    print("SARIMAX order used:", results["sarimax_order"])


if __name__ == "__main__":
    main()
