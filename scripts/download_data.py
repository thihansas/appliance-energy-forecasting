#!/usr/bin/env python
"""
scripts/download_data.py
==========================
Standalone script to download the raw dataset and produce the cleaned
hourly dataset, without running the rest of the pipeline. Useful for
Part 1 (EDA) exploration in notebooks/01_data_download_and_cleaning.ipynb.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force   # re-download even if cached
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy import data  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    hourly = data.load_appliance_data(force_download=args.force)
    print("\nHourly dataset shape:", hourly.shape)
    print(hourly.head())


if __name__ == "__main__":
    main()
