
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
