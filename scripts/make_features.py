
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy import config, data, features  # noqa: E402


def main():
    df = data.load_appliance_data()
    ml_table = features.make_ml_table(df, target=config.TARGET)

    out_path = config.INTERIM_DIR / "ml_feature_table.csv"
    ml_table.to_csv(out_path)

    print(f"Feature table shape: {ml_table.shape}")
    print(f"Saved to: {out_path}")
    print("\nColumns:")
    print(list(ml_table.columns))


if __name__ == "__main__":
    main()
