from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "January": ROOT / "data/raw/January_WL_1994-2025.csv",
    "August": ROOT / "data/processed/august/August_WL_1994-2025_clean.csv"
}

for name, path in FILES.items():

    print("\n" + "=" * 60)
    print(f"{name.upper()} DATASET")
    print("=" * 60)

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst five records:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDataset shape:")
    print(df.shape)

print("\nPrediction data inspection completed.")