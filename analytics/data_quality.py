from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/raw/January_2026.csv")

REQUIRED_COLUMNS = [
    "STATE_UT",
    "DISTRICT",
    "BLOCK",
    "VILLAGE",
    "LATITUDE",
    "LONGITUDE",
    "DATE",
    "DTWL",
]


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    return pd.read_csv(DATA_FILE)


def check_columns(df):
    print("\n--- COLUMN CHECK ---")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("Missing columns:")
        for column in missing_columns:
            print(f"  - {column}")
    else:
        print("All required columns are present.")


def check_missing_values(df):
    print("\n--- MISSING VALUES ---")

    missing = df[REQUIRED_COLUMNS].isnull().sum()

    for column, count in missing.items():
        percentage = (count / len(df)) * 100
        print(f"{column}: {count} ({percentage:.2f}%)")


def check_duplicates(df):
    print("\n--- DUPLICATES ---")

    duplicate_count = df.duplicated().sum()

    print(f"Exact duplicate rows: {duplicate_count}")


def check_coordinates(df):
    print("\n--- COORDINATE CHECK ---")

    invalid_latitude = (
        (df["LATITUDE"] < -90) |
        (df["LATITUDE"] > 90)
    ).sum()

    invalid_longitude = (
        (df["LONGITUDE"] < -180) |
        (df["LONGITUDE"] > 180)
    ).sum()

    print(f"Invalid latitudes: {invalid_latitude}")
    print(f"Invalid longitudes: {invalid_longitude}")


def check_dates(df):
    print("\n--- DATE CHECK ---")

    dates = pd.to_datetime(df["DATE"], errors="coerce")

    invalid_dates = dates.isna().sum()

    print(f"Invalid dates: {invalid_dates}")

    if invalid_dates < len(df):
        print(f"Earliest date: {dates.min()}")
        print(f"Latest date: {dates.max()}")


def check_water_levels(df):
    print("\n--- DTWL CHECK ---")

    dtwl = pd.to_numeric(df["DTWL"], errors="coerce")

    invalid_values = dtwl.isna().sum()

    print(f"Non-numeric DTWL values: {invalid_values}")

    print(f"Minimum DTWL: {dtwl.min()}")
    print(f"Maximum DTWL: {dtwl.max()}")
    print(f"Mean DTWL: {dtwl.mean():.2f}")
    print(f"Median DTWL: {dtwl.median():.2f}")


def dataset_summary(df):
    print("\n--- DATASET SUMMARY ---")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print(f"States/UTs: {df['STATE_UT'].nunique()}")
    print(f"Districts: {df['DISTRICT'].nunique()}")
    print(f"Blocks: {df['BLOCK'].nunique()}")
    print(f"Villages: {df['VILLAGE'].nunique()}")

    coordinates = df[["LATITUDE", "LONGITUDE"]].drop_duplicates()

    print(f"Unique coordinates: {len(coordinates)}")


def main():
    print("=" * 60)
    print("DWLR DATA QUALITY REPORT")
    print("=" * 60)

    df = load_data()

    check_columns(df)
    dataset_summary(df)
    check_missing_values(df)
    check_duplicates(df)
    check_coordinates(df)
    check_dates(df)
    check_water_levels(df)

    print("\n" + "=" * 60)
    print("DATA QUALITY CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()