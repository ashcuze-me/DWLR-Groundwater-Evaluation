from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw/January_2026.csv")
OUTPUT_FILE = Path("data/processed/January_2026_clean.csv")


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    return pd.read_csv(INPUT_FILE)


def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )

    return df


def clean_text_columns(df):
    text_columns = [
        "STATE_UT",
        "DISTRICT",
        "BLOCK",
        "VILLAGE"
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    return df


def clean_numeric_columns(df):
    df["LATITUDE"] = pd.to_numeric(
        df["LATITUDE"],
        errors="coerce"
    )

    df["LONGITUDE"] = pd.to_numeric(
        df["LONGITUDE"],
        errors="coerce"
    )

    df["DTWL"] = pd.to_numeric(
        df["DTWL"],
        errors="coerce"
    )

    return df


def clean_dates(df):
    df["DATE"] = pd.to_datetime(
        df["DATE"],
        errors="coerce"
    )

    return df


def remove_duplicates(df):
    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    print(f"Duplicate rows removed: {removed}")

    return df


def validate_data(df):
    print("\n--- FINAL VALIDATION ---")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print(f"Missing DTWL: {df['DTWL'].isna().sum()}")
    print(f"Missing latitude: {df['LATITUDE'].isna().sum()}")
    print(f"Missing longitude: {df['LONGITUDE'].isna().sum()}")
    print(f"Invalid dates: {df['DATE'].isna().sum()}")

    invalid_latitude = (
        (df["LATITUDE"] < -90) |
        (df["LATITUDE"] > 90)
    ).sum()

    invalid_longitude = (
        (df["LONGITUDE"] < -180) |
        (df["LONGITUDE"] > 180)
    ).sum()

    print(f"Invalid latitude values: {invalid_latitude}")
    print(f"Invalid longitude values: {invalid_longitude}")

    print(f"Remaining duplicates: {df.duplicated().sum()}")


def save_data(df):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nCleaned dataset saved to: {OUTPUT_FILE}")


def main():
    print("=" * 60)
    print("DWLR DATA CLEANING PIPELINE")
    print("=" * 60)

    df = load_data()

    print(f"\nOriginal rows: {len(df)}")

    df = clean_column_names(df)
    df = clean_text_columns(df)
    df = clean_numeric_columns(df)
    df = clean_dates(df)
    df = remove_duplicates(df)

    validate_data(df)

    save_data(df)

    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()