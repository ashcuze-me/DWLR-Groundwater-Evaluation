from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/January_2026_clean.csv")
OUTPUT_FILE = Path("data/processed/January_2026_standardized.csv")


TEXT_COLUMNS = [
    "STATE_UT",
    "DISTRICT",
    "BLOCK",
    "VILLAGE"
]


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    return pd.read_csv(INPUT_FILE)


def normalize_text(value):
    if pd.isna(value):
        return pd.NA

    return (
        str(value)
        .strip()
        .upper()
        .replace("  ", " ")
    )


def normalize_location_names(df):
    for column in TEXT_COLUMNS:
        df[f"{column}_NORMALIZED"] = (
            df[column]
            .apply(normalize_text)
        )

    return df


def create_location_key(df):
    latitude = df["LATITUDE"].round(6)
    longitude = df["LONGITUDE"].round(6)

    df["LOCATION_KEY"] = (
        latitude.astype(str)
        + "_"
        + longitude.astype(str)
    )

    return df


def report_standardization(df):
    print("\n--- LOCATION STANDARDIZATION ---")

    print(
        "Original districts:",
        df["DISTRICT"].nunique()
    )

    print(
        "Normalized districts:",
        df["DISTRICT_NORMALIZED"].nunique()
    )

    print(
        "Original villages:",
        df["VILLAGE"].nunique()
    )

    print(
        "Normalized villages:",
        df["VILLAGE_NORMALIZED"].nunique()
    )

    print(
        "Unique location keys:",
        df["LOCATION_KEY"].nunique()
    )


def save_data(df):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nStandardized dataset saved to: {OUTPUT_FILE}"
    )


def main():
    print("=" * 60)
    print("DWLR LOCATION STANDARDIZATION")
    print("=" * 60)

    df = load_data()

    print(f"\nInput rows: {len(df)}")

    df = normalize_location_names(df)
    df = create_location_key(df)

    report_standardization(df)

    save_data(df)

    print("\n" + "=" * 60)
    print("LOCATION STANDARDIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()