from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "January_WL_1994-2025.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "historical_audit"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {DATA_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "STATE_UT",
        "DISTRICT",
        "BLOCK",
        "VILLAGE",
        "LATITUDE",
        "LONGITUDE",
        "Date",
        "WL(mbgl)",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["Date_parsed"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%y",
        errors="coerce",
    )

    df["LATITUDE"] = pd.to_numeric(
        df["LATITUDE"], errors="coerce"
    )

    df["LONGITUDE"] = pd.to_numeric(
        df["LONGITUDE"], errors="coerce"
    )

    df["WL(mbgl)"] = pd.to_numeric(
        df["WL(mbgl)"], errors="coerce"
    )

    # Basic dataset information
    print("\n===== HISTORICAL DATA QUALITY REPORT =====")

    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns) - 1}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")

    print("\n===== MISSING VALUES =====")
    print(df[required_columns].isna().sum())

    # Date validation
    valid_dates = df["Date_parsed"].dropna()

    print("\n===== DATE COVERAGE =====")
    print(f"Invalid dates: {df['Date_parsed'].isna().sum()}")

    if not valid_dates.empty:
        print(f"Earliest date: {valid_dates.min().date()}")
        print(f"Latest date: {valid_dates.max().date()}")
        print(f"Unique dates: {valid_dates.nunique()}")

    df["YEAR"] = df["Date_parsed"].dt.year

    yearly = (
        df.dropna(subset=["YEAR"])
        .groupby("YEAR")
        .agg(
            records=("Date_parsed", "size"),
            unique_dates=("Date_parsed", "nunique"),
            unique_locations=("LATITUDE", "count"),
            mean_water_level=("WL(mbgl)", "mean"),
            median_water_level=("WL(mbgl)", "median"),
            min_water_level=("WL(mbgl)", "min"),
            max_water_level=("WL(mbgl)", "max"),
        )
        .reset_index()
    )

    yearly.to_csv(
        OUTPUT_DIR / "yearly_summary.csv",
        index=False,
    )

    print("\n===== YEARLY RECORD COUNTS =====")
    print(yearly[["YEAR", "records", "unique_dates"]].to_string(
        index=False
    ))

    # Coordinate validation
    invalid_lat = (
        (df["LATITUDE"] < -90)
        | (df["LATITUDE"] > 90)
    )

    invalid_lon = (
        (df["LONGITUDE"] < -180)
        | (df["LONGITUDE"] > 180)
    )

    print("\n===== COORDINATE VALIDATION =====")
    print(f"Invalid latitudes: {invalid_lat.sum()}")
    print(f"Invalid longitudes: {invalid_lon.sum()}")

    print(
        "Unique coordinate pairs:",
        df[["LATITUDE", "LONGITUDE"]]
        .dropna()
        .drop_duplicates()
        .shape[0],
    )

    # Potential duplicate observations
    potential_duplicates = df.duplicated(
        subset=[
            "LATITUDE",
            "LONGITUDE",
            "Date_parsed",
        ],
        keep=False,
    )

    repeated = df.loc[potential_duplicates].sort_values(
        ["LATITUDE", "LONGITUDE", "Date_parsed"]
    )

    repeated.to_csv(
        OUTPUT_DIR / "repeated_coordinate_dates.csv",
        index=False,
    )

    # Location coverage across years
    location_years = (
        df.dropna(subset=["LATITUDE", "LONGITUDE", "YEAR"])
        .groupby(["LATITUDE", "LONGITUDE"])
        .agg(
            first_year=("YEAR", "min"),
            last_year=("YEAR", "max"),
            years_observed=("YEAR", "nunique"),
            records=("YEAR", "size"),
        )
        .reset_index()
    )

    location_years.to_csv(
        OUTPUT_DIR / "location_temporal_coverage.csv",
        index=False,
    )

    print("\n===== TEMPORAL LOCATION COVERAGE =====")
    print(
        "Unique coordinate pairs:",
        len(location_years),
    )

    print(
        "Locations observed across multiple years:",
        (location_years["years_observed"] > 1).sum(),
    )

    print(
        "Locations spanning at least 10 calendar years:",
        (
            location_years["last_year"]
            - location_years["first_year"]
            >= 9
        ).sum(),
    )

    # Water-level summary
    print("\n===== WATER-LEVEL SUMMARY =====")
    print(df["WL(mbgl)"].describe())

    print(
        "Negative water-level values:",
        (df["WL(mbgl)"] < 0).sum(),
    )

    print(
        "Zero water-level values:",
        (df["WL(mbgl)"] == 0).sum(),
    )

    # Location-label consistency
    label_counts = (
        df.groupby("STATE_UT", dropna=False)["DISTRICT"]
        .nunique()
        .sort_values(ascending=False)
    )

    print("\n===== STATE/UT LABELS =====")
    print("Distinct state/UT labels:", df["STATE_UT"].nunique())
    print("Distinct district labels:", df["DISTRICT"].nunique())

    label_counts.to_csv(
        OUTPUT_DIR / "districts_per_state.csv"
    )

    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()