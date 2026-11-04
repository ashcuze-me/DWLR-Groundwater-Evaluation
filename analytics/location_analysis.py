from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/processed/January_2026_standardized.csv")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    print("=" * 60)
    print("DWLR LOCATION ANALYSIS")
    print("=" * 60)

    print("\n--- BASIC COUNTS ---")
    print("Total records:", len(df))

    print(
        "Unique coordinate pairs:",
        df[["LATITUDE", "LONGITUDE"]].drop_duplicates().shape[0]
    )

    print(
        "Unique location keys:",
        df["LOCATION_KEY"].nunique()
    )

    print("\n--- RECORDS PER LOCATION ---")

    records_per_location = df.groupby("LOCATION_KEY").size()

    print("Locations with one record:", (records_per_location == 1).sum())
    print("Locations with multiple records:", (records_per_location > 1).sum())
    print("Maximum records at one location:", records_per_location.max())

    print("\n--- LOCATIONS WITH MULTIPLE RECORDS ---")

    repeated_locations = (
        records_per_location[records_per_location > 1]
        .sort_values(ascending=False)
    )

    print(repeated_locations.head(10))

    print("\n--- COORDINATES WITH MULTIPLE DISTRICTS ---")

    district_counts = (
        df.groupby(["LATITUDE", "LONGITUDE"])["DISTRICT_NORMALIZED"]
        .nunique()
    )

    conflicting_districts = district_counts[district_counts > 1]

    print(
        "Coordinate pairs associated with multiple districts:",
        len(conflicting_districts)
    )

    print("\n--- COORDINATES WITH MULTIPLE STATES/UTS ---")

    state_counts = (
        df.groupby(["LATITUDE", "LONGITUDE"])["STATE_UT_NORMALIZED"]
        .nunique()
    )

    conflicting_states = state_counts[state_counts > 1]

    print(
        "Coordinate pairs associated with multiple States/UTs:",
        len(conflicting_states)
    )

    print("\n" + "=" * 60)
    print("LOCATION ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()