
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "January_WL_1994-2025.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "trend_validation"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%y",
        errors="coerce",
    )

    df["WL(mbgl)"] = pd.to_numeric(
        df["WL(mbgl)"],
        errors="coerce",
    )

    required = [
        "STATE_UT",
        "DISTRICT",
        "BLOCK",
        "VILLAGE",
        "LATITUDE",
        "LONGITUDE",
        "Date",
        "WL(mbgl)",
    ]

    df = df.dropna(subset=required).copy()

    exact_duplicates = int(df.duplicated().sum())

    # Remove only exact duplicate records in this analysis copy.
    df = df.drop_duplicates().copy()

    location_keys = [
        "STATE_UT",
        "DISTRICT",
        "BLOCK",
        "VILLAGE",
        "LATITUDE",
        "LONGITUDE",
    ]

    # Check whether coordinates map to multiple administrative labels.
    coordinate_locations = (
        df.groupby(["LATITUDE", "LONGITUDE"])
        .agg(
            records=("Date", "size"),
            villages=("VILLAGE", "nunique"),
            districts=("DISTRICT", "nunique"),
            blocks=("BLOCK", "nunique"),
            states=("STATE_UT", "nunique"),
        )
        .reset_index()
    )

    coordinate_locations["ambiguous_admin_labels"] = (
        (coordinate_locations["villages"] > 1)
        | (coordinate_locations["districts"] > 1)
        | (coordinate_locations["blocks"] > 1)
        | (coordinate_locations["states"] > 1)
    )

    coordinate_locations.to_csv(
        OUTPUT_DIR / "coordinate_location_ambiguity.csv",
        index=False,
    )

    # Inspect repeated dates within provisional location groups.
    date_keys = location_keys + ["Date"]

    repeated_dates = (
        df.groupby(date_keys, dropna=False)
        .agg(
            records=("WL(mbgl)", "size"),
            distinct_levels=("WL(mbgl)", "nunique"),
            minimum_level=("WL(mbgl)", "min"),
            maximum_level=("WL(mbgl)", "max"),
        )
        .reset_index()
    )

    repeated_dates = repeated_dates[
        repeated_dates["records"] > 1
    ].copy()

    repeated_dates.to_csv(
        OUTPUT_DIR / "repeated_provisional_location_dates.csv",
        index=False,
    )

    summary = {
        "valid_records_before_exact_deduplication": len(
            pd.read_csv(DATA_FILE)
        ),
        "exact_duplicate_rows_in_validated_data": exact_duplicates,
        "records_after_exact_deduplication": len(df),
        "unique_coordinate_pairs": len(
            df[["LATITUDE", "LONGITUDE"]].drop_duplicates()
        ),
        "unique_provisional_locations": len(
            df[location_keys].drop_duplicates()
        ),
        "coordinates_with_multiple_admin_labels": int(
            coordinate_locations["ambiguous_admin_labels"].sum()
        ),
        "repeated_provisional_location_date_groups": len(
            repeated_dates
        ),
        "repeated_groups_with_different_levels": int(
            (repeated_dates["distinct_levels"] > 1).sum()
        ),
        "extra_records_in_repeated_groups": int(
            (repeated_dates["records"] - 1).sum()
        ),
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"],
    )

    summary_df.to_csv(
        OUTPUT_DIR / "validation_summary.csv",
        index=False,
    )

    print("\n===== TREND LOCATION VALIDATION =====")

    for metric, value in summary.items():
        print(f"{metric}: {value}")

    print("\nReports saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
