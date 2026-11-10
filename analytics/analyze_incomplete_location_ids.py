
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" /
    "team_dataset" / "team_combined.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "team_dataset_anomalies"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_missing_identifier(series):
    values = series.fillna("").astype(str).str.strip().str.upper()
    return values.isin(["", "-", "NA", "N/A", "NONE", "NULL"])


def main():
    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    admin_columns = [
        "STATE_UT",
        "DISTRICT",
        "BLOCK",
        "VILLAGE",
    ]

    for col in admin_columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Identify missing administrative identifiers.
    for col in admin_columns:
        df[f"{col}_MISSING"] = is_missing_identifier(df[col])

    missing_summary = []

    for col in admin_columns:
        missing_summary.append({
            "column": col,
            "missing_records": int(df[f"{col}_MISSING"].sum()),
            "missing_percentage": (
                df[f"{col}_MISSING"].mean() * 100
            ),
        })

    summary = pd.DataFrame(missing_summary)

    print("\nMISSING ADMINISTRATIVE IDENTIFIERS")
    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "missing_admin_identifier_summary.csv",
        index=False
    )

    # Focus on records with missing block or village.
    incomplete = df[
        df["BLOCK_MISSING"] | df["VILLAGE_MISSING"]
    ].copy()

    incomplete.to_csv(
        OUTPUT_DIR / "incomplete_location_records.csv",
        index=False
    )

    # Analyse by source file.
    source_summary = (
        incomplete.groupby("source_file")
        .agg(
            records=("source_file", "size"),
            missing_blocks=("BLOCK_MISSING", "sum"),
            missing_villages=("VILLAGE_MISSING", "sum"),
        )
        .reset_index()
        .sort_values("records", ascending=False)
    )

    source_summary.to_csv(
        OUTPUT_DIR / "incomplete_location_by_source.csv",
        index=False
    )

    print("\nINCOMPLETE RECORDS")
    print(f"Total: {len(incomplete):,}")

    print("\nSOURCE FILE BREAKDOWN")
    print(source_summary.head(20).to_string(index=False))

    # Investigate repeated readings where identifiers are incomplete.
    valid = incomplete.dropna(
        subset=["LATITUDE", "LONGITUDE", "Date"]
    ).copy()

    valid["Date"] = pd.to_datetime(
        valid["Date"], errors="coerce"
    )

    valid = valid.dropna(subset=["Date"])

    group_columns = [
        "STATE_UT",
        "DISTRICT",
        "BLOCK",
        "VILLAGE",
        "LATITUDE",
        "LONGITUDE",
        "Date",
    ]

    repeated = (
        valid.groupby(group_columns, dropna=False)
        .agg(
            record_count=("WL(mbgl)", "size"),
            unique_levels=("WL(mbgl)", "nunique"),
            min_level=("WL(mbgl)", "min"),
            max_level=("WL(mbgl)", "max"),
            source_file=("source_file", "first"),
        )
        .reset_index()
    )

    conflicts = repeated[
        (repeated["record_count"] > 1) &
        (repeated["unique_levels"] > 1)
    ]

    conflicts.to_csv(
        OUTPUT_DIR / "incomplete_location_conflicts.csv",
        index=False
    )

    print("\nINCOMPLETE LOCATION CONFLICTS")
    print(f"Conflicting groups: {len(conflicts):,}")

    print("\nSAMPLE CONFLICTS")
    print(conflicts.head(25).to_string(index=False))

    print("\nAnalysis complete.")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
