
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" /
    "team_dataset" / "team_combined.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "team_dataset_anomalies"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "anomaly_report.txt"

# Broad bounding box covering mainland India and its island territories.
INDIA_BOUNDS = {
    "min_lat": 5.0,
    "max_lat": 38.0,
    "min_lon": 68.0,
    "max_lon": 98.0,
}

LOCATION_COLUMNS = [
    "STATE_UT",
    "DISTRICT",
    "BLOCK",
    "VILLAGE",
    "LATITUDE",
    "LONGITUDE",
]

EXACT_COLUMNS = [
    "STATE_UT",
    "DISTRICT",
    "BLOCK",
    "VILLAGE",
    "LATITUDE",
    "LONGITUDE",
    "Date",
    "WL(mbgl)",
]


def load_data():
    print("Loading consolidated team dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    for col in ["LATITUDE", "LONGITUDE", "WL(mbgl)"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    print(f"Loaded {len(df):,} records.")

    return df


def analyse_water_levels(df):
    print("\nAnalysing groundwater readings...")

    valid = df["WL(mbgl)"].dropna()

    negative = df[df["WL(mbgl)"] < 0].copy()
    zero = df[df["WL(mbgl)"] == 0].copy()

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    iqr_extremes = df[
        (df["WL(mbgl)"] < lower) |
        (df["WL(mbgl)"] > upper)
    ].copy()

    p001 = valid.quantile(0.001)
    p999 = valid.quantile(0.999)

    percentile_extremes = df[
        (df["WL(mbgl)"] < p001) |
        (df["WL(mbgl)"] > p999)
    ].copy()

    highest = df.nlargest(50, "WL(mbgl)")
    lowest = df.nsmallest(50, "WL(mbgl)")

    negative.to_csv(
        OUTPUT_DIR / "negative_readings.csv",
        index=False
    )

    zero.to_csv(
        OUTPUT_DIR / "zero_readings.csv",
        index=False
    )

    iqr_extremes.to_csv(
        OUTPUT_DIR / "iqr_extreme_readings.csv",
        index=False
    )

    percentile_extremes.to_csv(
        OUTPUT_DIR / "percentile_extreme_readings.csv",
        index=False
    )

    highest.to_csv(
        OUTPUT_DIR / "highest_50_readings.csv",
        index=False
    )

    lowest.to_csv(
        OUTPUT_DIR / "lowest_50_readings.csv",
        index=False
    )

    return {
        "negative": len(negative),
        "zero": len(zero),
        "iqr_extremes": len(iqr_extremes),
        "percentile_extremes": len(percentile_extremes),
        "q1": q1,
        "q3": q3,
        "iqr_lower": lower,
        "iqr_upper": upper,
        "p001": p001,
        "p999": p999,
        "minimum": valid.min(),
        "maximum": valid.max(),
        "mean": valid.mean(),
        "median": valid.median(),
    }


def analyse_coordinates(df):
    print("\nChecking geographical coordinates...")

    missing = df[
        df["LATITUDE"].isna() |
        df["LONGITUDE"].isna()
    ].copy()

    invalid_global = df[
        (df["LATITUDE"] < -90) |
        (df["LATITUDE"] > 90) |
        (df["LONGITUDE"] < -180) |
        (df["LONGITUDE"] > 180)
    ].copy()

    outside_india = df[
        df["LATITUDE"].notna() &
        df["LONGITUDE"].notna() &
        (
            (df["LATITUDE"] < INDIA_BOUNDS["min_lat"]) |
            (df["LATITUDE"] > INDIA_BOUNDS["max_lat"]) |
            (df["LONGITUDE"] < INDIA_BOUNDS["min_lon"]) |
            (df["LONGITUDE"] > INDIA_BOUNDS["max_lon"])
        )
    ].copy()

    missing.to_csv(
        OUTPUT_DIR / "missing_coordinates.csv",
        index=False
    )

    invalid_global.to_csv(
        OUTPUT_DIR / "invalid_global_coordinates.csv",
        index=False
    )

    outside_india.to_csv(
        OUTPUT_DIR / "coordinates_outside_india_bounds.csv",
        index=False
    )

    return {
        "missing": len(missing),
        "invalid_global": len(invalid_global),
        "outside_india": len(outside_india),
    }


def analyse_duplicates(df):
    print("\nChecking duplicate observations...")

    # Compare original observation fields, excluding source_file.
    exact_mask = df.duplicated(
        subset=EXACT_COLUMNS,
        keep=False
    )

    exact_duplicates = df[exact_mask].copy()

    # Identify duplicate rows within the same source file.
    within_mask = df.duplicated(
        subset=EXACT_COLUMNS + ["source_file"],
        keep=False
    )

    within_file = df[within_mask].copy()

    # Identify identical observations appearing across different files.
    cross_groups = (
        df.groupby(EXACT_COLUMNS, dropna=False)["source_file"]
        .nunique()
    )

    cross_keys = cross_groups[cross_groups > 1].index

    cross_file = df.merge(
        pd.DataFrame(
            list(cross_keys),
            columns=EXACT_COLUMNS
        ),
        on=EXACT_COLUMNS,
        how="inner"
    )

    exact_duplicates.to_csv(
        OUTPUT_DIR / "exact_duplicate_records.csv",
        index=False
    )

    within_file.to_csv(
        OUTPUT_DIR / "within_file_duplicates.csv",
        index=False
    )

    cross_file.to_csv(
        OUTPUT_DIR / "cross_file_duplicates.csv",
        index=False
    )

    return {
        "exact_duplicate_rows": len(exact_duplicates),
        "within_file_rows": len(within_file),
        "cross_file_rows": len(cross_file),
        "cross_file_groups": len(cross_keys),
    }


def analyse_repeated_location_dates(df):
    print("\nChecking repeated location-date observations...")

    valid = df.dropna(
        subset=["LATITUDE", "LONGITUDE", "Date"]
    ).copy()

    valid["location_date_key"] = (
        valid["LATITUDE"].round(4).astype(str) + "_" +
        valid["LONGITUDE"].round(4).astype(str) + "_" +
        valid["Date"].dt.strftime("%Y-%m-%d")
    )

    counts = valid["location_date_key"].value_counts()

    repeated_keys = counts[counts > 1].index

    repeated = valid[
        valid["location_date_key"].isin(repeated_keys)
    ].copy()

    repeated.to_csv(
        OUTPUT_DIR / "repeated_location_date_records.csv",
        index=False
    )

    # Distinguish repeated records with identical levels
    # from records with conflicting levels.
    groups = valid.groupby(
        "location_date_key"
    )["WL(mbgl)"]

    repeated_groups = groups.size()
    level_counts = groups.nunique(dropna=True)

    conflicting = level_counts[
        (level_counts > 1) &
        (repeated_groups > 1)
    ]

    return {
        "repeated_groups": len(repeated_keys),
        "repeated_rows": len(repeated),
        "conflicting_level_groups": len(conflicting),
    }


def analyse_temporal_coverage(df):
    print("\nAnalysing temporal coverage...")

    valid = df.dropna(subset=["Date"]).copy()

    valid["year"] = valid["Date"].dt.year
    valid["month"] = valid["Date"].dt.month

    yearly = (
        valid.groupby("year")
        .agg(
            records=("Date", "size"),
            unique_dates=("Date", "nunique"),
            locations=("LATITUDE", "count"),
        )
        .reset_index()
    )

    monthly = (
        valid.groupby("month")
        .size()
        .reindex(range(1, 13), fill_value=0)
        .rename("records")
        .reset_index()
    )

    yearly.to_csv(
        OUTPUT_DIR / "yearly_coverage.csv",
        index=False
    )

    monthly.to_csv(
        OUTPUT_DIR / "monthly_coverage.csv",
        index=False
    )

    return yearly, monthly


def analyse_source_files(df):
    print("\nChecking source file quality...")

    summary = (
        df.groupby("source_file")
        .agg(
            records=("source_file", "size"),
            missing_wl=("WL(mbgl)", lambda x: x.isna().sum()),
            negative_wl=("WL(mbgl)", lambda x: (x < 0).sum()),
            missing_latitude=("LATITUDE", lambda x: x.isna().sum()),
            missing_longitude=("LONGITUDE", lambda x: x.isna().sum()),
            missing_dates=("Date", lambda x: x.isna().sum()),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "source_file_quality.csv",
        index=False
    )

    return summary


def main():
    print("=" * 65)
    print("TEAM DATASET ANOMALY INVESTIGATION")
    print("=" * 65)

    df = load_data()

    water = analyse_water_levels(df)
    coordinates = analyse_coordinates(df)
    duplicates = analyse_duplicates(df)
    repeated = analyse_repeated_location_dates(df)

    yearly, monthly = analyse_temporal_coverage(df)
    sources = analyse_source_files(df)

    lines = []

    lines.append("=" * 65)
    lines.append("TEAM DATASET ANOMALY INVESTIGATION REPORT")
    lines.append("=" * 65)

    lines.append(f"\nTotal records: {len(df):,}")
    lines.append(f"Source files: {df['source_file'].nunique()}")

    lines.append("\nWATER LEVEL ANOMALIES")

    for key, value in water.items():
        lines.append(f"{key}: {value}")

    lines.append("\nCOORDINATE ANOMALIES")

    for key, value in coordinates.items():
        lines.append(f"{key}: {value:,}")

    lines.append("\nDUPLICATE ANALYSIS")

    for key, value in duplicates.items():
        lines.append(f"{key}: {value:,}")

    lines.append("\nREPEATED LOCATION-DATE ANALYSIS")

    for key, value in repeated.items():
        lines.append(f"{key}: {value:,}")

    lines.append("\nTEMPORAL COVERAGE")

    if not yearly.empty:
        lines.append(
            f"Earliest year: {yearly['year'].min()}"
        )
        lines.append(
            f"Latest year: {yearly['year'].max()}"
        )

    lines.append("\nMONTHLY RECORD COUNTS")

    for _, row in monthly.iterrows():
        lines.append(
            f"Month {int(row['month'])}: "
            f"{int(row['records']):,}"
        )

    lines.append("\nSOURCE FILES WITH MOST NEGATIVE READINGS")

    top_negative = sources.sort_values(
        "negative_wl",
        ascending=False
    ).head(10)

    lines.append(top_negative.to_string(index=False))

    lines.append("\n" + "=" * 65)

    report = "\n".join(lines)

    REPORT_FILE.write_text(
        report,
        encoding="utf-8"
    )

    print("\n" + report)

    print("\nOUTPUT DIRECTORY:")
    print(OUTPUT_DIR)

    print("\nANOMALY INVESTIGATION COMPLETE")


if __name__ == "__main__":
    main()
