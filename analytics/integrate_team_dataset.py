
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "raw" / "team_formatted_data"
OUTPUT_DIR = ROOT / "data" / "processed" / "team_dataset"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = [
    "STATE_UT",
    "DISTRICT",
    "BLOCK",
    "VILLAGE",
    "LATITUDE",
    "LONGITUDE",
    "Date",
    "WL(mbgl)",
]

OUTPUT_FILE = OUTPUT_DIR / "team_combined.csv"
REPORT_FILE = OUTPUT_DIR / "team_audit_report.txt"
INVALID_FILE = OUTPUT_DIR / "team_invalid_dates.csv"


def load_files():
    files = sorted(INPUT_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {INPUT_DIR}"
        )

    print(f"Found {len(files)} CSV files.")

    frames = []

    for index, file in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] Reading {file.name}")

        df = pd.read_csv(file)

        df.columns = df.columns.str.strip()

        missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

        if missing_columns:
            print(f"Skipping {file.name}: missing {missing_columns}")
            continue

        df = df[EXPECTED_COLUMNS].copy()

        df["source_file"] = file.name

        frames.append(df)

    if not frames:
        raise ValueError("No valid CSV files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)

    return combined, len(files), len(frames)


def prepare_data(df):
    df["LATITUDE"] = pd.to_numeric(
        df["LATITUDE"], errors="coerce"
    )

    df["LONGITUDE"] = pd.to_numeric(
        df["LONGITUDE"], errors="coerce"
    )

    df["WL(mbgl)"] = pd.to_numeric(
        df["WL(mbgl)"], errors="coerce"
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    return df


def generate_report(df, total_files, loaded_files):
    lines = []

    lines.append("=" * 65)
    lines.append("TEAM GROUNDWATER DATASET AUDIT")
    lines.append("=" * 65)

    lines.append(f"CSV files found: {total_files}")
    lines.append(f"CSV files loaded: {loaded_files}")
    lines.append(f"Total records: {len(df):,}")
    lines.append(f"Columns: {len(df.columns)}")

    lines.append("\nDATE COVERAGE")

    valid_dates = df["Date"].dropna()

    if not valid_dates.empty:
        lines.append(f"Earliest date: {valid_dates.min()}")
        lines.append(f"Latest date: {valid_dates.max()}")
        lines.append(f"Unique dates: {valid_dates.nunique()}")

    lines.append("\nGEOGRAPHICAL COVERAGE")

    lines.append(
        f"States/UTs: {df['STATE_UT'].nunique(dropna=True)}"
    )

    lines.append(
        f"Districts: {df['DISTRICT'].nunique(dropna=True)}"
    )

    lines.append(
        f"Blocks: {df['BLOCK'].nunique(dropna=True)}"
    )

    lines.append(
        f"Villages: {df['VILLAGE'].nunique(dropna=True)}"
    )

    coordinates = df[["LATITUDE", "LONGITUDE"]].dropna()

    lines.append(
        f"Unique coordinate pairs: {len(coordinates.drop_duplicates()):,}"
    )

    lines.append("\nMISSING VALUES")

    for column in EXPECTED_COLUMNS:
        missing = df[column].isna().sum()

        lines.append(
            f"{column}: {missing:,} "
            f"({missing / len(df) * 100:.2f}%)"
        )

    lines.append("\nDUPLICATE RECORDS")

    duplicate_count = df.duplicated(
        subset=EXPECTED_COLUMNS
    ).sum()

    lines.append(
        f"Exact duplicate records: {duplicate_count:,}"
    )

    lines.append("\nINVALID VALUES")

    lines.append(
        f"Negative groundwater readings: "
        f"{(df['WL(mbgl)'] < 0).sum():,}"
    )

    lines.append(
        f"Zero groundwater readings: "
        f"{(df['WL(mbgl)'] == 0).sum():,}"
    )

    invalid_lat = (
        (df["LATITUDE"] < -90) |
        (df["LATITUDE"] > 90)
    ).sum()

    invalid_lon = (
        (df["LONGITUDE"] < -180) |
        (df["LONGITUDE"] > 180)
    ).sum()

    lines.append(f"Invalid latitudes: {invalid_lat:,}")
    lines.append(f"Invalid longitudes: {invalid_lon:,}")

    lines.append("\nGROUNDWATER LEVEL STATISTICS")

    lines.append(
        df["WL(mbgl)"].describe().to_string()
    )

    lines.append("\nMONTHLY DISTRIBUTION")

    monthly = (
        df["Date"]
        .dt.month
        .value_counts()
        .sort_index()
    )

    for month, count in monthly.items():
        lines.append(f"Month {month}: {count:,}")

    lines.append("\nSOURCE FILE DISTRIBUTION")

    source_counts = df["source_file"].value_counts()

    for source, count in source_counts.items():
        lines.append(f"{source}: {count:,}")

    lines.append("\n" + "=" * 65)

    return "\n".join(lines)


def main():
    print("\nSTARTING TEAM DATA INTEGRATION")
    print("=" * 65)

    df, total_files, loaded_files = load_files()

    print(f"\nCombined records: {len(df):,}")

    df = prepare_data(df)

    invalid_dates = df[df["Date"].isna()].copy()

    if not invalid_dates.empty:
        invalid_dates.to_csv(
            INVALID_FILE,
            index=False
        )

        print(
            f"Invalid date records saved: "
            f"{len(invalid_dates):,}"
        )

    report = generate_report(
        df,
        total_files,
        loaded_files
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8"
    )

    print("\n" + report)

    print("\nOUTPUT FILES")
    print(f"Combined dataset: {OUTPUT_FILE}")
    print(f"Audit report: {REPORT_FILE}")

    if not invalid_dates.empty:
        print(f"Invalid dates: {INVALID_FILE}")

    print("\nINTEGRATION COMPLETE")


if __name__ == "__main__":
    main()
