
from pathlib import Path
import pandas as pd
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" /
    "team_dataset" / "team_combined.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "team_duplicate_reconciliation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_COLUMNS = ["STATE_UT", "DISTRICT", "BLOCK", "VILLAGE"]

MISSING_VALUES = {"", "-", "--", "NA", "N/A", "NONE", "NULL", "NAN"}

# --------------------------------------------------
# LOAD AND NORMALIZE
# --------------------------------------------------

print("Loading team dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
df["WL(mbgl)"] = pd.to_numeric(df["WL(mbgl)"], errors="coerce")


def normalize(value):
    if pd.isna(value):
        return pd.NA

    value = unicodedata.normalize("NFKC", str(value))
    value = " ".join(value.strip().upper().split())

    if value in MISSING_VALUES:
        return pd.NA

    return value


for column in ADMIN_COLUMNS:
    df[f"{column}_NORM"] = df[column].apply(normalize)

NORMALIZED_ADMIN = [f"{column}_NORM" for column in ADMIN_COLUMNS]

# --------------------------------------------------
# DEFINE GROUPING KEYS
# --------------------------------------------------

# Fill missing identifiers explicitly so incomplete records
# can still be investigated without treating them as real IDs.

for column in NORMALIZED_ADMIN:
    df[column] = df[column].fillna("<MISSING>")

LOCATION_DATE_KEYS = (
    NORMALIZED_ADMIN
    + ["LATITUDE", "LONGITUDE", "Date"]
)

MEASUREMENT_KEYS = LOCATION_DATE_KEYS + ["WL(mbgl)"]

valid = df.dropna(
    subset=["LATITUDE", "LONGITUDE", "Date", "WL(mbgl)"]
).copy()

print(f"Total input records: {len(df):,}")
print(f"Records usable for reconciliation: {len(valid):,}")

# --------------------------------------------------
# 1. EXACT DUPLICATE RECONCILIATION
# --------------------------------------------------

# A measurement signature includes all standardized measurement
# fields, but excludes source_file.
# Therefore, identical signatures across files are cross-file
# duplicate candidates.

exact_groups = (
    valid.groupby(MEASUREMENT_KEYS, dropna=False)
    .agg(
        record_count=("source_file", "size"),
        source_file_count=("source_file", "nunique"),
        source_files=("source_file",
                      lambda x: " | ".join(sorted(set(x))))
    )
    .reset_index()
)

exact_duplicates = exact_groups[
    exact_groups["record_count"] > 1
].copy()

within_file_exact = (
    valid.groupby(["source_file"] + MEASUREMENT_KEYS, dropna=False)
    .size()
    .reset_index(name="record_count")
)

within_file_exact = within_file_exact[
    within_file_exact["record_count"] > 1
].copy()

cross_file_exact = exact_duplicates[
    exact_duplicates["source_file_count"] > 1
].copy()

exact_duplicates.to_csv(
    OUTPUT_DIR / "exact_duplicate_measurement_groups.csv",
    index=False
)

within_file_exact.to_csv(
    OUTPUT_DIR / "within_file_exact_duplicate_groups.csv",
    index=False
)

cross_file_exact.to_csv(
    OUTPUT_DIR / "cross_file_exact_duplicate_groups.csv",
    index=False
)

# --------------------------------------------------
# 2. IDENTICAL-READING GROUPS
# --------------------------------------------------

location_groups = (
    valid.groupby(LOCATION_DATE_KEYS, dropna=False)
    .agg(
        record_count=("WL(mbgl)", "size"),
        unique_levels=("WL(mbgl)", "nunique"),
        min_level=("WL(mbgl)", "min"),
        max_level=("WL(mbgl)", "max"),
        source_file_count=("source_file", "nunique"),
        source_files=("source_file",
                      lambda x: " | ".join(sorted(set(x))))
    )
    .reset_index()
)

identical_groups = location_groups[
    (location_groups["record_count"] > 1)
    & (location_groups["unique_levels"] == 1)
].copy()

conflicting_groups = location_groups[
    (location_groups["record_count"] > 1)
    & (location_groups["unique_levels"] > 1)
].copy()

identical_groups.to_csv(
    OUTPUT_DIR / "identical_reading_groups.csv",
    index=False
)

conflicting_groups.to_csv(
    OUTPUT_DIR / "conflicting_reading_groups.csv",
    index=False
)

# --------------------------------------------------
# 3. CONFLICTING GROUP SIZE DISTRIBUTION
# --------------------------------------------------

conflict_distribution = (
    conflicting_groups.groupby("unique_levels")
    .agg(
        groups=("record_count", "size"),
        total_records=("record_count", "sum")
    )
    .reset_index()
    .sort_values("unique_levels")
)

conflict_distribution.to_csv(
    OUTPUT_DIR / "conflict_reading_count_distribution.csv",
    index=False
)

# --------------------------------------------------
# 4. SUMMARY
# --------------------------------------------------

summary = {
    "total_input_records": len(df),
    "records_usable_for_reconciliation": len(valid),

    "exact_duplicate_groups": len(exact_duplicates),
    "records_in_exact_duplicate_groups":
        int(exact_duplicates["record_count"].sum()),
    "excess_exact_duplicate_records":
        int((exact_duplicates["record_count"] - 1).sum()),

    "within_file_exact_duplicate_groups": len(within_file_exact),
    "cross_file_exact_duplicate_groups": len(cross_file_exact),

    "identical_reading_groups": len(identical_groups),
    "records_in_identical_reading_groups":
        int(identical_groups["record_count"].sum()),

    "conflicting_reading_groups": len(conflicting_groups),
    "records_in_conflicting_reading_groups":
        int(conflicting_groups["record_count"].sum()),

    "conflicting_groups_spanning_multiple_files":
        int((conflicting_groups["source_file_count"] > 1).sum())
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["metric", "value"]
)

summary_df.to_csv(
    OUTPUT_DIR / "duplicate_reconciliation_summary.csv",
    index=False
)

# --------------------------------------------------
# 5. PRINT RESULTS
# --------------------------------------------------

print("\nEXACT DUPLICATE ANALYSIS")

print(f"Exact duplicate groups: {summary['exact_duplicate_groups']:,}")
print(
    "Records in exact duplicate groups: "
    f"{summary['records_in_exact_duplicate_groups']:,}"
)
print(
    "Excess exact duplicate records: "
    f"{summary['excess_exact_duplicate_records']:,}"
)
print(
    "Within-file exact duplicate groups: "
    f"{summary['within_file_exact_duplicate_groups']:,}"
)
print(
    "Cross-file exact duplicate groups: "
    f"{summary['cross_file_exact_duplicate_groups']:,}"
)

print("\nIDENTICAL READING ANALYSIS")

print(f"Identical reading groups: {len(identical_groups):,}")
print(
    "Records in identical reading groups: "
    f"{summary['records_in_identical_reading_groups']:,}"
)

print("\nCONFLICTING READING ANALYSIS")

print(f"Conflicting groups: {len(conflicting_groups):,}")
print(
    "Records in conflicting groups: "
    f"{summary['records_in_conflicting_reading_groups']:,}"
)
print(
    "Conflicting groups spanning multiple files: "
    f"{summary['conflicting_groups_spanning_multiple_files']:,}"
)

print("\nCONFLICTING GROUP SIZE DISTRIBUTION")
print(conflict_distribution.to_string(index=False))

print("\nAnalysis complete.")
print(f"Outputs saved to: {OUTPUT_DIR}")
