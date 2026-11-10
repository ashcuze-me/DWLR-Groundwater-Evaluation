
from pathlib import Path
import pandas as pd
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "team_dataset"
    / "team_combined_deduplicated_candidate.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "team_duplicate_reconciliation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_COLUMNS = ["STATE_UT", "DISTRICT", "BLOCK", "VILLAGE"]

MISSING_VALUES = {"", "-", "--", "NA", "N/A", "NONE", "NULL", "NAN"}

print("Loading deduplicated candidate...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

original_count = len(df)

def normalize(value):
    if pd.isna(value):
        return "<MISSING>"

    value = unicodedata.normalize("NFKC", str(value))
    value = " ".join(value.strip().upper().split())

    return "<MISSING>" if value in MISSING_VALUES else value


for column in ADMIN_COLUMNS:
    df[f"_{column}"] = df[column].apply(normalize)

df["_DATE"] = pd.to_datetime(df["Date"], errors="coerce")
df["_LAT"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
df["_LON"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
df["_WL"] = pd.to_numeric(df["WL(mbgl)"], errors="coerce")

keys = (
    [f"_{column}" for column in ADMIN_COLUMNS]
    + ["_LAT", "_LON", "_DATE", "_WL", "source_file"]
)

valid = df.dropna(
    subset=["_LAT", "_LON", "_DATE", "_WL"]
).copy()

remaining_duplicates = valid.duplicated(
    subset=keys,
    keep=False
).sum()

location_keys = (
    [f"_{column}" for column in ADMIN_COLUMNS]
    + ["_LAT", "_LON", "_DATE"]
)

groups = (
    valid.groupby(location_keys, dropna=False)
    .agg(
        record_count=("_WL", "size"),
        unique_levels=("_WL", "nunique")
    )
    .reset_index()
)

repeated = groups[groups["record_count"] > 1]

identical = repeated[repeated["unique_levels"] == 1]
conflicting = repeated[repeated["unique_levels"] > 1]

print("\nDEDUPLICATED DATASET VALIDATION")
print(f"Total records: {original_count:,}")
print(f"Records usable for grouping: {len(valid):,}")
print(f"Records with incomplete measurement fields: {original_count - len(valid):,}")

print("\nDUPLICATE VALIDATION")
print(f"Remaining exact duplicate records: {remaining_duplicates:,}")

print("\nREPEATED OBSERVATION VALIDATION")
print(f"Repeated groups: {len(repeated):,}")
print(f"Identical-reading groups: {len(identical):,}")
print(f"Conflicting-reading groups: {len(conflicting):,}")

print("\nVALIDATION CHECKS")
print("No remaining exact duplicate signatures:", remaining_duplicates == 0)
print("Record count is positive:", original_count > 0)

summary = pd.DataFrame({
    "metric": [
        "total_records",
        "usable_records",
        "remaining_exact_duplicate_records",
        "repeated_groups",
        "identical_reading_groups",
        "conflicting_reading_groups"
    ],
    "value": [
        original_count,
        len(valid),
        remaining_duplicates,
        len(repeated),
        len(identical),
        len(conflicting)
    ]
})

summary.to_csv(
    OUTPUT_DIR / "deduplicated_candidate_validation.csv",
    index=False
)

print("\nValidation report saved.")
