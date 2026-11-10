
from pathlib import Path
import pandas as pd
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" /
    "team_dataset" / "team_combined.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "team_dataset"
AUDIT_DIR = ROOT / "outputs" / "team_duplicate_reconciliation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_COLUMNS = ["STATE_UT", "DISTRICT", "BLOCK", "VILLAGE"]

MISSING_VALUES = {"", "-", "--", "NA", "N/A", "NONE", "NULL", "NAN"}

print("Loading original integrated dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

original_count = len(df)

def normalize(value):
    if pd.isna(value):
        return "<MISSING>"

    value = unicodedata.normalize("NFKC", str(value))
    value = " ".join(value.strip().upper().split())

    if value in MISSING_VALUES:
        return "<MISSING>"

    return value


# Build normalized comparison columns without changing
# the original displayed values.

for column in ADMIN_COLUMNS:
    df[f"_{column}_KEY"] = df[column].apply(normalize)

df["_DATE_KEY"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

df["_LAT_KEY"] = pd.to_numeric(
    df["LATITUDE"], errors="coerce"
)

df["_LON_KEY"] = pd.to_numeric(
    df["LONGITUDE"], errors="coerce"
)

df["_WL_KEY"] = pd.to_numeric(
    df["WL(mbgl)"], errors="coerce"
)

# Exact standardized measurement signature.
# Source file is included so records from different
# source files are never automatically merged.

comparison_columns = (
    [f"_{column}_KEY" for column in ADMIN_COLUMNS]
    + ["_LAT_KEY", "_LON_KEY", "_DATE_KEY", "_WL_KEY",
       "source_file"]
)

# Only records with all measurement fields available
# can be safely compared using this signature.

comparable_mask = df[
    ["_LAT_KEY", "_LON_KEY", "_DATE_KEY", "_WL_KEY"]
].notna().all(axis=1)

comparable = df[comparable_mask].copy()
not_comparable = df[~comparable_mask].copy()

# Capture excess duplicate rows for audit.
duplicate_mask = comparable.duplicated(
    subset=comparison_columns,
    keep="first"
)

duplicate_rows = comparable[duplicate_mask].copy()

# Retain one copy of each standardized signature.
deduplicated = comparable.drop_duplicates(
    subset=comparison_columns,
    keep="first"
).copy()

# Preserve records that could not be compared.
candidate = pd.concat(
    [deduplicated, not_comparable],
    ignore_index=True
)

# Remove internal comparison columns from deliverables.
internal_columns = [
    column for column in candidate.columns
    if column.startswith("_")
]

candidate = candidate.drop(columns=internal_columns)

duplicate_rows = duplicate_rows.drop(
    columns=[
        column for column in duplicate_rows.columns
        if column.startswith("_")
    ]
)

# Save outputs.
candidate_file = OUTPUT_DIR / "team_combined_deduplicated_candidate.csv"

audit_file = AUDIT_DIR / "removed_exact_duplicate_candidates.csv"

candidate.to_csv(candidate_file, index=False)
duplicate_rows.to_csv(audit_file, index=False)

removed_count = original_count - len(candidate)

print("\nDEDUPLICATION SUMMARY")
print(f"Original records: {original_count:,}")
print(f"Comparable records: {len(comparable):,}")
print(f"Records not eligible for comparison: {len(not_comparable):,}")
print(f"Exact duplicate candidates removed: {removed_count:,}")
print(f"Final candidate records: {len(candidate):,}")

print("\nVALIDATION")
print(
    "Record count reconciles:",
    len(candidate) + removed_count == original_count
)

print("\nOUTPUT FILES")
print(candidate_file)
print(audit_file)

print("\nOriginal combined dataset remains unchanged.")
