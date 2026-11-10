
from pathlib import Path
import pandas as pd
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" /
    "team_dataset" / "team_combined.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "team_repeated_observations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["WL(mbgl)"] = pd.to_numeric(df["WL(mbgl)"], errors="coerce")
df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

missing_values = {"", "-", "--", "NA", "N/A", "NONE", "NULL", "NAN"}

def normalize(value):
    if pd.isna(value):
        return pd.NA

    value = unicodedata.normalize("NFKC", str(value))
    value = " ".join(value.strip().upper().split())

    return pd.NA if value in missing_values else value

admin_columns = ["STATE_UT", "DISTRICT", "BLOCK", "VILLAGE"]

for column in admin_columns:
    df[f"{column}_NORM"] = df[column].apply(normalize)

# Administrative completeness
norm_columns = [f"{column}_NORM" for column in admin_columns]

df["missing_admin_count"] = df[norm_columns].isna().sum(axis=1)

df["location_status"] = "COMPLETE"
df.loc[df["missing_admin_count"] > 0, "location_status"] = "INCOMPLETE"

# Records without sufficient information for grouping
valid = df.dropna(
    subset=[
        "LATITUDE",
        "LONGITUDE",
        "Date",
        "WL(mbgl)"
    ]
).copy()

group_columns = [
    *norm_columns,
    "LATITUDE",
    "LONGITUDE",
    "Date"
]

valid[group_columns[:4]] = valid[group_columns[:4]].fillna("<MISSING>")

# Group by administrative identifiers, coordinates and date
groups = (
    valid.groupby(group_columns, dropna=False)
    .agg(
        record_count=("WL(mbgl)", "size"),
        unique_levels=("WL(mbgl)", "nunique")
    )
    .reset_index()
)

# Classify each group
groups["observation_status"] = "VALID_SINGLE"

groups.loc[
    (groups["record_count"] > 1) &
    (groups["unique_levels"] == 1),
    "observation_status"
] = "REPEATED_IDENTICAL"

groups.loc[
    (groups["record_count"] > 1) &
    (groups["unique_levels"] > 1),
    "observation_status"
] = "REPEATED_CONFLICTING"

# Summarize groups and records separately
summary = (
    groups.groupby("observation_status")
    .agg(
        groups=("record_count", "size"),
        records=("record_count", "sum")
    )
    .reset_index()
)

summary["extra_records"] = (
    summary["records"] - summary["groups"]
)

summary.to_csv(
    OUTPUT_DIR / "observation_quality_summary.csv",
    index=False
)

groups.to_csv(
    OUTPUT_DIR / "observation_quality_groups.csv",
    index=False
)

# Incomplete location record count
incomplete_count = (
    valid["location_status"] == "INCOMPLETE"
).sum()

print("\nOBSERVATION QUALITY SUMMARY")
print(summary.to_string(index=False))

print("\nADDITIONAL INFORMATION")
print(f"Total input records: {len(df):,}")
print(f"Records usable for grouping: {len(valid):,}")
print(f"Records with incomplete identifiers: {incomplete_count:,}")
print(f"Records excluded from grouping: {len(df) - len(valid):,}")

print("\nSaved outputs to:", OUTPUT_DIR)
