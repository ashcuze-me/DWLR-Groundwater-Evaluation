
from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "team_dataset"
    / "team_combined_deduplicated_candidate.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "team_dataset"
REPORT_DIR = ROOT / "outputs" / "team_data_quality"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_COLUMNS = ["STATE_UT", "DISTRICT", "BLOCK", "VILLAGE"]

MISSING_VALUES = {"", "-", "--", "NA", "N/A", "NONE", "NULL", "NAN"}

print("Loading deduplicated candidate dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

initial_records = len(df)


# --------------------------------------------------
# 1. NORMALIZE AND VALIDATE FIELDS
# --------------------------------------------------

def normalize_identifier(value):
    if pd.isna(value):
        return pd.NA

    value = unicodedata.normalize("NFKC", str(value))
    value = " ".join(value.strip().upper().split())

    if value in MISSING_VALUES:
        return pd.NA

    return value


for column in ADMIN_COLUMNS:
    df[f"{column}_NORMALIZED"] = (
        df[column].apply(normalize_identifier)
    )

df["Date_PARSED"] = pd.to_datetime(
    df["Date"], errors="coerce", format="%Y-%m-%d"
)

df["LATITUDE_NUMERIC"] = pd.to_numeric(
    df["LATITUDE"], errors="coerce"
)

df["LONGITUDE_NUMERIC"] = pd.to_numeric(
    df["LONGITUDE"], errors="coerce"
)

df["WL_NUMERIC"] = pd.to_numeric(
    df["WL(mbgl)"], errors="coerce"
)


# --------------------------------------------------
# 2. ADMINISTRATIVE IDENTIFIER FLAGS
# --------------------------------------------------

for column in ADMIN_COLUMNS:
    df[f"missing_{column.lower()}"] = (
        df[f"{column}_NORMALIZED"].isna()
    )

df["incomplete_location"] = df[
    [f"missing_{column.lower()}" for column in ADMIN_COLUMNS]
].any(axis=1)


# --------------------------------------------------
# 3. MEASUREMENT QUALITY FLAGS
# --------------------------------------------------

df["missing_date"] = df["Date_PARSED"].isna()

df["missing_latitude"] = df["LATITUDE_NUMERIC"].isna()
df["missing_longitude"] = df["LONGITUDE_NUMERIC"].isna()

df["missing_groundwater_level"] = df["WL_NUMERIC"].isna()

df["invalid_latitude"] = (
    df["LATITUDE_NUMERIC"].notna()
    & ~df["LATITUDE_NUMERIC"].between(-90, 90)
)

df["invalid_longitude"] = (
    df["LONGITUDE_NUMERIC"].notna()
    & ~df["LONGITUDE_NUMERIC"].between(-180, 180)
)

df["negative_groundwater_level"] = (
    df["WL_NUMERIC"] < 0
)

df["zero_groundwater_level"] = (
    df["WL_NUMERIC"] == 0
)


# --------------------------------------------------
# 4. STATISTICAL EXTREME FLAGS
# --------------------------------------------------

valid_levels = df["WL_NUMERIC"].dropna()

lower_limit = valid_levels.quantile(0.001)
upper_limit = valid_levels.quantile(0.999)

df["extreme_groundwater_level"] = (
    df["WL_NUMERIC"].notna()
    & (
        (df["WL_NUMERIC"] < lower_limit)
        | (df["WL_NUMERIC"] > upper_limit)
    )
)

print("\nSTATISTICAL THRESHOLDS")
print(f"0.1 percentile: {lower_limit:.4f}")
print(f"99.9 percentile: {upper_limit:.4f}")


# --------------------------------------------------
# 5. CONFLICTING OBSERVATION FLAGS
# --------------------------------------------------

group_columns = [
    f"{column}_NORMALIZED" for column in ADMIN_COLUMNS
] + [
    "LATITUDE_NUMERIC",
    "LONGITUDE_NUMERIC",
    "Date_PARSED"
]

valid_for_grouping = df.dropna(
    subset=[
        "LATITUDE_NUMERIC",
        "LONGITUDE_NUMERIC",
        "Date_PARSED",
        "WL_NUMERIC"
    ]
).copy()

# Missing administrative identifiers remain explicit.
for column in group_columns[:4]:
    valid_for_grouping[column] = (
        valid_for_grouping[column].fillna("<MISSING>")
    )

valid_for_grouping["unique_levels"] = (
    valid_for_grouping.groupby(
        group_columns, dropna=False
    )["WL_NUMERIC"].transform("nunique")
)

df["conflicting_observation"] = False

df.loc[
    valid_for_grouping.index,
    "conflicting_observation"
] = valid_for_grouping["unique_levels"].gt(1)


# --------------------------------------------------
# 6. OVERALL QUALITY FLAGS
# --------------------------------------------------

df["measurement_incomplete"] = (
    df["missing_date"]
    | df["missing_latitude"]
    | df["missing_longitude"]
    | df["missing_groundwater_level"]
    | df["invalid_latitude"]
    | df["invalid_longitude"]
)

# These flags indicate review requirements, not automatic
# exclusion from the final dataset.
review_columns = [
    "incomplete_location",
    "measurement_incomplete",
    "negative_groundwater_level",
    "zero_groundwater_level",
    "extreme_groundwater_level",
    "conflicting_observation"
]

df["requires_quality_review"] = df[review_columns].any(axis=1)


# --------------------------------------------------
# 7. SAVE FLAGGED DATASET
# --------------------------------------------------

# Keep the original source fields and append quality flags.
# Do not overwrite the input candidate.

flagged_file = OUTPUT_DIR / "team_quality_flagged.csv"

df.to_csv(flagged_file, index=False)


# --------------------------------------------------
# 8. GENERATE QUALITY REPORT
# --------------------------------------------------

flag_columns = [
    "incomplete_location",
    "missing_state_ut",
    "missing_district",
    "missing_block",
    "missing_village",
    "missing_date",
    "missing_latitude",
    "missing_longitude",
    "missing_groundwater_level",
    "invalid_latitude",
    "invalid_longitude",
    "negative_groundwater_level",
    "zero_groundwater_level",
    "extreme_groundwater_level",
    "conflicting_observation",
    "measurement_incomplete",
    "requires_quality_review"
]

summary = pd.DataFrame([
    {
        "quality_flag": column,
        "records_flagged": int(df[column].sum()),
        "percentage": round(df[column].mean() * 100, 4)
    }
    for column in flag_columns
])

summary.to_csv(
    REPORT_DIR / "quality_flag_summary.csv",
    index=False
)

review_records = df[df["requires_quality_review"]].copy()

review_records.to_csv(
    REPORT_DIR / "records_requiring_quality_review.csv",
    index=False
)


# --------------------------------------------------
# 9. FINAL REPORT
# --------------------------------------------------

print("\nDATA QUALITY PIPELINE SUMMARY")
print(f"Input records: {initial_records:,}")
print(f"Output records: {len(df):,}")

print("\nQUALITY FLAGS")
print(summary.to_string(index=False))

print("\nADDITIONAL SUMMARY")
print(f"Records requiring review: {len(review_records):,}")
print(
    "Records preserved:",
    len(df) == initial_records
)

print("\nOUTPUT FILES")
print(flagged_file)
print(REPORT_DIR / "quality_flag_summary.csv")
print(REPORT_DIR / "records_requiring_quality_review.csv")

print("\nData quality pipeline completed.")
