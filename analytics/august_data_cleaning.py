from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "raw" / "August_WL_1994-2025.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "august"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_FILE = OUTPUT_DIR / "August_WL_1994-2025_clean.csv"
OUTLIER_FILE = OUTPUT_DIR / "august_extreme_readings.csv"
REPEATED_FILE = OUTPUT_DIR / "august_repeated_location_dates.csv"
SUMMARY_FILE = OUTPUT_DIR / "august_cleaning_summary.txt"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading August dataset...")

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)

# Standardise column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

# Map column names to standard names
column_mapping = {
    "state_ut": "state",
    "state": "state",
    "district": "district",
    "block": "block",
    "village": "village",
    "latitude": "latitude",
    "longitude": "longitude",
    "date": "date",
    "wl_mbgl": "water_level_mbgl",
    "wl_m_bgl": "water_level_mbgl",
    "water_level_mbgl": "water_level_mbgl"
}

df = df.rename(columns=column_mapping)

required_columns = [
    "state",
    "district",
    "block",
    "village",
    "latitude",
    "longitude",
    "date",
    "water_level_mbgl"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df[required_columns].copy()

# --------------------------------------------------
# DATA TYPE CONVERSION
# --------------------------------------------------

print("Converting data types...")

for col in ["state", "district", "block", "village"]:
    df[col] = df[col].astype("string").str.strip()

df["date"] = pd.to_datetime(
    df["date"],
    format="%Y-%m-%d",
    errors="coerce"
)

for col in ["latitude", "longitude", "water_level_mbgl"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------------------------------
# INVALID RECORDS
# --------------------------------------------------

invalid_mask = (
    df["date"].isna()
    | df["latitude"].isna()
    | df["longitude"].isna()
    | df["water_level_mbgl"].isna()
    | ~df["latitude"].between(-90, 90)
    | ~df["longitude"].between(-180, 180)
    | (df["water_level_mbgl"] < 0)
    | (df["date"].dt.month != 8)
)

invalid_records = df[invalid_mask].copy()
df = df[~invalid_mask].copy()

# --------------------------------------------------
# REMOVE EXACT DUPLICATES
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates()

duplicates_removed = before_duplicates - len(df)

# --------------------------------------------------
# FLAG EXTREME READINGS
# --------------------------------------------------

q1 = df["water_level_mbgl"].quantile(0.25)
q3 = df["water_level_mbgl"].quantile(0.75)

iqr = q3 - q1

upper_bound = q3 + (1.5 * iqr)
lower_bound = max(0, q1 - (1.5 * iqr))

df["extreme_level_flag"] = (
    (df["water_level_mbgl"] < lower_bound)
    | (df["water_level_mbgl"] > upper_bound)
)

extreme_readings = df[df["extreme_level_flag"]].copy()

# --------------------------------------------------
# IDENTIFY REPEATED LOCATION-DATE RECORDS
# --------------------------------------------------

location_columns = [
    "state",
    "district",
    "block",
    "village",
    "latitude",
    "longitude"
]

repeated_mask = df.duplicated(
    subset=location_columns + ["date"],
    keep=False
)

repeated_records = df[repeated_mask].copy()

# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------

df.to_csv(CLEANED_FILE, index=False)

extreme_readings.to_csv(
    OUTLIER_FILE,
    index=False
)

repeated_records.to_csv(
    REPEATED_FILE,
    index=False
)

# --------------------------------------------------
# CLEANING SUMMARY
# --------------------------------------------------

summary = f"""
AUGUST DATA CLEANING REPORT
========================================

Original records: {original_rows}

Invalid records removed: {len(invalid_records)}

Exact duplicate records removed: {duplicates_removed}

Final cleaned records: {len(df)}

Repeated location-date records flagged: {len(repeated_records)}

Extreme readings flagged: {len(extreme_readings)}

----------------------------------------
WATER LEVEL STATISTICS
----------------------------------------

Mean: {df['water_level_mbgl'].mean():.3f}
Median: {df['water_level_mbgl'].median():.3f}
Minimum: {df['water_level_mbgl'].min():.3f}
Maximum: {df['water_level_mbgl'].max():.3f}

----------------------------------------
EXTREME READING THRESHOLDS
----------------------------------------

Q1: {q1:.3f}
Q3: {q3:.3f}
IQR: {iqr:.3f}

Lower bound: {lower_bound:.3f}
Upper bound: {upper_bound:.3f}

----------------------------------------
DATE COVERAGE
----------------------------------------

Earliest date: {df['date'].min().date()}
Latest date: {df['date'].max().date()}

----------------------------------------
OUTPUT FILES
----------------------------------------

Cleaned dataset:
{CLEANED_FILE}

Extreme readings:
{OUTLIER_FILE}

Repeated location-date records:
{REPEATED_FILE}
"""

SUMMARY_FILE.write_text(summary, encoding="utf-8")

print(summary)
print("\nAugust cleaning completed successfully!")