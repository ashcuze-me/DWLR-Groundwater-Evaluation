from pathlib import Path
import pandas as pd
import re
import unicodedata

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "august"
    / "August_WL_1994-2025_clean.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "august"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "August_WL_1994-2025_standardized.csv"
MAPPING_FILE = OUTPUT_DIR / "august_location_mapping.csv"
CONFLICT_FILE = OUTPUT_DIR / "august_coordinate_conflicts.csv"
SUMMARY_FILE = OUTPUT_DIR / "august_standardization_report.txt"

# --------------------------------------------------
# NORMALISATION FUNCTION
# --------------------------------------------------

def normalize_name(value):
    """
    Standardise administrative names without changing
    their underlying meaning.
    """

    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip().upper()

    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )

    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)

    return value.strip() or "UNKNOWN"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading cleaned August dataset...")

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)

# --------------------------------------------------
# STANDARDISE ADMINISTRATIVE NAMES
# --------------------------------------------------

print("Normalising administrative names...")

location_columns = [
    "state",
    "district",
    "block",
    "village"
]

for col in location_columns:
    df[col] = df[col].apply(normalize_name)

# --------------------------------------------------
# STANDARDISE COORDINATES
# --------------------------------------------------

df["latitude"] = pd.to_numeric(
    df["latitude"], errors="coerce"
).round(6)

df["longitude"] = pd.to_numeric(
    df["longitude"], errors="coerce"
).round(6)

# --------------------------------------------------
# GENERATE PROVISIONAL LOCATION KEY
# --------------------------------------------------

print("Generating provisional location keys...")

df["location_key"] = (
    df["state"].astype(str) + "_"
    + df["district"].astype(str) + "_"
    + df["block"].astype(str) + "_"
    + df["village"].astype(str) + "_"
    + df["latitude"].astype(str) + "_"
    + df["longitude"].astype(str)
)

# --------------------------------------------------
# IDENTIFY UNIQUE LOCATIONS
# --------------------------------------------------

mapping_columns = [
    "location_key",
    "state",
    "district",
    "block",
    "village",
    "latitude",
    "longitude"
]

location_mapping = (
    df[mapping_columns]
    .drop_duplicates()
    .sort_values(["state", "district", "block", "village"])
    .reset_index(drop=True)
)

# --------------------------------------------------
# IDENTIFY COORDINATE CONFLICTS
# --------------------------------------------------

print("Checking administrative label conflicts...")

coordinate_groups = (
    df.groupby(["latitude", "longitude"])
    .agg(
        state_count=("state", "nunique"),
        district_count=("district", "nunique"),
        block_count=("block", "nunique"),
        village_count=("village", "nunique"),
        record_count=("location_key", "size")
    )
    .reset_index()
)

conflicts = coordinate_groups[
    (coordinate_groups["state_count"] > 1)
    | (coordinate_groups["district_count"] > 1)
    | (coordinate_groups["block_count"] > 1)
    | (coordinate_groups["village_count"] > 1)
].copy()

# --------------------------------------------------
# LOCATION STATISTICS
# --------------------------------------------------

unique_locations = df["location_key"].nunique()

unique_coordinates = df[
    ["latitude", "longitude"]
].drop_duplicates().shape[0]

unique_states = df["state"].nunique()
unique_districts = df["district"].nunique()
unique_blocks = df["block"].nunique()
unique_villages = df["village"].nunique()

# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------

print("Saving standardised dataset...")

df.to_csv(OUTPUT_FILE, index=False)

location_mapping.to_csv(
    MAPPING_FILE,
    index=False
)

conflicts.to_csv(
    CONFLICT_FILE,
    index=False
)

# --------------------------------------------------
# GENERATE REPORT
# --------------------------------------------------

summary = f"""
AUGUST LOCATION STANDARDISATION REPORT
==========================================

Input records: {original_rows}

Records in standardised dataset: {len(df)}

------------------------------------------
LOCATION STATISTICS
------------------------------------------

Unique states/UTs: {unique_states}
Unique districts: {unique_districts}
Unique blocks: {unique_blocks}
Unique villages: {unique_villages}

Unique coordinate pairs: {unique_coordinates}

Provisional location keys: {unique_locations}

------------------------------------------
COORDINATE CONFLICTS
------------------------------------------

Coordinates with multiple administrative labels:
{len(conflicts)}

------------------------------------------
OUTPUT FILES
------------------------------------------

Standardised dataset:
{OUTPUT_FILE}

Location mapping:
{MAPPING_FILE}

Coordinate conflicts:
{CONFLICT_FILE}
"""

SUMMARY_FILE.write_text(summary, encoding="utf-8")

print(summary)

print("\nAugust location standardisation completed!")