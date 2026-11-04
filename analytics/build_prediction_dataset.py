from pathlib import Path
import pandas as pd
import numpy as np
import re
import unicodedata

# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

JAN_FILE = ROOT / "data/raw/January_WL_1994-2025.csv"

AUG_FILE = (
    ROOT / "data/processed/august/"
    "August_WL_1994-2025_clean.csv"
)

OUTPUT_DIR = ROOT / "data/processed/prediction"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "groundwater_prediction_dataset.csv"

REPORT_FILE = OUTPUT_DIR / "prediction_dataset_report.txt"

# ==================================================
# LOCATION NORMALISATION
# ==================================================

def normalize_name(value):

    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip().upper()

    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    value = re.sub(r"[^\w\s]", "", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip() or "UNKNOWN"


def create_location_key(df):

    return (
        df["state"] + "_"
        + df["district"] + "_"
        + df["block"] + "_"
        + df["village"] + "_"
        + df["latitude"].astype(str) + "_"
        + df["longitude"].astype(str)
    )


# ==================================================
# LOAD JANUARY DATA
# ==================================================

print("\nLoading January historical dataset...")

jan = pd.read_csv(JAN_FILE)

jan = jan.rename(columns={
    "STATE_UT": "state",
    "DISTRICT": "district",
    "BLOCK": "block",
    "VILLAGE": "village",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    "Date": "date",
    "WL(mbgl)": "water_level_mbgl"
})

jan["date"] = pd.to_datetime(
    jan["date"],
    format="%d-%m-%y",
    errors="coerce"
)

jan["season"] = "January"

# ==================================================
# LOAD AUGUST DATA
# ==================================================

print("Loading August historical dataset...")

aug = pd.read_csv(AUG_FILE)

aug["date"] = pd.to_datetime(
    aug["date"],
    format="%Y-%m-%d",
    errors="coerce"
)

aug["season"] = "August"

# ==================================================
# STANDARDISE BOTH DATASETS
# ==================================================

print("Standardising location information...")

required = [
    "state",
    "district",
    "block",
    "village",
    "latitude",
    "longitude",
    "date",
    "water_level_mbgl",
    "season"
]

jan = jan[required].copy()
aug = aug[required].copy()

df = pd.concat(
    [jan, aug],
    ignore_index=True
)

for col in ["state", "district", "block", "village"]:

    df[col] = df[col].apply(normalize_name)

for col in ["latitude", "longitude", "water_level_mbgl"]:

    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

df["latitude"] = df["latitude"].round(6)
df["longitude"] = df["longitude"].round(6)

df["year"] = df["date"].dt.year

# ==================================================
# REMOVE INVALID RECORDS
# ==================================================

df = df.dropna(
    subset=[
        "date",
        "latitude",
        "longitude",
        "water_level_mbgl",
        "year"
    ]
).copy()

df = df[
    df["water_level_mbgl"] >= 0
].copy()

# ==================================================
# LOCATION IDENTIFIERS
# ==================================================

df["location_key"] = create_location_key(df)

# ==================================================
# AGGREGATE LOCATION-YEAR-SEASON
# ==================================================

print("Aggregating seasonal observations...")

seasonal = (
    df.groupby(
        [
            "location_key",
            "state",
            "district",
            "block",
            "village",
            "latitude",
            "longitude",
            "year",
            "season"
        ],
        dropna=False
    )
    .agg(
        water_level_mbgl=("water_level_mbgl", "median"),
        observation_count=("water_level_mbgl", "count")
    )
    .reset_index()
)

# ==================================================
# PIVOT JANUARY AND AUGUST
# ==================================================

print("Combining seasonal observations...")

levels = seasonal.pivot(
    index=[
        "location_key",
        "state",
        "district",
        "block",
        "village",
        "latitude",
        "longitude",
        "year"
    ],
    columns="season",
    values="water_level_mbgl"
).reset_index()

levels.columns.name = None

levels = levels.rename(columns={
    "January": "january_level_mbgl",
    "August": "august_level_mbgl"
})

# ==================================================
# ADD OBSERVATION FLAGS
# ==================================================

counts = seasonal.pivot(
    index=[
        "location_key",
        "year"
    ],
    columns="season",
    values="observation_count"
).reset_index()

counts.columns.name = None

counts = counts.rename(columns={
    "January": "january_observation_count",
    "August": "august_observation_count"
})

levels = levels.merge(
    counts,
    on=["location_key", "year"],
    how="left"
)

# ==================================================
# YEAR-OVER-YEAR FEATURES
# ==================================================

print("Engineering historical features...")

levels = levels.sort_values(
    ["location_key", "year"]
).reset_index(drop=True)

for col in [
    "january_level_mbgl",
    "august_level_mbgl"
]:

    levels[f"previous_{col}"] = (
        levels.groupby("location_key")[col]
        .shift(1)
    )

    levels[f"{col}_change"] = (
        levels.groupby("location_key")[col]
        .diff()
    )

# Previous year's available seasonal level
levels["previous_january_level_mbgl"] = (
    levels.groupby("location_key")["january_level_mbgl"]
    .shift(1)
)

levels["previous_august_level_mbgl"] = (
    levels.groupby("location_key")["august_level_mbgl"]
    .shift(1)
)

# ==================================================
# SEASONAL DIFFERENCE
# ==================================================

levels["august_january_difference_mbgl"] = (
    levels["august_level_mbgl"]
    - levels["january_level_mbgl"]
)

# ==================================================
# SAVE DATASET
# ==================================================

levels.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==================================================
# GENERATE REPORT
# ==================================================

report = f"""
GROUNDWATER PREDICTION DATASET REPORT
==========================================

Original January records: {len(jan)}

Original August records: {len(aug)}

Combined records before aggregation: {len(df)}

Seasonal location-year observations: {len(seasonal)}

Final location-year records: {len(levels)}

Unique locations: {levels['location_key'].nunique()}

Years covered:
{levels['year'].min()} - {levels['year'].max()}

------------------------------------------
SEASONAL COVERAGE
------------------------------------------

January observations:
{levels['january_level_mbgl'].notna().sum()}

August observations:
{levels['august_level_mbgl'].notna().sum()}

Both seasons available:
{levels[['january_level_mbgl', 'august_level_mbgl']].notna().all(axis=1).sum()}

------------------------------------------
MISSING VALUES
------------------------------------------

{levels.isnull().sum().to_string()}

------------------------------------------
OUTPUT
------------------------------------------

{OUTPUT_FILE}

IMPORTANT:
Seasonal observations were aggregated using
the median for each location-year-season.

Missing seasonal observations were retained
as missing values.

No groundwater readings were interpolated.
"""

REPORT_FILE.write_text(
    report,
    encoding="utf-8"
)

print(report)

print("\nPrediction dataset construction completed!")