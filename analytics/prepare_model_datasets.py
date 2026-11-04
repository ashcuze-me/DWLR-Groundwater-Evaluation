from pathlib import Path
import pandas as pd
import numpy as np
import re
import unicodedata

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

JAN_FILE = ROOT / "data/raw/January_WL_1994-2025.csv"

AUG_FILE = (
    ROOT / "data/processed/august/"
    "August_WL_1994-2025_clean.csv"
)

OUTPUT_DIR = ROOT / "data/processed/prediction"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JAN_OUTPUT = OUTPUT_DIR / "january_model_dataset.csv"
AUG_OUTPUT = OUTPUT_DIR / "august_model_dataset.csv"
COMBINED_OUTPUT = OUTPUT_DIR / "groundwater_prediction_dataset.csv"

REPORT_FILE = OUTPUT_DIR / "model_dataset_preparation_report.txt"


# ==========================================================
# CONFIGURATION
# ==========================================================

LOCATION_COLUMNS = [
    "state",
    "district",
    "block",
    "village",
    "latitude",
    "longitude"
]

SEASONS = ["January", "August"]


# ==========================================================
# NORMALISE LOCATION NAMES
# ==========================================================

def normalize_name(value):

    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip().upper()

    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
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


# ==========================================================
# LOAD JANUARY DATA
# ==========================================================

print("\nLoading January historical data...")

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


# ==========================================================
# LOAD AUGUST DATA
# ==========================================================

print("Loading August historical data...")

aug = pd.read_csv(AUG_FILE)

aug["date"] = pd.to_datetime(
    aug["date"],
    format="%Y-%m-%d",
    errors="coerce"
)

aug["season"] = "August"


# ==========================================================
# STANDARDISE DATA
# ==========================================================

print("Standardising both datasets...")

required_columns = LOCATION_COLUMNS + [
    "date",
    "water_level_mbgl",
    "season"
]

jan = jan[required_columns].copy()
aug = aug[required_columns].copy()

df = pd.concat(
    [jan, aug],
    ignore_index=True
)

for col in ["state", "district", "block", "village"]:

    df[col] = df[col].apply(normalize_name)

for col in ["latitude", "longitude", "water_level_mbgl"]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df["latitude"] = df["latitude"].round(6)
df["longitude"] = df["longitude"].round(6)

df["year"] = df["date"].dt.year

df = df.dropna(
    subset=[
        "date",
        "latitude",
        "longitude",
        "water_level_mbgl",
        "year"
    ]
).copy()

df = df[df["water_level_mbgl"] >= 0].copy()

# ==========================================================
# LOCATION IDENTIFICATION
# ==========================================================

print("Generating provisional location identifiers...")

df["location_key"] = create_location_key(df)

# ==========================================================
# AGGREGATE LOCATION-YEAR-SEASON OBSERVATIONS
# ==========================================================

print("Aggregating observations...")

seasonal = (
    df.groupby(
        [
            "location_key",
            *LOCATION_COLUMNS,
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

# ==========================================================
# CREATE UNIFIED SEASONAL DATASET
# ==========================================================

print("Combining January and August observations...")

index_columns = [
    "location_key",
    *LOCATION_COLUMNS,
    "year"
]

levels = seasonal.pivot(
    index=index_columns,
    columns="season",
    values="water_level_mbgl"
).reset_index()

levels.columns.name = None

levels = levels.rename(columns={
    "January": "january_level_mbgl",
    "August": "august_level_mbgl"
})

counts = seasonal.pivot(
    index=["location_key", "year"],
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

levels = levels.sort_values(
    ["location_key", "year"]
).reset_index(drop=True)

# ==========================================================
# EXACT PREVIOUS CALENDAR YEAR FEATURES
# ==========================================================

print("Generating previous-calendar-year features...")

previous_year = levels[
    [
        "location_key",
        "year",
        "january_level_mbgl",
        "august_level_mbgl"
    ]
].copy()

previous_year["year"] = previous_year["year"] + 1

previous_year = previous_year.rename(columns={
    "january_level_mbgl": "previous_year_january_mbgl",
    "august_level_mbgl": "previous_year_august_mbgl"
})

levels = levels.merge(
    previous_year,
    on=["location_key", "year"],
    how="left"
)

# ==========================================================
# HISTORICAL OBSERVATION GAP FEATURES
# ==========================================================

print("Calculating historical observation gaps...")

for season, column in [
    ("january", "january_level_mbgl"),
    ("august", "august_level_mbgl")
]:

    observed = levels[
        ["location_key", "year", column]
    ].dropna(subset=[column]).copy()

    observed = observed.sort_values(
        ["location_key", "year"]
    )

    observed["previous_observation_year"] = (
        observed.groupby("location_key")["year"].shift(1)
    )

    observed["observation_gap_years"] = (
        observed["year"]
        - observed["previous_observation_year"]
    )

    gap_mapping = observed[
        [
            "location_key",
            "year",
            "observation_gap_years"
        ]
    ]

    levels = levels.merge(
        gap_mapping.rename(columns={
            "observation_gap_years":
                f"{season}_observation_gap_years"
        }),
        on=["location_key", "year"],
        how="left"
    )

# ==========================================================
# YEAR-OVER-YEAR CHANGES
# ==========================================================

levels["january_change_from_previous_year"] = (
    levels["january_level_mbgl"]
    - levels["previous_year_january_mbgl"]
)

levels["august_change_from_previous_year"] = (
    levels["august_level_mbgl"]
    - levels["previous_year_august_mbgl"]
)

# ==========================================================
# SEASONAL DIFFERENCE
# ==========================================================

levels["august_minus_january_mbgl"] = (
    levels["august_level_mbgl"]
    - levels["january_level_mbgl"]
)

# ==========================================================
# CREATE NEXT-YEAR TARGETS
# ==========================================================

print("Creating next-calendar-year prediction targets...")

targets = levels[
    [
        "location_key",
        "year",
        "january_level_mbgl",
        "august_level_mbgl"
    ]
].copy()

targets["year"] = targets["year"] - 1

targets = targets.rename(columns={
    "year": "input_year",
    "january_level_mbgl": "target_january_mbgl",
    "august_level_mbgl": "target_august_mbgl"
})

levels = levels.merge(
    targets,
    left_on=["location_key", "year"],
    right_on=["location_key", "input_year"],
    how="left"
)

levels["target_year"] = levels["year"] + 1

# ==========================================================
# MODEL FEATURE DEFINITIONS
# ==========================================================

COMMON_FEATURES = [
    "year",
    "latitude",
    "longitude",
    "state",
    "district",
    "january_observation_count",
    "august_observation_count",
    "january_observation_gap_years",
    "august_observation_gap_years"
]

JANUARY_FEATURES = COMMON_FEATURES + [
    "january_level_mbgl",
    "august_level_mbgl",
    "previous_year_january_mbgl",
    "previous_year_august_mbgl",
    "january_change_from_previous_year",
    "august_change_from_previous_year",
    "august_minus_january_mbgl"
]

AUGUST_FEATURES = COMMON_FEATURES + [
    "january_level_mbgl",
    "august_level_mbgl",
    "previous_year_january_mbgl",
    "previous_year_august_mbgl",
    "january_change_from_previous_year",
    "august_change_from_previous_year",
    "august_minus_january_mbgl"
]

# ==========================================================
# PREPARE MODEL DATASETS
# ==========================================================

print("Preparing separate seasonal model datasets...")

january_model = levels[
    JANUARY_FEATURES
    + [
        "location_key",
        "target_year",
        "target_january_mbgl"
    ]
].copy()

august_model = levels[
    AUGUST_FEATURES
    + [
        "location_key",
        "target_year",
        "target_august_mbgl"
    ]
].copy()

# Retain only rows where the corresponding future target exists.
january_model = january_model.dropna(
    subset=["target_january_mbgl"]
)

august_model = august_model.dropna(
    subset=["target_august_mbgl"]
)

# ==========================================================
# SAVE DATASETS
# ==========================================================

levels.to_csv(COMBINED_OUTPUT, index=False)

january_model.to_csv(
    JAN_OUTPUT,
    index=False
)

august_model.to_csv(
    AUG_OUTPUT,
    index=False
)

# ==========================================================
# GENERATE REPORT
# ==========================================================

report = f"""
GROUNDWATER MODEL DATASET PREPARATION REPORT
================================================

COMBINED DATA
------------------------------------------------

Combined raw records: {len(df)}

Aggregated seasonal records: {len(seasonal)}

Location-year records: {len(levels)}

Unique provisional locations:
{levels['location_key'].nunique()}

Year range:
{levels['year'].min()} - {levels['year'].max()}

------------------------------------------------
JANUARY MODEL
------------------------------------------------

Training samples: {len(january_model)}

Unique locations:
{january_model['location_key'].nunique()}

Input years:
{january_model['year'].min()} - {january_model['year'].max()}

Target years:
{january_model['target_year'].min()} - {january_model['target_year'].max()}

Missing target values:
{january_model['target_january_mbgl'].isna().sum()}

------------------------------------------------
AUGUST MODEL
------------------------------------------------

Training samples: {len(august_model)}

Unique locations:
{august_model['location_key'].nunique()}

Input years:
{august_model['year'].min()} - {august_model['year'].max()}

Target years:
{august_model['target_year'].min()} - {august_model['target_year'].max()}

Missing target values:
{august_model['target_august_mbgl'].isna().sum()}

------------------------------------------------
FEATURE ENGINEERING
------------------------------------------------

Previous calendar year values are obtained
using exact year matching.

Missing previous-year observations remain missing.

Observation gaps are retained as features.

No future observations are used as predictors.

Multiple readings are aggregated using medians.

------------------------------------------------
OUTPUT FILES
------------------------------------------------

{COMBINED_OUTPUT}

{JAN_OUTPUT}

{AUG_OUTPUT}
"""

REPORT_FILE.write_text(
    report,
    encoding="utf-8"
)

print(report)

print("\nModel dataset preparation completed successfully!")