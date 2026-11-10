
from pathlib import Path
import pandas as pd

DATA = Path(
    "data/processed/team_dataset/team_quality_flagged.csv"
)

ELIGIBLE = Path(
    "outputs/team_temporal_coverage/eligible_locations.csv"
)

OUTPUT = Path("data/processed/team_dataset/multiseason")
OUTPUT.mkdir(parents=True, exist_ok=True)

SEASONS = [1, 5, 8, 11]
MONTH_NAMES = {
    1: "jan",
    5: "may",
    8: "aug",
    11: "nov",
}

print("Loading datasets...")

df = pd.read_csv(DATA, low_memory=False)
eligible = pd.read_csv(ELIGIBLE)

# Select locations with at least 10 complete four-season years.
selected = eligible.loc[
    eligible["years_with_all_four_seasons"] >= 10,
    ["LOCATION_KEY"]
].copy()

selected_keys = set(selected["LOCATION_KEY"])

# Reconstruct the same provisional location key.
location_columns = [
    "STATE_UT_NORMALIZED",
    "DISTRICT_NORMALIZED",
    "BLOCK_NORMALIZED",
    "VILLAGE_NORMALIZED",
]

df["LOCATION_KEY"] = (
    df[location_columns]
    .fillna("")
    .astype(str)
    .agg("|".join, axis=1)
    + "|"
    + pd.to_numeric(
        df["LATITUDE_NUMERIC"], errors="coerce"
    ).round(5).astype(str)
    + "|"
    + pd.to_numeric(
        df["LONGITUDE_NUMERIC"], errors="coerce"
    ).round(5).astype(str)
)

df["Date_PARSED"] = pd.to_datetime(
    df["Date_PARSED"], errors="coerce"
)

df["WL_NUMERIC"] = pd.to_numeric(
    df["WL_NUMERIC"], errors="coerce"
)

df["YEAR"] = df["Date_PARSED"].dt.year
df["MONTH"] = df["Date_PARSED"].dt.month

# Filter to selected locations and four major seasons.
df = df[
    df["LOCATION_KEY"].isin(selected_keys)
    & df["MONTH"].isin(SEASONS)
].copy()

# Keep only usable measurements.
df = df.dropna(
    subset=["WL_NUMERIC", "YEAR"]
).copy()

# Convert quality flags safely.
conflict = (
    df["conflicting_observation"]
    .fillna(False)
    .astype(bool)
)

df["CONFLICT_FLAG"] = conflict.astype(int)

print(f"Selected locations: {len(selected_keys):,}")
print(f"Usable seasonal records: {len(df):,}")

# Aggregate each location-year-month.
# Median is used to reduce sensitivity to extreme readings.
grouped = (
    df.groupby(
        ["LOCATION_KEY", "YEAR", "MONTH"],
        as_index=False
    )
    .agg(
        groundwater_level=("WL_NUMERIC", "median"),
        observation_count=("WL_NUMERIC", "size"),
        distinct_readings=("WL_NUMERIC", "nunique"),
        conflict_flag=("CONFLICT_FLAG", "max"),
    )
)

grouped["conflict_flag"] = (
    grouped["conflict_flag"]
    | (grouped["distinct_readings"] > 1).astype(int)
)

# Convert the seasonal measurements into columns.
level_wide = grouped.pivot(
    index=["LOCATION_KEY", "YEAR"],
    columns="MONTH",
    values="groundwater_level",
)

count_wide = grouped.pivot(
    index=["LOCATION_KEY", "YEAR"],
    columns="MONTH",
    values="observation_count",
)

conflict_wide = grouped.pivot(
    index=["LOCATION_KEY", "YEAR"],
    columns="MONTH",
    values="conflict_flag",
)

for month in SEASONS:
    if month not in level_wide.columns:
        level_wide[month] = float("nan")
    if month not in count_wide.columns:
        count_wide[month] = 0
    if month not in conflict_wide.columns:
        conflict_wide[month] = 0

seasonal = pd.DataFrame(index=level_wide.index)

for month in SEASONS:
    name = MONTH_NAMES[month]

    seasonal[f"{name}_level"] = level_wide[month]
    seasonal[f"{name}_count"] = count_wide[month]
    seasonal[f"{name}_conflict"] = conflict_wide[month]

seasonal = seasonal.reset_index()

# Retain years with all four seasonal measurements.
level_columns = [
    f"{MONTH_NAMES[m]}_level" for m in SEASONS
]

seasonal_complete = seasonal.dropna(
    subset=level_columns
).copy()

seasonal_complete["YEAR"] = (
    seasonal_complete["YEAR"].astype(int)
)

seasonal_complete = seasonal_complete.sort_values(
    ["LOCATION_KEY", "YEAR"]
)

# Save the location-year seasonal dataset.
seasonal_complete.to_csv(
    OUTPUT / "location_year_seasonal.csv",
    index=False
)

# Create next-year forecasting samples.
feature_columns = level_columns + [
    f"{MONTH_NAMES[m]}_count" for m in SEASONS
] + [
    f"{MONTH_NAMES[m]}_conflict" for m in SEASONS
]

target_columns = [
    f"target_{MONTH_NAMES[m]}_level" for m in SEASONS
]

current = seasonal_complete[
    ["LOCATION_KEY", "YEAR"] + feature_columns
].copy()

future = seasonal_complete[
    ["LOCATION_KEY", "YEAR"] + level_columns
].copy()

future["YEAR"] = future["YEAR"] - 1
future = future.rename(
    columns={
        col: f"target_{MONTH_NAMES[m]}_level"
        for m, col in zip(SEASONS, level_columns)
    }
)

samples = current.merge(
    future,
    on=["LOCATION_KEY", "YEAR"],
    how="inner",
    validate="one_to_one",
)

# The merge only matches consecutive calendar years.
samples["TARGET_YEAR"] = samples["YEAR"] + 1

# Chronological splits based on target year.
samples["SPLIT"] = "unused"

samples.loc[
    samples["TARGET_YEAR"] <= 2019,
    "SPLIT"
] = "train"

samples.loc[
    samples["TARGET_YEAR"].between(2020, 2022),
    "SPLIT"
] = "validation"

samples.loc[
    samples["TARGET_YEAR"].between(2023, 2024),
    "SPLIT"
] = "test"

samples = samples[
    samples["SPLIT"] != "unused"
].copy()

samples.to_csv(
    OUTPUT / "multiseason_prediction_dataset.csv",
    index=False
)

# Save a compact report.
print("\nLOCATION-YEAR DATASET")
print(f"Rows: {len(seasonal_complete):,}")
print(
    "Unique locations:",
    seasonal_complete["LOCATION_KEY"].nunique()
)
print(
    "Year range:",
    seasonal_complete["YEAR"].min(),
    "to",
    seasonal_complete["YEAR"].max(),
)

print("\nPREDICTION DATASET")
print(f"Samples: {len(samples):,}")
print(
    "Unique locations:",
    samples["LOCATION_KEY"].nunique()
)

print("\nSPLIT DISTRIBUTION")
print(
    samples.groupby(
        ["SPLIT", "TARGET_YEAR"]
    ).size().to_string()
)

print("\nQUALITY INDICATORS")
for col in [
    "jan_conflict",
    "may_conflict",
    "aug_conflict",
    "nov_conflict",
]:
    print(
        f"{col}:",
        int(samples[col].sum())
    )

print("\nOutput directory:", OUTPUT.resolve())
print("Dataset construction complete.")
