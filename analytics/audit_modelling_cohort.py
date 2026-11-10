
from pathlib import Path
import pandas as pd

DATA = Path(
    "data/processed/team_dataset/team_quality_flagged.csv"
)

ELIGIBLE = Path(
    "outputs/team_temporal_coverage/eligible_locations.csv"
)

OUTPUT = Path("outputs/team_temporal_coverage")
OUTPUT.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

df = pd.read_csv(DATA, low_memory=False)
eligible = pd.read_csv(ELIGIBLE)

# Select locations with at least 10 complete four-season years.
selected = eligible.loc[
    eligible["years_with_all_four_seasons"] >= 10,
    ["LOCATION_KEY"]
].copy()

selected_keys = set(selected["LOCATION_KEY"])

df["Date_PARSED"] = pd.to_datetime(
    df["Date_PARSED"], errors="coerce"
)

df["WL_NUMERIC"] = pd.to_numeric(
    df["WL_NUMERIC"], errors="coerce"
)

# Recreate the location key using the same logic
# as analyze_location_season_coverage.py.

location_columns = [
    "STATE_UT_NORMALIZED",
    "DISTRICT_NORMALIZED",
    "BLOCK_NORMALIZED",
    "VILLAGE_NORMALIZED",
]

for col in location_columns:
    if col not in df.columns:
        raise ValueError(f"Missing expected column: {col}")

df["LOCATION_KEY"] = (
    df[location_columns]
    .fillna("")
    .astype(str)
    .agg("|".join, axis=1)
    + "|"
    + df["LATITUDE_NUMERIC"].round(5).astype(str)
    + "|"
    + df["LONGITUDE_NUMERIC"].round(5).astype(str)
)

df["YEAR"] = df["Date_PARSED"].dt.year
df["MONTH"] = df["Date_PARSED"].dt.month

df = df[df["LOCATION_KEY"].isin(selected_keys)].copy()
df = df[df["MONTH"].isin([1, 5, 8, 11])].copy()

print("\nCOHORT SUMMARY")
print(f"Selected locations: {len(selected_keys):,}")
print(f"Raw seasonal records: {len(df):,}")

print("\nRECORD QUALITY")

for column in [
    "conflicting_observation",
    "negative_groundwater_level",
    "zero_groundwater_level",
    "extreme_groundwater_level",
]:
    if column in df.columns:
        count = df[column].fillna(False).astype(bool).sum()
        print(f"{column}: {count:,}")

print("\nSEASONAL RECORD COUNTS")

print(
    df.groupby("MONTH")
    .agg(
        records=("WL_NUMERIC", "count"),
        locations=("LOCATION_KEY", "nunique"),
        years=("YEAR", "nunique"),
    )
    .to_string()
)

# Check repeated observations within the same location/year/month.
groups = (
    df.groupby(["LOCATION_KEY", "YEAR", "MONTH"])
    .agg(
        observation_count=("WL_NUMERIC", "count"),
        distinct_readings=("WL_NUMERIC", "nunique"),
    )
    .reset_index()
)

print("\nLOCATION-YEAR-MONTH GROUPS")
print(f"Total groups: {len(groups):,}")
print(
    "Groups with multiple observations:",
    f"{(groups['observation_count'] > 1).sum():,}"
)
print(
    "Groups with conflicting readings:",
    f"{(groups['distinct_readings'] > 1).sum():,}"
)

groups.to_csv(
    OUTPUT / "cohort_location_month_groups.csv",
    index=False,
)

print("\nAudit complete.")
