
from pathlib import Path

import pandas as pd

INPUT = Path(
    "data/processed/team_dataset/team_quality_flagged.csv"
)

OUTPUT_DIR = Path("outputs/team_temporal_coverage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [1, 5, 8, 11]

print("Loading dataset...")

df = pd.read_csv(INPUT, low_memory=False)

df["Date_PARSED"] = pd.to_datetime(
    df["Date_PARSED"], errors="coerce"
)
df["WL_NUMERIC"] = pd.to_numeric(
    df["WL_NUMERIC"], errors="coerce"
)

# Retain records with usable dates, readings and coordinates.
df = df.dropna(
    subset=[
        "Date_PARSED",
        "WL_NUMERIC",
        "LATITUDE_NUMERIC",
        "LONGITUDE_NUMERIC",
    ]
).copy()

# Build a provisional location key.
# Administrative labels help distinguish locations
# that happen to share coordinates.
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

df = df[df["MONTH"].isin(SEASONS)].copy()

# Count observations for each location, year and season.
coverage = (
    df.groupby(["LOCATION_KEY", "YEAR", "MONTH"])
    .size()
    .reset_index(name="OBSERVATIONS")
)

coverage.to_csv(
    OUTPUT_DIR / "location_season_coverage.csv",
    index=False
)

# Count years represented in each season per location.
season_summary = (
    coverage.groupby(["LOCATION_KEY", "MONTH"])
    .agg(
        years_with_data=("YEAR", "nunique"),
        total_observations=("OBSERVATIONS", "sum"),
    )
    .reset_index()
)

season_wide = season_summary.pivot(
    index="LOCATION_KEY",
    columns="MONTH",
    values="years_with_data"
).fillna(0)

season_wide.columns = [
    f"years_month_{int(month)}"
    for month in season_wide.columns
]

observation_wide = season_summary.pivot(
    index="LOCATION_KEY",
    columns="MONTH",
    values="total_observations"
).fillna(0)

observation_wide.columns = [
    f"observations_month_{int(month)}"
    for month in observation_wide.columns
]

summary = season_wide.join(observation_wide).fillna(0)

# Count years where a location has all four seasons.
complete_years = (
    coverage.pivot_table(
        index=["LOCATION_KEY", "YEAR"],
        columns="MONTH",
        values="OBSERVATIONS",
        aggfunc="sum",
        fill_value=0,
    )
)

for month in SEASONS:
    if month not in complete_years.columns:
        complete_years[month] = 0

complete_years["ALL_FOUR_SEASONS"] = (
    complete_years[SEASONS] > 0
).all(axis=1)

complete_year_counts = (
    complete_years.groupby(level="LOCATION_KEY")[
        "ALL_FOUR_SEASONS"
    ].sum()
)

summary["years_with_all_four_seasons"] = (
    complete_year_counts
)

summary = summary.fillna(0)

# Initial eligibility criteria.
summary["eligible"] = (
    (summary["years_month_1"] >= 10)
    & (summary["years_month_5"] >= 8)
    & (summary["years_month_8"] >= 8)
    & (summary["years_month_11"] >= 8)
    & (summary["years_with_all_four_seasons"] >= 8)
)

summary.to_csv(
    OUTPUT_DIR / "location_coverage_summary.csv"
)

eligible = summary[summary["eligible"]].copy()

eligible.to_csv(
    OUTPUT_DIR / "eligible_locations.csv"
)

print("\nLOCATION COVERAGE SUMMARY")
print(f"Total locations: {len(summary)}")
print(f"Eligible locations: {len(eligible)}")
print(
    f"Eligibility rate: "
    f"{len(eligible) / max(len(summary), 1) * 100:.2f}%"
)

print("\nSEASONAL COVERAGE")
print(
    summary[
        [
            "years_month_1",
            "years_month_5",
            "years_month_8",
            "years_month_11",
            "years_with_all_four_seasons",
        ]
    ].describe().round(2)
)

print("\nELIGIBLE LOCATION COUNTS")
print(
    summary["eligible"].value_counts().rename(
        {True: "Eligible", False: "Not eligible"}
    )
)

print("\nAnalysis complete.")
print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
