
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "team_dataset"
    / "team_quality_flagged.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "team_eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading quality-flagged dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["Date_PARSED"] = pd.to_datetime(
    df["Date_PARSED"], errors="coerce"
)

df["WL_NUMERIC"] = pd.to_numeric(
    df["WL_NUMERIC"], errors="coerce"
)

# Use records with valid dates and groundwater readings.
# Negative and extreme readings are retained and flagged.

valid = df.dropna(
    subset=["Date_PARSED", "WL_NUMERIC"]
).copy()

valid["year"] = valid["Date_PARSED"].dt.year
valid["month"] = valid["Date_PARSED"].dt.month

# --------------------------------------------------
# 1. OVERALL DATASET SUMMARY
# --------------------------------------------------

summary = {
    "total_records": len(df),
    "valid_date_and_level_records": len(valid),
    "unique_states": df["STATE_UT"].nunique(),
    "unique_districts": df["DISTRICT"].nunique(),
    "unique_blocks": df["BLOCK"].nunique(),
    "unique_villages": df["VILLAGE"].nunique(),
    "unique_coordinate_pairs": df[
        ["LATITUDE_NUMERIC", "LONGITUDE_NUMERIC"]
    ].dropna().drop_duplicates().shape[0],
    "earliest_date": valid["Date_PARSED"].min(),
    "latest_date": valid["Date_PARSED"].max()
}

print("\nOVERALL DATASET SUMMARY")
for key, value in summary.items():
    print(f"{key}: {value}")

pd.DataFrame(
    list(summary.items()),
    columns=["metric", "value"]
).to_csv(
    OUTPUT_DIR / "overall_dataset_summary.csv",
    index=False
)

# --------------------------------------------------
# 2. GROUNDWATER STATISTICS
# --------------------------------------------------

statistics = valid["WL_NUMERIC"].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
)

statistics.to_csv(
    OUTPUT_DIR / "groundwater_statistics.csv",
    header=["value"]
)

print("\nGROUNDWATER STATISTICS")
print(statistics)

# --------------------------------------------------
# 3. YEARLY COVERAGE
# --------------------------------------------------

yearly = (
    valid.groupby("year")
    .agg(
        observations=("WL_NUMERIC", "size"),
        mean_level=("WL_NUMERIC", "mean"),
        median_level=("WL_NUMERIC", "median"),
        standard_deviation=("WL_NUMERIC", "std")
    )
    .reset_index()
)

yearly.to_csv(
    OUTPUT_DIR / "yearly_groundwater_summary.csv",
    index=False
)

print("\nYEARLY COVERAGE")
print(yearly.to_string(index=False))

# --------------------------------------------------
# 4. MONTHLY COVERAGE
# --------------------------------------------------

monthly = (
    valid.groupby("month")
    .agg(
        observations=("WL_NUMERIC", "size"),
        mean_level=("WL_NUMERIC", "mean"),
        median_level=("WL_NUMERIC", "median"),
        standard_deviation=("WL_NUMERIC", "std")
    )
    .reset_index()
)

monthly.to_csv(
    OUTPUT_DIR / "monthly_groundwater_summary.csv",
    index=False
)

print("\nMONTHLY COVERAGE")
print(monthly.to_string(index=False))

# --------------------------------------------------
# 5. STATE-WISE ANALYSIS
# --------------------------------------------------

state_summary = (
    valid.groupby("STATE_UT")
    .agg(
        observations=("WL_NUMERIC", "size"),
        mean_level=("WL_NUMERIC", "mean"),
        median_level=("WL_NUMERIC", "median"),
        standard_deviation=("WL_NUMERIC", "std"),
        districts=("DISTRICT", "nunique")
    )
    .reset_index()
    .sort_values("observations", ascending=False)
)

state_summary.to_csv(
    OUTPUT_DIR / "state_groundwater_summary.csv",
    index=False
)

print("\nTOP 15 STATES BY OBSERVATION COUNT")
print(state_summary.head(15).to_string(index=False))

# --------------------------------------------------
# 6. DISTRICT-WISE ANALYSIS
# --------------------------------------------------

district_summary = (
    valid.groupby(["STATE_UT", "DISTRICT"])
    .agg(
        observations=("WL_NUMERIC", "size"),
        mean_level=("WL_NUMERIC", "mean"),
        median_level=("WL_NUMERIC", "median")
    )
    .reset_index()
    .sort_values("observations", ascending=False)
)

district_summary.to_csv(
    OUTPUT_DIR / "district_groundwater_summary.csv",
    index=False
)

print("\nTOP 20 DISTRICTS BY OBSERVATION COUNT")
print(district_summary.head(20).to_string(index=False))

# --------------------------------------------------
# 7. OBSERVATION COVERAGE BY YEAR AND MONTH
# --------------------------------------------------

year_month = (
    valid.groupby(["year", "month"])
    .size()
    .reset_index(name="observations")
)

year_month.to_csv(
    OUTPUT_DIR / "year_month_observation_coverage.csv",
    index=False
)

# --------------------------------------------------
# 8. QUALITY FLAG DISTRIBUTION
# --------------------------------------------------

quality_summary = pd.DataFrame({
    "flag": [
        "negative_groundwater_level",
        "zero_groundwater_level",
        "extreme_groundwater_level",
        "conflicting_observation",
        "incomplete_location"
    ],
    "records": [
        int(df[column].sum())
        for column in [
            "negative_groundwater_level",
            "zero_groundwater_level",
            "extreme_groundwater_level",
            "conflicting_observation",
            "incomplete_location"
        ]
    ]
})

quality_summary.to_csv(
    OUTPUT_DIR / "quality_flag_distribution.csv",
    index=False
)

print("\nEDA COMPLETE")
print(f"Outputs saved to: {OUTPUT_DIR}")
