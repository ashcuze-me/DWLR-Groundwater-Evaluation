
from pathlib import Path
import pandas as pd

INPUT = Path(
    "outputs/team_temporal_coverage/eligible_locations.csv"
)

OUTPUT = Path("outputs/team_temporal_coverage")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

season_columns = [
    "years_month_1",
    "years_month_5",
    "years_month_8",
    "years_month_11",
]

print("ELIGIBLE LOCATION COVERAGE")

print(f"\nEligible locations: {len(df)}")

print("\nYears of coverage by season:")
print(df[season_columns].describe().round(2))

print("\nLocations by number of complete four-season years:")
print(
    df["years_with_all_four_seasons"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nCoverage thresholds:")
for threshold in [8, 10, 12, 15]:
    count = (
        df["years_with_all_four_seasons"] >= threshold
    ).sum()

    print(
        f"At least {threshold} complete years: "
        f"{count:,} locations"
    )

# Save a compact coverage distribution.
distribution = (
    df["years_with_all_four_seasons"]
    .value_counts()
    .sort_index()
    .rename_axis("complete_four_season_years")
    .reset_index(name="location_count")
)

distribution.to_csv(
    OUTPUT / "eligible_coverage_distribution.csv",
    index=False,
)

print("\nInspection complete.")
