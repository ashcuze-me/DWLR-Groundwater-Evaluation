
from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/processed/team_dataset/multiseason/"
    "location_year_seasonal.csv"
)

OUTPUT = Path(
    "outputs/team_temporal_coverage"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

print("Loading location-year dataset...")

df = pd.read_csv(INPUT)

df["YEAR"] = pd.to_numeric(
    df["YEAR"], errors="coerce"
).astype("Int64")

# Number of complete locations in each year.
yearly = (
    df.groupby("YEAR")
    .agg(
        complete_location_years=("LOCATION_KEY", "nunique")
    )
    .reset_index()
)

# Count consecutive-year location pairs.
current = df[["LOCATION_KEY", "YEAR"]].copy()

future = df[["LOCATION_KEY", "YEAR"]].copy()
future["YEAR"] = future["YEAR"] - 1

transitions = current.merge(
    future,
    on=["LOCATION_KEY", "YEAR"],
    how="inner",
    validate="one_to_one",
)

transition_counts = (
    transitions.groupby("YEAR")
    .agg(
        consecutive_location_pairs=("LOCATION_KEY", "nunique")
    )
    .reset_index()
)

transition_counts["TARGET_YEAR"] = (
    transition_counts["YEAR"] + 1
)

report = transition_counts.merge(
    yearly,
    on="YEAR",
    how="left",
)

report = report.merge(
    yearly.rename(
        columns={
            "YEAR": "TARGET_YEAR",
            "complete_location_years":
                "target_year_complete_locations",
        }
    ),
    on="TARGET_YEAR",
    how="left",
)

report = report.sort_values("TARGET_YEAR")

report.to_csv(
    OUTPUT / "multiseason_year_transition_audit.csv",
    index=False,
)

print("\nCOMPLETE LOCATION-YEAR COUNTS")
print(yearly.to_string(index=False))

print("\nCONSECUTIVE-YEAR TRANSITION COUNTS")
print(report.to_string(index=False))

print("\nTOTAL CONSECUTIVE-YEAR SAMPLES")
print(
    f"{report['consecutive_location_pairs'].sum():,}"
)

print("\nAudit complete.")
