from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "august"
    / "August_WL_1994-2025_standardized.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "august_temporal_coverage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COVERAGE_FILE = OUTPUT_DIR / "august_location_coverage.csv"
SUMMARY_FILE = OUTPUT_DIR / "august_temporal_coverage_report.txt"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading standardised August dataset...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year

df = df.dropna(subset=["date", "location_key"])

# --------------------------------------------------
# LOCATION-LEVEL TEMPORAL COVERAGE
# --------------------------------------------------

print("Calculating temporal coverage...")

coverage = (
    df.groupby("location_key")
    .agg(
        state=("state", "first"),
        district=("district", "first"),
        block=("block", "first"),
        village=("village", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        total_records=("date", "size"),
        unique_dates=("date", "nunique"),
        first_observation=("date", "min"),
        last_observation=("date", "max"),
        observed_years=("year", "nunique")
    )
    .reset_index()
)

# --------------------------------------------------
# CALENDAR YEAR COVERAGE
# --------------------------------------------------

coverage["calendar_year_span"] = (
    coverage["last_observation"].dt.year
    - coverage["first_observation"].dt.year
    + 1
)

coverage["year_coverage_ratio"] = (
    coverage["observed_years"]
    / coverage["calendar_year_span"]
)

# --------------------------------------------------
# ELIGIBILITY CRITERIA
# --------------------------------------------------

coverage["eligible_for_trend_analysis"] = (
    (coverage["observed_years"] >= 10)
    & (coverage["year_coverage_ratio"] >= 0.70)
    & (coverage["unique_dates"] >= 10)
)

# --------------------------------------------------
# SAVE LOCATION COVERAGE
# --------------------------------------------------

coverage.to_csv(COVERAGE_FILE, index=False)

# --------------------------------------------------
# SUMMARY STATISTICS
# --------------------------------------------------

total_locations = len(coverage)

ten_year_locations = (
    coverage["observed_years"] >= 10
).sum()

ten_calendar_year_locations = (
    coverage["calendar_year_span"] >= 10
).sum()

eligible_locations = (
    coverage["eligible_for_trend_analysis"]
).sum()

summary = f"""
AUGUST TEMPORAL COVERAGE ANALYSIS
==========================================

Total records analysed: {len(df)}

Total provisional locations: {total_locations}

------------------------------------------
TEMPORAL COVERAGE
------------------------------------------

Locations with >= 10 observed years:
{ten_year_locations}

Locations spanning >= 10 calendar years:
{ten_calendar_year_locations}

Locations meeting all trend eligibility criteria:
{eligible_locations}

------------------------------------------
ELIGIBILITY CRITERIA
------------------------------------------

1. At least 10 distinct observation years.
2. At least 70% calendar-year coverage.
3. At least 10 distinct observation dates.

------------------------------------------
OUTPUT
------------------------------------------

{COVERAGE_FILE}
"""

SUMMARY_FILE.write_text(summary, encoding="utf-8")

print(summary)

print("\nAugust temporal coverage analysis completed!")