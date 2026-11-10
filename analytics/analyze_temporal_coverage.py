
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "team_dataset"
    / "team_quality_flagged.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "team_eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["Date_PARSED"] = pd.to_datetime(
    df["Date_PARSED"], errors="coerce"
)

df["year"] = df["Date_PARSED"].dt.year
df["month"] = df["Date_PARSED"].dt.month

valid = df.dropna(
    subset=["year", "month", "WL_NUMERIC"]
).copy()

# --------------------------------------------------
# 1. YEAR-MONTH OBSERVATION MATRIX
# --------------------------------------------------

coverage = pd.crosstab(
    valid["year"],
    valid["month"]
)

coverage = coverage.reindex(
    columns=range(1, 13),
    fill_value=0
)

coverage.columns = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

coverage.to_csv(
    OUTPUT_DIR / "year_month_coverage_matrix.csv"
)

# --------------------------------------------------
# 2. YEARLY MONTH COVERAGE
# --------------------------------------------------

yearly = coverage.copy()

yearly["total_observations"] = yearly.sum(axis=1)

yearly["months_with_observations"] = (
    (yearly.iloc[:, :12] > 0).sum(axis=1)
)

yearly.to_csv(
    OUTPUT_DIR / "yearly_month_coverage.csv"
)

# --------------------------------------------------
# 3. COVERAGE BY MONTH
# --------------------------------------------------

monthly_summary = pd.DataFrame({
    "month": coverage.columns,
    "total_observations": coverage.sum().values,
    "years_with_observations": (coverage > 0).sum().values,
    "years_with_at_least_100_observations":
        (coverage >= 100).sum().values
})

monthly_summary.to_csv(
    OUTPUT_DIR / "monthly_coverage_completeness.csv",
    index=False
)

# --------------------------------------------------
# 4. FOUR MAJOR SEASONS
# --------------------------------------------------

major_months = ["Jan", "May", "Aug", "Nov"]

major_coverage = coverage[major_months].copy()

major_coverage["total_observations"] = (
    major_coverage.sum(axis=1)
)

major_coverage.to_csv(
    OUTPUT_DIR / "major_season_coverage.csv"
)

# --------------------------------------------------
# 5. PRINT RESULTS
# --------------------------------------------------

print("\nYEAR-MONTH COVERAGE MATRIX")
print(coverage.to_string())

print("\nMONTHLY COVERAGE COMPLETENESS")
print(monthly_summary.to_string(index=False))

print("\nMAJOR SEASON COVERAGE")
print(major_coverage.to_string())

print("\nYEARS WITH ALL FOUR MAJOR MONTHS")
complete_years = major_coverage[
    (major_coverage[major_months] > 0).all(axis=1)
]

print(complete_years.index.tolist())

print("\nAnalysis complete.")
print(f"Outputs saved to: {OUTPUT_DIR}")
