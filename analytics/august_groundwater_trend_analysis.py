from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import theilslopes, kendalltau

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "august"
    / "August_WL_1994-2025_standardized.csv"
)

COVERAGE_FILE = (
    ROOT / "outputs" / "august_temporal_coverage"
    / "august_location_coverage.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "august_groundwater_trends"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "august_location_trends.csv"
SUMMARY_FILE = OUTPUT_DIR / "august_trend_summary.txt"
REPEATED_FILE = OUTPUT_DIR / "august_excluded_repeated_records.csv"

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MIN_OBSERVED_YEARS = 10
MIN_UNIQUE_DATES = 10
MIN_COVERAGE_RATIO = 0.70
SIGNIFICANCE_LEVEL = 0.05

LOCATION_COLUMNS = [
    "state",
    "district",
    "block",
    "village",
    "latitude",
    "longitude"
]

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading August dataset...")

df = pd.read_csv(INPUT_FILE)
coverage = pd.read_csv(COVERAGE_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year

df["water_level_mbgl"] = pd.to_numeric(
    df["water_level_mbgl"], errors="coerce"
)

df = df.dropna(
    subset=["date", "water_level_mbgl", "location_key"]
).copy()

# --------------------------------------------------
# REMOVE EXACT DUPLICATES
# --------------------------------------------------

initial_rows = len(df)

df = df.drop_duplicates()

exact_duplicates_removed = initial_rows - len(df)

# --------------------------------------------------
# IDENTIFY REPEATED LOCATION-DATE GROUPS
# --------------------------------------------------

print("Identifying repeated location-date records...")

repeated_mask = df.duplicated(
    subset=["location_key", "date"],
    keep=False
)

repeated_records = df[repeated_mask].copy()

# Exclude all observations in repeated groups
analysis_df = df[~repeated_mask].copy()

repeated_records.to_csv(REPEATED_FILE, index=False)

# --------------------------------------------------
# SELECT ELIGIBLE LOCATIONS
# --------------------------------------------------

print("Selecting eligible locations...")

eligible = coverage[
    (coverage["observed_years"] >= MIN_OBSERVED_YEARS)
    & (coverage["year_coverage_ratio"] >= MIN_COVERAGE_RATIO)
    & (coverage["unique_dates"] >= MIN_UNIQUE_DATES)
]

eligible_keys = set(eligible["location_key"])

analysis_df = analysis_df[
    analysis_df["location_key"].isin(eligible_keys)
].copy()

# --------------------------------------------------
# TREND ANALYSIS FUNCTION
# --------------------------------------------------

def calculate_trend(group):

    group = group.sort_values("date")

    # Use elapsed calendar years for slope calculation
    x = (
        group["date"] - group["date"].min()
    ).dt.days.to_numpy() / 365.2425

    y = group["water_level_mbgl"].to_numpy(dtype=float)

    result = {
        "observations": len(group),
        "observed_years": group["year"].nunique(),
        "first_observation": group["date"].min(),
        "last_observation": group["date"].max(),
        "mean_water_level_mbgl": np.mean(y),
        "median_water_level_mbgl": np.median(y),
        "min_water_level_mbgl": np.min(y),
        "max_water_level_mbgl": np.max(y),
        "theil_sen_slope_mbgl_per_year": np.nan,
        "slope_ci_lower": np.nan,
        "slope_ci_upper": np.nan,
        "kendall_tau": np.nan,
        "p_value": np.nan,
        "trend": "Uncertain"
    }

    if len(group) < 10 or len(np.unique(x)) < 2:
        return pd.Series(result)

    try:
        slope, intercept, low, high = theilslopes(
            y,
            x,
            alpha=0.95
        )

        tau, p_value = kendalltau(
            x,
            y,
            nan_policy="omit"
        )

        result["theil_sen_slope_mbgl_per_year"] = slope
        result["slope_ci_lower"] = low
        result["slope_ci_upper"] = high
        result["kendall_tau"] = tau
        result["p_value"] = p_value

        # Positive slope = increasing depth below ground level
        if p_value < SIGNIFICANCE_LEVEL:

            if low > 0:
                result["trend"] = "Increasing"

            elif high < 0:
                result["trend"] = "Decreasing"

            else:
                result["trend"] = "Uncertain"

        else:
            result["trend"] = "Uncertain"

    except Exception:
        pass

    return pd.Series(result)

# --------------------------------------------------
# RUN ANALYSIS
# --------------------------------------------------

print("Calculating groundwater trends...")

results = []

for index, (location_key, group) in enumerate(
    analysis_df.groupby("location_key"),
    start=1
):

    trend = calculate_trend(group)

    trend["location_key"] = location_key

    for col in LOCATION_COLUMNS:
        trend[col] = group[col].iloc[0]

    results.append(trend)

    if index % 1000 == 0:
        print(f"Processed {index} locations...")

results_df = pd.DataFrame(results)

# --------------------------------------------------
# SUMMARY STATISTICS
# --------------------------------------------------

total_eligible = len(eligible_keys)
analysed_locations = len(results_df)

increasing = (
    results_df["trend"] == "Increasing"
).sum()

decreasing = (
    results_df["trend"] == "Decreasing"
).sum()

uncertain = (
    results_df["trend"] == "Uncertain"
).sum()

significant = (
    results_df["p_value"] < SIGNIFICANCE_LEVEL
).sum()

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results_df.to_csv(RESULTS_FILE, index=False)

# --------------------------------------------------
# REPORT
# --------------------------------------------------

summary = f"""
AUGUST GROUNDWATER TREND ANALYSIS
==========================================

Input records: {len(df)}

Exact duplicates removed: {exact_duplicates_removed}

Repeated location-date records excluded: {len(repeated_records)}

Records available for analysis:
{len(analysis_df)}

------------------------------------------
LOCATION STATISTICS
------------------------------------------

Eligible locations: {total_eligible}

Locations analysed: {analysed_locations}

------------------------------------------
TREND CLASSIFICATION
------------------------------------------

Increasing depth below ground level: {increasing}

Decreasing depth below ground level: {decreasing}

Uncertain trends: {uncertain}

Statistically significant Kendall tests:
{significant}

------------------------------------------
METHODOLOGY
------------------------------------------

Trend estimator: Theil-Sen slope

Significance test: Kendall's tau

Significance threshold: {SIGNIFICANCE_LEVEL}

Minimum observed years: {MIN_OBSERVED_YEARS}

Minimum unique dates: {MIN_UNIQUE_DATES}

Minimum calendar-year coverage: {MIN_COVERAGE_RATIO}

Repeated location-date groups were excluded
from primary trend calculations.

Extreme readings were retained.

------------------------------------------
OUTPUT FILES
------------------------------------------

Trend results:
{RESULTS_FILE}

Excluded repeated records:
{REPEATED_FILE}

IMPORTANT:
Positive slopes indicate increasing depth to
groundwater (a deeper water table).

Negative slopes indicate decreasing depth to
groundwater (a shallower water table).

These are exploratory statistical results.
Temporal dependence and multiple testing are
not accounted for in the reported p-values.
"""

SUMMARY_FILE.write_text(summary, encoding="utf-8")

print(summary)

print("\nAugust groundwater trend analysis completed!")