from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

JAN_TREND = (
    ROOT / "outputs" / "groundwater_trends_v2"
    / "location_trends.csv"
)

AUG_TREND = (
    ROOT / "outputs" / "august_groundwater_trends"
    / "august_location_trends.csv"
)

JAN_DATA = (
    ROOT / "data" / "raw"
    / "January_WL_1994-2025.csv"
)

AUG_DATA = (
    ROOT / "data" / "processed" / "august"
    / "August_WL_1994-2025_clean.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "seasonal_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHED_FILE = OUTPUT_DIR / "matched_location_trends.csv"
LEVEL_FILE = OUTPUT_DIR / "seasonal_groundwater_levels.csv"
TREND_FILE = OUTPUT_DIR / "seasonal_trend_comparison.csv"
SUMMARY_FILE = OUTPUT_DIR / "seasonal_comparison_report.txt"

MATCH_COLUMNS = [
    "state",
    "district",
    "block",
    "village",
    "latitude",
    "longitude"
]

# --------------------------------------------------
# LOAD AND NORMALISE TREND SCHEMAS
# --------------------------------------------------

def load_trend_file(path, season):

    df = pd.read_csv(path)

    if season == "january":

        df = df.rename(columns={
            "STATE_UT": "state",
            "DISTRICT": "district",
            "BLOCK": "block",
            "VILLAGE": "village",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "slope_m_per_year": "slope",
            "slope_lower_95": "slope_lower",
            "slope_upper_95": "slope_upper",
            "kendall_p_value": "p_value",
            "kendall_tau": "kendall_tau",
            "trend_direction": "trend",
            "years_observed": "observed_years",
            "observation_dates": "observations"
        })

        df["mean_water_level_mbgl"] = np.nan

    else:

        df = df.rename(columns={
            "theil_sen_slope_mbgl_per_year": "slope",
            "slope_ci_lower": "slope_lower",
            "slope_ci_upper": "slope_upper",
            "p_value": "p_value"
        })

    required = MATCH_COLUMNS + [
        "slope",
        "trend",
        "p_value"
    ]

    missing = [
        col for col in required if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{season} file missing columns: {missing}"
        )

    for col in ["state", "district", "block", "village"]:
        df[col] = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).round(6)

    return df


print("\nLoading January trend results...")
jan = load_trend_file(JAN_TREND, "january")

print("Loading August trend results...")
aug = load_trend_file(AUG_TREND, "august")

# --------------------------------------------------
# STANDARDISE TREND LABELS
# --------------------------------------------------

def standardise_trend(value):

    value = str(value).strip().lower()

    if value in [
        "increasing",
        "increasing recorded depth"
    ]:
        return "Increasing"

    if value in [
        "decreasing",
        "decreasing recorded depth"
    ]:
        return "Decreasing"

    return "Uncertain"


jan["trend"] = jan["trend"].apply(standardise_trend)
aug["trend"] = aug["trend"].apply(standardise_trend)

# --------------------------------------------------
# MATCHING
# --------------------------------------------------

print("Preparing location matching...")

jan = jan.dropna(subset=["latitude", "longitude"])
aug = aug.dropna(subset=["latitude", "longitude"])

# Exclude duplicate keys to avoid many-to-many joins
jan = jan[
    ~jan.duplicated(subset=MATCH_COLUMNS, keep=False)
].copy()

aug = aug[
    ~aug.duplicated(subset=MATCH_COLUMNS, keep=False)
].copy()

print("Matching January and August locations...")

jan_indexed = jan.set_index(MATCH_COLUMNS)
aug_indexed = aug.set_index(MATCH_COLUMNS)

matched = jan_indexed.join(
    aug_indexed,
    how="inner",
    lsuffix="_jan",
    rsuffix="_aug"
).reset_index()

# --------------------------------------------------
# TREND COMPARISON
# --------------------------------------------------

print("Comparing trend classifications...")

matched["trend_comparison"] = np.select(
    [
        (
            (matched["trend_jan"] == "Increasing")
            & (matched["trend_aug"] == "Increasing")
        ),
        (
            (matched["trend_jan"] == "Decreasing")
            & (matched["trend_aug"] == "Decreasing")
        ),
        (
            (matched["trend_jan"] == "Increasing")
            & (matched["trend_aug"] == "Decreasing")
        ),
        (
            (matched["trend_jan"] == "Decreasing")
            & (matched["trend_aug"] == "Increasing")
        )
    ],
    [
        "Increasing in both seasons",
        "Decreasing in both seasons",
        "Opposite seasonal trends",
        "Opposite seasonal trends"
    ],
    default="At least one uncertain trend"
)

matched["slope_difference"] = (
    matched["slope_jan"] - matched["slope_aug"]
)

matched.to_csv(MATCHED_FILE, index=False)

# --------------------------------------------------
# SEASONAL LEVEL COMPARISON
# --------------------------------------------------

print("Loading cleaned observation datasets...")

jan_data = pd.read_csv(JAN_DATA)
aug_data = pd.read_csv(AUG_DATA)


def prepare_observations(df):

    df = df.copy()
    df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    )
    df = df.rename(columns={
        "state_ut": "state",
        "wl(mbgl)": "water_level_mbgl",
        "wl_mbgl": "water_level_mbgl"
        })

    for col in MATCH_COLUMNS[:4]:
        df[col] = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).round(6)

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df["water_level_mbgl"] = pd.to_numeric(
        df["water_level_mbgl"], errors="coerce"
    )

    return df.dropna(
        subset=[
            "latitude",
            "longitude",
            "date",
            "water_level_mbgl"
        ]
    )


jan_data = prepare_observations(jan_data)
aug_data = prepare_observations(aug_data)

# Aggregate observations by location and year.
# This enables a fairer comparison using matching years.

jan_data["year"] = jan_data["date"].dt.year
aug_data["year"] = aug_data["date"].dt.year

jan_levels = (
    jan_data.groupby(MATCH_COLUMNS + ["year"])
    ["water_level_mbgl"]
    .mean()
    .reset_index(name="january_mean_mbgl")
)

aug_levels = (
    aug_data.groupby(MATCH_COLUMNS + ["year"])
    ["water_level_mbgl"]
    .mean()
    .reset_index(name="august_mean_mbgl")
)

seasonal_levels = jan_levels.merge(
    aug_levels,
    on=MATCH_COLUMNS + ["year"],
    how="inner"
)

seasonal_levels["august_minus_january_mbgl"] = (
    seasonal_levels["august_mean_mbgl"]
    - seasonal_levels["january_mean_mbgl"]
)

seasonal_levels.to_csv(LEVEL_FILE, index=False)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

trend_summary = matched[
    "trend_comparison"
].value_counts()

report = f"""
JANUARY-AUGUST SEASONAL COMPARISON
==========================================

TREND COMPARISON
------------------------------------------

January trend locations: {len(jan)}

August trend locations: {len(aug)}

Matched locations: {len(matched)}

TREND AGREEMENT
------------------------------------------

{trend_summary.to_string()}

SEASONAL LEVEL COMPARISON
------------------------------------------

Matched location-year observations:
{len(seasonal_levels)}

Unique matched locations:
{seasonal_levels[MATCH_COLUMNS].drop_duplicates().shape[0]}

Mean January groundwater depth:
{seasonal_levels['january_mean_mbgl'].mean():.3f} mbgl

Mean August groundwater depth:
{seasonal_levels['august_mean_mbgl'].mean():.3f} mbgl

Mean August minus January difference:
{seasonal_levels['august_minus_january_mbgl'].mean():.3f} mbgl

METHODOLOGICAL NOTES
------------------------------------------

Trend locations were matched using:
State, district, block, village,
latitude and longitude.

Duplicate matching keys were excluded.

Seasonal level comparisons use matching
location-year combinations.

Seasonal differences are descriptive and
should not automatically be interpreted as
causal seasonal effects.

OUTPUT FILES
------------------------------------------

{MATCHED_FILE}

{LEVEL_FILE}
"""

SUMMARY_FILE.write_text(report, encoding="utf-8")

print(report)

print("\nSeasonal comparison completed!")