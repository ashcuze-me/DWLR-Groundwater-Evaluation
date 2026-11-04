
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "January_WL_1994-2025.csv"
OUT = ROOT / "outputs" / "groundwater_trends_v2"

LOCATION = [
    "STATE_UT",
    "DISTRICT",
    "BLOCK",
    "VILLAGE",
    "LATITUDE",
    "LONGITUDE",
]

MIN_OBSERVATIONS = 10
MIN_YEARS = 10


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    df["Date"] = pd.to_datetime(
        df["Date"], format="%d-%m-%y", errors="coerce"
    )
    df["WL(mbgl)"] = pd.to_numeric(
        df["WL(mbgl)"], errors="coerce"
    )

    df = df.dropna(
        subset=LOCATION + ["Date", "WL(mbgl)"]
    ).copy()

    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Flag repeated dates within provisional locations.
    keys = LOCATION + ["Date"]
    daily = (
        df.groupby(keys, dropna=False)
        .agg(
            records=("WL(mbgl)", "size"),
            distinct_levels=("WL(mbgl)", "nunique"),
            minimum_level=("WL(mbgl)", "min"),
            maximum_level=("WL(mbgl)", "max"),
        )
        .reset_index()
    )

    repeated = daily[daily["records"] > 1].copy()
    repeated["level_range"] = (
        repeated["maximum_level"] - repeated["minimum_level"]
    )
    repeated.to_csv(OUT / "repeated_dates_flagged.csv", index=False)

    # Primary analysis: use only dates with one remaining record.
    # This avoids silently merging potentially different wells.
    clean_keys = daily.loc[daily["records"] == 1, keys]
    clean = df.merge(clean_keys, on=keys, how="inner")

    results = []

    for location, group in clean.groupby(LOCATION, dropna=False):
        series = group.set_index("Date")["WL(mbgl)"].sort_index()

        if len(series) < MIN_OBSERVATIONS:
            continue

        years = series.index.year.nunique()
        if years < MIN_YEARS:
            continue

        elapsed = (
            (series.index - series.index.min()).days / 365.25
        ).to_numpy()
        values = series.to_numpy(dtype=float)

        slope, intercept, lower, upper = theilslopes(
            values, elapsed, alpha=0.95
        )
        tau, p_value = kendalltau(elapsed, values)

        results.append({
            **dict(zip(LOCATION, location)),
            "first_date": series.index.min(),
            "last_date": series.index.max(),
            "observation_dates": len(series),
            "years_observed": years,
            "slope_m_per_year": slope,
            "slope_lower_95": lower,
            "slope_upper_95": upper,
            "kendall_tau": tau,
            "kendall_p_value": p_value,
            "trend_direction": (
                "Increasing recorded depth" if lower > 0
                else "Decreasing recorded depth" if upper < 0
                else "Uncertain"
            ),
        })

    trends = pd.DataFrame(results)
    trends.to_csv(OUT / "location_trends.csv", index=False)

    print("\n===== TREND ANALYSIS V2 =====")
    print("Input rows:", len(pd.read_csv(DATA)))
    print("Exact duplicate rows removed from analysis:", exact_duplicates)
    print("Repeated location-date groups flagged:", len(repeated))
    print("Dates retained for analysis:", len(clean))
    print("Locations analysed:", len(trends))

    if not trends.empty:
        print("\n===== TREND CLASSIFICATION =====")
        print(trends["trend_direction"].value_counts().to_string())

    print("\nSaved outputs to:", OUT)


if __name__ == "__main__":
    main()
