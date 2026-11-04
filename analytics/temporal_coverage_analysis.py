from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "January_WL_1994-2025.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "temporal_coverage"

START_YEAR = 1994
END_YEAR = 2025


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%y",
        errors="coerce",
    )

    df["YEAR"] = df["Date"].dt.year

    # Keep valid observations within the intended study period.
    df = df[
        df["Date"].notna()
        & df["LATITUDE"].notna()
        & df["LONGITUDE"].notna()
        & df["YEAR"].between(START_YEAR, END_YEAR)
    ].copy()

    # Provisional location identifier.
    location_columns = ["LATITUDE", "LONGITUDE"]

    # Number of observations per year.
    yearly = (
        df.groupby("YEAR")
        .agg(
            observations=("Date", "size"),
            unique_dates=("Date", "nunique"),
            unique_locations=(
                "LATITUDE",
                "count",
            ),
        )
        .reindex(range(START_YEAR, END_YEAR + 1), fill_value=0)
        .rename_axis("YEAR")
        .reset_index()
    )

    yearly.to_csv(
        OUTPUT_DIR / "yearly_coverage.csv",
        index=False,
    )

    # Number of distinct years and dates per location.
    location_summary = (
        df.groupby(location_columns)
        .agg(
            first_date=("Date", "min"),
            last_date=("Date", "max"),
            years_observed=("YEAR", "nunique"),
            observation_dates=("Date", "nunique"),
            observations=("Date", "size"),
        )
        .reset_index()
    )

    location_summary["calendar_span_years"] = (
        location_summary["last_date"].dt.year
        - location_summary["first_date"].dt.year
        + 1
    )

    location_summary["year_coverage_ratio"] = (
        location_summary["years_observed"]
        / location_summary["calendar_span_years"]
    )

    location_summary.to_csv(
        OUTPUT_DIR / "location_coverage.csv",
        index=False,
    )

    # Count consecutive-year gaps in the observed years.
    observed_years = (
        df.groupby(location_columns)["YEAR"]
        .apply(lambda values: sorted(set(values)))
    )

    gap_records = []

    for location, years in observed_years.items():
        if len(years) < 2:
            continue

        gaps = [
            years[i + 1] - years[i] - 1
            for i in range(len(years) - 1)
        ]

        gap_records.append({
            "LATITUDE": location[0],
            "LONGITUDE": location[1],
            "years_observed": len(years),
            "largest_missing_year_gap": max(gaps, default=0),
            "missing_year_gaps": sum(gap > 0 for gap in gaps),
        })

    gap_summary = pd.DataFrame(gap_records)

    gap_summary.to_csv(
        OUTPUT_DIR / "location_year_gaps.csv",
        index=False,
    )

    # Candidate locations for preliminary trend analysis.
    # These are screening criteria, not scientific guarantees.
    candidates = location_summary[
        (location_summary["years_observed"] >= 10)
        & (location_summary["year_coverage_ratio"] >= 0.70)
        & (location_summary["observation_dates"] >= 10)
    ].copy()

    candidates.to_csv(
        OUTPUT_DIR / "trend_analysis_candidates.csv",
        index=False,
    )

    # Plot 1: observations per year.
    plt.figure(figsize=(12, 5))
    plt.plot(
        yearly["YEAR"],
        yearly["observations"],
        marker="o",
    )
    plt.xlabel("Year")
    plt.ylabel("Number of observations")
    plt.title("Historical Groundwater Observation Coverage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "yearly_observations.png",
        dpi=200,
    )
    plt.close()

    # Plot 2: distribution of distinct observed years.
    plt.figure(figsize=(10, 5))
    plt.hist(
        location_summary["years_observed"],
        bins=range(
            1,
            int(location_summary["years_observed"].max()) + 2,
        ),
        align="left",
        edgecolor="black",
    )
    plt.xlabel("Distinct years with observations")
    plt.ylabel("Number of locations")
    plt.title("Temporal Coverage Across Monitoring Locations")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "location_year_distribution.png",
        dpi=200,
    )
    plt.close()

    print("\n===== TEMPORAL COVERAGE ANALYSIS =====")
    print("Valid records:", len(df))
    print("Unique coordinate pairs:", len(location_summary))
    print("Locations observed in >= 10 years:",
          (location_summary["years_observed"] >= 10).sum())
    print("Locations spanning >= 10 calendar years:",
          (location_summary["calendar_span_years"] >= 10).sum())
    print("Potential trend-analysis candidates:", len(candidates))

    print("\n===== YEARLY COVERAGE =====")
    print(yearly.to_string(index=False))

    print("\n===== LONGEST OBSERVED HISTORIES =====")
    print(
        location_summary.sort_values(
            ["years_observed", "observation_dates"],
            ascending=False,
        ).head(10).to_string(index=False)
    )

    print("\nReports saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()