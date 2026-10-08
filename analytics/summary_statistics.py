from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/raw/January_2026.csv")


def main():
    df = pd.read_csv(DATA_FILE)

    print("=" * 60)
    print("DWLR SUMMARY STATISTICS")
    print("=" * 60)

    print("\n--- DTWL STATISTICS ---")
    print(df["DTWL"].describe())

    print("\n--- TOP STATES ---")
    print(df["STATE_UT"].value_counts().head(10))

    print("\n--- TOP DISTRICTS ---")
    print(df["DISTRICT"].value_counts().head(10))

    print("\n--- DTWL BY STATE ---")
    state_stats = (
        df.groupby("STATE_UT")["DTWL"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values("mean", ascending=False)
    )

    print(state_stats)

    print("\n--- DTWL BY DISTRICT ---")
    district_stats = (
        df.groupby("DISTRICT")["DTWL"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values("mean", ascending=False)
    )

    print(district_stats.head(20))

    print("\n" + "=" * 60)
    print("SUMMARY COMPLETE")
    print("=" * 60)
    
if __name__ == "__main__":
    main()