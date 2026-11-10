
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" /
    "team_dataset" / "team_combined.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "team_repeated_observations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["WL(mbgl)"] = pd.to_numeric(
    df["WL(mbgl)"], errors="coerce"
)

# Inspect selected conflicting groups
cases = [
    {
        "name": "Gorantla",
        "district": "ANANTAPUR",
        "village": "GORANTLA",
        "latitude": 13.98333333,
        "longitude": 77.76666667,
        "dates": ["2012-11-05", "2013-01-05", "2013-11-05"]
    },
    {
        "name": "Kothapeta",
        "district": "EAST GODAVARI",
        "village": "KOTHAPETA",
        "latitude": 16.71666667,
        "longitude": 81.9,
        "dates": ["2012-11-05", "2013-01-05", "2014-01-05",
                  "2015-05-15", "2017-05-15"]
    }
]

all_results = []

for case in cases:
    print(f"\n{'=' * 60}")
    print(f"INSPECTING: {case['name']}")
    print("=" * 60)

    dates = pd.to_datetime(case["dates"])

    subset = df[
        (df["DISTRICT"].astype(str).str.strip().str.upper()
         == case["district"])
        &
        (df["VILLAGE"].astype(str).str.strip().str.upper()
         == case["village"])
        &
        (pd.to_numeric(df["LATITUDE"], errors="coerce")
         .sub(case["latitude"]).abs() < 0.000001)
        &
        (pd.to_numeric(df["LONGITUDE"], errors="coerce")
         .sub(case["longitude"]).abs() < 0.000001)
        &
        (df["Date"].isin(dates))
    ].copy()

    subset = subset.sort_values(["Date", "WL(mbgl)"])

    print(f"Original records found: {len(subset)}")
    print(subset.to_string(index=False))

    all_results.append(subset)

if all_results:
    result = pd.concat(all_results, ignore_index=True)

    output_file = (
        OUTPUT_DIR / "selected_original_conflicting_records.csv"
    )

    result.to_csv(output_file, index=False)

    print(f"\nSaved original records to: {output_file}")
