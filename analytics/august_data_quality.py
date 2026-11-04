from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "raw" / "August_WL_1994-2025.csv"
OUTPUT_DIR = ROOT / "outputs" / "august_audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\n" + "=" * 60)
print("AUGUST GROUNDWATER DATA QUALITY AUDIT")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# Standardise column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

# Map possible column-name variations
column_mapping = {
    "state_ut": "state",
    "state": "state",
    "district": "district",
    "block": "block",
    "village": "village",
    "latitute": "latitude",
    "latitude": "latitude",
    "longitude": "longitude",
    "date": "date",
    "wl_mbgl": "water_level_mbgl",
    "wl_m_bgl": "water_level_mbgl",
    "water_level_mbgl": "water_level_mbgl",
}

df = df.rename(columns=column_mapping)

required_columns = [
    "state",
    "district",
    "block",
    "village",
    "latitude",
    "longitude",
    "date",
    "water_level_mbgl",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )

df = df[required_columns].copy()


# --------------------------------------------------
# DATA TYPE CONVERSION
# --------------------------------------------------

for col in ["state", "district", "block", "village"]:
    df[col] = df[col].astype("string").str.strip()

for col in ["latitude", "longitude", "water_level_mbgl"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["date"] = pd.to_datetime(
    df["date"],
    format="%Y-%m-%d",
    errors="coerce"
)


# --------------------------------------------------
# BASIC DATA QUALITY
# --------------------------------------------------

total_rows = len(df)
total_columns = len(df.columns)

missing_values = df.isnull().sum()

exact_duplicates = df.duplicated().sum()

invalid_dates = df["date"].isna().sum()

invalid_latitude = (
    (df["latitude"] < -90) |
    (df["latitude"] > 90)
).sum()

invalid_longitude = (
    (df["longitude"] < -180) |
    (df["longitude"] > 180)
).sum()

invalid_coordinates = (
    df["latitude"].isna() |
    df["longitude"].isna() |
    (df["latitude"] < -90) |
    (df["latitude"] > 90) |
    (df["longitude"] < -180) |
    (df["longitude"] > 180)
).sum()

invalid_water_levels = (
    df["water_level_mbgl"].isna() |
    (df["water_level_mbgl"] < 0)
).sum()

non_august_dates = (
    df["date"].notna() &
    (df["date"].dt.month != 8)
).sum()


# --------------------------------------------------
# DUPLICATE ANALYSIS
# --------------------------------------------------

duplicate_records = df[df.duplicated(keep=False)]

duplicate_records.to_csv(
    OUTPUT_DIR / "exact_duplicate_records.csv",
    index=False
)


# --------------------------------------------------
# YEARLY COVERAGE
# --------------------------------------------------

valid_dates = df[df["date"].notna()].copy()

yearly_coverage = (
    valid_dates.groupby(valid_dates["date"].dt.year)
    .agg(
        records=("date", "size"),
        unique_dates=("date", "nunique"),
        unique_states=("state", "nunique"),
        unique_districts=("district", "nunique"),
        unique_coordinates=("latitude", "count"),
    )
    .reset_index()
    .rename(columns={"date": "year"})
)

yearly_coverage.to_csv(
    OUTPUT_DIR / "yearly_coverage.csv",
    index=False
)


# --------------------------------------------------
# LOCATION COVERAGE
# --------------------------------------------------

location_columns = [
    "state",
    "district",
    "block",
    "village",
    "latitude",
    "longitude",
]

location_coverage = (
    df.groupby(location_columns, dropna=False)
    .agg(
        total_observations=("date", "size"),
        unique_dates=("date", "nunique"),
        first_observation=("date", "min"),
        last_observation=("date", "max"),
        mean_water_level=("water_level_mbgl", "mean"),
        median_water_level=("water_level_mbgl", "median"),
    )
    .reset_index()
)

location_coverage.to_csv(
    OUTPUT_DIR / "location_coverage.csv",
    index=False
)


# --------------------------------------------------
# REPEATED LOCATION-DATE OBSERVATIONS
# --------------------------------------------------

valid_location_dates = df.dropna(
    subset=location_columns + ["date"]
)

repeated_groups = (
    valid_location_dates.groupby(
        location_columns + ["date"],
        dropna=False
    )
    .agg(
        records=("water_level_mbgl", "size"),
        distinct_readings=("water_level_mbgl", "nunique"),
        minimum_level=("water_level_mbgl", "min"),
        maximum_level=("water_level_mbgl", "max"),
    )
    .reset_index()
)

repeated_groups = repeated_groups[
    repeated_groups["records"] > 1
]

repeated_groups.to_csv(
    OUTPUT_DIR / "repeated_location_dates.csv",
    index=False
)


# --------------------------------------------------
# SUMMARY STATISTICS
# --------------------------------------------------

water_level_summary = df["water_level_mbgl"].describe()

state_counts = (
    df["state"]
    .value_counts()
    .head(15)
)

district_counts = (
    df["district"]
    .value_counts()
    .head(15)
)


# --------------------------------------------------
# GENERATE TEXT REPORT
# --------------------------------------------------

report = []

report.append("=" * 60)
report.append("AUGUST GROUNDWATER DATA QUALITY REPORT")
report.append("=" * 60)

report.append(f"\nTotal rows: {total_rows}")
report.append(f"Total columns: {total_columns}")

report.append("\n--- GEOGRAPHICAL COVERAGE ---")
report.append(f"States/UTs: {df['state'].nunique()}")
report.append(f"Districts: {df['district'].nunique()}")
report.append(f"Blocks: {df['block'].nunique()}")
report.append(f"Villages: {df['village'].nunique()}")

report.append(
    f"Unique coordinate pairs: "
    f"{df[['latitude', 'longitude']].drop_duplicates().shape[0]}"
)

report.append("\n--- DATE COVERAGE ---")

if valid_dates.empty:
    report.append("No valid dates found.")
else:
    report.append(f"Earliest date: {valid_dates['date'].min()}")
    report.append(f"Latest date: {valid_dates['date'].max()}")
    report.append(f"Unique dates: {valid_dates['date'].nunique()}")

report.append(f"Invalid dates: {invalid_dates}")
report.append(f"Non-August dates: {non_august_dates}")

report.append("\n--- MISSING VALUES ---")

for col, count in missing_values.items():
    percentage = (count / total_rows) * 100
    report.append(
        f"{col}: {count} ({percentage:.2f}%)"
    )

report.append("\n--- DUPLICATES ---")
report.append(f"Exact duplicate rows: {exact_duplicates}")
report.append(
    f"Repeated location-date groups: {len(repeated_groups)}"
)

report.append("\n--- INVALID VALUES ---")
report.append(f"Invalid latitude values: {invalid_latitude}")
report.append(f"Invalid longitude values: {invalid_longitude}")
report.append(f"Rows with invalid coordinates: {invalid_coordinates}")
report.append(f"Missing or negative water levels: {invalid_water_levels}")

report.append("\n--- WATER LEVEL STATISTICS (mbgl) ---")

for statistic, value in water_level_summary.items():
    report.append(f"{statistic}: {value:.3f}")

report.append("\n--- TOP 15 STATES ---")
report.append(state_counts.to_string())

report.append("\n--- TOP 15 DISTRICTS ---")
report.append(district_counts.to_string())

report.append("\n--- YEARLY COVERAGE ---")
report.append(yearly_coverage.to_string(index=False))

report_text = "\n".join(report)

with open(
    OUTPUT_DIR / "august_data_quality_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report_text)


# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------

print(report_text)

print("\n" + "=" * 60)
print("AUDIT COMPLETED")
print(f"Reports saved to: {OUTPUT_DIR}")
print("=" * 60)