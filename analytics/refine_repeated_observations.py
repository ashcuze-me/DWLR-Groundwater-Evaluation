
from pathlib import Path
import pandas as pd
import unicodedata

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "team_dataset" / "team_combined.csv"

OUTPUT_DIR = ROOT / "outputs" / "team_repeated_observations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ADMIN_COLUMNS = ["STATE_UT", "DISTRICT", "BLOCK", "VILLAGE"]

MISSING_VALUES = {
    "", "-", "--", "NA", "N/A", "NONE", "NULL", "NAN"
}

# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

def normalize_identifier(value):
    if pd.isna(value):
        return pd.NA

    value = unicodedata.normalize("NFKC", str(value))
    value = " ".join(value.strip().upper().split())

    if value in MISSING_VALUES:
        return pd.NA

    return value


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading combined dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for column in ADMIN_COLUMNS:
    df[f"{column}_NORM"] = df[column].apply(normalize_identifier)

df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
df["WL(mbgl)"] = pd.to_numeric(df["WL(mbgl)"], errors="coerce")

# --------------------------------------------------
# CLASSIFY IDENTIFIER COMPLETENESS
# --------------------------------------------------

normalized_columns = [f"{column}_NORM" for column in ADMIN_COLUMNS]

df["missing_admin_count"] = df[normalized_columns].isna().sum(axis=1)

df["identifier_status"] = "complete"

df.loc[
    df["missing_admin_count"].between(1, 2),
    "identifier_status"
] = "partially_complete"

df.loc[
    df["missing_admin_count"].ge(3),
    "identifier_status"
] = "highly_incomplete"

print("\nIDENTIFIER COMPLETENESS")
print(df["identifier_status"].value_counts(dropna=False))

# Save completeness summary
completeness = (
    df.groupby("identifier_status")
    .size()
    .reset_index(name="record_count")
)

completeness["percentage"] = (
    completeness["record_count"] / len(df) * 100
)

completeness.to_csv(
    OUTPUT_DIR / "identifier_completeness_summary.csv",
    index=False
)

# --------------------------------------------------
# REPEATED OBSERVATION ANALYSIS
# --------------------------------------------------


def analyse_repeated_records(data, group_columns, label):
    print(f"\nAnalysing {label} records...")

    # Only coordinates, date and groundwater level are
    # mandatory for this candidate-group analysis.
    required_columns = [
        "LATITUDE",
        "LONGITUDE",
        "Date",
        "WL(mbgl)"
    ]

    valid = data.dropna(subset=required_columns).copy()

    # Use an explicit marker for missing administrative IDs.
    # This allows incomplete records to be grouped without
    # pretending that the missing values are real identifiers.
    grouping_data = valid.copy()

    normalized_admin_columns = [
        "STATE_UT_NORM",
        "DISTRICT_NORM",
        "BLOCK_NORM",
        "VILLAGE_NORM"
    ]

    for column in normalized_admin_columns:
        grouping_data[column] = (
            grouping_data[column].fillna("<MISSING>")
        )

    grouped = (
        grouping_data.groupby(group_columns, dropna=False)
        .agg(
            record_count=("WL(mbgl)", "size"),
            unique_levels=("WL(mbgl)", "nunique"),
            min_level=("WL(mbgl)", "min"),
            max_level=("WL(mbgl)", "max"),
            source_files=("source_file", "nunique")
        )
        .reset_index()
    )

    repeated = grouped[grouped["record_count"] > 1].copy()

    conflicting = repeated[
        repeated["unique_levels"] > 1
    ].copy()

    repeated.to_csv(
        OUTPUT_DIR / f"{label}_repeated_groups.csv",
        index=False
    )

    conflicting.to_csv(
        OUTPUT_DIR / f"{label}_conflicting_groups.csv",
        index=False
    )

    print(f"  Valid records: {len(valid):,}")
    print(f"  Repeated groups: {len(repeated):,}")
    print(f"  Conflicting groups: {len(conflicting):,}")

    return {
        "category": label,
        "valid_records": len(valid),
        "repeated_groups": len(repeated),
        "conflicting_groups": len(conflicting)
    }

# --------------------------------------------------
# COMPLETE IDENTIFIERS
# --------------------------------------------------

complete = df[
    df["identifier_status"] == "complete"
].copy()

# Coordinates and date are included because administrative
# labels alone do not uniquely identify an observation.
complete_group_columns = [
    "STATE_UT_NORM",
    "DISTRICT_NORM",
    "BLOCK_NORM",
    "VILLAGE_NORM",
    "LATITUDE",
    "LONGITUDE",
    "Date"
]

results = []

results.append(
    analyse_repeated_records(
        complete,
        complete_group_columns,
        "complete_identifiers"
    )
)

# --------------------------------------------------
# INCOMPLETE IDENTIFIERS
# --------------------------------------------------

incomplete = df[
    df["identifier_status"] != "complete"
].copy()

# Incomplete records are analysed separately.
# These groups are candidates for investigation, NOT
# confirmed duplicate wells or duplicate observations.

incomplete_group_columns = [
    "STATE_UT_NORM",
    "DISTRICT_NORM",
    "BLOCK_NORM",
    "VILLAGE_NORM",
    "LATITUDE",
    "LONGITUDE",
    "Date"
]

results.append(
    analyse_repeated_records(
        incomplete,
        incomplete_group_columns,
        "incomplete_identifiers"
    )
)

# --------------------------------------------------
# SAVE SUMMARY
# --------------------------------------------------

summary = pd.DataFrame(results)

summary.to_csv(
    OUTPUT_DIR / "repeated_observation_summary.csv",
    index=False
)

print("\nFINAL SUMMARY")
print(summary.to_string(index=False))

print("\nAnalysis complete.")
print(f"Outputs saved to: {OUTPUT_DIR}")
