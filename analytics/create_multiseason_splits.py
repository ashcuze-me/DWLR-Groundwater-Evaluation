
from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/processed/team_dataset/multiseason/"
    "multiseason_prediction_dataset.csv"
)

OUTPUT = INPUT

print("Loading prediction dataset...")
df = pd.read_csv(INPUT)

# Chronological target-year allocation.
train_years = list(range(1997, 2012)) + [2014, 2015]
validation_years = [2018, 2019]
test_years = [2023]

df["SPLIT"] = "excluded"

df.loc[
    df["TARGET_YEAR"].isin(train_years),
    "SPLIT"
] = "train"

df.loc[
    df["TARGET_YEAR"].isin(validation_years),
    "SPLIT"
] = "validation"

df.loc[
    df["TARGET_YEAR"].isin(test_years),
    "SPLIT"
] = "test"

# Keep only samples assigned to an evaluation split.
model_data = df[df["SPLIT"] != "excluded"].copy()

print("\nSPLIT SUMMARY")

for split in ["train", "validation", "test"]:
    subset = model_data[model_data["SPLIT"] == split]

    print(f"\n{split.upper()}")
    print(f"Samples: {len(subset):,}")
    print(
        "Unique locations:",
        subset["LOCATION_KEY"].nunique()
    )

    if not subset.empty:
        print(
            "Target years:",
            sorted(subset["TARGET_YEAR"].unique())
        )

print("\nEXCLUDED SAMPLES")
print(
    f"{(df['SPLIT'] == 'excluded').sum():,}"
)

print("\nSAMPLES BY TARGET YEAR")
print(
    model_data.groupby(
        ["SPLIT", "TARGET_YEAR"]
    ).size().to_string()
)

model_data.to_csv(
    INPUT.parent / "multiseason_model_splits.csv",
    index=False
)

print("\nSaved:")
print(
    (INPUT.parent / "multiseason_model_splits.csv").resolve()
)
