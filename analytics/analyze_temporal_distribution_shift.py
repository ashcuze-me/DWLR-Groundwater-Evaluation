
from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/processed/team_dataset/multiseason/"
    "multiseason_model_splits.csv"
)

OUTPUT = Path("outputs/multiseason_models/diagnostics")
OUTPUT.mkdir(parents=True, exist_ok=True)

SEASONS = ["jan", "may", "aug", "nov"]
SPLITS = ["train", "validation", "test"]

df = pd.read_csv(INPUT)

results = []

for season in SEASONS:

    target = f"target_{season}_level"

    for split in SPLITS:

        subset = df[df["SPLIT"] == split]
        values = subset[target].dropna()

        results.append({
            "season": season,
            "split": split,
            "samples": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "p10": values.quantile(0.10),
            "p25": values.quantile(0.25),
            "p75": values.quantile(0.75),
            "p90": values.quantile(0.90),
            "minimum": values.min(),
            "maximum": values.max(),
        })

summary = pd.DataFrame(results)

summary.to_csv(
    OUTPUT / "temporal_distribution_summary.csv",
    index=False,
)

for season in SEASONS:

    print(f"\n{'=' * 55}")
    print(f"{season.upper()} TARGET DISTRIBUTION")
    print("=" * 55)

    print(
        summary[summary["season"] == season]
        .drop(columns="season")
        .round(3)
        .to_string(index=False)
    )

print("\nDistribution analysis complete.")
print("Saved to:", OUTPUT.resolve())
