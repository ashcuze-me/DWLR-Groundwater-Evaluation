
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

INPUT = Path(
    "data/processed/team_dataset/multiseason/"
    "multiseason_model_splits.csv"
)

MODEL_DIR = Path("outputs/multiseason_models/models")
OUTPUT = Path("outputs/multiseason_models/error_analysis")

OUTPUT.mkdir(parents=True, exist_ok=True)

SEASONS = ["jan", "may", "aug", "nov"]

FEATURES = (
    [f"{s}_level" for s in SEASONS]
    + [f"{s}_count" for s in SEASONS]
    + [f"{s}_conflict" for s in SEASONS]
)

df = pd.read_csv(INPUT)

all_results = []

# Groundwater depth bands in mbgl.
bins = [-np.inf, 2, 5, 10, 20, 50, np.inf]

labels = [
    "Below 2",
    "2 to 5",
    "5 to 10",
    "10 to 20",
    "20 to 50",
    "Above 50",
]

for season in SEASONS:

    print(f"\n{'=' * 55}")
    print(f"{season.upper()} ERROR ANALYSIS")
    print("=" * 55)

    model = joblib.load(
        MODEL_DIR / f"{season}_RandomForest.joblib"
    )

    target = f"target_{season}_level"

    for split in ["validation", "test"]:

        subset = df[df["SPLIT"] == split].copy()

        X = subset[FEATURES]
        actual = subset[target]

        predicted = model.predict(X)

        analysis = pd.DataFrame({
            "actual": actual.values,
            "predicted": predicted,
        })

        analysis["error"] = (
            analysis["predicted"] - analysis["actual"]
        )

        analysis["absolute_error"] = (
            analysis["error"].abs()
        )

        analysis["actual_range"] = pd.cut(
            analysis["actual"],
            bins=bins,
            labels=labels,
            right=False,
        )

        analysis["season"] = season
        analysis["split"] = split

        all_results.append(analysis)

        print(f"\n{split.upper()}")

        print(
            f"Actual mean: {analysis['actual'].mean():.4f}"
        )

        print(
            f"Predicted mean: {analysis['predicted'].mean():.4f}"
        )

        print(
            f"Overall bias: {analysis['error'].mean():.4f}"
        )

        print("\nERROR BY ACTUAL GROUNDWATER RANGE")

        summary = (
            analysis.groupby(
                "actual_range",
                observed=False,
            )
            .agg(
                samples=("actual", "size"),
                actual_mean=("actual", "mean"),
                predicted_mean=("predicted", "mean"),
                MAE=("absolute_error", "mean"),
                mean_error=("error", "mean"),
            )
            .round(4)
        )

        print(summary.to_string())

        print(
            "\nOverall MAE:",
            f"{mean_absolute_error(actual, predicted):.4f}"
        )

# Save all prediction-level diagnostics.
all_predictions = pd.concat(
    all_results,
    ignore_index=True,
)

all_predictions.to_csv(
    OUTPUT / "prediction_level_errors.csv",
    index=False,
)

# Save aggregated error report.
range_summary = (
    all_predictions.groupby(
        ["season", "split", "actual_range"],
        observed=False,
    )
    .agg(
        samples=("actual", "size"),
        actual_mean=("actual", "mean"),
        predicted_mean=("predicted", "mean"),
        MAE=("absolute_error", "mean"),
        mean_error=("error", "mean"),
    )
    .reset_index()
)

range_summary.to_csv(
    OUTPUT / "error_by_groundwater_range.csv",
    index=False,
)

print("\nError analysis complete.")
print("Outputs saved to:", OUTPUT.resolve())
