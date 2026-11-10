
from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

INPUT = Path(
    "data/processed/team_dataset/multiseason/"
    "multiseason_model_splits.csv"
)

MODEL_DIR = Path("outputs/multiseason_models/models")
OUTPUT = Path("outputs/multiseason_models/diagnostics")

OUTPUT.mkdir(parents=True, exist_ok=True)

SEASONS = ["jan", "may", "aug", "nov"]

FEATURES = (
    [f"{s}_level" for s in SEASONS]
    + [f"{s}_count" for s in SEASONS]
    + [f"{s}_conflict" for s in SEASONS]
)

df = pd.read_csv(INPUT)

importance_results = []
prediction_results = []

for season in SEASONS:

    model_path = MODEL_DIR / f"{season}_RandomForest.joblib"
    model = joblib.load(model_path)

    target = f"target_{season}_level"

    print(f"\n{'=' * 50}")
    print(f"{season.upper()} MODEL DIAGNOSTICS")
    print("=" * 50)

    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
        "season": season,
    })

    importance_results.append(importance)

    print("\nTop features:")
    print(
        importance.sort_values(
            "importance", ascending=False
        ).head(6).to_string(index=False)
    )

    # Prediction diagnostics
    for split in ["validation", "test"]:

        subset = df[df["SPLIT"] == split].copy()

        X = subset[FEATURES]
        y = subset[target]
        predictions = model.predict(X)

        errors = predictions - y

        mae = mean_absolute_error(y, predictions)
        rmse = mean_squared_error(
            y, predictions
        ) ** 0.5

        prediction_results.append({
            "season": season,
            "split": split,
            "samples": len(subset),
            "MAE": mae,
            "RMSE": rmse,
            "mean_error": errors.mean(),
            "median_absolute_error": errors.abs().median(),
            "p90_absolute_error": errors.abs().quantile(0.90),
        })

        print(f"\n{split.upper()} PREDICTION ERRORS")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Mean error: {errors.mean():.4f}")
        print(
            "90th percentile absolute error:",
            f"{errors.abs().quantile(0.90):.4f}"
        )

# Save results.
importance_df = pd.concat(
    importance_results,
    ignore_index=True,
)

predictions_df = pd.DataFrame(prediction_results)

importance_df.to_csv(
    OUTPUT / "feature_importance.csv",
    index=False,
)

predictions_df.to_csv(
    OUTPUT / "prediction_error_summary.csv",
    index=False,
)

print("\nDIAGNOSTICS COMPLETE")
print("Output directory:", OUTPUT.resolve())
