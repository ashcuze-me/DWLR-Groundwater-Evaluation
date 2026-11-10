
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

INPUT = Path(
    "data/processed/team_dataset/multiseason/"
    "multiseason_model_splits.csv"
)

OUTPUT = Path("outputs/multiseason_models")
MODEL_DIR = OUTPUT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = ["jan", "may", "aug", "nov"]

FEATURES = (
    [f"{s}_level" for s in SEASONS]
    + [f"{s}_count" for s in SEASONS]
    + [f"{s}_conflict" for s in SEASONS]
)

print("Loading modelling dataset...")
df = pd.read_csv(INPUT)

train = df[df["SPLIT"] == "train"].copy()
validation = df[df["SPLIT"] == "validation"].copy()
test = df[df["SPLIT"] == "test"].copy()

print(f"Training samples: {len(train):,}")
print(f"Validation samples: {len(validation):,}")
print(f"Test samples: {len(test):,}")

results = []

for season in SEASONS:

    target = f"target_{season}_level"

    print(f"\n{'=' * 55}")
    print(f"TRAINING {season.upper()} MODEL")
    print("=" * 55)

    X_train = train[FEATURES]
    y_train = train[target]

    X_val = validation[FEATURES]
    y_val = validation[target]

    X_test = test[FEATURES]
    y_test = test[target]

    # Persistence baseline:
    # predict next year's level using this year's same-season level.
    persistence_feature = f"{season}_level"

    for split_name, X, y in [
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        pred = X[persistence_feature]

        results.append({
            "season": season,
            "model": "Persistence",
            "split": split_name,
            "MAE": mean_absolute_error(y, pred),
            "RMSE": mean_squared_error(y, pred) ** 0.5,
            "R2": r2_score(y, pred),
        })

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        # Evaluate on validation and test.
        for split_name, X, y in [
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:

            predictions = model.predict(X)

            mae = mean_absolute_error(y, predictions)
            rmse = mean_squared_error(y, predictions) ** 0.5
            r2 = r2_score(y, predictions)

            results.append({
                "season": season,
                "model": model_name,
                "split": split_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            })

            print(
                f"{split_name.title()}: "
                f"MAE={mae:.4f}, "
                f"RMSE={rmse:.4f}, "
                f"R2={r2:.4f}"
            )

        # Save trained model.
        model_path = MODEL_DIR / f"{season}_{model_name}.joblib"
        joblib.dump(model, model_path)

        print(f"Saved: {model_path}")

# Save evaluation results.
results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "multiseason_model_metrics.csv",
    index=False,
)

print("\nFINAL MODEL COMPARISON")
print(
    results_df.round(4).to_string(index=False)
)

print("\nResults saved to:", OUTPUT.resolve())
print("Training complete.")
