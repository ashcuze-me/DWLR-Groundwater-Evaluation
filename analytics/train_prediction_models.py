from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "prediction"
OUTPUT_DIR = ROOT / "outputs" / "model_training"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = 2019
VALIDATION_START = 2020
VALIDATION_END = 2022
TEST_START = 2023
TEST_END = 2025


FEATURES = {
    "january": [
        "year",
        "latitude",
        "longitude",
        "state",
        "district",
        "january_observation_count",
        "january_observation_gap_years",
        "january_level_mbgl",
        "previous_year_january_mbgl",
        "previous_year_august_mbgl",
    ],
    "august": [
        "year",
        "latitude",
        "longitude",
        "state",
        "district",
        "january_observation_count",
        "august_observation_count",
        "january_observation_gap_years",
        "august_observation_gap_years",
        "january_level_mbgl",
        "august_level_mbgl",
        "previous_year_january_mbgl",
        "previous_year_august_mbgl",
        "january_change_from_previous_year",
        "august_change_from_previous_year",
        "august_minus_january_mbgl",
    ],
}


def calculate_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
                sparse_output=True,
                ),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def build_models(numeric_features, categorical_features):
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    return {
        "LinearRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]),

        "RandomForest": Pipeline([
            ("preprocessor", build_preprocessor(
                numeric_features, categorical_features
            )),
            ("model", RandomForestRegressor(
                n_estimators=80,
                max_depth=12,
                min_samples_leaf=5,
                max_features=0.7,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def train_season(season, filename, target_column):
    print("\n" + "=" * 65)
    print(f"TRAINING {season.upper()} MODEL")
    print("=" * 65)

    path = DATA_DIR / filename
    df = pd.read_csv(path)

    features = FEATURES[season]

    # Keep only the selected features and target.
    df = df[features + ["target_year", target_column]].copy()

    df["target_year"] = pd.to_numeric(
        df["target_year"], errors="coerce"
    )

    df[target_column] = pd.to_numeric(
        df[target_column], errors="coerce"
    )

    df = df.dropna(subset=["target_year", target_column])

    # Convert numeric features explicitly.
    numeric_features = [
        col for col in features
        if col not in ["state", "district"]
    ]

    categorical_features = ["state", "district"]

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where no usable current-season level exists.
    persistence_column = (
        "january_level_mbgl"
        if season == "january"
        else "august_level_mbgl"
    )

    # Chronological split.
    train = df[df["target_year"] <= TRAIN_END].copy()

    validation = df[
        df["target_year"].between(
            VALIDATION_START, VALIDATION_END
        )
    ].copy()

    test = df[
        df["target_year"].between(TEST_START, TEST_END)
    ].copy()

    print(f"Total usable records: {len(df):,}")
    print(f"Training records: {len(train):,}")
    print(f"Validation records: {len(validation):,}")
    print(f"Testing records: {len(test):,}")

    if train.empty or validation.empty or test.empty:
        print("ERROR: One or more chronological splits are empty.")
        return []

    X_train = train[features]
    y_train = train[target_column]

    X_validation = validation[features]
    y_validation = validation[target_column]

    X_test = test[features]
    y_test = test[target_column]

    models = build_models(
        numeric_features,
        categorical_features,
    )

    validation_results = []

    # Persistence baseline.
    baseline_predictions = validation[
        persistence_column
    ].fillna(
        train[persistence_column].median()
    )

    baseline_metrics = calculate_metrics(
        y_validation,
        baseline_predictions,
    )

    validation_results.append({
        "season": season,
        "model": "Persistence",
        **baseline_metrics,
    })

    print("\nValidation results:")

    print(
        f"Persistence | MAE: {baseline_metrics['MAE']:.4f}"
        f" | RMSE: {baseline_metrics['RMSE']:.4f}"
        f" | R2: {baseline_metrics['R2']:.4f}"
    )

    # Train and validate candidate models.
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        validation_results.append({
            "season": season,
            "model": model_name,
            **metrics,
        })

        print(
            f"{model_name} | "
            f"MAE: {metrics['MAE']:.4f} | "
            f"RMSE: {metrics['RMSE']:.4f} | "
            f"R2: {metrics['R2']:.4f}"
        )

    # Select model using validation MAE.
    results_df = pd.DataFrame(validation_results)

    candidates = results_df[
        results_df["model"] != "Persistence"
    ]

    best_model_name = candidates.loc[
        candidates["MAE"].idxmin(), "model"
    ]

    print(f"\nSelected model: {best_model_name}")

    # Retrain selected model using training + validation data.
    final_train = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    final_model = build_models(
        numeric_features,
        categorical_features,
    )[best_model_name]

    final_model.fit(
        final_train[features],
        final_train[target_column],
    )

    # Evaluate once on the held-out test period.
    test_predictions = final_model.predict(X_test)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    # Test persistence baseline on the same test records.
    test_baseline = test[
        persistence_column
    ].fillna(
        final_train[persistence_column].median()
    )

    test_baseline_metrics = calculate_metrics(
        y_test,
        test_baseline,
    )

    improvement = (
        (test_baseline_metrics["MAE"] - test_metrics["MAE"])
        / test_baseline_metrics["MAE"] * 100
        if test_baseline_metrics["MAE"] != 0
        else np.nan
    )

    print("\nFINAL TEST RESULTS")
    print("-" * 45)

    print(f"Model: {best_model_name}")
    print(f"MAE:  {test_metrics['MAE']:.4f}")
    print(f"RMSE: {test_metrics['RMSE']:.4f}")
    print(f"R2:   {test_metrics['R2']:.4f}")

    print("\nPersistence baseline:")
    print(f"MAE:  {test_baseline_metrics['MAE']:.4f}")
    print(f"RMSE: {test_baseline_metrics['RMSE']:.4f}")
    print(f"R2:   {test_baseline_metrics['R2']:.4f}")

    print(f"\nMAE improvement over baseline: {improvement:.2f}%")

    # Save model.
    model_path = MODEL_DIR / f"{season}_best_model.joblib"
    joblib.dump(final_model, model_path)

    # Save predictions for inspection.
    prediction_output = test[
        ["target_year", "state", "district",
         "latitude", "longitude", target_column]
    ].copy()

    prediction_output["predicted_level_mbgl"] = test_predictions
    prediction_output["absolute_error"] = np.abs(
        prediction_output[target_column] - test_predictions
    )

    prediction_output.to_csv(
        OUTPUT_DIR / f"{season}_test_predictions.csv",
        index=False,
    )

    print(f"\nModel saved: {model_path}")

    return [
        {
            "season": season,
            "model": "Persistence",
            "evaluation": "Test",
            **test_baseline_metrics,
        },
        {
            "season": season,
            "model": best_model_name,
            "evaluation": "Test",
            **test_metrics,
            "MAE_improvement_percent": improvement,
        },
    ]


def main():
    all_results = []

    all_results.extend(train_season(
        "january",
        "january_model_dataset.csv",
        "target_january_mbgl",
    ))

    all_results.extend(train_season(
        "august",
        "august_model_dataset.csv",
        "target_august_mbgl",
    ))

    if all_results:
        results_df = pd.DataFrame(all_results)

        results_df.to_csv(
            OUTPUT_DIR / "model_evaluation_results.csv",
            index=False,
        )

        print("\n" + "=" * 65)
        print("MODEL TRAINING COMPLETE")
        print("=" * 65)

        print(results_df.to_string(index=False))

        print(
            f"\nResults saved to: "
            f"{OUTPUT_DIR / 'model_evaluation_results.csv'}"
        )


if __name__ == "__main__":
    main()