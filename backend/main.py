from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import joblib
from math import isfinite

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "January_2026_standardized.csv"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "multiseason_models"
    / "models"
)

MODEL_FILES = {
    "jan": MODEL_DIR / "jan_RandomForest.joblib",
    "may": MODEL_DIR / "may_RandomForest.joblib",
    "nov": MODEL_DIR / "nov_RandomForest.joblib",
}

app = Flask(__name__)
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Run the data cleaning and standardization scripts first."
        )

    data = pd.read_csv(DATA_FILE)

    for column in ["LATITUDE", "LONGITUDE", "DTWL"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    data["DATE"] = data["DATE"].astype("string")

    return data


df = load_data()
def load_prediction_models():
    models = {}

    for season, model_path in MODEL_FILES.items():
        if not model_path.exists():
            print(
                f"WARNING: Prediction model not found: {model_path}"
            )
            continue

        models[season] = joblib.load(model_path)

    return models


prediction_models = load_prediction_models()

def clean_value(value):
    if pd.isna(value):
        return None
    return str(value).strip()


def observation_to_dict(row):
    return {
        "state": clean_value(row.get("STATE_UT")),
        "district": clean_value(row.get("DISTRICT")),
        "village": clean_value(row.get("VILLAGE")),
        "latitude": float(row["LATITUDE"]),
        "longitude": float(row["LONGITUDE"]),
        "date": clean_value(row.get("DATE")),
        "dtwl": (
            float(row["DTWL"])
            if pd.notna(row["DTWL"])
            else None
        ),
    }


def calculate_distance(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0

    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))

    return 2 * earth_radius_km * atan2(
        sqrt(a), sqrt(1 - a)
    )


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "records_loaded": len(df),
    })


@app.get("/api/locations/search")
def search_locations():
    query = request.args.get("q", "").strip()
    limit = request.args.get("limit", default=15, type=int)

    if not query:
        return jsonify({"results": []})

    limit = max(1, min(limit, 50))

    columns = ["STATE_UT", "DISTRICT", "VILLAGE"]

    mask = pd.Series(False, index=df.index)

    for column in columns:
        mask |= df[column].astype("string").str.contains(
            query, case=False, na=False, regex=False
        )

    results = (
        df.loc[mask]
        .drop_duplicates(subset=["LATITUDE", "LONGITUDE"])
        .head(limit)
    )

    return jsonify({
        "results": [
            observation_to_dict(row)
            for _, row in results.iterrows()
        ]
    })


@app.get("/api/locations/nearest")
def nearest_location():
    latitude = request.args.get("latitude", type=float)
    longitude = request.args.get("longitude", type=float)
    radius_km = request.args.get("radius_km", default=50.0, type=float)

    if latitude is None or longitude is None:
        return jsonify({
            "error": "latitude and longitude are required"
        }), 400

    if not (-90 <= latitude <= 90):
        return jsonify({"error": "Invalid latitude"}), 400

    if not (-180 <= longitude <= 180):
        return jsonify({"error": "Invalid longitude"}), 400

    if not (0 < radius_km <= 500):
        return jsonify({
            "error": "radius_km must be greater than 0 and at most 500"
        }), 400

    distances = df.apply(
        lambda row: calculate_distance(
            latitude,
            longitude,
            row["LATITUDE"],
            row["LONGITUDE"],
        ),
        axis=1,
    )

    nearest_index = distances.idxmin()
    nearest_distance = float(distances.loc[nearest_index])

    if nearest_distance > radius_km:
        return jsonify({
            "found": False,
            "message": "No monitoring location found within the selected radius.",
        })

    result = observation_to_dict(df.loc[nearest_index])
    result["distance_km"] = round(nearest_distance, 2)

    return jsonify({
        "found": True,
        "result": result,
    })


@app.get("/api/locations/observations")
def location_observations():
    latitude = request.args.get("latitude", type=float)
    longitude = request.args.get("longitude", type=float)

    if latitude is None or longitude is None:
        return jsonify({
            "error": "latitude and longitude are required"
        }), 400

    matches = df[
        (df["LATITUDE"] - latitude).abs() < 0.000001
    ]

    matches = matches[
        (matches["LONGITUDE"] - longitude).abs() < 0.000001
    ]

    return jsonify({
        "observations": [
            observation_to_dict(row)
            for _, row in matches.iterrows()
        ]
    })

SEASONS = ["jan", "may", "aug", "nov"]

PREDICTION_FEATURES = (
    [f"{season}_level" for season in SEASONS]
    + [f"{season}_count" for season in SEASONS]
    + [f"{season}_conflict" for season in SEASONS]
)


@app.post("/api/predictions/seasonal")
def predict_seasonal_groundwater():

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must contain valid JSON."
        }), 400

    # Validate the required historical inputs.
    input_data = {}

    for season in SEASONS:

        level_key = f"{season}_level"
        count_key = f"{season}_count"
        conflict_key = f"{season}_conflict"

        for key in [level_key, count_key, conflict_key]:
            if key not in payload:
                return jsonify({
                    "error": f"Missing required field: {key}"
                }), 400

        try:
            level = float(payload[level_key])
            count = int(payload[count_key])

            conflict_value = payload[conflict_key]

            if isinstance(conflict_value, bool):
                conflict = int(conflict_value)
            elif conflict_value in (0, 1):
                conflict = int(conflict_value)
            else:
                return jsonify({
                    "error": (
                        f"{conflict_key} must be "
                        "true, false, 0 or 1."
                    )
                }), 400

        except (TypeError, ValueError, OverflowError):
            return jsonify({
                "error": (
                    f"Invalid numeric input for {season}."
                )
            }), 400

        if not isfinite(level):
            return jsonify({
                "error": f"{level_key} must be finite."
            }), 400

        if count < 1:
            return jsonify({
                "error": f"{count_key} must be at least 1."
            }), 400

        if count > 100000:
            return jsonify({
                "error": f"{count_key} is outside the accepted range."
            }), 400

        input_data[level_key] = level
        input_data[count_key] = count
        input_data[conflict_key] = conflict

    # Preserve the exact feature order used during training.
    features = pd.DataFrame(
        [[input_data[col] for col in PREDICTION_FEATURES]],
        columns=PREDICTION_FEATURES,
    )

    predictions = {}

    for season in SEASONS:

        if season == "aug":
            # Validation results favoured persistence for August.
            predicted_level = input_data["aug_level"]
            model_name = "Persistence"

        else:
            model = prediction_models.get(season)

            if model is None:
                return jsonify({
                    "error": (
                        f"The {season.upper()} prediction model "
                        "is unavailable on the server."
                    )
                }), 503

            predicted_level = float(
                model.predict(features)[0]
            )

            model_name = "RandomForest"

        if not isfinite(predicted_level):
            return jsonify({
                "error": (
                    f"Invalid prediction generated for {season}."
                )
            }), 500

        predictions[season] = {
            "predicted_level_mbgl": round(
                predicted_level, 3
            ),
            "model": model_name,
            "unit": "mbgl",
        }

    return jsonify({
        "status": "success",
        "forecast_type": "next_year_seasonal",
        "predictions": predictions,
        "metadata": {
            "input_seasons": SEASONS,
            "unit": "metres below ground level",
            "model_scope": "known-location forecasting",
            "warning": (
                "Experimental forecasts based on historical "
                "seasonal data. Predictions are not verified "
                "groundwater measurements."
            ),
        },
    }), 200


@app.get("/")
def index():
    if (FRONTEND_DIR / "index.html").exists():
        return send_from_directory(FRONTEND_DIR, "index.html")

    return jsonify({
        "message": "Groundwater API is running.",
        "health": "/api/health",
    })


@app.get("/<path:filename>")
def frontend_files(filename):
    if (FRONTEND_DIR / filename).is_file():
        return send_from_directory(FRONTEND_DIR, filename)

    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)