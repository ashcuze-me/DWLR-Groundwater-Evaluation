from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "January_2026_standardized.csv"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

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