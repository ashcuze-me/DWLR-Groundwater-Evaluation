from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster


DATA_FILE = Path("data/processed/January_2026_standardized.csv")
OUTPUT_FILE = Path("outputs/dwlr_map.html")

INDIA_CENTER = [22.5, 82.0]


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE.resolve()}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "STATE_UT",
        "DISTRICT",
        "VILLAGE",
        "LATITUDE",
        "LONGITUDE",
        "DATE",
        "DTWL",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for column in ["LATITUDE", "LONGITUDE", "DTWL"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=["LATITUDE", "LONGITUDE"]
    ).copy()

    df = df[
        df["LATITUDE"].between(-90, 90)
        & df["LONGITUDE"].between(-180, 180)
    ].copy()

    df["DTWL"] = df["DTWL"].astype("Float64")

    return df


def create_map(df):
    groundwater_map = folium.Map(
        location=INDIA_CENTER,
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    marker_cluster = MarkerCluster(
        name="DWLR Observation Locations"
    ).add_to(groundwater_map)

    for row in df.itertuples(index=False):
        dtwl = (
            "Unavailable"
            if pd.isna(row.DTWL)
            else f"{row.DTWL:.2f}"
        )

        village = (
            "Not available"
            if pd.isna(row.VILLAGE)
            else str(row.VILLAGE)
        )

        district = (
            "Not available"
            if pd.isna(row.DISTRICT)
            else str(row.DISTRICT)
        )

        state = (
            "Not available"
            if pd.isna(row.STATE_UT)
            else str(row.STATE_UT)
        )

        popup_html = f"""
        <div style="width: 250px; font-family: Arial;">
            <h4>DWLR Observation</h4>
            <b>State/UT:</b> {state}<br>
            <b>District:</b> {district}<br>
            <b>Village:</b> {village}<br>
            <b>Date:</b> {row.DATE}<br>
            <b>DTWL:</b> {dtwl}<br>
            <b>Latitude:</b> {row.LATITUDE}<br>
            <b>Longitude:</b> {row.LONGITUDE}
        </div>
        """

        folium.CircleMarker(
            location=[
                row.LATITUDE,
                row.LONGITUDE,
            ],
            radius=4,
            color="#1769aa",
            weight=1,
            fill=True,
            fill_color="#2196f3",
            fill_opacity=0.7,
            popup=folium.Popup(
                popup_html,
                max_width=300,
            ),
            tooltip=f"{district}, {state} | DTWL: {dtwl}",
        ).add_to(marker_cluster)

    folium.LayerControl().add_to(groundwater_map)

    return groundwater_map


def main():
    print("=" * 60)
    print("DWLR INTERACTIVE MAP")
    print("=" * 60)

    df = load_data()

    print(f"Records loaded: {len(df)}")
    print(
        "Records with DTWL:",
        int(df["DTWL"].notna().sum()),
    )
    print(
        "Unique coordinate pairs:",
        df[["LATITUDE", "LONGITUDE"]]
        .drop_duplicates()
        .shape[0],
    )

    groundwater_map = create_map(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    groundwater_map.save(str(OUTPUT_FILE))

    print(f"\nMap saved to: {OUTPUT_FILE.resolve()}")
    print(f"Markers created: {len(df)}")
    print("Map generation successful.")
    print("=" * 60)


if __name__ == "__main__":
    main()