from pathlib import Path

import pandas as pd
import plotly.express as px


DATA_FILE = Path("data/processed/January_2026_standardized.csv")
OUTPUT_FILE = Path("outputs/dwlr_map_plotly.html")


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
            f"Missing columns: {missing_columns}"
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
        df["LATITUDE"].between(6, 38)
        & df["LONGITUDE"].between(68, 98)
    ].copy()

    return df


def create_map(df):
    fig = px.scatter_geo(
        df,
        lat="LATITUDE",
        lon="LONGITUDE",
        color="DTWL",
        color_continuous_scale="Viridis",
        hover_name="DISTRICT",
        hover_data={
            "STATE_UT": True,
            "VILLAGE": True,
            "DATE": True,
            "DTWL": ":.2f",
            "LATITUDE": ":.4f",
            "LONGITUDE": ":.4f",
        },
        opacity=0.75,
        title="DWLR Observation Locations — India",
        labels={
            "DTWL": "Depth to Water Level",
            "STATE_UT": "State / UT",
            "VILLAGE": "Village",
        },
    )

    fig.update_geos(
        scope="asia",
        projection_type="mercator",
        lonaxis_range=[68, 98],
        lataxis_range=[6, 38],
        showland=True,
        landcolor="#F2F2F2",
        showocean=True,
        oceancolor="#EAF4FA",
        showcountries=True,
        countrycolor="#888888",
        showcoastlines=True,
        coastlinecolor="#666666",
        showlakes=True,
        lakecolor="#EAF4FA",
        showframe=False,
        bgcolor="white",
    )

    fig.update_layout(
        title_x=0.5,
        height=750,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
        coloraxis_colorbar={
            "title": "DTWL",
        },
    )

    return fig


def main():
    print("=" * 60)
    print("DWLR PLOTLY MAP")
    print("=" * 60)

    df = load_data()

    print("Observations loaded:", len(df))
    print(
        "Unique coordinates:",
        df[["LATITUDE", "LONGITUDE"]]
        .drop_duplicates()
        .shape[0],
    )

    fig = create_map(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.write_html(
        str(OUTPUT_FILE.resolve()),
        include_plotlyjs=True,
    )

    print("\nMap saved to:", OUTPUT_FILE.resolve())
    print("Map generation successful!")
    print("=" * 60)


if __name__ == "__main__":
    main()