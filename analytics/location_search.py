from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DATA_FILE = Path("data/processed/January_2026_standardized.csv")
OUTPUT_FILE = Path("outputs/dwlr_location_search.html")


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

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

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

    for column in ["STATE_UT", "DISTRICT", "VILLAGE"]:
        df[column] = df[column].fillna("Not available").astype(str)

    return df


def create_map(df):
    fig = px.scatter_geo(
        df,
        lat="LATITUDE",
        lon="LONGITUDE",
        hover_name="DISTRICT",
        hover_data=[
            "STATE_UT",
            "VILLAGE",
            "DATE",
            "DTWL",
        ],
        color="DTWL",
        color_continuous_scale="Viridis",
        title="DWLR Groundwater Observation Explorer",
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
        showframe=False,
    )

    fig.update_layout(
        title_x=0.5,
        height=700,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return fig


def create_search_page(df):
    states = sorted(df["STATE_UT"].unique())

    fig = create_map(df)

    state_buttons = []

    for state in states:
        state_df = df[df["STATE_UT"] == state]

        state_buttons.append(
            dict(
                label=state,
                method="update",
                args=[
                    {
                        "visible": [
                            state == selected_state
                            for selected_state in states
                        ]
                    },
                    {
                        "title": (
                            "DWLR Observations — "
                            + state
                        )
                    },
                ],
            )
        )

    return fig


def main():
    print("=" * 60)
    print("DWLR LOCATION SEARCH PROTOTYPE")
    print("=" * 60)

    df = load_data()

    print("Observations loaded:", len(df))
    print("States/UTs:", df["STATE_UT"].nunique())
    print("Districts:", df["DISTRICT"].nunique())
    print("Villages:", df["VILLAGE"].nunique())

    fig = create_search_page(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.write_html(
        str(OUTPUT_FILE.resolve()),
        include_plotlyjs=True,
    )

    print("\nMap saved to:", OUTPUT_FILE.resolve())
    print("Location explorer generated successfully!")


if __name__ == "__main__":
    main()