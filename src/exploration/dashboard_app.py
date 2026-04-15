"""Interactive dashboard for exploring large bird sightings datasets.

Run:
streamlit run src/exploration/dashboard_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.exploration.build_eda_assets import build_assets


DEFAULT_SAMPLE_PATH = Path("data/exploration/sample_preview.csv")
DEFAULT_SUMMARY_PATH = Path("data/exploration/summary.json")
DEFAULT_RAW_PATH = Path("data/raw/bird_sightings.csv")


@st.cache_data(show_spinner=False)
def load_sample(sample_path: str) -> pd.DataFrame:
    df = pd.read_csv(sample_path)
    if "eventDate" in df.columns:
        df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce")
    return df


def maybe_build_assets_from_raw(
    raw_path: str,
    output_dir: str,
    max_rows: int,
    sample_rows: int,
    chunksize: int,
) -> None:
    build_assets(
        file_path=raw_path,
        output_dir=output_dir,
        max_rows=max_rows,
        sample_rows=sample_rows,
        chunksize=chunksize,
    )


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in view", f"{len(df):,}")
    c2.metric("Unique species", f"{df['species'].nunique():,}" if "species" in df else "-")
    c3.metric("Unique states", f"{df['stateProvince'].nunique():,}" if "stateProvince" in df else "-")


def render_time_chart(df: pd.DataFrame) -> None:
    if "eventDate" not in df.columns:
        return
    valid = df.dropna(subset=["eventDate"]).copy()
    if valid.empty:
        return
    valid["year_month"] = valid["eventDate"].dt.to_period("M").astype(str)
    monthly = valid.groupby("year_month", as_index=False).size()
    fig = px.line(
        monthly,
        x="year_month",
        y="size",
        title="Sightings Over Time (Monthly)",
        labels={"size": "Sightings", "year_month": "Year-Month"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top_species(df: pd.DataFrame, top_n: int) -> None:
    if "species" not in df.columns:
        return
    counts = (
        df["species"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(top_n)
        .rename_axis("species")
        .reset_index(name="count")
    )
    if counts.empty:
        return
    fig = px.bar(
        counts,
        x="species",
        y="count",
        title=f"Top {top_n} Species (Current Filter)",
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)


def render_state_month_heatmap(df: pd.DataFrame) -> None:
    required = {"stateProvince", "eventDate"}
    if not required.issubset(df.columns):
        return
    tmp = df.dropna(subset=["stateProvince", "eventDate"]).copy()
    if tmp.empty:
        return
    tmp["month"] = tmp["eventDate"].dt.month
    pivot = (
        tmp.groupby(["stateProvince", "month"], as_index=False)
        .size()
        .pivot(index="stateProvince", columns="month", values="size")
        .fillna(0)
    )
    if pivot.empty:
        return
    fig = px.imshow(
        pivot,
        labels={"x": "Month", "y": "State", "color": "Sightings"},
        title="State vs Month Intensity",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_map(df: pd.DataFrame) -> None:
    required = {"decimalLatitude", "decimalLongitude", "species", "stateProvince"}
    if not required.issubset(df.columns):
        return
    points = df.dropna(subset=["decimalLatitude", "decimalLongitude"]).copy()
    if points.empty:
        return
    points = points.head(10_000)
    fig = px.scatter_map(
        points,
        lat="decimalLatitude",
        lon="decimalLongitude",
        color="stateProvince",
        hover_name="species",
        zoom=3,
        title="Sighting Locations (Sampled)",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Bird Sightings Explorer", layout="wide")
    st.title("Bird Sightings Explorer")
    st.caption("Interactive exploration with filters for species, state, and date range.")

    with st.sidebar:
        st.header("Data Source")
        data_mode = st.radio(
            "Choose input mode",
            options=["Use existing sample", "Build sample from raw file"],
            index=0,
        )

        sample_path = st.text_input("Sample CSV path", str(DEFAULT_SAMPLE_PATH))
        raw_path = st.text_input("Raw file path", str(DEFAULT_RAW_PATH))
        output_dir = st.text_input("EDA output directory", "data/exploration")
        max_rows = st.number_input("Max rows to process", min_value=50_000, value=500_000, step=50_000)
        sample_rows = st.number_input("Sample rows for dashboard", min_value=2_000, value=20_000, step=1_000)
        chunksize = st.number_input("Chunksize", min_value=50_000, value=200_000, step=50_000)

        if data_mode == "Build sample from raw file":
            if st.button("Build/refresh sample"):
                with st.spinner("Building EDA assets from raw file..."):
                    maybe_build_assets_from_raw(
                        raw_path=raw_path,
                        output_dir=output_dir,
                        max_rows=int(max_rows),
                        sample_rows=int(sample_rows),
                        chunksize=int(chunksize),
                    )
                st.success("EDA assets generated.")
                sample_path = str(Path(output_dir) / "sample_preview.csv")

        st.divider()
        top_n = st.slider("Top species count", min_value=5, max_value=30, value=15)

    sample_file = Path(sample_path)
    if not sample_file.exists():
        st.warning("Sample file not found. Build one from raw file in the sidebar or update the sample path.")
        st.stop()

    df = load_sample(str(sample_file))
    if df.empty:
        st.warning("Sample file exists but contains no rows.")
        st.stop()

    with st.sidebar:
        st.header("Filters")
        species_vals = sorted(df["species"].dropna().astype(str).unique().tolist()) if "species" in df else []
        state_vals = sorted(df["stateProvince"].dropna().astype(str).unique().tolist()) if "stateProvince" in df else []

        selected_species = st.multiselect("Species", options=species_vals, default=[])
        selected_states = st.multiselect("States", options=state_vals, default=[])

    filtered = df.copy()
    if selected_species:
        filtered = filtered[filtered["species"].astype(str).isin(selected_species)]
    if selected_states:
        filtered = filtered[filtered["stateProvince"].astype(str).isin(selected_states)]

    if "eventDate" in filtered.columns and filtered["eventDate"].notna().any():
        min_date = filtered["eventDate"].min().date()
        max_date = filtered["eventDate"].max().date()
        with st.sidebar:
            date_range = st.slider("Date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        filtered = filtered[
            filtered["eventDate"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
        ]

    if filtered.empty:
        st.warning("No rows match current filters.")
        st.stop()

    render_overview(filtered)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        render_time_chart(filtered)
    with c2:
        render_top_species(filtered, top_n=top_n)

    render_state_month_heatmap(filtered)
    render_map(filtered)

    st.subheader("Filtered Data Preview")
    st.dataframe(filtered.head(200), use_container_width=True)
    st.download_button(
        label="Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_bird_sightings.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()