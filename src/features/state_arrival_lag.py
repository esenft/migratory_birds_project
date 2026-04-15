"""Build state-to-state migration lag features from large sightings data.

This script keeps all states and computes first-sighting dates by:
- species
- year
- stateProvince

Then it can compute lag days between two states for each species/year.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

try:
    from src.data_ingestion.google_drive_loader import iter_csv_chunks
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.data_ingestion.google_drive_loader import iter_csv_chunks


def build_first_sightings_table(
    raw_file_path: str | Path,
    output_path: str | Path,
    *,
    chunksize: int = 500_000,
    country_code: str | None = "US",
) -> pd.DataFrame:
    """Build first-sighting date per species/state/year from a large raw file."""
    keys = ["species", "stateProvince", "year"]
    running = pd.DataFrame(columns=[*keys, "first_seen_date"])

    for chunk in iter_csv_chunks(
        file_path=raw_file_path,
        chunksize=chunksize,
        usecols=["species", "stateProvince", "eventDate", "countryCode"],
        parse_dates=["eventDate"],
        on_bad_lines="skip",
    ):
        if country_code:
            chunk = chunk[chunk["countryCode"] == country_code]

        chunk = chunk.dropna(subset=["species", "stateProvince", "eventDate"])
        if chunk.empty:
            continue

        chunk["eventDate"] = pd.to_datetime(chunk["eventDate"], errors="coerce")
        chunk = chunk.dropna(subset=["eventDate"])
        if chunk.empty:
            continue

        chunk["year"] = chunk["eventDate"].dt.year.astype("int32")
        grouped = (
            chunk.groupby(keys, as_index=False)["eventDate"]
            .min()
            .rename(columns={"eventDate": "first_seen_date"})
        )

        running = pd.concat([running, grouped], ignore_index=True)
        running = running.groupby(keys, as_index=False)["first_seen_date"].min()

    running["first_seen_week"] = running["first_seen_date"].dt.isocalendar().week.astype(
        "int32"
    )
    running = running.sort_values(["species", "year", "stateProvince"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    running.to_parquet(output_path, index=False)
    return running


def compute_state_to_state_lag(
    first_sightings: pd.DataFrame,
    *,
    source_state: str,
    target_state: str,
) -> pd.DataFrame:
    """Compute lag in days from source state first sighting to target state first sighting."""
    src = first_sightings[first_sightings["stateProvince"] == source_state][
        ["species", "year", "first_seen_date", "first_seen_week"]
    ].rename(
        columns={
            "first_seen_date": "source_first_seen_date",
            "first_seen_week": "source_first_seen_week",
        }
    )

    tgt = first_sightings[first_sightings["stateProvince"] == target_state][
        ["species", "year", "first_seen_date", "first_seen_week"]
    ].rename(
        columns={
            "first_seen_date": "target_first_seen_date",
            "first_seen_week": "target_first_seen_week",
        }
    )

    merged = src.merge(tgt, on=["species", "year"], how="inner")
    merged["lag_days_source_to_target"] = (
        merged["target_first_seen_date"] - merged["source_first_seen_date"]
    ).dt.days.astype("int32")
    merged["lag_weeks_source_to_target"] = (
        merged["target_first_seen_week"] - merged["source_first_seen_week"]
    ).astype("int32")
    merged = merged.sort_values(["species", "year"])
    return merged


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build first-sighting tables and state-to-state lag features."
    )
    parser.add_argument(
        "--raw-file-path",
        default="data/raw/bird_sightings.csv",
        help="Raw sightings file path",
    )
    parser.add_argument(
        "--first-sightings-path",
        default="data/processed/first_sightings_by_species_state_year.parquet",
        help="Parquet output path for first sightings table",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Chunk size for raw ingestion",
    )
    parser.add_argument(
        "--country-code",
        default="US",
        help="Country code filter (set empty string to disable)",
    )
    parser.add_argument("--source-state", default=None, help="Source stateProvince")
    parser.add_argument("--target-state", default=None, help="Target stateProvince")
    parser.add_argument(
        "--lag-output-path",
        default="data/processed/state_to_state_lag.parquet",
        help="Parquet output path for lag feature table",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    country_code = args.country_code or None

    first_sightings = build_first_sightings_table(
        raw_file_path=args.raw_file_path,
        output_path=args.first_sightings_path,
        chunksize=args.chunksize,
        country_code=country_code,
    )
    print(
        "Built first sightings table with "
        f"{len(first_sightings):,} rows -> {args.first_sightings_path}"
    )

    if args.source_state and args.target_state:
        lag_df = compute_state_to_state_lag(
            first_sightings,
            source_state=args.source_state,
            target_state=args.target_state,
        )

        lag_output_path = Path(args.lag_output_path)
        lag_output_path.parent.mkdir(parents=True, exist_ok=True)
        lag_df.to_parquet(lag_output_path, index=False)

        print(
            "Built lag feature table with "
            f"{len(lag_df):,} rows -> {lag_output_path}"
        )
        if not lag_df.empty:
            print(
                "Lag summary (days): "
                f"median={lag_df['lag_days_source_to_target'].median():.1f}, "
                f"p10={lag_df['lag_days_source_to_target'].quantile(0.10):.1f}, "
                f"p90={lag_df['lag_days_source_to_target'].quantile(0.90):.1f}"
            )


if __name__ == "__main__":
    main()
