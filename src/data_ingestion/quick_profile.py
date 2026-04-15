"""Quick profiling for large bird sightings CSV files.

Example:
python src/data_ingestion/quick_profile.py \
  --file-path data/raw/bird_sightings.csv \
  --species-column species \
    --date-column eventDate
"""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from src.data_ingestion.google_drive_loader import iter_csv_chunks


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile large bird sightings CSV files.")
    parser.add_argument("--file-path", required=True, help="Path to CSV file")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Rows per chunk for memory-safe processing",
    )
    parser.add_argument("--species-column", default="species")
    parser.add_argument("--date-column", default="eventDate")
    return parser


def main() -> None:
    args = _build_cli().parse_args()

    rows_total = 0
    species_counts: Counter[str] = Counter()
    min_date = None
    max_date = None

    for chunk in iter_csv_chunks(
        file_path=args.file_path,
        chunksize=args.chunksize,
        usecols=[args.species_column, args.date_column],
        parse_dates=[args.date_column],
    ):
        rows_total += len(chunk)
        species_counts.update(chunk[args.species_column].dropna().astype(str).values)

        dates = chunk[args.date_column].dropna()
        if not dates.empty:
            chunk_min = pd.to_datetime(dates).min()
            chunk_max = pd.to_datetime(dates).max()

            min_date = chunk_min if min_date is None else min(min_date, chunk_min)
            max_date = chunk_max if max_date is None else max(max_date, chunk_max)

    print(f"Total rows: {rows_total:,}")
    print(f"Date range: {min_date} -> {max_date}")
    print("Top 10 species by record count:")
    for species, count in species_counts.most_common(10):
        print(f"  {species}: {count:,}")


if __name__ == "__main__":
    main()