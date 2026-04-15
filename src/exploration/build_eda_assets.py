"""Create lightweight exploration assets from a large sightings file.

Outputs:
- Summary JSON with row counts and top categories
- Sample CSV for quick inspection in notebooks/spreadsheets
- PNG plots for species/state/year/month distributions
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

try:
    from src.data_ingestion.google_drive_loader import iter_csv_chunks
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.data_ingestion.google_drive_loader import iter_csv_chunks


def _series_to_counter(series: pd.Series, *, max_items: int | None = None) -> Counter:
    values = series.dropna().astype(str)
    if max_items is not None:
        values = values.head(max_items)
    return Counter(values.tolist())


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_bar(counter: Counter, title: str, x_label: str, output_path: Path, top_n: int = 20) -> None:
    items = counter.most_common(top_n)
    if not items:
        return

    labels = [x for x, _ in items]
    values = [y for _, y in items]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_assets(
    file_path: str | Path,
    output_dir: str | Path,
    *,
    chunksize: int = 300_000,
    max_rows: int = 2_000_000,
    sample_rows: int = 20_000,
    random_seed: int = 42,
) -> dict:
    output_dir = _ensure_dir(output_dir)
    plots_dir = _ensure_dir(Path(output_dir) / "plots")

    total_rows = 0
    processed_rows = 0

    species_counter: Counter = Counter()
    state_counter: Counter = Counter()
    year_counter: Counter = Counter()
    month_counter: Counter = Counter()

    sample_frames: list[pd.DataFrame] = []
    sample_columns = [
        "species",
        "eventDate",
        "stateProvince",
        "decimalLatitude",
        "decimalLongitude",
        "individualCount",
        "locality",
    ]

    for chunk_index, chunk in enumerate(
        iter_csv_chunks(
            file_path=file_path,
            chunksize=chunksize,
            usecols=["species", "eventDate", "stateProvince", "decimalLatitude", "decimalLongitude", "individualCount", "locality"],
            parse_dates=["eventDate"],
            on_bad_lines="skip",
        )
    ):
        total_rows += len(chunk)
        if processed_rows >= max_rows:
            continue

        remaining = max_rows - processed_rows
        work_chunk = chunk.head(remaining)
        processed_rows += len(work_chunk)

        species_counter.update(_series_to_counter(work_chunk["species"]))
        state_counter.update(_series_to_counter(work_chunk["stateProvince"]))

        dates = pd.to_datetime(work_chunk["eventDate"], errors="coerce").dropna()
        if not dates.empty:
            year_counter.update(Counter(dates.dt.year.astype(str).tolist()))
            month_counter.update(Counter(dates.dt.month.astype(str).tolist()))

        if sample_rows > 0:
            per_chunk_target = max(1, int(sample_rows / max(1, max_rows // chunksize)))
            sampled = work_chunk.sample(
                n=min(per_chunk_target, len(work_chunk)),
                random_state=random_seed + chunk_index,
            )
            sample_frames.append(sampled[sample_columns])

        if processed_rows >= max_rows:
            break

    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame(columns=sample_columns)
    if len(sample_df) > sample_rows:
        sample_df = sample_df.sample(n=sample_rows, random_state=random_seed)

    sample_path = Path(output_dir) / "sample_preview.csv"
    sample_df.to_csv(sample_path, index=False)

    _plot_bar(species_counter, "Top Species (processed subset)", "Species", Path(plots_dir) / "top_species.png")
    _plot_bar(state_counter, "Top States (processed subset)", "State", Path(plots_dir) / "top_states.png")
    _plot_bar(year_counter, "Observations by Year (processed subset)", "Year", Path(plots_dir) / "by_year.png")
    _plot_bar(month_counter, "Observations by Month (processed subset)", "Month", Path(plots_dir) / "by_month.png")

    summary = {
        "file_path": str(file_path),
        "rows_seen_in_stream": total_rows,
        "rows_processed_for_eda": processed_rows,
        "sample_rows_written": int(len(sample_df)),
        "sample_preview_path": str(sample_path),
        "plots_dir": str(plots_dir),
        "top_species": species_counter.most_common(25),
        "top_states": state_counter.most_common(25),
        "top_years": year_counter.most_common(25),
        "top_months": month_counter.most_common(12),
    }

    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build exploration assets from large sightings data.")
    parser.add_argument("--file-path", default="data/raw/bird_sightings.csv", help="Input sightings file path")
    parser.add_argument("--output-dir", default="data/exploration", help="Output directory for EDA assets")
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--max-rows", type=int, default=2_000_000, help="Max rows to process for initial EDA")
    parser.add_argument("--sample-rows", type=int, default=20_000, help="Rows to keep in sample preview CSV")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    summary = build_assets(
        file_path=args.file_path,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        sample_rows=args.sample_rows,
        random_seed=args.random_seed,
    )
    print(f"EDA assets created in: {args.output_dir}")
    print(f"Rows processed: {summary['rows_processed_for_eda']:,}")
    print(f"Sample preview: {summary['sample_preview_path']}")


if __name__ == "__main__":
    main()
