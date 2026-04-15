"""Utilities for loading large bird sightings data from Google Drive.

This module supports two common workflows:
1) Local/dev container: download from Google Drive into `data/raw/`.
2) Google Colab: mount Drive and read directly from mounted path.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Iterable

import gdown
import pandas as pd


def mount_drive_if_colab(mount_point: str = "/content/drive") -> None:
    """Mount Google Drive when running in Colab; no-op otherwise."""
    try:
        from google.colab import drive  # type: ignore
    except ModuleNotFoundError:
        return

    drive.mount(mount_point)


def download_from_google_drive(
    destination: str | Path,
    *,
    drive_url: str | None = None,
    file_id: str | None = None,
    quiet: bool = False,
) -> Path:
    """Download a file from Google Drive to a local path.

    Args:
        destination: Local output path, e.g. data/raw/bird_sightings.csv
        drive_url: Full Google Drive share URL.
        file_id: Google Drive file ID.
        quiet: Reduce downloader output.

    Returns:
        Path to the downloaded file.
    """
    if not drive_url and not file_id:
        raise ValueError("Provide either `drive_url` or `file_id`.")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if drive_url:
        extracted_id = _extract_google_drive_file_id(drive_url)
        if extracted_id:
            file_id = extracted_id

    if not file_id and drive_url:
        # Fallback for older gdown versions that support URL download only.
        gdown.download(url=drive_url, output=str(destination), quiet=quiet)
    else:
        gdown.download(id=file_id, output=str(destination), quiet=quiet)

    if not destination.exists():
        raise FileNotFoundError(
            f"Download appears incomplete. File not found at: {destination}"
        )

    return destination


def _extract_google_drive_file_id(drive_url: str) -> str | None:
    """Extract a Google Drive file ID from common sharing URL formats."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            return match.group(1)
    return None


def iter_csv_chunks(
    file_path: str | Path,
    *,
    chunksize: int = 500_000,
    sep: str | None = None,
    usecols: list[str] | None = None,
    parse_dates: list[str] | None = None,
    dtype: dict[str, str] | None = None,
    on_bad_lines: str = "warn",
) -> Iterable[pd.DataFrame]:
    """Yield delimited data in chunks to keep memory usage controlled.

    If ``sep`` is omitted, the delimiter is inferred from the header line.
    """
    resolved_sep = sep or _infer_delimiter(file_path)
    return pd.read_csv(
        file_path,
        chunksize=chunksize,
        sep=resolved_sep,
        usecols=usecols,
        parse_dates=parse_dates,
        dtype=dtype,
        low_memory=False,
        on_bad_lines=on_bad_lines,
    )


def _infer_delimiter(file_path: str | Path) -> str:
    """Infer delimiter using csv.Sniffer with a simple tab/comma fallback."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
        sample = handle.readline()

    if not sample:
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        return dialect.delimiter
    except csv.Error:
        if "\t" in sample:
            return "\t"
        return ","


def read_parquet(
    file_path: str | Path,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a parquet file (optionally selecting columns)."""
    return pd.read_parquet(file_path, columns=columns)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a dataset from Google Drive for local analysis."
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="Local path for downloaded file, e.g. data/raw/bird_sightings.csv",
    )
    parser.add_argument("--drive-url", help="Google Drive share URL", default=None)
    parser.add_argument("--file-id", help="Google Drive file ID", default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce downloader output",
    )
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    downloaded_path = download_from_google_drive(
        destination=args.destination,
        drive_url=args.drive_url,
        file_id=args.file_id,
        quiet=args.quiet,
    )
    print(f"Downloaded file to: {downloaded_path}")


if __name__ == "__main__":
    main()