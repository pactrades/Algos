#!/usr/bin/env python3
"""Download parquet data files from Google Drive shared folder.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --output data/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gdown

# Google Drive shared folder URL
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1R6e4QSgIGlCC2Qhpr3bH9I6ox7oka_rA"


def download_data(output_dir: Path) -> None:
    """Download all files from the shared Google Drive folder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data files to: {output_dir}")
    print(f"Source: {GDRIVE_FOLDER_URL}")

    gdown.download_folder(
        url=GDRIVE_FOLDER_URL,
        output=str(output_dir),
        quiet=False,
    )

    # List downloaded files
    parquet_files = sorted(output_dir.glob("*.parquet"))
    if parquet_files:
        print(f"\nDownloaded {len(parquet_files)} parquet files:")
        for f in parquet_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")
    else:
        print("\nNo parquet files found. Check if the folder URL is correct.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download futures data from Google Drive")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()
    download_data(args.output)


if __name__ == "__main__":
    main()
