"""
01_create_dataset_index.py

Create a reproducible dataset index for Spanish and German read-text speech.

This script is intentionally configurable through command-line arguments.
It does not include private dataset paths.

Example:
    python scripts/01_create_dataset_index.py \
        --spanish-readtext data/spanish/readtext_PDvsHC \
        --german-readtext data/german/Parkinson_German/Recordings/readtext \
        --output metadata/dataset_index_readtext.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset_index import build_dataset_index, save_index, scan_audio_files, summarize_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Spanish-German readtext dataset index.")
    parser.add_argument("--spanish-readtext", type=Path, required=True, help="Path to Spanish readtext folder.")
    parser.add_argument("--german-readtext", type=Path, required=True, help="Path to German readtext folder.")
    parser.add_argument("--output", type=Path, default=Path("metadata/dataset_index_readtext.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/tables/dataset_index_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = []
    records.extend(
        scan_audio_files(
            root_dir=args.spanish_readtext,
            language="Spanish",
            task="readtext",
            dataset="PCGITA_readtext",
        )
    )
    records.extend(
        scan_audio_files(
            root_dir=args.german_readtext,
            language="German",
            task="readtext",
            dataset="German_Sabine_Skoda",
        )
    )

    df = build_dataset_index(records)
    save_index(df, args.output)

    summary = summarize_index(df)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)

    print(f"Dataset index saved to: {args.output}")
    print(f"Dataset summary saved to: {args.summary_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
