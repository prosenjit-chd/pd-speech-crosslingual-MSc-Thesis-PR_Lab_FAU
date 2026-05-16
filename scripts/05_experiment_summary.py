"""
05_experiment_summary.py

Generate a compact experiment summary table from classification result files.

This script is useful for thesis notes and GitHub documentation.

Example:
    python scripts/05_experiment_summary.py \
        --results outputs/tables/cross_language_results_layer4.csv \
        --output outputs/reports/baseline_summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate markdown summary from result CSV.")
    parser.add_argument("--results", type=Path, required=True, help="Result CSV file.")
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/baseline_summary.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.results)

    display_cols = [
        "train_language",
        "test_language",
        "classifier",
        "uar",
        "accuracy",
        "sensitivity",
        "specificity",
        "auc",
        "n_train",
        "n_test",
    ]

    available = [col for col in display_cols if col in df.columns]
    summary_table = df[available].copy()

    for metric in ["uar", "accuracy", "sensitivity", "specificity", "auc"]:
        if metric in summary_table.columns:
            summary_table[metric] = summary_table[metric].round(4)

    markdown = "# Baseline Experiment Summary\n\n"
    markdown += "This report summarizes the current Spanish-German PD vs HC speech classification experiment.\n\n"
    markdown += summary_table.to_markdown(index=False)
    markdown += "\n\n## Notes\n\n"
    markdown += "- UAR / balanced accuracy is the primary metric.\n"
    markdown += "- Cross-language results measure how well a model trained on one language transfers to another.\n"
    markdown += "- Low cross-language performance may indicate language mismatch in the embedding space.\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    print(f"Experiment summary saved to: {args.output}")


if __name__ == "__main__":
    main()
