"""
04_cross_language_classification.py

Run Spanish-German cross-language PD vs HC classification scenarios.

Scenarios:
1. Spanish -> German
2. German -> Spanish

Within-language nested CV can be added as the next milestone.

Example:
    python scripts/04_cross_language_classification.py \
        --features features/xlsr/xlsr_readtext_layer4.csv \
        --output outputs/tables/cross_language_results_layer4.csv
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.classification import train_and_test_language_transfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-language PD classification.")
    parser.add_argument("--features", type=Path, required=True, help="Feature CSV.")
    parser.add_argument("--classifier", type=str, default="svm", choices=["svm", "logreg"])
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/cross_language_results.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.features)
    results = []

    scenarios = [
        ("Spanish", "German"),
        ("German", "Spanish"),
    ]

    for train_lang, test_lang in scenarios:
        result = train_and_test_language_transfer(
            features_df=df,
            train_language=train_lang,
            test_language=test_lang,
            classifier_name=args.classifier,
        )
        results.append(asdict(result))

    result_df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)

    print(f"Cross-language results saved to: {args.output}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
