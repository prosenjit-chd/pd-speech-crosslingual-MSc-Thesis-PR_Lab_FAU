"""
03_visualize_embeddings.py

Generate t-SNE plots for disease labels, language labels, and combined groups.

Example:
    python scripts/03_visualize_embeddings.py \
        --features features/xlsr/xlsr_readtext_layer4.csv \
        --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.visualization import run_tsne, save_tsne_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize speech embeddings with t-SNE.")
    parser.add_argument("--features", type=Path, required=True, help="Feature CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--name", type=str, default=None, help="Optional output name prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.features)
    prefix = args.name or args.features.stem

    plot_df = run_tsne(df)

    save_tsne_plot(
        plot_df,
        color_by="label",
        output_path=args.output_dir / f"{prefix}_tsne_by_label.png",
        title=f"t-SNE by PD/HC Label: {prefix}",
    )

    save_tsne_plot(
        plot_df,
        color_by="language",
        output_path=args.output_dir / f"{prefix}_tsne_by_language.png",
        title=f"t-SNE by Language: {prefix}",
    )

    save_tsne_plot(
        plot_df,
        color_by="group",
        output_path=args.output_dir / f"{prefix}_tsne_by_group.png",
        title=f"t-SNE by Language and Label: {prefix}",
    )

    print(f"t-SNE plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
