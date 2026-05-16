"""
visualization.py

Embedding visualization utilities for multilingual PD speech experiments.

The goal is to inspect whether speech embeddings separate:
1. PD vs HC disease labels
2. Spanish vs German language labels
3. Combined groups such as Spanish-PD, Spanish-HC, German-PD, German-HC

Author: Prosenjit Chowdhury
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


FEATURE_PREFIX = "feature_"


def run_tsne(
    features_df: pd.DataFrame,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run t-SNE on feature columns and return a plotting DataFrame.
    """
    feature_cols = [col for col in features_df.columns if col.startswith(FEATURE_PREFIX)]
    if not feature_cols:
        raise ValueError("No feature columns found for t-SNE.")

    X = features_df[feature_cols].to_numpy()
    X = StandardScaler().fit_transform(X)

    n_samples = len(features_df)
    safe_perplexity = min(perplexity, max(2, (n_samples - 1) / 3))

    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )

    coords = tsne.fit_transform(X)

    plot_df = features_df[["speaker_id", "language", "task", "label"]].copy()
    plot_df["tsne_1"] = coords[:, 0]
    plot_df["tsne_2"] = coords[:, 1]
    plot_df["group"] = plot_df["language"].astype(str) + " - " + plot_df["label"].astype(str)

    return plot_df


def save_tsne_plot(
    plot_df: pd.DataFrame,
    color_by: str,
    output_path: str | Path,
    title: str,
) -> None:
    """
    Save a t-SNE scatter plot.

    color_by can be:
    - label
    - language
    - group
    """
    if color_by not in plot_df.columns:
        raise ValueError(f"Column not found in plotting DataFrame: {color_by}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for name, group_df in plot_df.groupby(color_by):
        plt.scatter(
            group_df["tsne_1"],
            group_df["tsne_2"],
            label=str(name),
            alpha=0.75,
            s=35,
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
