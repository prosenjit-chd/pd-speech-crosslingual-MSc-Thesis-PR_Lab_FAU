import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

def plot_embeddings(df, feature_cols, output_dir, prefix):
    """
    Generates t-SNE and PCA plots colored by label, language, and group.
    """
    if len(df) < 5:
        logger.warning(f"Not enough samples to run t-SNE for {prefix}")
        return

    features = df[feature_cols].values
    
    # Run PCA
    try:
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(features)
        df['pca-one'] = pca_result[:, 0]
        df['pca-two'] = pca_result[:, 1]
    except Exception as e:
        logger.error(f"PCA failed: {e}")
        return

    # Run t-SNE
    try:
        perplexity = min(30, max(5, len(df) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_result = tsne.fit_transform(features)
        df['tsne-one'] = tsne_result[:, 0]
        df['tsne-two'] = tsne_result[:, 1]
    except Exception as e:
        logger.error(f"t-SNE failed: {e}")
        return

    # Create Group column: e.g., "Spanish PD"
    df['group'] = df['language'] + ' ' + df['label']

    plot_types = [
        ('label', 'by_label'),
        ('language', 'by_language'),
        ('group', 'by_group')
    ]

    for hue_col, suffix in plot_types:
        # Plot t-SNE
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="tsne-one", y="tsne-two",
            hue=hue_col,
            palette=sns.color_palette("hls", df[hue_col].nunique()),
            data=df,
            legend="full",
            alpha=0.8
        )
        plt.title(f't-SNE {prefix} ({suffix})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tsne_{prefix}_{suffix}.png'), dpi=300)
        plt.close()
        
        # Plot PCA
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue=hue_col,
            palette=sns.color_palette("hls", df[hue_col].nunique()),
            data=df,
            legend="full",
            alpha=0.8
        )
        plt.title(f'PCA {prefix} ({suffix})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pca_{prefix}_{suffix}.png'), dpi=300)
        plt.close()
