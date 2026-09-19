#!/usr/bin/env python3
"""
08_generate_figures.py
Generates Publication-Quality Figures for Spanish -> German Final Refinement.

Implements Section 30:
- Figure 1: WER per speaker (Config C vs Final)
- Figure 2: CER per speaker (Config C vs Final)
- Figure 3: WIP per speaker (Config C vs Final)
- Figure 4: German LID confidence P(de) (Config C vs Final)
- Figure 5: Speaker similarity (Config C vs Final)
- Figure 6: Pathology feature-direction preservation summary
Export: Both PNG (300 DPI) and PDF for each figure.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement")
FIG_DIR = WORKSPACE / "16_figures"
TABLES_DIR = WORKSPACE / "17_tables"
PATHOLOGY_DIR = WORKSPACE / "15_pathology_sanity_check"

# Styling configuration
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": True
})

def save_fig(fig, name):
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf")

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    comp_csv = TABLES_DIR / "comparison_config_c_vs_final.csv"
    if not comp_csv.exists():
        print(f"Error: {comp_csv} not found! Run evaluation first.")
        sys.exit(1)
        
    df = pd.read_csv(comp_csv)
    speakers = df["safe_id"].tolist()
    labels = [f"{s}\n({d})" for s, d in zip(df["safe_id"], df["diagnosis"])]
    x = np.arange(len(speakers))
    width = 0.35
    
    color_c = "#4A90E2"      # Config C (Blue)
    color_final = "#50E3C2"  # Final Selected (Teal/Emerald)
    
    # -------------------------------------------------------------
    # Figure 1: WER per speaker (Config C vs Final)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, df["WER_before"], width, label="Config C Baseline", color=color_c, edgecolor="#1B365D", alpha=0.9)
    rects2 = ax.bar(x + width/2, df["WER_after"], width, label="Final Refined Method", color=color_final, edgecolor="#1D735C", alpha=0.95)
    
    ax.set_ylabel("Word Error Rate (WER) [Lower is Better]")
    ax.set_title("Figure 1: Spanish → German Word Error Rate (WER) per Speaker")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(1.05, float(df["WER_before"].max()) + 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    # Mean reference lines
    ax.axhline(df["WER_before"].mean(), color=color_c, linestyle=":", linewidth=1.5, label=f"Mean Baseline ({df['WER_before'].mean():.3f})")
    ax.axhline(df["WER_after"].mean(), color=color_final, linestyle="--", linewidth=1.8, label=f"Mean Final ({df['WER_after'].mean():.3f})")
    ax.legend(loc="upper right", frameon=True)
    
    save_fig(fig, "figure_1_wer_per_speaker")
    
    # -------------------------------------------------------------
    # Figure 2: CER per speaker (Config C vs Final)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, df["CER_before"], width, label="Config C Baseline", color=color_c, edgecolor="#1B365D", alpha=0.9)
    ax.bar(x + width/2, df["CER_after"], width, label="Final Refined Method", color=color_final, edgecolor="#1D735C", alpha=0.95)
    
    ax.set_ylabel("Character Error Rate (CER) [Lower is Better]")
    ax.set_title("Figure 2: Spanish → German Character Error Rate (CER) per Speaker")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(1.05, float(df["CER_before"].max()) + 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axhline(df["CER_before"].mean(), color=color_c, linestyle=":", linewidth=1.5, label=f"Mean Baseline ({df['CER_before'].mean():.3f})")
    ax.axhline(df["CER_after"].mean(), color=color_final, linestyle="--", linewidth=1.8, label=f"Mean Final ({df['CER_after'].mean():.3f})")
    ax.legend(loc="upper right", frameon=True)
    
    save_fig(fig, "figure_2_cer_per_speaker")
    
    # -------------------------------------------------------------
    # Figure 3: WIP per speaker (Config C vs Final)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, df["WIP_before"], width, label="Config C Baseline", color=color_c, edgecolor="#1B365D", alpha=0.9)
    ax.bar(x + width/2, df["WIP_after"], width, label="Final Refined Method", color=color_final, edgecolor="#1D735C", alpha=0.95)
    
    ax.set_ylabel("Word Information Preserved (WIP) [Higher is Better]")
    ax.set_title("Figure 3: Spanish → German Word Information Preserved (WIP) per Speaker")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axhline(df["WIP_before"].mean(), color=color_c, linestyle=":", linewidth=1.5, label=f"Mean Baseline ({df['WIP_before'].mean():.3f})")
    ax.axhline(df["WIP_after"].mean(), color=color_final, linestyle="--", linewidth=1.8, label=f"Mean Final ({df['WIP_after'].mean():.3f})")
    ax.legend(loc="upper right", frameon=True)
    
    save_fig(fig, "figure_3_wip_per_speaker")
    
    # -------------------------------------------------------------
    # Figure 4: German LID confidence P(de) (Config C vs Final)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, df["P_de_before"], width, label="Config C Baseline", color=color_c, edgecolor="#1B365D", alpha=0.9)
    ax.bar(x + width/2, df["P_de_after"], width, label="Final Refined Method", color=color_final, edgecolor="#1D735C", alpha=0.95)
    
    ax.set_ylabel("Target Language Confidence P(German)")
    ax.set_title("Figure 4: Target Language (German) LID Probability per Speaker")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.7, 1.02)
    ax.axhline(0.85, color="#D0021B", linestyle="--", linewidth=1.2, label="Hard Quality Gate (0.85)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    
    save_fig(fig, "figure_4_lid_confidence")
    
    # -------------------------------------------------------------
    # Figure 5: Speaker similarity (Config C vs Final)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, df["Similarity_before"], width, label="Config C Baseline", color=color_c, edgecolor="#1B365D", alpha=0.9)
    ax.bar(x + width/2, df["Similarity_after"], width, label="Final Refined Method", color=color_final, edgecolor="#1D735C", alpha=0.95)
    
    ax.set_ylabel("ECAPA Cosine Speaker Similarity")
    ax.set_title("Figure 5: Spanish Source → German Converted Speaker Identity Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.35, 0.85)
    ax.axhline(0.40, color="#D0021B", linestyle="--", linewidth=1.2, label="Hard Quality Gate (0.40)")
    ax.axhline(df["Similarity_after"].mean(), color=color_final, linestyle="--", linewidth=1.5, label=f"Mean Final ({df['Similarity_after'].mean():.3f})")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    save_fig(fig, "figure_5_speaker_similarity")
    
    # -------------------------------------------------------------
    # Figure 6: Pathology feature-direction preservation summary
    # -------------------------------------------------------------
    path_csv = PATHOLOGY_DIR / "pathology_direction_preservation.csv"
    if path_csv.exists():
        df_path = pd.read_csv(path_csv)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        feats = df_path["feature"].tolist()
        preserved = df_path["direction_preserved"].tolist()
        colors = ["#27AE60" if p else "#E74C3C" for p in preserved]
        
        y_pos = np.arange(len(feats))
        ax.barh(y_pos, [1]*len(feats), color=colors, edgecolor="#2C3E50", alpha=0.85, height=0.65)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "Preserved"])
        ax.set_title(f"Figure 6: Exploratory Acoustic Pathology Direction Preservation ({sum(preserved)}/{len(preserved)} Features Preserved)")
        ax.set_xlabel("Effect Direction Agreement (Source Spanish HC/PD vs Converted German HC/PD)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#27AE60", edgecolor="#2C3E50", label="Direction Preserved (Green)"),
            Patch(facecolor="#E74C3C", edgecolor="#2C3E50", label="Direction Inverted / Attenuated (Red)")
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True)
        
        save_fig(fig, "figure_6_pathology_direction_preservation")
        
    print("=== All 6 Figures Successfully Generated in PNG (300 DPI) and PDF! ===")

if __name__ == "__main__":
    main()
