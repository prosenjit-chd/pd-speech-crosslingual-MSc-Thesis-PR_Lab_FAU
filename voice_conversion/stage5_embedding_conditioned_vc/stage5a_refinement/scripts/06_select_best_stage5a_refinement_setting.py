#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_select_best_stage5a_refinement_setting.py

Purpose:
Analyzes the refinement classification summary and audio validation results
to identify the optimal conditioning configuration (model, layer, and alpha).
Generates:
- logs_refinement/stage5a_refinement_best_setting_summary.csv
- logs_refinement/stage5a_refinement_best_setting.md
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd

# Resolve directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
REFINEMENT_DIR = STAGE5_DIR / "stage5a_refinement"
LOGS_DIR = REFINEMENT_DIR / "logs_refinement"

VALIDATION_CSV = LOGS_DIR / "stage5a_refinement_audio_validation.csv"
CLASSIFICATION_SUMMARY = LOGS_DIR / "stage5a_refinement_classification_summary.csv"
BEST_SUMMARY_CSV = LOGS_DIR / "stage5a_refinement_best_setting_summary.csv"
BEST_SETTING_MD = LOGS_DIR / "stage5a_refinement_best_setting.md"

def main():
    print("=" * 60)
    print("Stage 5A-Refinement Step 6: Parameter Optimization & Selection")
    print("=" * 60)

    if not VALIDATION_CSV.exists() or not CLASSIFICATION_SUMMARY.exists():
        print(f"ERROR: Logs not found. Make sure step 4 and step 5 are complete.")
        sys.exit(1)

    df_val = pd.read_csv(VALIDATION_CSV)
    df_class = pd.read_csv(CLASSIFICATION_SUMMARY)

    # 1. Group validation results to find success rates for each setting
    # Group keys: condition_model, condition_layer, alpha
    val_grouped = df_val.groupby(["condition_model", "condition_layer", "alpha"])
    val_rates = {}
    for keys, group_df in val_grouped:
        success_rate = sum(group_df["status"] == "success") / len(group_df)
        val_rates[keys] = success_rate

    # 2. Analyze classification results by setting
    class_grouped = df_class.groupby(["condition_model", "condition_layer", "alpha"])
    
    summary_records = []
    
    for (model, layer, alpha), group_df in class_grouped:
        uar_deltas = group_df["uar_delta"].values
        abs_deltas = np.abs(uar_deltas)
        
        pos_count = int(sum(uar_deltas > 0))
        neg_count = int(sum(uar_deltas < 0))
        
        val_rate = val_rates.get((model, layer, alpha), 0.0)
        
        summary_records.append({
            "condition_model": model,
            "condition_layer": int(layer),
            "alpha": float(alpha),
            "mean_crosslingual_uar_delta": float(np.mean(uar_deltas)),
            "median_crosslingual_uar_delta": float(np.median(uar_deltas)),
            "mean_abs_uar_delta": float(np.mean(abs_deltas)),
            "number_of_positive_crosslingual_rows": pos_count,
            "number_of_negative_crosslingual_rows": neg_count,
            "audio_success_rate": val_rate
        })
        
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(BEST_SUMMARY_CSV, index=False)
    print(f"Saved best setting summary CSV: {BEST_SUMMARY_CSV}")

    # 3. Find the best setting based on scientific criteria:
    # Criteria:
    # 1. High audio success rate (must be 1.0)
    # 2. Maximum mean crosslingual UAR delta (we want positive or least negative delta)
    # 3. Stability: high positive count and low negative count
    # 4. Prefer moderate alpha (like 0.25 or 0.5) if performance is close
    
    df_eligible = df_summary[df_summary["audio_success_rate"] >= 0.99]
    if df_eligible.empty:
        print("WARNING: No configurations passed technical validation. Selection aborted.")
        sys.exit(1)
        
    # Sort primarily by mean UAR delta descending, then median UAR delta descending, then lower absolute delta
    df_sorted = df_eligible.sort_values(
        by=["mean_crosslingual_uar_delta", "median_crosslingual_uar_delta", "mean_abs_uar_delta"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    
    best_row = df_sorted.iloc[0]
    best_model = best_row["condition_model"]
    best_layer = int(best_row["condition_layer"])
    best_alpha = float(best_row["alpha"])
    
    print(f"\nOptimal Parameter Selection: {best_model.upper()} Layer {best_layer} | Alpha={best_alpha}")

    # 4. Generate recommendation MD
    rec_lines = [
        "# Parameter Optimization & Best Setting Selection",
        "",
        "Based on the grid search classification stability check and audio quality check, we have identified the optimal parameter set.",
        "",
        "## Selected Optimal Configuration",
        f"- **Conditioning Model**: {best_model.upper()}",
        f"- **Target Layer**: {best_layer}",
        f"- **Conditioning Scale (Alpha)**: {best_alpha}",
        f"- **Averages**: UAR Delta: `{best_row['mean_crosslingual_uar_delta']:+.4f}` | Abs UAR Delta: `{best_row['mean_abs_uar_delta']:.4f}`",
        f"- **Stability**: Positive Delta Runs: {best_row['number_of_positive_crosslingual_rows']} | Negative Delta Runs: {best_row['number_of_negative_crosslingual_rows']}",
        f"- **Technical Validation Success Rate**: {best_row['audio_success_rate']*100:.1f}%",
        "",
        "## Candidate Selection Performance Table",
        "",
        "| Model | Layer | Alpha | Mean UAR Delta | Median UAR Delta | Mean Abs Delta | Pos Rows | Neg Rows | Audio Success |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    ]
    
    for _, r in df_sorted.iterrows():
        rec_lines.append(
            f"| {r['condition_model'].upper()} | {int(r['condition_layer'])} | {r['alpha']} | "
            f"{r['mean_crosslingual_uar_delta']:+.4f} | {r['median_crosslingual_uar_delta']:+.4f} | {r['mean_abs_uar_delta']:.4f} | "
            f"{r['number_of_positive_crosslingual_rows']} | {r['number_of_negative_crosslingual_rows']} | {r['audio_success_rate']*100:.1f}% |"
        )
        
    rec_lines.append("")
    rec_lines.append("## Recommendation for Stage 5B")
    
    # Simple logic to determine recommendation status:
    # If the best mean UAR delta is reasonable (e.g. > -0.05 or better than others), we proceed.
    if best_row["mean_crosslingual_uar_delta"] >= -0.05:
        rec_lines.append(f"**Decision**: `Proceed to Stage 5B 80-file subset with selected setting`")
        rec_lines.append(f"**Rationale**: The configuration `{best_model}_layer{best_layer}` with `alpha={best_alpha}` shows the best preservation of downstream crosslingual PD/HC diagnostic features while yielding 100% technically validated synthetic speech.")
    else:
        rec_lines.append(f"**Decision**: `Refine method further before 80 files`")
        rec_lines.append(f"**Rationale**: All settings showed significant crosslingual UAR degradation (mean UAR delta < -0.05), suggesting representation drift is too high.")
        
    with open(BEST_SETTING_MD, mode="w", encoding="utf-8") as f:
        f.write("\n".join(rec_lines))
        
    print(f"Best setting report written to: {BEST_SETTING_MD}")
    print("Step 6 Finished Successfully!\n")

if __name__ == "__main__":
    main()
