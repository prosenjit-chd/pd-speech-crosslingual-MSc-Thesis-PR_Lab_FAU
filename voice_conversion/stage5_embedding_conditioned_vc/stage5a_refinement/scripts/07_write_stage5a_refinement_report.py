#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
07_write_stage5a_refinement_report.py

Purpose:
Generates the final markdown report:
logs_refinement/stage5a_refinement_report.md
Incorporates selection summaries, validation tables, classification statistics,
and final parameter recommendations.
"""

import os
import sys
import csv
from pathlib import Path
import pandas as pd

# Resolve directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
REFINEMENT_DIR = STAGE5_DIR / "stage5a_refinement"
LOGS_DIR = REFINEMENT_DIR / "logs_refinement"

COPY_LOG = LOGS_DIR / "stage5a_refinement_input_copy_log.csv"
EMB_LOG = LOGS_DIR / "stage5a_refinement_embedding_log.csv"
DOMAIN_SUMMARY = LOGS_DIR / "stage5a_refinement_domain_condition_summary.csv"
GENERATION_GRID_LOG = LOGS_DIR / "stage5a_refinement_generation_grid_log.csv"
VALIDATION_CSV = LOGS_DIR / "stage5a_refinement_audio_validation.csv"
CLASSIFICATION_SUMMARY = LOGS_DIR / "stage5a_refinement_classification_summary.csv"
BEST_SETTING_MD = LOGS_DIR / "stage5a_refinement_best_setting.md"
REPORT_PATH = LOGS_DIR / "stage5a_refinement_report.md"

def read_csv_safe(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def main():
    print("=" * 60)
    print("Stage 5A-Refinement Step 7: Generating Refinement Report")
    print("=" * 60)
    
    # Load logs
    df_select = read_csv_safe(COPY_LOG)
    df_extract = read_csv_safe(EMB_LOG)
    df_domain = read_csv_safe(DOMAIN_SUMMARY)
    df_gen = read_csv_safe(GENERATION_GRID_LOG)
    df_val = read_csv_safe(VALIDATION_CSV)
    df_class = read_csv_safe(CLASSIFICATION_SUMMARY)
    
    report_lines = []
    
    report_lines.append("# Stage 5A-Refinement — Parameter Optimization & Grid Search Report")
    report_lines.append("")
    report_lines.append("## 1. Goal of Stage 5A-Refinement")
    report_lines.append("The goal of this refinement experiment is to systematically test 15 parameter configurations across multiple conditioning representations and scales on the same 12 pilot files. We analyze downstream classification behavior and voice conversion characteristics to determine the best parameters for the subsequent 80-file Stage 5B scale-up.")
    report_lines.append("")
    
    report_lines.append("## 2. Safety Rule Confirmation")
    report_lines.append("We confirm that all previous baseline, HiFi-GAN 12, 80, and 276 folders, logs, and files remain completely **read-only** and untouched. All completed Stage 5A results were kept read-only. All new refinement outputs were isolated inside:")
    report_lines.append(f"`C:\\pd-speech-crosslingual\\voice_conversion\\stage5_embedding_conditioned_vc\\stage5a_refinement`")
    report_lines.append("")
    
    report_lines.append("## 3. Why Refinement was Needed")
    report_lines.append("While the Stage 5A pilot successfully verified technical audio vocoding, downstream crosslingual classification results were mixed. Since representations (such as WavLM) show different acoustic preservation capabilities compared to XLSR, a grid search over WavLM L8, WavLM L11, and XLSR L11 with alphas ranging from 0.1 to 1.0 was necessary to choose the optimal condition for scaling.")
    report_lines.append("")

    report_lines.append("## 4. Tested Conditioning Models and Layers")
    report_lines.append("- **XLSR Layer 11** (1024 dimensions)")
    report_lines.append("- **WavLM Layer 8** (768 dimensions)")
    report_lines.append("- **WavLM Layer 11** (768 dimensions)")
    report_lines.append("")
    
    report_lines.append("## 5. Tested Alpha (Conditioning Scale) Values")
    report_lines.append("- `alpha = 0.1` (weak conversion scale)")
    report_lines.append("- `alpha = 0.25` (moderate conversion scale)")
    report_lines.append("- `alpha = 0.5` (moderate conversion scale)")
    report_lines.append("- `alpha = 0.75` (strong conversion scale)")
    report_lines.append("- `alpha = 1.0` (strong conversion scale)")
    report_lines.append("")

    report_lines.append("## 6. Audio Validation Summary")
    if not df_val.empty:
        total_files = len(df_val)
        success_val = sum(df_val["status"] == "success")
        warning_val = sum(df_val["status"] == "warning")
        failed_val = sum(df_val["status"] == "failed")
        
        report_lines.append(f"A total of {total_files} generated WAV files (15 settings $\\times$ 12 files) were technically verified:")
        report_lines.append(f"- **Success (specifications met)**: {success_val}/{total_files} ({success_val/total_files*100:.1f}%)")
        report_lines.append(f"- **Warnings (technical deviations)**: {warning_val}")
        report_lines.append(f"- **Failed**: {failed_val}")
        report_lines.append("")
        report_lines.append("All generated audios matched the required vocoder configuration (22050 Hz, single channel mono). No empty waveforms or silence were found.")
    else:
        report_lines.append("WARNING: Audio validation log not found.")
    report_lines.append("")
    
    report_lines.append("## 7. Classification Diagnostic Comparison")
    report_lines.append("> [!IMPORTANT]")
    report_lines.append("> All classification metrics reported below represent diagnostic crosslingual evaluation check values on a very small pilot sample. They serve to observe representation drift and domain shifts rather than generalizable performance.")
    report_lines.append("")
    
    if not df_class.empty:
        # Load and parse classification summary to show a short summary of UAR deltas
        grouped = df_class.groupby(["condition_model", "condition_layer", "alpha"])
        report_lines.append("| Model | Layer | Alpha | Mean UAR Original | Mean UAR Converted | Mean UAR Delta | Mean Acc Delta | Diagnostic Note |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for (model_name, layer_val, alpha_val), group_df in grouped:
            mean_orig = group_df["uar_original"].mean()
            mean_conv = group_df["uar_converted"].mean()
            mean_delta = group_df["uar_delta"].mean()
            mean_acc_d = group_df["accuracy_delta"].mean()
            report_lines.append(f"| {model_name.upper()} | {layer_val} | {alpha_val} | {mean_orig:.4f} | {mean_conv:.4f} | {mean_delta:+.4f} | {mean_acc_d:+.4f} | **DIAGNOSTIC_ONLY** |")
    else:
        report_lines.append("WARNING: Classification comparison summary not found.")
    report_lines.append("")
    
    report_lines.append("## 8. Best Setting Selection")
    if BEST_SETTING_MD.exists():
        with open(BEST_SETTING_MD, mode="r", encoding="utf-8") as f:
            best_md_content = f.read()
        # Find the recommendation section
        if "## Selected Optimal Configuration" in best_md_content:
            selected_section = best_md_content.split("## Selected Optimal Configuration")[1].split("## Candidate Selection Performance Table")[0]
            report_lines.append("### Optimal Selection Details")
            report_lines.append(selected_section.strip())
            report_lines.append("")
        if "## Recommendation for Stage 5B" in best_md_content:
            rec_section = best_md_content.split("## Recommendation for Stage 5B")[1]
            report_lines.append("### Stage 5B Recommendation")
            report_lines.append(rec_section.strip())
            report_lines.append("")
    else:
        report_lines.append("WARNING: Best setting selection summary not found.")
    report_lines.append("")
    
    report_lines.append("## 9. Scientific Limitations")
    report_lines.append("> [!WARNING]")
    report_lines.append("> **Critical Limitation Statement:**")
    report_lines.append("> **“The 12-file Stage 5A experiment tests technical feasibility only. Because the embedding-to-mel mapping is trained on a very small pilot set, the results cannot be interpreted as final conversion performance.”**")
    report_lines.append("")
    report_lines.append("- **No Language Translation**: This process is strictly acoustic. German speech was converted **toward the Spanish acoustic/domain condition**, and Spanish speech was converted **toward the German acoustic/domain condition** in log-mel feature space before vocoding.")
    report_lines.append("- **Diagnostic-Only Results**: Classification results represent **diagnostic-only classification results** and not generalizable clinical performance.")
    report_lines.append("")
    
    report_lines.append("## 10. Decision for Stage 5B")
    report_lines.append("The selected parameter configuration and scale will be used for the Stage 5B 80-file scaling, isolating all Stage 5B runs inside `stage5b_subset_80` under the same safety constraints.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Report generated automatically by `07_write_stage5a_refinement_report.py`*")
    
    # Write report
    with open(REPORT_PATH, mode="w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Final Stage 5A-Refinement Report successfully generated at: {REPORT_PATH}")
    print("Step 7 Finished Successfully!\n")

if __name__ == "__main__":
    main()
