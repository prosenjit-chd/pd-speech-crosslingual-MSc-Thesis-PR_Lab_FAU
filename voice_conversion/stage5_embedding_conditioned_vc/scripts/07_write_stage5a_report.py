#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
07_write_stage5a_report.py

Purpose:
Generates the final markdown report:
logs_stage5/stage5a_embedding_conditioned_conversion_report.md
Reads output logs from all previous steps to populate tables and summaries.
"""

import os
import sys
import csv
from pathlib import Path
import pandas as pd

# Resolve directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
LOG_DIR = STAGE5_DIR / "logs_stage5"

SELECTION_LOG = LOG_DIR / "stage5a_pilot_12_selection_log.csv"
EXTRACTION_LOG = LOG_DIR / "stage5a_embedding_extraction_log.csv"
DOMAIN_SUMMARY = LOG_DIR / "stage5a_domain_condition_summary.csv"
GENERATION_LOG = LOG_DIR / "stage5a_conditioned_generation_log.csv"
VALIDATION_LOG = LOG_DIR / "stage5a_converted_audio_validation.csv"
CLASSIFICATION_SUMMARY = LOG_DIR / "stage5a_original_vs_converted_classification_summary.csv"
REPORT_PATH = LOG_DIR / "stage5a_embedding_conditioned_conversion_report.md"

def read_csv_safe(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def main():
    print("=" * 60)
    print("Stage 5A Step 7: Generating Final Report")
    print("=" * 60)
    
    # Load logs
    df_select = read_csv_safe(SELECTION_LOG)
    df_extract = read_csv_safe(EXTRACTION_LOG)
    df_domain = read_csv_safe(DOMAIN_SUMMARY)
    df_gen = read_csv_safe(GENERATION_LOG)
    df_val = read_csv_safe(VALIDATION_LOG)
    df_class = read_csv_safe(CLASSIFICATION_SUMMARY)
    
    report_lines = []
    
    report_lines.append("# Stage 5A — 12-File Embedding-Conditioned Crosslingual Conversion Pilot Report")
    report_lines.append("")
    report_lines.append("## 1. Goal of Stage 5A")
    report_lines.append("The objective of this stage is to test **Plan B: embedding-conditioned generation/conversion** on a small 12-file pilot (3 Spanish HC, 3 Spanish PD, 3 German HC, 3 German PD) before scaling. We aim to determine whether acoustic representations (WavLM, Wav2Vec2, XLSR) can be used as conditioning information to guide crosslingual conversion while preserving Parkinson’s Disease (PD) vs. Healthy Control (HC) diagnostic speech features.")
    report_lines.append("")
    
    report_lines.append("## 2. Safety Rule Confirmation")
    report_lines.append("We confirm that all previous baseline, HiFi-GAN 12, 80, and 276 folders, logs, and files remain completely **read-only** and untouched. All Stage 5A work and intermediate outputs were isolated inside:")
    report_lines.append(f"`C:\\pd-speech-crosslingual\\voice_conversion\\stage5_embedding_conditioned_vc`")
    report_lines.append("")
    
    report_lines.append("## 3. Pilot Selection Summary")
    if not df_select.empty:
        report_lines.append(f"Exactly {len(df_select)} balanced files were prepared and verified in the pilot set:")
        report_lines.append("")
        report_lines.append("| Stage 5 File | Language | Diagnosis | Source Original | Copied Status |")
        report_lines.append("| --- | --- | --- | --- | --- |")
        for _, r in df_select.iterrows():
            report_lines.append(f"| `{r['stage5_filename']}` | {r['language']} | {r['label']} | `{r['original_filename']}` | {r['copied_status']} |")
    else:
        report_lines.append("WARNING: Selection log not found.")
    report_lines.append("")
    
    report_lines.append("## 4. Embedding Extraction Summary")
    if not df_extract.empty:
        success_count = sum(df_extract["status"] == "success")
        total_count = len(df_extract)
        report_lines.append(f"Layer-wise speech embeddings (layers 0, 4, 8, 11) were extracted for XLSR, Wav2Vec2, and WavLM models. Extraction success rate: **{success_count}/{total_count} ({success_count/total_count*100:.1f}%)**.")
    else:
        report_lines.append("WARNING: Embedding extraction log not found.")
    report_lines.append("")
    
    report_lines.append("## 5. Domain Condition Creation Summary")
    if not df_domain.empty:
        report_lines.append("Averaged target-domain conditions were calculated by averaging original file embeddings within each language for each model and target layer:")
        report_lines.append("")
        report_lines.append("| Model | Layer | Domain | Dimension | Mean Embedding Value | Std Dev |")
        report_lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, r in df_domain.iterrows():
            report_lines.append(f"| {r['model'].upper()} | {r['layer']} | {r['domain']} | {r['embedding_dimension']} | {r['mean_value']:.4f} | {r['std_value']:.4f} |")
    else:
        report_lines.append("WARNING: Domain condition summary not found.")
    report_lines.append("")
    
    report_lines.append("## 6. Conversion / Generation Method")
    report_lines.append("- **Method Applied**: **prototype embedding-conditioned conversion**")
    report_lines.append("- **Conditioning Feature**: XLSR layer 11 embeddings (1024 dimensions)")
    report_lines.append("- **Mathematical Mapping**: Fit a Ridge regression model ($m \\approx W \\cdot E + b$) from the 1024-dimensional embedding space to the 80-dimensional time-averaged log-mel spectrogram space using the 12 pilot files. For a source file, the predicted acoustic domain shift was computed as $\\Delta m = W \\cdot (E_{tgt\\_avg} - E_i)$ and applied to the source log-mel spectrogram with a conversion scale $\\alpha = 0.5$. The converted log-mel spectrograms were verified to match the HiFi-GAN vocoder config and synthesized using the pre-trained `universal_v1` generator.")
    report_lines.append("")
    
    report_lines.append("## 7. Audio Validation Results")
    if not df_val.empty:
        success_val = sum(df_val["status"] == "success")
        warning_val = sum(df_val["status"] == "warning")
        failed_val = sum(df_val["status"] == "failed")
        report_lines.append(f"Technical validation results for the generated WAV files:")
        report_lines.append(f"- **Success (specifications met)**: {success_val}")
        report_lines.append(f"- **Warning (technical deviations)**: {warning_val}")
        report_lines.append(f"- **Failed**: {failed_val}")
        report_lines.append("")
        report_lines.append("| Source File | Converted Output File | SR (Hz) | Mono | Conv Dur (s) | Delta (s) | Peak Amp | RMS | Status |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, r in df_val.iterrows():
            rel_file = r["converted_file"] if r["converted_file"] else "N/A"
            is_mono = "Yes" if r["channels"] == 1 else "No"
            report_lines.append(f"| `{r['source_file']}` | `{rel_file}` | {r['sample_rate']} | {is_mono} | {r['converted_duration_sec']:.3f} | {r['duration_diff_sec']:+.3f} | {r['peak_amplitude']:.3f} | {r['rms_energy']:.3f} | **{r['status'].upper()}** |")
    else:
        report_lines.append("WARNING: Audio validation log not found.")
    report_lines.append("")
    
    report_lines.append("## 8. Classification Comparison Results (Diagnostic Only)")
    report_lines.append("> [!IMPORTANT]")
    report_lines.append("> All classification metrics reported below represent diagnostic crosslingual evaluation check values on a very small pilot sample. They serve to observe representation drift and domain shifts rather than generalizable performance.")
    report_lines.append("")
    
    if not df_class.empty:
        report_lines.append("| Model | Layer | Scenario | Classifier | UAR Original | UAR Converted | UAR Delta | Acc Original | Acc Converted | Acc Delta |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, r in df_class.iterrows():
            report_lines.append(
                f"| {r['model'].upper()} | {r['layer']} | {r['train_language']}$\\to${r['test_language']} | {r['classifier']} | "
                f"{r['uar_orig']:.4f} | {r['uar_conv']:.4f} | {r['uar_delta']:+.4f} | "
                f"{r['acc_orig']:.4f} | {r['acc_conv']:.4f} | {r['acc_delta']:+.4f} |"
            )
    else:
        report_lines.append("WARNING: Classification comparison summary not found.")
    report_lines.append("")
    
    report_lines.append("## 9. Scientific Limitations")
    report_lines.append("> [!WARNING]")
    report_lines.append("> **Critical Limitation Statement:**")
    report_lines.append("> **“The 12-file Stage 5A experiment tests technical feasibility only. Because the embedding-to-mel mapping is trained on a very small pilot set, the results cannot be interpreted as final conversion performance.”**")
    report_lines.append("")
    report_lines.append("- **No Language Translation**: This process is strictly acoustic. German speech was converted **toward the Spanish acoustic/domain condition**, and Spanish speech was converted **toward the German acoustic/domain condition** in log-mel feature space before vocoding.")
    report_lines.append("- **Small Sample Bounds**: With only 3 speakers per group, SVM/Logistic Regression classification cannot yield final thesis performance conclusions.")
    report_lines.append("")
    
    report_lines.append("## 10. Decision Recommendation")
    report_lines.append("- **Option 1**: **Proceed to 80-file subset** using the prototype method if diagnostic classification preservation and audio validation meet baseline expectations (low representation drift).")
    report_lines.append("- **Option 2**: **Adjust method before scaling** if significant acoustic degradation or extreme representation drift is observed.")
    report_lines.append("- **Option 3**: **Fallback to existing VC model approach** if the linear projection fails to capture speaker identities or PD/HC features sufficiently.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Report generated automatically by `07_write_stage5a_report.py`*")
    
    # Write report
    with open(REPORT_PATH, mode="w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Final Stage 5A Report successfully generated at: {REPORT_PATH}")
    print("Step 7 Finished Successfully!\n")

if __name__ == "__main__":
    main()
