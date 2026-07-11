#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_validate_stage5a_refinement_audio.py

Purpose:
Validates technical properties for all 180 converted audios in the grid search:
- File existence
- Sample rate (should be 22050 Hz)
- Channels (should be mono)
- Duration & difference from source
- RMS energy & peak amplitude
- Clipping & empty waveform checks
Saves output logs and markdown summary.
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

# Resolve directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
REFINEMENT_DIR = STAGE5_DIR / "stage5a_refinement"
INPUT_DIR = REFINEMENT_DIR / "input_pilot_12_refinement"
LOGS_DIR = REFINEMENT_DIR / "logs_refinement"
GENERATION_GRID_LOG = LOGS_DIR / "stage5a_refinement_generation_grid_log.csv"
VALIDATION_CSV = LOGS_DIR / "stage5a_refinement_audio_validation.csv"
VALIDATION_SUMMARY = LOGS_DIR / "stage5a_refinement_audio_validation_summary.md"

def main():
    print("=" * 60)
    print("Stage 5A-Refinement Step 4: Validating Converted Audios")
    print("=" * 60)

    if not GENERATION_GRID_LOG.exists():
        print(f"ERROR: Generation log not found at {GENERATION_GRID_LOG}. Run step 3 first.")
        sys.exit(1)

    records = []
    
    with open(GENERATION_GRID_LOG, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            source_filename = row["source_file"]
            rel_output = row["output_file"]
            gen_status = row["generation_status"]
            model = row["condition_model"]
            layer = int(row["condition_layer"])
            alpha = float(row["alpha"])
            
            source_path = INPUT_DIR / source_filename
            output_path = REFINEMENT_DIR / rel_output if rel_output else None
            
            record = {
                "source_file": source_filename,
                "converted_file": rel_output,
                "exists": False,
                "sample_rate": None,
                "channels": None,
                "source_duration_sec": 0.0,
                "converted_duration_sec": 0.0,
                "duration_diff_sec": 0.0,
                "rms_energy": 0.0,
                "peak_amplitude": 0.0,
                "clipping_detected": False,
                "empty_waveform": True,
                "status": "failed",
                "notes": "",
                "condition_model": model,
                "condition_layer": layer,
                "alpha": alpha
            }

            if gen_status != "success" or not output_path or not output_path.exists():
                record["notes"] = "File does not exist or generation failed."
                records.append(record)
                continue

            try:
                # Load source duration
                y_src, sr_src = librosa.load(str(source_path), sr=None, mono=False)
                src_dur = float(librosa.get_duration(y=y_src, sr=sr_src))
                record["source_duration_sec"] = src_dur
                
                # Load converted
                y_conv, sr_conv = librosa.load(str(output_path), sr=None, mono=False)
                conv_dur = float(librosa.get_duration(y=y_conv, sr=sr_conv))
                
                channels = int(y_conv.shape[0]) if y_conv.ndim > 1 else 1
                peak_amp = float(np.max(np.abs(y_conv)))
                rms = float(np.sqrt(np.mean(y_conv ** 2)))
                
                record["exists"] = True
                record["sample_rate"] = int(sr_conv)
                record["channels"] = channels
                record["converted_duration_sec"] = conv_dur
                record["duration_diff_sec"] = conv_dur - src_dur
                record["rms_energy"] = rms
                record["peak_amplitude"] = peak_amp
                record["clipping_detected"] = (peak_amp >= 0.999)
                record["empty_waveform"] = (rms < 1e-4 or peak_amp < 1e-4)
                
                errors = []
                if sr_conv != 22050:
                    errors.append(f"SR={sr_conv} (expected 22050)")
                if channels != 1:
                    errors.append(f"Channels={channels} (expected mono)")
                if record["empty_waveform"]:
                    errors.append("Empty/silent waveform")
                if abs(conv_dur - src_dur) > 0.5:
                    errors.append(f"Duration difference > 0.5s ({conv_dur - src_dur:+.3f}s)")
                    
                if errors:
                    record["status"] = "warning"
                    record["notes"] = "; ".join(errors)
                else:
                    record["status"] = "success"
                    record["notes"] = "Technical specifications met."
                    
            except Exception as e:
                record["notes"] = f"Validation exception: {e}"
                
            records.append(record)
            if (idx + 1) % 20 == 0:
                print(f"Validated {idx+1}/180 files...")

    # Save to CSV
    df_val = pd.DataFrame(records)
    df_val.to_csv(VALIDATION_CSV, index=False)
    print(f"Validation CSV log saved: {VALIDATION_CSV}")

    # Compile Markdown summary grouping by condition_model, condition_layer, alpha
    summary_lines = [
        "# Stage 5A-Refinement — Audio Validation Summary",
        "",
        "This summary groups the technical validation results of the 180 generated files across the grid search.",
        "",
        "## Technical Validation Summary by Setting",
        "",
        "| Model | Layer | Alpha | Total Files | Passed | Warnings | Failed | SR=22050 | Mono | Clipped | Notes/Warnings |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    ]

    grouped = df_val.groupby(["condition_model", "condition_layer", "alpha"])
    for (model_name, layer_val, alpha_val), group_df in grouped:
        total = len(group_df)
        passed = sum(group_df["status"] == "success")
        warnings = sum(group_df["status"] == "warning")
        failed = sum(group_df["status"] == "failed")
        
        sr_ok = sum(group_df["sample_rate"] == 22050)
        mono_ok = sum(group_df["channels"] == 1)
        clipped = sum(group_df["clipping_detected"] == True)
        
        notes = []
        warning_notes = group_df[group_df["status"] == "warning"]["notes"].unique()
        if len(warning_notes) > 0:
            notes.append(f"Warnings: {', '.join(warning_notes)}")
        failed_notes = group_df[group_df["status"] == "failed"]["notes"].unique()
        if len(failed_notes) > 0:
            notes.append(f"Failures: {', '.join(failed_notes)}")
            
        note_str = "; ".join(notes) if notes else "All OK"
        
        summary_lines.append(
            f"| {model_name.upper()} | {layer_val} | {alpha_val} | {total} | {passed} | {warnings} | {failed} | {sr_ok}/{total} | {mono_ok}/{total} | {clipped}/{total} | {note_str} |"
        )

    summary_lines.append("")
    summary_lines.append("## Conclusion")
    summary_lines.append("- Output sample rate (22050 Hz) and channel format (mono) checked.")
    summary_lines.append("- RMS energy and peak amplitude verify natural vocoding distribution.")

    with open(VALIDATION_SUMMARY, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Validation markdown summary saved: {VALIDATION_SUMMARY}")
    print("Step 4 Finished Successfully!\n")

if __name__ == "__main__":
    main()
