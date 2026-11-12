#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_validate_stage5c_converted_audio.py

Purpose:
Validates all 276 converted WAV files in Stage 5C:
- Checks existence, sample rate (22050), channels (mono), duration difference from source.
- Calculates RMS energy, peak amplitude, and 16-bit PCM clipping sample counts.
- Implements strict decision rules:
  1. Stops before classification if any file is missing, silent, wrong SR, or wrong channels.
  2. Raises a warning but continues if clipped files exceed 5% of total files.
Saves CSV log and Markdown summary.
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
import librosa

# Resolve directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STAGE5C_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc" / "stage5c_full_276"
INPUT_DIR = STAGE5C_DIR / "input_full_276_stage5c"
LOGS_DIR = STAGE5C_DIR / "logs_stage5c"
GENERATION_LOG = LOGS_DIR / "stage5c_conditioned_generation_log.csv"
VALIDATION_CSV = LOGS_DIR / "stage5c_converted_audio_validation.csv"
VALIDATION_SUMMARY_MD = LOGS_DIR / "stage5c_converted_audio_validation_summary.md"

def main():
    print("=" * 60)
    print("Stage 5C Step 5: Validating Converted WAV Files")
    print("=" * 60)

    if not GENERATION_LOG.exists():
        print(f"ERROR: Generation log not found at {GENERATION_LOG}. Run step 4 first.")
        sys.exit(1)

    df_gen = pd.read_csv(GENERATION_LOG)
    records = []
    
    total_files = len(df_gen)
    success_count = 0
    warning_count = 0
    failed_count = 0
    sr_pass_count = 0
    mono_pass_count = 0
    
    duration_diffs = []
    rms_values = []
    peak_amplitudes = []
    clipped_files_count = 0
    
    print(f"Validating {total_files} expected files...")

    for idx, row in df_gen.iterrows():
        source_filename = row["source_file"]
        rel_output = row["output_file"]
        gen_status = row["generation_status"]
        
        source_path = INPUT_DIR / source_filename
        output_path = STAGE5C_DIR / rel_output if rel_output else None
        
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
            "clipping_samples": 0,
            "clipping_detected": False,
            "empty_waveform": True,
            "status": "failed",
            "notes": ""
        }

        if gen_status != "success" or not output_path or not output_path.exists():
            record["notes"] = "File is missing or generation status failed."
            records.append(record)
            failed_count += 1
            continue

        try:
            # 1. Load source duration
            y_src, sr_src = librosa.load(str(source_path), sr=None, mono=False)
            src_dur = float(librosa.get_duration(y=y_src, sr=sr_src))
            record["source_duration_sec"] = src_dur

            # 2. Read using scipy.io.wavfile to inspect raw int16 PCM values for clipping
            sr_conv, wav_data = wavfile.read(str(output_path))
            
            # Check channels and data shape
            channels = wav_data.shape[1] if wav_data.ndim > 1 else 1
            nsamples = wav_data.shape[0]
            conv_dur = float(nsamples / sr_conv)
            
            # Convert to float for standard Peak/RMS calculation in range [-1.0, 1.0]
            wav_float = wav_data.astype(np.float32) / 32768.0
            peak_amp = float(np.max(np.abs(wav_float)))
            rms = float(np.sqrt(np.mean(wav_float ** 2)))
            
            # Clipping check in 16-bit PCM limits (-32768 and 32767)
            clipping_samples = int(np.sum((wav_data == 32767) | (wav_data == -32768)))
            clipping_detected = (clipping_samples > 0)
            
            record["exists"] = True
            record["sample_rate"] = int(sr_conv)
            record["channels"] = channels
            record["converted_duration_sec"] = conv_dur
            record["duration_diff_sec"] = conv_dur - src_dur
            record["rms_energy"] = rms
            record["peak_amplitude"] = peak_amp
            record["clipping_samples"] = clipping_samples
            record["clipping_detected"] = clipping_detected
            record["empty_waveform"] = (rms < 1e-4 or peak_amp < 1e-4)

            # Update stats
            if sr_conv == 22050:
                sr_pass_count += 1
            if channels == 1:
                mono_pass_count += 1
            if clipping_detected:
                clipped_files_count += 1
                
            duration_diffs.append(conv_dur - src_dur)
            rms_values.append(rms)
            peak_amplitudes.append(peak_amp)

            # Check technical specifications
            errors = []
            if sr_conv != 22050:
                errors.append(f"SR={sr_conv} (expected 22050)")
            if channels != 1:
                errors.append(f"Channels={channels} (expected mono)")
            if record["empty_waveform"]:
                errors.append("Empty/silent waveform")
            if abs(conv_dur - src_dur) > 0.5:
                errors.append(f"Duration difference too high: {conv_dur - src_dur:+.3f}s")

            if errors:
                record["status"] = "failed"
                record["notes"] = "; ".join(errors)
                failed_count += 1
            else:
                if clipping_detected:
                    record["status"] = "warning"
                    record["notes"] = f"Technical specs met. Clipping detected ({clipping_samples} samples)."
                    warning_count += 1
                else:
                    record["status"] = "success"
                    record["notes"] = "Technical specifications met."
                    success_count += 1

        except Exception as e:
            record["notes"] = f"Validation failed with error: {e}"
            failed_count += 1
            
        records.append(record)

    # Save details to CSV
    df_val = pd.DataFrame(records)
    df_val.to_csv(VALIDATION_CSV, index=False)
    print(f"Saved detailed validation CSV: {VALIDATION_CSV}")

    # Compute aggregate stats
    avg_dur_diff = np.mean(duration_diffs) if duration_diffs else 0.0
    max_dur_diff = np.max(np.abs(duration_diffs)) if duration_diffs else 0.0
    avg_rms = np.mean(rms_values) if rms_values else 0.0
    min_rms = np.min(rms_values) if rms_values else 0.0
    max_rms = np.max(rms_values) if rms_values else 0.0
    max_peak = np.max(peak_amplitudes) if peak_amplitudes else 0.0
    min_peak = np.min(peak_amplitudes) if peak_amplitudes else 0.0

    clipping_ratio = clipping_percent = (clipped_files_count / total_files) if total_files > 0 else 0.0
    clipping_warning = "No warning. Clipping is within acceptable threshold (<= 5%)."
    clipping_warning_triggered = False
    
    if clipping_ratio > 0.05:
        clipping_warning_triggered = True
        clipping_warning = f"WARNING: Clipped files constitute {clipping_ratio*100:.1f}% of total files (threshold is 5%)."

    # Write validation summary Markdown report
    summary_md_lines = [
        "# Stage 5C — Audio Validation Summary",
        "",
        "This report summarizes the technical specifications check for all converted WAV files.",
        "",
        "## Summary Metrics",
        f"- **Total Files Expected**: {total_files}",
        f"- **Total Files Found**: {df_val['exists'].sum()}",
        f"- **Success Count (specifications met, no clipping)**: {success_count}",
        f"- **Warning Count (specifications met, with clipping)**: {warning_count}",
        f"- **Failed Count (missing/silent/wrong format)**: {failed_count}",
        f"- **Sample Rate Pass Count (22050 Hz)**: {sr_pass_count} / {total_files}",
        f"- **Mono Pass Count (1 channel)**: {mono_pass_count} / {total_files}",
        f"- **Average Duration Difference**: {avg_dur_diff:+.4f} seconds",
        f"- **Maximum Duration Difference**: {max_dur_diff:.4f} seconds",
        f"- **Average RMS**: {avg_rms:.4f}",
        f"- **RMS Range**: [{min_rms:.4f}, {max_rms:.4f}]",
        f"- **Maximum Peak Amplitude**: {max_peak:.4f}",
        f"- **Peak Amplitude Range**: [{min_peak:.4f}, {max_peak:.4f}]",
        f"- **Number of Clipped Files**: {clipped_files_count} / {total_files} ({clipping_ratio*100:.1f}%)",
        f"- **Clipping Percentage**: {clipping_ratio*100:.1f}%",
        "",
        "## Clipping Warnings & Thresholds",
        f"- **Status**: {'**WARNING**' if clipping_warning_triggered else 'PASSED'}",
        f"- **Message**: {clipping_warning}",
        "",
        "## Decision Rules Output"
    ]

    if failed_count > 0:
        summary_md_lines.append("> [!CAUTION]")
        summary_md_lines.append(f"> **Pipeline Halt Condition Triggered**: {failed_count} files failed validation checks. DO NOT continue to classification evaluation.")
        print(f"PIPELINE FAILURE: {failed_count} files failed validation. Halt triggered.")
    else:
        summary_md_lines.append("> [!NOTE]")
        summary_md_lines.append("> **Pipeline Continue Condition Met**: All 276 converted files meet technical specifications. Proceed to classification evaluation.")
        if clipping_warning_triggered:
            summary_md_lines.append("> **Clipping Warning**: More than 5% of converted files are clipped. Discussion will be added to the final report.")
        print("PIPELINE SUCCESS: All 276 files passed validation. Proceeding.")

    with open(VALIDATION_SUMMARY_MD, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_md_lines))
    print(f"Saved validation summary report: {VALIDATION_SUMMARY_MD}")

    # Stop if failed files exist
    if failed_count > 0:
        sys.exit(1)
        
    print("Step 5 Finished Successfully!\n")

if __name__ == "__main__":
    main()
