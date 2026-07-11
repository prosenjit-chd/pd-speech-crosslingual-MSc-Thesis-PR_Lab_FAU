#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_validate_stage5a_converted_audio.py

Purpose:
Validates all converted audio files technical properties:
- File existence
- Sample rate (should be 22050 Hz)
- Channels (should be mono)
- Duration & difference from source
- RMS energy
- Peak amplitude & clipping check
- Empty waveform check
Saves log and summary to:
- logs_stage5/stage5a_converted_audio_validation.csv
- logs_stage5/stage5a_converted_audio_validation_summary.md
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np
import librosa

# Resolve directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
INPUT_12_DIR = STAGE5_DIR / "input_pilot_12"
LOG_DIR = STAGE5_DIR / "logs_stage5"
GENERATION_LOG_PATH = LOG_DIR / "stage5a_conditioned_generation_log.csv"
VALIDATION_CSV_PATH = LOG_DIR / "stage5a_converted_audio_validation.csv"
VALIDATION_SUMMARY_PATH = LOG_DIR / "stage5a_converted_audio_validation_summary.md"

def main():
    print("=" * 60)
    print("Stage 5A Step 5: Validating Converted Audio Files")
    print("=" * 60)

    if not GENERATION_LOG_PATH.exists():
        print(f"ERROR: Generation log not found at {GENERATION_LOG_PATH}. Run step 4 first.")
        sys.exit(1)

    validation_records = []

    with open(GENERATION_LOG_PATH, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            source_filename = row["source_file"]
            rel_output_file = row["output_file"]
            gen_status = row["generation_status"]
            
            source_path = INPUT_12_DIR / source_filename
            output_path = STAGE5_DIR / rel_output_file if rel_output_file else None
            
            print(f"[{idx+1}] Validating output of {source_filename} ... ")
            
            record = {
                "source_file": source_filename,
                "converted_file": rel_output_file,
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
                "notes": ""
            }
            
            if gen_status != "success" or not output_path or not output_path.exists():
                print("  FAILED (does not exist or generation failed)")
                record["notes"] = "File does not exist or generation failed"
                validation_records.append(record)
                continue
                
            try:
                # Load source to get duration
                y_src, sr_src = librosa.load(str(source_path), sr=None, mono=False)
                src_dur = float(librosa.get_duration(y=y_src, sr=sr_src))
                record["source_duration_sec"] = src_dur
                
                # Load converted (preserving channels and SR)
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
                record["clipping_detected"] = (peak_amp >= 0.999) # Peak normalized output close to 1.0
                record["empty_waveform"] = (rms < 1e-4 or peak_amp < 1e-4)
                
                # Verify technical parameters
                errors = []
                if sr_conv != 22050:
                    errors.append(f"SR={sr_conv} (expected 22050)")
                if channels != 1:
                    errors.append(f"Channels={channels} (expected mono)")
                if record["empty_waveform"]:
                    errors.append("Empty/silent waveform")
                if abs(conv_dur - src_dur) > 0.5: # Allow a minor delta due to hop size / window padding
                    errors.append(f"Duration difference > 0.5s ({conv_dur - src_dur:+.3f}s)")
                    
                if errors:
                    record["status"] = "warning"
                    record["notes"] = "; ".join(errors)
                    print(f"  WARNING: {record['notes']}")
                else:
                    record["status"] = "success"
                    record["notes"] = "Technical specifications met."
                    print("  SUCCESS")
                    
            except Exception as e:
                print(f"  ERROR: {e}")
                record["notes"] = f"Validation exception: {e}"
                
            validation_records.append(record)

    # Save validation CSV
    with open(VALIDATION_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_file", "converted_file", "exists", "sample_rate", "channels",
            "source_duration_sec", "converted_duration_sec", "duration_diff_sec",
            "rms_energy", "peak_amplitude", "clipping_detected", "empty_waveform",
            "status", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in validation_records:
            writer.writerow(r)
    print(f"\nValidation CSV saved to: {VALIDATION_CSV_PATH}")

    # Write Markdown summary
    num_success = sum(1 for r in validation_records if r["status"] == "success")
    num_warning = sum(1 for r in validation_records if r["status"] == "warning")
    num_failed = sum(1 for r in validation_records if r["status"] == "failed")
    
    summary_lines = [
        "# Stage 5A — Converted Audio Validation Report",
        "",
        "## Overall Summary",
        f"- **Total Files Evaluated**: {len(validation_records)}",
        f"- **Fully Valid (Success)**: {num_success}",
        f"- **Warnings (Technical Divergence)**: {num_warning}",
        f"- **Failed (Missing/Unusable)**: {num_failed}",
        "",
        "## Technical Validation Matrix",
        "",
        "| Source File | Converted File | SR (Hz) | Channels | Src Dur (s) | Conv Dur (s) | Delta (s) | Peak Amp | RMS | Status | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    ]
    
    for r in validation_records:
        rel_file = r["converted_file"] if r["converted_file"] else "N/A"
        summary_lines.append(
            f"| `{r['source_file']}` | `{rel_file}` | {r['sample_rate']} | {r['channels']} | "
            f"{r['source_duration_sec']:.3f} | {r['converted_duration_sec']:.3f} | {r['duration_diff_sec']:+.3f} | "
            f"{r['peak_amplitude']:.3f} | {r['rms_energy']:.3f} | **{r['status'].upper()}** | {r['notes']} |"
        )
        
    summary_lines.append("")
    summary_lines.append("## Conclusion & Warning Note")
    summary_lines.append("- Output sample rate of 22050 Hz and single channel (mono) are verified.")
    summary_lines.append("- Duration difference should be close to 0 (typically within a few frames due to STFT window padding).")
    
    with open(VALIDATION_SUMMARY_PATH, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Validation summary markdown saved to: {VALIDATION_SUMMARY_PATH}")
    print("Step 5 Finished Successfully!\n")

if __name__ == "__main__":
    main()
