#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_subset_80_stage.py

Purpose:
Orchestrates the preparation and verification of the Stage 3 (80-file) subset:
1. Reads C:\pd-speech-crosslingual\metadata\dataset_index_readtext.csv.
2. Selects 20 Spanish PD, 20 Spanish HC, 20 German PD, and 20 German HC readtext files (total 80 files).
3. Copies and renames them to C:\pd-speech-crosslingual\voice_conversion\input_subset_80\ using standard formats.
4. Logs selection metadata to C:\pd-speech-crosslingual\voice_conversion\logs_subset_80\subset_80_selection_log.csv.
5. Invokes the prepare_hifigan_subset_80_audio.py script to downsample, convert to mono, and normalize waveforms.
6. Runs inspection on input_subset_80_22050 and saves results to C:\pd-speech-crosslingual\voice_conversion\logs_subset_80\subset_80_inspection_summary.csv.
7. Updates hifigan_stage3_experiment_log.md inside logs_subset_80 with the details.
"""

import os
import sys
import csv
import shutil
import traceback
from pathlib import Path
import numpy as np
import soundfile as sf

# Resolve project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_CSV_PATH = PROJECT_ROOT / "metadata" / "dataset_index_readtext.csv"
INPUT_SUBSET_DIR = PROJECT_ROOT / "voice_conversion" / "input_subset_80"
INPUT_SUBSET_22050_DIR = PROJECT_ROOT / "voice_conversion" / "input_subset_80_22050"
LOG_DIR = PROJECT_ROOT / "voice_conversion" / "logs_subset_80"
SELECTION_LOG_PATH = LOG_DIR / "subset_80_selection_log.csv"
INSPECTION_SUMMARY_PATH = LOG_DIR / "subset_80_inspection_summary.csv"
EXPERIMENT_LOG_PATH = LOG_DIR / "hifigan_stage3_experiment_log.md"

# Add script folder to path to import helper and inspect functions
sys.path.append(str(Path(__file__).resolve().parent))
try:
    import prepare_hifigan_subset_80_audio
    import inspect_audio_folder
except ImportError:
    pass

def select_and_copy_files():
    print("=" * 60)
    print("Step 1: Selecting and Copying Subset 80 WAV files")
    print(f"Index CSV: {INDEX_CSV_PATH}")
    print("=" * 60)
    
    if not INDEX_CSV_PATH.exists():
        print(f"ERROR: Dataset index CSV not found at {INDEX_CSV_PATH}")
        sys.exit(1)
        
    INPUT_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filter groups
    groups = {
        ("Spanish", "PD"): [],
        ("Spanish", "HC"): [],
        ("German", "PD"): [],
        ("German", "HC"): []
    }
    
    # Read the dataset index CSV
    with open(INDEX_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row.get("language")
            lbl = row.get("label")
            task = row.get("task")
            
            # Ensure we only pick readtext task
            if task != "readtext":
                continue
                
            key = (lang, lbl)
            if key in groups and len(groups[key]) < 20:
                groups[key].append(row)
                
    # Check if we have 20 files in each group
    missing_groups = []
    for key, items in groups.items():
        if len(items) < 20:
            missing_groups.append(f"{key[0]} {key[1]} (found {len(items)})")
            
    if missing_groups:
        print(f"ERROR: Insufficient balanced files found in index: {', '.join(missing_groups)}")
        sys.exit(1)
        
    # Naming format mapping
    prefix_map = {
        ("Spanish", "PD"): "SP_PD",
        ("Spanish", "HC"): "SP_HC",
        ("German", "PD"): "DE_PD",
        ("German", "HC"): "DE_HC"
    }
    
    selection_log_entries = []
    
    for key, row_list in groups.items():
        prefix = prefix_map[key]
        for idx, row in enumerate(row_list, 1):
            new_filename = f"{prefix}_{idx:03d}.wav"
            rel_src_path = row.get("file_path")
            src_full_path = PROJECT_ROOT / rel_src_path
            dest_full_path = INPUT_SUBSET_DIR / new_filename
            
            log_entry = {
                "new_filename": new_filename,
                "original_filename": Path(rel_src_path).name,
                "original_path": rel_src_path,
                "language": key[0],
                "label": key[1],
                "duration_if_available": 0.0,
                "status": "failed"
            }
            
            print(f"Copying {rel_src_path} -> {new_filename} ... ", end="")
            
            if not src_full_path.exists():
                print("FAILED (source file does not exist)")
                selection_log_entries.append(log_entry)
                continue
                
            try:
                # Copy without modifying original file
                shutil.copy2(src_full_path, dest_full_path)
                
                # Get duration
                info = sf.info(dest_full_path)
                log_entry["duration_if_available"] = float(info.duration)
                log_entry["status"] = "copied"
                print("SUCCESS")
            except Exception as e:
                log_entry["status"] = f"error ({type(e).__name__})"
                print(f"FAILED ({e})")
                
            selection_log_entries.append(log_entry)
            
    # Write selection log
    with open(SELECTION_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["new_filename", "original_filename", "original_path", "language", "label", "duration_if_available", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in selection_log_entries:
            writer.writerow(entry)
            
    print(f"\nSelection log successfully written to: {SELECTION_LOG_PATH}\n")
    return selection_log_entries

def run_preprocessing_pipeline():
    print("=" * 60)
    print("Step 2: Preprocessing Subset 80 Files (Downsampling & Normalization)")
    print("=" * 60)
    try:
        # Run preparation script
        prepare_hifigan_subset_80_audio.main()
        return "success"
    except Exception as e:
        print(f"ERROR: Preprocessing failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return "failed"

def run_inspection_pipeline():
    print("=" * 60)
    print("Step 3: Technical Audio Folder Inspection")
    print("=" * 60)
    try:
        # Inspect files in the input_subset_80_22050 directory
        wav_files = sorted(list(INPUT_SUBSET_22050_DIR.rglob("*.wav")))
        if not wav_files:
            print(f"No WAV files found in '{INPUT_SUBSET_22050_DIR}' to inspect.")
            return "empty"
            
        results = []
        for idx, f_path in enumerate(wav_files, 1):
            print(f"[{idx}/{len(wav_files)}] Inspecting: {f_path.name}")
            info = inspect_audio_folder.inspect_file(f_path)
            results.append(info)
            
        fieldnames = ["filename", "path", "sample_rate", "duration_sec", "num_samples", "channels", "max_abs_amplitude", "rms", "status"]
        with open(INSPECTION_SUMMARY_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
                
        print(f"Inspection complete. Summary report saved to: {INSPECTION_SUMMARY_PATH}\n")
        return "success"
    except Exception as e:
        print(f"ERROR: Inspection summary failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return "failed"

def update_experiment_log(selection_entries, preprocess_status):
    print("=" * 60)
    print("Step 4: Updating Experiment Log Markdown")
    print("=" * 60)
    
    selected_list = []
    for entry in selection_entries:
        if entry["status"] == "copied":
            selected_list.append(f"- `{entry['new_filename']}` (Original: `{entry['original_filename']}`, Language: `{entry['language']}`, Diagnosis: `{entry['label']}`, Duration: {entry['duration_if_available']:.2f}s)")
            
    selected_text = "\n".join(selected_list) if selected_list else "- *No files copied successfully.*"
    copied_count = sum(1 for e in selection_entries if e["status"] == "copied")
    
    experiment_log_content = f"""# HiFi-GAN Stage 3: Controlled Subset Experiment (80 Files)

## Goal
Test whether HiFi-GAN can successfully reconstruct Spanish and German PD/HC speech samples on a larger, balanced 80-file subset.

## Current Status
- **Environment Verification**: **COMPLETED** (Python virtual environment verified).
- **Stage 3 Scripts Status**: **CREATED**
  - Selection & Orchestration: `run_subset_80_stage.py`
  - Preprocessing: `prepare_hifigan_subset_80_audio.py`
  - Inspect & Compare: `inspect_and_compare_generated_subset_80.py`
  - Evaluation: `evaluate_reconstructed_subset_80.py`
- **Data Selection Status**: **COMPLETED** ({copied_count} files selected and copied).
- **Preprocessing Status**: **{preprocess_status.upper()}**

## Selected Files Summary
The following {copied_count} balanced readtext WAV files were copied and renamed from the index:
{selected_text}

## Preprocessing Details
- Input directory: `voice_conversion/input_subset_80`
- Processed directory: `voice_conversion/input_subset_80_22050` (Mono, 22050 Hz, Amplitude Normalized)
- Preprocessing log location: `voice_conversion/logs_subset_80/subset_80_preprocessing_log.csv`
- Technical inspection summary: `voice_conversion/logs_subset_80/subset_80_inspection_summary.csv`

## Planned Evaluation Criteria
1. **Audio duration comparison**: Check if the reconstructed WAV duration matches the original file exactly.
2. **Audio technical statistics**: Compare sample rate, channels, peak amplitude, and RMS energy between original and reconstructed audio.
3. **Downstream Classification**: Run baseline embedding extraction (WavLM, Wav2Vec2, XLSR) and cross-language classification scenarios.
"""
    
    with open(EXPERIMENT_LOG_PATH, mode="w", encoding="utf-8") as f:
        f.write(experiment_log_content)
    print(f"Updated Stage 3 experiment log at: {EXPERIMENT_LOG_PATH}\n")

def main():
    selection_entries = select_and_copy_files()
    preprocess_status = run_preprocessing_pipeline()
    inspect_status = run_inspection_pipeline()
    update_experiment_log(selection_entries, preprocess_status)
    
    print("=" * 60)
    print("Subset 80 Stage 3 Preparation Execution Finished!")
    print("=" * 60)

if __name__ == "__main__":
    main()
