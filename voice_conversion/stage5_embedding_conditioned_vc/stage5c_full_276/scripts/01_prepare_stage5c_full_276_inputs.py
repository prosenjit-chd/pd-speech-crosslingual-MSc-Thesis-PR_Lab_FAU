#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_prepare_stage5c_full_276_inputs.py

Purpose:
Prepares the folder structure and inputs for Stage 5C:
1. Creates the required subdirectories under stage5c_full_276/.
2. Copies the 276 full dataset files from input_full/ to input_full_276_stage5c/.
3. Creates copy log CSV and summary MD.
"""

import os
import sys
import csv
import shutil
from pathlib import Path

# Resolve project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
STAGE5C_DIR = STAGE5_DIR / "stage5c_full_276"

# Inputs
SRC_INPUT_DIR = PROJECT_ROOT / "voice_conversion" / "input_full"
# Outputs
DEST_INPUT_DIR = STAGE5C_DIR / "input_full_276_stage5c"
LOGS_DIR = STAGE5C_DIR / "logs_stage5c"
COPY_LOG_PATH = LOGS_DIR / "stage5c_input_copy_log.csv"
SUMMARY_MD_PATH = LOGS_DIR / "stage5c_input_summary.md"

def main():
    print("=" * 60)
    print("Stage 5C Step 1: Preparing Directories and Inputs")
    print("=" * 60)

    # 1. Create directory structure
    subdirs = [
        "input_full_276_stage5c",
        "source_embeddings_stage5c",
        "target_domain_embeddings_stage5c",
        "converted_mels_stage5c",
        "converted_audio_stage5c/spanish_to_german",
        "converted_audio_stage5c/german_to_spanish",
        "features_original_stage5c",
        "features_converted_stage5c",
        "outputs_original_stage5c",
        "outputs_converted_stage5c",
        "logs_stage5c",
        "scripts"
    ]

    for d in subdirs:
        path = STAGE5C_DIR / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory verified: {path.relative_to(STAGE5C_DIR)}")

    # 2. Copy files
    if not SRC_INPUT_DIR.exists():
        print(f"ERROR: Stage 5C input directory not found at {SRC_INPUT_DIR}")
        sys.exit(1)

    # Scan SRC_INPUT_DIR for all wav files
    wav_files = sorted([f for f in SRC_INPUT_DIR.glob("*.wav")])
    if len(wav_files) != 276:
        print(f"WARNING: Expected 276 wav files, but found {len(wav_files)} at {SRC_INPUT_DIR}")

    copied_entries = []

    for src_file_path in wav_files:
        filename = src_file_path.name
        dest_file_path = DEST_INPUT_DIR / filename
        
        # Parse info from name: e.g. SP_HC_001.wav, DE_PD_088.wav
        parts = filename.split(".")[0].split("_")
        lang_code = parts[0] # SP or DE
        label_code = parts[1] # HC or PD
        
        language = "Spanish" if lang_code == "SP" else "German"
        label = label_code # HC or PD
        group = label_code
        
        copied_status = "failed"
        try:
            shutil.copy2(src_file_path, dest_file_path)
            copied_status = "copied"
        except Exception as e:
            print(f"FAILED to copy {filename}: {e}")

        copied_entries.append({
            "stage5c_filename": filename,
            "source_path": str(src_file_path),
            "language": language,
            "label": label,
            "group": group,
            "original_filename": filename,
            "copied_status": copied_status
        })

    # Write CSV copy log
    with open(COPY_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["stage5c_filename", "source_path", "language", "label", "group", "original_filename", "copied_status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in copied_entries:
            writer.writerow(entry)
    print(f"Created input copy log: {COPY_LOG_PATH}")

    # Write Markdown summary
    summary_lines = [
        "# Stage 5C — Full 276-File Input Summary",
        "",
        "This dataset composition was copied from the predefined full folder `voice_conversion/input_full`.",
        "",
        "## Selected Files Inventory",
        "",
        "| File | Language | Diagnosis | Source Original File | Status |",
        "| --- | --- | --- | --- | --- |"
    ]

    for entry in copied_entries:
        summary_lines.append(
            f"| `{entry['stage5c_filename']}` | {entry['language']} | {entry['label']} | `{entry['original_filename']}` | {entry['copied_status']} |"
        )

    summary_lines.append("")
    summary_lines.append("## Setup Verification")
    summary_lines.append(f"- Input directory: [input_full_276_stage5c](file:///{DEST_INPUT_DIR.as_posix()})")
    
    de_hc = sum(1 for e in copied_entries if e["language"] == "German" and e["label"] == "HC")
    de_pd = sum(1 for e in copied_entries if e["language"] == "German" and e["label"] == "PD")
    sp_hc = sum(1 for e in copied_entries if e["language"] == "Spanish" and e["label"] == "HC")
    sp_pd = sum(1 for e in copied_entries if e["language"] == "Spanish" and e["label"] == "PD")
    
    summary_lines.append(f"- **German HC**: {de_hc} files")
    summary_lines.append(f"- **German PD**: {de_pd} files")
    summary_lines.append(f"- **Spanish HC**: {sp_hc} files")
    summary_lines.append(f"- **Spanish PD**: {sp_pd} files")
    summary_lines.append(f"- **Total Balanced Files**: {len(copied_entries)} files")

    with open(SUMMARY_MD_PATH, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Created summary markdown: {SUMMARY_MD_PATH}")
    print("Step 1 Finished Successfully!\n")

if __name__ == "__main__":
    main()
