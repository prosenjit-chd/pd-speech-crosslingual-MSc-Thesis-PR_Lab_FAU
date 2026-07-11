#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_prepare_stage5a_refinement_inputs.py

Purpose:
Prepares the folder structure and inputs for Stage 5A-Refinement:
1. Creates the required subdirectories under stage5a_refinement/.
2. Copies the 12 pilot files from input_pilot_12/ to stage5a_refinement/input_pilot_12_refinement/.
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
REFINEMENT_DIR = STAGE5_DIR / "stage5a_refinement"

# Inputs
SRC_INPUT_DIR = STAGE5_DIR / "input_pilot_12"
DEST_INPUT_DIR = REFINEMENT_DIR / "input_pilot_12_refinement"
OLD_SELECTION_LOG = STAGE5_DIR / "logs_stage5" / "stage5a_pilot_12_selection_log.csv"

# Outputs
LOGS_DIR = REFINEMENT_DIR / "logs_refinement"
COPY_LOG_PATH = LOGS_DIR / "stage5a_refinement_input_copy_log.csv"
SUMMARY_MD_PATH = LOGS_DIR / "stage5a_refinement_input_summary.md"

def main():
    print("=" * 60)
    print("Stage 5A-Refinement Step 1: Preparing Directories and Inputs")
    print("=" * 60)

    # 1. Create directory structure
    subdirs = [
        "input_pilot_12_refinement",
        "source_embeddings_refinement",
        "target_domain_embeddings_refinement",
        "converted_mels_refinement",
        "converted_audio_refinement/xlsr_layer11",
        "converted_audio_refinement/wavlm_layer8",
        "converted_audio_refinement/wavlm_layer11",
        "features_original_refinement",
        "features_converted_refinement",
        "outputs_original_refinement",
        "outputs_converted_refinement",
        "logs_refinement",
        "scripts"
    ]

    for d in subdirs:
        path = REFINEMENT_DIR / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory verified: {path.relative_to(REFINEMENT_DIR)}")

    # 2. Copy files
    if not SRC_INPUT_DIR.exists():
        print(f"ERROR: Stage 5A input directory not found at {SRC_INPUT_DIR}")
        sys.exit(1)

    if not OLD_SELECTION_LOG.exists():
        print(f"ERROR: Stage 5A selection log not found at {OLD_SELECTION_LOG}")
        sys.exit(1)

    copied_entries = []

    with open(OLD_SELECTION_LOG, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["stage5_filename"]
            src_file_path = SRC_INPUT_DIR / filename
            dest_file_path = DEST_INPUT_DIR / filename

            print(f"Copying {filename} ... ", end="")
            copied_status = "failed"
            if src_file_path.exists():
                try:
                    shutil.copy2(src_file_path, dest_file_path)
                    copied_status = "copied"
                    print("SUCCESS")
                except Exception as e:
                    print(f"FAILED ({e})")
            else:
                print("FAILED (does not exist)")

            copied_entries.append({
                "stage5_filename": filename,
                "source_path": row["source_path"],
                "language": row["language"],
                "label": row["label"],
                "group": row["group"],
                "original_filename": row["original_filename"],
                "copied_status": copied_status
            })

    # Write CSV copy log
    with open(COPY_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["stage5_filename", "source_path", "language", "label", "group", "original_filename", "copied_status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in copied_entries:
            writer.writerow(entry)
    print(f"Created selection copy log: {COPY_LOG_PATH}")

    # Write Markdown summary
    summary_lines = [
        "# Stage 5A-Refinement — 12-File Input Summary",
        "",
        "This refinement dataset uses the same 12 balanced files from the Stage 5A pilot.",
        "",
        "## Selected Files Inventory",
        "",
        "| Refinement File | Language | Diagnosis | Source Original File | Status |",
        "| --- | --- | --- | --- | --- |"
    ]

    for entry in copied_entries:
        summary_lines.append(
            f"| `{entry['stage5_filename']}` | {entry['language']} | {entry['label']} | `{entry['original_filename']}` | {entry['copied_status']} |"
        )

    summary_lines.append("")
    summary_lines.append("## Setup Verification")
    summary_lines.append(f"- Input directory: [input_pilot_12_refinement](file:///{DEST_INPUT_DIR.as_posix()})")
    summary_lines.append("- Balanced structure: 3 Spanish HC, 3 Spanish PD, 3 German HC, 3 German PD.")

    with open(SUMMARY_MD_PATH, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Created summary markdown: {SUMMARY_MD_PATH}")
    print("Step 1 Finished Successfully!\n")

if __name__ == "__main__":
    main()
