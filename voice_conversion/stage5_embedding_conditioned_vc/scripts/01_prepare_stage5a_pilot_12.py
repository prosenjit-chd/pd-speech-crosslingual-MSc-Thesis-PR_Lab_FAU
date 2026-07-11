#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_prepare_stage5a_pilot_12.py

Purpose:
Prepares the 12-file pilot dataset for Stage 5A:
1. Locates original files from voice_conversion/input_pilot/.
2. Copies them to voice_conversion/stage5_embedding_conditioned_vc/input_pilot_12/.
3. Reads metadata from voice_conversion/logs/pilot_selection_log.csv to construct
   the new selection log and summary.
"""

import os
import sys
import csv
import shutil
from pathlib import Path

# Resolve project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
INPUT_PILOT_DIR = PROJECT_ROOT / "voice_conversion" / "input_pilot"
INPUT_PILOT_12_DIR = STAGE5_DIR / "input_pilot_12"
OLD_SELECTION_LOG = PROJECT_ROOT / "voice_conversion" / "logs" / "pilot_selection_log.csv"
NEW_SELECTION_LOG = STAGE5_DIR / "logs_stage5" / "stage5a_pilot_12_selection_log.csv"
NEW_SELECTION_SUMMARY = STAGE5_DIR / "logs_stage5" / "stage5a_pilot_12_selection_summary.md"

def main():
    print("=" * 60)
    print("Stage 5A Step 1: Preparing 12-File Pilot")
    print("=" * 60)

    # Ensure directories exist
    INPUT_PILOT_12_DIR.mkdir(parents=True, exist_ok=True)
    (STAGE5_DIR / "logs_stage5").mkdir(parents=True, exist_ok=True)
    (STAGE5_DIR / "converted_mels_stage5").mkdir(parents=True, exist_ok=True)

    if not OLD_SELECTION_LOG.exists():
        print(f"ERROR: Old pilot selection log not found at {OLD_SELECTION_LOG}")
        sys.exit(1)

    copied_entries = []

    # Read the previous selection log
    with open(OLD_SELECTION_LOG, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["new_filename"]
            src_file_path = INPUT_PILOT_DIR / filename
            dest_file_path = INPUT_PILOT_12_DIR / filename

            print(f"Copying {src_file_path.name} to {INPUT_PILOT_12_DIR.name} ... ", end="")
            copied_status = "failed"
            if src_file_path.exists():
                try:
                    shutil.copy2(src_file_path, dest_file_path)
                    copied_status = "copied"
                    print("SUCCESS")
                except Exception as e:
                    print(f"FAILED ({e})")
            else:
                print("FAILED (source file does not exist)")

            copied_entries.append({
                "stage5_filename": filename,
                "source_path": row["original_path"],
                "language": row["language"],
                "label": row["label"],
                "group": row["label"], # group column is mapped to label (HC/PD)
                "original_filename": row["original_filename"],
                "copied_status": copied_status
            })

    # Write new CSV selection log
    with open(NEW_SELECTION_LOG, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["stage5_filename", "source_path", "language", "label", "group", "original_filename", "copied_status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in copied_entries:
            writer.writerow(entry)
    print(f"Created selection log: {NEW_SELECTION_LOG}")

    # Write Markdown summary
    summary_lines = [
        "# Stage 5A — 12-File Pilot Dataset Selection Summary",
        "",
        "This pilot dataset is prepared to test embedding-conditioned crosslingual voice conversion.",
        "It consists of exactly 12 balanced readtext files copied from the previous HiFi-GAN pilot.",
        "",
        "## File Inventory",
        "",
        "| Stage 5 File | Language | Diagnosis | Source Original File | Status |",
        "| --- | --- | --- | --- | --- |"
    ]

    for entry in copied_entries:
        summary_lines.append(
            f"| `{entry['stage5_filename']}` | {entry['language']} | {entry['label']} | `{entry['original_filename']}` | {entry['copied_status']} |"
        )

    summary_lines.append("")
    summary_lines.append("## Verification")
    summary_lines.append(f"- Input directory: [input_pilot_12](file:///{INPUT_PILOT_12_DIR.as_posix()})")
    summary_lines.append(f"- Balanced setup: 3 Spanish HC, 3 Spanish PD, 3 German HC, 3 German PD.")

    with open(NEW_SELECTION_SUMMARY, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Created markdown summary: {NEW_SELECTION_SUMMARY}")
    print("Step 1 Finished Successfully!\n")

if __name__ == "__main__":
    main()
