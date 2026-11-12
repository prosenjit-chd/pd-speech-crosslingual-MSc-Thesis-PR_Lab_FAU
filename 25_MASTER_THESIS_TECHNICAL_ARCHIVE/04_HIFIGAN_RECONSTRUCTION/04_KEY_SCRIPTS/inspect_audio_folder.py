#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inspect_audio_folder.py

Purpose:
Inspects all WAV files in a specified folder (recursively) and writes a summary CSV
documenting sample rate, duration, channels, sample count, peak amplitude, and RMS energy.
"""

import os
import sys
import csv
import argparse
import traceback
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

def inspect_file(file_path: Path) -> dict:
    """
    Reads a WAV file and returns metadata and statistics.
    """
    info = {
        "filename": file_path.name,
        "path": str(file_path),
        "sample_rate": None,
        "duration_sec": 0.0,
        "num_samples": 0,
        "channels": 0,
        "max_abs_amplitude": 0.0,
        "rms": 0.0,
        "status": "failed"
    }
    
    try:
        # Load audio with original sample rate and channels
        y, sr = librosa.load(file_path, sr=None, mono=False)
        
        info["sample_rate"] = int(sr)
        info["channels"] = int(y.shape[0]) if y.ndim > 1 else 1
        info["num_samples"] = int(y.shape[-1])
        
        # Calculate duration
        duration = librosa.get_duration(y=y, sr=sr)
        info["duration_sec"] = float(duration)
        
        # Statistics
        max_abs = np.max(np.abs(y))
        info["max_abs_amplitude"] = float(max_abs)
        
        # Root Mean Square (RMS) energy
        rms_val = np.sqrt(np.mean(y ** 2))
        info["rms"] = float(rms_val)
        
        info["status"] = "success"
        
    except Exception as e:
        print(f"WARNING: Failed to inspect {file_path}: {e}", file=sys.stderr)
        info["status"] = f"failed ({type(e).__name__})"
        
    return info

def main():
    parser = argparse.ArgumentParser(description="Inspect audio folder and output a CSV report.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing WAV files to inspect.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to write the CSV report.")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_csv).resolve()
    
    print("=" * 60)
    print("Starting Audio Folder Inspection")
    print(f"Input Directory:  {input_path}")
    print(f"Output CSV:       {output_path}")
    print("=" * 60)
    
    if not input_path.exists():
        print(f"ERROR: Input directory '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    # Ensure parent output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Recursively find all WAV files
    wav_files = sorted(list(input_path.rglob("*.wav")))
    
    if not wav_files:
        print(f"No WAV files found in '{input_path}'.")
        # Still write header to CSV
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "path", "sample_rate", "duration_sec", 
                "num_samples", "channels", "max_abs_amplitude", "rms", "status"
            ])
        print(f"Empty report written to: {output_path}")
        sys.exit(0)
        
    print(f"Found {len(wav_files)} WAV files to inspect.")
    
    results = []
    for idx, f_path in enumerate(wav_files, 1):
        print(f"[{idx}/{len(wav_files)}] Inspecting: {f_path.name}")
        info = inspect_file(f_path)
        results.append(info)
        
    # Write report
    fieldnames = [
        "filename", "path", "sample_rate", "duration_sec", 
        "num_samples", "channels", "max_abs_amplitude", "rms", "status"
    ]
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"Inspection complete. Report saved to: {output_path}")

if __name__ == "__main__":
    main()
