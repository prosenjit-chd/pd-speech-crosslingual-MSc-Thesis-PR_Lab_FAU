#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prepare_hifigan_pilot_audio.py

Purpose:
Preprocesses raw audio files for HiFi-GAN synthesis/reconstruction:
- Converts input audio to mono.
- Downsamples to 22050 Hz.
- Normalizes amplitude (peak normalization).
- Saves short, clean WAV files in the target directory.
- Logs processing metadata to a CSV file.
"""

import os
import sys
import csv
import traceback
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "voice_conversion" / "input_pilot"
OUTPUT_DIR = PROJECT_ROOT / "voice_conversion" / "input_pilot_22050"
LOG_DIR = PROJECT_ROOT / "voice_conversion" / "logs"
CSV_LOG_PATH = LOG_DIR / "pilot_preprocessing_log.csv"

def preprocess_audio(file_path: Path, output_dir: Path) -> dict:
    """
    Preprocesses a single audio file:
    - Loads audio (keeps original sample rate and channels initially)
    - Downsamples to 22050 Hz, converts to mono, and normalizes amplitude
    - Saves output WAV
    """
    rel_path = file_path.relative_to(INPUT_DIR)
    out_file_path = output_dir / rel_path
    
    # Ensure nested subdirectories are created if they exist in pilot
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize info dict
    file_info = {
        "filename": file_path.name,
        "relative_path": str(rel_path),
        "original_sr": None,
        "output_sr": 22050,
        "duration_sec": 0.0,
        "original_channels": 1,
        "max_abs_amplitude": 0.0,
        "status": "failed",
        "error_message": ""
    }
    
    try:
        # Load audio with original sample rate, preserving channels (mono=False)
        y, orig_sr = librosa.load(file_path, sr=None, mono=False)
        
        file_info["original_sr"] = orig_sr
        
        # Determine channels
        if y.ndim > 1:
            file_info["original_channels"] = y.shape[0]
            # Convert to mono
            y_mono = librosa.to_mono(y)
        else:
            file_info["original_channels"] = 1
            y_mono = y
            
        # Resample to 22050 Hz if necessary
        if orig_sr != 22050:
            y_resampled = librosa.resample(y_mono, orig_sr=orig_sr, target_sr=22050)
        else:
            y_resampled = y_mono
            
        # Peak normalization
        max_abs = np.max(np.abs(y_resampled))
        file_info["max_abs_amplitude"] = float(max_abs)
        if max_abs > 0:
            y_normalized = y_resampled / max_abs
        else:
            y_normalized = y_resampled
            
        # Calculate duration
        duration = librosa.get_duration(y=y_normalized, sr=22050)
        file_info["duration_sec"] = float(duration)
        
        # Save output WAV file
        sf.write(out_file_path, y_normalized, 22050, subtype='PCM_16')
        
        file_info["status"] = "success"
        
        print(f"Processed: {file_info['relative_path']}")
        print(f"  Original SR: {orig_sr} Hz | Channels: {file_info['original_channels']}")
        print(f"  Duration: {duration:.2f} sec | Saved: {out_file_path}\n")
        
    except Exception as e:
        err_msg = f"Error processing {file_path}: {e}"
        print(f"WARNING: {err_msg}", file=sys.stderr)
        traceback.print_exc()
        file_info["status"] = "failed"
        file_info["error_message"] = str(e)
        
    return file_info

def main():
    print("=" * 60)
    print("Starting HiFi-GAN Pilot Audio Preprocessing")
    print(f"Input Directory:  {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory {INPUT_DIR} does not exist.")
        sys.exit(1)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Recursively find all WAV files
    wav_files = sorted(list(INPUT_DIR.rglob("*.wav")))
    
    if not wav_files:
        print(f"No WAV files found in {INPUT_DIR}.")
        print("Please place pilot WAV files there first.")
        # We will still create an empty CSV log with headers and a descriptive note row
        with open(CSV_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "relative_path", "original_sr", "output_sr", 
                "duration_sec", "original_channels", "max_abs_amplitude", 
                "status", "error_message"
            ])
            writer.writerow([
                "NO_FILES_FOUND", "N/A", "N/A", "N/A", 
                "N/A", "N/A", "N/A", 
                "empty", "No pilot WAV files are present in input_pilot yet"
            ])
        sys.exit(0)
        
    results = []
    for f_path in wav_files:
        info = preprocess_audio(f_path, OUTPUT_DIR)
        results.append(info)
        
    # Write to CSV log
    with open(CSV_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename", "relative_path", "original_sr", "output_sr", 
            "duration_sec", "original_channels", "max_abs_amplitude", 
            "status", "error_message"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"Preprocessing completed. Log written to: {CSV_LOG_PATH}")

if __name__ == "__main__":
    main()
