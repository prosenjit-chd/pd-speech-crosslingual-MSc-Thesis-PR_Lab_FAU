#!/usr/bin/env python3
"""
02_extract_reference_audio.py
Deterministic Reference Audio Extraction for Spanish -> German Pilot Speakers.

Implements Section 12:
- R1: Clean 10-second voiced reference
- R2: Clean 15-second voiced reference
- R3: Two clean non-overlapping reference segments (5-second each) for multi-segment conditioning

Speakers evaluated (all 6 Spanish pilot recordings):
- SP_HC_001, SP_HC_002, SP_HC_003
- SP_PD_001, SP_PD_002, SP_PD_003

All reference selections are completely deterministic and quality-controlled:
- Maximize voiced fraction and signal-to-silence ratio
- Verify absence of clipping
- Apply 25ms cosine fade in/out to prevent boundary clicks.
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd

# Paths
WORKSPACE = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement")
REF_DIR = WORKSPACE / "03_reference_audio_analysis"
CLEAN_DIR = REF_DIR / "clean_references"
PILOT_CSV = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\04_pilot_selection\pilot_12_selection.csv")

def compute_audio_properties(y: np.ndarray, sr: int) -> dict:
    duration = len(y) / sr
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    is_clipped = peak >= 0.999
    
    # Energy-based silence & voicing approximation
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    num_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    
    frame_rms = []
    for i in range(num_frames):
        st = i * hop_len
        frame = y[st:st + frame_len]
        frame_rms.append(np.sqrt(np.mean(frame ** 2)))
    frame_rms = np.array(frame_rms)
    
    silence_threshold = max(0.005, 0.1 * rms)
    silent_frames = np.sum(frame_rms < silence_threshold)
    silence_ratio = float(silent_frames / len(frame_rms)) if len(frame_rms) > 0 else 0.0
    voiced_ratio = float(1.0 - silence_ratio)
    
    return {
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "is_clipped": is_clipped,
        "silence_ratio": silence_ratio,
        "voiced_ratio": voiced_ratio
    }

def find_best_segment(y: np.ndarray, sr: int, target_sec: float, exclude_ranges=None) -> tuple:
    """
    Deterministically scan audio using sliding window to find optimal voiced segment.
    """
    target_samples = int(target_sec * sr)
    if len(y) <= target_samples:
        return y.copy(), 0.0, len(y) / sr
    
    hop = int(0.5 * sr)  # 0.5s step
    best_score = -1.0
    best_start = 0
    
    for start in range(0, len(y) - target_samples + 1, hop):
        end = start + target_samples
        
        # Check collision with excluded ranges
        if exclude_ranges:
            overlap = False
            for ex_st, ex_end in exclude_ranges:
                if not (end <= ex_st or start >= ex_end):
                    overlap = True
                    break
            if overlap:
                continue
        
        seg = y[start:end]
        props = compute_audio_properties(seg, sr)
        
        if props["is_clipped"]:
            score = 0.0
        else:
            rms_factor = min(1.0, props["rms"] / 0.04)
            score = props["voiced_ratio"] * (1.0 - props["silence_ratio"]) * rms_factor
        
        if score > best_score:
            best_score = score
            best_start = start
            
    best_seg = y[best_start:best_start + target_samples].copy()
    
    # 25ms cosine fade in/out
    fade_len = int(0.025 * sr)
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
    fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
    best_seg[:fade_len] *= fade_in
    best_seg[-fade_len:] *= fade_out
    
    return best_seg, best_start / sr, (best_start + target_samples) / sr

def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    
    df_pilot = pd.read_csv(PILOT_CSV)
    df_spanish = df_pilot[df_pilot["language"] == "Spanish"].copy()
    
    records = []
    
    print("=== Extracting Deterministic Reference Audio Variants (R1, R2, R3) ===")
    for _, row in df_spanish.iterrows():
        sid = row["safe_id"]
        base_id = sid.replace(".wav", "")
        diag = row["diagnosis"]
        audio_path = Path(row["audio_path"])
        
        y, sr = sf.read(str(audio_path))
        if y.ndim > 1:
            y = y[:, 0]
            
        full_props = compute_audio_properties(y, sr)
        total_dur = len(y) / sr
        
        # 1. R1: 10-second reference
        r1_seg, r1_st, r1_end = find_best_segment(y, sr, target_sec=10.0)
        r1_path = CLEAN_DIR / f"{base_id}_R1_10s.wav"
        sf.write(str(r1_path), r1_seg, sr)
        r1_props = compute_audio_properties(r1_seg, sr)
        
        # 2. R2: 15-second reference (or max available duration minus padding)
        target_r2_sec = min(15.0, max(10.5, total_dur - 0.2))
        r2_seg, r2_st, r2_end = find_best_segment(y, sr, target_sec=target_r2_sec)
        r2_path = CLEAN_DIR / f"{base_id}_R2_15s.wav"
        sf.write(str(r2_path), r2_seg, sr)
        r2_props = compute_audio_properties(r2_seg, sr)
        
        # 3. R3: Two 5-second non-overlapping clean reference segments
        r3_seg_a, r3a_st, r3a_end = find_best_segment(y, sr, target_sec=5.0)
        r3_path_a = CLEAN_DIR / f"{base_id}_R3_part1_5s.wav"
        sf.write(str(r3_path_a), r3_seg_a, sr)
        
        # Find second 5s segment non-overlapping with part 1
        exclude = [(int(r3a_st * sr), int(r3a_end * sr))]
        r3_seg_b, r3b_st, r3b_end = find_best_segment(y, sr, target_sec=5.0, exclude_ranges=exclude)
        r3_path_b = CLEAN_DIR / f"{base_id}_R3_part2_5s.wav"
        sf.write(str(r3_path_b), r3_seg_b, sr)
        
        records.append({
            "safe_id": base_id,
            "diagnosis": diag,
            "total_duration": total_dur,
            "R1_path": str(r1_path),
            "R1_start": r1_st,
            "R1_end": r1_end,
            "R1_rms": r1_props["rms"],
            "R1_voiced_ratio": r1_props["voiced_ratio"],
            "R2_path": str(r2_path),
            "R2_start": r2_st,
            "R2_end": r2_end,
            "R2_rms": r2_props["rms"],
            "R2_voiced_ratio": r2_props["voiced_ratio"],
            "R3_part1_path": str(r3_path_a),
            "R3_part1_start": r3a_st,
            "R3_part1_end": r3a_end,
            "R3_part2_path": str(r3_path_b),
            "R3_part2_start": r3b_st,
            "R3_part2_end": r3b_end
        })
        print(f"[{base_id}] R1: {r1_st:.1f}-{r1_end:.1f}s | R2: {r2_st:.1f}-{r2_end:.1f}s | R3: ({r3a_st:.1f}-{r3a_end:.1f}s, {r3b_st:.1f}-{r3b_end:.1f}s)")
        
    df_refs = pd.DataFrame(records)
    out_csv = REF_DIR / "reference_audio_strategies.csv"
    df_refs.to_csv(out_csv, index=False)
    print(f"Saved reference audio strategy manifest to: {out_csv}")

if __name__ == "__main__":
    main()
