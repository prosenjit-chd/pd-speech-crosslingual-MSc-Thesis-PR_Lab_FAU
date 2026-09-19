#!/usr/bin/env python3
"""
07_pathology_sanity_check.py
Exploratory Pathology Sanity Check for Spanish -> German Pilot Recordings (3 HC vs 3 PD).

Implements Section 28:
- Extracts 17 acoustic and phonatory features:
  F0 mean, F0 median, F0 SD, F0 range, voiced fraction, pause ratio, pause count,
  mean pause duration, RMS, spectral centroid, spectral bandwidth, spectral rolloff,
  HNR, jitter, shimmer, MFCC summaries (1-4).
- Compares original Spanish vs final regenerated German outputs.
- Computes HC vs PD feature directions (PD > HC vs HC > PD) in both domains.
- Descriptive only (3 HC vs 3 PD); no unwarranted statistical claims.
- Saves artifacts to:
  15_pathology_sanity_check/
  17_tables/pathology_feature_preservation.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torchaudio
import torch

WORKSPACE = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement")
PATHOLOGY_DIR = WORKSPACE / "15_pathology_sanity_check"
TABLES_DIR = WORKSPACE / "17_tables"
FINAL_AUDIO_DIR = WORKSPACE / "10_final_regeneration" / "spanish_to_german"
PILOT_CSV = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\04_pilot_selection\pilot_12_selection.csv")

def extract_features(filepath):
    y, sr = sf.read(str(filepath))
    if y.ndim > 1:
        y = y[:, 0]
        
    # Resample to 16kHz
    if sr != 16000:
        t_y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        y = resampler(t_y).squeeze(0).numpy()
        sr = 16000
        
    hop_length = int(sr * 0.01)   # 10ms
    frame_length = int(sr * 0.03) # 30ms
    
    # 1. Energy & RMS
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms_frames))
    rms_std = float(np.std(rms_frames))
    peak = float(np.max(np.abs(y)))
    
    # 2. Timing & Pauses
    silence_thresh = max(1e-5, peak * 0.018)
    speech_mask = rms_frames >= silence_thresh
    pause_mask = ~speech_mask
    pause_ratio = float(np.mean(pause_mask))
    
    pause_runs = []
    curr_len = 0
    for is_pause in pause_mask:
        if is_pause:
            curr_len += 1
        else:
            if curr_len >= 10:  # >=100ms
                pause_runs.append(curr_len * 0.01)
            curr_len = 0
    if curr_len >= 10:
        pause_runs.append(curr_len * 0.01)
    pause_count = len(pause_runs)
    mean_pause_dur = float(np.mean(pause_runs)) if pause_count > 0 else 0.0
    
    # 3. F0 / Pitch (via pyin)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=60, fmax=450, sr=sr, hop_length=hop_length
    )
    voiced_f0 = f0[~np.isnan(f0) & (f0 > 0)]
    voiced_fraction = float(len(voiced_f0) / len(f0)) if len(f0) > 0 else 0.0
    
    if len(voiced_f0) > 10:
        f0_mean = float(np.mean(voiced_f0))
        f0_median = float(np.median(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_range = float(np.percentile(voiced_f0, 95) - np.percentile(voiced_f0, 5))
        
        periods = 1.0 / voiced_f0
        period_diffs = np.abs(np.diff(periods))
        jitter_local = float(np.mean(period_diffs) / np.mean(periods)) if np.mean(periods) > 0 else 0.0
    else:
        f0_mean, f0_median, f0_std, f0_range, jitter_local = 0.0, 0.0, 0.0, 0.0, 0.0
        
    # 4. Spectral Features
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    
    spec_cent_mean = float(np.mean(cent))
    spec_bw_mean = float(np.mean(bw))
    spec_rolloff_mean = float(np.mean(rolloff))
    
    # 5. Harmonics-to-Noise Ratio (HNR proxy via HPSS)
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.sum(y_harm ** 2)
    perc_energy = np.sum(y_perc ** 2)
    hnr_db = float(10 * np.log10(max(1e-6, harm_energy) / max(1e-6, perc_energy)))
    
    # 6. Shimmer (local relative difference in voiced speech RMS)
    if np.any(speech_mask):
        speech_rms = rms_frames[speech_mask]
        if len(speech_rms) > 10:
            rms_diffs = np.abs(np.diff(speech_rms))
            shimmer_local = float(np.mean(rms_diffs) / np.mean(speech_rms))
        else:
            shimmer_local = 0.0
    else:
        shimmer_local = 0.0
        
    # 7. MFCC 1-4
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_means = np.mean(mfccs, axis=1)
    
    return {
        "duration": round(len(y) / sr, 3),
        "f0_mean_hz": round(f0_mean, 2),
        "f0_median_hz": round(f0_median, 2),
        "f0_std_hz": round(f0_std, 2),
        "f0_range_hz": round(f0_range, 2),
        "voiced_fraction": round(voiced_fraction, 4),
        "pause_ratio": round(pause_ratio, 4),
        "pause_count": pause_count,
        "mean_pause_dur_s": round(mean_pause_dur, 3),
        "rms_mean": round(rms_mean, 4),
        "rms_std": round(rms_std, 4),
        "spec_centroid_hz": round(spec_cent_mean, 1),
        "spec_bandwidth_hz": round(spec_bw_mean, 1),
        "spec_rolloff_hz": round(spec_rolloff_mean, 1),
        "hnr_db": round(hnr_db, 2),
        "jitter_local": round(jitter_local, 5),
        "shimmer_local": round(shimmer_local, 5),
        "mfcc_1": round(float(mfcc_means[0]), 2),
        "mfcc_2": round(float(mfcc_means[1]), 2),
        "mfcc_3": round(float(mfcc_means[2]), 2),
        "mfcc_4": round(float(mfcc_means[3]), 2)
    }

def main():
    PATHOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    df_pilot = pd.read_csv(PILOT_CSV)
    df_es = df_pilot[df_pilot["language"] == "Spanish"].copy()
    
    orig_rows = []
    final_rows = []
    
    print("=== Extracting Pathology Features for 6 Spanish Speakers (3 HC vs 3 PD) ===")
    for _, row in df_es.iterrows():
        sid = row["safe_id"].replace(".wav", "")
        diag = row["diagnosis"]
        orig_audio = Path(row["audio_path"])
        final_audio = FINAL_AUDIO_DIR / f"{sid}_final_to_German.wav"
        
        print(f"[{sid}] ({diag}) Extracting features...")
        feat_orig = extract_features(orig_audio)
        feat_orig["safe_id"] = sid
        feat_orig["diagnosis"] = diag
        orig_rows.append(feat_orig)
        
        feat_final = extract_features(final_audio)
        feat_final["safe_id"] = sid
        feat_final["diagnosis"] = diag
        final_rows.append(feat_final)
        
    df_orig = pd.DataFrame(orig_rows)
    df_final = pd.DataFrame(final_rows)
    
    df_orig.to_csv(PATHOLOGY_DIR / "pathology_features_original_spanish.csv", index=False)
    df_final.to_csv(PATHOLOGY_DIR / "pathology_features_final_german.csv", index=False)
    
    # Feature direction comparison
    feature_keys = [
        "f0_mean_hz", "f0_median_hz", "f0_std_hz", "f0_range_hz",
        "voiced_fraction", "pause_ratio", "pause_count", "mean_pause_dur_s",
        "rms_mean", "spec_centroid_hz", "spec_bandwidth_hz", "spec_rolloff_hz",
        "hnr_db", "jitter_local", "shimmer_local",
        "mfcc_1", "mfcc_2"
    ]
    
    direction_records = []
    matches = 0
    total_feats = len(feature_keys)
    
    for k in feature_keys:
        # Original Spanish direction
        hc_orig = df_orig[df_orig["diagnosis"] == "HC"][k].mean()
        pd_orig = df_orig[df_orig["diagnosis"] == "PD"][k].mean()
        orig_dir = "PD > HC" if pd_orig > hc_orig else ("PD < HC" if pd_orig < hc_orig else "EQUAL")
        
        # Final German direction
        hc_fin = df_final[df_final["diagnosis"] == "HC"][k].mean()
        pd_fin = df_final[df_final["diagnosis"] == "PD"][k].mean()
        fin_dir = "PD > HC" if pd_fin > hc_fin else ("PD < HC" if pd_fin < hc_fin else "EQUAL")
        
        agree = (orig_dir == fin_dir)
        if agree:
            matches += 1
            
        direction_records.append({
            "feature": k,
            "orig_HC_mean": round(hc_orig, 3),
            "orig_PD_mean": round(pd_orig, 3),
            "orig_direction": orig_dir,
            "final_HC_mean": round(hc_fin, 3),
            "final_PD_mean": round(pd_fin, 3),
            "final_direction": fin_dir,
            "direction_preserved": agree
        })
        
    df_dir = pd.DataFrame(direction_records)
    dir_csv = PATHOLOGY_DIR / "pathology_direction_preservation.csv"
    df_dir.to_csv(dir_csv, index=False)
    df_dir.to_csv(TABLES_DIR / "pathology_feature_preservation.csv", index=False)
    
    pct_preserved = round(matches / total_feats * 100, 1)
    
    report_md = f"""# Exploratory Pathology Sanity Check: Spanish Source -> Converted German

> [!NOTE]
> Sample size: N = 6 (3 HC vs 3 PD). This sanity check is descriptive and exploratory only.
> It evaluates whether acoustic and phonatory distinctions present in the source recordings
> are preserved in their directional tendency after true cross-language voice conversion.

## Directional Agreement Summary
- Total Acoustic Features Evaluated: **{total_feats}**
- Features with Preserved Effect Direction: **{matches}/{total_feats} ({pct_preserved}%)**

## Feature Direction Matrix
{df_dir.to_markdown(index=False)}
"""
    (PATHOLOGY_DIR / "pathology_sanity_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nPathology Direction Preserved: {matches}/{total_feats} ({pct_preserved}%)")
    print(f"Report saved to: {PATHOLOGY_DIR / 'pathology_sanity_report.md'}")

if __name__ == "__main__":
    main()
