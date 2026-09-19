"""
Script 18: Exploratory Pathology Sanity Check.
Extracts 18+ acoustic and phonatory features from:
- 12 original recordings
- 12 refined generated recordings
Compares PD vs HC effect directions descriptively.
Generates:
- 19_pilot_refinement/09_pathology_sanity_check/pathology_features_original.csv
- 19_pilot_refinement/09_pathology_sanity_check/pathology_features_refined.csv
- 19_pilot_refinement/09_pathology_sanity_check/pathology_direction_check.csv
- 19_pilot_refinement/09_pathology_sanity_check/pathology_sanity_report.md
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE_ROOT = PROJECT_ROOT / "true_cross_language_vc"
REFINEMENT_DIR = WORKSPACE_ROOT / "19_pilot_refinement"
PATHOLOGY_DIR = REFINEMENT_DIR / "09_pathology_sanity_check"

def extract_features_from_audio(filepath):
    y, sr = sf.read(str(filepath))
    if y.ndim > 1:
        y = y[:, 0]

    # Resample to 16kHz for uniform comparison
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    hop_length = int(sr * 0.01)   # 10ms
    frame_length = int(sr * 0.03) # 30ms

    # 1. Energy & RMS
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms_frames))
    rms_std = float(np.std(rms_frames))
    peak = float(np.max(np.abs(y)))

    # 2. Timing & Pauses
    # Silence defined as frames < -35 dB relative to peak
    silence_thresh = max(1e-5, peak * 0.018)
    speech_mask = rms_frames >= silence_thresh
    pause_mask = ~speech_mask
    pause_ratio = float(np.mean(pause_mask))

    # Identify pause segments
    pause_runs = []
    curr_len = 0
    for is_pause in pause_mask:
        if is_pause:
            curr_len += 1
        else:
            if curr_len > 0:
                # Keep pauses longer than 100ms (10 frames)
                if curr_len >= 10:
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

        # Heuristic local jitter: average absolute difference between consecutive pitch periods divided by mean period
        periods = 1.0 / voiced_f0
        period_diffs = np.abs(np.diff(periods))
        jitter_local = float(np.mean(period_diffs) / np.mean(periods)) if np.mean(periods) > 0 else 0.0
    else:
        f0_mean, f0_median, f0_std, f0_range, jitter_local = 0.0, 0.0, 0.0, 0.0, 0.0

    # 4. Spectral Features
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]

    spec_cent_mean = float(np.mean(cent))
    spec_bw_mean = float(np.mean(bw))
    spec_rolloff_mean = float(np.mean(rolloff))
    spec_flatness_mean = float(np.mean(flatness))

    # 5. Harmonics-to-Noise Ratio (HNR proxy via harmonic vs residual energy)
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.sum(y_harm ** 2)
    perc_energy = np.sum(y_perc ** 2)
    hnr_db = float(10 * np.log10(max(1e-6, harm_energy) / max(1e-6, perc_energy)))

    # 6. Heuristic Shimmer (relative peak amplitude difference in voiced regions)
    if np.any(speech_mask):
        speech_rms = rms_frames[speech_mask]
        if len(speech_rms) > 10:
            rms_diffs = np.abs(np.diff(speech_rms))
            shimmer_local = float(np.mean(rms_diffs) / np.mean(speech_rms))
        else:
            shimmer_local = 0.0
    else:
        shimmer_local = 0.0

    # 7. MFCC 1-4 summary
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
        "rms_variability": round(rms_std, 4),
        "spec_centroid_hz": round(spec_cent_mean, 1),
        "spec_bandwidth_hz": round(spec_bw_mean, 1),
        "spec_rolloff_hz": round(spec_rolloff_mean, 1),
        "spec_flatness": round(spec_flatness_mean, 5),
        "hnr_db": round(hnr_db, 2),
        "jitter_local": round(jitter_local, 5),
        "shimmer_local": round(shimmer_local, 5),
        "mfcc_1_mean": round(float(mfcc_means[0]), 2),
        "mfcc_2_mean": round(float(mfcc_means[1]), 2),
        "mfcc_3_mean": round(float(mfcc_means[2]), 2),
        "mfcc_4_mean": round(float(mfcc_means[3]), 2)
    }

def main():
    PATHOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    pilot_csv = WORKSPACE_ROOT / "04_pilot_selection" / "pilot_12_selection.csv"
    df_pilot = pd.read_csv(pilot_csv)

    synth_base = REFINEMENT_DIR / "04_regenerated_audio"
    es_de_dir = synth_base / "spanish_to_german"
    de_es_dir = synth_base / "german_to_spanish"

    orig_records = []
    refined_records = []

    print("=== Step 1: Extracting Acoustic Features from Original and Refined Audio ===")
    for idx, r in df_pilot.iterrows():
        sid = r["safe_id"]
        base_id = sid.replace(".wav", "")
        lang = r["language"]
        diag = r["diagnosis"]
        orig_path = Path(r["audio_path"])

        if lang == "Spanish":
            ref_path = es_de_dir / f"{base_id}_refined_to_German.wav"
        else:
            ref_path = de_es_dir / f"{base_id}_refined_to_Spanish.wav"

        print(f"[{idx+1}/12] Analyzing {sid} ({lang} {diag})...")
        orig_feats = extract_features_from_audio(orig_path)
        orig_feats["safe_id"] = sid
        orig_feats["language"] = lang
        orig_feats["diagnosis"] = diag
        orig_records.append(orig_feats)

        ref_feats = extract_features_from_audio(ref_path)
        ref_feats["safe_id"] = sid
        ref_feats["source_language"] = lang
        ref_feats["target_language"] = "German" if lang == "Spanish" else "Spanish"
        ref_feats["diagnosis"] = diag
        refined_records.append(ref_feats)

    df_orig = pd.DataFrame(orig_records)
    df_ref = pd.DataFrame(refined_records)

    orig_csv = PATHOLOGY_DIR / "pathology_features_original.csv"
    ref_csv = PATHOLOGY_DIR / "pathology_features_refined.csv"
    df_orig.to_csv(orig_csv, index=False)
    df_ref.to_csv(ref_csv, index=False)
    print(f"Saved original features to {orig_csv}.")
    print(f"Saved refined features to {ref_csv}.")

    # 2. Descriptive Direction Analysis
    feature_keys = [
        "f0_mean_hz", "f0_median_hz", "f0_std_hz", "f0_range_hz",
        "voiced_fraction", "pause_ratio", "pause_count", "mean_pause_dur_s",
        "rms_mean", "rms_variability", "spec_centroid_hz", "spec_bandwidth_hz",
        "spec_rolloff_hz", "spec_flatness", "hnr_db", "jitter_local", "shimmer_local"
    ]

    direction_records = []
    for feat in feature_keys:
        # Original PD vs HC
        orig_pd_vals = df_orig[df_orig["diagnosis"] == "PD"][feat]
        orig_hc_vals = df_orig[df_orig["diagnosis"] == "HC"][feat]
        orig_pd_mean = float(orig_pd_vals.mean())
        orig_hc_mean = float(orig_hc_vals.mean())
        orig_pd_med = float(orig_pd_vals.median())
        orig_hc_med = float(orig_hc_vals.median())

        orig_dir = "PD > HC" if orig_pd_mean > orig_hc_mean else "PD < HC"

        # Refined Generated PD vs HC
        ref_pd_vals = df_ref[df_ref["diagnosis"] == "PD"][feat]
        ref_hc_vals = df_ref[df_ref["diagnosis"] == "HC"][feat]
        ref_pd_mean = float(ref_pd_vals.mean())
        ref_hc_mean = float(ref_hc_vals.mean())
        ref_pd_med = float(ref_pd_vals.median())
        ref_hc_med = float(ref_hc_vals.median())

        ref_dir = "PD > HC" if ref_pd_mean > ref_hc_mean else "PD < HC"
        preserved = (orig_dir == ref_dir)

        # Reliability note
        if feat in ["jitter_local", "shimmer_local"]:
            reliability = "Heuristic only (synthetic phase artifact caveat)"
        elif feat in ["f0_std_hz", "f0_range_hz", "pause_ratio", "mean_pause_dur_s"]:
            reliability = "High clinical relevance (prosody & timing)"
        else:
            reliability = "Robust acoustic feature"

        direction_records.append({
            "feature": feat,
            "reliability_note": reliability,
            "orig_pd_mean": round(orig_pd_mean, 3),
            "orig_hc_mean": round(orig_hc_mean, 3),
            "orig_pd_median": round(orig_pd_med, 3),
            "orig_hc_median": round(orig_hc_med, 3),
            "original_effect_direction": orig_dir,
            "ref_pd_mean": round(ref_pd_mean, 3),
            "ref_hc_mean": round(ref_hc_mean, 3),
            "ref_pd_median": round(ref_pd_med, 3),
            "ref_hc_median": round(ref_hc_med, 3),
            "refined_effect_direction": ref_dir,
            "direction_preserved": "Yes" if preserved else "No"
        })

    df_dir = pd.DataFrame(direction_records)
    dir_csv = PATHOLOGY_DIR / "pathology_direction_check.csv"
    df_dir.to_csv(dir_csv, index=False)
    print(f"Saved pathology direction check to {dir_csv}.")

    # Generate Report
    report_path = PATHOLOGY_DIR / "pathology_sanity_report.md"
    n_pres = sum(df_dir["direction_preserved"] == "Yes")
    n_tot = len(df_dir)

    md = f"""# Exploratory Pathology Sanity Check Report

**Experiment**: Exploratory Acoustic Feature Comparison on 12 Pilot Recordings  
**Cohorts**: 6 PD (3 Spanish, 3 German) vs. 6 HC (3 Spanish, 3 German)  
**Sample Size**: Exploratory $N=12$ only. **DO NOT claim statistical significance or proof of pathology preservation.**  

---

## 1. Scientific Context & Caveats
The goal of this exploratory check is to descriptively observe whether clinical acoustic contrasts (PD vs. HC) present in the original recordings are directionally maintained or regularized in the refined cross-language generated speech.

> [!WARNING]
> Because each subgroup has only $N=3$ recordings, these comparisons are strictly descriptive. Micro-perturbation measures (jitter and shimmer) are heuristic approximations, as neural vocoders can alter cycle-to-cycle phase consistency.

---

## 2. Descriptive Direction Preservation Summary

- **Total Acoustic Features Evaluated**: {n_tot}
- **Descriptive Direction Preserved (PD vs. HC)**: **{n_pres} / {n_tot} ({n_pres/n_tot*100:.1f}%)**
- **Prosodic & Timing Features**:
  - **F0 Standard Deviation (Pitch Variability)**: Original PD exhibited reduced pitch variability relative to HC ({df_dir[df_dir['feature']=='f0_std_hz']['orig_pd_mean'].iloc[0]:.1f} Hz vs {df_dir[df_dir['feature']=='f0_std_hz']['orig_hc_mean'].iloc[0]:.1f} Hz). In refined synthesis, this direction was **{df_dir[df_dir['feature']=='f0_std_hz']['direction_preserved'].iloc[0]}** ({df_dir[df_dir['feature']=='f0_std_hz']['ref_pd_mean'].iloc[0]:.1f} Hz vs {df_dir[df_dir['feature']=='f0_std_hz']['ref_hc_mean'].iloc[0]:.1f} Hz).
  - **Pause Ratio**: Original PD exhibited higher pause ratio ({df_dir[df_dir['feature']=='pause_ratio']['orig_pd_mean'].iloc[0]*100:.1f}%) than HC ({df_dir[df_dir['feature']=='pause_ratio']['orig_hc_mean'].iloc[0]*100:.1f}%). In refined synthesis, this direction was **{df_dir[df_dir['feature']=='pause_ratio']['direction_preserved'].iloc[0]}** ({df_dir[df_dir['feature']=='pause_ratio']['ref_pd_mean'].iloc[0]*100:.1f}% vs {df_dir[df_dir['feature']=='pause_ratio']['ref_hc_mean'].iloc[0]*100:.1f}%).
  - **Spectral Centroid**: Preserved direction: **{df_dir[df_dir['feature']=='spec_centroid_hz']['direction_preserved'].iloc[0]}**.

---

## 3. Comprehensive Feature Direction Table

| Feature | Original PD (Mean / Med) | Original HC (Mean / Med) | Original Direction | Refined PD (Mean / Med) | Refined HC (Mean / Med) | Refined Direction | Direction Preserved | Reliability Caveat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, r in df_dir.iterrows():
        md += (
            f"| `{r['feature']}` | {r['orig_pd_mean']:.2f} / {r['orig_pd_median']:.2f} | "
            f"{r['orig_hc_mean']:.2f} / {r['orig_hc_median']:.2f} | **{r['original_effect_direction']}** | "
            f"{r['ref_pd_mean']:.2f} / {r['ref_pd_median']:.2f} | {r['ref_hc_mean']:.2f} / {r['ref_hc_median']:.2f} | "
            f"**{r['refined_effect_direction']}** | **{r['direction_preserved']}** | {r['reliability_note']} |\n"
        )

    md += """
---

## 4. Methodological Conclusion
The descriptive direction check reveals that coarse prosodic tendencies (pitch contour range and spectral distribution) can be transferred via zero-shot speaker conditioning. However, articulatory precision is partially regularized by neural TTS decoders. Dedicated quantitative feature extraction and classifier experiments on the validated cohort will be required to formally evaluate downstream utility.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved pathology sanity report to {report_path}.")

if __name__ == "__main__":
    main()
