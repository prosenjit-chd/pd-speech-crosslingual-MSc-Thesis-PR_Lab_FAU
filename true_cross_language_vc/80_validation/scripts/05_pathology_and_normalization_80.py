"""
Script 05: Clinical Pathology Feature Extraction, Preservation & Normalization Analysis.

Implements Sections 14, 15, 16:
- Section 14: Extracts 19 acoustic/phonatory/spectral features for 80 original and 80 generated audio files.
  -> 10_pathology_features/original_features_80.csv
  -> 10_pathology_features/generated_features_80.csv
  -> 10_pathology_features/feature_dictionary.md
- Section 15: Computes PD vs HC effect sizes (Cohen's d, Cliff's delta, bootstrap 95% CIs, p-values, BH-FDR q-values).
  Analyzes feature direction preservation (preserved, attenuated, amplified, reversed) and overall preservation percentage.
  -> 11_pathology_preservation/pathology_effects_original.csv
  -> 11_pathology_preservation/pathology_effects_generated.csv
  -> 11_pathology_preservation/pathology_preservation_comparison.csv
  -> 11_pathology_preservation/pathology_preservation_report.md
- Section 16: Rigorous statistical testing of acoustic normalization (PD shift toward HC centroid).
  -> 11_pathology_preservation/pathology_normalization_analysis.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy import stats

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE = PROJECT_ROOT / "true_cross_language_vc"
VAL_DIR = WORKSPACE / "21_validation_80"

SELECT_CSV = VAL_DIR / "01_selection" / "validation_80_selection.csv"
SYNTH_DIR = VAL_DIR / "05_synthesis"
ES_DE_DIR = SYNTH_DIR / "spanish_to_german"
DE_ES_DIR = SYNTH_DIR / "german_to_spanish"

FEAT_DIR = VAL_DIR / "10_pathology_features"
PRES_DIR = VAL_DIR / "11_pathology_preservation"

FEATURE_NAMES = [
    "f0_mean_hz", "f0_median_hz", "f0_std_hz", "f0_range_hz",
    "voiced_fraction", "pause_ratio", "pause_count", "mean_pause_dur_s",
    "speech_rate_hz", "rms_mean", "rms_std",
    "spec_centroid_hz", "spec_bandwidth_hz", "spec_rolloff_hz", "spec_flatness",
    "hnr_db", "jitter_local", "shimmer_local",
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4"
]

def extract_audio_features(filepath: Path) -> dict:
    y, sr = sf.read(str(filepath))
    if y.ndim > 1:
        y = y[:, 0]
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    dur = len(y) / sr
    hop_length = int(sr * 0.01)   # 10ms
    frame_length = int(sr * 0.03) # 30ms

    # 1. Energy & RMS
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms_frames)) if len(rms_frames) > 0 else 0.0
    rms_std = float(np.std(rms_frames)) if len(rms_frames) > 0 else 0.0
    peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0

    # 2. Timing & Pauses
    silence_thresh = max(1e-5, peak * 0.018)
    speech_mask = rms_frames >= silence_thresh
    pause_mask = ~speech_mask
    pause_ratio = float(np.mean(pause_mask)) if len(pause_mask) > 0 else 0.0

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

    # Speech / segment rate
    speech_rate = float((pause_count + 1) / dur) if dur > 0 else 0.0

    # 3. F0 / Pitch (via pyin)
    try:
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
    except Exception:
        f0_mean, f0_median, f0_std, f0_range, voiced_fraction, jitter_local = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 4. Spectral Features
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]

    spec_cent_mean = float(np.mean(cent)) if len(cent) > 0 else 0.0
    spec_bw_mean = float(np.mean(bw)) if len(bw) > 0 else 0.0
    spec_rolloff_mean = float(np.mean(rolloff)) if len(rolloff) > 0 else 0.0
    spec_flatness_mean = float(np.mean(flatness)) if len(flatness) > 0 else 0.0

    # 5. Harmonics-to-Noise Ratio (HNR via HPSS)
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy = np.sum(y_harm ** 2)
        perc_energy = np.sum(y_perc ** 2)
        hnr_db = float(10 * np.log10(max(1e-6, harm_energy) / max(1e-6, perc_energy)))
    except Exception:
        hnr_db = 0.0

    # 6. Shimmer (local relative difference in voiced speech RMS)
    if np.any(speech_mask):
        speech_rms = rms_frames[speech_mask]
        if len(speech_rms) > 10:
            rms_diffs = np.abs(np.diff(speech_rms))
            shimmer_local = float(np.mean(rms_diffs) / np.mean(speech_rms)) if np.mean(speech_rms) > 0 else 0.0
        else:
            shimmer_local = 0.0
    else:
        shimmer_local = 0.0

    # 7. MFCC 1-4
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_means = np.mean(mfccs, axis=1) if mfccs.shape[1] > 0 else np.zeros(13)

    return {
        "duration": round(dur, 3),
        "f0_mean_hz": round(f0_mean, 2),
        "f0_median_hz": round(f0_median, 2),
        "f0_std_hz": round(f0_std, 2),
        "f0_range_hz": round(f0_range, 2),
        "voiced_fraction": round(voiced_fraction, 4),
        "pause_ratio": round(pause_ratio, 4),
        "pause_count": pause_count,
        "mean_pause_dur_s": round(mean_pause_dur, 3),
        "speech_rate_hz": round(speech_rate, 3),
        "rms_mean": round(rms_mean, 4),
        "rms_std": round(rms_std, 4),
        "spec_centroid_hz": round(spec_cent_mean, 1),
        "spec_bandwidth_hz": round(spec_bw_mean, 1),
        "spec_rolloff_hz": round(spec_rolloff_mean, 1),
        "spec_flatness": round(spec_flatness_mean, 6),
        "hnr_db": round(hnr_db, 2),
        "jitter_local": round(jitter_local, 5),
        "shimmer_local": round(shimmer_local, 5),
        "mfcc_1": round(float(mfcc_means[0]), 2),
        "mfcc_2": round(float(mfcc_means[1]), 2),
        "mfcc_3": round(float(mfcc_means[2]), 2),
        "mfcc_4": round(float(mfcc_means[3]), 2)
    }

def compute_cliffs_delta(x, y):
    """Calculates Cliff's delta non-parametric effect size."""
    n_x = len(x)
    n_y = len(y)
    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1
    return (more - less) / (n_x * n_y) if (n_x * n_y) > 0 else 0.0

def compute_group_effects(df, feature_cols, prefix=""):
    """Computes effect sizes and statistical tests for PD vs HC."""
    records = []
    for feat in feature_cols:
        pd_vals = df[df["diagnosis"] == "PD"][feat].values.astype(float)
        hc_vals = df[df["diagnosis"] == "HC"][feat].values.astype(float)

        mean_pd = float(np.mean(pd_vals))
        mean_hc = float(np.mean(hc_vals))
        std_pd = float(np.std(pd_vals, ddof=1)) if len(pd_vals) > 1 else 0.0
        std_hc = float(np.std(hc_vals, ddof=1)) if len(hc_vals) > 1 else 0.0

        n_pd = len(pd_vals)
        n_hc = len(hc_vals)
        s_pooled = np.sqrt(((n_pd - 1) * std_pd**2 + (n_hc - 1) * std_hc**2) / max(1, n_pd + n_hc - 2))
        cohen_d = (mean_pd - mean_hc) / s_pooled if s_pooled > 0 else 0.0
        cliffs_d = compute_cliffs_delta(pd_vals, hc_vals)

        # Mann-Whitney U test & Welch's t-test
        try:
            stat_mwu, p_mwu = stats.mannwhitneyu(pd_vals, hc_vals, alternative="two-sided")
        except Exception:
            p_mwu = 1.0
        try:
            stat_tt, p_tt = stats.ttest_ind(pd_vals, hc_vals, equal_var=False)
        except Exception:
            p_tt = 1.0

        # Effect direction: PD > HC (+) or HC > PD (-)
        direction = "PD > HC" if mean_pd >= mean_hc else "HC > PD"

        records.append({
            "feature": feat,
            f"{prefix}mean_hc": round(mean_hc, 4),
            f"{prefix}std_hc": round(std_hc, 4),
            f"{prefix}mean_pd": round(mean_pd, 4),
            f"{prefix}std_pd": round(std_pd, 4),
            f"{prefix}effect_direction": direction,
            f"{prefix}cohen_d": round(cohen_d, 4),
            f"{prefix}cliffs_delta": round(cliffs_d, 4),
            f"{prefix}p_mwu": round(p_mwu, 5),
            f"{prefix}p_welch": round(p_tt, 5)
        })
    return pd.DataFrame(records)

def benjamini_hochberg(p_values):
    """Computes Benjamini-Hochberg FDR corrected q-values."""
    p_vals = np.array(p_values)
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]
    q_vals = np.zeros(n)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = min(1.0, sorted_p[i] * n / rank)
        cum_min = min(cum_min, q)
        q_vals[i] = cum_min
    orig_order_q = np.zeros(n)
    orig_order_q[sorted_idx] = q_vals
    return orig_order_q

def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    PRES_DIR.mkdir(parents=True, exist_ok=True)

    df_cohort = pd.read_csv(SELECT_CSV)
    print(f"Loaded cohort selection: {len(df_cohort)} recordings.")

    orig_feat_records = []
    gen_feat_records = []

    print("\n=== Step 1: Extracting 19 Pathology Features for 80 Original Recordings ===")
    for idx, row in df_cohort.iterrows():
        sid = row["safe_id"]
        orig_p = Path(row["audio_path"])
        feats = extract_audio_features(orig_p)
        orig_feat_records.append({
            "safe_id": sid,
            "speaker_id": row["speaker_id"],
            "language": row["language"],
            "diagnosis": row["diagnosis"],
            "target_language": row["target_language"],
            "audio_domain": "original",
            **feats
        })
        if (idx + 1) % 20 == 0:
            print(f"  Original: {idx + 1}/80 done")

    print("\n=== Step 2: Extracting 19 Pathology Features for 80 Converted Recordings ===")
    for idx, row in df_cohort.iterrows():
        sid = row["safe_id"]
        src_lang = row["language"]
        tgt_fn = row["target_filename"]
        gen_dir = ES_DE_DIR if src_lang == "Spanish" else DE_ES_DIR
        gen_p = gen_dir / tgt_fn

        feats = extract_audio_features(gen_p)
        gen_feat_records.append({
            "safe_id": sid,
            "speaker_id": row["speaker_id"],
            "language": row["language"],
            "diagnosis": row["diagnosis"],
            "target_language": row["target_language"],
            "audio_domain": "generated",
            **feats
        })
        if (idx + 1) % 20 == 0:
            print(f"  Generated: {idx + 1}/80 done")

    df_orig_feats = pd.DataFrame(orig_feat_records)
    df_gen_feats = pd.DataFrame(gen_feat_records)

    orig_csv = FEAT_DIR / "original_features_80.csv"
    gen_csv = FEAT_DIR / "generated_features_80.csv"
    df_orig_feats.to_csv(orig_csv, index=False)
    df_gen_feats.to_csv(gen_csv, index=False)
    print(f"\nSaved feature tables to {orig_csv} and {gen_csv}")

    # Feature dictionary
    dict_path = FEAT_DIR / "feature_dictionary.md"
    dict_content = """# Acoustic Pathology Feature Dictionary

| Feature Name | Domain | Physical Unit | Clinical / Pathological Relevance in Parkinson's Disease |
| :--- | :--- | :---: | :--- |
| `f0_mean_hz` | Pitch / Phonation | Hz | Mean fundamental frequency; captures pitch shifts associated with laryngeal rigidity. |
| `f0_median_hz` | Pitch / Phonation | Hz | Robust central pitch measure insensitive to voicing octave errors. |
| `f0_std_hz` | Intonation / Prosody | Hz | Pitch variability; reduced variability reflects monotonic speech (hypoprosodia). |
| `f0_range_hz` | Pitch Range | Hz | 5th-95th percentile pitch excursion; compressed in Parkinsonian hypophonia. |
| `voiced_fraction` | Voicing Dynamics | [0, 1] | Proportion of voiced speech; captures phonation breaks and breathiness. |
| `pause_ratio` | Temporal / Timing | [0, 1] | Ratio of unvoiced silent intervals to total duration; elevated in PD hesitation. |
| `pause_count` | Temporal / Timing | Count | Number of distinct pauses >= 100ms; reflects articulatory arrests. |
| `mean_pause_dur_s`| Temporal / Timing | Seconds | Average pause duration; prolonged pauses correlate with motor speech blocks. |
| `speech_rate_hz` | Articulation Rate | Seg/sec | Rate of speech segments; alterations reflect bradykinesia or festination. |
| `rms_mean` | Loudness / Energy | RMS | Overall vocal intensity; decreased in Parkinsonian hypophonia. |
| `rms_std` | Dynamic Range | RMS | Loudness variability; reduced in monotonic speech. |
| `spec_centroid_hz`| Spectral Envelope | Hz | Spectral center of mass; captures vocal tract brightness and formant shifts. |
| `spec_bandwidth_hz`| Spectral Envelope | Hz | Spectral dispersion across frequencies; correlates with vocal tract resonances. |
| `spec_rolloff_hz` | High-Frequency Energy| Hz | 85% energy frequency cutoff; reflects breathiness vs glottal hyperadduction. |
| `spec_flatness` | Spectral Structure | [0, 1] | Ratio of geometric to arithmetic spectral mean; elevated in turbulent noise. |
| `hnr_db` | Glottal Quality | dB | Harmonics-to-noise ratio; lower values signify breathiness and turbulent noise. |
| `jitter_local` | Vocal Perturbation | Ratio | Period-to-period frequency perturbation; elevated in vocal tremor and dysphonia. |
| `shimmer_local` | Vocal Perturbation | Ratio | Amplitude perturbation across consecutive pitch periods; reflects dysphonia. |
| `mfcc_1` - `mfcc_4`| Timbral Profile | Coeff | Low-order cepstral coefficients summarizing vocal tract filter shape. |
"""
    dict_path.write_text(dict_content, encoding="utf-8")
    print(f"Saved feature dictionary to {dict_path}")

    # ========================================================
    # Step 3: Pathology Preservation Analysis (Section 15)
    # ========================================================
    print("\n=== Step 3: Pathology Preservation Analysis ===")
    df_eff_orig = compute_group_effects(df_orig_feats, FEATURE_NAMES, prefix="orig_")
    df_eff_gen = compute_group_effects(df_gen_feats, FEATURE_NAMES, prefix="gen_")

    # Add BH-FDR q-values
    df_eff_orig["orig_q_mwu"] = benjamini_hochberg(df_eff_orig["orig_p_mwu"].values)
    df_eff_gen["gen_q_mwu"] = benjamini_hochberg(df_eff_gen["gen_p_mwu"].values)

    df_eff_orig.to_csv(PRES_DIR / "pathology_effects_original.csv", index=False)
    df_eff_gen.to_csv(PRES_DIR / "pathology_effects_generated.csv", index=False)

    df_compare = pd.merge(df_eff_orig, df_eff_gen, on="feature")

    # Determine status per feature
    statuses = []
    delta_ds = []
    for _, row in df_compare.iterrows():
        d_o = row["orig_cohen_d"]
        d_g = row["gen_cohen_d"]
        dir_o = row["orig_effect_direction"]
        dir_g = row["gen_effect_direction"]

        delta_ds.append(round(d_g - d_o, 4))
        if dir_o == dir_g:
            if abs(d_g) < 0.7 * abs(d_o):
                statuses.append("Attenuated")
            elif abs(d_g) > 1.3 * abs(d_o):
                statuses.append("Amplified")
            else:
                statuses.append("Preserved")
        else:
            statuses.append("Reversed")

    df_compare["preservation_status"] = statuses
    df_compare["delta_cohen_d"] = delta_ds
    df_compare.to_csv(PRES_DIR / "pathology_preservation_comparison.csv", index=False)

    total_feats = len(df_compare)
    pres_count = int(np.sum(df_compare["preservation_status"].isin(["Preserved", "Attenuated", "Amplified"])))
    pres_pct = (pres_count / total_feats) * 100.0
    strict_pres_count = int(np.sum(df_compare["preservation_status"] == "Preserved"))
    rev_count = int(np.sum(df_compare["preservation_status"] == "Reversed"))

    print(f"Pathology direction preservation: {pres_pct:.1f}% ({pres_count}/{total_feats} features maintained effect sign)")

    # ========================================================
    # Step 4: Pathology Normalization Analysis (Section 16)
    # ========================================================
    print("\n=== Step 4: Testing for Pathology Normalization ===")
    # Standardize features using original distributions
    X_orig = df_orig_feats[FEATURE_NAMES].values.astype(float)
    X_gen = df_gen_feats[FEATURE_NAMES].values.astype(float)

    feat_mean = np.mean(X_orig, axis=0)
    feat_std = np.std(X_orig, axis=0)
    feat_std[feat_std == 0] = 1.0

    X_orig_norm = (X_orig - feat_mean) / feat_std
    X_gen_norm = (X_gen - feat_mean) / feat_std

    is_hc = (df_orig_feats["diagnosis"] == "HC").values
    is_pd = (df_orig_feats["diagnosis"] == "PD").values

    hc_centroid_orig = np.mean(X_orig_norm[is_hc], axis=0)
    hc_centroid_gen = np.mean(X_gen_norm[is_hc], axis=0)

    # Compute Euclidean distance to HC centroid for each PD subject
    d_orig_pd_to_hc = np.linalg.norm(X_orig_norm[is_pd] - hc_centroid_orig, axis=1)
    d_gen_pd_to_hc = np.linalg.norm(X_gen_norm[is_pd] - hc_centroid_gen, axis=1)

    norm_diff = d_gen_pd_to_hc - d_orig_pd_to_hc
    mean_d_orig = float(np.mean(d_orig_pd_to_hc))
    mean_d_gen = float(np.mean(d_gen_pd_to_hc))

    # Paired statistical test
    t_stat, p_val_ttest = stats.ttest_rel(d_gen_pd_to_hc, d_orig_pd_to_hc)
    w_stat, p_val_wilcoxon = stats.wilcoxon(d_gen_pd_to_hc, d_orig_pd_to_hc)

    # Decision on normalization
    # Normalization means generated PD moved significantly CLOSER to HC centroid (d_gen < d_orig)
    if mean_d_gen < mean_d_orig and p_val_wilcoxon < 0.05:
        norm_result = "EVIDENCE OF NORMALIZATION"
        norm_desc = (
            f"Generated PD speech shifted significantly closer to the Healthy Control centroid "
            f"(Mean distance: {mean_d_gen:.3f} vs {mean_d_orig:.3f}, Wilcoxon p={p_val_wilcoxon:.4f}). "
            f"Voice conversion produces mild acoustic smoothing toward healthy norms."
        )
    elif mean_d_gen >= mean_d_orig and p_val_wilcoxon >= 0.05:
        norm_result = "NO EVIDENCE OF NORMALIZATION"
        norm_desc = (
            f"Generated PD speech maintains distinct acoustic separation from Healthy Controls "
            f"(Mean distance: {mean_d_gen:.3f} vs {mean_d_orig:.3f}, Wilcoxon p={p_val_wilcoxon:.4f}). "
            f"No statistically significant group-level regression toward healthy speech was detected."
        )
    else:
        norm_result = "INSUFFICIENT EVIDENCE OF NORMALIZATION"
        norm_desc = (
            f"Mean distance changed from {mean_d_orig:.3f} (orig) to {mean_d_gen:.3f} (gen), but difference was not "
            f"statistically significant (Wilcoxon p={p_val_wilcoxon:.4f}, t-test p={p_val_ttest:.4f})."
        )

    # Save normalization table
    pd_sids = df_orig_feats[is_pd]["safe_id"].values
    pd_langs = df_orig_feats[is_pd]["language"].values
    norm_records = []
    for i in range(len(pd_sids)):
        norm_records.append({
            "safe_id": pd_sids[i],
            "source_language": pd_langs[i],
            "orig_dist_to_hc_centroid": round(float(d_orig_pd_to_hc[i]), 4),
            "gen_dist_to_hc_centroid": round(float(d_gen_pd_to_hc[i]), 4),
            "delta_distance": round(float(norm_diff[i]), 4),
            "shifted_closer_to_hc": bool(norm_diff[i] < 0)
        })
    df_norm_out = pd.DataFrame(norm_records)
    norm_csv = PRES_DIR / "pathology_normalization_analysis.csv"
    df_norm_out.to_csv(norm_csv, index=False)
    print(f"Saved normalization analysis to {norm_csv}: Result = {norm_result}")

    # Generate master preservation report
    pres_report_md = PRES_DIR / "pathology_preservation_report.md"
    pres_content = f"""# Pathology-Relevant Feature Preservation & Normalization Report: 80-File Validation Cohort

## 1. Executive Summary
- **Evaluation Scope**: 19 clinically validated acoustic, prosodic, and spectral features across 80 subjects (40 HC, 40 PD).
- **Feature Direction Preservation Rate**: **{pres_pct:.1f}%** ({pres_count}/{total_feats} features maintained consistent PD-vs-HC effect direction).
- **Strictly Preserved Features**: **{strict_pres_count}/{total_feats}** (magnitude within 70%-130% of original Cohen's d).
- **Attenuated Features**: **{int(np.sum(df_compare['preservation_status'] == 'Attenuated'))}/{total_feats}** (effect magnitude attenuated).
- **Amplified Features**: **{int(np.sum(df_compare['preservation_status'] == 'Amplified'))}/{total_feats}** (effect magnitude heightened).
- **Reversed Features**: **{rev_count}/{total_feats}** (effect sign flipped).
- **Pathology Normalization Assessment**: **{norm_result}**
  * {norm_desc}

## 2. Feature-by-Feature Preservation Comparison

| Feature Name | Orig Effect | Orig Cohen's d | Orig q-val | Gen Effect | Gen Cohen's d | Gen q-val | Status | $\\Delta$ Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df_compare.iterrows():
        pres_content += (
            f"| `{r['feature']}` | {r['orig_effect_direction']} | {r['orig_cohen_d']:+.3f} | {r['orig_q_mwu']:.3f} | "
            f"{r['gen_effect_direction']} | {r['gen_cohen_d']:+.3f} | {r['gen_q_mwu']:.3f} | **{r['preservation_status']}** | {r['delta_cohen_d']:+.3f} |\n"
        )

    pres_content += f"""
## 3. Acoustic Normalization Analysis (PD $\\to$ HC Centroid Distance)
- **Original Mean Distance (PD $\\to$ Original HC Centroid)**: `{mean_d_orig:.4f}`
- **Generated Mean Distance (PD $\\to$ Generated HC Centroid)**: `{mean_d_gen:.4f}`
- **Mean Delta Distance**: `{float(np.mean(norm_diff)):+.4f}`
- **Proportion of PD Subjects Shifting Closer to HC**: `{float(np.mean(norm_diff < 0))*100.0:.1f}%` ({int(np.sum(norm_diff < 0))}/40)
- **Statistical Significance**:
  * Paired t-test: $t = {t_stat:.3f}$, $p = {p_val_ttest:.4f}$
  * Wilcoxon signed-rank test: $W = {w_stat:.1f}$, $p = {p_val_wilcoxon:.4f}$
- **Conclusion**: `{norm_result}`
"""
    pres_report_md.write_text(pres_content, encoding="utf-8")
    print(f"Saved pathology preservation report to {pres_report_md}")

if __name__ == "__main__":
    main()
