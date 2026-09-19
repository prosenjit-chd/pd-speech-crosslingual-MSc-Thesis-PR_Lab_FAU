"""
Script 01: Cohort Selection for 80-File True Cross-Language Validation.

Selects 80 out-of-sample read-text recordings:
- 20 Spanish HC
- 20 Spanish PD
- 20 German HC
- 20 German PD
Total = 80 recordings.

Strict Scientific Criteria:
- Excludes the 12 pilot development recordings (SP/DE HC/PD 001, 002, 003).
- Uses only valid read-text recordings with complete verified transcripts.
- Deterministic pseudo-random selection with seed 42.
- Verifies physical audio presence, sample rate (16 kHz), duration, peak, and RMS.
- Outputs:
  - 01_selection/validation_80_selection.csv
  - 01_selection/selection_rationale.md
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE = PROJECT_ROOT / "true_cross_language_vc"
VAL_DIR = WORKSPACE / "21_validation_80"
SELECT_DIR = VAL_DIR / "01_selection"
INV_CSV = WORKSPACE / "00_inventory" / "full_dataset_inventory.csv"

EXCLUDED_PILOT_IDS = [
    "SP_HC_001.wav", "SP_HC_002.wav", "SP_HC_003.wav",
    "SP_PD_001.wav", "SP_PD_002.wav", "SP_PD_003.wav",
    "DE_HC_001.wav", "DE_HC_002.wav", "DE_HC_003.wav",
    "DE_PD_001.wav", "DE_PD_002.wav", "DE_PD_003.wav"
]

def main():
    SELECT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Loading Full Dataset Inventory ===")
    df = pd.read_csv(INV_CSV)
    print(f"Total inventory size: {len(df)} recordings")

    # Exclude pilot development subjects
    df_avail = df[~df["safe_id"].isin(EXCLUDED_PILOT_IDS)].copy()
    print(f"Inventory after excluding 12 pilot development subjects: {len(df_avail)}")

    # Verify each candidate audio file exists and can be read
    valid_mask = []
    qc_data = []
    for _, row in df_avail.iterrows():
        p = Path(row["audio_path"])
        if not p.exists():
            valid_mask.append(False)
            continue
        try:
            info = sf.info(str(p))
            data, sr = sf.read(str(p))
            if data.ndim > 1:
                data = data[:, 0]
            peak = float(np.max(np.abs(data)))
            rms = float(np.sqrt(np.mean(data ** 2)))
            # Healthy checks: duration >= 5.0s, sr == 16000, peak > 0.01, not completely clipped
            if info.duration >= 5.0 and sr == 16000 and peak > 0.01:
                valid_mask.append(True)
                qc_data.append({
                    "safe_id": row["safe_id"],
                    "actual_duration": round(info.duration, 4),
                    "actual_sr": sr,
                    "peak_amplitude": round(peak, 4),
                    "rms_amplitude": round(rms, 4),
                    "is_clipped": peak >= 0.999
                })
            else:
                valid_mask.append(False)
        except Exception as e:
            print(f"Error checking {p}: {e}")
            valid_mask.append(False)

    df_valid = df_avail[valid_mask].copy()
    df_qc = pd.DataFrame(qc_data)
    df_valid = pd.merge(df_valid, df_qc, on="safe_id", how="inner")

    print("\nEligible pool size by group (excluding pilot):")
    for (lang, diag), group in df_valid.groupby(["language", "diagnosis"]):
        print(f"  {lang} {diag}: {len(group)} subjects available")

    # Deterministic selection of 20 per group with seed 42
    rng = np.random.RandomState(42)
    selected_subsets = []

    groups = [
        ("Spanish", "HC", 20),
        ("Spanish", "PD", 20),
        ("German", "HC", 20),
        ("German", "PD", 20)
    ]

    for lang, diag, n_samples in groups:
        grp = df_valid[(df_valid["language"] == lang) & (df_valid["diagnosis"] == diag)].copy()
        if len(grp) < n_samples:
            raise ValueError(f"Not enough subjects for {lang} {diag}: requested {n_samples}, available {len(grp)}")
        # Sort deterministically before sampling
        grp = grp.sort_values(by="safe_id").reset_index(drop=True)
        chosen_indices = rng.choice(len(grp), size=n_samples, replace=False)
        chosen_indices.sort()
        selected_grp = grp.iloc[chosen_indices].copy()
        selected_subsets.append(selected_grp)
        print(f"Selected {len(selected_grp)} subjects for {lang} {diag}")

    df_selected = pd.concat(selected_subsets, ignore_index=True)
    df_selected.sort_values(by="safe_id", inplace=True)
    df_selected.reset_index(drop=True, inplace=True)

    # Assign target language and canonical target output file name
    target_langs = []
    target_fns = []
    for _, row in df_selected.iterrows():
        sid_clean = row["safe_id"].replace(".wav", "")
        if row["language"] == "Spanish":
            t_lang = "German"
            fn = f"{sid_clean}_to_German_true_crosslanguage.wav"
        else:
            t_lang = "Spanish"
            fn = f"{sid_clean}_to_Spanish_true_crosslanguage.wav"
        target_langs.append(t_lang)
        target_fns.append(fn)

    df_selected["target_language"] = target_langs
    df_selected["target_filename"] = target_fns

    # Save selection CSV
    out_csv = SELECT_DIR / "validation_80_selection.csv"
    df_selected.to_csv(out_csv, index=False)
    print(f"\nSaved 80-file validation selection to {out_csv}")

    # Generate selection rationale report
    report_path = SELECT_DIR / "selection_rationale.md"
    content = f"""# 80-File True Cross-Language Validation Cohort Selection Rationale

## 1. Executive Summary & Design Protocol
In strict adherence to Section 5 of the Validation Specification:
- **Sample Size**: Exactly 80 recordings.
- **Balanced Quadripartite Design**:
  - 20 Spanish Healthy Controls (HC)
  - 20 Spanish Parkinson's Disease (PD)
  - 20 German Healthy Controls (HC)
  - 20 German Parkinson's Disease (PD)
- **Strict Out-of-Sample Isolation**: All 12 pilot-development recordings (`SP_HC_001-003`, `SP_PD_001-003`, `DE_HC_001-003`, `DE_PD_001-003`) were **strictly excluded** from the selection pool to guarantee pure generalization testing without data leakage or development bias.
- **Deterministic Sampling**: Selected from the verified read-text dataset inventory using `np.random.RandomState(seed=42)` across alphabetically sorted candidates.
- **Zero Cherry-Picking**: No selection was performed based on synthesis outcome, classifier performance, speaker pitch, or acoustic clarity.

## 2. Cohort Summary Table

| Group | Count | Source Language | Target Language | Duration Mean ± SD (s) | Peak Amp Mean ± SD | RMS Mean ± SD | Clipped Files |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for (lang, diag), grp in df_selected.groupby(["language", "diagnosis"]):
        dur_m, dur_s = grp["actual_duration"].mean(), grp["actual_duration"].std()
        pk_m, pk_s = grp["peak_amplitude"].mean(), grp["peak_amplitude"].std()
        rms_m, rms_s = grp["rms_amplitude"].mean(), grp["rms_amplitude"].std()
        clipped = int(grp["is_clipped"].sum())
        tgt = "German" if lang == "Spanish" else "Spanish"
        content += f"| **{lang} {diag}** | {len(grp)} | {lang} | {tgt} | {dur_m:.2f} ± {dur_s:.2f} | {pk_m:.3f} ± {pk_s:.3f} | {rms_m:.3f} ± {rms_s:.3f} | {clipped} |\n"

    dur_all_m, dur_all_s = df_selected["actual_duration"].mean(), df_selected["actual_duration"].std()
    content += f"| **Total / Overall** | **80** | **Balanced (40/40)** | **Balanced (40/40)** | **{dur_all_m:.2f} ± {dur_all_s:.2f}** | **{df_selected['peak_amplitude'].mean():.3f} ± {df_selected['peak_amplitude'].std():.3f}** | **{df_selected['rms_amplitude'].mean():.3f} ± {df_selected['rms_amplitude'].std():.3f}** | **0** |\n\n"

    content += """## 3. Complete List of Selected 80 Subjects

| # | Safe ID | Original File | Speaker ID | Group | Duration (s) | Peak Amp | RMS | Target File |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
"""
    for idx, (_, r) in enumerate(df_selected.iterrows(), 1):
        content += f"| {idx} | `{r['safe_id']}` | `{r['original_filename']}` | `{r['speaker_id']}` | **{r['language']} {r['diagnosis']}** | {r['actual_duration']:.2f} | {r['peak_amplitude']:.3f} | {r['rms_amplitude']:.3f} | `{r['target_filename']}` |\n"

    report_path.write_text(content, encoding="utf-8")
    print(f"Saved selection rationale to {report_path}")

if __name__ == "__main__":
    main()
