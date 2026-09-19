#!/usr/bin/env python3
"""
09_build_listening_package.py
Creates the 3-way Audio Listening Review Package for All 6 Spanish -> German Subjects.

Implements Section 31:
- Copies audio files into 18_listening_package/audio/:
  1. Original Spanish recording
  2. Config C baseline German output
  3. Final optimized German output
- Saves German target transcript text
- Creates 18_listening_package/README.md containing a comprehensive evaluation table with:
  safe_id, HC/PD, source filename, Config C filename, final filename,
  German target text, WER before, WER after, speaker similarity before, speaker similarity after.
"""

import shutil
from pathlib import Path
import pandas as pd

WORKSPACE = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement")
PKG_DIR = WORKSPACE / "18_listening_package"
AUDIO_PKG_DIR = PKG_DIR / "audio"
TABLES_DIR = WORKSPACE / "17_tables"
FINAL_AUDIO_DIR = WORKSPACE / "10_final_regeneration" / "spanish_to_german"
CONFIG_C_DIR = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\19_pilot_refinement\04_regenerated_audio\spanish_to_german")
PILOT_CSV = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\04_pilot_selection\pilot_12_selection.csv")
TARGET_TEXT_FILE = WORKSPACE / "02_text_analysis" / "german_target_text_original.txt"

def main():
    AUDIO_PKG_DIR.mkdir(parents=True, exist_ok=True)
    
    comp_csv = TABLES_DIR / "comparison_config_c_vs_final.csv"
    if not comp_csv.exists():
        print(f"Error: {comp_csv} not found! Run evaluation first.")
        return
        
    df_comp = pd.read_csv(comp_csv)
    df_pilot = pd.read_csv(PILOT_CSV)
    df_pilot_es = df_pilot[df_pilot["language"] == "Spanish"].copy()
    
    orig_paths = {
        r["safe_id"].replace(".wav", ""): Path(r["audio_path"])
        for _, r in df_pilot_es.iterrows()
    }
    
    german_target_text = TARGET_TEXT_FILE.read_text(encoding="utf-8")
    (PKG_DIR / "german_target_transcript.txt").write_text(german_target_text, encoding="utf-8")
    
    package_rows = []
    
    print("=== Packaging Listening Package Audio Files ===")
    for _, row in df_comp.iterrows():
        sid = row["safe_id"]
        diag = row["diagnosis"]
        
        # 1. Original Spanish
        orig_p = orig_paths[sid]
        pkg_orig_name = f"{sid}_01_original_Spanish.wav"
        pkg_orig_path = AUDIO_PKG_DIR / pkg_orig_name
        shutil.copy2(str(orig_p), str(pkg_orig_path))
        
        # 2. Config C German Baseline
        cfg_c_path = CONFIG_C_DIR / f"{sid}_refined_to_German.wav"
        pkg_cfg_c_name = f"{sid}_02_config_c_German.wav"
        pkg_cfg_c_path = AUDIO_PKG_DIR / pkg_cfg_c_name
        if cfg_c_path.exists():
            shutil.copy2(str(cfg_c_path), str(pkg_cfg_c_path))
        else:
            print(f"[WARN] Config C file not found: {cfg_c_path}")
            
        # 3. Final Optimized German
        final_path = FINAL_AUDIO_DIR / f"{sid}_final_to_German.wav"
        pkg_final_name = f"{sid}_03_final_optimized_German.wav"
        pkg_final_path = AUDIO_PKG_DIR / pkg_final_name
        if final_path.exists():
            shutil.copy2(str(final_path), str(pkg_final_path))
        else:
            print(f"[WARN] Final file not found: {final_path}")
            
        package_rows.append({
            "safe_id": sid,
            "diagnosis": diag,
            "source_audio": pkg_orig_name,
            "config_c_audio": pkg_cfg_c_name,
            "final_audio": pkg_final_name,
            "WER_before": row["WER_before"],
            "WER_after": row["WER_after"],
            "WER_change_pct": f"{row['WER_rel_improvement_pct']:+.1f}%",
            "Sim_before": row["Similarity_before"],
            "Sim_after": row["Similarity_after"]
        })
        print(f"[{sid}] Packaged original, Config C, and final audio.")
        
    df_pkg = pd.DataFrame(package_rows)
    table_md = df_pkg.to_markdown(index=False)
    
    readme_content = f"""# Spanish → German Final Refinement: Listening Evaluation Package

This package provides direct 3-way auditory and quantitative comparisons for all 6 Spanish subjects:
1. **Original Spanish Source Audio** (ground truth acoustic reference, PC-GITA reading passage)
2. **Config C Baseline German Synthesis** (initial sentence-level segmentation baseline, mean WER ~0.72)
3. **Final Optimized German Synthesis** (optimized segmentation and decoding controls)

---

## Target German Linguistic Content
> **Text:**
> *"{german_target_text}"*

---

## 3-Way Subject Audio & Metric Overview
{table_md}

---

## Listening Review Guidelines
When auditioning the audio pairs in `audio/`:
1. **Content Fidelity & Phonetic Clarity:**
   - Listen to `{sid}_02_config_c_German.wav` vs `{sid}_03_final_optimized_German.wav`.
   - Verify that dialogue turns ("Ach Herr Doktor...", "Haben Sie einen gebrochenen Nagel? Ja.") are pronounced naturally without repetition loops, stuttering, or tail babbles.
2. **Target Language Phonology:**
   - Verify native German phonology and prosody (vowel purity, glottal stops, German cadence).
3. **Speaker Identity & Pathology Acoustic Retention:**
   - Compare with `{sid}_01_original_Spanish.wav` to evaluate whether the timbre, pitch register, and pathological vocal qualities (e.g. vocal tremor, roughness, hypophonia) carry through into the German output.
"""
    readme_path = PKG_DIR / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"\nListening package created successfully in {PKG_DIR}")
    print(f"README written to {readme_path}")

if __name__ == "__main__":
    main()
