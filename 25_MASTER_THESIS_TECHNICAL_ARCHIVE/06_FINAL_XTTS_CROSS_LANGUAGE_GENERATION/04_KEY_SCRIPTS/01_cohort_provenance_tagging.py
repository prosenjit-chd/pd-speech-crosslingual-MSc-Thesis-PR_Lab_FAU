"""
Script 01: Cohort Provenance Tagging and Analysis Set Partitioning.

Implements Sections 6 & 7:
- Tags every subject with exact historical provenance:
  * PILOT_DEVELOPMENT (12 subjects)
  * VALIDATION_80 (80 subjects)
  * FULL_COHORT_ONLY (184 subjects)
- Defines analysis roles across the three predefined sets:
  * SET A: Full Cohort (N = 276)
  * SET B: Development-Excluded Sensitivity Cohort (N = 264)
  * SET C: Independent Validation Benchmark (Archived 80-file cohort)
- Generates 02_cohort_tags/cohort_provenance.csv and cohort_provenance_report.md
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE = PROJECT_ROOT / "true_cross_language_vc"
EVAL_DIR = WORKSPACE / "22_full_276_evaluation"

INV_CSV = EVAL_DIR / "01_inventory" / "full_276_inventory.csv"
PILOT_CSV = WORKSPACE / "04_pilot_selection" / "pilot_12_selection.csv"
VAL80_CSV = WORKSPACE / "21_validation_80" / "01_selection" / "validation_80_selection.csv"

def main():
    print("=== Processing Cohort Provenance Tags ===")
    df_inv = pd.read_csv(INV_CSV)
    df_pilot = pd.read_csv(PILOT_CSV)
    df_val80 = pd.read_csv(VAL80_CSV)

    pilot_sids = set(df_pilot["safe_id"].tolist())
    val80_sids = set(df_val80["safe_id"].tolist())

    print(f"Pilot Development IDs: {len(pilot_sids)}")
    print(f"Validation 80 IDs:    {len(val80_sids)}")

    # Verify disjointness of pilot and validation 80
    overlap = pilot_sids.intersection(val80_sids)
    assert len(overlap) == 0, f"Error: Pilot and Validation 80 cohorts overlap on {overlap}!"
    print("Confirmed: Pilot development and Validation 80 cohorts are strictly mutually exclusive.")

    records = []
    for _, row in df_inv.iterrows():
        sid = row["safe_id"]
        lang = row["language"]
        diag = row["diagnosis"]

        is_pilot = sid in pilot_sids
        is_val80 = sid in val80_sids
        is_full_only = (not is_pilot) and (not is_val80)

        if is_pilot:
            tag = "PILOT_DEVELOPMENT"
            role = "Set A (Full); Excluded from Set B; Excluded from Set C"
        elif is_val80:
            tag = "VALIDATION_80"
            role = "Set A (Full); Included in Set B (Dev-Excluded); Set C Confirmatory"
        else:
            tag = "FULL_COHORT_ONLY"
            role = "Set A (Full); Included in Set B (Dev-Excluded); Excluded from Set C"

        records.append({
            "safe_id": sid,
            "language": lang,
            "diagnosis": diag,
            "pilot_subject": is_pilot,
            "validation80_subject": is_val80,
            "full_cohort_only": is_full_only,
            "cohort_tag": tag,
            "analysis_role": role,
            "in_set_a_full_276": True,
            "in_set_b_dev_excluded_264": not is_pilot,
            "in_set_c_val_80": is_val80
        })

    df_tags = pd.DataFrame(records)
    out_csv = EVAL_DIR / "02_cohort_tags" / "cohort_provenance.csv"
    df_tags.to_csv(out_csv, index=False)
    print(f"Saved cohort provenance tags to {out_csv} ({len(df_tags)} records)")

    # Breakdown summary
    counts = df_tags["cohort_tag"].value_counts()
    print("\nCohort Tag Breakdown:")
    for tag, cnt in counts.items():
        print(f"  {tag}: {cnt}")

    set_a_cnt = df_tags["in_set_a_full_276"].sum()
    set_b_cnt = df_tags["in_set_b_dev_excluded_264"].sum()
    set_c_cnt = df_tags["in_set_c_val_80"].sum()

    print(f"\nAnalysis Sets:")
    print(f"  SET A (Full Cohort):                      N = {set_a_cnt}")
    print(f"  SET B (Development-Excluded Sensitivity): N = {set_b_cnt}")
    print(f"  SET C (Independent Validation Benchmark): N = {set_c_cnt}")

    rep_md = EVAL_DIR / "02_cohort_tags" / "cohort_provenance_report.md"
    content = f"""# Cohort Provenance and Partitioning Report

**Experiment:** Autonomous Full 276-File True Cross-Language Evaluation  
**Directory:** `true_cross_language_vc/22_full_276_evaluation/02_cohort_tags/`  
**Date:** 2026-09-18  

---

## 1. Provenance Hierarchy and Full Disclosure

In strict compliance with open science and rigorous validation principles, the 276 recordings in this evaluation are categorized by historical experimental exposure:

| Provenance Tag | Spanish HC | Spanish PD | German HC | German PD | Total Count | Analysis Eligibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`PILOT_DEVELOPMENT`** | 3 | 3 | 3 | 3 | **12** | Included in Set A; Excluded from Set B & C |
| **`VALIDATION_80`** | 20 | 20 | 20 | 20 | **80** | Primary confirmatory cohort; Included in Set A, B, C |
| **`FULL_COHORT_ONLY`** | 27 | 27 | 65 | 65 | **184** | Extension subjects; Included in Set A & B |
| **TOTAL** | **50** | **50** | **88** | **88** | **276** | Set A (Master Full Cohort) |

## 2. Predefined Analysis Sets

1. **SET A — FULL COHORT ($N = 276$)**:
   - Comprehensive sensitivity analysis across the entire available dataset.
   - 100 Spanish speakers (50 HC, 50 PD) $\to$ German.
   - 176 German speakers (88 HC, 88 PD) $\to$ Spanish.
2. **SET B — DEVELOPMENT-EXCLUDED COHORT ($N = 264$)**:
   - Strict sensitivity test excluding the 12 pilot development subjects ($276 - 12 = 264$).
   - Completely removes any possibility of pilot tuning circularity across all 264 remaining subjects.
3. **SET C — INDEPENDENT VALIDATION REFERENCE ($N = 80$)**:
   - The frozen, archived confirmatory validation result from `21_validation_80`.
   - Maintained as the immutable ground-truth benchmark.
"""
    rep_md.write_text(content, encoding="utf-8")
    print(f"Saved cohort provenance report to {rep_md}")

if __name__ == "__main__":
    main()
