# Cohort Provenance and Partitioning Report

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
   - 100 Spanish speakers (50 HC, 50 PD) $	o$ German.
   - 176 German speakers (88 HC, 88 PD) $	o$ Spanish.
2. **SET B — DEVELOPMENT-EXCLUDED COHORT ($N = 264$)**:
   - Strict sensitivity test excluding the 12 pilot development subjects ($276 - 12 = 264$).
   - Completely removes any possibility of pilot tuning circularity across all 264 remaining subjects.
3. **SET C — INDEPENDENT VALIDATION REFERENCE ($N = 80$)**:
   - The frozen, archived confirmatory validation result from `21_validation_80`.
   - Maintained as the immutable ground-truth benchmark.
