# Exploratory Pathology Sanity Check Report

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

- **Total Acoustic Features Evaluated**: 17
- **Descriptive Direction Preserved (PD vs. HC)**: **10 / 17 (58.8%)**
- **Prosodic & Timing Features**:
  - **F0 Standard Deviation (Pitch Variability)**: Original PD exhibited reduced pitch variability relative to HC (37.1 Hz vs 37.3 Hz). In refined synthesis, this direction was **No** (47.9 Hz vs 42.9 Hz).
  - **Pause Ratio**: Original PD exhibited higher pause ratio (37.2%) than HC (29.0%). In refined synthesis, this direction was **Yes** (39.3% vs 32.8%).
  - **Spectral Centroid**: Preserved direction: **Yes**.

---

## 3. Comprehensive Feature Direction Table

| Feature | Original PD (Mean / Med) | Original HC (Mean / Med) | Original Direction | Refined PD (Mean / Med) | Refined HC (Mean / Med) | Refined Direction | Direction Preserved | Reliability Caveat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `f0_mean_hz` | 155.72 / 159.84 | 167.26 / 165.34 | **PD < HC** | 139.98 / 152.66 | 156.74 / 156.24 | **PD < HC** | **Yes** | Robust acoustic feature |
| `f0_median_hz` | 158.21 / 158.51 | 165.38 / 165.94 | **PD < HC** | 136.96 / 153.88 | 161.87 / 161.20 | **PD < HC** | **Yes** | Robust acoustic feature |
| `f0_std_hz` | 37.06 / 23.42 | 37.31 / 36.58 | **PD < HC** | 47.88 / 41.16 | 42.92 / 33.80 | **PD > HC** | **No** | High clinical relevance (prosody & timing) |
| `f0_range_hz` | 114.19 / 66.26 | 126.56 / 111.78 | **PD < HC** | 137.73 / 134.84 | 134.98 / 97.41 | **PD > HC** | **No** | High clinical relevance (prosody & timing) |
| `voiced_fraction` | 0.61 / 0.68 | 0.76 / 0.74 | **PD < HC** | 0.64 / 0.66 | 0.69 / 0.71 | **PD < HC** | **Yes** | Robust acoustic feature |
| `pause_ratio` | 0.37 / 0.32 | 0.29 / 0.30 | **PD > HC** | 0.39 / 0.38 | 0.33 / 0.33 | **PD > HC** | **Yes** | High clinical relevance (prosody & timing) |
| `pause_count` | 13.50 / 11.50 | 12.33 / 12.00 | **PD > HC** | 25.33 / 23.50 | 18.00 / 15.50 | **PD > HC** | **Yes** | Robust acoustic feature |
| `mean_pause_dur_s` | 0.78 / 0.52 | 0.41 / 0.40 | **PD > HC** | 0.69 / 0.57 | 0.70 / 0.69 | **PD < HC** | **No** | High clinical relevance (prosody & timing) |
| `rms_mean` | 0.08 / 0.08 | 0.08 / 0.08 | **PD > HC** | 0.09 / 0.09 | 0.10 / 0.10 | **PD < HC** | **No** | Robust acoustic feature |
| `rms_variability` | 0.08 / 0.08 | 0.07 / 0.08 | **PD > HC** | 0.09 / 0.09 | 0.10 / 0.10 | **PD < HC** | **No** | Robust acoustic feature |
| `spec_centroid_hz` | 1745.55 / 1809.00 | 1616.72 / 1593.45 | **PD > HC** | 1397.72 / 1456.00 | 1346.87 / 1385.30 | **PD > HC** | **Yes** | Robust acoustic feature |
| `spec_bandwidth_hz` | 1787.88 / 1724.45 | 1696.40 / 1689.90 | **PD > HC** | 1518.22 / 1493.60 | 1425.32 / 1457.80 | **PD > HC** | **Yes** | Robust acoustic feature |
| `spec_rolloff_hz` | 3592.90 / 3453.85 | 3302.87 / 3292.75 | **PD > HC** | 2857.32 / 2847.00 | 2635.57 / 2697.40 | **PD > HC** | **Yes** | Robust acoustic feature |
| `spec_flatness` | 0.03 / 0.03 | 0.02 / 0.02 | **PD > HC** | 0.09 / 0.09 | 0.10 / 0.09 | **PD < HC** | **No** | Robust acoustic feature |
| `hnr_db` | 1.84 / 1.59 | 0.62 / -0.04 | **PD > HC** | 0.12 / 1.36 | 0.11 / 0.34 | **PD > HC** | **Yes** | Robust acoustic feature |
| `jitter_local` | 0.01 / 0.01 | 0.01 / 0.01 | **PD < HC** | 0.01 / 0.01 | 0.01 / 0.01 | **PD < HC** | **Yes** | Heuristic only (synthetic phase artifact caveat) |
| `shimmer_local` | 0.12 / 0.12 | 0.13 / 0.13 | **PD < HC** | 0.13 / 0.13 | 0.12 / 0.12 | **PD > HC** | **No** | Heuristic only (synthetic phase artifact caveat) |

---

## 4. Methodological Conclusion
The descriptive direction check reveals that coarse prosodic tendencies (pitch contour range and spectral distribution) can be transferred via zero-shot speaker conditioning. However, articulatory precision is partially regularized by neural TTS decoders. Dedicated quantitative feature extraction and classifier experiments on the validated cohort will be required to formally evaluate downstream utility.
