# Pathology-Relevant Feature Preservation & Normalization Report: 80-File Validation Cohort

## 1. Executive Summary
- **Evaluation Scope**: 19 clinically validated acoustic, prosodic, and spectral features across 80 subjects (40 HC, 40 PD).
- **Feature Direction Preservation Rate**: **68.2%** (15/22 features maintained consistent PD-vs-HC effect direction).
- **Strictly Preserved Features**: **4/22** (magnitude within 70%-130% of original Cohen's d).
- **Attenuated Features**: **5/22** (effect magnitude attenuated).
- **Amplified Features**: **6/22** (effect magnitude heightened).
- **Reversed Features**: **7/22** (effect sign flipped).
- **Pathology Normalization Assessment**: **NO EVIDENCE OF NORMALIZATION**
  * Generated PD speech maintains distinct acoustic separation from Healthy Controls (Mean distance: 4.715 vs 4.708, Wilcoxon p=0.4125). No statistically significant group-level regression toward healthy speech was detected.

## 2. Feature-by-Feature Preservation Comparison

| Feature Name | Orig Effect | Orig Cohen's d | Orig q-val | Gen Effect | Gen Cohen's d | Gen q-val | Status | $\Delta$ Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `f0_mean_hz` | HC > PD | -0.075 | 0.776 | HC > PD | -0.369 | 0.273 | **Amplified** | -0.294 |
| `f0_median_hz` | HC > PD | -0.079 | 0.735 | HC > PD | -0.160 | 0.578 | **Amplified** | -0.081 |
| `f0_std_hz` | HC > PD | -0.426 | 0.062 | HC > PD | -0.152 | 0.775 | **Attenuated** | +0.273 |
| `f0_range_hz` | HC > PD | -0.542 | 0.043 | HC > PD | -0.439 | 0.273 | **Preserved** | +0.104 |
| `voiced_fraction` | PD > HC | +0.001 | 0.848 | PD > HC | +0.013 | 0.775 | **Amplified** | +0.012 |
| `pause_ratio` | HC > PD | -0.053 | 0.735 | PD > HC | +0.355 | 0.273 | **Reversed** | +0.408 |
| `pause_count` | HC > PD | -0.358 | 0.597 | HC > PD | -0.078 | 0.775 | **Attenuated** | +0.280 |
| `mean_pause_dur_s` | PD > HC | +0.313 | 0.361 | PD > HC | +0.419 | 0.115 | **Amplified** | +0.106 |
| `speech_rate_hz` | HC > PD | -0.371 | 0.330 | HC > PD | -0.010 | 0.954 | **Attenuated** | +0.361 |
| `rms_mean` | PD > HC | +0.293 | 0.412 | HC > PD | -0.065 | 0.818 | **Reversed** | -0.358 |
| `rms_std` | PD > HC | +0.331 | 0.330 | PD > HC | +0.483 | 0.115 | **Amplified** | +0.152 |
| `spec_centroid_hz` | HC > PD | -0.289 | 0.525 | HC > PD | -0.141 | 0.815 | **Attenuated** | +0.149 |
| `spec_bandwidth_hz` | HC > PD | -0.056 | 0.928 | PD > HC | +0.167 | 0.775 | **Reversed** | +0.223 |
| `spec_rolloff_hz` | HC > PD | -0.130 | 0.796 | PD > HC | +0.112 | 0.775 | **Reversed** | +0.241 |
| `spec_flatness` | PD > HC | +0.086 | 0.735 | PD > HC | +0.191 | 0.775 | **Amplified** | +0.105 |
| `hnr_db` | PD > HC | +0.722 | 0.016 | PD > HC | +0.508 | 0.115 | **Preserved** | -0.214 |
| `jitter_local` | HC > PD | -0.360 | 0.330 | HC > PD | -0.418 | 0.273 | **Preserved** | -0.058 |
| `shimmer_local` | HC > PD | -0.425 | 0.330 | PD > HC | +0.176 | 0.775 | **Reversed** | +0.601 |
| `mfcc_1` | PD > HC | +0.137 | 0.735 | HC > PD | -0.232 | 0.662 | **Reversed** | -0.369 |
| `mfcc_2` | PD > HC | +0.283 | 0.330 | PD > HC | +0.244 | 0.538 | **Preserved** | -0.038 |
| `mfcc_3` | HC > PD | -0.004 | 0.939 | PD > HC | +0.129 | 0.775 | **Reversed** | +0.132 |
| `mfcc_4` | PD > HC | +0.018 | 0.939 | PD > HC | +0.004 | 0.939 | **Attenuated** | -0.014 |

## 3. Acoustic Normalization Analysis (PD $\to$ HC Centroid Distance)
- **Original Mean Distance (PD $\to$ Original HC Centroid)**: `4.7080`
- **Generated Mean Distance (PD $\to$ Generated HC Centroid)**: `4.7150`
- **Mean Delta Distance**: `+0.0070`
- **Proportion of PD Subjects Shifting Closer to HC**: `60.0%` (24/40)
- **Statistical Significance**:
  * Paired t-test: $t = 0.025$, $p = 0.9802$
  * Wilcoxon signed-rank test: $W = 348.0$, $p = 0.4125$
- **Conclusion**: `NO EVIDENCE OF NORMALIZATION`
