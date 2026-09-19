# 80-File True Cross-Language Validation Cohort Selection Rationale

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
| **German HC** | 20 | German | Spanish | 48.42 ± 7.68 | 0.763 ± 0.293 | 0.103 ± 0.081 | 9 |
| **German PD** | 20 | German | Spanish | 44.70 ± 7.26 | 0.782 ± 0.323 | 0.141 ± 0.087 | 12 |
| **Spanish HC** | 20 | Spanish | German | 17.62 ± 2.76 | 1.000 ± 0.000 | 0.102 ± 0.021 | 20 |
| **Spanish PD** | 20 | Spanish | German | 18.77 ± 5.93 | 1.000 ± 0.000 | 0.104 ± 0.028 | 20 |
| **Total / Overall** | **80** | **Balanced (40/40)** | **Balanced (40/40)** | **32.38 ± 15.58** | **0.886 ± 0.243** | **0.112 ± 0.063** | **0** |

## 3. Complete List of Selected 80 Subjects

| # | Safe ID | Original File | Speaker ID | Group | Duration (s) | Peak Amp | RMS | Target File |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `DE_HC_019.wav` | `023.u1.02.wav` | `023.u1.02` | **German HC** | 53.82 | 0.532 | 0.046 | `DE_HC_019_to_Spanish_true_crosslanguage.wav` |
| 2 | `DE_HC_020.wav` | `024.u1.02.wav` | `024.u1.02` | **German HC** | 44.39 | 0.262 | 0.032 | `DE_HC_020_to_Spanish_true_crosslanguage.wav` |
| 3 | `DE_HC_021.wav` | `025.u1.02.wav` | `025.u1.02` | **German HC** | 39.01 | 1.000 | 0.110 | `DE_HC_021_to_Spanish_true_crosslanguage.wav` |
| 4 | `DE_HC_023.wav` | `027.u1.02.wav` | `027.u1.02` | **German HC** | 51.33 | 1.000 | 0.278 | `DE_HC_023_to_Spanish_true_crosslanguage.wav` |
| 5 | `DE_HC_027.wav` | `032.u1.02.wav` | `032.u1.02` | **German HC** | 54.52 | 0.992 | 0.025 | `DE_HC_027_to_Spanish_true_crosslanguage.wav` |
| 6 | `DE_HC_028.wav` | `033.u1.02.wav` | `033.u1.02` | **German HC** | 42.70 | 1.000 | 0.165 | `DE_HC_028_to_Spanish_true_crosslanguage.wav` |
| 7 | `DE_HC_039.wav` | `047.u1.02.wav` | `047.u1.02` | **German HC** | 48.04 | 1.000 | 0.120 | `DE_HC_039_to_Spanish_true_crosslanguage.wav` |
| 8 | `DE_HC_041.wav` | `049.u1.02.wav` | `049.u1.02` | **German HC** | 67.71 | 0.580 | 0.042 | `DE_HC_041_to_Spanish_true_crosslanguage.wav` |
| 9 | `DE_HC_046.wav` | `054.u1.02.wav` | `054.u1.02` | **German HC** | 44.48 | 0.431 | 0.041 | `DE_HC_046_to_Spanish_true_crosslanguage.wav` |
| 10 | `DE_HC_049.wav` | `058.u1.02.wav` | `058.u1.02` | **German HC** | 47.28 | 0.498 | 0.046 | `DE_HC_049_to_Spanish_true_crosslanguage.wav` |
| 11 | `DE_HC_052.wav` | `061.u1.02.wav` | `061.u1.02` | **German HC** | 40.86 | 0.248 | 0.023 | `DE_HC_052_to_Spanish_true_crosslanguage.wav` |
| 12 | `DE_HC_054.wav` | `063.u1.02.wav` | `063.u1.02` | **German HC** | 44.50 | 1.000 | 0.184 | `DE_HC_054_to_Spanish_true_crosslanguage.wav` |
| 13 | `DE_HC_055.wav` | `064.u1.02.wav` | `064.u1.02` | **German HC** | 51.18 | 0.236 | 0.032 | `DE_HC_055_to_Spanish_true_crosslanguage.wav` |
| 14 | `DE_HC_056.wav` | `065.u1.02.wav` | `065.u1.02` | **German HC** | 45.30 | 1.000 | 0.204 | `DE_HC_056_to_Spanish_true_crosslanguage.wav` |
| 15 | `DE_HC_060.wav` | `071.u1.02.wav` | `071.u1.02` | **German HC** | 56.94 | 0.797 | 0.073 | `DE_HC_060_to_Spanish_true_crosslanguage.wav` |
| 16 | `DE_HC_072.wav` | `085.u1.02.wav` | `085.u1.02` | **German HC** | 38.27 | 1.000 | 0.107 | `DE_HC_072_to_Spanish_true_crosslanguage.wav` |
| 17 | `DE_HC_073.wav` | `086.u1.02.wav` | `086.u1.02` | **German HC** | 38.47 | 1.000 | 0.280 | `DE_HC_073_to_Spanish_true_crosslanguage.wav` |
| 18 | `DE_HC_077.wav` | `090.u1.02.wav` | `090.u1.02` | **German HC** | 60.28 | 0.789 | 0.059 | `DE_HC_077_to_Spanish_true_crosslanguage.wav` |
| 19 | `DE_HC_083.wav` | `096.u1.02.wav` | `096.u1.02` | **German HC** | 51.87 | 1.000 | 0.059 | `DE_HC_083_to_Spanish_true_crosslanguage.wav` |
| 20 | `DE_HC_085.wav` | `098.u1.02.wav` | `098.u1.02` | **German HC** | 47.47 | 0.885 | 0.140 | `DE_HC_085_to_Spanish_true_crosslanguage.wav` |
| 21 | `DE_PD_013.wav` | `032.u1.02.wav` | `032.u1.02` | **German PD** | 38.89 | 0.857 | 0.109 | `DE_PD_013_to_Spanish_true_crosslanguage.wav` |
| 22 | `DE_PD_016.wav` | `043.u1.02.wav` | `043.u1.02` | **German PD** | 56.02 | 0.839 | 0.123 | `DE_PD_016_to_Spanish_true_crosslanguage.wav` |
| 23 | `DE_PD_017.wav` | `044.u1.02.wav` | `044.u1.02` | **German PD** | 45.34 | 1.000 | 0.176 | `DE_PD_017_to_Spanish_true_crosslanguage.wav` |
| 24 | `DE_PD_018.wav` | `046.u1.02.wav` | `046.u1.02` | **German PD** | 40.43 | 1.000 | 0.191 | `DE_PD_018_to_Spanish_true_crosslanguage.wav` |
| 25 | `DE_PD_024.wav` | `059.u2.02.wav` | `059.u2.02` | **German PD** | 51.69 | 0.195 | 0.014 | `DE_PD_024_to_Spanish_true_crosslanguage.wav` |
| 26 | `DE_PD_026.wav` | `064.u4.02.wav` | `064.u4.02` | **German PD** | 36.89 | 1.000 | 0.173 | `DE_PD_026_to_Spanish_true_crosslanguage.wav` |
| 27 | `DE_PD_033.wav` | `082.u2.02.wav` | `082.u2.02` | **German PD** | 55.55 | 1.000 | 0.212 | `DE_PD_033_to_Spanish_true_crosslanguage.wav` |
| 28 | `DE_PD_038.wav` | `090.u1.02.wav` | `090.u1.02` | **German PD** | 38.36 | 0.213 | 0.031 | `DE_PD_038_to_Spanish_true_crosslanguage.wav` |
| 29 | `DE_PD_044.wav` | `100.u1.02.wav` | `100.u1.02` | **German PD** | 42.27 | 1.000 | 0.167 | `DE_PD_044_to_Spanish_true_crosslanguage.wav` |
| 30 | `DE_PD_045.wav` | `106.u2.02.wav` | `106.u2.02` | **German PD** | 52.41 | 1.000 | 0.074 | `DE_PD_045_to_Spanish_true_crosslanguage.wav` |
| 31 | `DE_PD_046.wav` | `107.u2.02.wav` | `107.u2.02` | **German PD** | 37.64 | 1.000 | 0.196 | `DE_PD_046_to_Spanish_true_crosslanguage.wav` |
| 32 | `DE_PD_051.wav` | `118.u1.02.wav` | `118.u1.02` | **German PD** | 48.00 | 1.000 | 0.197 | `DE_PD_051_to_Spanish_true_crosslanguage.wav` |
| 33 | `DE_PD_053.wav` | `123.u1.02.wav` | `123.u1.02` | **German PD** | 54.15 | 0.386 | 0.037 | `DE_PD_053_to_Spanish_true_crosslanguage.wav` |
| 34 | `DE_PD_054.wav` | `124.u1.02.wav` | `124.u1.02` | **German PD** | 45.30 | 1.000 | 0.271 | `DE_PD_054_to_Spanish_true_crosslanguage.wav` |
| 35 | `DE_PD_060.wav` | `138.u1.02.wav` | `138.u1.02` | **German PD** | 54.04 | 0.629 | 0.082 | `DE_PD_060_to_Spanish_true_crosslanguage.wav` |
| 36 | `DE_PD_067.wav` | `147.u1.02.wav` | `147.u1.02` | **German PD** | 44.83 | 1.000 | 0.132 | `DE_PD_067_to_Spanish_true_crosslanguage.wav` |
| 37 | `DE_PD_068.wav` | `148.u3.02.wav` | `148.u3.02` | **German PD** | 32.27 | 0.319 | 0.044 | `DE_PD_068_to_Spanish_true_crosslanguage.wav` |
| 38 | `DE_PD_071.wav` | `157.u1.02.wav` | `157.u1.02` | **German PD** | 43.50 | 1.000 | 0.315 | `DE_PD_071_to_Spanish_true_crosslanguage.wav` |
| 39 | `DE_PD_079.wav` | `182.u1.02.wav` | `182.u1.02` | **German PD** | 40.59 | 0.199 | 0.029 | `DE_PD_079_to_Spanish_true_crosslanguage.wav` |
| 40 | `DE_PD_080.wav` | `183.u1.02.wav` | `183.u1.02` | **German PD** | 35.90 | 1.000 | 0.245 | `DE_PD_080_to_Spanish_true_crosslanguage.wav` |
| 41 | `SP_HC_007.wav` | `AVPEPUDEAC0008_readtext.wav` | `AVPEPUDEAC0008` | **Spanish HC** | 17.91 | 1.000 | 0.141 | `SP_HC_007_to_German_true_crosslanguage.wav` |
| 42 | `SP_HC_008.wav` | `AVPEPUDEAC0010_readtext.wav` | `AVPEPUDEAC0010` | **Spanish HC** | 19.35 | 1.000 | 0.107 | `SP_HC_008_to_German_true_crosslanguage.wav` |
| 43 | `SP_HC_010.wav` | `AVPEPUDEAC0012_readtext.wav` | `AVPEPUDEAC0012` | **Spanish HC** | 17.36 | 1.000 | 0.135 | `SP_HC_010_to_German_true_crosslanguage.wav` |
| 44 | `SP_HC_012.wav` | `AVPEPUDEAC0014_readtext.wav` | `AVPEPUDEAC0014` | **Spanish HC** | 19.52 | 1.000 | 0.077 | `SP_HC_012_to_German_true_crosslanguage.wav` |
| 45 | `SP_HC_013.wav` | `AVPEPUDEAC0015_readtext.wav` | `AVPEPUDEAC0015` | **Spanish HC** | 15.92 | 1.000 | 0.128 | `SP_HC_013_to_German_true_crosslanguage.wav` |
| 46 | `SP_HC_016.wav` | `AVPEPUDEAC0018_readtext.wav` | `AVPEPUDEAC0018` | **Spanish HC** | 17.41 | 1.000 | 0.091 | `SP_HC_016_to_German_true_crosslanguage.wav` |
| 47 | `SP_HC_017.wav` | `AVPEPUDEAC0019_readtext.wav` | `AVPEPUDEAC0019` | **Spanish HC** | 18.31 | 1.000 | 0.100 | `SP_HC_017_to_German_true_crosslanguage.wav` |
| 48 | `SP_HC_019.wav` | `AVPEPUDEAC0021_readtext.wav` | `AVPEPUDEAC0021` | **Spanish HC** | 17.74 | 1.000 | 0.094 | `SP_HC_019_to_German_true_crosslanguage.wav` |
| 49 | `SP_HC_021.wav` | `AVPEPUDEAC0023_readtext.wav` | `AVPEPUDEAC0023` | **Spanish HC** | 16.07 | 1.000 | 0.076 | `SP_HC_021_to_German_true_crosslanguage.wav` |
| 50 | `SP_HC_023.wav` | `AVPEPUDEAC0025_readtext.wav` | `AVPEPUDEAC0025` | **Spanish HC** | 20.60 | 1.000 | 0.080 | `SP_HC_023_to_German_true_crosslanguage.wav` |
| 51 | `SP_HC_028.wav` | `AVPEPUDEAC0030_readtext.wav` | `AVPEPUDEAC0030` | **Spanish HC** | 15.19 | 1.000 | 0.092 | `SP_HC_028_to_German_true_crosslanguage.wav` |
| 52 | `SP_HC_029.wav` | `AVPEPUDEAC0031_readtext.wav` | `AVPEPUDEAC0031` | **Spanish HC** | 13.65 | 1.000 | 0.115 | `SP_HC_029_to_German_true_crosslanguage.wav` |
| 53 | `SP_HC_030.wav` | `AVPEPUDEAC0033_readtext.wav` | `AVPEPUDEAC0033` | **Spanish HC** | 16.45 | 1.000 | 0.125 | `SP_HC_030_to_German_true_crosslanguage.wav` |
| 54 | `SP_HC_031.wav` | `AVPEPUDEAC0034_readtext.wav` | `AVPEPUDEAC0034` | **Spanish HC** | 20.83 | 1.000 | 0.099 | `SP_HC_031_to_German_true_crosslanguage.wav` |
| 55 | `SP_HC_037.wav` | `AVPEPUDEAC0042_readtext.wav` | `AVPEPUDEAC0042` | **Spanish HC** | 20.06 | 1.000 | 0.102 | `SP_HC_037_to_German_true_crosslanguage.wav` |
| 56 | `SP_HC_040.wav` | `AVPEPUDEAC0045_readtext.wav` | `AVPEPUDEAC0045` | **Spanish HC** | 20.27 | 1.000 | 0.084 | `SP_HC_040_to_German_true_crosslanguage.wav` |
| 57 | `SP_HC_043.wav` | `AVPEPUDEAC0048_readtext.wav` | `AVPEPUDEAC0048` | **Spanish HC** | 13.97 | 1.000 | 0.123 | `SP_HC_043_to_German_true_crosslanguage.wav` |
| 58 | `SP_HC_044.wav` | `AVPEPUDEAC0049_readtext.wav` | `AVPEPUDEAC0049` | **Spanish HC** | 23.31 | 1.000 | 0.108 | `SP_HC_044_to_German_true_crosslanguage.wav` |
| 59 | `SP_HC_047.wav` | `AVPEPUDEAC0052_readtext.wav` | `AVPEPUDEAC0052` | **Spanish HC** | 12.16 | 1.000 | 0.091 | `SP_HC_047_to_German_true_crosslanguage.wav` |
| 60 | `SP_HC_048.wav` | `AVPEPUDEAC0053_readtext.wav` | `AVPEPUDEAC0053` | **Spanish HC** | 16.33 | 1.000 | 0.070 | `SP_HC_048_to_German_true_crosslanguage.wav` |
| 61 | `SP_PD_004.wav` | `AVPEPUDEA0005_readtext.wav` | `AVPEPUDEA0005` | **Spanish PD** | 20.61 | 1.000 | 0.078 | `SP_PD_004_to_German_true_crosslanguage.wav` |
| 62 | `SP_PD_006.wav` | `AVPEPUDEA0007_readtext.wav` | `AVPEPUDEA0007` | **Spanish PD** | 23.64 | 1.000 | 0.067 | `SP_PD_006_to_German_true_crosslanguage.wav` |
| 63 | `SP_PD_008.wav` | `AVPEPUDEA0009_readtext.wav` | `AVPEPUDEA0009` | **Spanish PD** | 18.96 | 1.000 | 0.065 | `SP_PD_008_to_German_true_crosslanguage.wav` |
| 64 | `SP_PD_014.wav` | `AVPEPUDEA0016_readtext.wav` | `AVPEPUDEA0016` | **Spanish PD** | 20.54 | 1.000 | 0.069 | `SP_PD_014_to_German_true_crosslanguage.wav` |
| 65 | `SP_PD_016.wav` | `AVPEPUDEA0020_readtext.wav` | `AVPEPUDEA0020` | **Spanish PD** | 14.32 | 1.000 | 0.116 | `SP_PD_016_to_German_true_crosslanguage.wav` |
| 66 | `SP_PD_018.wav` | `AVPEPUDEA0022_readtext.wav` | `AVPEPUDEA0022` | **Spanish PD** | 12.28 | 1.000 | 0.087 | `SP_PD_018_to_German_true_crosslanguage.wav` |
| 67 | `SP_PD_022.wav` | `AVPEPUDEA0026_readtext.wav` | `AVPEPUDEA0026` | **Spanish PD** | 17.03 | 1.000 | 0.124 | `SP_PD_022_to_German_true_crosslanguage.wav` |
| 68 | `SP_PD_024.wav` | `AVPEPUDEA0029_readtext.wav` | `AVPEPUDEA0029` | **Spanish PD** | 18.58 | 1.000 | 0.102 | `SP_PD_024_to_German_true_crosslanguage.wav` |
| 69 | `SP_PD_026.wav` | `AVPEPUDEA0031_readtext.wav` | `AVPEPUDEA0031` | **Spanish PD** | 21.54 | 1.000 | 0.143 | `SP_PD_026_to_German_true_crosslanguage.wav` |
| 70 | `SP_PD_027.wav` | `AVPEPUDEA0032_readtext.wav` | `AVPEPUDEA0032` | **Spanish PD** | 14.42 | 1.000 | 0.115 | `SP_PD_027_to_German_true_crosslanguage.wav` |
| 71 | `SP_PD_030.wav` | `AVPEPUDEA0037_readtext.wav` | `AVPEPUDEA0037` | **Spanish PD** | 23.05 | 1.000 | 0.100 | `SP_PD_030_to_German_true_crosslanguage.wav` |
| 72 | `SP_PD_032.wav` | `AVPEPUDEA0039_readtext.wav` | `AVPEPUDEA0039` | **Spanish PD** | 20.03 | 1.000 | 0.103 | `SP_PD_032_to_German_true_crosslanguage.wav` |
| 73 | `SP_PD_036.wav` | `AVPEPUDEA0045_readtext.wav` | `AVPEPUDEA0045` | **Spanish PD** | 13.93 | 1.000 | 0.076 | `SP_PD_036_to_German_true_crosslanguage.wav` |
| 74 | `SP_PD_039.wav` | `AVPEPUDEA0048_readtext.wav` | `AVPEPUDEA0048` | **Spanish PD** | 12.54 | 1.000 | 0.132 | `SP_PD_039_to_German_true_crosslanguage.wav` |
| 75 | `SP_PD_042.wav` | `AVPEPUDEA0051_readtext.wav` | `AVPEPUDEA0051` | **Spanish PD** | 14.74 | 1.000 | 0.101 | `SP_PD_042_to_German_true_crosslanguage.wav` |
| 76 | `SP_PD_043.wav` | `AVPEPUDEA0052_readtext.wav` | `AVPEPUDEA0052` | **Spanish PD** | 14.03 | 1.000 | 0.134 | `SP_PD_043_to_German_true_crosslanguage.wav` |
| 77 | `SP_PD_044.wav` | `AVPEPUDEA0053_readtext.wav` | `AVPEPUDEA0053` | **Spanish PD** | 37.34 | 1.000 | 0.083 | `SP_PD_044_to_German_true_crosslanguage.wav` |
| 78 | `SP_PD_045.wav` | `AVPEPUDEA0054_readtext.wav` | `AVPEPUDEA0054` | **Spanish PD** | 17.29 | 1.000 | 0.098 | `SP_PD_045_to_German_true_crosslanguage.wav` |
| 79 | `SP_PD_048.wav` | `AVPEPUDEA0057_readtext.wav` | `AVPEPUDEA0057` | **Spanish PD** | 14.23 | 1.000 | 0.170 | `SP_PD_048_to_German_true_crosslanguage.wav` |
| 80 | `SP_PD_050.wav` | `AVPEPUDEA0059_readtext.wav` | `AVPEPUDEA0059` | **Spanish PD** | 26.36 | 1.000 | 0.110 | `SP_PD_050_to_German_true_crosslanguage.wav` |
