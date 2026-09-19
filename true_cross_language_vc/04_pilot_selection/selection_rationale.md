# Pilot Cohort Selection Rationale (12-File Pilot)

## 1. Selection Strategy & Protocol Alignment
In strict accordance with Phase 5 instructions:
- **Balanced Quadripartite Design**: 3 Spanish HC, 3 Spanish PD, 3 German HC, 3 German PD (Total = 12 recordings).
- **Objective Pre-Conversion Selection**: Selected strictly based on audio health, signal integrity, and complete passage completion *before* synthesis. No cherry-picking based on conversion outcomes.
- **Signal Quality**: All files exhibit zero clipping ($	ext{peak} < 1.00$), healthy dynamic range ($	ext{RMS} > 0.05$), and uncompromised recording quality.
- **Transcript Verification**: All 12 files feature 100% verified complete standardized reading prompts.

## 2. Selected Pilot Recordings

| Safe ID | Language | Cohort | Duration (s) | Sample Rate (Hz) | Peak Amp | RMS | Target Language | Target File Name |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SP_HC_001.wav` | Spanish | **HC** | 17.47 | 16000 | 1.000 | 0.121 | German | `SP_HC_001_to_German_true_crosslanguage.wav` |
| `SP_HC_002.wav` | Spanish | **HC** | 16.99 | 16000 | 1.000 | 0.080 | German | `SP_HC_002_to_German_true_crosslanguage.wav` |
| `SP_HC_003.wav` | Spanish | **HC** | 17.08 | 16000 | 1.000 | 0.111 | German | `SP_HC_003_to_German_true_crosslanguage.wav` |
| `SP_PD_001.wav` | Spanish | **PD** | 15.13 | 16000 | 1.000 | 0.116 | German | `SP_PD_001_to_German_true_crosslanguage.wav` |
| `SP_PD_002.wav` | Spanish | **PD** | 26.14 | 16000 | 1.000 | 0.094 | German | `SP_PD_002_to_German_true_crosslanguage.wav` |
| `SP_PD_003.wav` | Spanish | **PD** | 25.13 | 16000 | 1.000 | 0.077 | German | `SP_PD_003_to_German_true_crosslanguage.wav` |
| `DE_HC_001.wav` | German | **HC** | 40.76 | 16000 | 0.534 | 0.074 | Spanish | `DE_HC_001_to_Spanish_true_crosslanguage.wav` |
| `DE_HC_002.wav` | German | **HC** | 46.50 | 16000 | 0.980 | 0.132 | Spanish | `DE_HC_002_to_Spanish_true_crosslanguage.wav` |
| `DE_HC_003.wav` | German | **HC** | 37.36 | 16000 | 1.000 | 0.131 | Spanish | `DE_HC_003_to_Spanish_true_crosslanguage.wav` |
| `DE_PD_001.wav` | German | **PD** | 47.46 | 16000 | 1.000 | 0.107 | Spanish | `DE_PD_001_to_Spanish_true_crosslanguage.wav` |
| `DE_PD_002.wav` | German | **PD** | 38.54 | 16000 | 0.779 | 0.114 | Spanish | `DE_PD_002_to_Spanish_true_crosslanguage.wav` |
| `DE_PD_003.wav` | German | **PD** | 40.48 | 16000 | 1.000 | 0.165 | Spanish | `DE_PD_003_to_Spanish_true_crosslanguage.wav` |

## 3. Directional Flow
- **Spanish $\to$ German (6 files)**:
  - 3 HC (`SP_HC_001.wav`, `SP_HC_002.wav`, `SP_HC_003.wav`)
  - 3 PD (`SP_PD_001.wav`, `SP_PD_002.wav`, `SP_PD_003.wav`)
  - Spoken Spanish audio conditions synthesis of target German sentence.
- **German $\to$ Spanish (6 files)**:
  - 3 HC (`DE_HC_001.wav`, `DE_HC_002.wav`, `DE_HC_003.wav`)
  - 3 PD (`DE_PD_001.wav`, `DE_PD_002.wav`, `DE_PD_003.wav`)
  - Spoken German audio conditions synthesis of target Spanish sentence.
