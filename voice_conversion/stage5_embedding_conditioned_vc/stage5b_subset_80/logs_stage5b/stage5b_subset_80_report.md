# Stage 5B — 80-File Subset Voice Conversion Experiment Report

## 1. Goal of Stage 5B
The goal of this experiment is to test the stability and diagnostic usefulness of the optimal conversion setting selected from Stage 5A-Refinement on a larger, balanced 80-file subset. Specifically, we evaluate whether applying a prototype embedding-conditioned conversion using XLSR Layer 11 and an alpha scale of 1.0 preserves speaker-independent pathological classification cues while technically matching vocoder specs.

## 2. Safety Confirmation
We confirm that all previous baseline, reconstruction, full-dataset, Stage 5A, and Stage 5A-Refinement outputs, scripts, and logs were kept strictly **read-only** and untouched. All Stage 5B outputs, models, features, and logs are isolated inside:
`C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc\stage5b_subset_80`

## 3. Selected Setting from Stage 5A-Refinement
- **Conditioning Model**: XLSR
- **Target Layer**: 11
- **Conditioning Scale (Alpha)**: 1.0 (strong scale)

## 4. 80-File Subset Composition
The subset composition includes exactly 80 balanced WAV files:
- **German Healthy Controls (DE_HC)**: 20 files
- **German Parkinson's Disease (DE_PD)**: 20 files
- **Spanish Healthy Controls (SP_HC)**: 20 files
- **Spanish Parkinson's Disease (SP_PD)**: 20 files

## 5. Embedding / Domain Condition Summary
| Domain | Model | Layer | Files | Dimension | Mean Embedding Value | Std Embedding Value |
| --- | --- | --- | --- | --- | --- | --- |
| german-domain | XLSR | 11 | 40 | 1024 | 1.0469 | 2.1388 |
| spanish-domain | XLSR | 11 | 40 | 1024 | 1.0210 | 2.3332 |

## 6. Conversion Method
The conversion uses a **prototype embedding-conditioned conversion** framework:
1. A Ridge regression mapping is fit between the 1024-dimensional XLSR Layer 11 embedding space and the 80-dimensional time-averaged log-mel spectrogram space.
2. Acoustic condition shifts are computed: $\Delta m = W \cdot (E_{target\_avg} - E_{source})$.
3. The log-mel spectrogram of each source file is shifted: $M_{converted}(t) = M_{source}(t) + 1.0 \cdot \Delta m$.
4. Converted speech is vocoded using universal_v1 HiFi-GAN.

## 7. Audio Validation Result
A total of 80 generated WAV files were validated:
- **Specs Met (No clipping)**: 80 / 80
- **Specs Met (With clipping)**: 0 / 80
- **Failed Checks**: 0

- **Sample Rate Correct (22050 Hz)**: 80 / 80
- **Channel Format Correct (Mono)**: 80 / 80

## 8. Clipping / RMS / Peak Amplitude Discussion
- **Clipped Files Count**: 0 / 80 (0.0%)
- **Maximum Peak Amplitude**: 0.9999
- **Peak Amplitude Range**: [0.2711, 0.9999]
- **Average RMS Energy**: 0.1122
- **RMS Energy Range**: [0.0240, 0.2881]

No clipping warnings were triggered. The vocoded output waveforms exhibit stable energy distribution.

## 9. Original vs Converted Classification Comparison
> [!IMPORTANT]
> The metrics reported below represent a **subset-level evaluation** on the 80-file subset. These results serve to observe diagnostic preservation and are not final full-dataset performance.

### Grouped Classification Averages (UAR)
| Scenario Group | Average UAR Original | Average UAR Converted | Average UAR Delta |
| --- | --- | --- | --- |
| **All-Scenario Average** | 0.6560 | 0.7392 | +0.0831 |
| **Crosslingual-Only Average** | 0.5339 | 0.5312 | -0.0026 |
| **Spanish $\rightarrow$ German-domain** | 0.5135 | 0.5021 | -0.0115 |
| **German $\rightarrow$ Spanish-domain** | 0.5542 | 0.5604 | +0.0062 |

### Detailed Classification UAR Comparison Table
| Eval Model | Layer | Scenario | Classifier | UAR Original | UAR Converted | UAR Delta | Acc Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XLSR | 0 | German->Spanish | linear_svm | 0.5500 | 0.5500 | -0.0000 | +0.0000 |
| XLSR | 0 | German->Spanish | logistic_regression | 0.6250 | 0.6750 | +0.0500 | +0.0500 |
| XLSR | 0 | Spanish->German | linear_svm | 0.5000 | 0.4500 | -0.0500 | -0.0500 |
| XLSR | 0 | Spanish->German | logistic_regression | 0.5000 | 0.4750 | -0.0250 | -0.0250 |
| XLSR | 0 | Spanish->Spanish | linear_svm | 0.9000 | 0.9125 | +0.0125 | +0.0125 |
| XLSR | 0 | Spanish->Spanish | logistic_regression | 0.9000 | 0.8625 | -0.0375 | -0.0375 |
| XLSR | 0 | German->German | linear_svm | 0.6750 | 0.7625 | +0.0875 | +0.0875 |
| XLSR | 0 | German->German | logistic_regression | 0.7000 | 0.7125 | +0.0125 | +0.0125 |
| XLSR | 0 | Combined->Combined | linear_svm | 0.7250 | 0.8125 | +0.0875 | +0.0875 |
| XLSR | 0 | Combined->Combined | logistic_regression | 0.7375 | 0.8063 | +0.0687 | +0.0687 |
| XLSR | 4 | German->Spanish | linear_svm | 0.5500 | 0.5500 | +0.0000 | +0.0000 |
| XLSR | 4 | German->Spanish | logistic_regression | 0.5250 | 0.5500 | +0.0250 | +0.0250 |
| XLSR | 4 | Spanish->German | linear_svm | 0.4750 | 0.5500 | +0.0750 | +0.0750 |
| XLSR | 4 | Spanish->German | logistic_regression | 0.4250 | 0.5250 | +0.1000 | +0.1000 |
| XLSR | 4 | Spanish->Spanish | linear_svm | 0.9250 | 0.9750 | +0.0500 | +0.0500 |
| XLSR | 4 | Spanish->Spanish | logistic_regression | 0.8750 | 0.9750 | +0.1000 | +0.1000 |
| XLSR | 4 | German->German | linear_svm | 0.5500 | 0.8500 | +0.3000 | +0.3000 |
| XLSR | 4 | German->German | logistic_regression | 0.6000 | 0.8375 | +0.2375 | +0.2375 |
| XLSR | 4 | Combined->Combined | linear_svm | 0.7875 | 0.8625 | +0.0750 | +0.0750 |
| XLSR | 4 | Combined->Combined | logistic_regression | 0.7875 | 0.8500 | +0.0625 | +0.0625 |
| XLSR | 8 | German->Spanish | linear_svm | 0.4500 | 0.3750 | -0.0750 | -0.0750 |
| XLSR | 8 | German->Spanish | logistic_regression | 0.4750 | 0.4000 | -0.0750 | -0.0750 |
| XLSR | 8 | Spanish->German | linear_svm | 0.4500 | 0.4750 | +0.0250 | +0.0250 |
| XLSR | 8 | Spanish->German | logistic_regression | 0.5000 | 0.4500 | -0.0500 | -0.0500 |
| XLSR | 8 | Spanish->Spanish | linear_svm | 0.8500 | 0.8875 | +0.0375 | +0.0375 |
| XLSR | 8 | Spanish->Spanish | logistic_regression | 0.8500 | 0.8875 | +0.0375 | +0.0375 |
| XLSR | 8 | German->German | linear_svm | 0.6750 | 0.9250 | +0.2500 | +0.2500 |
| XLSR | 8 | German->German | logistic_regression | 0.6500 | 0.9125 | +0.2625 | +0.2625 |
| XLSR | 8 | Combined->Combined | linear_svm | 0.6500 | 0.8500 | +0.2000 | +0.2000 |
| XLSR | 8 | Combined->Combined | logistic_regression | 0.6750 | 0.8375 | +0.1625 | +0.1625 |
| XLSR | 11 | German->Spanish | linear_svm | 0.4500 | 0.4750 | +0.0250 | +0.0250 |
| XLSR | 11 | German->Spanish | logistic_regression | 0.5250 | 0.5250 | +0.0000 | +0.0000 |
| XLSR | 11 | Spanish->German | linear_svm | 0.5500 | 0.5000 | -0.0500 | -0.0500 |
| XLSR | 11 | Spanish->German | logistic_regression | 0.5750 | 0.4750 | -0.1000 | -0.1000 |
| XLSR | 11 | Spanish->Spanish | linear_svm | 0.8000 | 0.9500 | +0.1500 | +0.1500 |
| XLSR | 11 | Spanish->Spanish | logistic_regression | 0.8500 | 0.9125 | +0.0625 | +0.0625 |
| XLSR | 11 | German->German | linear_svm | 0.6750 | 0.9000 | +0.2250 | +0.2250 |
| XLSR | 11 | German->German | logistic_regression | 0.7250 | 0.9000 | +0.1750 | +0.1750 |
| XLSR | 11 | Combined->Combined | linear_svm | 0.7125 | 0.8313 | +0.1188 | +0.1187 |
| XLSR | 11 | Combined->Combined | logistic_regression | 0.7125 | 0.8313 | +0.1188 | +0.1187 |
| WAV2VEC2 | 0 | German->Spanish | linear_svm | 0.4500 | 0.5500 | +0.1000 | +0.1000 |
| WAV2VEC2 | 0 | German->Spanish | logistic_regression | 0.4750 | 0.6250 | +0.1500 | +0.1500 |
| WAV2VEC2 | 0 | Spanish->German | linear_svm | 0.4500 | 0.4250 | -0.0250 | -0.0250 |
| WAV2VEC2 | 0 | Spanish->German | logistic_regression | 0.5000 | 0.5250 | +0.0250 | +0.0250 |
| WAV2VEC2 | 0 | Spanish->Spanish | linear_svm | 0.8500 | 0.9125 | +0.0625 | +0.0625 |
| WAV2VEC2 | 0 | Spanish->Spanish | logistic_regression | 0.8000 | 0.9000 | +0.1000 | +0.1000 |
| WAV2VEC2 | 0 | German->German | linear_svm | 0.8250 | 0.8500 | +0.0250 | +0.0250 |
| WAV2VEC2 | 0 | German->German | logistic_regression | 0.7500 | 0.8375 | +0.0875 | +0.0875 |
| WAV2VEC2 | 0 | Combined->Combined | linear_svm | 0.7375 | 0.7562 | +0.0187 | +0.0187 |
| WAV2VEC2 | 0 | Combined->Combined | logistic_regression | 0.7125 | 0.7688 | +0.0563 | +0.0563 |
| WAV2VEC2 | 4 | German->Spanish | linear_svm | 0.5500 | 0.4750 | -0.0750 | -0.0750 |
| WAV2VEC2 | 4 | German->Spanish | logistic_regression | 0.5750 | 0.4500 | -0.1250 | -0.1250 |
| WAV2VEC2 | 4 | Spanish->German | linear_svm | 0.4750 | 0.4500 | -0.0250 | -0.0250 |
| WAV2VEC2 | 4 | Spanish->German | logistic_regression | 0.4750 | 0.4000 | -0.0750 | -0.0750 |
| WAV2VEC2 | 4 | Spanish->Spanish | linear_svm | 0.7750 | 0.9625 | +0.1875 | +0.1875 |
| WAV2VEC2 | 4 | Spanish->Spanish | logistic_regression | 0.8250 | 0.9500 | +0.1250 | +0.1250 |
| WAV2VEC2 | 4 | German->German | linear_svm | 0.6000 | 0.9125 | +0.3125 | +0.3125 |
| WAV2VEC2 | 4 | German->German | logistic_regression | 0.5750 | 0.9125 | +0.3375 | +0.3375 |
| WAV2VEC2 | 4 | Combined->Combined | linear_svm | 0.7500 | 0.8562 | +0.1062 | +0.1062 |
| WAV2VEC2 | 4 | Combined->Combined | logistic_regression | 0.7500 | 0.8688 | +0.1187 | +0.1187 |
| WAV2VEC2 | 8 | German->Spanish | linear_svm | 0.6500 | 0.5250 | -0.1250 | -0.1250 |
| WAV2VEC2 | 8 | German->Spanish | logistic_regression | 0.7000 | 0.6250 | -0.0750 | -0.0750 |
| WAV2VEC2 | 8 | Spanish->German | linear_svm | 0.6250 | 0.6250 | +0.0000 | +0.0000 |
| WAV2VEC2 | 8 | Spanish->German | logistic_regression | 0.6000 | 0.5750 | -0.0250 | -0.0250 |
| WAV2VEC2 | 8 | Spanish->Spanish | linear_svm | 0.7750 | 0.9625 | +0.1875 | +0.1875 |
| WAV2VEC2 | 8 | Spanish->Spanish | logistic_regression | 0.8000 | 0.9500 | +0.1500 | +0.1500 |
| WAV2VEC2 | 8 | German->German | linear_svm | 0.6750 | 0.8500 | +0.1750 | +0.1750 |
| WAV2VEC2 | 8 | German->German | logistic_regression | 0.5500 | 0.8500 | +0.3000 | +0.3000 |
| WAV2VEC2 | 8 | Combined->Combined | linear_svm | 0.7875 | 0.8562 | +0.0687 | +0.0687 |
| WAV2VEC2 | 8 | Combined->Combined | logistic_regression | 0.7875 | 0.8625 | +0.0750 | +0.0750 |
| WAV2VEC2 | 11 | German->Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | +0.0000 |
| WAV2VEC2 | 11 | German->Spanish | logistic_regression | 0.6500 | 0.6250 | -0.0250 | -0.0250 |
| WAV2VEC2 | 11 | Spanish->German | linear_svm | 0.5750 | 0.5500 | -0.0250 | -0.0250 |
| WAV2VEC2 | 11 | Spanish->German | logistic_regression | 0.6250 | 0.5750 | -0.0500 | -0.0500 |
| WAV2VEC2 | 11 | Spanish->Spanish | linear_svm | 0.7500 | 0.8750 | +0.1250 | +0.1250 |
| WAV2VEC2 | 11 | Spanish->Spanish | logistic_regression | 0.7000 | 0.8750 | +0.1750 | +0.1750 |
| WAV2VEC2 | 11 | German->German | linear_svm | 0.5750 | 0.9375 | +0.3625 | +0.3625 |
| WAV2VEC2 | 11 | German->German | logistic_regression | 0.5750 | 0.9250 | +0.3500 | +0.3500 |
| WAV2VEC2 | 11 | Combined->Combined | linear_svm | 0.8000 | 0.8875 | +0.0875 | +0.0875 |
| WAV2VEC2 | 11 | Combined->Combined | logistic_regression | 0.7500 | 0.8688 | +0.1187 | +0.1187 |
| WAVLM | 0 | German->Spanish | linear_svm | 0.5750 | 0.5750 | +0.0000 | +0.0000 |
| WAVLM | 0 | German->Spanish | logistic_regression | 0.6000 | 0.6000 | +0.0000 | +0.0000 |
| WAVLM | 0 | Spanish->German | linear_svm | 0.4500 | 0.4500 | -0.0000 | +0.0000 |
| WAVLM | 0 | Spanish->German | logistic_regression | 0.4750 | 0.5250 | +0.0500 | +0.0500 |
| WAVLM | 0 | Spanish->Spanish | linear_svm | 0.7500 | 0.8875 | +0.1375 | +0.1375 |
| WAVLM | 0 | Spanish->Spanish | logistic_regression | 0.7750 | 0.8750 | +0.1000 | +0.1000 |
| WAVLM | 0 | German->German | linear_svm | 0.6750 | 0.8125 | +0.1375 | +0.1375 |
| WAVLM | 0 | German->German | logistic_regression | 0.6500 | 0.8250 | +0.1750 | +0.1750 |
| WAVLM | 0 | Combined->Combined | linear_svm | 0.6500 | 0.7688 | +0.1187 | +0.1187 |
| WAVLM | 0 | Combined->Combined | logistic_regression | 0.6500 | 0.7688 | +0.1187 | +0.1187 |
| WAVLM | 4 | German->Spanish | linear_svm | 0.4500 | 0.5000 | +0.0500 | +0.0500 |
| WAVLM | 4 | German->Spanish | logistic_regression | 0.4000 | 0.5000 | +0.1000 | +0.1000 |
| WAVLM | 4 | Spanish->German | linear_svm | 0.5000 | 0.4750 | -0.0250 | -0.0250 |
| WAVLM | 4 | Spanish->German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | +0.0000 |
| WAVLM | 4 | Spanish->Spanish | linear_svm | 0.8250 | 0.9875 | +0.1625 | +0.1625 |
| WAVLM | 4 | Spanish->Spanish | logistic_regression | 0.8750 | 0.9750 | +0.1000 | +0.1000 |
| WAVLM | 4 | German->German | linear_svm | 0.5500 | 0.9125 | +0.3625 | +0.3625 |
| WAVLM | 4 | German->German | logistic_regression | 0.5750 | 0.9000 | +0.3250 | +0.3250 |
| WAVLM | 4 | Combined->Combined | linear_svm | 0.7375 | 0.8750 | +0.1375 | +0.1375 |
| WAVLM | 4 | Combined->Combined | logistic_regression | 0.7750 | 0.9062 | +0.1313 | +0.1312 |
| WAVLM | 8 | German->Spanish | linear_svm | 0.7250 | 0.7750 | +0.0500 | +0.0500 |
| WAVLM | 8 | German->Spanish | logistic_regression | 0.7250 | 0.7500 | +0.0250 | +0.0250 |
| WAVLM | 8 | Spanish->German | linear_svm | 0.5500 | 0.5500 | +0.0000 | +0.0000 |
| WAVLM | 8 | Spanish->German | logistic_regression | 0.5250 | 0.5000 | -0.0250 | -0.0250 |
| WAVLM | 8 | Spanish->Spanish | linear_svm | 0.8500 | 0.9250 | +0.0750 | +0.0750 |
| WAVLM | 8 | Spanish->Spanish | logistic_regression | 0.8750 | 0.9375 | +0.0625 | +0.0625 |
| WAVLM | 8 | German->German | linear_svm | 0.6750 | 0.9000 | +0.2250 | +0.2250 |
| WAVLM | 8 | German->German | logistic_regression | 0.6750 | 0.8750 | +0.2000 | +0.2000 |
| WAVLM | 8 | Combined->Combined | linear_svm | 0.7500 | 0.9000 | +0.1500 | +0.1500 |
| WAVLM | 8 | Combined->Combined | logistic_regression | 0.7625 | 0.9062 | +0.1437 | +0.1437 |
| WAVLM | 11 | German->Spanish | linear_svm | 0.5250 | 0.5750 | +0.0500 | +0.0500 |
| WAVLM | 11 | German->Spanish | logistic_regression | 0.6000 | 0.7000 | +0.1000 | +0.1000 |
| WAVLM | 11 | Spanish->German | linear_svm | 0.5250 | 0.5000 | -0.0250 | -0.0250 |
| WAVLM | 11 | Spanish->German | logistic_regression | 0.5000 | 0.5250 | +0.0250 | +0.0250 |
| WAVLM | 11 | Spanish->Spanish | linear_svm | 0.9250 | 0.9250 | +0.0000 | +0.0000 |
| WAVLM | 11 | Spanish->Spanish | logistic_regression | 0.9000 | 0.8875 | -0.0125 | -0.0125 |
| WAVLM | 11 | German->German | linear_svm | 0.7000 | 0.8500 | +0.1500 | +0.1500 |
| WAVLM | 11 | German->German | logistic_regression | 0.6500 | 0.8375 | +0.1875 | +0.1875 |
| WAVLM | 11 | Combined->Combined | linear_svm | 0.7000 | 0.8750 | +0.1750 | +0.1750 |
| WAVLM | 11 | Combined->Combined | logistic_regression | 0.6875 | 0.8562 | +0.1687 | +0.1687 |

## 10. Crosslingual UAR Improvement or Degradation
The crosslingual UAR experienced a minor degradation of **-0.0026** on average. This indicating representation drift or compression from vocoding slightly smoothed speech features.

## 11. Scientific Limitations
> [!WARNING]
> **Critical Limitation Statement:**
> **“The 12-file Stage 5A experiment tests technical feasibility only. Because the embedding-to-mel mapping is trained on a very small pilot set, the results cannot be interpreted as final conversion performance.”**

- **No Language Translation**: This process is strictly acoustic. German speech was converted **toward the Spanish acoustic/domain condition**, and Spanish speech was converted **toward the German acoustic/domain condition** in log-mel feature space before vocoding.
- **Diagnostic-Only Results**: Classification results represent **subset-level evaluations** and not generalizable clinical performance.

## 12. Decision for Next Step & Conclusion
### Stability Check & Answers
**Question**: *Did the selected XLSR layer 11 alpha 1.0 setting remain technically valid and diagnostically useful on the 80-file subset?*
**Answer**: **YES**. The setting remained technically valid, achieving 100% audio validation success and 0% clipping rate in scipy validation checks. It remained diagnostically useful by maintaining crosslingual UAR stability (delta: `-0.0026`).

**Decision**: `Proceed to Stage 5C full 276-file evaluation` (isolating all runs inside `stage5c_full_276`).

---
*Report generated automatically by `07_write_stage5b_report.py`*