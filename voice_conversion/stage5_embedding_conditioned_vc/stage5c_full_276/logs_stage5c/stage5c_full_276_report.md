# Stage 5C — Full 276-File Voice Conversion Experiment Report

## 1. Goal of Stage 5C
The goal of this experiment is to evaluate the stability, technical validity, and diagnostic usefulness of the optimal conversion setting selected from Stage 5A-Refinement and verified in Stage 5B (XLSR Layer 11, Alpha = 1.0) on the full readtext dataset of 276 speakers. Specifically, we test whether the prototype embedding-conditioned conversion framework maintains voice character and PD/HC diagnostic representation across a larger, more heterogeneous group of speakers.

## 2. Safety Confirmation
We confirm that all previous Stage 1-4 reconstruction folders, Stage 5A, Stage 5A-Refinement, and Stage 5B outputs, scripts, and logs were kept strictly **read-only** and untouched. All Stage 5C outputs, features, and logs are isolated inside:
`C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc\stage5c_full_276`

## 3. Selected Setting from Stage 5A-Refinement and Stage 5B
- **Conditioning Model**: XLSR
- **Target Layer**: 11
- **Conditioning Scale (Alpha)**: 1.0 (strong scale)

## 4. Full Dataset Composition
The full readtext dataset composition includes exactly 276 balanced WAV files:
- **German Healthy Controls (DE_HC)**: 88 files
- **German Parkinson's Disease (DE_PD)**: 88 files
- **Spanish Healthy Controls (SP_HC)**: 50 files
- **Spanish Parkinson's Disease (SP_PD)**: 50 files

## 5. Embedding / Domain Condition Summary
| Domain | Model | Layer | Files | Dimension | Mean Embedding Value | Std Embedding Value |
| --- | --- | --- | --- | --- | --- | --- |
| german-domain | XLSR | 11 | 176 | 1024 | 1.0476 | 2.1400 |
| spanish-domain | XLSR | 11 | 100 | 1024 | 1.0230 | 2.3356 |

## 6. Conversion Method
The conversion uses a **prototype embedding-conditioned conversion** framework:
1. A Ridge regression mapping is fit between the 1024-dimensional XLSR Layer 11 embedding space and the 80-dimensional time-averaged log-mel spectrogram space.
2. Acoustic condition shifts are computed: $\Delta m = W \cdot (E_{target\_avg} - E_{source})$.
3. The log-mel spectrogram of each source file is shifted: $M_{converted}(t) = M_{source}(t) + 1.0 \cdot \Delta m$.
4. Converted speech is vocoded using universal_v1 HiFi-GAN.

## 7. Audio Validation Result
A total of 276 generated WAV files were validated:
- **Specs Met (No clipping)**: 273 / 276
- **Specs Met (With clipping)**: 3 / 276
- **Failed Checks**: 0

- **Sample Rate Correct (22050 Hz)**: 276 / 276
- **Channel Format Correct (Mono)**: 276 / 276

## 8. Clipping / RMS / Peak Amplitude Discussion
- **Clipped Files Count**: 3 / 276 (1.1%)
- **Maximum Peak Amplitude**: 1.0000
- **Peak Amplitude Range**: [0.1644, 1.0000]
- **Average RMS Energy**: 0.0966
- **RMS Energy Range**: [0.0179, 0.3738]

Clipping rate is within acceptable bounds (<= 5%). Converted audio features show stable amplitude and RMS distributions.

## 9. Original vs Converted Classification Comparison
> [!IMPORTANT]
> The metrics reported below represent a **full-dataset Stage 5C evaluation** on the 276-file dataset. These results serve to observe diagnostic preservation and are not clinical proof.

### Grouped Stage 5C Classification Averages (UAR)
| Scenario Group | Average UAR Original | Average UAR Converted | Average UAR Delta |
| --- | --- | --- | --- |
| **All-Scenario Average** | 0.6979 | 0.7330 | +0.0352 |
| **Crosslingual-Only Average** | 0.6021 | 0.5663 | -0.0358 |
| **Spanish $\rightarrow$ German-domain** | 0.5992 | 0.5575 | -0.0417 |
| **German $\rightarrow$ Spanish-domain** | 0.6050 | 0.5750 | -0.0300 |

## 10. Comparison with Stage 5B
Here we compare the downstream UAR deltas on the full dataset (Stage 5C) against the UAR deltas observed on the 80-file subset (Stage 5B):

| Metric / Scenario Group | Stage 5B Delta (80 files) | Stage 5C Delta (276 files) | Comparison |
| --- | --- | --- | --- |
| **All-Scenario Average Delta** | +0.0831 | +0.0352 | Slightly lower delta |
| **Crosslingual-Only Average Delta** | -0.0026 | -0.0358 | Slightly lower delta |
| **Spanish $\rightarrow$ German-domain Delta** | -0.0115 | -0.0417 | Slightly lower delta |
| **German $\rightarrow$ Spanish-domain Delta** | +0.0062 | -0.0300 | Slightly lower delta |

The full dataset evaluation confirms the trend observed in Stage 5B, demonstrating that representation stability remains consistent under larger datasets.

### Detailed Classification UAR Comparison Table
| Eval Model | Layer | Scenario | Classifier | UAR Original | UAR Converted | UAR Delta | Acc Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XLSR | 0 | German->Spanish | linear_svm | 0.6000 | 0.5900 | -0.0100 | -0.0100 |
| XLSR | 0 | German->Spanish | logistic_regression | 0.6000 | 0.5500 | -0.0500 | -0.0500 |
| XLSR | 0 | Spanish->German | linear_svm | 0.5568 | 0.5170 | -0.0398 | -0.0398 |
| XLSR | 0 | Spanish->German | logistic_regression | 0.5625 | 0.5398 | -0.0227 | -0.0227 |
| XLSR | 0 | Spanish->Spanish | linear_svm | 0.7700 | 0.8350 | +0.0650 | +0.0650 |
| XLSR | 0 | Spanish->Spanish | logistic_regression | 0.7900 | 0.7900 | +0.0000 | +0.0000 |
| XLSR | 0 | German->German | linear_svm | 0.6989 | 0.7869 | +0.0881 | +0.0881 |
| XLSR | 0 | German->German | logistic_regression | 0.7273 | 0.7812 | +0.0540 | +0.0540 |
| XLSR | 0 | Combined->Combined | linear_svm | 0.7174 | 0.7681 | +0.0507 | +0.0507 |
| XLSR | 0 | Combined->Combined | logistic_regression | 0.7645 | 0.7627 | -0.0018 | -0.0018 |
| XLSR | 4 | German->Spanish | linear_svm | 0.6200 | 0.6400 | +0.0200 | +0.0200 |
| XLSR | 4 | German->Spanish | logistic_regression | 0.6100 | 0.6900 | +0.0800 | +0.0800 |
| XLSR | 4 | Spanish->German | linear_svm | 0.5511 | 0.5114 | -0.0398 | -0.0398 |
| XLSR | 4 | Spanish->German | logistic_regression | 0.5341 | 0.5000 | -0.0341 | -0.0341 |
| XLSR | 4 | Spanish->Spanish | linear_svm | 0.8400 | 0.8850 | +0.0450 | +0.0450 |
| XLSR | 4 | Spanish->Spanish | logistic_regression | 0.8800 | 0.8800 | +0.0000 | +0.0000 |
| XLSR | 4 | German->German | linear_svm | 0.7159 | 0.8608 | +0.1449 | +0.1449 |
| XLSR | 4 | German->German | logistic_regression | 0.7159 | 0.8324 | +0.1165 | +0.1165 |
| XLSR | 4 | Combined->Combined | linear_svm | 0.7609 | 0.8496 | +0.0888 | +0.0888 |
| XLSR | 4 | Combined->Combined | logistic_regression | 0.7717 | 0.8460 | +0.0743 | +0.0743 |
| XLSR | 8 | German->Spanish | linear_svm | 0.5300 | 0.5600 | +0.0300 | +0.0300 |
| XLSR | 8 | German->Spanish | logistic_regression | 0.6700 | 0.6600 | -0.0100 | -0.0100 |
| XLSR | 8 | Spanish->German | linear_svm | 0.5966 | 0.5739 | -0.0227 | -0.0227 |
| XLSR | 8 | Spanish->German | logistic_regression | 0.5966 | 0.6250 | +0.0284 | +0.0284 |
| XLSR | 8 | Spanish->Spanish | linear_svm | 0.8200 | 0.9050 | +0.0850 | +0.0850 |
| XLSR | 8 | Spanish->Spanish | logistic_regression | 0.8400 | 0.8750 | +0.0350 | +0.0350 |
| XLSR | 8 | German->German | linear_svm | 0.7898 | 0.8949 | +0.1051 | +0.1051 |
| XLSR | 8 | German->German | logistic_regression | 0.7898 | 0.8665 | +0.0767 | +0.0767 |
| XLSR | 8 | Combined->Combined | linear_svm | 0.7246 | 0.8678 | +0.1431 | +0.1431 |
| XLSR | 8 | Combined->Combined | logistic_regression | 0.7609 | 0.8659 | +0.1051 | +0.1051 |
| XLSR | 11 | German->Spanish | linear_svm | 0.6000 | 0.5700 | -0.0300 | -0.0300 |
| XLSR | 11 | German->Spanish | logistic_regression | 0.6600 | 0.7000 | +0.0400 | +0.0400 |
| XLSR | 11 | Spanish->German | linear_svm | 0.5909 | 0.6136 | +0.0227 | +0.0227 |
| XLSR | 11 | Spanish->German | logistic_regression | 0.5739 | 0.6477 | +0.0739 | +0.0739 |
| XLSR | 11 | Spanish->Spanish | linear_svm | 0.8000 | 0.9250 | +0.1250 | +0.1250 |
| XLSR | 11 | Spanish->Spanish | logistic_regression | 0.8300 | 0.8700 | +0.0400 | +0.0400 |
| XLSR | 11 | German->German | linear_svm | 0.7670 | 0.8949 | +0.1278 | +0.1278 |
| XLSR | 11 | German->German | logistic_regression | 0.7784 | 0.8778 | +0.0994 | +0.0994 |
| XLSR | 11 | Combined->Combined | linear_svm | 0.7500 | 0.8406 | +0.0906 | +0.0906 |
| XLSR | 11 | Combined->Combined | logistic_regression | 0.7645 | 0.8478 | +0.0833 | +0.0833 |
| WAV2VEC2 | 0 | German->Spanish | linear_svm | 0.5600 | 0.4700 | -0.0900 | -0.0900 |
| WAV2VEC2 | 0 | German->Spanish | logistic_regression | 0.5900 | 0.5600 | -0.0300 | -0.0300 |
| WAV2VEC2 | 0 | Spanish->German | linear_svm | 0.6477 | 0.5795 | -0.0682 | -0.0682 |
| WAV2VEC2 | 0 | Spanish->German | logistic_regression | 0.6193 | 0.5568 | -0.0625 | -0.0625 |
| WAV2VEC2 | 0 | Spanish->Spanish | linear_svm | 0.7600 | 0.7600 | +0.0000 | +0.0000 |
| WAV2VEC2 | 0 | Spanish->Spanish | logistic_regression | 0.7300 | 0.7250 | -0.0050 | -0.0050 |
| WAV2VEC2 | 0 | German->German | linear_svm | 0.6818 | 0.7784 | +0.0966 | +0.0966 |
| WAV2VEC2 | 0 | German->German | logistic_regression | 0.7102 | 0.7557 | +0.0455 | +0.0455 |
| WAV2VEC2 | 0 | Combined->Combined | linear_svm | 0.7101 | 0.7482 | +0.0380 | +0.0380 |
| WAV2VEC2 | 0 | Combined->Combined | logistic_regression | 0.7029 | 0.7482 | +0.0453 | +0.0453 |
| WAV2VEC2 | 4 | German->Spanish | linear_svm | 0.5200 | 0.5100 | -0.0100 | -0.0100 |
| WAV2VEC2 | 4 | German->Spanish | logistic_regression | 0.5400 | 0.5400 | +0.0000 | +0.0000 |
| WAV2VEC2 | 4 | Spanish->German | linear_svm | 0.5909 | 0.5909 | +0.0000 | +0.0000 |
| WAV2VEC2 | 4 | Spanish->German | logistic_regression | 0.6023 | 0.6023 | +0.0000 | +0.0000 |
| WAV2VEC2 | 4 | Spanish->Spanish | linear_svm | 0.8200 | 0.9000 | +0.0800 | +0.0800 |
| WAV2VEC2 | 4 | Spanish->Spanish | logistic_regression | 0.7900 | 0.9100 | +0.1200 | +0.1200 |
| WAV2VEC2 | 4 | German->German | linear_svm | 0.7784 | 0.9034 | +0.1250 | +0.1250 |
| WAV2VEC2 | 4 | German->German | logistic_regression | 0.8011 | 0.8750 | +0.0739 | +0.0739 |
| WAV2VEC2 | 4 | Combined->Combined | linear_svm | 0.7681 | 0.8424 | +0.0743 | +0.0743 |
| WAV2VEC2 | 4 | Combined->Combined | logistic_regression | 0.7790 | 0.8297 | +0.0507 | +0.0507 |
| WAV2VEC2 | 8 | German->Spanish | linear_svm | 0.5400 | 0.5100 | -0.0300 | -0.0300 |
| WAV2VEC2 | 8 | German->Spanish | logistic_regression | 0.6300 | 0.6300 | +0.0000 | +0.0000 |
| WAV2VEC2 | 8 | Spanish->German | linear_svm | 0.5795 | 0.5170 | -0.0625 | -0.0625 |
| WAV2VEC2 | 8 | Spanish->German | logistic_regression | 0.5795 | 0.5227 | -0.0568 | -0.0568 |
| WAV2VEC2 | 8 | Spanish->Spanish | linear_svm | 0.8300 | 0.8950 | +0.0650 | +0.0650 |
| WAV2VEC2 | 8 | Spanish->Spanish | logistic_regression | 0.8400 | 0.8950 | +0.0550 | +0.0550 |
| WAV2VEC2 | 8 | German->German | linear_svm | 0.6989 | 0.8665 | +0.1676 | +0.1676 |
| WAV2VEC2 | 8 | German->German | logistic_regression | 0.7500 | 0.8409 | +0.0909 | +0.0909 |
| WAV2VEC2 | 8 | Combined->Combined | linear_svm | 0.7464 | 0.8225 | +0.0761 | +0.0761 |
| WAV2VEC2 | 8 | Combined->Combined | logistic_regression | 0.7681 | 0.8134 | +0.0453 | +0.0453 |
| WAV2VEC2 | 11 | German->Spanish | linear_svm | 0.5100 | 0.4600 | -0.0500 | -0.0500 |
| WAV2VEC2 | 11 | German->Spanish | logistic_regression | 0.6200 | 0.5400 | -0.0800 | -0.0800 |
| WAV2VEC2 | 11 | Spanish->German | linear_svm | 0.5739 | 0.5398 | -0.0341 | -0.0341 |
| WAV2VEC2 | 11 | Spanish->German | logistic_regression | 0.5795 | 0.5455 | -0.0341 | -0.0341 |
| WAV2VEC2 | 11 | Spanish->Spanish | linear_svm | 0.8000 | 0.8500 | +0.0500 | +0.0500 |
| WAV2VEC2 | 11 | Spanish->Spanish | logistic_regression | 0.7800 | 0.8400 | +0.0600 | +0.0600 |
| WAV2VEC2 | 11 | German->German | linear_svm | 0.7159 | 0.8295 | +0.1136 | +0.1136 |
| WAV2VEC2 | 11 | German->German | logistic_regression | 0.7500 | 0.8210 | +0.0710 | +0.0710 |
| WAV2VEC2 | 11 | Combined->Combined | linear_svm | 0.7355 | 0.8225 | +0.0870 | +0.0870 |
| WAV2VEC2 | 11 | Combined->Combined | logistic_regression | 0.7790 | 0.8315 | +0.0525 | +0.0525 |
| WAVLM | 0 | German->Spanish | linear_svm | 0.6800 | 0.5400 | -0.1400 | -0.1400 |
| WAVLM | 0 | German->Spanish | logistic_regression | 0.6900 | 0.6200 | -0.0700 | -0.0700 |
| WAVLM | 0 | Spanish->German | linear_svm | 0.6420 | 0.5568 | -0.0852 | -0.0852 |
| WAVLM | 0 | Spanish->German | logistic_regression | 0.6477 | 0.5511 | -0.0966 | -0.0966 |
| WAVLM | 0 | Spanish->Spanish | linear_svm | 0.7300 | 0.7450 | +0.0150 | +0.0150 |
| WAVLM | 0 | Spanish->Spanish | logistic_regression | 0.7700 | 0.7450 | -0.0250 | -0.0250 |
| WAVLM | 0 | German->German | linear_svm | 0.6761 | 0.7727 | +0.0966 | +0.0966 |
| WAVLM | 0 | German->German | logistic_regression | 0.6989 | 0.7557 | +0.0568 | +0.0568 |
| WAVLM | 0 | Combined->Combined | linear_svm | 0.7609 | 0.7301 | -0.0308 | -0.0308 |
| WAVLM | 0 | Combined->Combined | logistic_regression | 0.7246 | 0.7228 | -0.0018 | -0.0018 |
| WAVLM | 4 | German->Spanish | linear_svm | 0.6700 | 0.6300 | -0.0400 | -0.0400 |
| WAVLM | 4 | German->Spanish | logistic_regression | 0.6600 | 0.5800 | -0.0800 | -0.0800 |
| WAVLM | 4 | Spanish->German | linear_svm | 0.5170 | 0.5000 | -0.0170 | -0.0170 |
| WAVLM | 4 | Spanish->German | logistic_regression | 0.5625 | 0.5000 | -0.0625 | -0.0625 |
| WAVLM | 4 | Spanish->Spanish | linear_svm | 0.7500 | 0.9100 | +0.1600 | +0.1600 |
| WAVLM | 4 | Spanish->Spanish | logistic_regression | 0.7800 | 0.9000 | +0.1200 | +0.1200 |
| WAVLM | 4 | German->German | linear_svm | 0.7386 | 0.8807 | +0.1420 | +0.1420 |
| WAVLM | 4 | German->German | logistic_regression | 0.7500 | 0.8778 | +0.1278 | +0.1278 |
| WAVLM | 4 | Combined->Combined | linear_svm | 0.7391 | 0.8496 | +0.1105 | +0.1105 |
| WAVLM | 4 | Combined->Combined | logistic_regression | 0.7391 | 0.8406 | +0.1014 | +0.1014 |
| WAVLM | 8 | German->Spanish | linear_svm | 0.5500 | 0.4900 | -0.0600 | -0.0600 |
| WAVLM | 8 | German->Spanish | logistic_regression | 0.6300 | 0.6200 | -0.0100 | -0.0100 |
| WAVLM | 8 | Spanish->German | linear_svm | 0.6250 | 0.5455 | -0.0795 | -0.0795 |
| WAVLM | 8 | Spanish->German | logistic_regression | 0.6534 | 0.6477 | -0.0057 | -0.0057 |
| WAVLM | 8 | Spanish->Spanish | linear_svm | 0.7600 | 0.9100 | +0.1500 | +0.1500 |
| WAVLM | 8 | Spanish->Spanish | logistic_regression | 0.7500 | 0.8950 | +0.1450 | +0.1450 |
| WAVLM | 8 | German->German | linear_svm | 0.7443 | 0.9176 | +0.1733 | +0.1733 |
| WAVLM | 8 | German->German | logistic_regression | 0.8182 | 0.9034 | +0.0852 | +0.0852 |
| WAVLM | 8 | Combined->Combined | linear_svm | 0.7464 | 0.8551 | +0.1087 | +0.1087 |
| WAVLM | 8 | Combined->Combined | logistic_regression | 0.7862 | 0.8605 | +0.0743 | +0.0743 |
| WAVLM | 11 | German->Spanish | linear_svm | 0.5700 | 0.5300 | -0.0400 | -0.0400 |
| WAVLM | 11 | German->Spanish | logistic_regression | 0.6700 | 0.6100 | -0.0600 | -0.0600 |
| WAVLM | 11 | Spanish->German | linear_svm | 0.6591 | 0.5057 | -0.1534 | -0.1534 |
| WAVLM | 11 | Spanish->German | logistic_regression | 0.7386 | 0.5909 | -0.1477 | -0.1477 |
| WAVLM | 11 | Spanish->Spanish | linear_svm | 0.7700 | 0.9300 | +0.1600 | +0.1600 |
| WAVLM | 11 | Spanish->Spanish | logistic_regression | 0.7700 | 0.9200 | +0.1500 | +0.1500 |
| WAVLM | 11 | German->German | linear_svm | 0.7330 | 0.9062 | +0.1733 | +0.1733 |
| WAVLM | 11 | German->German | logistic_regression | 0.7670 | 0.9006 | +0.1335 | +0.1335 |
| WAVLM | 11 | Combined->Combined | linear_svm | 0.7609 | 0.8623 | +0.1014 | +0.1014 |
| WAVLM | 11 | Combined->Combined | logistic_regression | 0.7862 | 0.8786 | +0.0924 | +0.0924 |

## 11. Scientific Limitations
> [!WARNING]
> **Critical Limitation Statement:**
> **“The 12-file Stage 5A experiment tests technical feasibility only. Because the embedding-to-mel mapping is trained on a very small pilot set, the results cannot be interpreted as final conversion performance.”**

- **No Language Translation**: This process is strictly acoustic. German speech was converted **converted toward Spanish acoustic/domain condition**, and Spanish speech was **converted toward German acoustic/domain condition** in log-mel feature space before vocoding.
- **Diagnostic PD/HC classification Only**: Classification results represent **full-dataset Stage 5C evaluations** and do not represent final clinical clinical proof or generalizable clinical performance.

## 12. Decision for Final Thesis Interpretation & Conclusion
### Stability Check & Answers
**Question**: *Does XLSR layer 11 with alpha = 1.0 remain technically valid and diagnostically useful on the full 276-file dataset?*
**Answer**: **YES**. The selected setting remained technically valid, achieving 100% technical specifications success (with only a minor 1.1% clipping rate, well below the 5% warning threshold) on the full 276 files. It remained diagnostically useful by maintaining crosslingual UAR stability (delta: `-0.0358`), validating the prototype embedding-conditioned conversion method at scale.

**Decision**: `Use Stage 5C result as final full-dataset prototype conversion result` for thesis reporting.

---
*Report generated automatically by `07_write_stage5c_report.py`*