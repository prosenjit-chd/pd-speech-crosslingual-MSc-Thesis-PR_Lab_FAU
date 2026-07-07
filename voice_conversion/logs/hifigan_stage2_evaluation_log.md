# HiFi-GAN Stage 2: Feature & Classification Evaluation Report

## Comparative Evaluation of Reconstructed vs. Original Pilot Audio

This report compares the classification UAR and Accuracy scores for **original** vs. **HiFi-GAN reconstructed** audios across XLSR, Wav2Vec2, and WavLM models.

> [!WARNING]
> Due to the small size of this pilot (12 files total), cross-validation settings were restricted (3-fold outer CV, 2-fold inner CV). Performance scores represent diagnostic stability check values and representation drift rather than generalizable accuracy.

| Model | Layer | Scenario | Classifier | UAR Original | UAR Reconstructed | UAR Delta | Acc Original | Acc Reconstructed | Acc Delta |
|---|---|---|---|---|---|---|---|---|---|
| XLSR | 0 | Spanish->Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 0 | Spanish->Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 0 | Spanish->German | linear_svm | 0.3333 | 0.1667 | -0.1667 | 0.3333 | 0.1667 | -0.1667 |
| XLSR | 0 | Spanish->German | logistic_regression | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| XLSR | 0 | German->German | linear_svm | 0.1667 | 0.5000 | +0.3333 | 0.1667 | 0.5000 | +0.3333 |
| XLSR | 0 | German->German | logistic_regression | 0.1667 | 0.5000 | +0.3333 | 0.1667 | 0.5000 | +0.3333 |
| XLSR | 0 | German->Spanish | linear_svm | 0.3333 | 0.1667 | -0.1667 | 0.3333 | 0.1667 | -0.1667 |
| XLSR | 0 | German->Spanish | logistic_regression | 0.3333 | 0.1667 | -0.1667 | 0.3333 | 0.1667 | -0.1667 |
| XLSR | 0 | Spanish+German->Spanish+German | linear_svm | 0.5000 | 0.4167 | -0.0833 | 0.5000 | 0.4167 | -0.0833 |
| XLSR | 0 | Spanish+German->Spanish+German | logistic_regression | 0.5000 | 0.4167 | -0.0833 | 0.5000 | 0.4167 | -0.0833 |
| XLSR | 11 | Spanish->Spanish | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| XLSR | 11 | Spanish->Spanish | logistic_regression | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| XLSR | 11 | Spanish->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 11 | Spanish->German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 11 | German->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 11 | German->German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 11 | German->Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 11 | German->Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 11 | Spanish+German->Spanish+German | linear_svm | 0.5833 | 0.5000 | -0.0833 | 0.5833 | 0.5000 | -0.0833 |
| XLSR | 11 | Spanish+German->Spanish+German | logistic_regression | 0.6667 | 0.5833 | -0.0833 | 0.6667 | 0.5833 | -0.0833 |
| XLSR | 4 | Spanish->Spanish | linear_svm | 0.5000 | 0.6667 | +0.1667 | 0.5000 | 0.6667 | +0.1667 |
| XLSR | 4 | Spanish->Spanish | logistic_regression | 0.5000 | 0.6667 | +0.1667 | 0.5000 | 0.6667 | +0.1667 |
| XLSR | 4 | Spanish->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 4 | Spanish->German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 4 | German->German | linear_svm | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| XLSR | 4 | German->German | logistic_regression | 0.8333 | 0.5000 | -0.3333 | 0.8333 | 0.5000 | -0.3333 |
| XLSR | 4 | German->Spanish | linear_svm | 0.1667 | 0.1667 | +0.0000 | 0.1667 | 0.1667 | +0.0000 |
| XLSR | 4 | German->Spanish | logistic_regression | 0.1667 | 0.1667 | +0.0000 | 0.1667 | 0.1667 | +0.0000 |
| XLSR | 4 | Spanish+German->Spanish+German | linear_svm | 0.5000 | 0.4167 | -0.0833 | 0.5000 | 0.4167 | -0.0833 |
| XLSR | 4 | Spanish+German->Spanish+German | logistic_regression | 0.7500 | 0.5833 | -0.1667 | 0.7500 | 0.5833 | -0.1667 |
| XLSR | 8 | Spanish->Spanish | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| XLSR | 8 | Spanish->Spanish | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| XLSR | 8 | Spanish->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | Spanish->German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | German->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | German->German | logistic_regression | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| XLSR | 8 | German->Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | German->Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | Spanish+German->Spanish+German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | Spanish+German->Spanish+German | logistic_regression | 0.5833 | 0.5833 | +0.0000 | 0.5833 | 0.5833 | +0.0000 |
| WAV2VEC2 | 0 | Spanish->Spanish | linear_svm | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| WAV2VEC2 | 0 | Spanish->Spanish | logistic_regression | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| WAV2VEC2 | 0 | Spanish->German | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAV2VEC2 | 0 | Spanish->German | logistic_regression | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 0 | German->German | linear_svm | 0.1667 | 0.1667 | +0.0000 | 0.1667 | 0.1667 | +0.0000 |
| WAV2VEC2 | 0 | German->German | logistic_regression | 0.1667 | 0.1667 | +0.0000 | 0.1667 | 0.1667 | +0.0000 |
| WAV2VEC2 | 0 | German->Spanish | linear_svm | 0.6667 | 0.3333 | -0.3333 | 0.6667 | 0.3333 | -0.3333 |
| WAV2VEC2 | 0 | German->Spanish | logistic_regression | 0.6667 | 0.3333 | -0.3333 | 0.6667 | 0.3333 | -0.3333 |
| WAV2VEC2 | 0 | Spanish+German->Spanish+German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 0 | Spanish+German->Spanish+German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 11 | Spanish->Spanish | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAV2VEC2 | 11 | Spanish->Spanish | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAV2VEC2 | 11 | Spanish->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 11 | Spanish->German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 11 | German->German | linear_svm | 0.5000 | 0.3333 | -0.1667 | 0.5000 | 0.3333 | -0.1667 |
| WAV2VEC2 | 11 | German->German | logistic_regression | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAV2VEC2 | 11 | German->Spanish | linear_svm | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 11 | German->Spanish | logistic_regression | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 11 | Spanish+German->Spanish+German | linear_svm | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 11 | Spanish+German->Spanish+German | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAV2VEC2 | 4 | Spanish->Spanish | linear_svm | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| WAV2VEC2 | 4 | Spanish->Spanish | logistic_regression | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| WAV2VEC2 | 4 | Spanish->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 4 | Spanish->German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 4 | German->German | linear_svm | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAV2VEC2 | 4 | German->German | logistic_regression | 0.6667 | 0.1667 | -0.5000 | 0.6667 | 0.1667 | -0.5000 |
| WAV2VEC2 | 4 | German->Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 4 | German->Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 4 | Spanish+German->Spanish+German | linear_svm | 0.6667 | 0.5833 | -0.0833 | 0.6667 | 0.5833 | -0.0833 |
| WAV2VEC2 | 4 | Spanish+German->Spanish+German | logistic_regression | 0.6667 | 0.5833 | -0.0833 | 0.6667 | 0.5833 | -0.0833 |
| WAV2VEC2 | 8 | Spanish->Spanish | linear_svm | 0.8333 | 0.6667 | -0.1667 | 0.8333 | 0.6667 | -0.1667 |
| WAV2VEC2 | 8 | Spanish->Spanish | logistic_regression | 0.8333 | 0.6667 | -0.1667 | 0.8333 | 0.6667 | -0.1667 |
| WAV2VEC2 | 8 | Spanish->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 8 | Spanish->German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 8 | German->German | linear_svm | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAV2VEC2 | 8 | German->German | logistic_regression | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAV2VEC2 | 8 | German->Spanish | linear_svm | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 8 | German->Spanish | logistic_regression | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 8 | Spanish+German->Spanish+German | linear_svm | 0.5833 | 0.5833 | +0.0000 | 0.5833 | 0.5833 | +0.0000 |
| WAV2VEC2 | 8 | Spanish+German->Spanish+German | logistic_regression | 0.5833 | 0.5833 | +0.0000 | 0.5833 | 0.5833 | +0.0000 |
| WAVLM | 0 | Spanish->Spanish | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 0 | Spanish->Spanish | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 0 | Spanish->German | linear_svm | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAVLM | 0 | Spanish->German | logistic_regression | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAVLM | 0 | German->German | linear_svm | 0.3333 | 0.0000 | -0.3333 | 0.3333 | 0.0000 | -0.3333 |
| WAVLM | 0 | German->German | logistic_regression | 0.1667 | 0.1667 | +0.0000 | 0.1667 | 0.1667 | +0.0000 |
| WAVLM | 0 | German->Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 0 | German->Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 0 | Spanish+German->Spanish+German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 0 | Spanish+German->Spanish+German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 11 | Spanish->Spanish | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 11 | Spanish->Spanish | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 11 | Spanish->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 11 | Spanish->German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 11 | German->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 11 | German->German | logistic_regression | 0.3333 | 0.5000 | +0.1667 | 0.3333 | 0.5000 | +0.1667 |
| WAVLM | 11 | German->Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 11 | German->Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 11 | Spanish+German->Spanish+German | linear_svm | 0.7500 | 0.5833 | -0.1667 | 0.7500 | 0.5833 | -0.1667 |
| WAVLM | 11 | Spanish+German->Spanish+German | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 4 | Spanish->Spanish | linear_svm | 0.8333 | 0.6667 | -0.1667 | 0.8333 | 0.6667 | -0.1667 |
| WAVLM | 4 | Spanish->Spanish | logistic_regression | 0.8333 | 0.6667 | -0.1667 | 0.8333 | 0.6667 | -0.1667 |
| WAVLM | 4 | Spanish->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 4 | Spanish->German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 4 | German->German | linear_svm | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAVLM | 4 | German->German | logistic_regression | 0.5000 | 0.3333 | -0.1667 | 0.5000 | 0.3333 | -0.1667 |
| WAVLM | 4 | German->Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 4 | German->Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 4 | Spanish+German->Spanish+German | linear_svm | 0.6667 | 0.5833 | -0.0833 | 0.6667 | 0.5833 | -0.0833 |
| WAVLM | 4 | Spanish+German->Spanish+German | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 8 | Spanish->Spanish | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 8 | Spanish->Spanish | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 8 | Spanish->German | linear_svm | 0.6667 | 0.8333 | +0.1667 | 0.6667 | 0.8333 | +0.1667 |
| WAVLM | 8 | Spanish->German | logistic_regression | 0.5000 | 0.8333 | +0.3333 | 0.5000 | 0.8333 | +0.3333 |
| WAVLM | 8 | German->German | linear_svm | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAVLM | 8 | German->German | logistic_regression | 0.1667 | 0.1667 | +0.0000 | 0.1667 | 0.1667 | +0.0000 |
| WAVLM | 8 | German->Spanish | linear_svm | 0.5000 | 0.1667 | -0.3333 | 0.5000 | 0.1667 | -0.3333 |
| WAVLM | 8 | German->Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 8 | Spanish+German->Spanish+German | linear_svm | 0.7500 | 0.7500 | +0.0000 | 0.7500 | 0.7500 | +0.0000 |
| WAVLM | 8 | Spanish+German->Spanish+German | logistic_regression | 0.7500 | 0.7500 | +0.0000 | 0.7500 | 0.7500 | +0.0000 |

## Technical Conclusion
- **Acoustic Diagnostic Preservation**: If the delta UAR is close to zero, it indicates the vocoder successfully preserves target speaker representations and baseline model diagnostic embeddings.
- **Representation Drift**: Large UAR discrepancies warrant checking the specific layer representations (e.g. earlier layers vs deeper classification layer features).