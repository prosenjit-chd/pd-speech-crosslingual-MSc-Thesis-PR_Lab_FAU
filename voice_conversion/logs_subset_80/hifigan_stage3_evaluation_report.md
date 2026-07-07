# HiFi-GAN Stage 3: Feature & Classification Evaluation Report

## Comparative Evaluation of Reconstructed vs. Original Subset 80 Audio

This report compares the classification UAR and Accuracy scores for **original preprocessed** vs. **HiFi-GAN reconstructed** audios across XLSR, Wav2Vec2, and WavLM models on the controlled 80-file subset.

> [!NOTE]
> Cross-validation was run using 5-fold outer and 4-fold inner splits. Performance scores represent representation preservation checks and potential feature drift after voice conversion reconstruction.

| Model | Layer | Scenario | Classifier | UAR Original | UAR Reconstructed | UAR Delta | Acc Original | Acc Reconstructed | Acc Delta |
|---|---|---|---|---|---|---|---|---|---|
| XLSR | 0 | Spanish->Spanish | linear_svm | 0.9250 | 0.8750 | -0.0500 | 0.9250 | 0.8750 | -0.0500 |
| XLSR | 0 | Spanish->Spanish | logistic_regression | 0.8250 | 0.8500 | +0.0250 | 0.8250 | 0.8500 | +0.0250 |
| XLSR | 0 | Spanish->German | linear_svm | 0.5000 | 0.4750 | -0.0250 | 0.5000 | 0.4750 | -0.0250 |
| XLSR | 0 | Spanish->German | logistic_regression | 0.5000 | 0.4750 | -0.0250 | 0.5000 | 0.4750 | -0.0250 |
| XLSR | 0 | German->German | linear_svm | 0.6000 | 0.5750 | -0.0250 | 0.6000 | 0.5750 | -0.0250 |
| XLSR | 0 | German->German | logistic_regression | 0.6250 | 0.5750 | -0.0500 | 0.6250 | 0.5750 | -0.0500 |
| XLSR | 0 | German->Spanish | linear_svm | 0.5250 | 0.6000 | +0.0750 | 0.5250 | 0.6000 | +0.0750 |
| XLSR | 0 | German->Spanish | logistic_regression | 0.6500 | 0.6000 | -0.0500 | 0.6500 | 0.6000 | -0.0500 |
| XLSR | 0 | Spanish+German->Spanish+German | linear_svm | 0.7000 | 0.6500 | -0.0500 | 0.7000 | 0.6500 | -0.0500 |
| XLSR | 0 | Spanish+German->Spanish+German | logistic_regression | 0.7000 | 0.6375 | -0.0625 | 0.7000 | 0.6375 | -0.0625 |
| XLSR | 11 | Spanish->Spanish | linear_svm | 0.7750 | 0.8500 | +0.0750 | 0.7750 | 0.8500 | +0.0750 |
| XLSR | 11 | Spanish->Spanish | logistic_regression | 0.8000 | 0.8500 | +0.0500 | 0.8000 | 0.8500 | +0.0500 |
| XLSR | 11 | Spanish->German | linear_svm | 0.5500 | 0.5750 | +0.0250 | 0.5500 | 0.5750 | +0.0250 |
| XLSR | 11 | Spanish->German | logistic_regression | 0.5750 | 0.5750 | +0.0000 | 0.5750 | 0.5750 | +0.0000 |
| XLSR | 11 | German->German | linear_svm | 0.6000 | 0.6500 | +0.0500 | 0.6000 | 0.6500 | +0.0500 |
| XLSR | 11 | German->German | logistic_regression | 0.6250 | 0.6500 | +0.0250 | 0.6250 | 0.6500 | +0.0250 |
| XLSR | 11 | German->Spanish | linear_svm | 0.4500 | 0.5250 | +0.0750 | 0.4500 | 0.5250 | +0.0750 |
| XLSR | 11 | German->Spanish | logistic_regression | 0.5250 | 0.5750 | +0.0500 | 0.5250 | 0.5750 | +0.0500 |
| XLSR | 11 | Spanish+German->Spanish+German | linear_svm | 0.6750 | 0.7250 | +0.0500 | 0.6750 | 0.7250 | +0.0500 |
| XLSR | 11 | Spanish+German->Spanish+German | logistic_regression | 0.6875 | 0.7625 | +0.0750 | 0.6875 | 0.7625 | +0.0750 |
| XLSR | 4 | Spanish->Spanish | linear_svm | 0.8750 | 0.8500 | -0.0250 | 0.8750 | 0.8500 | -0.0250 |
| XLSR | 4 | Spanish->Spanish | logistic_regression | 0.8500 | 0.8000 | -0.0500 | 0.8500 | 0.8000 | -0.0500 |
| XLSR | 4 | Spanish->German | linear_svm | 0.4500 | 0.5500 | +0.1000 | 0.4500 | 0.5500 | +0.1000 |
| XLSR | 4 | Spanish->German | logistic_regression | 0.4250 | 0.5500 | +0.1250 | 0.4250 | 0.5500 | +0.1250 |
| XLSR | 4 | German->German | linear_svm | 0.6750 | 0.6500 | -0.0250 | 0.6750 | 0.6500 | -0.0250 |
| XLSR | 4 | German->German | logistic_regression | 0.6500 | 0.6500 | +0.0000 | 0.6500 | 0.6500 | +0.0000 |
| XLSR | 4 | German->Spanish | linear_svm | 0.5250 | 0.5750 | +0.0500 | 0.5250 | 0.5750 | +0.0500 |
| XLSR | 4 | German->Spanish | logistic_regression | 0.5250 | 0.5500 | +0.0250 | 0.5250 | 0.5500 | +0.0250 |
| XLSR | 4 | Spanish+German->Spanish+German | linear_svm | 0.7125 | 0.6500 | -0.0625 | 0.7125 | 0.6500 | -0.0625 |
| XLSR | 4 | Spanish+German->Spanish+German | logistic_regression | 0.7250 | 0.7000 | -0.0250 | 0.7250 | 0.7000 | -0.0250 |
| XLSR | 8 | Spanish->Spanish | linear_svm | 0.7750 | 0.7750 | +0.0000 | 0.7750 | 0.7750 | +0.0000 |
| XLSR | 8 | Spanish->Spanish | logistic_regression | 0.8000 | 0.8250 | +0.0250 | 0.8000 | 0.8250 | +0.0250 |
| XLSR | 8 | Spanish->German | linear_svm | 0.4500 | 0.4500 | +0.0000 | 0.4500 | 0.4500 | +0.0000 |
| XLSR | 8 | Spanish->German | logistic_regression | 0.4750 | 0.4500 | -0.0250 | 0.4750 | 0.4500 | -0.0250 |
| XLSR | 8 | German->German | linear_svm | 0.6250 | 0.6250 | +0.0000 | 0.6250 | 0.6250 | +0.0000 |
| XLSR | 8 | German->German | logistic_regression | 0.6250 | 0.6500 | +0.0250 | 0.6250 | 0.6500 | +0.0250 |
| XLSR | 8 | German->Spanish | linear_svm | 0.4500 | 0.4500 | -0.0000 | 0.4500 | 0.4500 | +0.0000 |
| XLSR | 8 | German->Spanish | logistic_regression | 0.4750 | 0.4750 | +0.0000 | 0.4750 | 0.4750 | +0.0000 |
| XLSR | 8 | Spanish+German->Spanish+German | linear_svm | 0.5750 | 0.6500 | +0.0750 | 0.5750 | 0.6500 | +0.0750 |
| XLSR | 8 | Spanish+German->Spanish+German | logistic_regression | 0.6500 | 0.6625 | +0.0125 | 0.6500 | 0.6625 | +0.0125 |
| WAV2VEC2 | 0 | Spanish->Spanish | linear_svm | 0.8750 | 0.8500 | -0.0250 | 0.8750 | 0.8500 | -0.0250 |
| WAV2VEC2 | 0 | Spanish->Spanish | logistic_regression | 0.8750 | 0.7750 | -0.1000 | 0.8750 | 0.7750 | -0.1000 |
| WAV2VEC2 | 0 | Spanish->German | linear_svm | 0.4750 | 0.5000 | +0.0250 | 0.4750 | 0.5000 | +0.0250 |
| WAV2VEC2 | 0 | Spanish->German | logistic_regression | 0.4750 | 0.4750 | +0.0000 | 0.4750 | 0.4750 | +0.0000 |
| WAV2VEC2 | 0 | German->German | linear_svm | 0.6750 | 0.7000 | +0.0250 | 0.6750 | 0.7000 | +0.0250 |
| WAV2VEC2 | 0 | German->German | logistic_regression | 0.7000 | 0.6750 | -0.0250 | 0.7000 | 0.6750 | -0.0250 |
| WAV2VEC2 | 0 | German->Spanish | linear_svm | 0.4750 | 0.3250 | -0.1500 | 0.4750 | 0.3250 | -0.1500 |
| WAV2VEC2 | 0 | German->Spanish | logistic_regression | 0.4750 | 0.4250 | -0.0500 | 0.4750 | 0.4250 | -0.0500 |
| WAV2VEC2 | 0 | Spanish+German->Spanish+German | linear_svm | 0.6375 | 0.6500 | +0.0125 | 0.6375 | 0.6500 | +0.0125 |
| WAV2VEC2 | 0 | Spanish+German->Spanish+German | logistic_regression | 0.6375 | 0.6500 | +0.0125 | 0.6375 | 0.6500 | +0.0125 |
| WAV2VEC2 | 11 | Spanish->Spanish | linear_svm | 0.6750 | 0.8250 | +0.1500 | 0.6750 | 0.8250 | +0.1500 |
| WAV2VEC2 | 11 | Spanish->Spanish | logistic_regression | 0.7750 | 0.8250 | +0.0500 | 0.7750 | 0.8250 | +0.0500 |
| WAV2VEC2 | 11 | Spanish->German | linear_svm | 0.5500 | 0.5750 | +0.0250 | 0.5500 | 0.5750 | +0.0250 |
| WAV2VEC2 | 11 | Spanish->German | logistic_regression | 0.5750 | 0.5750 | +0.0000 | 0.5750 | 0.5750 | +0.0000 |
| WAV2VEC2 | 11 | German->German | linear_svm | 0.5750 | 0.6500 | +0.0750 | 0.5750 | 0.6500 | +0.0750 |
| WAV2VEC2 | 11 | German->German | logistic_regression | 0.6000 | 0.6500 | +0.0500 | 0.6000 | 0.6500 | +0.0500 |
| WAV2VEC2 | 11 | German->Spanish | linear_svm | 0.5000 | 0.6000 | +0.1000 | 0.5000 | 0.6000 | +0.1000 |
| WAV2VEC2 | 11 | German->Spanish | logistic_regression | 0.6500 | 0.6250 | -0.0250 | 0.6500 | 0.6250 | -0.0250 |
| WAV2VEC2 | 11 | Spanish+German->Spanish+German | linear_svm | 0.7500 | 0.7875 | +0.0375 | 0.7500 | 0.7875 | +0.0375 |
| WAV2VEC2 | 11 | Spanish+German->Spanish+German | logistic_regression | 0.7500 | 0.7875 | +0.0375 | 0.7500 | 0.7875 | +0.0375 |
| WAV2VEC2 | 4 | Spanish->Spanish | linear_svm | 0.8250 | 0.8500 | +0.0250 | 0.8250 | 0.8500 | +0.0250 |
| WAV2VEC2 | 4 | Spanish->Spanish | logistic_regression | 0.8250 | 0.8250 | +0.0000 | 0.8250 | 0.8250 | +0.0000 |
| WAV2VEC2 | 4 | Spanish->German | linear_svm | 0.4500 | 0.5000 | +0.0500 | 0.4500 | 0.5000 | +0.0500 |
| WAV2VEC2 | 4 | Spanish->German | logistic_regression | 0.5000 | 0.4500 | -0.0500 | 0.5000 | 0.4500 | -0.0500 |
| WAV2VEC2 | 4 | German->German | linear_svm | 0.6750 | 0.6750 | +0.0000 | 0.6750 | 0.6750 | +0.0000 |
| WAV2VEC2 | 4 | German->German | logistic_regression | 0.6500 | 0.6750 | +0.0250 | 0.6500 | 0.6750 | +0.0250 |
| WAV2VEC2 | 4 | German->Spanish | linear_svm | 0.5750 | 0.5500 | -0.0250 | 0.5750 | 0.5500 | -0.0250 |
| WAV2VEC2 | 4 | German->Spanish | logistic_regression | 0.5250 | 0.5500 | +0.0250 | 0.5250 | 0.5500 | +0.0250 |
| WAV2VEC2 | 4 | Spanish+German->Spanish+German | linear_svm | 0.7000 | 0.7250 | +0.0250 | 0.7000 | 0.7250 | +0.0250 |
| WAV2VEC2 | 4 | Spanish+German->Spanish+German | logistic_regression | 0.7000 | 0.7125 | +0.0125 | 0.7000 | 0.7125 | +0.0125 |
| WAV2VEC2 | 8 | Spanish->Spanish | linear_svm | 0.8250 | 0.8750 | +0.0500 | 0.8250 | 0.8750 | +0.0500 |
| WAV2VEC2 | 8 | Spanish->Spanish | logistic_regression | 0.7750 | 0.8750 | +0.1000 | 0.7750 | 0.8750 | +0.1000 |
| WAV2VEC2 | 8 | Spanish->German | linear_svm | 0.6250 | 0.5500 | -0.0750 | 0.6250 | 0.5500 | -0.0750 |
| WAV2VEC2 | 8 | Spanish->German | logistic_regression | 0.6000 | 0.5750 | -0.0250 | 0.6000 | 0.5750 | -0.0250 |
| WAV2VEC2 | 8 | German->German | linear_svm | 0.6000 | 0.5750 | -0.0250 | 0.6000 | 0.5750 | -0.0250 |
| WAV2VEC2 | 8 | German->German | logistic_regression | 0.6500 | 0.6000 | -0.0500 | 0.6500 | 0.6000 | -0.0500 |
| WAV2VEC2 | 8 | German->Spanish | linear_svm | 0.6500 | 0.6500 | +0.0000 | 0.6500 | 0.6500 | +0.0000 |
| WAV2VEC2 | 8 | German->Spanish | logistic_regression | 0.7250 | 0.6500 | -0.0750 | 0.7250 | 0.6500 | -0.0750 |
| WAV2VEC2 | 8 | Spanish+German->Spanish+German | linear_svm | 0.7625 | 0.7250 | -0.0375 | 0.7625 | 0.7250 | -0.0375 |
| WAV2VEC2 | 8 | Spanish+German->Spanish+German | logistic_regression | 0.7250 | 0.7375 | +0.0125 | 0.7250 | 0.7375 | +0.0125 |
| WAVLM | 0 | Spanish->Spanish | linear_svm | 0.8500 | 0.8750 | +0.0250 | 0.8500 | 0.8750 | +0.0250 |
| WAVLM | 0 | Spanish->Spanish | logistic_regression | 0.8500 | 0.8500 | +0.0000 | 0.8500 | 0.8500 | +0.0000 |
| WAVLM | 0 | Spanish->German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 0 | Spanish->German | logistic_regression | 0.5000 | 0.5250 | +0.0250 | 0.5000 | 0.5250 | +0.0250 |
| WAVLM | 0 | German->German | linear_svm | 0.6750 | 0.6750 | +0.0000 | 0.6750 | 0.6750 | +0.0000 |
| WAVLM | 0 | German->German | logistic_regression | 0.6000 | 0.6500 | +0.0500 | 0.6000 | 0.6500 | +0.0500 |
| WAVLM | 0 | German->Spanish | linear_svm | 0.5500 | 0.5000 | -0.0500 | 0.5500 | 0.5000 | -0.0500 |
| WAVLM | 0 | German->Spanish | logistic_regression | 0.5750 | 0.4500 | -0.1250 | 0.5750 | 0.4500 | -0.1250 |
| WAVLM | 0 | Spanish+German->Spanish+German | linear_svm | 0.6125 | 0.6250 | +0.0125 | 0.6125 | 0.6250 | +0.0125 |
| WAVLM | 0 | Spanish+German->Spanish+German | logistic_regression | 0.6875 | 0.6500 | -0.0375 | 0.6875 | 0.6500 | -0.0375 |
| WAVLM | 11 | Spanish->Spanish | linear_svm | 0.8750 | 0.8750 | +0.0000 | 0.8750 | 0.8750 | +0.0000 |
| WAVLM | 11 | Spanish->Spanish | logistic_regression | 0.8500 | 0.8500 | -0.0000 | 0.8500 | 0.8500 | +0.0000 |
| WAVLM | 11 | Spanish->German | linear_svm | 0.5250 | 0.5750 | +0.0500 | 0.5250 | 0.5750 | +0.0500 |
| WAVLM | 11 | Spanish->German | logistic_regression | 0.5000 | 0.5500 | +0.0500 | 0.5000 | 0.5500 | +0.0500 |
| WAVLM | 11 | German->German | linear_svm | 0.6750 | 0.6750 | +0.0000 | 0.6750 | 0.6750 | +0.0000 |
| WAVLM | 11 | German->German | logistic_regression | 0.6250 | 0.7000 | +0.0750 | 0.6250 | 0.7000 | +0.0750 |
| WAVLM | 11 | German->Spanish | linear_svm | 0.5500 | 0.5500 | -0.0000 | 0.5500 | 0.5500 | +0.0000 |
| WAVLM | 11 | German->Spanish | logistic_regression | 0.5750 | 0.5500 | -0.0250 | 0.5750 | 0.5500 | -0.0250 |
| WAVLM | 11 | Spanish+German->Spanish+German | linear_svm | 0.7125 | 0.7500 | +0.0375 | 0.7125 | 0.7500 | +0.0375 |
| WAVLM | 11 | Spanish+German->Spanish+German | logistic_regression | 0.7375 | 0.7500 | +0.0125 | 0.7375 | 0.7500 | +0.0125 |
| WAVLM | 4 | Spanish->Spanish | linear_svm | 0.8750 | 0.8750 | +0.0000 | 0.8750 | 0.8750 | +0.0000 |
| WAVLM | 4 | Spanish->Spanish | logistic_regression | 0.9250 | 0.9250 | +0.0000 | 0.9250 | 0.9250 | +0.0000 |
| WAVLM | 4 | Spanish->German | linear_svm | 0.5000 | 0.5500 | +0.0500 | 0.5000 | 0.5500 | +0.0500 |
| WAVLM | 4 | Spanish->German | logistic_regression | 0.4750 | 0.5250 | +0.0500 | 0.4750 | 0.5250 | +0.0500 |
| WAVLM | 4 | German->German | linear_svm | 0.6500 | 0.6500 | +0.0000 | 0.6500 | 0.6500 | +0.0000 |
| WAVLM | 4 | German->German | logistic_regression | 0.6750 | 0.6000 | -0.0750 | 0.6750 | 0.6000 | -0.0750 |
| WAVLM | 4 | German->Spanish | linear_svm | 0.5000 | 0.6250 | +0.1250 | 0.5000 | 0.6250 | +0.1250 |
| WAVLM | 4 | German->Spanish | logistic_regression | 0.4500 | 0.6000 | +0.1500 | 0.4500 | 0.6000 | +0.1500 |
| WAVLM | 4 | Spanish+German->Spanish+German | linear_svm | 0.6750 | 0.6875 | +0.0125 | 0.6750 | 0.6875 | +0.0125 |
| WAVLM | 4 | Spanish+German->Spanish+German | logistic_regression | 0.6625 | 0.6875 | +0.0250 | 0.6625 | 0.6875 | +0.0250 |
| WAVLM | 8 | Spanish->Spanish | linear_svm | 0.7750 | 0.8250 | +0.0500 | 0.7750 | 0.8250 | +0.0500 |
| WAVLM | 8 | Spanish->Spanish | logistic_regression | 0.8500 | 0.8000 | -0.0500 | 0.8500 | 0.8000 | -0.0500 |
| WAVLM | 8 | Spanish->German | linear_svm | 0.5250 | 0.5000 | -0.0250 | 0.5250 | 0.5000 | -0.0250 |
| WAVLM | 8 | Spanish->German | logistic_regression | 0.5500 | 0.5000 | -0.0500 | 0.5500 | 0.5000 | -0.0500 |
| WAVLM | 8 | German->German | linear_svm | 0.6750 | 0.6750 | +0.0000 | 0.6750 | 0.6750 | +0.0000 |
| WAVLM | 8 | German->German | logistic_regression | 0.6500 | 0.7000 | +0.0500 | 0.6500 | 0.7000 | +0.0500 |
| WAVLM | 8 | German->Spanish | linear_svm | 0.7250 | 0.6000 | -0.1250 | 0.7250 | 0.6000 | -0.1250 |
| WAVLM | 8 | German->Spanish | logistic_regression | 0.7250 | 0.5750 | -0.1500 | 0.7250 | 0.5750 | -0.1500 |
| WAVLM | 8 | Spanish+German->Spanish+German | linear_svm | 0.7625 | 0.8000 | +0.0375 | 0.7625 | 0.8000 | +0.0375 |
| WAVLM | 8 | Spanish+German->Spanish+German | logistic_regression | 0.7625 | 0.7875 | +0.0250 | 0.7625 | 0.7875 | +0.0250 |

## Technical Conclusion
- **Representation Drift Analysis**: If the delta UAR is near zero, it signifies that HiFi-GAN synthetic vocoding successfully preserves target speaker representations and diagnostic features.
- **Acoustic and Feature Drift**: Significant degradation or change in UAR suggests representation drift across layers (e.g. earlier layers preserving features differently than deep classifier layer embeddings).