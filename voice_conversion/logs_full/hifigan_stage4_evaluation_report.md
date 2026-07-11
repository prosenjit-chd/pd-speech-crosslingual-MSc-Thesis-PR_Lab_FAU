# HiFi-GAN Stage 4: Feature & Classification Evaluation Report

## Comparative Evaluation of Reconstructed vs. Original Full Dataset

This report compares the classification UAR and Accuracy scores for **original preprocessed** vs. **HiFi-GAN reconstructed** audios across XLSR, Wav2Vec2, and WavLM models on the full dataset.

> [!NOTE]
> Cross-validation was run using standard 10-fold outer and 9-fold inner splits as configured in baseline. Comparative results check diagnostic embedding robustness after full reconstruction.

| Model | Layer | Scenario | Classifier | UAR Original | UAR Reconstructed | UAR Delta | Acc Original | Acc Reconstructed | Acc Delta |
|---|---|---|---|---|---|---|---|---|---|
| XLSR | 0 | Spanish->Spanish | linear_svm | 0.7800 | 0.7900 | +0.0100 | 0.7800 | 0.7900 | +0.0100 |
| XLSR | 0 | Spanish->Spanish | logistic_regression | 0.7800 | 0.7800 | +0.0000 | 0.7800 | 0.7800 | +0.0000 |
| XLSR | 0 | Spanish->German | linear_svm | 0.5511 | 0.5682 | +0.0170 | 0.5511 | 0.5682 | +0.0170 |
| XLSR | 0 | Spanish->German | logistic_regression | 0.5966 | 0.5909 | -0.0057 | 0.5966 | 0.5909 | -0.0057 |
| XLSR | 0 | German->German | linear_svm | 0.7102 | 0.7216 | +0.0114 | 0.7102 | 0.7216 | +0.0114 |
| XLSR | 0 | German->German | logistic_regression | 0.7386 | 0.7557 | +0.0170 | 0.7386 | 0.7557 | +0.0170 |
| XLSR | 0 | German->Spanish | linear_svm | 0.6100 | 0.6000 | -0.0100 | 0.6100 | 0.6000 | -0.0100 |
| XLSR | 0 | German->Spanish | logistic_regression | 0.6100 | 0.6500 | +0.0400 | 0.6100 | 0.6500 | +0.0400 |
| XLSR | 0 | Spanish+German->Spanish+German | linear_svm | 0.7681 | 0.7717 | +0.0036 | 0.7681 | 0.7717 | +0.0036 |
| XLSR | 0 | Spanish+German->Spanish+German | logistic_regression | 0.7645 | 0.7862 | +0.0217 | 0.7645 | 0.7862 | +0.0217 |
| XLSR | 11 | Spanish->Spanish | linear_svm | 0.8000 | 0.7800 | -0.0200 | 0.8000 | 0.7800 | -0.0200 |
| XLSR | 11 | Spanish->Spanish | logistic_regression | 0.8000 | 0.8200 | +0.0200 | 0.8000 | 0.8200 | +0.0200 |
| XLSR | 11 | Spanish->German | linear_svm | 0.5909 | 0.5398 | -0.0511 | 0.5909 | 0.5398 | -0.0511 |
| XLSR | 11 | Spanish->German | logistic_regression | 0.6420 | 0.6591 | +0.0170 | 0.6420 | 0.6591 | +0.0170 |
| XLSR | 11 | German->German | linear_svm | 0.7045 | 0.7557 | +0.0511 | 0.7045 | 0.7557 | +0.0511 |
| XLSR | 11 | German->German | logistic_regression | 0.7557 | 0.7670 | +0.0114 | 0.7557 | 0.7670 | +0.0114 |
| XLSR | 11 | German->Spanish | linear_svm | 0.6100 | 0.5700 | -0.0400 | 0.6100 | 0.5700 | -0.0400 |
| XLSR | 11 | German->Spanish | logistic_regression | 0.6600 | 0.5500 | -0.1100 | 0.6600 | 0.5500 | -0.1100 |
| XLSR | 11 | Spanish+German->Spanish+German | linear_svm | 0.8007 | 0.7717 | -0.0290 | 0.8007 | 0.7717 | -0.0290 |
| XLSR | 11 | Spanish+German->Spanish+German | logistic_regression | 0.7826 | 0.8007 | +0.0181 | 0.7826 | 0.8007 | +0.0181 |
| XLSR | 4 | Spanish->Spanish | linear_svm | 0.8400 | 0.8200 | -0.0200 | 0.8400 | 0.8200 | -0.0200 |
| XLSR | 4 | Spanish->Spanish | logistic_regression | 0.8200 | 0.8100 | -0.0100 | 0.8200 | 0.8100 | -0.0100 |
| XLSR | 4 | Spanish->German | linear_svm | 0.5284 | 0.5227 | -0.0057 | 0.5284 | 0.5227 | -0.0057 |
| XLSR | 4 | Spanish->German | logistic_regression | 0.5455 | 0.5341 | -0.0114 | 0.5455 | 0.5341 | -0.0114 |
| XLSR | 4 | German->German | linear_svm | 0.7670 | 0.7159 | -0.0511 | 0.7670 | 0.7159 | -0.0511 |
| XLSR | 4 | German->German | logistic_regression | 0.7727 | 0.7841 | +0.0114 | 0.7727 | 0.7841 | +0.0114 |
| XLSR | 4 | German->Spanish | linear_svm | 0.6200 | 0.6600 | +0.0400 | 0.6200 | 0.6600 | +0.0400 |
| XLSR | 4 | German->Spanish | logistic_regression | 0.6200 | 0.7300 | +0.1100 | 0.6200 | 0.7300 | +0.1100 |
| XLSR | 4 | Spanish+German->Spanish+German | linear_svm | 0.7754 | 0.7826 | +0.0072 | 0.7754 | 0.7826 | +0.0072 |
| XLSR | 4 | Spanish+German->Spanish+German | logistic_regression | 0.7862 | 0.7935 | +0.0072 | 0.7862 | 0.7935 | +0.0072 |
| XLSR | 8 | Spanish->Spanish | linear_svm | 0.7900 | 0.8200 | +0.0300 | 0.7900 | 0.8200 | +0.0300 |
| XLSR | 8 | Spanish->Spanish | logistic_regression | 0.8400 | 0.8600 | +0.0200 | 0.8400 | 0.8600 | +0.0200 |
| XLSR | 8 | Spanish->German | linear_svm | 0.5966 | 0.6818 | +0.0852 | 0.5966 | 0.6818 | +0.0852 |
| XLSR | 8 | Spanish->German | logistic_regression | 0.6477 | 0.6591 | +0.0114 | 0.6477 | 0.6591 | +0.0114 |
| XLSR | 8 | German->German | linear_svm | 0.7102 | 0.7386 | +0.0284 | 0.7102 | 0.7386 | +0.0284 |
| XLSR | 8 | German->German | logistic_regression | 0.7500 | 0.7670 | +0.0170 | 0.7500 | 0.7670 | +0.0170 |
| XLSR | 8 | German->Spanish | linear_svm | 0.5500 | 0.6800 | +0.1300 | 0.5500 | 0.6800 | +0.1300 |
| XLSR | 8 | German->Spanish | logistic_regression | 0.6600 | 0.6600 | +0.0000 | 0.6600 | 0.6600 | +0.0000 |
| XLSR | 8 | Spanish+German->Spanish+German | linear_svm | 0.7536 | 0.7935 | +0.0399 | 0.7536 | 0.7935 | +0.0399 |
| XLSR | 8 | Spanish+German->Spanish+German | logistic_regression | 0.7935 | 0.8043 | +0.0109 | 0.7935 | 0.8043 | +0.0109 |
| WAV2VEC2 | 0 | Spanish->Spanish | linear_svm | 0.7800 | 0.7200 | -0.0600 | 0.7800 | 0.7200 | -0.0600 |
| WAV2VEC2 | 0 | Spanish->Spanish | logistic_regression | 0.7500 | 0.7400 | -0.0100 | 0.7500 | 0.7400 | -0.0100 |
| WAV2VEC2 | 0 | Spanish->German | linear_svm | 0.6420 | 0.5909 | -0.0511 | 0.6420 | 0.5909 | -0.0511 |
| WAV2VEC2 | 0 | Spanish->German | logistic_regression | 0.6136 | 0.6136 | +0.0000 | 0.6136 | 0.6136 | +0.0000 |
| WAV2VEC2 | 0 | German->German | linear_svm | 0.7159 | 0.7159 | -0.0000 | 0.7159 | 0.7159 | +0.0000 |
| WAV2VEC2 | 0 | German->German | logistic_regression | 0.7386 | 0.7045 | -0.0341 | 0.7386 | 0.7045 | -0.0341 |
| WAV2VEC2 | 0 | German->Spanish | linear_svm | 0.6200 | 0.6300 | +0.0100 | 0.6200 | 0.6300 | +0.0100 |
| WAV2VEC2 | 0 | German->Spanish | logistic_regression | 0.5500 | 0.6300 | +0.0800 | 0.5500 | 0.6300 | +0.0800 |
| WAV2VEC2 | 0 | Spanish+German->Spanish+German | linear_svm | 0.7717 | 0.7283 | -0.0435 | 0.7717 | 0.7283 | -0.0435 |
| WAV2VEC2 | 0 | Spanish+German->Spanish+German | logistic_regression | 0.7609 | 0.7210 | -0.0399 | 0.7609 | 0.7210 | -0.0399 |
| WAV2VEC2 | 11 | Spanish->Spanish | linear_svm | 0.7200 | 0.7300 | +0.0100 | 0.7200 | 0.7300 | +0.0100 |
| WAV2VEC2 | 11 | Spanish->Spanish | logistic_regression | 0.7400 | 0.7300 | -0.0100 | 0.7400 | 0.7300 | -0.0100 |
| WAV2VEC2 | 11 | Spanish->German | linear_svm | 0.6136 | 0.5227 | -0.0909 | 0.6136 | 0.5227 | -0.0909 |
| WAV2VEC2 | 11 | Spanish->German | logistic_regression | 0.5682 | 0.5625 | -0.0057 | 0.5682 | 0.5625 | -0.0057 |
| WAV2VEC2 | 11 | German->German | linear_svm | 0.7500 | 0.7557 | +0.0057 | 0.7500 | 0.7557 | +0.0057 |
| WAV2VEC2 | 11 | German->German | logistic_regression | 0.7443 | 0.7784 | +0.0341 | 0.7443 | 0.7784 | +0.0341 |
| WAV2VEC2 | 11 | German->Spanish | linear_svm | 0.5200 | 0.6500 | +0.1300 | 0.5200 | 0.6500 | +0.1300 |
| WAV2VEC2 | 11 | German->Spanish | logistic_regression | 0.6200 | 0.7100 | +0.0900 | 0.6200 | 0.7100 | +0.0900 |
| WAV2VEC2 | 11 | Spanish+German->Spanish+German | linear_svm | 0.7355 | 0.7283 | -0.0072 | 0.7355 | 0.7283 | -0.0072 |
| WAV2VEC2 | 11 | Spanish+German->Spanish+German | logistic_regression | 0.7645 | 0.7536 | -0.0109 | 0.7645 | 0.7536 | -0.0109 |
| WAV2VEC2 | 4 | Spanish->Spanish | linear_svm | 0.8100 | 0.8300 | +0.0200 | 0.8100 | 0.8300 | +0.0200 |
| WAV2VEC2 | 4 | Spanish->Spanish | logistic_regression | 0.8000 | 0.7700 | -0.0300 | 0.8000 | 0.7700 | -0.0300 |
| WAV2VEC2 | 4 | Spanish->German | linear_svm | 0.5852 | 0.6364 | +0.0511 | 0.5852 | 0.6364 | +0.0511 |
| WAV2VEC2 | 4 | Spanish->German | logistic_regression | 0.6591 | 0.6648 | +0.0057 | 0.6591 | 0.6648 | +0.0057 |
| WAV2VEC2 | 4 | German->German | linear_svm | 0.7614 | 0.7500 | -0.0114 | 0.7614 | 0.7500 | -0.0114 |
| WAV2VEC2 | 4 | German->German | logistic_regression | 0.7784 | 0.7614 | -0.0170 | 0.7784 | 0.7614 | -0.0170 |
| WAV2VEC2 | 4 | German->Spanish | linear_svm | 0.5300 | 0.6000 | +0.0700 | 0.5300 | 0.6000 | +0.0700 |
| WAV2VEC2 | 4 | German->Spanish | logistic_regression | 0.5600 | 0.6400 | +0.0800 | 0.5600 | 0.6400 | +0.0800 |
| WAV2VEC2 | 4 | Spanish+German->Spanish+German | linear_svm | 0.7681 | 0.7754 | +0.0072 | 0.7681 | 0.7754 | +0.0072 |
| WAV2VEC2 | 4 | Spanish+German->Spanish+German | logistic_regression | 0.7862 | 0.7862 | -0.0000 | 0.7862 | 0.7862 | +0.0000 |
| WAV2VEC2 | 8 | Spanish->Spanish | linear_svm | 0.7500 | 0.7800 | +0.0300 | 0.7500 | 0.7800 | +0.0300 |
| WAV2VEC2 | 8 | Spanish->Spanish | logistic_regression | 0.7800 | 0.7900 | +0.0100 | 0.7800 | 0.7900 | +0.0100 |
| WAV2VEC2 | 8 | Spanish->German | linear_svm | 0.5398 | 0.5341 | -0.0057 | 0.5398 | 0.5341 | -0.0057 |
| WAV2VEC2 | 8 | Spanish->German | logistic_regression | 0.5795 | 0.5682 | -0.0114 | 0.5795 | 0.5682 | -0.0114 |
| WAV2VEC2 | 8 | German->German | linear_svm | 0.7159 | 0.7386 | +0.0227 | 0.7159 | 0.7386 | +0.0227 |
| WAV2VEC2 | 8 | German->German | logistic_regression | 0.7727 | 0.7784 | +0.0057 | 0.7727 | 0.7784 | +0.0057 |
| WAV2VEC2 | 8 | German->Spanish | linear_svm | 0.5400 | 0.5300 | -0.0100 | 0.5400 | 0.5300 | -0.0100 |
| WAV2VEC2 | 8 | German->Spanish | logistic_regression | 0.6200 | 0.6700 | +0.0500 | 0.6200 | 0.6700 | +0.0500 |
| WAV2VEC2 | 8 | Spanish+German->Spanish+German | linear_svm | 0.7428 | 0.7500 | +0.0072 | 0.7428 | 0.7500 | +0.0072 |
| WAV2VEC2 | 8 | Spanish+German->Spanish+German | logistic_regression | 0.7935 | 0.7790 | -0.0145 | 0.7935 | 0.7790 | -0.0145 |
| WAVLM | 0 | Spanish->Spanish | linear_svm | 0.7400 | 0.7800 | +0.0400 | 0.7400 | 0.7800 | +0.0400 |
| WAVLM | 0 | Spanish->Spanish | logistic_regression | 0.7100 | 0.7600 | +0.0500 | 0.7100 | 0.7600 | +0.0500 |
| WAVLM | 0 | Spanish->German | linear_svm | 0.6364 | 0.6420 | +0.0057 | 0.6364 | 0.6420 | +0.0057 |
| WAVLM | 0 | Spanish->German | logistic_regression | 0.6136 | 0.6136 | +0.0000 | 0.6136 | 0.6136 | +0.0000 |
| WAVLM | 0 | German->German | linear_svm | 0.6591 | 0.6761 | +0.0170 | 0.6591 | 0.6761 | +0.0170 |
| WAVLM | 0 | German->German | logistic_regression | 0.7045 | 0.6932 | -0.0114 | 0.7045 | 0.6932 | -0.0114 |
| WAVLM | 0 | German->Spanish | linear_svm | 0.6900 | 0.6700 | -0.0200 | 0.6900 | 0.6700 | -0.0200 |
| WAVLM | 0 | German->Spanish | logistic_regression | 0.7000 | 0.6600 | -0.0400 | 0.7000 | 0.6600 | -0.0400 |
| WAVLM | 0 | Spanish+German->Spanish+German | linear_svm | 0.7391 | 0.7319 | -0.0072 | 0.7391 | 0.7319 | -0.0072 |
| WAVLM | 0 | Spanish+German->Spanish+German | logistic_regression | 0.7500 | 0.7428 | -0.0072 | 0.7500 | 0.7428 | -0.0072 |
| WAVLM | 11 | Spanish->Spanish | linear_svm | 0.7400 | 0.7000 | -0.0400 | 0.7400 | 0.7000 | -0.0400 |
| WAVLM | 11 | Spanish->Spanish | logistic_regression | 0.7800 | 0.7300 | -0.0500 | 0.7800 | 0.7300 | -0.0500 |
| WAVLM | 11 | Spanish->German | linear_svm | 0.6818 | 0.7045 | +0.0227 | 0.6818 | 0.7045 | +0.0227 |
| WAVLM | 11 | Spanish->German | logistic_regression | 0.7330 | 0.7045 | -0.0284 | 0.7330 | 0.7045 | -0.0284 |
| WAVLM | 11 | German->German | linear_svm | 0.7443 | 0.7557 | +0.0114 | 0.7443 | 0.7557 | +0.0114 |
| WAVLM | 11 | German->German | logistic_regression | 0.8011 | 0.8068 | +0.0057 | 0.8011 | 0.8068 | +0.0057 |
| WAVLM | 11 | German->Spanish | linear_svm | 0.5600 | 0.5700 | +0.0100 | 0.5600 | 0.5700 | +0.0100 |
| WAVLM | 11 | German->Spanish | logistic_regression | 0.6800 | 0.6900 | +0.0100 | 0.6800 | 0.6900 | +0.0100 |
| WAVLM | 11 | Spanish+German->Spanish+German | linear_svm | 0.7790 | 0.7717 | -0.0072 | 0.7790 | 0.7717 | -0.0072 |
| WAVLM | 11 | Spanish+German->Spanish+German | logistic_regression | 0.7971 | 0.7935 | -0.0036 | 0.7971 | 0.7935 | -0.0036 |
| WAVLM | 4 | Spanish->Spanish | linear_svm | 0.7800 | 0.7800 | +0.0000 | 0.7800 | 0.7800 | +0.0000 |
| WAVLM | 4 | Spanish->Spanish | logistic_regression | 0.8000 | 0.8200 | +0.0200 | 0.8000 | 0.8200 | +0.0200 |
| WAVLM | 4 | Spanish->German | linear_svm | 0.5227 | 0.5511 | +0.0284 | 0.5227 | 0.5511 | +0.0284 |
| WAVLM | 4 | Spanish->German | logistic_regression | 0.5625 | 0.5568 | -0.0057 | 0.5625 | 0.5568 | -0.0057 |
| WAVLM | 4 | German->German | linear_svm | 0.7386 | 0.7443 | +0.0057 | 0.7386 | 0.7443 | +0.0057 |
| WAVLM | 4 | German->German | logistic_regression | 0.7670 | 0.7727 | +0.0057 | 0.7670 | 0.7727 | +0.0057 |
| WAVLM | 4 | German->Spanish | linear_svm | 0.6600 | 0.6100 | -0.0500 | 0.6600 | 0.6100 | -0.0500 |
| WAVLM | 4 | German->Spanish | logistic_regression | 0.6600 | 0.6100 | -0.0500 | 0.6600 | 0.6100 | -0.0500 |
| WAVLM | 4 | Spanish+German->Spanish+German | linear_svm | 0.7428 | 0.7572 | +0.0145 | 0.7428 | 0.7572 | +0.0145 |
| WAVLM | 4 | Spanish+German->Spanish+German | logistic_regression | 0.7681 | 0.7717 | +0.0036 | 0.7681 | 0.7717 | +0.0036 |
| WAVLM | 8 | Spanish->Spanish | linear_svm | 0.7500 | 0.7600 | +0.0100 | 0.7500 | 0.7600 | +0.0100 |
| WAVLM | 8 | Spanish->Spanish | logistic_regression | 0.7800 | 0.7700 | -0.0100 | 0.7800 | 0.7700 | -0.0100 |
| WAVLM | 8 | Spanish->German | linear_svm | 0.6193 | 0.5398 | -0.0795 | 0.6193 | 0.5398 | -0.0795 |
| WAVLM | 8 | Spanish->German | logistic_regression | 0.6534 | 0.6193 | -0.0341 | 0.6534 | 0.6193 | -0.0341 |
| WAVLM | 8 | German->German | linear_svm | 0.7557 | 0.7557 | +0.0000 | 0.7557 | 0.7557 | +0.0000 |
| WAVLM | 8 | German->German | logistic_regression | 0.8068 | 0.7557 | -0.0511 | 0.8068 | 0.7557 | -0.0511 |
| WAVLM | 8 | German->Spanish | linear_svm | 0.5500 | 0.5000 | -0.0500 | 0.5500 | 0.5000 | -0.0500 |
| WAVLM | 8 | German->Spanish | logistic_regression | 0.6500 | 0.6500 | +0.0000 | 0.6500 | 0.6500 | +0.0000 |
| WAVLM | 8 | Spanish+German->Spanish+German | linear_svm | 0.7681 | 0.7536 | -0.0145 | 0.7681 | 0.7536 | -0.0145 |
| WAVLM | 8 | Spanish+German->Spanish+German | logistic_regression | 0.8043 | 0.7899 | -0.0145 | 0.8043 | 0.7899 | -0.0145 |

## Technical Conclusion
- **Vocoding Integrity**: Near-zero delta values across all layers suggest vocoding has successfully preserved target properties needed by pre-trained feature extractors.
- **Diagnostic Drift**: High deltas highlight potential drift or voice conversion distortion. Analyzing layers individually helps isolate which transformers are sensitive to reconstruction artifacts.