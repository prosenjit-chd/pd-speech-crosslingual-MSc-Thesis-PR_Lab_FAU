# Stage 5A — 12-File Embedding-Conditioned Crosslingual Conversion Pilot Report

## 1. Goal of Stage 5A
The objective of this stage is to test **Plan B: embedding-conditioned generation/conversion** on a small 12-file pilot (3 Spanish HC, 3 Spanish PD, 3 German HC, 3 German PD) before scaling. We aim to determine whether acoustic representations (WavLM, Wav2Vec2, XLSR) can be used as conditioning information to guide crosslingual conversion while preserving Parkinson’s Disease (PD) vs. Healthy Control (HC) diagnostic speech features.

## 2. Safety Rule Confirmation
We confirm that all previous baseline, HiFi-GAN 12, 80, and 276 folders, logs, and files remain completely **read-only** and untouched. All Stage 5A work and intermediate outputs were isolated inside:
`C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc`

## 3. Pilot Selection Summary
Exactly 12 balanced files were prepared and verified in the pilot set:

| Stage 5 File | Language | Diagnosis | Source Original | Copied Status |
| --- | --- | --- | --- | --- |
| `SP_PD_001.wav` | Spanish | PD | `AVPEPUDEA0001_readtext.wav` | copied |
| `SP_PD_002.wav` | Spanish | PD | `AVPEPUDEA0002_readtext.wav` | copied |
| `SP_PD_003.wav` | Spanish | PD | `AVPEPUDEA0003_readtext.wav` | copied |
| `SP_HC_001.wav` | Spanish | HC | `AVPEPUDEAC0001_readtext.wav` | copied |
| `SP_HC_002.wav` | Spanish | HC | `AVPEPUDEAC0003_readtext.wav` | copied |
| `SP_HC_003.wav` | Spanish | HC | `AVPEPUDEAC0004_readtext.wav` | copied |
| `DE_PD_001.wav` | German | PD | `002.u1.02.wav` | copied |
| `DE_PD_002.wav` | German | PD | `003.u1.02.wav` | copied |
| `DE_PD_003.wav` | German | PD | `007.u2.02.wav` | copied |
| `DE_HC_001.wav` | German | HC | `001.u1.02.wav` | copied |
| `DE_HC_002.wav` | German | HC | `003.u1.02.wav` | copied |
| `DE_HC_003.wav` | German | HC | `005.u1.02.wav` | copied |

## 4. Embedding Extraction Summary
Layer-wise speech embeddings (layers 0, 4, 8, 11) were extracted for XLSR, Wav2Vec2, and WavLM models. Extraction success rate: **144/144 (100.0%)**.

## 5. Domain Condition Creation Summary
Averaged target-domain conditions were calculated by averaging original file embeddings within each language for each model and target layer:

| Model | Layer | Domain | Dimension | Mean Embedding Value | Std Dev |
| --- | --- | --- | --- | --- | --- |
| XLSR | 0 | german-domain | 1024 | 1.1634 | 2.2768 |
| XLSR | 0 | spanish-domain | 1024 | 1.1317 | 2.1969 |
| XLSR | 4 | german-domain | 1024 | 1.0819 | 1.4987 |
| XLSR | 4 | spanish-domain | 1024 | 1.0513 | 1.5752 |
| XLSR | 8 | german-domain | 1024 | 1.0681 | 1.7875 |
| XLSR | 8 | spanish-domain | 1024 | 1.0399 | 1.9719 |
| XLSR | 11 | german-domain | 1024 | 1.0596 | 2.1263 |
| XLSR | 11 | spanish-domain | 1024 | 1.0281 | 2.3308 |
| WAV2VEC2 | 0 | german-domain | 768 | 0.0001 | 0.0854 |
| WAV2VEC2 | 0 | spanish-domain | 768 | -0.0000 | 0.0965 |
| WAV2VEC2 | 4 | german-domain | 768 | -0.0024 | 0.0909 |
| WAV2VEC2 | 4 | spanish-domain | 768 | -0.0003 | 0.1146 |
| WAV2VEC2 | 8 | german-domain | 768 | -0.0056 | 0.1129 |
| WAV2VEC2 | 8 | spanish-domain | 768 | -0.0046 | 0.1294 |
| WAV2VEC2 | 11 | german-domain | 768 | -0.0002 | 0.1003 |
| WAV2VEC2 | 11 | spanish-domain | 768 | 0.0001 | 0.0983 |
| WAVLM | 0 | german-domain | 768 | 0.0039 | 0.0629 |
| WAVLM | 0 | spanish-domain | 768 | 0.0046 | 0.0722 |
| WAVLM | 4 | german-domain | 768 | 0.0059 | 0.0649 |
| WAVLM | 4 | spanish-domain | 768 | 0.0036 | 0.0706 |
| WAVLM | 8 | german-domain | 768 | 0.0025 | 0.0489 |
| WAVLM | 8 | spanish-domain | 768 | 0.0016 | 0.0602 |
| WAVLM | 11 | german-domain | 768 | 0.0005 | 0.0721 |
| WAVLM | 11 | spanish-domain | 768 | -0.0003 | 0.0718 |

## 6. Conversion / Generation Method
- **Method Applied**: **prototype embedding-conditioned conversion**
- **Conditioning Feature**: XLSR layer 11 embeddings (1024 dimensions)
- **Mathematical Mapping**: Fit a Ridge regression model ($m \approx W \cdot E + b$) from the 1024-dimensional embedding space to the 80-dimensional time-averaged log-mel spectrogram space using the 12 pilot files. For a source file, the predicted acoustic domain shift was computed as $\Delta m = W \cdot (E_{tgt\_avg} - E_i)$ and applied to the source log-mel spectrogram with a conversion scale $\alpha = 0.5$. The converted log-mel spectrograms were verified to match the HiFi-GAN vocoder config and synthesized using the pre-trained `universal_v1` generator.

## 7. Audio Validation Results
Technical validation results for the generated WAV files:
- **Success (specifications met)**: 12
- **Warning (technical deviations)**: 0
- **Failed**: 0

| Source File | Converted Output File | SR (Hz) | Mono | Conv Dur (s) | Delta (s) | Peak Amp | RMS | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `SP_PD_001.wav` | `converted_spanish_to_german\SP_PD_001_to_DE_domain.wav` | 22050 | Yes | 15.128 | -0.001 | 0.974 | 0.125 | **SUCCESS** |
| `SP_PD_002.wav` | `converted_spanish_to_german\SP_PD_002_to_DE_domain.wav` | 22050 | Yes | 26.134 | -0.004 | 0.997 | 0.180 | **SUCCESS** |
| `SP_PD_003.wav` | `converted_spanish_to_german\SP_PD_003_to_DE_domain.wav` | 22050 | Yes | 25.124 | -0.010 | 0.992 | 0.119 | **SUCCESS** |
| `SP_HC_001.wav` | `converted_spanish_to_german\SP_HC_001_to_DE_domain.wav` | 22050 | Yes | 17.461 | -0.009 | 0.980 | 0.185 | **SUCCESS** |
| `SP_HC_002.wav` | `converted_spanish_to_german\SP_HC_002_to_DE_domain.wav` | 22050 | Yes | 16.985 | -0.007 | 0.983 | 0.121 | **SUCCESS** |
| `SP_HC_003.wav` | `converted_spanish_to_german\SP_HC_003_to_DE_domain.wav` | 22050 | Yes | 17.067 | -0.010 | 0.966 | 0.141 | **SUCCESS** |
| `DE_PD_001.wav` | `converted_german_to_spanish\DE_PD_001_to_SP_domain.wav` | 22050 | Yes | 47.450 | -0.010 | 0.513 | 0.057 | **SUCCESS** |
| `DE_PD_002.wav` | `converted_german_to_spanish\DE_PD_002_to_SP_domain.wav` | 22050 | Yes | 38.534 | -0.010 | 0.605 | 0.069 | **SUCCESS** |
| `DE_PD_003.wav` | `converted_german_to_spanish\DE_PD_003_to_SP_domain.wav` | 22050 | Yes | 40.472 | -0.004 | 0.701 | 0.074 | **SUCCESS** |
| `DE_HC_001.wav` | `converted_german_to_spanish\DE_HC_001_to_SP_domain.wav` | 22050 | Yes | 40.763 | -0.001 | 0.524 | 0.064 | **SUCCESS** |
| `DE_HC_002.wav` | `converted_german_to_spanish\DE_HC_002_to_SP_domain.wav` | 22050 | Yes | 46.498 | -0.004 | 0.613 | 0.063 | **SUCCESS** |
| `DE_HC_003.wav` | `converted_german_to_spanish\DE_HC_003_to_SP_domain.wav` | 22050 | Yes | 37.349 | -0.010 | 0.972 | 0.077 | **SUCCESS** |

## 8. Classification Comparison Results (Diagnostic Only)
> [!IMPORTANT]
> All classification metrics reported below represent diagnostic crosslingual evaluation check values on a very small pilot sample. They serve to observe representation drift and domain shifts rather than generalizable performance.

| Model | Layer | Scenario | Classifier | UAR Original | UAR Converted | UAR Delta | Acc Original | Acc Converted | Acc Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XLSR | 0 | Spanish$\to$Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 0 | Spanish$\to$Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 0 | Spanish$\to$German | linear_svm | 0.3333 | 0.5000 | +0.1667 | 0.3333 | 0.5000 | +0.1667 |
| XLSR | 0 | Spanish$\to$German | logistic_regression | 0.3333 | 0.5000 | +0.1667 | 0.3333 | 0.5000 | +0.1667 |
| XLSR | 0 | German$\to$German | linear_svm | 0.1667 | 0.1667 | +0.0000 | 0.1667 | 0.1667 | +0.0000 |
| XLSR | 0 | German$\to$German | logistic_regression | 0.1667 | 0.3333 | +0.1667 | 0.1667 | 0.3333 | +0.1667 |
| XLSR | 0 | German$\to$Spanish | linear_svm | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| XLSR | 0 | German$\to$Spanish | logistic_regression | 0.3333 | 0.1667 | -0.1667 | 0.3333 | 0.1667 | -0.1667 |
| XLSR | 0 | Spanish+German$\to$Spanish+German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 0 | Spanish+German$\to$Spanish+German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 4 | Spanish$\to$Spanish | linear_svm | 0.5000 | 0.6667 | +0.1667 | 0.5000 | 0.6667 | +0.1667 |
| XLSR | 4 | Spanish$\to$Spanish | logistic_regression | 0.5000 | 0.6667 | +0.1667 | 0.5000 | 0.6667 | +0.1667 |
| XLSR | 4 | Spanish$\to$German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 4 | Spanish$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 4 | German$\to$German | linear_svm | 0.6667 | 0.3333 | -0.3333 | 0.6667 | 0.3333 | -0.3333 |
| XLSR | 4 | German$\to$German | logistic_regression | 0.8333 | 0.5000 | -0.3333 | 0.8333 | 0.5000 | -0.3333 |
| XLSR | 4 | German$\to$Spanish | linear_svm | 0.1667 | 0.3333 | +0.1667 | 0.1667 | 0.3333 | +0.1667 |
| XLSR | 4 | German$\to$Spanish | logistic_regression | 0.1667 | 0.1667 | +0.0000 | 0.1667 | 0.1667 | +0.0000 |
| XLSR | 4 | Spanish+German$\to$Spanish+German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 4 | Spanish+German$\to$Spanish+German | logistic_regression | 0.7500 | 0.5833 | -0.1667 | 0.7500 | 0.5833 | -0.1667 |
| XLSR | 8 | Spanish$\to$Spanish | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| XLSR | 8 | Spanish$\to$Spanish | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| XLSR | 8 | Spanish$\to$German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | Spanish$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | German$\to$German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | German$\to$German | logistic_regression | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| XLSR | 8 | German$\to$Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | German$\to$Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | Spanish+German$\to$Spanish+German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 8 | Spanish+German$\to$Spanish+German | logistic_regression | 0.5833 | 0.5833 | +0.0000 | 0.5833 | 0.5833 | +0.0000 |
| XLSR | 11 | Spanish$\to$Spanish | linear_svm | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| XLSR | 11 | Spanish$\to$Spanish | logistic_regression | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| XLSR | 11 | Spanish$\to$German | linear_svm | 0.5000 | 0.3333 | -0.1667 | 0.5000 | 0.3333 | -0.1667 |
| XLSR | 11 | Spanish$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 11 | German$\to$German | linear_svm | 0.5000 | 0.3333 | -0.1667 | 0.5000 | 0.3333 | -0.1667 |
| XLSR | 11 | German$\to$German | logistic_regression | 0.3333 | 0.5000 | +0.1667 | 0.3333 | 0.5000 | +0.1667 |
| XLSR | 11 | German$\to$Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 11 | German$\to$Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| XLSR | 11 | Spanish+German$\to$Spanish+German | linear_svm | 0.5000 | 0.5833 | +0.0833 | 0.5000 | 0.5833 | +0.0833 |
| XLSR | 11 | Spanish+German$\to$Spanish+German | logistic_regression | 0.6667 | 0.5833 | -0.0833 | 0.6667 | 0.5833 | -0.0833 |
| WAV2VEC2 | 0 | Spanish$\to$Spanish | linear_svm | 0.8333 | 0.6667 | -0.1667 | 0.8333 | 0.6667 | -0.1667 |
| WAV2VEC2 | 0 | Spanish$\to$Spanish | logistic_regression | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| WAV2VEC2 | 0 | Spanish$\to$German | linear_svm | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 0 | Spanish$\to$German | logistic_regression | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 0 | German$\to$German | linear_svm | 0.1667 | 0.1667 | +0.0000 | 0.1667 | 0.1667 | +0.0000 |
| WAV2VEC2 | 0 | German$\to$German | logistic_regression | 0.1667 | 0.1667 | +0.0000 | 0.1667 | 0.1667 | +0.0000 |
| WAV2VEC2 | 0 | German$\to$Spanish | linear_svm | 0.6667 | 0.3333 | -0.3333 | 0.6667 | 0.3333 | -0.3333 |
| WAV2VEC2 | 0 | German$\to$Spanish | logistic_regression | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 0 | Spanish+German$\to$Spanish+German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 0 | Spanish+German$\to$Spanish+German | logistic_regression | 0.5000 | 0.5833 | +0.0833 | 0.5000 | 0.5833 | +0.0833 |
| WAV2VEC2 | 4 | Spanish$\to$Spanish | linear_svm | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| WAV2VEC2 | 4 | Spanish$\to$Spanish | logistic_regression | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| WAV2VEC2 | 4 | Spanish$\to$German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 4 | Spanish$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 4 | German$\to$German | linear_svm | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAV2VEC2 | 4 | German$\to$German | logistic_regression | 0.6667 | 0.1667 | -0.5000 | 0.6667 | 0.1667 | -0.5000 |
| WAV2VEC2 | 4 | German$\to$Spanish | linear_svm | 0.5000 | 0.3333 | -0.1667 | 0.5000 | 0.3333 | -0.1667 |
| WAV2VEC2 | 4 | German$\to$Spanish | logistic_regression | 0.5000 | 0.3333 | -0.1667 | 0.5000 | 0.3333 | -0.1667 |
| WAV2VEC2 | 4 | Spanish+German$\to$Spanish+German | linear_svm | 0.6667 | 0.5833 | -0.0833 | 0.6667 | 0.5833 | -0.0833 |
| WAV2VEC2 | 4 | Spanish+German$\to$Spanish+German | logistic_regression | 0.6667 | 0.5833 | -0.0833 | 0.6667 | 0.5833 | -0.0833 |
| WAV2VEC2 | 8 | Spanish$\to$Spanish | linear_svm | 0.8333 | 0.6667 | -0.1667 | 0.8333 | 0.6667 | -0.1667 |
| WAV2VEC2 | 8 | Spanish$\to$Spanish | logistic_regression | 0.8333 | 0.6667 | -0.1667 | 0.8333 | 0.6667 | -0.1667 |
| WAV2VEC2 | 8 | Spanish$\to$German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 8 | Spanish$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 8 | German$\to$German | linear_svm | 0.3333 | 0.1667 | -0.1667 | 0.3333 | 0.1667 | -0.1667 |
| WAV2VEC2 | 8 | German$\to$German | logistic_regression | 0.3333 | 0.1667 | -0.1667 | 0.3333 | 0.1667 | -0.1667 |
| WAV2VEC2 | 8 | German$\to$Spanish | linear_svm | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 8 | German$\to$Spanish | logistic_regression | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 8 | Spanish+German$\to$Spanish+German | linear_svm | 0.5833 | 0.5000 | -0.0833 | 0.5833 | 0.5000 | -0.0833 |
| WAV2VEC2 | 8 | Spanish+German$\to$Spanish+German | logistic_regression | 0.5833 | 0.5833 | +0.0000 | 0.5833 | 0.5833 | +0.0000 |
| WAV2VEC2 | 11 | Spanish$\to$Spanish | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAV2VEC2 | 11 | Spanish$\to$Spanish | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAV2VEC2 | 11 | Spanish$\to$German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 11 | Spanish$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 11 | German$\to$German | linear_svm | 0.5000 | 0.3333 | -0.1667 | 0.5000 | 0.3333 | -0.1667 |
| WAV2VEC2 | 11 | German$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAV2VEC2 | 11 | German$\to$Spanish | linear_svm | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 11 | German$\to$Spanish | logistic_regression | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 11 | Spanish+German$\to$Spanish+German | linear_svm | 0.6667 | 0.5000 | -0.1667 | 0.6667 | 0.5000 | -0.1667 |
| WAV2VEC2 | 11 | Spanish+German$\to$Spanish+German | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 0 | Spanish$\to$Spanish | linear_svm | 0.6667 | 0.8333 | +0.1667 | 0.6667 | 0.8333 | +0.1667 |
| WAVLM | 0 | Spanish$\to$Spanish | logistic_regression | 0.6667 | 0.8333 | +0.1667 | 0.6667 | 0.8333 | +0.1667 |
| WAVLM | 0 | Spanish$\to$German | linear_svm | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAVLM | 0 | Spanish$\to$German | logistic_regression | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAVLM | 0 | German$\to$German | linear_svm | 0.3333 | 0.1667 | -0.1667 | 0.3333 | 0.1667 | -0.1667 |
| WAVLM | 0 | German$\to$German | logistic_regression | 0.1667 | 0.0000 | -0.1667 | 0.1667 | 0.0000 | -0.1667 |
| WAVLM | 0 | German$\to$Spanish | linear_svm | 0.5000 | 0.3333 | -0.1667 | 0.5000 | 0.3333 | -0.1667 |
| WAVLM | 0 | German$\to$Spanish | logistic_regression | 0.5000 | 0.3333 | -0.1667 | 0.5000 | 0.3333 | -0.1667 |
| WAVLM | 0 | Spanish+German$\to$Spanish+German | linear_svm | 0.5000 | 0.4167 | -0.0833 | 0.5000 | 0.4167 | -0.0833 |
| WAVLM | 0 | Spanish+German$\to$Spanish+German | logistic_regression | 0.5000 | 0.4167 | -0.0833 | 0.5000 | 0.4167 | -0.0833 |
| WAVLM | 4 | Spanish$\to$Spanish | linear_svm | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| WAVLM | 4 | Spanish$\to$Spanish | logistic_regression | 0.8333 | 0.8333 | +0.0000 | 0.8333 | 0.8333 | +0.0000 |
| WAVLM | 4 | Spanish$\to$German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 4 | Spanish$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 4 | German$\to$German | linear_svm | 0.3333 | 0.5000 | +0.1667 | 0.3333 | 0.5000 | +0.1667 |
| WAVLM | 4 | German$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 4 | German$\to$Spanish | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 4 | German$\to$Spanish | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 4 | Spanish+German$\to$Spanish+German | linear_svm | 0.6667 | 0.5833 | -0.0833 | 0.6667 | 0.5833 | -0.0833 |
| WAVLM | 4 | Spanish+German$\to$Spanish+German | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 8 | Spanish$\to$Spanish | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 8 | Spanish$\to$Spanish | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 8 | Spanish$\to$German | linear_svm | 0.6667 | 0.8333 | +0.1667 | 0.6667 | 0.8333 | +0.1667 |
| WAVLM | 8 | Spanish$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 8 | German$\to$German | linear_svm | 0.3333 | 0.3333 | +0.0000 | 0.3333 | 0.3333 | +0.0000 |
| WAVLM | 8 | German$\to$German | logistic_regression | 0.1667 | 0.3333 | +0.1667 | 0.1667 | 0.3333 | +0.1667 |
| WAVLM | 8 | German$\to$Spanish | linear_svm | 0.5000 | 0.6667 | +0.1667 | 0.5000 | 0.6667 | +0.1667 |
| WAVLM | 8 | German$\to$Spanish | logistic_regression | 0.5000 | 0.6667 | +0.1667 | 0.5000 | 0.6667 | +0.1667 |
| WAVLM | 8 | Spanish+German$\to$Spanish+German | linear_svm | 0.7500 | 0.5833 | -0.1667 | 0.7500 | 0.5833 | -0.1667 |
| WAVLM | 8 | Spanish+German$\to$Spanish+German | logistic_regression | 0.7500 | 0.7500 | +0.0000 | 0.7500 | 0.7500 | +0.0000 |
| WAVLM | 11 | Spanish$\to$Spanish | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 11 | Spanish$\to$Spanish | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 11 | Spanish$\to$German | linear_svm | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 11 | Spanish$\to$German | logistic_regression | 0.5000 | 0.5000 | +0.0000 | 0.5000 | 0.5000 | +0.0000 |
| WAVLM | 11 | German$\to$German | linear_svm | 0.5000 | 0.3333 | -0.1667 | 0.5000 | 0.3333 | -0.1667 |
| WAVLM | 11 | German$\to$German | logistic_regression | 0.3333 | 0.5000 | +0.1667 | 0.3333 | 0.5000 | +0.1667 |
| WAVLM | 11 | German$\to$Spanish | linear_svm | 0.5000 | 0.6667 | +0.1667 | 0.5000 | 0.6667 | +0.1667 |
| WAVLM | 11 | German$\to$Spanish | logistic_regression | 0.5000 | 0.6667 | +0.1667 | 0.5000 | 0.6667 | +0.1667 |
| WAVLM | 11 | Spanish+German$\to$Spanish+German | linear_svm | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |
| WAVLM | 11 | Spanish+German$\to$Spanish+German | logistic_regression | 0.6667 | 0.6667 | +0.0000 | 0.6667 | 0.6667 | +0.0000 |

## 9. Scientific Limitations
> [!WARNING]
> **Critical Limitation Statement:**
> **“The 12-file Stage 5A experiment tests technical feasibility only. Because the embedding-to-mel mapping is trained on a very small pilot set, the results cannot be interpreted as final conversion performance.”**

- **No Language Translation**: This process is strictly acoustic. German speech was converted **toward the Spanish acoustic/domain condition**, and Spanish speech was converted **toward the German acoustic/domain condition** in log-mel feature space before vocoding.
- **Small Sample Bounds**: With only 3 speakers per group, SVM/Logistic Regression classification cannot yield final thesis performance conclusions.

## 10. Decision Recommendation
- **Option 1**: **Proceed to 80-file subset** using the prototype method if diagnostic classification preservation and audio validation meet baseline expectations (low representation drift).
- **Option 2**: **Adjust method before scaling** if significant acoustic degradation or extreme representation drift is observed.
- **Option 3**: **Fallback to existing VC model approach** if the linear projection fails to capture speaker identities or PD/HC features sufficiently.

---
*Report generated automatically by `07_write_stage5a_report.py`*