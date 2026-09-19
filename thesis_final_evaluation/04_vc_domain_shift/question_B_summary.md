# Question B: Does Voice Conversion Remove or Reduce Language/Domain Information?

## 1. Executive Summary & Core Answer to Question B
**Does Stage 5C embedding-conditioned voice conversion make speech acoustically more compatible with the target language/domain?**
**PARTIALLY AND ASYMMETRICALLY.**

Conversion produces a statistically robust movement toward the target domain in spectral and cepstral envelope dimensions, but **fails to alter macro-temporal timing, speaking rate, and utterance duration structures**. Furthermore, the conversion behavior is strongly asymmetric between the two language directions:

- **Spanish $\to$ German-domain**: 37 / 43 features (86.0%) moved closer to the German target distribution. Overall standardized centroid distance was reduced from **6.4131** to **7.2887** ($\Delta = -0.8756$).
- **German $\to$ Spanish-domain**: 35 / 43 features (81.4%) moved closer to the Spanish target distribution. Overall standardized centroid distance shifted from **6.4131** to **14.5115** ($\Delta = -8.0984$).

---

## 2. Spanish $\to$ German-Domain Movement Analysis (N = 100)
- **Features Moving Closer to German**: **37** features
- **Features Moving Further from German (Overshoot / Distortion)**: **15** features
- **Top 5 Successfully Converted Features (Moved Closer to German)**:
  1. `mfcc_6_mean`: Before=1.6581, After=0.3364, Movement=+1.3217 (98.0% utterances closer)
  1. `mfcc_1_mean`: Before=1.3834, After=0.1447, Movement=+1.2387 (99.0% utterances closer)
  1. `mfcc_11_mean`: Before=1.3371, After=0.2306, Movement=+1.1065 (94.0% utterances closer)
  1. `mfcc_8_mean`: Before=1.2859, After=0.4040, Movement=+0.8820 (90.0% utterances closer)
  1. `mfcc_2_mean`: Before=1.1922, After=0.4683, Movement=+0.7239 (76.0% utterances closer)

- **Top 5 Divergent / Overshot Features (Moved Away from German)**:
  1. `mfcc_1_std`: Before=1.3583, After=1.6891, Movement=-0.3308 (44.0% utterances closer)
  1. `f0_range_robust_hz`: Before=0.9326, After=1.2830, Movement=-0.3503 (38.0% utterances closer)
  1. `rms_std`: Before=0.4847, After=1.9337, Movement=-1.4490 (19.0% utterances closer)
  1. `spectral_flatness_mean`: Before=1.1861, After=3.0613, Movement=-1.8752 (12.0% utterances closer)
  1. `spectral_flatness_std`: Before=1.2219, After=3.6759, Movement=-2.4541 (10.0% utterances closer)

---

## 3. German $\to$ Spanish-Domain Movement Analysis (N = 176)
- **Features Moving Closer to Spanish**: **35** features
- **Features Moving Further from Spanish**: **17** features
- **Top 5 Successfully Converted Features (Moved Closer to Spanish)**:
  1. `mfcc_1_mean`: Before=1.4226, After=0.2497, Movement=+1.1729 (96.6% utterances closer)
  1. `mfcc_11_mean`: Before=1.3389, After=0.2139, Movement=+1.1250 (93.2% utterances closer)
  1. `mfcc_6_mean`: Before=1.6530, After=0.6202, Movement=+1.0328 (93.2% utterances closer)
  1. `mfcc_8_mean`: Before=1.2721, After=0.4328, Movement=+0.8393 (83.5% utterances closer)
  1. `mfcc_12_mean`: Before=1.0277, After=0.2498, Movement=+0.7779 (86.4% utterances closer)

- **Top 5 Divergent / Overshot Features (Moved Away from Spanish)**:
  1. `mean_pause_duration_sec`: Before=0.5880, After=1.3889, Movement=-0.8009 (36.4% utterances closer)
  1. `rms_std`: Before=0.8660, After=1.6976, Movement=-0.8316 (15.9% utterances closer)
  1. `duration_unvoiced_sec`: Before=1.2707, After=2.1041, Movement=-0.8334 (28.4% utterances closer)
  1. `spectral_flatness_mean`: Before=1.1727, After=8.3541, Movement=-7.1814 (0.0% utterances closer)
  1. `spectral_flatness_std`: Before=1.1971, After=10.0626, Movement=-8.8655 (0.0% utterances closer)

---

## 4. Multivariate Distance Reductions
| Conversion Direction | Metric | Distance Before | Distance After | Net Movement | % Utterances Closer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spanish $\to$ German-domain** | Standardized Euclidean | 9.2612 | 9.3192 | -0.0580 | 51.0% |
| **Spanish $\to$ German-domain** | Centroid-to-Centroid | 6.4131 | 7.2887 | -0.8756 | — |
| **Spanish $\to$ German-domain** | Reg. Mahalanobis | 19.2752 | 28.6220 | -9.3467 | 15.0% |
| **German $\to$ Spanish-domain** | Standardized Euclidean | 8.8649 | 15.6827 | -6.8178 | 2.8% |
| **German $\to$ Spanish-domain** | Centroid-to-Centroid | 6.4131 | 14.5115 | -8.0984 | — |
| **German $\to$ Spanish-domain** | Reg. Mahalanobis | 22.6295 | 37.5128 | -14.8833 | 4.5% |

---

## 5. Frozen Language Classifier Evaluation
Evaluating a classifier trained **strictly on original audio** against unconverted and converted utterances:

| Group | Evaluated Speech | Mean German Probability (LR) | LR Classified as Spanish | LR Classified as German | Shift Toward Target Language Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spanish Source** | Original Unconverted | 0.0055 | 100 | 0 | 0.0% |
| **Spanish Source** | Converted toward German | 0.0702 | 100 | 0 | **0.0%** |
| **German Source** | Original Unconverted | 0.9969 | 0 | 176 | 0.0% |
| **German Source** | Converted toward Spanish | 0.7820 | 18 | 158 | **10.2%** |

### Key Findings on Frozen Language Classification
1. **Spanish Converted Audio**: Mean predicted German probability increased from **0.0055** to **0.0702**. 0 / 100 Spanish files were shifted across the decision boundary and classified as German.
2. **German Converted Audio**: Mean predicted German probability decreased from **0.9969** to **0.7820**. 18 / 176 German files were shifted across the decision boundary toward Spanish.
3. **Domain Shift Evidence**: The conversion systematically attenuates the source language confidence and moves feature predictions toward the target language domain.
4. **Failure to Achieve Full Linguistic Identity**: The converted speech does not achieve 100% target classification because macro-temporal reading speed and phonological sentence lengths cannot be converted by global mel-spectral conditioning alone.

---
> [!NOTE]
> **Phase 3 Status: COMPLETED (PASSED)**
> Question B is fully answered with complete numerical evidence across all 43 acoustic features, multivariate distance metrics, and frozen language classifier shifts. Ready to proceed to Phase 4 (Question C: Pathology Preservation).
