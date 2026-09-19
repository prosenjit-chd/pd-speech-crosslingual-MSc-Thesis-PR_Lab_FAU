# Cross-Language Parkinson's Disease Classification Report: 80-File Validation Cohort

## 1. Executive Summary
- **Baseline Feature Extractor**: `facebook/wav2vec2-large-xlsr-53` (Layer 11, frozen).
- **Classifiers**: Linear Support Vector Machine (`SVC`) & Logistic Regression (`liblinear`).
- **Validation Design**: Strict out-of-sample testing across 80 subjects (20 Spanish HC, 20 Spanish PD, 20 German HC, 20 German PD).

### Critical Comparison Table

| Direction | Classifier | Within-Language Baseline UAR | Original Foreign Cross-Language UAR | Generated True Cross-Language UAR | Absolute $\Delta$ UAR | Relative Gain | Bootstrap $p$-value | BH-FDR $q$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spanish -> German** | `linear_svm` | 0.6500 | 0.5000 | **0.6250** | **+0.1250** | **+25.0%** | 0.0620 | 0.1240 |
| **German -> Spanish** | `linear_svm` | 0.8500 | 0.5750 | **0.4250** | **-0.1500** | **-26.1%** | 1.0000 | 1.0000 |
| **Spanish -> German** | `logistic_regression` | 0.6750 | 0.5000 | **0.6500** | **+0.1500** | **+30.0%** | 0.0250 | 0.1000 |
| **German -> Spanish** | `logistic_regression` | 0.8500 | 0.5750 | **0.5250** | **-0.0500** | **-8.7%** | 0.8780 | 1.0000 |

---

## 2. Complete Scenario Results Table

| Classifier | Scenario | Training Domain | Test Domain | UAR [95% CI] | Sensitivity (PD Recall) | Specificity (HC Recall) | Confusion Matrix [TN, FP, FN, TP] | Best $C$ |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `linear_svm` | Within-Language (German->German) | German | Real German | **0.6500** [0.5126, 0.8001] | 0.5500 | 0.7500 | [15, 5, 9, 11] | 0.01 |
| `linear_svm` | Original Cross-Language (German->Spanish) | German | Real Spanish | **0.5000** [0.5000, 0.5000] | 1.0000 | 0.0000 | [0, 20, 0, 20] | 0.01 |
| `linear_svm` | Generated True Cross-Language (Spanish in German) | German | Generated German | **0.6250** [0.4749, 0.7674] | 0.6500 | 0.6000 | [12, 8, 7, 13] | 0.01 |
| `linear_svm` | Within-Language (Spanish->Spanish) | Spanish | Real Spanish | **0.8500** [0.7323, 0.9500] | 0.7500 | 0.9500 | [19, 1, 5, 15] | 0.01 |
| `linear_svm` | Original Cross-Language (Spanish->German) | Spanish | Real German | **0.5750** [0.4500, 0.7085] | 0.3000 | 0.8500 | [17, 3, 14, 6] | 0.01 |
| `linear_svm` | Generated True Cross-Language (German in Spanish) | Spanish | Generated Spanish | **0.4250** [0.2903, 0.5632] | 0.2000 | 0.6500 | [13, 7, 16, 4] | 0.01 |
| `logistic_regression` | Within-Language (German->German) | German | Real German | **0.6750** [0.5429, 0.8208] | 0.5500 | 0.8000 | [16, 4, 9, 11] | 1.0 |
| `logistic_regression` | Original Cross-Language (German->Spanish) | German | Real Spanish | **0.5000** [0.5000, 0.5000] | 1.0000 | 0.0000 | [0, 20, 0, 20] | 1.0 |
| `logistic_regression` | Generated True Cross-Language (Spanish in German) | German | Generated German | **0.6500** [0.5012, 0.7812] | 0.7000 | 0.6000 | [12, 8, 6, 14] | 1.0 |
| `logistic_regression` | Within-Language (Spanish->Spanish) | Spanish | Real Spanish | **0.8500** [0.7323, 0.9500] | 0.7500 | 0.9500 | [19, 1, 5, 15] | 0.1 |
| `logistic_regression` | Original Cross-Language (Spanish->German) | Spanish | Real German | **0.5750** [0.4500, 0.7085] | 0.3000 | 0.8500 | [17, 3, 14, 6] | 0.1 |
| `logistic_regression` | Generated True Cross-Language (German in Spanish) | Spanish | Generated Spanish | **0.5250** [0.4271, 0.6253] | 0.1500 | 0.9000 | [18, 2, 17, 3] | 0.1 |
