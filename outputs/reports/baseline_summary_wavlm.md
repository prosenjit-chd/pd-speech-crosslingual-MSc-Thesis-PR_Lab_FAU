# Multilingual Parkinson's Speech Detection Baseline - WAVLM

## 1. Dataset Summary
- **Task**: readtext
- **Total Recordings**: 276
- **Languages**: Spanish (100), German (176)
- **Labels**: PD (138), HC (138)

## 2. Experimental Setup
- **Model Used**: microsoft/wavlm-base
- **Layers Evaluated**: [0, 4, 8, 11]
- **Classifiers Used**: ['linear_svm', 'logistic_regression']

## 3. Best Performing Configurations (by UAR)

**Best Overall Layer**: Layer 8 (UAR: 0.8182 via German->German)

### Best Within-Language Result (Spanish → Spanish)
- **UAR**: 0.8000
- **Model/Layer**: wavlm / Layer 4
- **Classifier**: logistic_regression
- **Sensitivity**: 0.6800
- **Specificity**: 0.9200
- **AUC**: 0.8332

### Cross-lingual: Spanish to German
- **UAR**: 0.7273
- **Model/Layer**: wavlm / Layer 11
- **Classifier**: logistic_regression
- **Sensitivity**: 0.8750
- **Specificity**: 0.5795
- **AUC**: 0.7487

### Best Within-Language Result (German → German)
- **UAR**: 0.8182
- **Model/Layer**: wavlm / Layer 8
- **Classifier**: logistic_regression
- **Sensitivity**: 0.8068
- **Specificity**: 0.8295
- **AUC**: 0.8680

### Cross-lingual: German to Spanish
- **UAR**: 0.6900
- **Model/Layer**: wavlm / Layer 0
- **Classifier**: logistic_regression
- **Sensitivity**: 0.6200
- **Specificity**: 0.7600
- **AUC**: 0.7488

### Spanish+German → Spanish+German
- **UAR**: 0.7971
- **Model/Layer**: wavlm / Layer 8
- **Classifier**: logistic_regression
- **Sensitivity**: 0.7899
- **Specificity**: 0.8043
- **AUC**: 0.8414

## 4. Interpretation on Language Mismatch
Yes, there appears to be a notable domain mismatch. Training on Spanish and testing on German caused a performance drop of 0.0727 UAR compared to within-language testing.

## 5. Generated Deliverable Files
- `metadata/dataset_index_readtext.csv` (Full dataset index)
- `features/xlsr/*.csv` (Layer-wise embedded features)
- `outputs/figures/tsne_*.png` (t-SNE plots by PD/HC label and language)
- `outputs/figures/pca_*.png` (PCA plots by PD/HC label and language)
- `outputs/tables/classification_results_*.csv` (Raw result tables per layer)
- `outputs/tables/model_layer_comparison.csv` (Aggregated cross-lingual classification results)

## 6. Full Results Table

| model   |   layer | train_language   | test_language   | classifier          |    uar |   accuracy |   sensitivity |   specificity |    auc |   n_train |   n_test |
|:--------|--------:|:-----------------|:----------------|:--------------------|-------:|-----------:|--------------:|--------------:|-------:|----------:|---------:|
| wavlm   |       0 | Spanish          | Spanish         | linear_svm          | 0.77   |     0.77   |        0.74   |        0.8    | 0.7944 |       100 |      100 |
| wavlm   |       0 | Spanish          | Spanish         | logistic_regression | 0.73   |     0.73   |        0.68   |        0.78   | 0.7972 |       100 |      100 |
| wavlm   |       0 | Spanish          | German          | linear_svm          | 0.642  |     0.642  |        0.5114 |        0.7727 | 0.7161 |       100 |      176 |
| wavlm   |       0 | Spanish          | German          | logistic_regression | 0.6477 |     0.6477 |        0.5455 |        0.75   | 0.7123 |       100 |      176 |
| wavlm   |       0 | German           | German          | linear_svm          | 0.7045 |     0.7045 |        0.7614 |        0.6477 | 0.7714 |       176 |      176 |
| wavlm   |       0 | German           | German          | logistic_regression | 0.7273 |     0.7273 |        0.7727 |        0.6818 | 0.7646 |       176 |      176 |
| wavlm   |       0 | German           | Spanish         | linear_svm          | 0.68   |     0.68   |        0.56   |        0.8    | 0.746  |       176 |      100 |
| wavlm   |       0 | German           | Spanish         | logistic_regression | 0.69   |     0.69   |        0.62   |        0.76   | 0.7488 |       176 |      100 |
| wavlm   |       0 | Spanish+German   | Spanish+German  | linear_svm          | 0.75   |     0.75   |        0.7681 |        0.7319 | 0.8004 |       276 |      276 |
| wavlm   |       0 | Spanish+German   | Spanish+German  | logistic_regression | 0.7536 |     0.7536 |        0.7754 |        0.7319 | 0.8056 |       276 |      276 |
| wavlm   |      11 | Spanish          | Spanish         | linear_svm          | 0.74   |     0.74   |        0.7    |        0.78   | 0.8016 |       100 |      100 |
| wavlm   |      11 | Spanish          | Spanish         | logistic_regression | 0.78   |     0.78   |        0.7    |        0.86   | 0.8368 |       100 |      100 |
| wavlm   |      11 | Spanish          | German          | linear_svm          | 0.6591 |     0.6591 |        0.9205 |        0.3977 | 0.7243 |       100 |      176 |
| wavlm   |      11 | Spanish          | German          | logistic_regression | 0.7273 |     0.7273 |        0.875  |        0.5795 | 0.7487 |       100 |      176 |
| wavlm   |      11 | German           | German          | linear_svm          | 0.733  |     0.733  |        0.7273 |        0.7386 | 0.8063 |       176 |      176 |
| wavlm   |      11 | German           | German          | logistic_regression | 0.7955 |     0.7955 |        0.8523 |        0.7386 | 0.8323 |       176 |      176 |
| wavlm   |      11 | German           | Spanish         | linear_svm          | 0.57   |     0.57   |        0.16   |        0.98   | 0.6458 |       176 |      100 |
| wavlm   |      11 | German           | Spanish         | logistic_regression | 0.67   |     0.67   |        0.46   |        0.88   | 0.7448 |       176 |      100 |
| wavlm   |      11 | Spanish+German   | Spanish+German  | linear_svm          | 0.7572 |     0.7572 |        0.7609 |        0.7536 | 0.827  |       276 |      276 |
| wavlm   |      11 | Spanish+German   | Spanish+German  | logistic_regression | 0.7754 |     0.7754 |        0.7971 |        0.7536 | 0.8366 |       276 |      276 |
| wavlm   |       4 | Spanish          | Spanish         | linear_svm          | 0.78   |     0.78   |        0.72   |        0.84   | 0.8296 |       100 |      100 |
| wavlm   |       4 | Spanish          | Spanish         | logistic_regression | 0.8    |     0.8    |        0.68   |        0.92   | 0.8332 |       100 |      100 |
| wavlm   |       4 | Spanish          | German          | linear_svm          | 0.517  |     0.517  |        0.9773 |        0.0568 | 0.6574 |       100 |      176 |
| wavlm   |       4 | Spanish          | German          | logistic_regression | 0.5625 |     0.5625 |        0.9886 |        0.1364 | 0.6569 |       100 |      176 |
| wavlm   |       4 | German           | German          | linear_svm          | 0.7557 |     0.7557 |        0.75   |        0.7614 | 0.7944 |       176 |      176 |
| wavlm   |       4 | German           | German          | logistic_regression | 0.75   |     0.75   |        0.75   |        0.75   | 0.802  |       176 |      176 |
| wavlm   |       4 | German           | Spanish         | linear_svm          | 0.68   |     0.68   |        0.66   |        0.7    | 0.674  |       176 |      100 |
| wavlm   |       4 | German           | Spanish         | logistic_regression | 0.66   |     0.66   |        0.8    |        0.52   | 0.704  |       176 |      100 |
| wavlm   |       4 | Spanish+German   | Spanish+German  | linear_svm          | 0.7826 |     0.7826 |        0.7754 |        0.7899 | 0.8281 |       276 |      276 |
| wavlm   |       4 | Spanish+German   | Spanish+German  | logistic_regression | 0.7862 |     0.7862 |        0.7754 |        0.7971 | 0.8364 |       276 |      276 |
| wavlm   |       8 | Spanish          | Spanish         | linear_svm          | 0.75   |     0.75   |        0.7    |        0.8    | 0.8176 |       100 |      100 |
| wavlm   |       8 | Spanish          | Spanish         | logistic_regression | 0.78   |     0.78   |        0.7    |        0.86   | 0.8468 |       100 |      100 |
| wavlm   |       8 | Spanish          | German          | linear_svm          | 0.625  |     0.625  |        0.9091 |        0.3409 | 0.683  |       100 |      176 |
| wavlm   |       8 | Spanish          | German          | logistic_regression | 0.6534 |     0.6534 |        0.7841 |        0.5227 | 0.6897 |       100 |      176 |
| wavlm   |       8 | German           | German          | linear_svm          | 0.7386 |     0.7386 |        0.75   |        0.7273 | 0.8134 |       176 |      176 |
| wavlm   |       8 | German           | German          | logistic_regression | 0.8182 |     0.8182 |        0.8068 |        0.8295 | 0.868  |       176 |      176 |
| wavlm   |       8 | German           | Spanish         | linear_svm          | 0.55   |     0.55   |        0.72   |        0.38   | 0.582  |       176 |      100 |
| wavlm   |       8 | German           | Spanish         | logistic_regression | 0.63   |     0.63   |        0.8    |        0.46   | 0.7124 |       176 |      100 |
| wavlm   |       8 | Spanish+German   | Spanish+German  | linear_svm          | 0.7754 |     0.7754 |        0.7681 |        0.7826 | 0.848  |       276 |      276 |
| wavlm   |       8 | Spanish+German   | Spanish+German  | logistic_regression | 0.7971 |     0.7971 |        0.7899 |        0.8043 | 0.8414 |       276 |      276 |