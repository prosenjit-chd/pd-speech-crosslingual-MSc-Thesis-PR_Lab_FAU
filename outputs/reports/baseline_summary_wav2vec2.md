# Multilingual Parkinson's Speech Detection Baseline - WAV2VEC2

## 1. Dataset Summary
- **Task**: readtext
- **Total Recordings**: 276
- **Languages**: Spanish (100), German (176)
- **Labels**: PD (138), HC (138)

## 2. Experimental Setup
- **Model Used**: facebook/wav2vec2-base
- **Layers Evaluated**: [0, 4, 8, 11]
- **Classifiers Used**: ['linear_svm', 'logistic_regression']

## 3. Best Performing Configurations (by UAR)

**Best Overall Layer**: Layer 4 (UAR: 0.8100 via Spanish->Spanish)

### Best Within-Language Result (Spanish → Spanish)
- **UAR**: 0.8100
- **Model/Layer**: wav2vec2 / Layer 4
- **Classifier**: linear_svm
- **Sensitivity**: 0.7800
- **Specificity**: 0.8400
- **AUC**: 0.8360

### Cross-lingual: Spanish to German
- **UAR**: 0.6648
- **Model/Layer**: wav2vec2 / Layer 4
- **Classifier**: logistic_regression
- **Sensitivity**: 0.8636
- **Specificity**: 0.4659
- **AUC**: 0.6934

### Best Within-Language Result (German → German)
- **UAR**: 0.7784
- **Model/Layer**: wav2vec2 / Layer 4
- **Classifier**: logistic_regression
- **Sensitivity**: 0.7727
- **Specificity**: 0.7841
- **AUC**: 0.8439

### Cross-lingual: German to Spanish
- **UAR**: 0.6300
- **Model/Layer**: wav2vec2 / Layer 8
- **Classifier**: logistic_regression
- **Sensitivity**: 0.4200
- **Specificity**: 0.8400
- **AUC**: 0.7808

### Spanish+German → Spanish+German
- **UAR**: 0.8080
- **Model/Layer**: wav2vec2 / Layer 8
- **Classifier**: logistic_regression
- **Sensitivity**: 0.8116
- **Specificity**: 0.8043
- **AUC**: 0.8427

## 4. Interpretation on Language Mismatch
Yes, there appears to be a notable domain mismatch. Training on Spanish and testing on German caused a performance drop of 0.1452 UAR compared to within-language testing.

## 5. Generated Deliverable Files
- `metadata/dataset_index_readtext.csv` (Full dataset index)
- `features/xlsr/*.csv` (Layer-wise embedded features)
- `outputs/figures/tsne_*.png` (t-SNE plots by PD/HC label and language)
- `outputs/figures/pca_*.png` (PCA plots by PD/HC label and language)
- `outputs/tables/classification_results_*.csv` (Raw result tables per layer)
- `outputs/tables/model_layer_comparison.csv` (Aggregated cross-lingual classification results)

## 6. Full Results Table

| model    |   layer | train_language   | test_language   | classifier          |    uar |   accuracy |   sensitivity |   specificity |    auc |   n_train |   n_test |
|:---------|--------:|:-----------------|:----------------|:--------------------|-------:|-----------:|--------------:|--------------:|-------:|----------:|---------:|
| wav2vec2 |       0 | Spanish          | Spanish         | linear_svm          | 0.76   |     0.76   |        0.74   |        0.78   | 0.8186 |       100 |      100 |
| wav2vec2 |       0 | Spanish          | Spanish         | logistic_regression | 0.75   |     0.75   |        0.74   |        0.76   | 0.7744 |       100 |      100 |
| wav2vec2 |       0 | Spanish          | German          | linear_svm          | 0.6477 |     0.6477 |        0.5341 |        0.7614 | 0.7073 |       100 |      176 |
| wav2vec2 |       0 | Spanish          | German          | logistic_regression | 0.6023 |     0.6023 |        0.4545 |        0.75   | 0.6905 |       100 |      176 |
| wav2vec2 |       0 | German           | German          | linear_svm          | 0.733  |     0.733  |        0.75   |        0.7159 | 0.7749 |       176 |      176 |
| wav2vec2 |       0 | German           | German          | logistic_regression | 0.7216 |     0.7216 |        0.7386 |        0.7045 | 0.7868 |       176 |      176 |
| wav2vec2 |       0 | German           | Spanish         | linear_svm          | 0.56   |     0.56   |        0.68   |        0.44   | 0.616  |       176 |      100 |
| wav2vec2 |       0 | German           | Spanish         | logistic_regression | 0.56   |     0.56   |        0.46   |        0.66   | 0.6568 |       176 |      100 |
| wav2vec2 |       0 | Spanish+German   | Spanish+German  | linear_svm          | 0.7645 |     0.7645 |        0.7826 |        0.7464 | 0.8102 |       276 |      276 |
| wav2vec2 |       0 | Spanish+German   | Spanish+German  | logistic_regression | 0.7428 |     0.7428 |        0.7536 |        0.7319 | 0.8004 |       276 |      276 |
| wav2vec2 |      11 | Spanish          | Spanish         | linear_svm          | 0.73   |     0.73   |        0.68   |        0.78   | 0.772  |       100 |      100 |
| wav2vec2 |      11 | Spanish          | Spanish         | logistic_regression | 0.75   |     0.75   |        0.7    |        0.8    | 0.7788 |       100 |      100 |
| wav2vec2 |      11 | Spanish          | German          | linear_svm          | 0.6023 |     0.6023 |        0.7841 |        0.4205 | 0.6267 |       100 |      176 |
| wav2vec2 |      11 | Spanish          | German          | logistic_regression | 0.5625 |     0.5625 |        0.9545 |        0.1705 | 0.6592 |       100 |      176 |
| wav2vec2 |      11 | German           | German          | linear_svm          | 0.75   |     0.75   |        0.7727 |        0.7273 | 0.8167 |       176 |      176 |
| wav2vec2 |      11 | German           | German          | logistic_regression | 0.75   |     0.75   |        0.75   |        0.75   | 0.8097 |       176 |      176 |
| wav2vec2 |      11 | German           | Spanish         | linear_svm          | 0.51   |     0.51   |        0.86   |        0.16   | 0.6656 |       176 |      100 |
| wav2vec2 |      11 | German           | Spanish         | logistic_regression | 0.62   |     0.62   |        0.84   |        0.4    | 0.7356 |       176 |      100 |
| wav2vec2 |      11 | Spanish+German   | Spanish+German  | linear_svm          | 0.7572 |     0.7572 |        0.7174 |        0.7971 | 0.8427 |       276 |      276 |
| wav2vec2 |      11 | Spanish+German   | Spanish+German  | logistic_regression | 0.75   |     0.75   |        0.7319 |        0.7681 | 0.8424 |       276 |      276 |
| wav2vec2 |       4 | Spanish          | Spanish         | linear_svm          | 0.81   |     0.81   |        0.78   |        0.84   | 0.836  |       100 |      100 |
| wav2vec2 |       4 | Spanish          | Spanish         | logistic_regression | 0.79   |     0.79   |        0.74   |        0.84   | 0.8256 |       100 |      100 |
| wav2vec2 |       4 | Spanish          | German          | linear_svm          | 0.5909 |     0.5909 |        0.75   |        0.4318 | 0.6171 |       100 |      176 |
| wav2vec2 |       4 | Spanish          | German          | logistic_regression | 0.6648 |     0.6648 |        0.8636 |        0.4659 | 0.6934 |       100 |      176 |
| wav2vec2 |       4 | German           | German          | linear_svm          | 0.7159 |     0.7159 |        0.7045 |        0.7273 | 0.7995 |       176 |      176 |
| wav2vec2 |       4 | German           | German          | logistic_regression | 0.7784 |     0.7784 |        0.7727 |        0.7841 | 0.8439 |       176 |      176 |
| wav2vec2 |       4 | German           | Spanish         | linear_svm          | 0.52   |     0.52   |        0.08   |        0.96   | 0.6524 |       176 |      100 |
| wav2vec2 |       4 | German           | Spanish         | logistic_regression | 0.56   |     0.56   |        0.16   |        0.96   | 0.7676 |       176 |      100 |
| wav2vec2 |       4 | Spanish+German   | Spanish+German  | linear_svm          | 0.7428 |     0.7428 |        0.7609 |        0.7246 | 0.8247 |       276 |      276 |
| wav2vec2 |       4 | Spanish+German   | Spanish+German  | logistic_regression | 0.7754 |     0.7754 |        0.7681 |        0.7826 | 0.859  |       276 |      276 |
| wav2vec2 |       8 | Spanish          | Spanish         | linear_svm          | 0.76   |     0.76   |        0.74   |        0.78   | 0.792  |       100 |      100 |
| wav2vec2 |       8 | Spanish          | Spanish         | logistic_regression | 0.79   |     0.79   |        0.76   |        0.82   | 0.8296 |       100 |      100 |
| wav2vec2 |       8 | Spanish          | German          | linear_svm          | 0.5568 |     0.5568 |        0.7955 |        0.3182 | 0.601  |       100 |      176 |
| wav2vec2 |       8 | Spanish          | German          | logistic_regression | 0.5739 |     0.5739 |        0.875  |        0.2727 | 0.6609 |       100 |      176 |
| wav2vec2 |       8 | German           | German          | linear_svm          | 0.6932 |     0.6932 |        0.6591 |        0.7273 | 0.7555 |       176 |      176 |
| wav2vec2 |       8 | German           | German          | logistic_regression | 0.7614 |     0.7614 |        0.7614 |        0.7614 | 0.7957 |       176 |      176 |
| wav2vec2 |       8 | German           | Spanish         | linear_svm          | 0.54   |     0.54   |        0.1    |        0.98   | 0.7316 |       176 |      100 |
| wav2vec2 |       8 | German           | Spanish         | logistic_regression | 0.63   |     0.63   |        0.42   |        0.84   | 0.7808 |       176 |      100 |
| wav2vec2 |       8 | Spanish+German   | Spanish+German  | linear_svm          | 0.7536 |     0.7536 |        0.7464 |        0.7609 | 0.8137 |       276 |      276 |
| wav2vec2 |       8 | Spanish+German   | Spanish+German  | logistic_regression | 0.808  |     0.808  |        0.8116 |        0.8043 | 0.8427 |       276 |      276 |