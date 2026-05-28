# Multilingual Parkinson's Speech Detection Baseline

## 1. Dataset Summary
- **Task**: readtext
- **Total Recordings**: 276
- **Languages**: Spanish (100), German (176)
- **Labels**: PD (138), HC (138)

## 2. Experimental Setup
- **Model Used**: facebook/wav2vec2-large-xlsr-53
- **Layers Evaluated**: [0, 4, 8, 11]
- **Classifiers Used**: ['linear_svm', 'logistic_regression']

## 3. Best Performing Configurations (by UAR)

**Best Overall Layer**: Layer 4 (UAR: 0.8400 via Spanish->Spanish)

### Best Within-Language Result (Spanish → Spanish)
- **UAR**: 0.8400
- **Model/Layer**: xlsr / Layer 4
- **Classifier**: linear_svm
- **Sensitivity**: 0.8400
- **Specificity**: 0.8400
- **AUC**: 0.8636

### Cross-lingual: Spanish to German
- **UAR**: 0.6477
- **Model/Layer**: xlsr / Layer 8
- **Classifier**: logistic_regression
- **Sensitivity**: 0.5568
- **Specificity**: 0.7386
- **AUC**: 0.7136

### Best Within-Language Result (German → German)
- **UAR**: 0.7670
- **Model/Layer**: xlsr / Layer 11
- **Classifier**: logistic_regression
- **Sensitivity**: 0.7386
- **Specificity**: 0.7955
- **AUC**: 0.8319

### Cross-lingual: German to Spanish
- **UAR**: 0.6700
- **Model/Layer**: xlsr / Layer 8
- **Classifier**: logistic_regression
- **Sensitivity**: 0.7200
- **Specificity**: 0.6200
- **AUC**: 0.7452

### Spanish+German → Spanish+German
- **UAR**: 0.7826
- **Model/Layer**: xlsr / Layer 8
- **Classifier**: logistic_regression
- **Sensitivity**: 0.7681
- **Specificity**: 0.7971
- **AUC**: 0.8512

## 4. Interpretation on Language Mismatch
Yes, there appears to be a notable domain mismatch. Training on Spanish and testing on German caused a performance drop of 0.1923 UAR compared to within-language testing.

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
| xlsr    |       0 | Spanish          | Spanish         | linear_svm          | 0.78   |     0.78   |        0.8    |        0.76   | 0.8348 |       100 |      100 |
| xlsr    |       0 | Spanish          | Spanish         | logistic_regression | 0.78   |     0.78   |        0.78   |        0.78   | 0.8296 |       100 |      100 |
| xlsr    |       0 | Spanish          | German          | linear_svm          | 0.5568 |     0.5568 |        0.2386 |        0.875  | 0.6455 |       100 |      176 |
| xlsr    |       0 | Spanish          | German          | logistic_regression | 0.5852 |     0.5852 |        0.2955 |        0.875  | 0.6719 |       100 |      176 |
| xlsr    |       0 | German           | German          | linear_svm          | 0.7443 |     0.7443 |        0.75   |        0.7386 | 0.7722 |       176 |      176 |
| xlsr    |       0 | German           | German          | logistic_regression | 0.75   |     0.75   |        0.7614 |        0.7386 | 0.7905 |       176 |      176 |
| xlsr    |       0 | German           | Spanish         | linear_svm          | 0.6    |     0.6    |        0.72   |        0.48   | 0.6954 |       176 |      100 |
| xlsr    |       0 | German           | Spanish         | logistic_regression | 0.6    |     0.6    |        0.68   |        0.52   | 0.6668 |       176 |      100 |
| xlsr    |       0 | Spanish+German   | Spanish+German  | linear_svm          | 0.75   |     0.75   |        0.7536 |        0.7464 | 0.8083 |       276 |      276 |
| xlsr    |       0 | Spanish+German   | Spanish+German  | logistic_regression | 0.7572 |     0.7572 |        0.7464 |        0.7681 | 0.8179 |       276 |      276 |
| xlsr    |      11 | Spanish          | Spanish         | linear_svm          | 0.78   |     0.78   |        0.72   |        0.84   | 0.876  |       100 |      100 |
| xlsr    |      11 | Spanish          | Spanish         | logistic_regression | 0.81   |     0.81   |        0.72   |        0.9    | 0.872  |       100 |      100 |
| xlsr    |      11 | Spanish          | German          | linear_svm          | 0.5909 |     0.5909 |        0.2955 |        0.8864 | 0.6856 |       100 |      176 |
| xlsr    |      11 | Spanish          | German          | logistic_regression | 0.5739 |     0.5739 |        0.2386 |        0.9091 | 0.7129 |       100 |      176 |
| xlsr    |      11 | German           | German          | linear_svm          | 0.75   |     0.75   |        0.7386 |        0.7614 | 0.821  |       176 |      176 |
| xlsr    |      11 | German           | German          | logistic_regression | 0.767  |     0.767  |        0.7386 |        0.7955 | 0.8319 |       176 |      176 |
| xlsr    |      11 | German           | Spanish         | linear_svm          | 0.6    |     0.6    |        0.68   |        0.52   | 0.6328 |       176 |      100 |
| xlsr    |      11 | German           | Spanish         | logistic_regression | 0.66   |     0.66   |        0.88   |        0.44   | 0.7756 |       176 |      100 |
| xlsr    |      11 | Spanish+German   | Spanish+German  | linear_svm          | 0.7536 |     0.7536 |        0.7681 |        0.7391 | 0.833  |       276 |      276 |
| xlsr    |      11 | Spanish+German   | Spanish+German  | logistic_regression | 0.779  |     0.779  |        0.7609 |        0.7971 | 0.8545 |       276 |      276 |
| xlsr    |       4 | Spanish          | Spanish         | linear_svm          | 0.84   |     0.84   |        0.84   |        0.84   | 0.8636 |       100 |      100 |
| xlsr    |       4 | Spanish          | Spanish         | logistic_regression | 0.82   |     0.82   |        0.78   |        0.86   | 0.8808 |       100 |      100 |
| xlsr    |       4 | Spanish          | German          | linear_svm          | 0.5511 |     0.5511 |        0.8636 |        0.2386 | 0.6109 |       100 |      176 |
| xlsr    |       4 | Spanish          | German          | logistic_regression | 0.5341 |     0.5341 |        0.8523 |        0.2159 | 0.6373 |       100 |      176 |
| xlsr    |       4 | German           | German          | linear_svm          | 0.7159 |     0.7159 |        0.7386 |        0.6932 | 0.7979 |       176 |      176 |
| xlsr    |       4 | German           | German          | logistic_regression | 0.7614 |     0.7614 |        0.7955 |        0.7273 | 0.8283 |       176 |      176 |
| xlsr    |       4 | German           | Spanish         | linear_svm          | 0.62   |     0.62   |        0.24   |        1      | 0.7276 |       176 |      100 |
| xlsr    |       4 | German           | Spanish         | logistic_regression | 0.61   |     0.61   |        0.22   |        1      | 0.7604 |       176 |      100 |
| xlsr    |       4 | Spanish+German   | Spanish+German  | linear_svm          | 0.7609 |     0.7609 |        0.7609 |        0.7609 | 0.8401 |       276 |      276 |
| xlsr    |       4 | Spanish+German   | Spanish+German  | logistic_regression | 0.779  |     0.779  |        0.7826 |        0.7754 | 0.8265 |       276 |      276 |
| xlsr    |       8 | Spanish          | Spanish         | linear_svm          | 0.79   |     0.79   |        0.74   |        0.84   | 0.8528 |       100 |      100 |
| xlsr    |       8 | Spanish          | Spanish         | logistic_regression | 0.83   |     0.83   |        0.78   |        0.88   | 0.8796 |       100 |      100 |
| xlsr    |       8 | Spanish          | German          | linear_svm          | 0.5966 |     0.5966 |        0.4205 |        0.7727 | 0.6525 |       100 |      176 |
| xlsr    |       8 | Spanish          | German          | logistic_regression | 0.6477 |     0.6477 |        0.5568 |        0.7386 | 0.7136 |       100 |      176 |
| xlsr    |       8 | German           | German          | linear_svm          | 0.7273 |     0.7273 |        0.7273 |        0.7273 | 0.7949 |       176 |      176 |
| xlsr    |       8 | German           | German          | logistic_regression | 0.7386 |     0.7386 |        0.7386 |        0.7386 | 0.8224 |       176 |      176 |
| xlsr    |       8 | German           | Spanish         | linear_svm          | 0.56   |     0.56   |        0.34   |        0.78   | 0.6308 |       176 |      100 |
| xlsr    |       8 | German           | Spanish         | logistic_regression | 0.67   |     0.67   |        0.72   |        0.62   | 0.7452 |       176 |      100 |
| xlsr    |       8 | Spanish+German   | Spanish+German  | linear_svm          | 0.7681 |     0.7681 |        0.7681 |        0.7681 | 0.8437 |       276 |      276 |
| xlsr    |       8 | Spanish+German   | Spanish+German  | logistic_regression | 0.7826 |     0.7826 |        0.7681 |        0.7971 | 0.8512 |       276 |      276 |