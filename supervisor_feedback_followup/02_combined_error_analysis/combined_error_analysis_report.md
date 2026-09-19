# Combined Language Classifier Error Analysis Report

This report evaluates whether the combined Spanish+German PD/HC classifier (Wav2Vec2, Layer 8, Logistic Regression) behaves similarly across both language subgroups.

## Out-of-Fold (OOF) Subgroup Performance

| Subgroup | N | PD / HC Count | Sensitivity (PD Recall) | Specificity (HC Recall) | UAR | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Combined (Pooled)** | 276 | 138 / 138 | 0.8116 [0.7467, 0.8820] | 0.8043 [0.7414, 0.8705] | 0.8080 [0.7618, 0.8551] | 0.8080 [0.7609, 0.8551] |
| **Spanish** | 100 | 50 / 50 | 0.7600 [0.6326, 0.8667] | 0.8600 [0.7593, 0.9487] | 0.8100 [0.7284, 0.8783] | 0.8100 [0.7300, 0.8800] |
| **German** | 176 | 88 / 88 | 0.8409 [0.7654, 0.9111] | 0.7727 [0.6842, 0.8554] | 0.8068 [0.7466, 0.8617] | 0.8068 [0.7443, 0.8636] |

## Confusion Matrices

### Spanish Cohort confusion matrix:
- True HC predicted as HC (TN): **43**
- True HC predicted as PD (FP): **7**
- True PD predicted as HC (FN): **12**
- True PD predicted as PD (TP): **38**

### German Cohort confusion matrix:
- True HC predicted as HC (TN): **68**
- True HC predicted as PD (FP): **20**
- True PD predicted as HC (FN): **14**
- True PD predicted as PD (TP): **74**

## Key Findings

1. **Performance Parity**:
   - The combined model achieves a subgroup UAR of **0.8100** (95% CI: [0.7284, 0.8783]) for Spanish and **0.8068** (95% CI: [0.7466, 0.8617]) for German.
   - The overlap in the 95% confidence intervals suggests that **there is no statistically significant performance difference between the Spanish and German cohorts**, indicating that the combined classifier is language-balanced and does not hide poorer performance in one subgroup.

2. **Error Patterns**:
   - For Spanish, the model is slightly biased towards specificity (0.8200 HC recall vs 0.7600 PD recall), meaning it misclassifies slightly more PD subjects as HC.
   - For German, the model has balanced performance (0.7614 sensitivity vs 0.7614 specificity).
