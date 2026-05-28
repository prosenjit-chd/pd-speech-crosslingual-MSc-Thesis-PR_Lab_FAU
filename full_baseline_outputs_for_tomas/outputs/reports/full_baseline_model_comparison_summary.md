# Full Baseline Model Comparison Summary

## 1. Dataset Summary
- **Task**: readtext
- **Total Recordings**: 276
- **Languages**: Spanish (100), German (176)
- **Labels**: PD (138), HC (138)

## 2. Models Evaluated
- wav2vec2
- wavlm
- xlsr

## 3. Layers & Classifiers
- **Layers Evaluated**: [0, 4, 8, 11]
- **Classifiers**: Linear SVM, Logistic Regression

## 4. Best Results Per Scenario

### Best Within-Language (Spanish)
- **Best UAR**: 0.8400
- **Model**: xlsr
- **Layer**: 4
- **Classifier**: linear_svm
- **Accuracy**: 0.8400
- **Sensitivity**: 0.8400
- **Specificity**: 0.8400
- **AUC**: 0.8636

### Best Within-Language (German)
- **Best UAR**: 0.8182
- **Model**: wavlm
- **Layer**: 8
- **Classifier**: logistic_regression
- **Accuracy**: 0.8182
- **Sensitivity**: 0.8068
- **Specificity**: 0.8295
- **AUC**: 0.8680

### Best Cross-Language (Spanish -> German)
- **Best UAR**: 0.7273
- **Model**: wavlm
- **Layer**: 11
- **Classifier**: logistic_regression
- **Accuracy**: 0.7273
- **Sensitivity**: 0.8750
- **Specificity**: 0.5795
- **AUC**: 0.7487

### Best Cross-Language (German -> Spanish)
- **Best UAR**: 0.6900
- **Model**: wavlm
- **Layer**: 0
- **Classifier**: logistic_regression
- **Accuracy**: 0.6900
- **Sensitivity**: 0.6200
- **Specificity**: 0.7600
- **AUC**: 0.7488

### Best Combined Language (Spanish+German)
- **Best UAR**: 0.8080
- **Model**: wav2vec2
- **Layer**: 8
- **Classifier**: logistic_regression
- **Accuracy**: 0.8080
- **Sensitivity**: 0.8116
- **Specificity**: 0.8043
- **AUC**: 0.8427

## 5. Model Rankings & Interpretation

**Overall Best Performing Configuration**:
- Model: xlsr, Layer: 4, Scenario: Spanish->Spanish (UAR: 0.8400)

**Language Mismatch Conclusion**:
Yes, a domain mismatch is clearly visible. For instance, the best Spanish->Spanish UAR is 0.8400, but drops to 0.7273 when transferring to German.

**Final Recommendation for Baseline**:
The **xlsr** model provides the strongest overall representations for speech classification in this dataset. This model should be used as the definitive reference baseline prior to introducing Voice Conversion.

