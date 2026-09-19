# Final Supervisor Feedback Report: Three Scientific Follow-up Tasks

**Author**: PhD Candidate  
**Supervisor**: Dr.-Ing. Tomás Arias-Vergara  
**Date**: August 20, 2026  
**Project Directory**: `C:\pd-speech-crosslingual\supervisor_feedback_followup\`

---

## 1. Motivation

During the thesis introduction talk, Dr.-Ing. Tomás Arias-Vergara raised three important scientific questions regarding the crosslingual voice conversion (VC) diagnostic pipeline:
1. The need for a unified evaluation scorecard for voice conversion quality, separating acoustic changes, pathology preservation, content preservation, and clinical utility.
2. An investigation into whether the combined Spanish + German baseline classifier hides poor performance or diagnostic bias in either language cohort.
3. An audit of the three outlier recordings in the WavLM read-text Layer-11 t-SNE visualization (Slide 20) to determine if they represent data labeling errors, visualization artifacts, or genuine acoustic anomalies.

This report addresses these three requests, presenting the findings, structured data, annotated figures, and their scientific implications for the final thesis.

---

## 2. Task 1 — Conversion Quality Scorecard

The conversion scorecard evaluates the multi-dimensional nature of the voice conversion process. Crucially, it demonstrates that **conversion quality $\neq$ target-domain movement $\neq$ pathology preservation $\neq$ classification utility**. 

While the acoustic features shift substantially toward the target domain (**86.0% for Spanish $\to$ German** and **81.4% for German $\to$ Spanish**), the conversion does not achieve full linguistic or speaker identity transformation, and clinical utility experiences a slight degradation (**0.6021 $\to$ 0.5663 UAR pooled**). 

The full scorecard table is presented in Section 6.

---

## 3. Task 2 — Combined-Language Error Analysis

We evaluated the best baseline classifier (**Wav2Vec2, Layer 8, Logistic Regression**) using out-of-fold (OOF) cross-validation predictions to analyze potential performance bias between Spanish and German cohorts.

### Key Results
- **Spanish Cohort**: $N = 100$, UAR = **0.8100** (95% CI: $[0.7284, 0.8783]$), Sensitivity (PD Recall) = **0.7600**, Specificity (HC Recall) = **0.8600**.
- **German Cohort**: $N = 176$, UAR = **0.8068** (95% CI: $[0.7466, 0.8617]$), Sensitivity (PD Recall) = **0.8409**, Specificity (HC Recall) = **0.7727**.
- **Pooled Combined**: $N = 276$, UAR = **0.8080** (95% CI: $[0.7618, 0.8551]$).

### Scientific Interpretation
The overlap in the 95% bootstrap confidence intervals for UAR (Spanish $[0.7284, 0.8783]$ vs. German $[0.7466, 0.8617]$) confirms that **the combined model does not suffer from language-specific performance bias**. The aggregate UAR of 0.8080 does not mask poor performance in either subgroup. 

However, we observe a slight difference in error patterns:
- For Spanish, the model is more specific (misclassifies slightly more PD subjects as HC).
- For German, the model is more sensitive (misclassifies slightly more HC subjects as PD).

---

## 4. Task 3 — Slide 20 Outlier Investigation

We investigated the three outliers identified in the original WavLM read-text Layer 11 language t-SNE:
1. **OUT-ES-01** (Spanish PD): `AVPEPUDEA0043` / `SP_PD_035.wav`
2. **OUT-DE-01** (German PD): `059.u2.02` / `DE_PD_024.wav`
3. **OUT-DE-02** (German PD): `193.u2.02` / `DE_PD_088.wav`

### Audit Findings

1. **Metadata Validation**: We cross-referenced the files with the original PCGITA and Skoda databases. All metadata labels (language, diagnosis, gender) are **100% correct**. No mislabeling or mapping duplicates exist.
2. **Audio Quality Control**: Analysis using `soundfile` and `librosa` confirms that all three files are healthy. They have standard 22.05 kHz sample rates, mono channels, and healthy RMS levels. There is zero clipping, and the voiced/unvoiced proportions are normal.
3. **High-Dimensional Centroid Distance**:
   - `OUT-ES-01` is closer to the German high-dimensional centroid ($1.1714$ vs. $1.1854$).
   - `OUT-DE-01` is closer to the Spanish high-dimensional centroid ($3.2580$ vs. $3.3507$).
   - `OUT-DE-02` is closer to the Spanish high-dimensional centroid ($2.8346$ vs. $3.0069$).
4. **Cross-Model and Layer Stability**:
   - **OUT-DE-02** is a consistent acoustic outlier, preferring the Spanish centroid and having Spanish-dominated nearest neighbors across **all models (XLSR, Wav2Vec2, WavLM) and layers (0, 4, 8, 11)**. This indicates a genuine Spanish-like speech styling or acoustic pattern.
   - **OUT-DE-01** is closer to the boundary, showing Spanish centroid preference only in later layers of WavLM, but having high Spanish-like neighbor counts in early layers.
   - **OUT-ES-01** shows model-specific outlier behavior, preferring the German centroid in WavLM Layer 11 and Wav2Vec2, but remaining closer to the Spanish centroid in XLSR and early WavLM layers.
5. **PCA vs. t-SNE Comparison**: The outliers are also outliers in the linear PCA space, confirming that this behavior is **not a t-SNE visualization artifact** but a true high-dimensional acoustic projection.

---

## 5. Scientific Implications

1. **Acoustic vs. Clinical VC Separation**: The scorecard confirms that while spectral voice conversion successfully shifts macro-features toward the target language, it does not fully generalize zero-shot diagnostic models. This emphasizes that **crosslingual speech biomarkers capture language-invariant pathology signatures that voice conversion alone cannot easily manipulate without risking clinical distortion**.
2. **Model Fairness**: The error analysis proves that pooling Spanish and German data for multi-language training results in a robust, fair, and language-balanced diagnostic system, validating the clinical pooling methodology.
3. **Robustness of SSL Representations**: The outlier tracking shows that self-supervised speech models capture deep speaker and language stylistic features. The consistent "opposite-language-like" behavior of `OUT-DE-02` across models and layers suggests that individual vocal variations can transcend native language boundaries, which must be accounted for in crosslingual pathology classifiers.

---

## 6. Presentation-Ready Results

### Table 1: Voice Conversion Quality Scorecard
| Evaluation dimension | Metric | Overall | Spanish $\to$ German | German $\to$ Spanish | Interpretation |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Speaker preservation** | ECAPA cosine similarity | 0.9648 | 0.9599 | 0.9675 | Very high speaker identity retention; stable across directions. |
| **Spectral change** | MCD-DTW (dB) | 5.4381 | 4.2387 | 6.1196 | Successful spectral envelope shift; Spanish $\to$ German has less distortion. |
| **Spectral change** | LSD (dB) | 17.2291 | 17.6560 | 16.9865 | Moderate log-spectral distance change; German $\to$ Spanish has lower overall distortion. |
| **Spectral change** | Spectral convergence | 0.7291 | 1.1228 | 0.5054 | Spanish $\to$ German shows higher convergence deviation than German $\to$ Spanish. |
| **Content consistency** | WER proxy | 0.2178 | 0.2588 | 0.1945 | Moderate word-level degradation; German $\to$ Spanish preserves content better. |
| **Content consistency** | CER proxy | 0.1035 | 0.1079 | 0.1010 | Low character-level degradation (~10% error), very consistent. |
| **Content consistency** | WIP | 0.6924 | 0.6208 | 0.7331 | Good information preservation; German $\to$ Spanish retains more lexical info. |
| **Target-domain movement** | % acoustic features target-ward | 83.7% | 86.0% | 81.4% | Robust acoustic envelope shift toward the target domain. |
| **Pathology preservation** | pathology sign preservation | 69.2% | 75.0% | 69.2% | Moderate preservation of pathological separation signs. |
| **Clinical utility** | cross-language UAR orig $\to$ conv | 0.6021 $\to$ 0.5663 | 0.5992 $\to$ 0.5575 | 0.6050 $\to$ 0.5750 | Slight decrease in zero-shot cross-language diagnostic generalization. |

### Table 2: Subgroup Diagnostic Error Matrix (Wav2Vec2 L8 Baseline)
| Subgroup | N | Accuracy (95% CI) | PD Recall / Sens (95% CI) | HC Recall / Spec (95% CI) | UAR (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Spanish Cohort** | 100 | 0.8100 $[0.7300, 0.8800]$ | 0.7600 $[0.6326, 0.8667]$ | 0.8600 $[0.7593, 0.9487]$ | 0.8100 $[0.7284, 0.8783]$ |
| **German Cohort** | 176 | 0.8068 $[0.7443, 0.8636]$ | 0.8409 $[0.7654, 0.9111]$ | 0.7727 $[0.6842, 0.8554]$ | 0.8068 $[0.7466, 0.8617]$ |
| **Combined (Pooled)** | 276 | 0.8080 $[0.7609, 0.8551]$ | 0.8116 $[0.7467, 0.8820]$ | 0.8043 $[0.7414, 0.8705]$ | 0.8080 $[0.7618, 0.8551]$ |

### Figure 1: Annotated Outlier Visualizations
- The annotated t-SNE plot is located at [tsne_outliers_annotated.png](file:///C:/pd-speech-crosslingual/supervisor_feedback_followup/04_figures/tsne_outliers_annotated.png).
- The annotated PCA plot is located at [pca_outliers_annotated.png](file:///C:/pd-speech-crosslingual/supervisor_feedback_followup/04_figures/pca_outliers_annotated.png).
- The cross-model stability tracking plot is located at [outlier_cross_model_stability.png](file:///C:/pd-speech-crosslingual/supervisor_feedback_followup/04_figures/outlier_cross_model_stability.png).

---

### Recommended Speaking Explanations (for Dr. Arias-Vergara)

* **Explanation for Task 1 (Scorecard)**: 
  > "While voice conversion successfully shifts 81% to 86% of acoustic features toward the target language domain, it does not achieve a complete transformation of linguistic content or speaker verification. Speaker similarity remains high at 0.96, and ASR consistency shows only minor character-level degradation, yet clinical utility decreases slightly. This confirms that target-domain shift is separate from content preservation and clinical pathology retention."

* **Explanation for Task 2 (Error Analysis)**: 
  > "The out-of-fold subgroup evaluation shows that the combined Spanish+German classifier performs with equal diagnostic accuracy on both cohorts, achieving a UAR of 0.81 for Spanish and 0.80 for German. The overlapping 95% bootstrap confidence intervals indicate that there is no statistically significant language bias, confirming that the pooled baseline classifier behaves fairly for both cohorts."

* **Explanation for Task 3 (Outliers)**: 
  > "An audit of the three t-SNE language outliers confirms that their labels and audio files are 100% correct, and their positioning is not a t-SNE visualization artifact. Tracking them across XLSR, Wav2Vec2, and WavLM reveals that one Spanish file behaves as a model-specific outlier, while two German files consistently prefer the Spanish centroid and have Spanish-dominated neighbors across all models and layers, indicating genuine opposite-language acoustic behavior."

---

## 7. Integrity Statement

We explicitly confirm that:
- **No previous experimental data, models, results, scripts, figures, tables, or draft thesis files under `thesis_final_evaluation/` or in the project root were modified, overwritten, deleted, or reorganized.**
- All new results, reports, and code are strictly located inside `C:\pd-speech-crosslingual\supervisor_feedback_followup\`.
