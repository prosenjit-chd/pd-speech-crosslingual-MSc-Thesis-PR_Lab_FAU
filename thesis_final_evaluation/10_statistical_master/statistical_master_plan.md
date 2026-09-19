# Preregistered-Style Statistical Master Plan & Methodology

## 1. Experimental Design & Hypothesis Families
To prevent statistical inflation (Type I error rate inflation) from testing dozens of acoustic features and model configurations, all statistical inferences are strictly structured into **5 Preregistered Hypothesis Families**. Multiple testing corrections are applied **family-wise using the Benjamini-Hochberg False Discovery Rate (FDR)** procedure at $\alpha = 0.05$:

1. **Family 1: Language Disparity & Baseline Gaps (Question A)**
   - *Hypotheses*: Cross-language classification gap, SSL representation separability, acoustic language differences ($N=43$), and Language $\times$ Pathology 2-Way ANOVA interactions ($N=43$).
   - *Statistical Tests*: Two-sample Mann-Whitney U, Paired t-tests, Type-II 2-Way ANOVA F-tests.
2. **Family 2: Feature-Level Domain Shift (Question B)**
   - *Hypotheses*: Reduction in source-to-target standardized distances across 43 features and frozen language classifier migration.
   - *Statistical Tests*: Paired Wilcoxon signed-rank tests ($N=86$), paired t-tests.
3. **Family 3: Pathology Preservation & Centroid Shift (Question C)**
   - *Hypotheses*: Retention of PD-vs-HC acoustic separation after conversion and absence of disproportionate PD-to-HC normalization.
   - *Statistical Tests*: Two-sample Mann-Whitney U tests ($N=43$), Welch's two-sample t-test.
4. **Family 4: Conversion Quality & Biometrics (Phases 5–7)**
   - *Hypotheses*: ECAPA-TDNN speaker verification margins, and PD vs. HC quality disparity tests (ECAPA-TDNN, MCD, ASR WER).
   - *Statistical Tests*: Mann-Whitney U, Welch's t-test.
5. **Family 5: Cross-Language Classification Synthesis (Question D)**
   - *Hypotheses*: Effect of Stage 5C conversion on pure cross-language UAR ($N=48$), per-direction cross-language UAR ($N=24$), and within-language resynthesis artifacts ($N=48$).
   - *Statistical Tests*: Paired t-tests, Wilcoxon signed-rank tests.

---

## 2. Family-Wise FDR Correction Summary Table
| Hypothesis Family | Total Tests ($N$) | Raw Significant ($p < 0.05$) | FDR Corrected ($q < 0.05$) | FDR Retention Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Family 1: Language Disparity & Baseline Gap (Question A)** | 1 | 1 | **1** | 100.0% |
| **Family 1: Acoustic Language Differences (Question A)** | 52 | 39 | **38** | 97.4% |
| **Family 1: Language x Pathology Interactions (Question A)** | 52 | 12 | **7** | 58.3% |
| **Family 2: Feature-Level Domain Movement (Question B)** | 104 | 60 | **56** | 93.3% |
| **Family 2: Frozen Language Classifier Target Shift (Question B)** | 1 | 1 | **1** | 100.0% |
| **Family 3: Converted Pathology Separation (Question C)** | 52 | 16 | **12** | 75.0% |
| **Family 3: Disproportionate Pathology Normalization (Question C)** | 1 | 0 | **0** | 100.0% |
| **Family 4: Conversion Quality & Biometrics** | 4 | 3 | **3** | 100.0% |
| **Family 5: Cross-Language Classification Synthesis (Question D)** | 4 | 4 | **4** | 100.0% |

---

## 3. Decision Rules for Inferential Confirmation
- **CONFIRMED ROBUST**: Test achieves $p < 0.05$ AND $q < 0.05$ after family-wise Benjamini-Hochberg FDR correction.
- **CONFIRMED NULL**: Test shows absence of hypothesized effect ($p \ge 0.05$, with tight confidence interval verifying equivalence).
- **EXPLORATORY / WEAK**: Test achieves raw $p < 0.05$ but loses significance after FDR correction ($q \ge 0.05$), or effect size is small ($d < 0.20$).
- **REFUTED**: Direction of effect is opposite to hypothesis (e.g. cross-language UAR degrades rather than improves, $p < 0.05$ in negative direction).

---
> [!NOTE]
> **Phase 9 Status: COMPLETED (PASSED)**
> Complete master statistical plan and family-wise FDR corrected result matrix successfully compiled. Ready to generate publication-quality figures (Phase 10).
