# Scientific Interpretation Corrections Note: 80-File Validation Report

**Experiment:** Independent 80-File True Cross-Language Validation  
**Location:** `true_cross_language_vc/21_validation_80/`  
**Date:** 2026-09-18  
**Document Purpose:** Formal record of rigorous scientific corrections and clarified interpretations for the independent 80-file validation report prior to full-cohort extension.

---

## 1. Context and Rationale

The independent 80-file validation provides the primary confirmatory validation evidence for the true cross-language voice conversion pipeline. To maintain rigorous scientific standards, eliminate potential over-claiming, and adhere strictly to formal statistical criteria, this note records four specific interpretive corrections to the master validation report.

The original report remains preserved in its historical form; this document serves as the authoritative scientific addendum and interpretation guide.

---

## 2. Mandatory Scientific Corrections

### A. Pathology Feature Direction Preservation Gate
- **Observed Result:** 68.2% (15 of 22 evaluated feature contrasts preserved in effect sign).
- **Predefined Engineering Target:** 70.0%.
- **Corrected Scientific Statement:**
  > The observed pathology direction preservation rate (68.2%) is **below the predefined 70% engineering target**, although a majority of pathology-related feature directions were retained. The gate status is strictly recorded as **BELOW PREDEFINED TARGET (68.2% vs. 70.0%)**, rather than passed.

### B. Statistical Significance in Spanish $\to$ German PD Classification
- **Observed Test Statistics (Logistic Regression):** Nominal bootstrap $p = 0.0250$; Benjamini-Hochberg FDR corrected $q = 0.1000$.
- **Corrected Scientific Statement:**
  > The classification improvement in the Spanish $\to$ German direction ($\Delta\text{UAR} = +0.1500$) is **nominally significant before FDR correction ($p = 0.025$) but not significant after FDR correction ($q = 0.100$)**. Consequently, it must not be reported as statistically significant under formal family-wise multiple testing control.

### C. Scope of Cross-Language Diagnostic Generalization
- **Observed Finding:** Spanish $\to$ German showed substantial classification gains ($\Delta\text{UAR} = +0.1250$ to $+0.1500$), whereas German $\to$ Spanish exhibited a decrement ($\Delta\text{UAR} = -0.0500$ to $-0.1500$).
- **Corrected Scientific Statement:**
  > The experimental evidence does not warrant claiming that the linguistic language barrier was "fully overcome" or "proven resolved." Instead:  
  > **Spanish $\to$ German conversion substantially improved cross-language PD classification in the independent 80-file validation, whereas German $\to$ Spanish conversion did not.**

### D. Pathology Normalization Hypothesis Testing
- **Observed Centroid Distance:** Original PD to HC centroid = 4.708; Generated PD to HC centroid = 4.715; Wilcoxon signed-rank $W = 181.0, p = 0.4125$.
- **Corrected Scientific Statement:**
  > A failure to reject the null hypothesis ($p = 0.4125 > 0.05$) does not demonstrate equivalence or the absence of normalization. The correct scientific conclusion is:  
  > **No statistically significant evidence of disproportionate PD-to-HC normalization was detected.**

---

## 3. Scientific Hierarchy Moving Forward

1. **Independent 80-File Validation (`21_validation_80`)**: Remains the **PRIMARY CONFIRMATORY VALIDATION** result of the project.
2. **Full 276-File Evaluation (`22_full_276_evaluation`)**: Represents a **FULL-COHORT EXTENSION / SENSITIVITY ANALYSIS**, not another independent validation, because it subsumes development and validation subjects.
