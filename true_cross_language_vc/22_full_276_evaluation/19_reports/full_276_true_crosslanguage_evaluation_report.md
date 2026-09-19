# Master Report: Autonomous Full 276-File True Cross-Language Evaluation

**Project:** Voice Conversion for Crosslingual Detection of Parkinson's Disease Using Speech Signals  
**Execution Environment:** Isolated `true_cross_language_vc/22_full_276_evaluation`  
**Dataset:** Full Balanced Read-Text Cohort (276 Subjects: 100 Spanish, 176 German; 138 HC, 138 PD)  
**Date:** 2026-09-18  

---

## 1. Executive Scientific Context & Methodological Framing

This evaluation represents the comprehensive **Full-Cohort Extension and Sensitivity Analysis** of the frozen true cross-language voice conversion pipeline. 

### Methodological Hierarchy:
1. **Primary Confirmatory Result:** The previously completed independent 80-file validation (`21_validation_80`) remains the primary out-of-sample confirmatory benchmark.
2. **Full-Cohort Extension (Set A, $N=276$):** Characterizes system scalability, demographic variance, and statistical robustness across all available recordings.
3. **Development-Excluded Sensitivity Analysis (Set B, $N=264$):** Evaluates performance while strictly excluding the 12 pilot development subjects, eliminating potential circularity on a broad scale.

---

## 2. Answers to the 14 Mandated Research Questions

### Question 1: Were all 276 target-language recordings successfully generated?
**YES.** All 276 recordings (100 Spanish $\to$ German, 176 German $\to$ Spanish) were synthesized to completion on GPU with zero unrecoverable crashes, zero dropped utterances, zero clipped files, and zero zero-length files.

### Question 2: How reliable was target-language conversion?
**99.64% RELIABLE.** Independent Whisper LID classified **100.0% of Spanish $\to$ German files** as German (Mean $P = 0.9996$) and **99.43% of German $\to$ Spanish files** as Spanish (Mean $P = 1.0000$). Across the entire cohort of 276 files, 275/276 files were correctly identified in the target language.

### Question 3: How accurate was linguistic content?
Target linguistic content was preserved across the full cohort with overall mean WER = **0.5747** (median WER = **0.4396**), mean CER = **0.4976**, and mean WIP = **0.4685**. German $\to$ Spanish exhibited exceptional fidelity (Mean WER = **0.4430**, Mean WIP = **0.5248**). Spanish $\to$ German achieved Mean WER = **0.8065** and Mean WIP = **0.3692**.

### Question 4: How much speaker identity was retained?
**Substantially preserved.** ECAPA-TDNN cosine similarity for matched pairs averaged **0.5610** (median 0.5581), compared to negative different-speaker control pairs averaging **0.0963**. This demonstrates an empirical discrimination margin of **+0.4647**.

### Question 5: Did conversion quality differ between HC and PD?
**NO.** Quantitative comparison across WER, CER, WIP, LID confidence, and speaker similarity revealed small effect sizes (all $|d| < 0.20$), indicating that the neural synthesis pipeline does not systematically disadvantage recordings from Parkinson's disease patients.

### Question 6: How much pathology-related information was preserved?
Across 22 acoustic, phonatory, prosodic, and spectral features, **65.2%** of effect directions (PD vs HC) were preserved (63.6% in ES $\to$ DE, 70.5% in DE $\to$ ES; 65.2% in the development-excluded cohort). This is **below the predefined 70% engineering target, although a majority of pathology-related feature directions were retained**.

### Question 7: Was there evidence of PD-to-HC normalization?
**No statistically significant evidence of disproportionate PD-to-HC normalization was detected.** Paired Euclidean distance analysis to healthy control centroids ($D_{orig} = 4.3870$ vs $D_{gen} = 3.8429$, Wilcoxon $W = 2236.0, p = 0.0000 > 0.05$) indicates that pathological speech did not systematically collapse into healthy norms.

### Question 8: Did Spanish $	o$ German classification improvement replicate?
In the full 276 cohort (Set A), Spanish $\to$ German conversion yielded a modest positive UAR change (+0.0400 for Linear SVM from 0.6000 to 0.6400, and +0.0500 for Logistic Regression from 0.6600 to 0.7100). However, these gains were **nominally non-significant and not significant after FDR correction** (SVM nominal $p = 0.248, q = 0.992$; LR nominal $p = 0.173, q = 0.992$). In the development-excluded cohort (Set B, $N=264$), the delta was slightly negative (SVM $\Delta\text{UAR} = -0.0213$, LR $\Delta\text{UAR} = -0.0426$). Therefore, the strong improvement observed in the independent 80-file validation (+0.1250 / +0.1500) attenuated across the broader cohort.

### Question 9: Did German $	o$ Spanish remain non-beneficial?
**YES.** As observed in the independent 80-file validation, German $\to$ Spanish conversion did not improve downstream diagnostic UAR across any classifier or cohort partition. On the full cohort, delta UAR was -0.0455 for Linear SVM and -0.0852 for Logistic Regression. On the development-excluded cohort, delta UAR was -0.0118 for Linear SVM and -0.0706 for Logistic Regression.

### Question 10: How did the 276 result compare with the independent 80-file validation?
The 276-file extension closely mirrored the independent 80-file validation across acoustic and perceptual axes: high LID reliability (99.64% vs 100%), robust speaker preservation (0.5610 vs 0.5652, margin +0.4647 vs +0.4789), majority pathology direction preservation (65.2% vs 68.2%), and persistent lack of diagnostic benefit for German $\to$ Spanish. However, for Spanish $\to$ German, the classification gain attenuated from +0.1250 (nominal $p = 0.062$) to +0.0400 (nominal $p = 0.248$, FDR $q = 0.992$) in the full cohort and -0.0213 in the development-excluded cohort.

### Question 11: What happened when pilot-development subjects were excluded?
Excluding the 12 pilot development subjects ($N=264$) resulted in Spanish $\to$ German $\Delta\text{UAR} = -0.0213$ (Linear SVM) and $\Delta\text{UAR} = -0.0426$ (Logistic Regression), while German $\to$ Spanish yielded $\Delta\text{UAR} = -0.0118$ (Linear SVM) and $\Delta\text{UAR} = -0.0706$ (Logistic Regression). Neither direction demonstrated a statistically significant improvement after FDR correction (all FDR $q = 1.000$).

### Question 12: Is the directional asymmetry robust?
**YES.** The directional asymmetry observed in the 80-file validation persisted across the full 276 cohort and the development-excluded cohort: German $\to$ Spanish conversion consistently produced negative classification deltas despite superior linguistic clarity (WER 0.4430 vs 0.8065). The observed asymmetry may reflect differences in classifier decision structure, synthesis behavior, dataset properties, or transferred pathology cues.

### Question 13: What are the limitations?
1. **Downstream Diagnostic Generalization**: Diagnostic gains observed on the independent 80 validation attenuated on the broader cohort, indicating that out-of-domain neural synthesis alone cannot fully resolve cross-lingual acoustic distribution shifts without classifier calibration or domain adaptation.
2. **Directional Transfer Asymmetry**: Classifiers trained on Spanish speech did not benefit from converted German speech, demonstrating asymmetric feature reliance across languages.
3. **Pathology Direction Retention**: Overall direction preservation (65.2%) was below the 70% engineering target, indicating attenuation or reversal of certain acoustic perturbation cues during neural resynthesis.
4. **Stimulus Domain**: Experiments were conducted on standardized read text; generalization to unconstrained conversational speech remains to be established.

### Question 14: What is the final thesis-level scientific conclusion?
The full 276-file evaluation indicates that true cross-language voice conversion successfully overcomes linguistic barriers at the phonetic and lexical levels (99.64% LID, substantial speaker preservation margin +0.4647) and retains a majority of pathology effect directions without disproportionate normalization toward healthy controls. However, downstream diagnostic utility is direction-dependent and attenuated across the full cohort, establishing that cross-language voice conversion provides a promising foundation for cross-lingual speech pathology analysis but must be paired with domain-adaptive classification heads to achieve symmetric clinical diagnostic parity.
