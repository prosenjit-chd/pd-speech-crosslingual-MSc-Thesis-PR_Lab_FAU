# Master Thesis Chapter-by-Chapter Actionable Revision Map

**Master's Thesis:** Voice Conversion for Crosslingual Detection of Parkinson's Disease Using Speech Signals  
**Student:** Prosenjit Chowdhury  
**Supervisor:** Dr.-Ing. Tomás Arias-Vergara  
**Institution:** Pattern Recognition Lab (PRL), Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)  

> [!CRITICAL]
> **Strict Non-Destructive Directive:** In accordance with scientific research protocols, the existing thesis draft files remain completely unmodified in read-only status. This revision map specifies exact paragraph-level updates, table/figure insertions, and claim calibrations for the final thesis manuscript.

---

## Chapter 1: Introduction & Research Hypotheses
### 1. Context and Problem Statement
- **Current Draft Issue:** May frame voice conversion as an unreservedly successful technique that bridges the cross-language diagnostic gap.
- **Required Textual Modification:**
  * Frame the thesis investigation around four systematic, preregistered research questions:
    1. **Question A (The Problem):** What is the exact magnitude and acoustic origin of the cross-language classification gap in Parkinson's detection?
    2. **Question B (Domain Shift):** Does embedding-conditioned voice conversion effectively shift source-language acoustic feature distributions toward the target domain?
    3. **Question C (Pathology Retention):** Does voice conversion preserve subtle dysarthric biomarkers without artificially normalizing PD speech toward healthy control distributions?
    4. **Question D (Cross-Language Efficacy):** Does voice conversion improve cross-language Parkinson's disease classification over unconverted baselines?
  * Clarify upfront that empirical results reveal a nuanced scientific dichotomy: while VC achieves strong spectral adaptation, speaker preservation, and biomarker retention, spectral shift alone is insufficient to bridge cross-lingual diagnostic performance due to macro-temporal and phonetic invariance.

---

## Chapter 2: State of the Art & Related Work
- **Required Additions:**
  * Review recent advances in Self-Supervised Learning (SSL) representations (Wav2Vec 2.0, XLSR-53, WavLM) for dysarthria detection.
  * Cite literature on the acoustic-phonetic disparities between Romance (Spanish) and Germanic (German) languages, specifically pitch range, vowel space expansion, and pause structures.
  * Contrast parallel vs. non-parallel voice conversion paradigms and identify the structural limitations of frame-level vocoders (e.g. HiFi-GAN) regarding temporal rhythm transformation.

---

## Chapter 3: Methodology & Experimental Framework
### 1. Dataset Description
- **Insert [Table A](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_A_inventory_summary.md):** Demographics and audio inventory summary detailing the 276 Spanish (PC-GITA) and German (G-Read) original recordings and their 276 matched Stage 5C conversions.
### 2. Feature Extraction & Quality Metrics
- **Add Section 3.X (Interpretable Acoustic Features):** Reference the standardized 43-feature Praat/Librosa acoustic extraction pipeline ([Feature Dictionary](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/01_acoustic_features/feature_dictionary.md)).
- **Add Section 3.Y (Objective Quality Metrics):**
  * Speaker Verification: Pretrained ECAPA-TDNN cosine similarity vs. negative reference distributions.
  * Spectral Distortion: Kubichek (1993) Mel-Cepstral Distortion with dynamic time warping (MCD-DTW), Log-Spectral Distance (LSD), and Spectral Convergence (SC).
  * Linguistic Consistency: OpenAI Whisper-base multilingual ASR with forced language decoding and Word Information Preserved (WIP) metrics.
### 3. Statistical Testing Framework
- **Insert Description of the 5 Preregistered Hypothesis Families:** Detail the application of Benjamini-Hochberg False Discovery Rate (FDR) corrections at $lpha = 0.05$ across 271 statistical tests ([Statistical Master Plan](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/10_statistical_master/statistical_master_plan.md)).

---

## Chapter 4: Question A — The Cross-Language Problem
### 1. Cross-Language Performance Gap
- **Insert [Table B](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_B_crosslanguage_gap.md) & [Fig 01](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig01_cross_language_performance_gap.png):**
  * Report Within-Language Baseline UAR = **0.7666** vs. Cross-Language Baseline UAR = **0.6021**.
  * Quantify the statistically verified gap: $\Delta = -0.1645$ (-21.46% drop; paired $t = -18.41, p = 1.2 	imes 10^{-23}, q < 10^{-20}$).
### 2. SSL Representation Separability
- **Insert [Table C](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_C_ssl_separability.md) & [Fig 02](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig02_ssl_layerwise_language_separability.png):**
  * Document that frozen SSL representations encode language identity with up to **100% UAR** across middle/deep layers.
### 3. Acoustic & Two-Way ANOVA Analysis
- **Insert [Table D](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_D_acoustic_language_differences.md) & [Fig 03](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig03_acoustic_feature_language_effect_sizes.png):** 38 of 43 acoustic features show FDR-significant language differences ($q < 0.05$).
- **Insert [Table E](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_E_dysarthria_biomarkers_interactions.md) & [Fig 04](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig04_language_pathology_interaction_heatmap.png):** Two-way ANOVA proving that language variance dominates over clinical pathology variance (Language $\eta_p^2$ up to 0.411 vs Diagnosis $\eta_p^2$ up to 0.093), with 7 significant interaction markers.

---

## Chapter 5: Question B — Voice Conversion Domain Shift
### 1. Feature-Level Domain Adaptation
- **Insert [Table F](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_F_feature_domain_movement.md), [Fig 05](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig05_vc_domain_shift_distances_spanish_to_german.png), and [Fig 06](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig06_vc_domain_shift_distances_german_to_spanish.png):**
  * Spanish $	o$ German: 86.0% of features shift toward the German domain ($p < 0.05$).
  * German $	o$ Spanish: 81.4% of features shift toward the Spanish domain ($p < 0.05$).
### 2. Frozen Language Classifier Migration
- **Insert [Table G](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_G_multivariate_domain_shift.md):**
  * Report that German recordings converted toward Spanish drop in German probability from 0.9969 to 0.7820 ($p = 4.5 	imes 10^{-17}$), with 10.23% crossing the decision boundary.
### 3. Macro-Temporal Invariance
- **Insert [Fig 07](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig07_multivariate_domain_distance_summary.png):**
  * Emphasize the crucial finding: sentence duration, pause ratios, and syllable rate remain unshifted ($\Delta 	ext{movement} = 0.000\sigma$).

---

## Chapter 6: Question C — Pathology Contrast Retention
### 1. Dysarthric Biomarker Preservation
- **Insert [Table H](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_H_pathology_preservation.md) & [Fig 08](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig08_pathology_effect_size_preservation.png):**
  * Report 69.2% overall sign preservation and retention of 8 primary FDR biomarkers (F0 range $R=0.812$, MFCC 5 std $R=1.061$, jitter proxy $R=0.870$).
### 2. Absence of Artificial Normalization
- **Cite the Welch Centroid Equivalence Test:** Converted PD shift ($2.316\sigma$) vs Converted HC shift ($2.593\sigma$) difference = $-0.277\sigma$ ($t = -0.4000, p = 0.6896$).
- **Scientific Conclusion:** Voice conversion does **not** artificially normalize or "cure" Parkinsonian speech.

---

## Chapter 7: Voice Conversion Quality & Biometrics
### 1. Speaker Verification
- **Insert [Table I](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_I_speaker_verification.md) & [Fig 09](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig09_speaker_similarity_distribution.png):**
  * ECAPA-TDNN mean same-speaker cosine similarity = **0.9648** (100% $\ge 0.70$) vs. negative pairs = **0.1315** ($p < 10^{-300}$).
### 2. Spectral Distortion
- **Insert [Table J](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_J_spectral_distortion.md) & [Fig 10](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig10_spectral_distortion_mcd_distribution.png):**
  * Mean MCD-DTW = **5.4381 dB**, LSD = 17.23 dB, SC = 0.729. Distortion is equitable between PD (5.50 dB) and HC (5.37 dB; $p = 0.3933$).
### 3. Linguistic Content Preservation
- **Insert [Table K](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_K_asr_consistency.md) & [Fig 11](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig11_asr_consistency_proxy_distribution.png):**
  * Whisper-base mean $	ext{WER}_{	ext{proxy}} = 0.2178$, $	ext{WIP} = 0.6924$.

---

## Chapter 8: Question D — Master Classification Synthesis
### 1. The Definitive 3-Way Classification Synthesis
- **Insert [Table L](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/table_L_master_question_D_synthesis.md) & [Fig 12](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/11_figures/fig12_master_question_d_classification_comparison.png):**
  * **D.1 Within-Language Baseline:** UAR = **0.7666**
  * **D.2 Cross-Language Baseline (Unconverted):** UAR = **0.6021**
  * **D.3 Cross-Language with Stage 5C Conversion:** UAR = **0.5663** ($\Delta = -0.0358$, paired $t = -5.0358, p = 7.3 	imes 10^{-6}, q < 10^{-5}$)
### 2. Disentangling the All-Scenario Reporting Pitfall
- **Critical Clarification:**
  * In "All Scenarios Combined", aggregate UAR rises from 0.6979 to 0.7330 ($\Delta = +0.0352$).
  * **This increase is entirely driven by within-language self-resynthesis (0.7666 $	o$ 0.8557)** due to vocoder compression.
  * In **pure cross-language testing**, converted speech shows a statistically significant decrease ($p < 0.0001$).
  * **The crosslingual gap is not closed by Stage 5C voice conversion.**

---

## Chapter 9: Discussion, Mechanistic Analysis & Limitations
- **Insert Section 9.1 (Mechanistic Explanation of Failure):**
  1. *Temporal and Phonetic Invariance:* Spanish syllable-timed rhythm vs. German stress-timed rhythm cannot be altered by static spectral conversion.
  2. *Vocoder Sensitivity in SSL Models:* Pretrained SSL transformers are sensitive to vocoder high-frequency spectral flatness inflation.
- **Insert Section 9.2 (Methodological Limitations):** Lack of forced phoneme alignments in read-text corpora, two-language scope, and proxy WER evaluation.

---

## Chapter 10: Conclusions & Future Research Directions
- **Insert [Research Question Answer Matrix](file:///c:/pd-speech-crosslingual/thesis_final_evaluation/12_tables/research_question_answer_matrix.md):** Complete summary answering Questions A–D.
- **Final Thesis Takeaway:** Voice conversion is technically robust and preserves speaker identity and dysarthric markers, but cross-lingual diagnostic generalization requires sequence-level prosodic, rhythmic, and durational transformation.
