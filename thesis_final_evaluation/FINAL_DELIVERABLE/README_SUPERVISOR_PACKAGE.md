# Master's Thesis Evaluation & Results Package: Executive Guide for Supervisor

**Master's Thesis:** Voice Conversion for Crosslingual Detection of Parkinson's Disease Using Speech Signals  
**Student:** Prosenjit Chowdhury  
**Supervisor:** Dr.-Ing. Tomás Arias-Vergara  
**Institution:** Pattern Recognition Lab (PRL), Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)  
**Deliverable Package Path:** `thesis_final_evaluation/FINAL_DELIVERABLE/`  

---

## 1. Executive Summary for Dr.-Ing. Tomás Arias-Vergara
This deliverable package contains the complete, rigorous, and fully validated scientific evaluation of the Master's Thesis. All evaluations were conducted on the unified dataset of **276 Spanish (PC-GITA) and German (G-Read) original recordings** and their **276 Stage 5C embedding-conditioned voice conversions** (552 waveforms total), spanning 43 acoustic features and 120 SSL classification pipelines across Wav2Vec 2.0, XLSR-53, and WavLM Base+.

### Summary of Core Scientific Answers:
1. **Question A (The Cross-Language Problem): CONFIRMED ROBUST**
   - Cross-language Parkinson's classification suffers a massive degradation (Within UAR **0.7666** vs. Cross-Language UAR **0.6021**, drop = **-21.46%**, $p = 1.2 	imes 10^{-23}$).
   - Frozen SSL embeddings encode language identity with up to **100% UAR**. 38 of 43 acoustic features differ significantly ($q < 0.05$), with language variance heavily dominating over pathology variance.
2. **Question B (Voice Conversion Domain Shift): CONFIRMED ROBUST (Partial / Spectral Shift)**
   - Stage 5C conversion successfully shifts spectral envelopes toward target language distributions across **86.0% (Spanish $	o$ German)** and **81.4% (German $	o$ Spanish)** of acoustic features ($p < 0.05$).
   - A frozen language classifier confirms source identity attenuation (German probability drops from 0.9969 to 0.7820, $p < 10^{-16}$, with 10.23% crossing the decision boundary). However, **macro-temporal duration and pause features remain unshifted**.
3. **Question C (Dysarthric Pathology Retention): CONFIRMED ROBUST**
   - **69.2% of acoustic features preserve directional pathology contrast**, with 8 of 14 primary dysarthric biomarkers retained (e.g. F0 range $R=0.812$, MFCC 5 std $R=1.061$, jitter proxy $R=0.870$).
   - Converted PD speech does **not** collapse toward healthy control centroids ($p = 0.6896$), proving that voice conversion does not artificially "cure" or normalize dysarthria.
4. **Question D (Cross-Language Classification Efficacy): REFUTED (Negative Finding Confirmed)**
   - **Stage 5C voice conversion does NOT improve cross-language Parkinson's disease classification.**
   - Across 48 pure cross-language models, unconverted baseline UAR is **0.6021**, which drops to **0.5663** following conversion ($\Delta = -0.0358$, paired $t = -5.0358, p = 7.3 	imes 10^{-6}, q < 10^{-5}$).
   - The apparent increase in all-scenario aggregate UAR (0.6979 $	o$ 0.7330) is an artifact of **within-language vocoder self-resynthesis (0.7666 $	o$ 0.8557, $p < 10^{-13}$)**.
   - **Conclusion:** Spectral envelope conversion alone cannot overcome the cross-language gap; sequence-level prosodic and rhythm transformation is required.

---

## 2. Directory Structure of this Deliverable Package
```
FINAL_DELIVERABLE/
├── README_SUPERVISOR_PACKAGE.md                        <-- This guide
├── 01_executive_reports/
│   ├── tomas_core_thesis_evaluation_report.md          <-- Full thesis evaluation report (Sections 1 to 9)
│   ├── research_question_answer_matrix.md / .csv       <-- Executive answer matrix for Questions A-D
│   ├── thesis_draft_revision_instructions.md           <-- Overview revision guide
│   └── thesis_chapter_by_chapter_revision_map.md       <-- Detailed chapter-by-chapter revision blueprint
├── 02_master_tables/
│   ├── table_A_inventory_summary.md / .csv             <-- Dataset & audio demographics
│   ├── table_B_crosslanguage_gap.md / .csv             <-- Baseline cross-language gap
│   ├── table_C_ssl_separability.md / .csv              <-- SSL layerwise language separability
│   ├── table_D_acoustic_language_differences.md / .csv <-- Top 20 acoustic language differences
│   ├── table_E_dysarthria_biomarkers_interactions.md   <-- 2-Way ANOVA interactions
│   ├── table_F_feature_domain_movement.md / .csv       <-- Top feature-level domain shift
│   ├── table_G_multivariate_domain_shift.md / .csv     <-- Frozen language classifier shift
│   ├── table_H_pathology_preservation.md / .csv        <-- Dysarthria effect size preservation
│   ├── table_I_speaker_verification.md / .csv          <-- ECAPA-TDNN speaker similarity
│   ├── table_J_spectral_distortion.md / .csv           <-- Mel-Cepstral Distortion (MCD-DTW)
│   ├── table_K_asr_consistency.md / .csv               <-- Whisper-base ASR consistency
│   ├── table_L_master_question_D_synthesis.md / .csv   <-- Master Question D synthesis
│   └── table_manifest.md                               <-- Full data traceability matrix
├── 03_publication_figures/
│   ├── fig01 to fig12 (.png [300 DPI] and .pdf [vector]) <-- 12 publication-quality thesis figures
│   └── figure_manifest.md                              <-- Figure manifest and interpretation guide
└── 04_statistical_and_reproducibility_package/
    ├── statistical_master_plan.md                      <-- Preregistered hypothesis testing plan
    ├── master_hypothesis_tests.csv                     <-- 271 statistical tests with FDR q-values
    ├── family_wise_corrections.csv                     <-- FDR correction summary across 5 families
    ├── confirmed_findings_summary.md                   <-- Summary of confirmed findings
    ├── reproducibility_summary.md                      <-- Environment, software, seeds, pipeline
    ├── input_output_manifest.csv                       <-- Full manifest of generated artifacts
    └── final_integrity_report.md                       <-- Final integrity verification certificate
```

---
> [!IMPORTANT]
> **Data Privacy & Scientific Integrity Notice:** This deliverable contains only aggregated, public-safe statistical results, publication figures, and executive markdown reports. No private raw audio, converted audio, speaker subject names, or proprietary paths are included.
