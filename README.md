# Voice Conversion for Crosslingual Detection of Parkinson’s Disease Using Speech Signals

### Master’s Thesis Project | Pattern Recognition Lab (PRL) | Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

**Author:** Prosenjit Chowdhury  
**Program:** M.Sc. Artificial Intelligence  
**Registration No.:** 23361276  
**Supervisor:** Dr.-Ing. Tomás Arias-Vergara  
**Professor / Thesis Advisor:** Prof. Dr.-Ing. habil. Andreas Maier  

I am an M.Sc. Artificial Intelligence student at FAU Erlangen-Nürnberg with hands-on experience in Speech AI, Machine Learning, Biomedical AI, Data Analysis, Enterprise Systems, Process Automation, and Digital Transformation.

Alongside my thesis research, I work as a Working Student at SAP SE, where I have contributed across SAP LeanIX and SAP Signavio Content Marketing, Industry Content Coordination for Professional Services and Engineering, Construction & Operations, and ERP PCX / Enterprise Systems and Process Automation.
---

## Project Status

> **Research implementation, full evaluation, and scientific audit are complete.**  
> The project now contains the complete experimental path from baseline classification and HiFi-GAN reconstruction to an earlier acoustic-domain conversion prototype and the final **true Spanish↔German cross-language speech-generation pipeline**.

| Component | Status | Final role |
|---|---|---|
| Original-speech PD/HC baseline | ✅ Complete | Quantifies the Spanish–German cross-language gap |
| HiFi-GAN reconstruction validation | ✅ Complete | Verifies that neural waveform reconstruction does not substantially collapse diagnostic behavior |
| Earlier acoustic-domain conversion prototype | ✅ Complete | Historical experiment; showed acoustic target-domain movement but no cross-language PD benefit |
| True cross-language pilot | ✅ Complete | Verified actual target-language generation |
| Pilot refinement | ✅ Complete | Improved language correctness and content stability |
| Spanish→German final refinement | ✅ Complete | Frozen final Spanish→German synthesis configuration |
| Independent 80-file validation | ✅ Complete | Primary out-of-sample validation |
| Full 276-file evaluation | ✅ Complete | Full-cohort extension / sensitivity analysis |
| Development-excluded analysis | ✅ Complete | 264-file sensitivity analysis |
| Final scientific audit | ✅ Complete | Reconciled metrics, statistics, and final interpretation |
| Thesis reporting | ✅ Finalization | Final thesis and supervisor-ready reports prepared |

---

## Project Overview

This repository contains the implementation and evaluation work for the Master’s thesis:

> **Voice Conversion for Crosslingual Detection of Parkinson’s Disease Using Speech Signals**

The thesis investigates a central problem in clinical speech AI:

> A Parkinson’s disease classifier trained on one language may perform substantially worse when evaluated on another language because speech representations contain language- and corpus-related information in addition to pathology-related information.

The project asks whether speech can be converted across **Spanish and German** while preserving the speaker and clinically relevant Parkinson’s disease (PD) information, and whether doing so improves cross-language PD-versus-healthy-control (HC) classification.

The final project distinguishes three different questions that must not be confused:

1. **Can the spoken language actually be changed?**
2. **Can speaker identity and pathology-related speech information survive the conversion?**
3. **Does the converted speech improve downstream cross-language PD classification?**

The final answer is:

> **True cross-language generation is technically successful, speaker identity is substantially preserved, and a meaningful proportion of pathology-related information remains; however, voice conversion alone does not reliably solve the cross-language PD-classification problem.**

---

## Important Conceptual Correction

The thesis contains an earlier experimental branch that performed **embedding-conditioned acoustic-domain conversion**. That method shifted acoustic characteristics toward the opposite language domain, but **did not translate the spoken words**.

That experiment was scientifically useful, but it is **not the final definition of true cross-language voice conversion used in this thesis**.

### Earlier acoustic-domain prototype

```text
source speech
   ↓
SSL/domain representation
   ↓
acoustic-domain shift
   ↓
HiFi-GAN reconstruction
   ↓
same linguistic content, acoustically shifted speech
```

### Final true cross-language pipeline

```text
source speech
   ↓
known source read-text / transcript representation
   ↓
frozen target-language translation
   ↓
speaker-conditioned multilingual speech synthesis
   ↓
actual target-language speech
```

Because the task uses fixed read-text passages, the final implementation uses **known canonical source text and frozen target-language translations** rather than relying on speaker-specific ASR text as the synthesis target. ASR is then used independently for **evaluation** of generated content.

For example:

```text
German source speaker
        ↓
German read-text content
        ↓
Spanish translation
        ↓
XTTS-v2 conditioned on the German speaker
        ↓
Spanish speech in a voice resembling the original speaker
```

The reverse direction is evaluated analogously for Spanish→German.

---

# Dataset

The full read-text dataset contains **276 recordings**.

| Language / Corpus | HC | PD | Total |
|---|---:|---:|---:|
| Spanish / PC-GITA | 50 | 50 | 100 |
| German / G-Read | 88 | 88 | 176 |
| **Total** | **138** | **138** | **276** |

Only the **read-text** task is used for the principal cross-language experiments because it is available in both language cohorts and provides a controlled comparison.

### Core labels

| Label | Meaning |
|---|---|
| PD | Parkinson’s disease |
| HC | Healthy control |
| ES / SP | Spanish |
| DE | German |
| UAR | Unweighted Average Recall |

---

# Research Questions

The final thesis can be summarized through the following questions:

| Question | Final answer |
|---|---|
| Is there a measurable Spanish–German cross-language PD-detection problem? | **Yes.** |
| Can neural reconstruction preserve downstream PD/HC behavior? | **Largely yes.** |
| Can actual Spanish↔German linguistic speech generation be achieved? | **Yes, very reliably.** |
| Is target-language content preserved? | **Partially and asymmetrically.** |
| Is speaker identity retained? | **Substantially.** |
| Are pathology-related acoustic characteristics retained? | **A majority, but not all.** |
| Does synthesis normalize PD speech toward HC? | **Partial acoustic normalization appears in the full cohort.** |
| Does true cross-language conversion improve PD detection? | **Not consistently.** |
| Is the diagnostic effect symmetric across directions? | **No.** |

---

# 1. Original-Speech Baseline

The baseline evaluates original Spanish and German speech without conversion.

### Models

- **XLSR-53**
- **Wav2Vec 2.0**
- **WavLM**

### Representation layers

- 0
- 4
- 8
- 11

### Classifiers

- Linear SVM
- Logistic Regression

### Evaluation scenarios

| Scenario | Type |
|---|---|
| Spanish→Spanish | Within-language |
| German→German | Within-language |
| Spanish→German | Cross-language |
| German→Spanish | Cross-language |
| Spanish + German → combined | Combined-language |

### Best scenario-specific baseline results

| Scenario | Best model | Layer | Classifier | UAR |
|---|---|---:|---|---:|
| Spanish→Spanish | XLSR | 4 | Linear SVM | **0.8400** |
| German→German | WavLM | 8 | Logistic Regression | **0.8182** |
| Spanish→German | WavLM | 11 | Logistic Regression | **0.7273** |
| German→Spanish | WavLM | 0 | Logistic Regression | **0.6900** |
| Combined | Wav2Vec2 | 8 | Logistic Regression | **0.8080** |

Across matched configurations:

- **Mean within-language UAR:** 0.7666
- **Mean cross-language UAR:** 0.6021
- **Absolute gap:** 0.1645
- **Relative cross-language degradation:** **21.46%**

This establishes the central problem: the models retain strong language/domain information, and PD/HC performance drops when the train and test languages differ.

Additional analysis found that **38 of 43 acoustic features** differed significantly between Spanish and German after multiple-testing correction.

---

# 2. HiFi-GAN Reconstruction Validation

Before attempting voice conversion, HiFi-GAN Universal V1 was evaluated as a waveform reconstruction component.

The purpose was **not** to change language. The goal was to determine whether neural resynthesis itself destroys the information required for downstream PD/HC classification.

### Controlled scaling

| Evaluation | Files | Result |
|---|---:|---|
| Reconstruction pilot | 12 | Technically successful |
| Reconstruction validation | 80 | Stable |
| Full reconstruction | 276 | 276/276 generated |

### Full 276-file reconstruction result

| Metric | Original | Reconstructed |
|---|---:|---:|
| Mean UAR across 120 matched configurations | 0.7016 | 0.7046 |
| Mean UAR change |  | **+0.0030** |
| Mean absolute UAR difference |  | **0.0264** |

**Interpretation:** HiFi-GAN reconstruction did not produce a substantial aggregate collapse in PD/HC classification behavior, supporting its use as a waveform synthesis component in the earlier prototype work.

---

# 3. Earlier Acoustic-Domain Conversion Prototype — Historical Experiment

An earlier branch developed an embedding-conditioned method that moved speech acoustics toward the opposite language/domain condition.

This branch remains part of the thesis because it provides an important negative scientific result, but it must not be confused with true linguistic conversion.

### Method concept

```text
source log-mel representation
        +
embedding-derived source→target domain shift
        ↓
modified log-mel representation
        ↓
HiFi-GAN
        ↓
acoustically shifted waveform
```

The selected configuration used:

- XLSR layer 11
- conversion strength α = 1.0

### Full 276-file result

| Metric | Original | Converted | Delta |
|---|---:|---:|---:|
| All-scenario mean UAR | 0.6979 | 0.7330 | +0.0352 |
| Pure cross-language mean UAR | 0.6021 | 0.5663 | **-0.0358** |

Additional analysis showed:

- substantial acoustic target-domain movement,
- strong speaker preservation,
- substantial pathology preservation,
- but **no improvement in pure cross-language PD classification**.

This experiment established an important distinction:

> **Moving acoustic features toward another language domain is not the same as genuinely changing the linguistic language, and acoustic-domain movement alone does not guarantee improved clinical transfer.**

---

# 4. True Cross-Language Speech Generation

The final branch implements actual linguistic Spanish↔German speech generation.

## Model

**XTTS-v2** is used for multilingual, speaker-conditioned synthesis.

The target text is the frozen translation of the known read-text passage, while the speaker reference is extracted from the original source recording.

### Evaluation dimensions

The generated audio is evaluated using:

1. Technical audio quality
2. Target-language identification
3. Content fidelity
   - WER
   - CER
   - WIP
4. Speaker preservation
   - ECAPA-TDNN cosine similarity
   - different-speaker controls
5. Pathology-related acoustic feature preservation
6. PD→HC normalization analysis
7. Frozen downstream PD/HC classification
8. Bootstrap statistics and Benjamini-Hochberg FDR correction

---

# 5. 12-File True Cross-Language Pilot and Refinement

The first pilot used:

- 3 Spanish HC
- 3 Spanish PD
- 3 German HC
- 3 German PD

The initial 12-file pipeline demonstrated that true target-language generation was feasible, but content stability required refinement.

### Refinement goals

Refinement focused on:

- sentence / semantic segmentation,
- cleaner deterministic speaker references,
- pause handling,
- synthesis temperature,
- repetition control,
- target-language stability,
- speaker preservation.

The refined pilot achieved:

- **100% target-language identification**
- zero clipping
- improved stability

---

# 6. Final Spanish→German Refinement

Spanish→German was the more difficult content-generation direction and therefore received one additional controlled refinement before the independent validation.

The final frozen configuration used:

- XTTS-v2
- semantic 4-chunk target segmentation
- deterministic ~15 s clean voiced speaker reference
- temperature = 0.65
- repetition penalty = 3.5
- top-p = 0.85
- speed = 1.00
- peak normalization = 0.95

### Final six-speaker development result

| Metric | Result |
|---|---:|
| German LID | 6/6 = 100% |
| Mean target P(German) | 0.9637 |
| Mean WER | 0.5109 |
| Mean CER | 0.4599 |
| Mean WIP | 0.5579 |
| Mean speaker cosine similarity | 0.6245 |
| Clipping | 0 |
| Exploratory pathology direction preservation | 15/17 = 88.2% |

This configuration was then **frozen** before the larger validation.

---

# 7. Independent 80-File True Cross-Language Validation

The independent validation cohort contained:

- 20 Spanish HC
- 20 Spanish PD
- 20 German HC
- 20 German PD

**Total: 80**

All 12 pilot-development subjects were excluded.

No per-speaker parameter tuning was performed.

> This 80-file experiment is the **primary independent validation** for the final true cross-language method.

## Technical and linguistic results

| Metric | Result |
|---|---:|
| Synthesis success | 80/80 |
| Audio QC pass | 96.2% |
| Clipping | 0% |
| Target LID | **100%** |
| ES→DE LID | 40/40 |
| DE→ES LID | 40/40 |
| Overall mean WER | 0.4739 |
| Overall median WER | 0.2500 |
| Overall mean CER | 0.3593 |
| Overall mean WIP | 0.6537 |

### Content fidelity by direction

| Direction | Mean WER | Mean WIP |
|---|---:|---:|
| Spanish→German | 0.8130 | 0.4862 |
| German→Spanish | 0.1347 | 0.8212 |

This shows pronounced linguistic asymmetry: German→Spanish had substantially better content fidelity than Spanish→German.

## Speaker preservation

| Measure | Value |
|---|---:|
| Matched source/generated mean cosine | 0.5654 |
| Matched median | 0.5698 |
| Different-speaker control mean | 0.1080 |
| Different-speaker control median | 0.0977 |
| Discrimination margin | **+0.4573** |

The different-speaker control used 6,320 strictly mismatched pairs.

## Pathology preservation

- **68.2% (15/22)** evaluated pathology-effect directions preserved their sign.
- This is a **majority**, but it is **below the predefined 70% engineering target**.

### Normalization analysis

- Original PD→HC centroid distance: 4.708
- Generated PD→HC centroid distance: 4.715
- Wilcoxon W = 181.0
- p = 0.4125

Correct interpretation:

> **No statistically significant evidence of disproportionate PD-to-HC normalization was detected in the independent 80-file validation.**

This does **not** prove equivalence or absence of normalization.

## PD classification — 80-file validation

### Spanish→German  
German-trained classifier evaluated on Spanish speakers before and after generating German speech.

| Classifier | Original UAR | Generated UAR | Delta | p | FDR q |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.5000 | 0.6250 | **+0.1250** | 0.062 | 0.124 |
| Logistic Regression | 0.5000 | 0.6500 | **+0.1500** | 0.025 | 0.100 |

The Logistic Regression gain is **nominally significant before FDR correction**, but **not significant after FDR correction**.

### German→Spanish

| Classifier | Original UAR | Generated UAR | Delta |
|---|---:|---:|---:|
| Linear SVM | 0.5750 | 0.4250 | **-0.1500** |
| Logistic Regression | 0.5750 | 0.5250 | **-0.0500** |

**Interpretation:** the independent validation produced a promising Spanish→German gain but no German→Spanish benefit.

---

# 8. Full 276-File True Cross-Language Evaluation

The frozen system was then applied to all available read-text recordings.

### Generation

| Direction | Files |
|---|---:|
| Spanish→German | 100 |
| German→Spanish | 176 |
| **Total** | **276** |

### Technical validation

| Metric | Result |
|---|---:|
| Generated successfully | **276/276** |
| Failed | 0 |
| Audio QC pass | **100.0%** |
| Clipped files | 0 |

## Target-language correctness

| Direction | LID accuracy |
|---|---:|
| Spanish→German | **100.00%** |
| German→Spanish | **99.43%** |
| Overall | **99.64%** |

This confirms that the final system performs genuine linguistic cross-language generation at scale.

## Content fidelity

| Metric | Result |
|---|---:|
| ES→DE mean WER | 0.8065 |
| DE→ES mean WER | 0.4430 |
| Overall mean WER | 0.5747 |
| Median WER | 0.4396 |
| Mean CER | 0.4976 |
| Mean WIP | 0.4685 |

The directional content asymmetry remained present.

## Speaker preservation

| Measure | Value |
|---|---:|
| Matched mean cosine | 0.5610 |
| Matched median | 0.5581 |
| Different-speaker mean | 0.0963 |
| Different-speaker median | 0.0875 |
| Discrimination margin | **+0.4647** |

Speaker identity remains clearly more similar for matched original/generated pairs than for different-speaker controls.

---

# 9. Full-Cohort Pathology Analysis

The final scientific audit reconciled the pathology-preservation calculations.

## Authoritative preservation results

| Analysis | Direction preservation |
|---|---:|
| Spanish→German | 63.6% |
| German→Spanish | 72.7% |
| **Pooled overall** | **59.1%** |
| Development-excluded cohort | 63.6% |

The earlier reported overall value of **65.2%** was incorrect because it was an unweighted arithmetic mean of directional percentages rather than a pooled contrast calculation.

Therefore:

> **59.1% is the authoritative overall pathology-direction preservation value for the full cohort.**

A majority of pathology-related acoustic contrasts were retained, but preservation was incomplete.

---

# 10. PD→HC Acoustic Normalization — Final Corrected Result

The full-cohort normalization analysis produced a different result from the independent 80-file subset.

| Measure | Value |
|---|---:|
| Original PD→HC centroid distance | 4.3870 |
| Generated PD→HC centroid distance | 3.8429 |
| Paired change | -0.5441 |
| Wilcoxon W | 2236.0 |
| p-value | < 0.0001 |

Because generated PD speech became significantly closer to the HC centroid, the final correct interpretation is:

> **Generated PD speech showed partial acoustic normalization toward the HC centroid under the evaluated feature-space distance measure.**

This does **not** mean that all Parkinsonian information disappeared. Pathology-preservation and normalization analyses address different aspects of the generated speech.

---

# 11. Full-Cohort PD Classification

## Full 276 cohort

### Spanish→German

| Classifier | Original UAR | Generated UAR | Delta | p | FDR q |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.6000 | 0.6400 | +0.0400 | 0.248 | 0.992 |
| Logistic Regression | 0.6000 | 0.6500 | +0.0500 | 0.173 | 0.992 |

### German→Spanish

| Classifier | Original UAR | Generated UAR | Delta | p | FDR q |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.5909 | 0.5455 | -0.0455 | 0.885 | 1.000 |
| Logistic Regression | 0.5909 | 0.5057 | -0.0852 | 1.000 | 1.000 |

The Spanish→German improvement observed in the independent 80-file validation attenuated to small, non-significant gains in the full cohort.

German→Spanish again showed no diagnostic benefit.

---

# 12. Development-Excluded Sensitivity Analysis

The full 276 cohort includes the 12 pilot-development subjects, so a separate analysis excludes those subjects.

**N = 264**

| Direction | Linear SVM ΔUAR | Logistic Regression ΔUAR |
|---|---:|---:|
| Spanish→German | -0.0213 | -0.0426 |
| German→Spanish | -0.0118 | -0.0706 |

This analysis is important because it shows that the positive Spanish→German effect seen in the independent 80-file validation does **not** persist robustly across the broader development-excluded cohort.

---

# 13. Evidence Hierarchy

The final interpretation uses the following hierarchy:

1. **Independent 80-file validation**  
   Primary out-of-sample validation result.

2. **Full 276-file evaluation**  
   Full-cohort extension / descriptive sensitivity analysis.

3. **Development-excluded 264-file analysis**  
   Broad sensitivity analysis excluding pilot-development subjects.

The full 276 cohort must **not** be described as a second independent validation because it contains earlier development and validation subjects.

---

# 14. Final Scientific Finding

The final audited result is:

1. Genuine Spanish↔German linguistic speech generation was technically successful at full scale.
2. Target-language identification was extremely reliable.
3. Speaker identity remained substantially distinguishable from different-speaker controls.
4. A majority, but not all, pathology-related acoustic effect directions were retained.
5. The full cohort showed evidence of partial acoustic normalization of PD speech toward HC.
6. Spanish→German showed a promising improvement in the independent 80-file validation.
7. That improvement attenuated in the full cohort and disappeared in the development-excluded analysis.
8. German→Spanish showed no consistent diagnostic benefit.
9. Better linguistic fidelity did not necessarily produce better PD classification.

Therefore:

> **Successful language conversion, speaker preservation, and partial pathology preservation are not sufficient by themselves to guarantee reliable cross-language Parkinson’s disease classification.**

---

# Key Thesis Impact

The main contribution is not simply a positive classification score.

The thesis demonstrates that:

- the Spanish–German language mismatch is a real barrier for PD speech classification;
- true cross-language speech generation can reliably change the spoken language;
- source-speaker identity can survive the language change to a substantial degree;
- clinically relevant acoustic information can partially survive neural cross-language synthesis;
- synthesis can also partially normalize PD speech acoustically;
- high linguistic fidelity is not a reliable predictor of downstream diagnostic benefit;
- and voice conversion alone does not robustly eliminate the cross-language classification shift.

This motivates future approaches that combine generative speech conversion with **classifier domain adaptation, recalibration, or pathology-aware representation learning**.

---

# Authoritative Final Metrics

| Category | Final verified result |
|---|---|
| Dataset | 276 read-text recordings |
| Class balance | 138 HC / 138 PD |
| Full synthesis | 276/276 successful |
| Audio QC | 100% pass, 0 clipping |
| Target LID | 99.64% overall |
| ES→DE LID | 100.0% |
| DE→ES LID | 99.43% |
| Overall WER | 0.5747 |
| Overall CER | 0.4976 |
| Overall WIP | 0.4685 |
| Matched speaker cosine | 0.5610 |
| Different-speaker mean | 0.0963 |
| Speaker discrimination margin | +0.4647 |
| Full-cohort pathology direction preservation | **59.1%** |
| Full-cohort normalization | **Partial normalization; p < 0.0001** |
| 80-file ES→DE LR ΔUAR | +0.1500, p=0.025, q=0.100 |
| Full 276 ES→DE SVM ΔUAR | +0.0400, q=0.992 |
| Full 276 DE→ES SVM ΔUAR | -0.0455, q=1.000 |
| Dev-excluded ES→DE SVM ΔUAR | -0.0213 |
| Dev-excluded DE→ES SVM ΔUAR | -0.0118 |

---

# Important Repository Locations

## Original baseline and reconstruction

```text
outputs/
voice_conversion/
```

## Earlier acoustic-domain prototype

```text
voice_conversion/stage5_embedding_conditioned_vc/
```

This branch is preserved as historical experimental evidence.

## Final true cross-language work

```text
true_cross_language_vc/
```

### True cross-language pilot

```text
true_cross_language_vc/17_reports/12_file_true_crosslanguage_pilot_report.md
```

### Refined 12-file pilot

```text
true_cross_language_vc/19_pilot_refinement/12_report/
```

### Final Spanish→German refinement

```text
true_cross_language_vc/20_spanish_to_german_final_refinement/
```

Key report:

```text
true_cross_language_vc/20_spanish_to_german_final_refinement/
19_reports/final_spanish_to_german_refinement_report.md
```

### Independent 80-file validation

```text
true_cross_language_vc/21_validation_80/
```

Key reports:

```text
true_cross_language_vc/21_validation_80/
17_reports/80_file_true_crosslanguage_validation_report.md

true_cross_language_vc/21_validation_80/
17_reports/scientific_interpretation_corrections.md
```

### Full 276-file evaluation

```text
true_cross_language_vc/22_full_276_evaluation/
```

Key report:

```text
true_cross_language_vc/22_full_276_evaluation/
19_reports/full_276_true_crosslanguage_evaluation_report.md
```

### Final scientific audit — highest-authority interpretation

```text
true_cross_language_vc/22_full_276_evaluation/
21_final_scientific_audit/
```

Use these files for thesis-level interpretation:

```text
08_corrected_reports/full_276_true_crosslanguage_evaluation_VERIFIED.md
08_corrected_reports/scientific_errata_and_corrections.md
09_thesis_summary/final_thesis_results_summary.md
09_thesis_summary/final_research_question_matrix.csv
09_thesis_summary/supervisor_update_short.md
```

---

# Full 276 Evaluation Outputs

```text
true_cross_language_vc/22_full_276_evaluation/
├── 01_inventory/
├── 02_cohort_tags/
├── 03_transcripts/
├── 04_translation/
├── 07_audio_qc/
├── 08_language_id/
├── 09_content_accuracy/
├── 10_speaker_preservation/
├── 11_pathology_features/
├── 12_pathology_preservation/
├── 13_pd_classification/
├── 14_statistics/
├── 15_subgroup_analysis/
├── 16_figures/
├── 17_tables/
├── 18_listening_examples/
├── 19_reports/
├── 20_reproducibility/
├── 21_final_scientific_audit/
├── logs/
└── scripts/
```

The full evaluation includes:

- 13 figures
- 18 formal tables
- supervisor listening examples
- execution manifests
- checksums
- environment information
- frozen configuration documentation
- statistical analysis outputs
- final scientific audit artifacts

---

# Full 276 Pipeline Scripts

The full evaluation is organized through the following scripts:

```text
true_cross_language_vc/22_full_276_evaluation/scripts/
├── 00_preflight_and_inventory.py
├── 01_cohort_provenance_tagging.py
├── 02_prepare_texts_and_references.py
├── 03_synthesize_276.py
├── 04_audio_qc_lid_asr_spk_276.py
├── 05_pathology_and_normalization_276.py
├── 06_pd_classification_and_statistics_276.py
├── 07_subgroup_and_error_analyses_276.py
├── 08_figures_tables_listening_repro_276.py
├── 09_compile_full_276_report.py
├── run_full_276_pipeline.py
└── run_final_audit.py
```

The final experiment is frozen. These scripts are retained for reproducibility, **not for further tuning**.

---

# Data Privacy and Repository Policy

This project uses sensitive medical speech data.

Raw audio, speaker-identifiable data, patient metadata, and private local paths must not be committed to a public repository.

## Keep private / local

- raw `.wav`, `.mp3`, `.flac`, `.m4a` files
- generated patient speech unless explicitly approved for sharing
- private metadata
- speaker-identifying filenames
- local dataset paths
- embedding tables tied to individual speakers
- model checkpoints if licensing or size makes them inappropriate for redistribution
- private clinical information

## Safe repository content

- source code
- configuration files
- anonymized aggregate results
- safe reports
- non-identifying figures
- non-sensitive tables
- reproducibility documentation
- environment manifests
- public model references

This repository is a research implementation and documentation repository, **not a public medical speech dataset**.

---

# Reproducibility

The project uses a staged, frozen-evaluation philosophy.

Important reproducibility artifacts include:

- environment snapshots
- execution manifests
- frozen synthesis configuration
- statistical analysis plan
- checksums
- failure logs
- cohort provenance
- corrected final reports

The final full-cohort reproducibility package is located at:

```text
true_cross_language_vc/22_full_276_evaluation/20_reproducibility/
```

The final audit reproducibility material is located at:

```text
true_cross_language_vc/22_full_276_evaluation/
21_final_scientific_audit/10_reproducibility/
```

---

# Scientific Reporting Rules Used in This Repository

To avoid over-interpretation:

- `p < 0.05` with `q >= 0.05` is described as **nominally significant before FDR correction, but not significant after correction**.
- `p > 0.05` is **not** interpreted as proof of equivalence.
- The 80-file validation remains the **primary independent validation**.
- The 276-file cohort is a **full-cohort extension**, not a second independent validation.
- The 264-file analysis is a **development-excluded sensitivity analysis**.
- Acoustic normalization, pathology preservation, speaker preservation, content fidelity, and PD classification are reported as **different outcomes**.
- No causal explanation for directional asymmetry is claimed without direct evidence.

---

# Limitations

| Limitation | Interpretation |
|---|---|
| Read-text only | Generalization to spontaneous speech remains unknown |
| Spanish and German only | Findings do not establish universal multilingual generalization |
| Directional asymmetry | ES→DE and DE→ES behave differently |
| Content fidelity asymmetry | ES→DE remains more difficult than DE→ES |
| Partial pathology preservation | Not all clinical acoustic contrasts survive synthesis |
| Partial acoustic normalization | Generated PD speech may move toward HC in feature space |
| Classification benefit not robust | Positive ES→DE validation gains attenuate in broader cohorts |
| Private medical data | Full public reproducibility is constrained by data access |
| Clinical scope | The system is a research experiment, not a medical diagnostic device |

---

# Future Work

The final results motivate several directions:

- explicit classifier domain adaptation,
- classifier recalibration for generated speech,
- pathology-aware speech synthesis objectives,
- pathology-preserving representation learning,
- spontaneous-speech and monologue validation,
- additional languages and cohorts,
- stronger semantic/content constraints for Spanish→German synthesis,
- clinical validation on independent external datasets.

The main lesson is that the generator and the diagnostic classifier should not be treated as independent components if the end goal is cross-language clinical generalization.

---

# Skills Demonstrated

| Category | Evidence |
|---|---|
| Speech AI | XLSR, Wav2Vec2, WavLM, HiFi-GAN, XTTS-v2 |
| Machine Learning | Linear SVM, Logistic Regression, cross-validation, UAR |
| Crosslingual AI | Spanish↔German domain shift and target-language generation |
| Biomedical AI | PD-versus-HC speech analysis |
| Audio Analysis | LID, ASR, WER, CER, WIP, spectral/acoustic features |
| Speaker Analysis | ECAPA-TDNN speaker embeddings and control-pair evaluation |
| Statistical Analysis | bootstrap inference, Wilcoxon testing, BH-FDR correction |
| Research Engineering | frozen configs, staged validation, reproducibility manifests |
| Python | modular experiment pipelines and automated evaluation |
| Scientific Reporting | error audits, corrected interpretation, figures and formal tables |
| Data Privacy | private clinical data handling and GitHub exclusion policy |

---

# Professional Context

**Prosenjit Chowdhury**  
M.Sc. Artificial Intelligence  
Friedrich-Alexander-Universität Erlangen-Nürnberg  
Pattern Recognition Lab

GitHub: [github.com/prosenjit-chd](https://github.com/prosenjit-chd)

Research interests include Speech AI, Biomedical AI, Crosslingual AI, Machine Learning, Voice Conversion, Data Analysis, and applied AI systems.
My SAP experience includes:

| SAP Area | Experience Focus |
|---|---|
| SAP LeanIX and SAP Signavio Content Marketing | Digital content operations, publishing workflows, product communication, global content coordination |
| Professional Services and Engineering, Construction & Operations Industries | Industry content coordination, stakeholder communication, customer success stories, go-live stories |
| ERP PCX / Enterprise Systems and Process Automation | Data validation, metadata quality, enterprise system support, Excel automation, VBA, Power Automate |

This project demonstrates skills relevant to AI research, data analysis, business analysis, process automation, technical documentation, and cross-functional collaboration.

---

## 🧩 Skills Demonstrated

| Category | Evidence |
|---|---|
| Machine Learning | SVM, Logistic Regression, cross-validation, UAR, AUC |
| Deep Learning | XLSR, Wav2Vec2, WavLM, HiFi-GAN |
| Speech Processing | Audio preprocessing, sampling rates, mel spectrograms, vocoding |
| Biomedical AI | Parkinson’s Disease vs Healthy Control classification |
| Crosslingual AI | Spanish–German transfer, language/domain mismatch analysis |
| Research Engineering | staged experiments, reproducibility, logs, reports |
| Python Development | modular scripts, experiment automation, file validation |
| Data Privacy | local-only medical data handling, GitHub exclusion policy |
| Documentation | supervisor-ready reports, GitHub README, result summaries |
| Professional Communication | SAP stakeholder communication, content coordination, structured reporting |
---

# Final One-Sentence Summary

> **This thesis demonstrates that genuine Spanish–German cross-language speech generation can preserve substantial speaker and Parkinson’s-related information, but also shows that successful language conversion alone is insufficient to guarantee reliable cross-language Parkinson’s disease classification.**

---

## Disclaimer

This repository documents academic research. The models and results presented here are **not intended for clinical diagnosis or medical decision-making**.
