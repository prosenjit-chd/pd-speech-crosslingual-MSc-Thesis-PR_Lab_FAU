# Voice Conversion for Crosslingual Detection of Parkinson’s Disease Using Speech Signals

### Spanish–German Parkinson’s Disease Speech Classification using XLSR, Wav2Vec2, WavLM, HiFi-GAN, and Prototype Embedding-Conditioned Voice Conversion
### Master’s Thesis Project | Pattern Recognition Lab | FAU Erlangen-Nürnberg

---

## 👤 Author & Professional Profile

**Prosenjit Chowdhury**  
M.Sc. Artificial Intelligence, Friedrich-Alexander-Universität Erlangen-Nürnberg  
Master’s Thesis Researcher, Pattern Recognition Lab, FAU  
Working Student at SAP SE  
GitHub: [github.com/prosenjit-chd](https://github.com/prosenjit-chd)

I am an M.Sc. Artificial Intelligence student at FAU Erlangen-Nürnberg with hands-on experience in Speech AI, Machine Learning, Biomedical AI, Data Analysis, Enterprise Systems, Process Automation, and Digital Transformation.

Alongside my thesis research, I work as a Working Student at SAP SE, where I have contributed across SAP LeanIX and SAP Signavio Content Marketing, Industry Content Coordination for Professional Services and Engineering, Construction & Operations, and ERP PCX / Enterprise Systems and Process Automation.

This project demonstrates my ability to design, implement, evaluate, document, and communicate a complete applied AI research pipeline from baseline modeling to speech generation, voice conversion, and full-dataset experimental analysis.

| Area | Skills Demonstrated |
|---|---|
| AI / ML | Speech embeddings, classification, cross-validation, UAR/AUC evaluation |
| Speech AI | XLSR, Wav2Vec2, WavLM, HiFi-GAN, audio preprocessing |
| Biomedical AI | Parkinson’s Disease vs Healthy Control speech analysis |
| Research Engineering | Reproducible pipeline, staged experiments, structured reporting |
| Data Analysis | Result comparison, model/layer/scenario analysis |
| Software Engineering | Python pipeline, modular scripts, local privacy-safe execution |
| Enterprise Experience | SAP SE, content operations, process automation, stakeholder communication |

---

## 🚀 Current Project Status

| Phase | Status | Description |
|---|---|---|
| Project Proposal | ✅ Completed | Research direction defined with Spanish–German PD speech and voice conversion |
| Baseline Model Comparison | ✅ Completed | XLSR, Wav2Vec2, and WavLM evaluated across layers and scenarios |
| HiFi-GAN Reconstruction Pilot | ✅ Completed | 12-file reconstruction validation |
| HiFi-GAN Reconstruction Subset | ✅ Completed | 80-file reconstruction validation |
| HiFi-GAN Reconstruction Full Dataset | ✅ Completed | 276-file reconstruction validation |
| Stage 5A Voice Conversion Pilot | ✅ Completed | 12-file prototype embedding-conditioned conversion |
| Stage 5A-Refinement | ✅ Completed | 15-setting alpha/model grid search |
| Stage 5B Subset Evaluation | ✅ Completed | 80-file prototype conversion evaluation |
| Stage 5C Full Evaluation | ✅ Completed | 276-file full prototype conversion evaluation |
| Thesis Writing / Final Interpretation | 🔄 In Progress | Formal thesis documentation and result interpretation |

---

## 📌 Project Overview

This repository contains the implementation work for my Master’s thesis at the Pattern Recognition Lab, FAU Erlangen-Nürnberg:

**Voice Conversion for Crosslingual Detection of Parkinson’s Disease Using Speech Signals**

The project investigates whether AI-based speech representations and voice conversion can support Parkinson’s Disease detection across languages. The work focuses on Spanish and German read-text speech recordings and evaluates whether converted speech can reduce language/domain mismatch while preserving Parkinson’s Disease-related acoustic cues.

The project follows a staged research pipeline:

1. Build a strong baseline using original Spanish and German speech.
2. Compare XLSR, Wav2Vec2, and WavLM embeddings for PD vs HC classification.
3. Validate HiFi-GAN reconstruction to ensure generated speech preserves diagnostic information.
4. Develop and test a prototype embedding-conditioned conversion method.
5. Scale the conversion experiments from 12 files to 80 files and finally to the full 276-file dataset.
6. Compare original vs converted speech using UAR, accuracy, sensitivity, specificity, and AUC.

The final Stage 5C experiment confirms that the prototype conversion method is technically feasible on the full dataset and preserves most diagnostic information, although strong crosslingual improvement remains limited.

---

## 🎯 Main Research Question

Can prototype embedding-conditioned voice conversion reduce Spanish–German language/domain mismatch in Parkinson’s Disease speech classification while preserving disease-relevant acoustic information?

In simpler terms:

> If a classifier is trained in one language, can converted speech from another language become acoustically closer to the training domain without losing PD/HC diagnostic cues?

---

## 🗂️ Dataset Overview

The current experiments use Spanish and German read-text Parkinson’s Disease speech recordings.

| Group | Count |
|---|---:|
| Spanish HC | 50 |
| Spanish PD | 50 |
| German HC | 88 |
| German PD | 88 |
| **Total** | **276** |

| Label | Meaning |
|---|---|
| PD | Parkinson’s Disease patient |
| HC | Healthy Control |

Only read-text speech is used in the current full pipeline because this task is available in both Spanish and German datasets and supports controlled crosslingual comparison.

---

## 🔒 Data Privacy and Repository Policy

This project uses sensitive medical speech data. Raw speech recordings, patient metadata, speaker-identifiable information, and local dataset paths are **not included** in this repository.

The following must remain local and must not be uploaded to GitHub:

- raw audio files (`*.wav`, `*.mp3`, `*.flac`, `*.m4a`)
- private metadata
- patient-related information
- local dataset paths
- raw input folders
- large derived features if they expose dataset structure
- checkpoint files and local generated audio folders unless explicitly safe

This repository is intended to contain:

- source code
- configuration files
- documentation
- safe result summaries
- non-identifying plots
- non-sensitive tables
- reproducible experiment structure

---

# 🧪 Experimental Pipeline and Results

## Phase 1 — Project Proposal

The original research direction was to investigate voice conversion for multilingual Parkinson’s Disease detection. After supervisor feedback, the wording was updated from **Multilingual** to **Crosslingual**, because the main focus is language transfer between Spanish and German.

The thesis goal became:

> Investigate whether voice conversion can support crosslingual Parkinson’s Disease detection from speech signals by reducing Spanish–German language mismatch while preserving disease-relevant acoustic cues.

---

## Phase 2 — Baseline Model Comparison

The baseline evaluates original speech without conversion. It compares XLSR, Wav2Vec2, and WavLM embeddings across layers 0, 4, 8, and 11 using Linear SVM and Logistic Regression.

### Baseline Scenarios

| Scenario | Purpose |
|---|---|
| Spanish → Spanish | Within-language Spanish baseline |
| German → German | Within-language German baseline |
| Spanish → German | Crosslingual transfer from Spanish to German |
| German → Spanish | Crosslingual transfer from German to Spanish |
| Spanish + German → Spanish + German | Combined bilingual training/testing |

### Best Baseline Results

| Scenario | Best Model | Layer | Classifier | UAR |
|---|---|---:|---|---:|
| Spanish → Spanish | XLSR | 4 | Linear SVM | 0.8400 |
| German → German | WavLM | 8 | Logistic Regression | 0.8182 |
| Spanish → German | WavLM | 11 | Logistic Regression | 0.7273 |
| German → Spanish | WavLM | 0 | Logistic Regression | 0.6900 |
| Spanish + German → Spanish + German | Wav2Vec2 | 8 | Logistic Regression | 0.8080 |

### Baseline Interpretation

The baseline confirms that within-language classification is stronger than crosslingual transfer. This shows a clear Spanish–German language/domain mismatch and motivates the voice conversion stage.

---

## Phase 3 — HiFi-GAN Reconstruction / Vocoder Validation

Before testing true conversion, HiFi-GAN reconstruction was validated to check whether generated/reconstructed audio can preserve PD/HC diagnostic information.

This stage is not true Spanish↔German conversion. It is a vocoder validation stage.

### HiFi-GAN Setup

| Item | Value |
|---|---|
| Vocoder | HiFi-GAN |
| Repository | `jik876/hifi-gan` |
| Checkpoint | `voice_conversion/checkpoints/universal_v1/generator_v1` |
| Config | `voice_conversion/checkpoints/universal_v1/config.json` |
| Target sample rate | 22050 Hz |
| Output format | mono WAV |

### Reconstruction Stages

| Stage | Files | Purpose | Result |
|---|---:|---|---|
| Pilot reconstruction | 12 | Small technical validation | Successful |
| Controlled subset reconstruction | 80 | Balanced subset validation | Successful |
| Full reconstruction | 276 | Full-dataset reconstruction validation | Successful |

### 80-File Reconstruction Result

| Metric | Value |
|---|---:|
| Files generated | 80/80 |
| Average UAR original | 0.6467 |
| Average UAR reconstructed | 0.6538 |
| Average UAR delta | +0.0071 |
| Rows within ±0.05 UAR change | 96/120 |
| Rows within ±0.15 UAR change | 120/120 |

### 276-File Reconstruction Result

| Metric | Value |
|---|---:|
| Files generated | 276/276 |
| Average UAR original | 0.7016 |
| Average UAR reconstructed | 0.7047 |
| Average UAR delta | +0.0030 |
| Mean absolute UAR change | 0.0264 |
| Median absolute UAR change | 0.0158 |

### Reconstruction Interpretation

HiFi-GAN reconstruction did not collapse diagnostic performance. This means generated speech remained usable for downstream PD/HC classification and justified moving toward voice conversion.

---

## Phase 4 — Stage 5A: 12-File Prototype Embedding-Conditioned Conversion

Stage 5A introduced the first prototype embedding-conditioned conversion experiment.

### Stage 5A Setup

| Item | Value |
|---|---|
| Files | 12 |
| Composition | 3 Spanish HC, 3 Spanish PD, 3 German HC, 3 German PD |
| Conversion method | Prototype embedding-conditioned conversion |
| Conditioning feature | XLSR layer 11 |
| Alpha | 0.5 |
| Vocoder | HiFi-GAN universal_v1 |

### Stage 5A Output

| Output | Result |
|---|---:|
| Spanish → German-domain files | 6 |
| German → Spanish-domain files | 6 |
| Total converted files | 12 |
| Audio validation success | 12/12 |

### Stage 5A Interpretation

The first Stage 5A pilot showed that the conversion pipeline could generate technically valid converted audio. However, classification results were mixed. Therefore, a refinement experiment was needed before scaling to 80 files.

---

## Phase 5 — Stage 5A-Refinement: Parameter Search

Because the first 12-file conversion result was mixed, a refinement grid search was performed before scaling.

### Tested Settings

| Conditioning Model | Layer |
|---|---:|
| XLSR | 11 |
| WavLM | 8 |
| WavLM | 11 |

| Alpha Values |
|---|
| 0.1 |
| 0.25 |
| 0.5 |
| 0.75 |
| 1.0 |

Total:

| Item | Count |
|---|---:|
| Parameter settings | 15 |
| Files per setting | 12 |
| Converted files | 180 |

### Stage 5A-Refinement Result

| Setting | Mean UAR Delta | Positive Rows | Negative Rows | Audio Success |
|---|---:|---:|---:|---:|
| XLSR layer 11, alpha 1.0 | +0.1250 | 5 | 0 | 100% |
| XLSR layer 11, alpha 0.75 | +0.0972 | 5 | 0 | 100% |
| XLSR layer 11, alpha 0.5 | +0.0278 | 2 | 0 | 100% |
| WavLM layer 8, all alpha values | +0.0139 | 1 | 0 | 100% |
| WavLM layer 11, all alpha values | +0.0139 | 1 | 0 | 100% |

### Selected Setting

| Parameter | Selected Value |
|---|---|
| Conditioning model | XLSR |
| Layer | 11 |
| Alpha | 1.0 |

### Interpretation

Stage 5A-Refinement solved the parameter-selection problem. It identified XLSR layer 11 with alpha 1.0 as the strongest prototype setting for the next scale-up.

---

## Phase 6 — Stage 5B: 80-File Subset Evaluation

Stage 5B tested whether the selected setting from Stage 5A-Refinement remains stable on a larger 80-file subset.

### Stage 5B Setup

| Item | Value |
|---|---|
| Files | 80 |
| Composition | 20 Spanish HC, 20 Spanish PD, 20 German HC, 20 German PD |
| Conditioning model | XLSR |
| Layer | 11 |
| Alpha | 1.0 |
| Converted files | 80 |

### Stage 5B Audio Validation

| Metric | Value |
|---|---:|
| Converted files | 80/80 |
| Validation success | 100% |
| Sample rate | 22050 Hz |
| Mono | Yes |
| Clipping rate | 0% |
| Maximum peak amplitude | 0.9999 |
| Average RMS | 0.1122 |

### Stage 5B Classification Summary

| Metric | Original | Converted | Delta |
|---|---:|---:|---:|
| All-scenario average UAR | 0.6560 | 0.7392 | +0.0831 |
| Crosslingual-only average UAR | 0.5339 | 0.5312 | -0.0026 |
| Spanish → German-domain UAR delta | — | — | -0.0115 |
| German → Spanish-domain UAR delta | — | — | +0.0062 |

### Stage 5B Interpretation

Stage 5B confirmed that the selected setting scales from 12 files to 80 files. It achieved perfect technical validation and improved the all-scenario average UAR. Crosslingual-only UAR remained nearly stable but did not clearly improve.

---

## Phase 7 — Stage 5C: Full 276-File Evaluation

Stage 5C applied the selected prototype conversion setting to the full 276-file read-text dataset.

### Stage 5C Setup

| Item | Value |
|---|---|
| Files | 276 |
| Spanish HC | 50 |
| Spanish PD | 50 |
| German HC | 88 |
| German PD | 88 |
| Conditioning model | XLSR |
| Layer | 11 |
| Alpha | 1.0 |
| Converted files | 276 |

### Stage 5C Audio Validation

| Metric | Value |
|---|---:|
| Converted files | 276/276 |
| Validation success | 100% |
| Sample rate | 22050 Hz |
| Mono | Yes |
| Clipped files | 3/276 |
| Clipping rate | 1.1% |
| Warning threshold | 5% |
| Validation status | Passed |

### Stage 5C Classification Summary

| Metric | Original | Converted | Delta |
|---|---:|---:|---:|
| All-scenario average UAR | 0.6979 | 0.7330 | +0.0352 |
| Crosslingual-only average UAR | 0.6021 | 0.5663 | -0.0358 |

### Stage 5C Interpretation

Stage 5C confirms that the prototype embedding-conditioned conversion method is technically valid on the full 276-file dataset. The all-scenario average improved after conversion, while the crosslingual-only average decreased slightly.

Therefore, the method supports technical feasibility and diagnostic preservation, but it does not yet prove strong crosslingual performance improvement.

---

## 📊 Full Stage-by-Stage Result Summary

| Phase | Files | Method | Key Output | Result |
|---|---:|---|---|---|
| Baseline | 276 | XLSR/Wav2Vec2/WavLM embeddings | Original speech classification | Completed |
| HiFi-GAN Pilot | 12 | Reconstruction | Reconstructed speech | 12/12 generated |
| HiFi-GAN Subset | 80 | Reconstruction | Reconstructed speech | 80/80 generated |
| HiFi-GAN Full | 276 | Reconstruction | Reconstructed speech | 276/276 generated |
| Stage 5A | 12 | Prototype conversion, XLSR L11, alpha 0.5 | Converted audio | 12/12 valid |
| Stage 5A-Refinement | 180 generated files | Grid search | Best setting selected | XLSR L11 alpha 1.0 |
| Stage 5B | 80 | Prototype conversion, XLSR L11, alpha 1.0 | Subset conversion | 80/80 valid |
| Stage 5C | 276 | Prototype conversion, XLSR L11, alpha 1.0 | Full conversion | 276/276 valid |

---

## 📁 Important Local Paths

| Item | Local Path |
|---|---|
| Project root | `C:\pd-speech-crosslingual` |
| Voice conversion root | `C:\pd-speech-crosslingual\voice_conversion` |
| HiFi-GAN repo | `C:\pd-speech-crosslingual\voice_conversion\hifi-gan` |
| HiFi-GAN environment | `C:\pd-speech-crosslingual\voice_conversion\hifi-gan\hifigan_env` |
| HiFi-GAN checkpoint | `C:\pd-speech-crosslingual\voice_conversion\checkpoints\universal_v1\generator_v1` |
| HiFi-GAN config | `C:\pd-speech-crosslingual\voice_conversion\checkpoints\universal_v1\config.json` |
| Stage 5 root | `C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc` |
| Stage 5A | `C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc` |
| Stage 5A-Refinement | `C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc\stage5a_refinement` |
| Stage 5B | `C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc\stage5b_subset_80` |
| Stage 5C | `C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc\stage5c_full_276` |

---

## 🧾 Key Output Files

| Stage | Key Output |
|---|---|
| Baseline | `outputs/reports/full_baseline_model_comparison_summary.md` |
| Baseline | `outputs/tables/full_model_comparison.csv` |
| HiFi-GAN Full | `voice_conversion/logs_full/hifigan_stage4_evaluation_report.md` |
| Stage 5A | `voice_conversion/stage5_embedding_conditioned_vc/logs_stage5/stage5a_embedding_conditioned_conversion_report.md` |
| Stage 5A-Refinement | `voice_conversion/stage5_embedding_conditioned_vc/stage5a_refinement/logs_refinement/stage5a_refinement_report.md` |
| Stage 5A-Refinement Best Setting | `voice_conversion/stage5_embedding_conditioned_vc/stage5a_refinement/logs_refinement/stage5a_refinement_best_setting.md` |
| Stage 5B | `voice_conversion/stage5_embedding_conditioned_vc/stage5b_subset_80/logs_stage5b/stage5b_subset_80_report.md` |
| Stage 5C | `voice_conversion/stage5_embedding_conditioned_vc/stage5c_full_276/logs_stage5c/stage5c_full_276_report.md` |

---

## 🏗️ Repository Structure

```text
pd-speech-crosslingual/
│
├── configs/
├── metadata/
├── features/
├── outputs/
├── scripts/
├── src/
│
├── voice_conversion/
│   ├── hifi-gan/
│   ├── checkpoints/
│   ├── input_pilot/
│   ├── input_subset_80/
│   ├── input_full/
│   ├── generated/
│   ├── generated_subset_80/
│   ├── generated_full/
│   ├── logs_subset_80/
│   ├── logs_full/
│   │
│   └── stage5_embedding_conditioned_vc/
│       ├── input_pilot_12/
│       ├── converted_spanish_to_german/
│       ├── converted_german_to_spanish/
│       ├── logs_stage5/
│       │
│       ├── stage5a_refinement/
│       ├── stage5b_subset_80/
│       └── stage5c_full_276/
│
├── README.md
├── requirements.txt
└── .gitignore
```

Sensitive raw audio, private metadata, generated audio, checkpoints, large features, and local experiment outputs should remain excluded from public GitHub unless explicitly reviewed and anonymized.

---

## ⚙️ How to Run Baseline

```bash
python scripts/06_run_full_baseline.py --model xlsr
python scripts/06_run_full_baseline.py --model wav2vec2
python scripts/06_run_full_baseline.py --model wavlm
python scripts/08_compare_all_models.py
```

## ⚙️ How to Run Stage 5C Full Evaluation

```bash
python voice_conversion/stage5_embedding_conditioned_vc/stage5c_full_276/scripts/01_prepare_stage5c_full_276_inputs.py
python voice_conversion/stage5_embedding_conditioned_vc/stage5c_full_276/scripts/02_extract_stage5c_xlsr_layer11_embeddings.py
python voice_conversion/stage5_embedding_conditioned_vc/stage5c_full_276/scripts/03_create_stage5c_domain_conditions.py
python voice_conversion/stage5_embedding_conditioned_vc/stage5c_full_276/scripts/04_generate_stage5c_converted_audio.py
python voice_conversion/stage5_embedding_conditioned_vc/stage5c_full_276/scripts/05_validate_stage5c_converted_audio.py
python voice_conversion/stage5_embedding_conditioned_vc/stage5c_full_276/scripts/06_evaluate_stage5c_classification.py
python voice_conversion/stage5_embedding_conditioned_vc/stage5c_full_276/scripts/07_write_stage5c_report.py
```

These commands require local private data, local HiFi-GAN checkpoints, and local generated feature folders. They will not run from a public repository without the private research data.

---

## 🧠 Final Current Scientific Interpretation

The completed experiments show that speech representation models can detect Parkinson’s Disease from Spanish and German read-text speech with strong within-language performance and weaker crosslingual transfer.

HiFi-GAN reconstruction experiments showed that generated/reconstructed audio remains diagnostically usable and does not collapse PD/HC classification.

The prototype embedding-conditioned conversion pipeline successfully generated technically valid converted speech across 12, 80, and 276-file stages. The full 276-file Stage 5C result improved the all-scenario average UAR from 0.6979 to 0.7330, while crosslingual-only UAR decreased from 0.6021 to 0.5663.

Therefore, the current conclusion is:

> The proposed prototype conversion method is technically feasible and diagnostically stable, but it does not yet provide strong evidence of crosslingual improvement.

This result is valuable because it gives a complete, reproducible experimental answer rather than only a positive claim.

---

## 💼 Professional Context

This thesis project connects my academic research in Artificial Intelligence with my professional experience in enterprise technology at SAP SE.

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

## ⚠️ Limitations

| Limitation | Explanation |
|---|---|
| Prototype conversion | The embedding-conditioned conversion is a research prototype, not a production VC system |
| No language translation | Speech is converted toward acoustic/domain conditions, not translated linguistically |
| Read-text only | Current experiments focus on read-text speech |
| Crosslingual improvement limited | Stage 5C did not improve crosslingual-only UAR |
| Sensitive dataset | Raw speech data cannot be shared publicly |
| Local reproducibility | Full reproduction requires private data and local checkpoints |
| Clinical limitation | This is research, not a medical diagnostic system |

---

## 📬 Contact

**Prosenjit Chowdhury**  
M.Sc. Artificial Intelligence  
Friedrich-Alexander-Universität Erlangen-Nürnberg  
Pattern Recognition Lab, FAU  
Working Student at SAP SE  

GitHub: [github.com/prosenjit-chd](https://github.com/prosenjit-chd)

Research interests: Speech AI, Biomedical AI, Machine Learning, Crosslingual AI, Voice Conversion, Data Analysis, Process Automation, and Enterprise AI.
