<div align="center">

# 🧠 Voice Conversion for Multilingual Detection of Parkinson’s Disease Using Speech Signals

### Spanish-German Read-Text Classification using XLSR Speech Embeddings  
### Master’s Thesis Project | Pattern Recognition Lab | FAU Erlangen-Nürnberg

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Speech AI](https://img.shields.io/badge/Speech%20AI-XLSR%20%7C%20Wav2Vec2%20%7C%20WavLM-purple)]()
[![Machine Learning](https://img.shields.io/badge/ML-SVM%20%7C%20Logistic%20Regression-green)]()
[![Research](https://img.shields.io/badge/Research-Parkinson's%20Disease-orange)]()
[![Privacy](https://img.shields.io/badge/Data-Private%20Medical%20Speech-red)]()

</div>

---

## 📌 Project Information

| Field | Details |
|---|---|
| **Author** | Prosenjit Chowdhury |
| **Degree Program** | M.Sc. Artificial Intelligence |
| **University** | Friedrich-Alexander-Universität Erlangen-Nürnberg |
| **Research Lab** | Pattern Recognition Lab, FAU Erlangen-Nürnberg |
| **Supervisor** | Dr.-Ing. Tomás Arias-Vergara |
| **Project Repository** | `pd-speech-crosslingual-MSc-Thesis-PR_Lab_FAU` |
| **GitHub** | [github.com/prosenjit-chd](https://github.com/prosenjit-chd) |
| **Current Milestone** | First XLSR baseline implementation completed |
| **Main Research Area** | Speech AI, Biomedical AI, Multilingual Machine Learning, Voice Conversion |

---

## 📖 Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Research Motivation](#2-research-motivation)
- [3. Thesis Goal](#3-thesis-goal)
- [4. Current Project Phase](#4-current-project-phase)
- [5. Dataset Overview](#5-dataset-overview)
- [6. Data Privacy and Repository Policy](#6-data-privacy-and-repository-policy)
- [7. AI and Machine Learning Approach](#7-ai-and-machine-learning-approach)
- [8. Model Used in the First Baseline](#8-model-used-in-the-first-baseline)
- [9. Planned Follow-up Models](#9-planned-follow-up-models)
- [10. Processing Pipeline](#10-processing-pipeline)
- [11. Classification Scenarios](#11-classification-scenarios)
- [12. Evaluation Metrics](#12-evaluation-metrics)
- [13. Current Baseline Results](#13-current-baseline-results)
- [14. Layer-Level Observations](#14-layer-level-observations)
- [15. Visualization Results](#15-visualization-results)
- [16. Current Implementation Status](#16-current-implementation-status)
- [17. Repository Structure](#17-repository-structure)
- [18. How to Run the Project](#18-how-to-run-the-project)
- [19. Running Scripts Individually](#19-running-scripts-individually)
- [20. Output Files](#20-output-files)
- [21. Deliverables Generated](#21-deliverables-generated)
- [22. Interpretation of the First Baseline](#22-interpretation-of-the-first-baseline)
- [23. Limitations](#23-limitations)
- [24. Next Steps](#24-next-steps)
- [25. Future Extension: Voice Conversion](#25-future-extension-voice-conversion)
- [26. Technical Skills Demonstrated](#26-technical-skills-demonstrated)
- [27. Professional Context](#27-professional-context)
- [28. Selected References](#28-selected-references)
- [29. Contact](#29-contact)

---

## 1. Project Overview

This repository contains the implementation work for my Master’s thesis:

> **Voice Conversion for Multilingual Detection of Parkinson’s Disease Using Speech Signals**

The thesis investigates whether speech-based artificial intelligence can detect **Parkinson’s Disease (PD)** from multilingual speech recordings and whether **Voice Conversion (VC)** can reduce the language mismatch between different speech datasets.

The current implementation focuses on Spanish and German Parkinson’s speech data. The first goal is to build a reliable baseline before applying voice conversion. This baseline measures how well self-supervised speech embeddings can support PD vs healthy control classification within and across languages.

The broader research direction is to evaluate Parkinson’s Disease classification across different languages and later apply voice conversion or speech-domain transformation to improve cross-language generalization.

> **Main Research Question**  
> Can voice conversion reduce language mismatch in multilingual Parkinson’s Disease speech classification while preserving pathology-related acoustic information?

---

## 2. Research Motivation

Parkinson’s Disease can affect speech production. Symptoms may appear in:

- phonation,
- articulation,
- prosody,
- loudness,
- pitch variation,
- rhythm,
- pauses,
- voice stability.

These speech characteristics can be measured from audio and used as non-invasive digital biomarkers for computer-aided Parkinson’s Disease assessment.

However, speech is strongly language-dependent. A model trained on Spanish speech may learn Spanish-specific acoustic or phonetic patterns instead of disease-related speech characteristics. When the same model is tested on German speech, performance may drop because the language domain is different.

This project focuses on the following core challenge:

> A Parkinson’s Disease classifier should learn disease-relevant speech patterns, not only language-specific patterns.

Therefore, the first stage of the thesis builds a strong baseline system before moving to voice conversion.

---

## 3. Thesis Goal

The main goal of this thesis is to investigate whether **voice conversion** can reduce language or domain mismatch in multilingual Parkinson’s Disease speech classification while preserving disease-related acoustic cues.

In simple terms:

```text
Original speech
        ↓
Speech representation model
        ↓
PD vs HC classifier
        ↓
Cross-language performance analysis
        ↓
Voice conversion / speech-domain transformation
        ↓
Re-evaluate PD classification
```

The final objective is not only to generate natural-sounding converted speech. The scientific goal is to test whether converted speech still preserves the Parkinson’s-related acoustic information needed for reliable classification.

---

## 4. Current Project Phase

<div align="center">

### ✅ First Baseline Implementation Completed

</div>

The current completed milestone is the **first XLSR baseline implementation**.

This stage does **not yet apply voice conversion**. Instead, it establishes a reproducible baseline for Parkinson’s Disease classification using Spanish and German read-text speech recordings.

The baseline answers the following questions:

| Question | Purpose |
|---|---|
| Can XLSR speech embeddings support PD vs HC classification? | Tests whether pretrained speech embeddings contain disease-related information |
| Which XLSR layer gives the strongest baseline result? | Compares layers 0, 4, 8, and 11 |
| How much performance is lost across languages? | Measures Spanish-German language mismatch |
| Do embeddings separate more by disease label or by language? | Uses PCA and t-SNE visual analysis |
| Which baseline should be used before voice conversion? | Establishes the reference point for the next stage |

The first baseline was completed using the multilingual XLSR speech model and Spanish-German read-text speech data.

---

## 5. Dataset Overview

The current baseline uses Spanish and German Parkinson’s Disease speech recordings.

### Current Baseline Task

Only the **read-text speech task** is used in the first implementation.

This decision was made because read-text recordings are available in both Spanish and German datasets. This makes the first comparison more controlled and fair.

### Dataset Summary

| Item | Value |
|---|---:|
| **Task** | Read-text only |
| **Total recordings** | 276 |
| **Spanish recordings** | 100 |
| **German recordings** | 176 |
| **PD recordings** | 138 |
| **HC recordings** | 138 |
| **Dataset index file** | `metadata/dataset_index_readtext.csv` |

### Label Meaning

| Label | Meaning |
|---|---|
| **PD** | Parkinson’s Disease patient |
| **HC** | Healthy Control |

The Spanish and German read-text setup follows the planned first dataset decision: use comparable read-text recordings before expanding to other speech tasks such as monologue, vowel, pataka/DDK, words, sentences, or full recordings.

---

## 6. Data Privacy and Repository Policy

> **Important:** This project uses sensitive medical speech data.

The raw speech recordings and patient metadata are not included in this repository.

The following files and folders must remain local and must not be uploaded to GitHub:

```text
input/
data/
*.wav
*.mp3
*.flac
*.m4a
*.xlsx
raw metadata files
speaker-identifiable files
local dataset paths
```

This repository is intended to contain only:

- source code,
- configuration files,
- documentation,
- non-sensitive result summaries,
- non-identifying plots,
- safe result tables.

Raw audio files, clinical metadata, and private patient-related information are excluded through `.gitignore`.

---

## 7. AI and Machine Learning Approach

This project uses both **Deep Learning** and **Machine Learning**.

### Deep Learning Part

Pretrained self-supervised speech models are used to convert raw audio into high-dimensional speech embeddings.

In the first baseline, the main model is:

```text
facebook/wav2vec2-large-xlsr-53
```

This is a pretrained multilingual XLSR wav2vec 2.0 model originally released by Facebook AI Research, now Meta AI. In this project, it is used as a **feature extractor**, not as a speech-to-text system.

### Machine Learning Part

The extracted embeddings are used to train classical machine learning classifiers:

```text
Linear SVM
Logistic Regression
```

These classifiers predict:

```text
PD = Parkinson’s Disease
HC = Healthy Control
```

### Overall AI Flow

```text
Raw speech audio
        ↓
Pretrained deep learning speech model
        ↓
Speech embeddings
        ↓
Classical machine learning classifier
        ↓
PD or HC prediction
```

---

## 8. Model Used in the First Baseline

The completed first implementation uses:

```text
facebook/wav2vec2-large-xlsr-53
```

For README readability, this can also be described as:

> **Pretrained multilingual XLSR wav2vec 2.0 speech representation model**

Selected hidden layers:

```text
0, 4, 8, 11
```

For each audio file:

```text
audio recording
        ↓
XLSR model
        ↓
hidden layer output
        ↓
mean pooling over time
        ↓
one fixed-length embedding vector
```

These embeddings are then used for visualization and classification.

---

## 9. Planned Follow-up Models

The first completed baseline uses only XLSR.

The following models are planned for follow-up comparison:

| Model | Provider / Model ID | Planned Role |
|---|---|---|
| **XLSR** | `facebook/wav2vec2-large-xlsr-53` | Completed first baseline |
| **Wav2Vec2** | `facebook/wav2vec2-base` | Planned comparison model |
| **WavLM** | `microsoft/wavlm-base` | Planned comparison model |

These models will allow comparison between multilingual XLSR representations and other self-supervised speech representations.

---

## 10. Processing Pipeline

The baseline pipeline follows this structure:

```text
Audio recordings
        ↓
Audio preprocessing
        ↓
XLSR feature extraction
        ↓
Layer-wise embedding creation
        ↓
PCA and t-SNE visualization
        ↓
PD vs HC classification
        ↓
Within-language and cross-language evaluation
        ↓
Result tables, figures, and summary report
```

### Pipeline Steps

| Step | Description | Output |
|---|---|---|
| **1. Dataset indexing** | Scans Spanish and German read-text folders and creates a structured dataset index | `metadata/dataset_index_readtext.csv` |
| **2. Audio preprocessing** | Loads audio, converts to mono, resamples to 16 kHz, and normalizes | Preprocessed waveform |
| **3. Feature extraction** | Extracts hidden states from selected XLSR layers | Layer-wise hidden states |
| **4. Embedding creation** | Mean-pools hidden states over time | One fixed-length vector per recording |
| **5. Visualization** | Generates PCA and t-SNE plots | `outputs/figures/` |
| **6. Classification** | Trains Linear SVM and Logistic Regression | Classification result tables |
| **7. Evaluation** | Tests within-language, cross-language, and multilingual scenarios | UAR, accuracy, sensitivity, specificity, AUC |
| **8. Reporting** | Generates summary tables and reports | `outputs/reports/` |

---

## 11. Classification Scenarios

The first baseline evaluates five scenarios:

| Scenario | Purpose |
|---|---|
| **Spanish → Spanish** | Within-language Spanish baseline |
| **Spanish → German** | Cross-language transfer from Spanish to German |
| **German → German** | Within-language German baseline |
| **German → Spanish** | Cross-language transfer from German to Spanish |
| **Spanish + German → Spanish + German** | Combined multilingual baseline |

These scenarios are designed to measure whether the classifier learns disease-relevant cues or language-dependent patterns.

---

## 12. Evaluation Metrics

The main metric is:

```text
UAR = Unweighted Average Recall
```

UAR is also commonly interpreted as balanced accuracy. It is useful because it treats both classes equally.

Additional metrics:

| Metric | Meaning |
|---|---|
| **Accuracy** | Overall percentage of correct predictions |
| **Sensitivity** | Ability to correctly detect PD |
| **Specificity** | Ability to correctly detect HC |
| **AUC** | Area under the ROC curve |

---

## 13. Current Baseline Results

The first XLSR baseline has been completed successfully.

### Best Result Summary

| Result | Layer | Classifier | UAR | Accuracy | Sensitivity | Specificity | AUC |
|---|---:|---|---:|---:|---:|---:|---:|
| **Best Spanish → Spanish** | 4 | Linear SVM | 0.8400 | 0.8400 | 0.8400 | 0.8400 | 0.8636 |
| **Best German → German** | 11 | Logistic Regression | 0.7670 | 0.7670 | 0.7386 | 0.7955 | 0.8319 |
| **Best Spanish → German** | 8 | Logistic Regression | 0.6477 | 0.6477 | 0.5568 | 0.7386 | 0.7136 |
| **Best German → Spanish** | 8 | Logistic Regression | 0.6700 | 0.6700 | 0.7200 | 0.6200 | 0.7452 |
| **Best Combined** | 8 | Logistic Regression | 0.7826 | 0.7826 | 0.7681 | 0.7971 | 0.8512 |

### Main Finding

The results show that within-language performance is stronger than cross-language transfer.

Example:

```text
Spanish → Spanish: UAR = 0.8400
Spanish → German: UAR = 0.6477
```

This is a performance drop of approximately:

```text
0.1923 UAR
```

This confirms a clear Spanish-German language/domain mismatch.

---

## 14. Layer-Level Observations

The first baseline shows that layer choice matters.

| Layer | Observation |
|---:|---|
| **0** | Useful acoustic-level baseline |
| **4** | Best Spanish within-language result |
| **8** | Best cross-language transfer result |
| **11** | Best German within-language result |

The strongest cross-language transfer results were obtained from **layer 8**, suggesting that mid-level XLSR representations may provide a better balance between acoustic detail and abstract speech structure.

---

## 15. Visualization Results

PCA and t-SNE plots were generated for every evaluated XLSR layer.

Each layer was visualized in three ways:

| Plot Type | Purpose |
|---|---|
| **By label** | Checks whether PD and HC form distinguishable regions |
| **By language** | Checks whether Spanish and German dominate the embedding space |
| **By combined group** | Checks Spanish HC, Spanish PD, German HC, and German PD together |

The main visualization finding is:

> The embedding space separates more strongly by language than by PD/HC label.

This means that Spanish and German recordings often form clearer clusters than Parkinson’s Disease and Healthy Control recordings.

This supports the classification result:

```text
Within-language performance > Cross-language performance
```

Therefore, the visualizations confirm that language/domain mismatch is a major challenge for this thesis.

---

## 16. Current Implementation Status

<div align="center">

### ✅ First Baseline Milestone Completed

</div>

Completed tasks:

- Created project folder structure.
- Created dataset index for Spanish and German read-text data.
- Processed 276 recordings.
- Extracted XLSR embeddings from layers 0, 4, 8, and 11.
- Generated PCA and t-SNE plots.
- Ran Linear SVM and Logistic Regression classifiers.
- Evaluated all five Spanish-German scenarios.
- Generated result tables.
- Generated baseline summary report.
- Created a safe deliverable package for supervisor review.
- Excluded raw audio, private metadata, and sensitive files.

Current safe package:

```text
baseline_outputs_for_tomas.zip
```

---

## 17. Repository Structure

```text
pd-speech-crosslingual/
│
├── configs/
│   └── baseline_config.yaml
│
├── input/
│   ├── Spanish/
│   ├── German/
│   └── metadata files
│
├── metadata/
│   └── dataset_index_readtext.csv
│
├── features/
│   └── xlsr/
│       ├── xlsr_readtext_layer0.csv
│       ├── xlsr_readtext_layer4.csv
│       ├── xlsr_readtext_layer8.csv
│       └── xlsr_readtext_layer11.csv
│
├── outputs/
│   ├── figures/
│   │   ├── pca_*.png
│   │   └── tsne_*.png
│   │
│   ├── tables/
│   │   ├── classification_results_xlsr_readtext_layer0.csv
│   │   ├── classification_results_xlsr_readtext_layer4.csv
│   │   ├── classification_results_xlsr_readtext_layer8.csv
│   │   ├── classification_results_xlsr_readtext_layer11.csv
│   │   └── model_layer_comparison.csv
│   │
│   └── reports/
│       └── baseline_summary.md
│
├── scripts/
│   ├── 00_prepare_project.py
│   ├── 01_create_dataset_index.py
│   ├── 02_extract_embeddings.py
│   ├── 03_visualize_embeddings.py
│   ├── 04_cross_language_classification.py
│   ├── 05_experiment_summary.py
│   ├── 06_run_full_baseline.py
│   └── 07_package_deliverable.py
│
├── src/
│   ├── audio_utils.py
│   ├── classification.py
│   ├── dataset_index.py
│   ├── embedding_extractor.py
│   ├── metrics.py
│   ├── utils.py
│   └── visualization.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:**  
> The `input/`, raw audio files, private metadata, and possibly large feature files should remain excluded from GitHub depending on privacy and file-size requirements.

---

## 18. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/prosenjit-chd/pd-speech-crosslingual-MSc-Thesis-PR_Lab_FAU.git
cd pd-speech-crosslingual-MSc-Thesis-PR_Lab_FAU
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare local data

Place the local datasets inside:

```text
input/
```

Expected local structure:

```text
input/
├── Spanish/
├── German/
└── metadata files
```

Raw data is not included in the repository.

### Step 5: Initialize project folders

```bash
python scripts/00_prepare_project.py
```

### Step 6: Create dataset index

```bash
python scripts/01_create_dataset_index.py
```

Expected output:

```text
metadata/dataset_index_readtext.csv
```

### Step 7: Run the full XLSR baseline

```bash
python scripts/06_run_full_baseline.py
```

This runs:

- dataset indexing,
- embedding extraction,
- visualization,
- classification,
- summary generation.

### Step 8: Package safe deliverables

```bash
python scripts/07_package_deliverable.py
```

Expected output:

```text
baseline_outputs_for_tomas/
baseline_outputs_for_tomas.zip
```

---

## 19. Running Scripts Individually

The full pipeline can also be run step by step.

Create the dataset index:

```bash
python scripts/01_create_dataset_index.py
```

Extract XLSR embeddings:

```bash
python scripts/02_extract_embeddings.py --model xlsr --layers 0 4 8 11
```

Generate PCA and t-SNE visualizations:

```bash
python scripts/03_visualize_embeddings.py
```

Run cross-language classification:

```bash
python scripts/04_cross_language_classification.py
```

Generate experiment summary:

```bash
python scripts/05_experiment_summary.py
```

---

## 20. Output Files

### Reports

```text
outputs/reports/baseline_summary.md
```

### Tables

```text
outputs/tables/model_layer_comparison.csv
outputs/tables/classification_results_xlsr_readtext_layer0.csv
outputs/tables/classification_results_xlsr_readtext_layer4.csv
outputs/tables/classification_results_xlsr_readtext_layer8.csv
outputs/tables/classification_results_xlsr_readtext_layer11.csv
```

### Figures

```text
outputs/figures/pca_*.png
outputs/figures/tsne_*.png
```

The figure files include visualizations by:

```text
label
language
combined group
```

---

## 21. Deliverables Generated

The first baseline generated the following safe deliverables:

| Deliverable | Purpose |
|---|---|
| `baseline_summary.md` | Human-readable summary of setup, dataset, results, and interpretation |
| `model_layer_comparison.csv` | Aggregated comparison across layers, classifiers, and scenarios |
| `classification_results_xlsr_readtext_layer*.csv` | Detailed layer-wise classification results |
| `pca_*.png` | PCA visualizations |
| `tsne_*.png` | t-SNE visualizations |
| `README.md` | Explanation of package contents and privacy exclusions |
| `baseline_outputs_for_tomas.zip` | Safe package prepared for supervisor review |

The package intentionally excludes:

```text
raw audio
private metadata
input folder
speaker-identifiable data
large derived feature files
```

---

## 22. Interpretation of the First Baseline

The first baseline provides a clear and useful result for the thesis.

The model performs well within the same language:

```text
Spanish → Spanish: 0.8400 UAR
German → German: 0.7670 UAR
```

But cross-language transfer is weaker:

```text
Spanish → German: 0.6477 UAR
German → Spanish: 0.6700 UAR
```

This confirms that the model representation and classifier are affected by language/domain differences.

Therefore, the first baseline supports the next research step:

> Apply voice conversion or speech-domain adaptation and test whether cross-language PD classification improves while preserving disease-relevant cues.

---

## 23. Limitations

The current implementation has several limitations:

| Limitation | Explanation |
|---|---|
| **Read-text only** | Only the read-text task was used in the first baseline. |
| **One model completed so far** | Only XLSR has been evaluated in the current completed implementation. |
| **No fine-tuning yet** | The speech model was used as a fixed feature extractor. |
| **Exploratory plots** | PCA and t-SNE help interpretation but do not replace classification metrics. |
| **Setup dependency** | Results depend on preprocessing, speaker split, random seed, and cross-validation setup. |
| **Other tasks pending** | Monologue, vowel, pataka/DDK, words, sentences, and full recordings remain for later experiments. |

---

## 24. Next Steps

The next planned steps are:

1. Review the first baseline results with **Dr.-Ing. Tomás Arias-Vergara** and align the next experimental direction based on his feedback.
2. Use the current read-text XLSR baseline as the initial reference point for future experiments.
3. Run the same baseline pipeline with **Wav2Vec2** and **WavLM**.
4. Evaluate whether additional XLSR layers improve PD vs HC classification.
5. Analyze PCA and t-SNE plots together with classification metrics.
6. Extend the dataset tasks if needed, including monologue, vowel, pataka/DDK, sentences, words, and full recordings.
7. Start the voice conversion stage after the baseline is reviewed.
8. Compare PD classification before and after voice conversion.
9. Evaluate whether language mismatch is reduced while PD-relevant speech characteristics are preserved.

---

## 25. Future Extension: Voice Conversion

After the baseline stage, this thesis will move toward voice conversion or speech-domain transformation.

Example idea:

```text
German speech
        ↓
Voice conversion
        ↓
Spanish-like speech/domain
        ↓
Embedding extraction
        ↓
Spanish-trained classifier
        ↓
PD/HC prediction
```

The key scientific challenge is:

> Reduce language mismatch without removing Parkinson’s Disease information.

Important PD-related cues may include:

- articulation difficulty,
- reduced pitch variation,
- monotone voice,
- rhythm irregularity,
- breathiness,
- pause patterns,
- voice instability.

The final evaluation will compare classification performance before and after voice conversion.

---

## 26. Technical Skills Demonstrated

This project demonstrates skills in:

| Skill Area | Evidence in This Project |
|---|---|
| **Speech AI** | Audio preprocessing, 16 kHz conversion, speech embeddings |
| **Deep Learning** | XLSR/wav2vec2 representation extraction |
| **Machine Learning** | SVM, Logistic Regression, cross-validation, UAR/AUC evaluation |
| **Multilingual AI** | Spanish-German transfer and language mismatch analysis |
| **Biomedical AI** | PD vs HC classification using medical speech data |
| **Research Engineering** | Reproducible pipeline, structured results, safe deliverables |
| **Data Privacy** | Exclusion of raw medical data and private metadata |
| **Technical Communication** | Report writing, result interpretation, supervisor-ready documentation |

---

## 27. Professional Context

I am **Prosenjit Chowdhury**, an M.Sc. Artificial Intelligence student at **Friedrich-Alexander-Universität Erlangen-Nürnberg**.

Alongside my studies, I work as a **Working Student at SAP SE**. My professional experience includes work across three SAP areas:

| SAP Area | Experience Focus |
|---|---|
| **SAP LeanIX and SAP Signavio Content Marketing** | Digital content operations, publishing workflows, product communication, content coordination, and marketing content support |
| **Professional Services and Engineering, Construction & Operations Industries** | Business content coordination, stakeholder communication, customer success stories, go-live stories, and industry-focused content support |
| **ERP PCX / Enterprise Systems and Process Automation** | Data validation, metadata quality, enterprise system support, Excel automation, VBA, and Power Automate |

This thesis connects my academic focus in artificial intelligence with practical experience in enterprise technology, structured documentation, data-driven analysis, responsible AI project execution, and cross-functional collaboration.

---

## 28. Selected References

1. J. C. Vásquez-Correa et al., “Convolutional Neural Networks and a Transfer Learning Strategy to Classify Parkinson’s Disease from Speech in Three Different Languages,” Springer, 2019.

2. A. Hernandez et al., “Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria Detection in Parkinson’s Disease,” arXiv, 2026.

3. C.-J. Li et al., “Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages,” Interspeech, 2025.

4. A. Suppa et al., “Voice in Parkinson’s Disease: A Machine Learning Study,” Frontiers in Neurology, 2022.

5. Y. A. Li et al., “StyleTTS 2: Towards Human-Level Text-to-Speech,” NeurIPS, 2023.

---

## 29. Contact

<div align="center">

### 👤 Prosenjit Chowdhury

**M.Sc. Artificial Intelligence | FAU Erlangen-Nürnberg**  
**Pattern Recognition Lab | Master’s Thesis Research**  
**Working Student at SAP SE**

</div>

<br>

| Category | Information |
|---|---|
| **Name** | Prosenjit Chowdhury |
| **Degree Program** | M.Sc. Artificial Intelligence |
| **University** | Friedrich-Alexander-Universität Erlangen-Nürnberg |
| **Research Lab** | Pattern Recognition Lab, FAU Erlangen-Nürnberg |
| **Supervisor** | Dr.-Ing. Tomás Arias-Vergara |
| **GitHub** | [github.com/prosenjit-chd](https://github.com/prosenjit-chd) |
| **Research Project** | Voice Conversion for Multilingual Detection of Parkinson’s Disease Using Speech Signals |
| **Project Focus** | Speech AI, Parkinson’s Disease detection, multilingual speech processing, XLSR/Wav2Vec2 embeddings, and voice conversion |

---

## 💼 SAP Professional Experience

| Area | Details |
|---|---|
| **Current Role** | Working Student at SAP SE |
| **Professional Track** | Enterprise Systems, Data Operations, Business Transformation, Content Operations, and Process Automation |
| **SAP Area 1** | SAP LeanIX and SAP Signavio Content Marketing |
| **SAP Area 2** | Professional Services and Engineering, Construction & Operations Industries |
| **SAP Area 3** | ERP PCX / Enterprise Systems and Process Automation |

---

## 🧩 Professional Focus Areas

<div align="center">

**Business & Data Analysis** • **Digital Transformation** • **Product Operations** • **Process Automation** • **AI Research** • **Speech Technology**

</div>

---

<div align="center">

### 🚀 Current Status

**First XLSR baseline completed.**  
**Next stage: review, model comparison, and preparation for voice conversion experiments.**

</div>


