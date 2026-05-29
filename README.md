<div align="center">

# 🧠 Voice Conversion for Multilingual Detection of Parkinson’s Disease Using Speech Signals

### Spanish-German Read-Text Classification using XLSR, Wav2Vec2, and WavLM Speech Embeddings  
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
| **Current Milestone** | Full first baseline model comparison completed |
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
- [8. Speech Representation Models](#8-speech-representation-models)
- [9. Processing Pipeline](#9-processing-pipeline)
- [10. Classification Scenarios](#10-classification-scenarios)
- [11. Evaluation Metrics](#11-evaluation-metrics)
- [12. Full Baseline Model Comparison Results](#12-full-baseline-model-comparison-results)
- [13. Model-Level Observations](#13-model-level-observations)
- [14. Visualization Results](#14-visualization-results)
- [15. Current Implementation Status](#15-current-implementation-status)
- [16. Repository Structure](#16-repository-structure)
- [17. How to Run the Project](#17-how-to-run-the-project)
- [18. Running Scripts Individually](#18-running-scripts-individually)
- [19. Output Files](#19-output-files)
- [20. Deliverables Generated](#20-deliverables-generated)
- [21. Interpretation of the Full Baseline](#21-interpretation-of-the-full-baseline)
- [22. Limitations](#22-limitations)
- [23. Next Steps](#23-next-steps)
- [24. Future Extension: Voice Conversion](#24-future-extension-voice-conversion)
- [25. Technical Skills Demonstrated](#25-technical-skills-demonstrated)
- [26. Professional Context](#26-professional-context)
- [27. Selected References](#27-selected-references)
- [28. Contact](#28-contact)

---

## 1. Project Overview

This repository contains the implementation work for my Master’s thesis:

> **Voice Conversion for Multilingual Detection of Parkinson’s Disease Using Speech Signals**

The thesis investigates whether speech-based artificial intelligence can detect **Parkinson’s Disease (PD)** from multilingual speech recordings and whether **Voice Conversion (VC)** can reduce language or domain mismatch between different speech datasets.

The current implementation focuses on Spanish and German Parkinson’s speech data. The first major objective was to build a reproducible baseline before applying voice conversion. This baseline measures how well self-supervised speech embeddings can support PD vs healthy control classification within and across languages.

The baseline stage has now been extended from the first XLSR implementation to a complete model comparison using:

```text
XLSR
Wav2Vec2
WavLM
````

The broader research direction is to evaluate Parkinson’s Disease classification across languages and later apply voice conversion or speech-domain transformation to improve cross-language generalization.

> **Main Research Question**
> Can voice conversion reduce language mismatch in multilingual Parkinson’s Disease speech classification while preserving pathology-related acoustic information?

---

## 2. Research Motivation

Parkinson’s Disease can affect speech production. Symptoms may appear in:

* phonation,
* articulation,
* prosody,
* loudness,
* pitch variation,
* rhythm,
* pauses,
* voice stability.

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

### ✅ Full First Baseline Model Comparison Completed

</div>

The current completed milestone is the **full first baseline model comparison** for Spanish-German Parkinson’s speech detection.

The first implementation started with **XLSR**, and the initial direction was reviewed and approved by the supervisor. The baseline was then extended to include **standard Wav2Vec2** and **WavLM**, using the same dataset, classification scenarios, selected layers, metrics, and evaluation structure.

This stage does **not yet apply voice conversion**. Instead, it establishes a reproducible baseline for Parkinson’s Disease classification using Spanish and German read-text speech recordings across multiple state-of-the-art speech representation models.

The full baseline answers the following questions:

| Question                                                          | Purpose                                                                  |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Can pretrained speech embeddings support PD vs HC classification? | Tests whether speech representations contain disease-related information |
| Which model performs best among XLSR, Wav2Vec2, and WavLM?        | Compares three self-supervised speech representation models              |
| Which layer gives the strongest result?                           | Compares layers 0, 4, 8, and 11                                          |
| How much performance is lost across languages?                    | Measures Spanish-German language mismatch                                |
| Do embeddings separate more by disease label or by language?      | Uses PCA and t-SNE visual analysis                                       |
| Which baseline should be used before voice conversion?            | Establishes the reference point for the next thesis stage                |

---

## 5. Dataset Overview

The current baseline uses Spanish and German Parkinson’s Disease speech recordings.

### Current Baseline Task

Only the **read-text speech task** is used in the current implementation.

This decision was made because read-text recordings are available in both Spanish and German datasets. This makes the first comparison more controlled and fair.

### Dataset Summary

| Item                   |                                 Value |
| ---------------------- | ------------------------------------: |
| **Task**               |                        Read-text only |
| **Total recordings**   |                                   276 |
| **Spanish recordings** |                                   100 |
| **German recordings**  |                                   176 |
| **PD recordings**      |                                   138 |
| **HC recordings**      |                                   138 |
| **Dataset index file** | `metadata/dataset_index_readtext.csv` |

### Label Meaning

| Label  | Meaning                     |
| ------ | --------------------------- |
| **PD** | Parkinson’s Disease patient |
| **HC** | Healthy Control             |

The Spanish and German read-text setup follows the planned first dataset decision: use comparable read-text recordings before expanding to other speech tasks such as monologue, vowel, pataka/DDK, words, sentences, or full recordings.

---

## 6. Data Privacy and Repository Policy

> **Important:** This project uses sensitive medical speech data.

The raw speech recordings and patient metadata are not included in this repository.

The following files and folders must remain local and must not be uploaded to GitHub:

```text
input/
data/
metadata/
features/
outputs/
*.wav
*.mp3
*.flac
*.m4a
*.xlsx
raw metadata files
speaker-identifiable files
local dataset paths
raw embedding feature files
```

This repository is intended to contain only:

* source code,
* configuration files,
* documentation,
* non-sensitive result summaries,
* non-identifying plots,
* safe result tables.

Raw audio files, clinical metadata, dataset index files with local paths, and private patient-related information are excluded through `.gitignore`.

---

## 7. AI and Machine Learning Approach

This project uses both **Deep Learning** and **Machine Learning**.

### Deep Learning Part

Pretrained self-supervised speech models are used to convert raw audio into high-dimensional speech embeddings.

The baseline uses:

```text
XLSR
Wav2Vec2
WavLM
```

These models are used as **feature extractors**, not as speech-to-text systems.

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

## 8. Speech Representation Models

The completed baseline compares three pretrained speech representation models.

| Model        | Provider / Model ID               | Role in Baseline                                                                                      |
| ------------ | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **XLSR**     | `facebook/wav2vec2-large-xlsr-53` | Multilingual Wav2Vec2-based model and first approved baseline                                         |
| **Wav2Vec2** | `facebook/wav2vec2-base`          | Standard Wav2Vec2 baseline for comparison                                                             |
| **WavLM**    | `microsoft/wavlm-base`            | Strong general-purpose speech representation model with best cross-language transfer in this baseline |

Selected hidden layers:

```text
0, 4, 8, 11
```

For each audio file:

```text
audio recording
        ↓
speech representation model
        ↓
hidden layer output
        ↓
mean pooling over time
        ↓
one fixed-length embedding vector
```

These embeddings are then used for visualization and classification.

---

## 9. Processing Pipeline

The baseline pipeline follows this structure:

```text
Audio recordings
        ↓
Audio preprocessing
        ↓
XLSR / Wav2Vec2 / WavLM feature extraction
        ↓
Layer-wise embedding creation
        ↓
PCA and t-SNE visualization
        ↓
PD vs HC classification
        ↓
Within-language and cross-language evaluation
        ↓
Model comparison
        ↓
Result tables, figures, and summary report
```

### Pipeline Steps

| Step                       | Description                                                                       | Output                                       |
| -------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------- |
| **1. Dataset indexing**    | Scans Spanish and German read-text folders and creates a structured dataset index | `metadata/dataset_index_readtext.csv`        |
| **2. Audio preprocessing** | Loads audio, converts to mono, resamples to 16 kHz, and normalizes                | Preprocessed waveform                        |
| **3. Feature extraction**  | Extracts hidden states from selected model layers                                 | Layer-wise hidden states                     |
| **4. Embedding creation**  | Mean-pools hidden states over time                                                | One fixed-length vector per recording        |
| **5. Visualization**       | Generates PCA and t-SNE plots                                                     | `outputs/figures/`                           |
| **6. Classification**      | Trains Linear SVM and Logistic Regression                                         | Classification result tables                 |
| **7. Evaluation**          | Tests within-language, cross-language, and multilingual scenarios                 | UAR, accuracy, sensitivity, specificity, AUC |
| **8. Model comparison**    | Compares XLSR, Wav2Vec2, and WavLM                                                | `full_model_comparison.csv`                  |
| **9. Reporting**           | Generates summary tables and reports                                              | `outputs/reports/`                           |

---

## 10. Classification Scenarios

The baseline evaluates five scenarios:

| Scenario                                | Purpose                                        |
| --------------------------------------- | ---------------------------------------------- |
| **Spanish → Spanish**                   | Within-language Spanish baseline               |
| **Spanish → German**                    | Cross-language transfer from Spanish to German |
| **German → German**                     | Within-language German baseline                |
| **German → Spanish**                    | Cross-language transfer from German to Spanish |
| **Spanish + German → Spanish + German** | Combined multilingual baseline                 |

These scenarios are designed to measure whether the classifier learns disease-relevant cues or language-dependent patterns.

---

## 11. Evaluation Metrics

The main metric is:

```text
UAR = Unweighted Average Recall
```

UAR is also commonly interpreted as balanced accuracy. It is useful because it treats both classes equally.

Additional metrics:

| Metric          | Meaning                                   |
| --------------- | ----------------------------------------- |
| **Accuracy**    | Overall percentage of correct predictions |
| **Sensitivity** | Ability to correctly detect PD            |
| **Specificity** | Ability to correctly detect HC            |
| **AUC**         | Area under the ROC curve                  |

---

## 12. Full Baseline Model Comparison Results

The full model comparison has been completed successfully.

### Best Result Summary

| Scenario                                     | Best Model | Layer | Classifier          |    UAR | Accuracy | Sensitivity | Specificity |    AUC |
| -------------------------------------------- | ---------- | ----: | ------------------- | -----: | -------: | ----------: | ----------: | -----: |
| **Best Spanish → Spanish**                   | XLSR       |     4 | Linear SVM          | 0.8400 |   0.8400 |      0.8400 |      0.8400 | 0.8636 |
| **Best German → German**                     | WavLM      |     8 | Logistic Regression | 0.8182 |   0.8182 |      0.8068 |      0.8295 | 0.8680 |
| **Best Spanish → German**                    | WavLM      |    11 | Logistic Regression | 0.7273 |   0.7273 |      0.8750 |      0.5795 | 0.7487 |
| **Best German → Spanish**                    | WavLM      |     0 | Logistic Regression | 0.6900 |   0.6900 |      0.6200 |      0.7600 | 0.7488 |
| **Best Spanish + German → Spanish + German** | Wav2Vec2   |     8 | Logistic Regression | 0.8080 |   0.8080 |      0.8116 |      0.8043 | 0.8427 |

### Main Finding

The results show that within-language performance is stronger than cross-language transfer.

Example:

```text
Spanish → Spanish: UAR = 0.8400
Spanish → German: UAR = 0.7273
```

This is a performance drop of approximately:

```text
0.1127 UAR
```

For the German direction:

```text
German → German: UAR = 0.8182
German → Spanish: UAR = 0.6900
```

This is a performance drop of approximately:

```text
0.1282 UAR
```

This confirms a clear Spanish-German language/domain mismatch, even after comparing multiple speech representation models.

---

## 13. Model-Level Observations

The full comparison shows that each model contributes a different insight.

| Model        | Best Result                                                         | Main Observation                                |
| ------------ | ------------------------------------------------------------------- | ----------------------------------------------- |
| **XLSR**     | 0.8400 UAR for Spanish → Spanish                                    | Strongest single within-language Spanish result |
| **Wav2Vec2** | 0.8080 UAR for Spanish + German → Spanish + German                  | Strongest combined multilingual result          |
| **WavLM**    | 0.7273 UAR for Spanish → German and 0.6900 UAR for German → Spanish | Strongest cross-language transfer model         |

### Interpretation

* **XLSR** achieved the strongest single result in the Spanish within-language setting.
* **WavLM** achieved the best cross-language transfer in both transfer directions.
* **Wav2Vec2** achieved the strongest combined multilingual training result.
* **Layer 8** appears important for several strong results, especially German within-language and combined multilingual classification.
* **Layer 11** produced the strongest Spanish to German transfer result with WavLM.
* **Layer 0** produced the strongest German to Spanish transfer result with WavLM.

---

## 14. Visualization Results

PCA and t-SNE plots were generated for every model and every evaluated layer.

Each layer was visualized in three ways:

| Plot Type             | Purpose                                                          |
| --------------------- | ---------------------------------------------------------------- |
| **By label**          | Checks whether PD and HC form distinguishable regions            |
| **By language**       | Checks whether Spanish and German dominate the embedding space   |
| **By combined group** | Checks Spanish HC, Spanish PD, German HC, and German PD together |

The main visualization finding is:

> The embedding space often separates more strongly by language than by PD/HC label.

This means that Spanish and German recordings can form clearer regions than Parkinson’s Disease and Healthy Control recordings.

This supports the classification result:

```text
Within-language performance > Cross-language performance
```

Therefore, the visualizations confirm that language/domain mismatch is a major challenge for this thesis.

Recommended visualizations to inspect:

```text
baseline_multimodel_pipeline_diagram.png
pca_xlsr_readtext_layer4_by_language.png
tsne_wavlm_readtext_layer11_by_language.png
tsne_wavlm_readtext_layer11_by_group.png
pca_wav2vec2_readtext_layer8_by_label.png
tsne_wavlm_readtext_layer8_by_label.png
pca_wavlm_readtext_layer8_by_language.png
```

---

## 15. Current Implementation Status

<div align="center">

### ✅ Full First Baseline Task Completed

</div>

Completed tasks:

* Created project folder structure.
* Created dataset index for Spanish and German read-text data.
* Processed 276 recordings.
* Extracted embeddings using XLSR, Wav2Vec2, and WavLM.
* Extracted features from layers 0, 4, 8, and 11.
* Generated PCA and t-SNE plots.
* Ran Linear SVM and Logistic Regression classifiers.
* Evaluated all five Spanish-German scenarios.
* Generated model-specific result tables.
* Generated full model comparison table.
* Generated model-specific summary reports.
* Generated full baseline comparison summary report.
* Created a safe deliverable package.
* Excluded raw audio, private metadata, dataset index, and sensitive files.

Current safe package:

```text
full_baseline_outputs_for_tomas.zip
```

---

## 16. Repository Structure

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
│   ├── xlsr/
│   │   ├── xlsr_readtext_layer0.csv
│   │   ├── xlsr_readtext_layer4.csv
│   │   ├── xlsr_readtext_layer8.csv
│   │   └── xlsr_readtext_layer11.csv
│   │
│   ├── wav2vec2/
│   │   ├── wav2vec2_readtext_layer0.csv
│   │   ├── wav2vec2_readtext_layer4.csv
│   │   ├── wav2vec2_readtext_layer8.csv
│   │   └── wav2vec2_readtext_layer11.csv
│   │
│   └── wavlm/
│       ├── wavlm_readtext_layer0.csv
│       ├── wavlm_readtext_layer4.csv
│       ├── wavlm_readtext_layer8.csv
│       └── wavlm_readtext_layer11.csv
│
├── outputs/
│   ├── figures/
│   │   ├── pca_*.png
│   │   └── tsne_*.png
│   │
│   ├── tables/
│   │   ├── classification_results_xlsr_readtext_layer*.csv
│   │   ├── classification_results_wav2vec2_readtext_layer*.csv
│   │   ├── classification_results_wavlm_readtext_layer*.csv
│   │   ├── model_layer_comparison_xlsr.csv
│   │   ├── model_layer_comparison_wav2vec2.csv
│   │   ├── model_layer_comparison_wavlm.csv
│   │   └── full_model_comparison.csv
│   │
│   └── reports/
│       ├── baseline_summary_xlsr.md
│       ├── baseline_summary_wav2vec2.md
│       ├── baseline_summary_wavlm.md
│       └── full_baseline_model_comparison_summary.md
│
├── scripts/
│   ├── 00_prepare_project.py
│   ├── 01_create_dataset_index.py
│   ├── 02_extract_embeddings.py
│   ├── 03_visualize_embeddings.py
│   ├── 04_cross_language_classification.py
│   ├── 05_experiment_summary.py
│   ├── 06_run_full_baseline.py
│   ├── 07_package_deliverable.py
│   ├── 08_compare_all_models.py
│   └── 09_package_full_baseline_deliverable.py
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
> The `input/`, raw audio files, private metadata, dataset index, and large feature files should remain excluded from GitHub depending on privacy and file-size requirements.

---

## 17. How to Run the Project

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

### Step 7: Run the full baseline for each model

Run XLSR:

```bash
python scripts/06_run_full_baseline.py --model xlsr
```

Run Wav2Vec2:

```bash
python scripts/06_run_full_baseline.py --model wav2vec2
```

Run WavLM:

```bash
python scripts/06_run_full_baseline.py --model wavlm
```

### Step 8: Compare all models

```bash
python scripts/08_compare_all_models.py
```

Expected outputs:

```text
outputs/tables/full_model_comparison.csv
outputs/reports/full_baseline_model_comparison_summary.md
```

### Step 9: Package safe deliverables

```bash
python scripts/09_package_full_baseline_deliverable.py
```

Expected output:

```text
full_baseline_outputs_for_tomas/
full_baseline_outputs_for_tomas.zip
```

---

## 18. Running Scripts Individually

The pipeline can also be run step by step.

Create the dataset index:

```bash
python scripts/01_create_dataset_index.py
```

Extract embeddings:

```bash
python scripts/02_extract_embeddings.py --model xlsr --layers 0 4 8 11
python scripts/02_extract_embeddings.py --model wav2vec2 --layers 0 4 8 11
python scripts/02_extract_embeddings.py --model wavlm --layers 0 4 8 11
```

Generate PCA and t-SNE visualizations:

```bash
python scripts/03_visualize_embeddings.py --model xlsr
python scripts/03_visualize_embeddings.py --model wav2vec2
python scripts/03_visualize_embeddings.py --model wavlm
```

Run cross-language classification:

```bash
python scripts/04_cross_language_classification.py --model xlsr
python scripts/04_cross_language_classification.py --model wav2vec2
python scripts/04_cross_language_classification.py --model wavlm
```

Generate model-specific experiment summaries:

```bash
python scripts/05_experiment_summary.py --model xlsr
python scripts/05_experiment_summary.py --model wav2vec2
python scripts/05_experiment_summary.py --model wavlm
```

Compare all models:

```bash
python scripts/08_compare_all_models.py
```

Package final safe deliverable:

```bash
python scripts/09_package_full_baseline_deliverable.py
```

---

## 19. Output Files

### Reports

```text
outputs/reports/baseline_summary_xlsr.md
outputs/reports/baseline_summary_wav2vec2.md
outputs/reports/baseline_summary_wavlm.md
outputs/reports/full_baseline_model_comparison_summary.md
```

### Tables

```text
outputs/tables/full_model_comparison.csv
outputs/tables/model_layer_comparison_xlsr.csv
outputs/tables/model_layer_comparison_wav2vec2.csv
outputs/tables/model_layer_comparison_wavlm.csv
outputs/tables/classification_results_xlsr_readtext_layer0.csv
outputs/tables/classification_results_xlsr_readtext_layer4.csv
outputs/tables/classification_results_xlsr_readtext_layer8.csv
outputs/tables/classification_results_xlsr_readtext_layer11.csv
outputs/tables/classification_results_wav2vec2_readtext_layer0.csv
outputs/tables/classification_results_wav2vec2_readtext_layer4.csv
outputs/tables/classification_results_wav2vec2_readtext_layer8.csv
outputs/tables/classification_results_wav2vec2_readtext_layer11.csv
outputs/tables/classification_results_wavlm_readtext_layer0.csv
outputs/tables/classification_results_wavlm_readtext_layer4.csv
outputs/tables/classification_results_wavlm_readtext_layer8.csv
outputs/tables/classification_results_wavlm_readtext_layer11.csv
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

## 20. Deliverables Generated

The full baseline generated the following safe deliverables:

| Deliverable                                 | Purpose                                                                     |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| `full_baseline_model_comparison_summary.md` | Human-readable summary of the full XLSR, Wav2Vec2, and WavLM comparison     |
| `full_model_comparison.csv`                 | Complete result table across all models, layers, scenarios, and classifiers |
| `baseline_summary_xlsr.md`                  | Model-specific summary for XLSR                                             |
| `baseline_summary_wav2vec2.md`              | Model-specific summary for Wav2Vec2                                         |
| `baseline_summary_wavlm.md`                 | Model-specific summary for WavLM                                            |
| `classification_results_*.csv`              | Detailed layer-wise classification results                                  |
| `pca_*.png`                                 | PCA visualizations                                                          |
| `tsne_*.png`                                | t-SNE visualizations                                                        |
| `README.md`                                 | Explanation of package contents and privacy exclusions                      |
| `full_baseline_outputs_for_tomas.zip`       | Safe package prepared for supervisor review                                 |

The package intentionally excludes:

```text
raw audio
private metadata
input folder
dataset index with speaker IDs or local paths
raw feature embeddings
speaker-identifiable data
large derived feature files
```

---

## 21. Interpretation of the Full Baseline

The full baseline provides a clear and useful result for the thesis.

The model performs well within the same language:

```text
Spanish → Spanish: 0.8400 UAR using XLSR
German → German: 0.8182 UAR using WavLM
```

But cross-language transfer is weaker:

```text
Spanish → German: 0.7273 UAR using WavLM
German → Spanish: 0.6900 UAR using WavLM
```

This confirms that the model representations and classifiers are affected by language/domain differences.

However, the extended comparison also shows that WavLM improves cross-language transfer compared with the first XLSR-only baseline.

First XLSR-only transfer:

```text
Spanish → German: 0.6477 UAR
German → Spanish: 0.6700 UAR
```

Best full comparison transfer:

```text
Spanish → German: 0.7273 UAR
German → Spanish: 0.6900 UAR
```

This supports the next research step:

> Apply voice conversion or speech-domain adaptation and test whether cross-language PD classification improves while preserving disease-relevant cues.

---

## 22. Limitations

The current implementation has several limitations:

| Limitation                  | Explanation                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| **Read-text only**          | Only the read-text task was used in the first baseline.                                           |
| **No voice conversion yet** | The current stage is the baseline stage before voice conversion.                                  |
| **No fine-tuning yet**      | The speech models were used as fixed feature extractors.                                          |
| **Selected layers only**    | Only layers 0, 4, 8, and 11 were evaluated.                                                       |
| **Mean-pooled embeddings**  | Temporal information was summarized into fixed-length vectors.                                    |
| **Exploratory plots**       | PCA and t-SNE help interpretation but do not replace classification metrics.                      |
| **Setup dependency**        | Results depend on preprocessing, speaker split, random seed, and cross-validation setup.          |
| **Other tasks pending**     | Monologue, vowel, pataka/DDK, words, sentences, and full recordings remain for later experiments. |

---

## 23. Next Steps

The next planned steps are:

1. Review the full baseline comparison with **Dr.-Ing. Tomás Arias-Vergara** and align the next experimental direction based on his feedback.
2. Use the current read-text baseline as the reference point for future experiments.
3. Discuss whether WavLM should be treated as the main cross-lingual reference model before voice conversion.
4. Evaluate whether additional layers improve PD vs HC classification.
5. Analyze PCA and t-SNE plots together with classification metrics.
6. Extend the dataset tasks if needed, including monologue, vowel, pataka/DDK, sentences, words, and full recordings.
7. Start the voice conversion stage after the baseline is reviewed.
8. Compare PD classification before and after voice conversion.
9. Evaluate whether language mismatch is reduced while PD-relevant speech characteristics are preserved.

---

## 24. Future Extension: Voice Conversion

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

* articulation difficulty,
* reduced pitch variation,
* monotone voice,
* rhythm irregularity,
* breathiness,
* pause patterns,
* voice instability.

The final evaluation will compare classification performance before and after voice conversion.

---

## 25. Technical Skills Demonstrated

This project demonstrates skills in:

| Skill Area                  | Evidence in This Project                                              |
| --------------------------- | --------------------------------------------------------------------- |
| **Speech AI**               | Audio preprocessing, 16 kHz conversion, speech embeddings             |
| **Deep Learning**           | XLSR, Wav2Vec2, and WavLM representation extraction                   |
| **Machine Learning**        | SVM, Logistic Regression, cross-validation, UAR/AUC evaluation        |
| **Multilingual AI**         | Spanish-German transfer and language mismatch analysis                |
| **Biomedical AI**           | PD vs HC classification using medical speech data                     |
| **Research Engineering**    | Reproducible pipeline, structured results, safe deliverables          |
| **Data Privacy**            | Exclusion of raw medical data and private metadata                    |
| **Technical Communication** | Report writing, result interpretation, supervisor-ready documentation |

---

## 26. Professional Context

I am **Prosenjit Chowdhury**, an M.Sc. Artificial Intelligence student at **Friedrich-Alexander-Universität Erlangen-Nürnberg**.

Alongside my studies, I work as a **Working Student at SAP SE**. My professional experience includes work across three SAP areas:

| SAP Area                                                                        | Experience Focus                                                                                                                          |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **SAP LeanIX and SAP Signavio Content Marketing**                               | Digital content operations, publishing workflows, product communication, content coordination, and marketing content support              |
| **Professional Services and Engineering, Construction & Operations Industries** | Business content coordination, stakeholder communication, customer success stories, go-live stories, and industry-focused content support |
| **ERP PCX / Enterprise Systems and Process Automation**                         | Data validation, metadata quality, enterprise system support, Excel automation, VBA, and Power Automate                                   |

This thesis connects my academic focus in artificial intelligence with practical experience in enterprise technology, structured documentation, data-driven analysis, responsible AI project execution, and cross-functional collaboration.

---

## 27. Selected References

1. J. C. Vásquez-Correa et al., “Convolutional Neural Networks and a Transfer Learning Strategy to Classify Parkinson’s Disease from Speech in Three Different Languages,” Springer, 2019.

2. A. Hernandez et al., “Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria Detection in Parkinson’s Disease,” arXiv, 2026.

3. C.-J. Li et al., “Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages,” Interspeech, 2025.

4. A. Suppa et al., “Voice in Parkinson’s Disease: A Machine Learning Study,” Frontiers in Neurology, 2022.

5. Y. A. Li et al., “StyleTTS 2: Towards Human-Level Text-to-Speech,” NeurIPS, 2023.

---

## 28. Contact

<div align="center">

### 👤 Prosenjit Chowdhury

**M.Sc. Artificial Intelligence | FAU Erlangen-Nürnberg**
**Pattern Recognition Lab | Master’s Thesis Research**
**Working Student at SAP SE**

</div>

<br>

| Category             | Information                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Name**             | Prosenjit Chowdhury                                                                                                            |
| **Degree Program**   | M.Sc. Artificial Intelligence                                                                                                  |
| **University**       | Friedrich-Alexander-Universität Erlangen-Nürnberg                                                                              |
| **Research Lab**     | Pattern Recognition Lab, FAU Erlangen-Nürnberg                                                                                 |
| **Supervisor**       | Dr.-Ing. Tomás Arias-Vergara                                                                                                   |
| **GitHub**           | [github.com/prosenjit-chd](https://github.com/prosenjit-chd)                                                                   |
| **Research Project** | Voice Conversion for Multilingual Detection of Parkinson’s Disease Using Speech Signals                                        |
| **Project Focus**    | Speech AI, Parkinson’s Disease detection, multilingual speech processing, XLSR/Wav2Vec2/WavLM embeddings, and voice conversion |

---

## 💼 SAP Professional Experience

| Area                   | Details                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| **Current Role**       | Working Student at SAP SE                                                                                |
| **Professional Track** | Enterprise Systems, Data Operations, Business Transformation, Content Operations, and Process Automation |
| **SAP Area 1**         | SAP LeanIX and SAP Signavio Content Marketing                                                            |
| **SAP Area 2**         | Professional Services and Engineering, Construction & Operations Industries                              |
| **SAP Area 3**         | ERP PCX / Enterprise Systems and Process Automation                                                      |

---

## 🧩 Professional Focus Areas

<div align="center">

**Business & Data Analysis** • **Digital Transformation** • **Product Operations** • **Process Automation** • **AI Research** • **Speech Technology**

</div>

---

<div align="center">

### 🚀 Current Status

**Full first baseline model comparison completed.**
**XLSR, Wav2Vec2, and WavLM evaluated on Spanish-German read-text PD vs HC classification.**
**Next stage: supervisor review, deeper analysis, and preparation for voice conversion experiments.**

</div>
```
