````markdown
# Voice Conversion for Multilingual Detection of Parkinson’s Disease Using Speech Signals

**Author:** Prosenjit Chowdhury  
**Degree Program:** M.Sc. Artificial Intelligence  
**University:** Friedrich-Alexander-Universität Erlangen-Nürnberg  
**Research Lab:** Pattern Recognition Lab, FAU Erlangen-Nürnberg  
**Supervisor:** Dr.-Ing. Tomás Arias-Vergara  
**Project Repository:** `pd-speech-crosslingual-MSc-Thesis-PR_Lab_FAU`  
**GitHub:** https://github.com/prosenjit-chd  

---

## 1. Project Overview

This repository contains the implementation work for my Master’s thesis:

**Voice Conversion for Multilingual Detection of Parkinson’s Disease Using Speech Signals**

The thesis investigates whether speech-based artificial intelligence can detect Parkinson’s Disease (PD) from multilingual speech recordings and whether **voice conversion** can reduce language mismatch between different speech datasets.

The main research direction is to evaluate Parkinson’s Disease classification across different languages, especially Spanish and German, and later apply voice conversion or speech-domain transformation to improve cross-language generalization.

The long-term research question is:

> Can voice conversion reduce language mismatch in multilingual Parkinson’s Disease speech classification while preserving pathology-related acoustic information?

---

## 2. Research Motivation

Parkinson’s Disease can affect speech production. Symptoms may appear in phonation, articulation, prosody, loudness, pitch variation, rhythm, and pauses. These speech characteristics can be measured from audio and used as non-invasive digital biomarkers for computer-aided Parkinson’s Disease assessment.

However, speech is strongly language-dependent. A model trained on Spanish speech may learn Spanish-specific phonetic or acoustic patterns instead of disease-related speech characteristics. When the same model is tested on German speech, performance may drop because the language domain is different.

This project focuses on this core challenge:

> A Parkinson’s Disease classifier should learn disease-relevant speech patterns, not only language-specific patterns.

The first stage of the project therefore builds a strong baseline system before moving to voice conversion.

---

## 3. Current Project Phase

The current completed milestone is the:

# First Baseline Implementation

This stage does **not yet apply voice conversion**. Instead, it establishes a reproducible baseline for Parkinson’s Disease classification using Spanish and German read-text speech recordings.

The baseline answers these questions:

- Can XLSR speech embeddings support PD vs HC classification?
- Which XLSR layer gives the strongest baseline result?
- How much performance is lost when training on one language and testing on another?
- Do the embeddings separate more strongly by disease label or by language?
- Which baseline should be used as the reference point before voice conversion?

The first baseline was completed using the multilingual XLSR speech model and Spanish-German read-text speech data.

---

## 4. Thesis Goal

The main goal of this thesis is to investigate whether **voice conversion** can reduce language/domain mismatch in multilingual Parkinson’s Disease speech classification while preserving disease-related acoustic cues.

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
Voice conversion / domain transformation
        ↓
Re-evaluate PD classification
````

The final objective is not only to generate natural-sounding converted speech. The scientific goal is to test whether converted speech still preserves Parkinson’s-related information needed for classification.

---

## 5. Datasets

The current baseline uses Spanish and German Parkinson’s Disease speech recordings.

### Current baseline task

Only the **read-text speech task** is used in the first implementation.

This decision was made because read-text recordings are available in both Spanish and German datasets, which makes the first comparison more controlled and fair.

### Dataset summary for the first baseline

| Item               |                                 Value |
| ------------------ | ------------------------------------: |
| Task               |                        Read-text only |
| Total recordings   |                                   276 |
| Spanish recordings |                                   100 |
| German recordings  |                                   176 |
| PD recordings      |                                   138 |
| HC recordings      |                                   138 |
| Dataset index file | `metadata/dataset_index_readtext.csv` |

The Spanish and German read-text setup follows the planned first dataset decision: use comparable read-text recordings before expanding to other tasks such as monologue, vowel, pataka/DDK, words, sentences, or full recordings.

---

## 6. Data Privacy and Repository Policy

The speech data used in this project contains sensitive medical information. Therefore, raw data is **not included** in this repository.

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

* source code,
* configuration files,
* documentation,
* non-sensitive result summaries,
* non-identifying plots,
* safe result tables.

Raw audio files, clinical metadata, and private patient-related information are excluded through `.gitignore`.

---

## 7. AI and Machine Learning Approach

This project uses both **Deep Learning** and **Machine Learning**.

### Deep Learning part

Pretrained self-supervised speech models are used to convert raw audio into high-dimensional speech embeddings.

In the first baseline, the main model is:

```text
facebook/wav2vec2-large-xlsr-53
```

This is a pretrained multilingual XLSR wav2vec 2.0 model originally released by Facebook AI Research, now Meta AI. In this project, it is used as a **feature extractor**, not as a speech-to-text system.

### Machine Learning part

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

---

## 8. Model Used in the First Baseline

The completed first implementation uses:

```text
facebook/wav2vec2-large-xlsr-53
```

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

The first completed baseline uses only XLSR. The following models are planned for follow-up comparison:

```text
facebook/wav2vec2-base
microsoft/wavlm-base
```

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

### Step-by-step pipeline

1. **Dataset indexing**
   The pipeline scans the Spanish and German read-text folders and creates a structured dataset index.

2. **Audio preprocessing**
   Audio files are loaded, converted to mono, resampled to 16 kHz, and normalized.

3. **Feature extraction**
   The pretrained XLSR model extracts hidden states from selected layers.

4. **Embedding creation**
   Hidden states are mean-pooled over time to create one fixed-length vector per recording.

5. **Visualization**
   PCA and t-SNE plots are generated by:

   * disease label,
   * language,
   * combined group.

6. **Classification**
   Linear SVM and Logistic Regression classifiers are trained.

7. **Evaluation**
   The system evaluates within-language, cross-language, and multilingual scenarios.

8. **Reporting**
   Result tables, plots, and a markdown summary are generated.

---

## 11. Classification Scenarios

The first baseline evaluates five scenarios:

| Scenario                            | Purpose                                        |
| ----------------------------------- | ---------------------------------------------- |
| Spanish → Spanish                   | Within-language Spanish baseline               |
| Spanish → German                    | Cross-language transfer from Spanish to German |
| German → German                     | Within-language German baseline                |
| German → Spanish                    | Cross-language transfer from German to Spanish |
| Spanish + German → Spanish + German | Combined multilingual baseline                 |

These scenarios are designed to measure whether the classifier learns disease-relevant cues or language-dependent patterns.

---

## 12. Evaluation Metrics

The main metric is:

```text
UAR = Unweighted Average Recall
```

UAR is also commonly interpreted as balanced accuracy. It is useful because it treats both classes equally.

Additional metrics:

```text
Accuracy
Sensitivity
Specificity
AUC
```

Where:

```text
Sensitivity = ability to correctly detect PD
Specificity = ability to correctly detect HC
AUC = area under the ROC curve
```

---

## 13. Current Baseline Results

The first XLSR baseline has been completed successfully.

### Best result summary

| Result                 | Layer | Classifier          |    UAR | Accuracy | Sensitivity | Specificity |    AUC |
| ---------------------- | ----: | ------------------- | -----: | -------: | ----------: | ----------: | -----: |
| Best Spanish → Spanish |     4 | Linear SVM          | 0.8400 |   0.8400 |      0.8400 |      0.8400 | 0.8636 |
| Best German → German   |    11 | Logistic Regression | 0.7670 |   0.7670 |      0.7386 |      0.7955 | 0.8319 |
| Best Spanish → German  |     8 | Logistic Regression | 0.6477 |   0.6477 |      0.5568 |      0.7386 | 0.7136 |
| Best German → Spanish  |     8 | Logistic Regression | 0.6700 |   0.6700 |      0.7200 |      0.6200 | 0.7452 |
| Best Combined          |     8 | Logistic Regression | 0.7826 |   0.7826 |      0.7681 |      0.7971 | 0.8512 |

### Main finding

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

| Layer | Observation                         |
| ----: | ----------------------------------- |
|     0 | Useful acoustic-level baseline      |
|     4 | Best Spanish within-language result |
|     8 | Best cross-language transfer result |
|    11 | Best German within-language result  |

The strongest cross-language transfer results were obtained from **layer 8**, suggesting that mid-level XLSR representations may provide a better balance between acoustic detail and abstract speech structure.

---

## 15. Visualization Results

PCA and t-SNE plots were generated for every evaluated XLSR layer.

Each layer was visualized in three ways:

```text
by label
by language
by combined group
```

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

The first baseline milestone is completed.

Completed tasks:

* Created project folder structure.
* Created dataset index for Spanish and German read-text data.
* Processed 276 recordings.
* Extracted XLSR embeddings from layers 0, 4, 8, and 11.
* Generated PCA and t-SNE plots.
* Ran Linear SVM and Logistic Regression classifiers.
* Evaluated all five Spanish-German scenarios.
* Generated result tables.
* Generated baseline summary report.
* Created a safe deliverable package for supervisor review.
* Excluded raw audio, private metadata, and sensitive files.

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

Note:

```text
input/
features/
raw audio files
private metadata
```

may be excluded from GitHub depending on privacy and file-size requirements.

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

Activate it on Linux/macOS:

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

```text
dataset indexing
embedding extraction
visualization
classification
summary generation
```

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

```bash
python scripts/01_create_dataset_index.py
```

```bash
python scripts/02_extract_embeddings.py --model xlsr --layers 0 4 8 11
```

```bash
python scripts/03_visualize_embeddings.py
```

```bash
python scripts/04_cross_language_classification.py
```

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

| Deliverable                                       | Purpose                                                               |
| ------------------------------------------------- | --------------------------------------------------------------------- |
| `baseline_summary.md`                             | Human-readable summary of setup, dataset, results, and interpretation |
| `model_layer_comparison.csv`                      | Aggregated comparison across layers, classifiers, and scenarios       |
| `classification_results_xlsr_readtext_layer*.csv` | Detailed layer-wise classification results                            |
| `pca_*.png`                                       | PCA visualizations                                                    |
| `tsne_*.png`                                      | t-SNE visualizations                                                  |
| `README.md`                                       | Explanation of package contents and privacy exclusions                |
| `baseline_outputs_for_tomas.zip`                  | Safe package prepared for supervisor review                           |

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

1. Only the read-text task was used in the first baseline.
2. Only one pretrained speech model, XLSR, has been evaluated so far.
3. The model was used as a fixed feature extractor and was not fine-tuned.
4. PCA and t-SNE are exploratory visualizations and should not replace classification metrics.
5. Results depend on preprocessing, split strategy, random seed, and cross-validation setup.
6. Other speech tasks such as monologue, vowel, pataka/DDK, words, sentences, and full recordings remain for future experiments.

---

## 24. Next Steps

The next planned steps are:

1. Review the first baseline results with Dr.-Ing. Tomás Arias-Vergara and align the next experimental direction based on his feedback.
2. Use the current read-text XLSR baseline as the initial reference point for future experiments.
3. Run the same baseline pipeline with Wav2Vec2 and WavLM.
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

```text
articulation difficulty
reduced pitch variation
monotone voice
rhythm irregularity
breathiness
pause patterns
voice instability
```

The final evaluation will compare classification performance before and after voice conversion.

---

## 26. Technical Skills Demonstrated

This project demonstrates skills in:

| Skill Area              | Evidence in This Project                                              |
| ----------------------- | --------------------------------------------------------------------- |
| Speech AI               | Audio preprocessing, 16 kHz conversion, speech embeddings             |
| Deep Learning           | XLSR/wav2vec2 representation extraction                               |
| Machine Learning        | SVM, Logistic Regression, cross-validation, UAR/AUC evaluation        |
| Multilingual AI         | Spanish-German transfer and language mismatch analysis                |
| Biomedical AI           | PD vs HC classification using medical speech data                     |
| Research Engineering    | Reproducible pipeline, structured results, safe deliverables          |
| Data Privacy            | Exclusion of raw medical data and private metadata                    |
| Technical Communication | Report writing, result interpretation, supervisor-ready documentation |

---

## 27. Professional Context

I am Prosenjit Chowdhury, an M.Sc. Artificial Intelligence student at Friedrich-Alexander-Universität Erlangen-Nürnberg.

Alongside my studies, I work as a Working Student at SAP SE. My professional experience includes work across SAP LeanIX and SAP Signavio Content Marketing, Professional Services and Engineering, Construction & Operations industries, and ERP PCX.

This thesis connects my academic focus in artificial intelligence with practical experience in enterprise technology, structured documentation, data-driven analysis, and responsible AI project execution.

---

## 28. Selected References

1. J. C. Vásquez-Correa et al., “Convolutional Neural Networks and a Transfer Learning Strategy to Classify Parkinson’s Disease from Speech in Three Different Languages,” Springer, 2019.

2. A. Hernandez et al., “Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria Detection in Parkinson’s Disease,” arXiv, 2026.

3. C.-J. Li et al., “Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages,” Interspeech, 2025.

4. A. Suppa et al., “Voice in Parkinson’s Disease: A Machine Learning Study,” Frontiers in Neurology, 2022.

5. Y. A. Li et al., “StyleTTS 2: Towards Human-Level Text-to-Speech,” NeurIPS, 2023.

---

## 29. Contact

**Name:** Prosenjit Chowdhury  
**Degree Program:** M.Sc. Artificial Intelligence  
**University:** Friedrich-Alexander-Universität Erlangen-Nürnberg  
**Research Lab:** Pattern Recognition Lab, FAU Erlangen-Nürnberg  
**Supervisor:** Dr.-Ing. Tomás Arias-Vergara  

**Project:** Voice Conversion for Multilingual Detection of Parkinson’s Disease Using Speech Signals  
**Project Focus:** Speech AI, Parkinson’s Disease detection, multilingual speech processing, XLSR/Wav2Vec2 embeddings, and voice conversion  

**GitHub:** [https://github.com/prosenjit-chd](https://github.com/prosenjit-chd)  

**Professional Experience** Enterprise Systems, Data Operations & Business Transformation  
**Professional Context:**  
Working Student at SAP-SE with experience across three (3) SAP core departments:  
- SAP LeanIX and SAP Signavio Content Marketing  
- Professional Services and Engineering, Construction & Operations Industries  
- ERP PCX / Enterprise Systems and Process Automation  

**Professional Focus:**  
Business & Data Analysis | Digital Transformation | Product Operations | Process Automation | AI Research

```


