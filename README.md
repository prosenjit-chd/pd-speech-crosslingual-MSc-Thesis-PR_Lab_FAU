Yes Prosenjit, I understood the style from your previous README. It starts with a clear project title, explains the objective, dataset, workflow, feature engineering, models, evaluation, project structure, running steps, and author profile in a professional way. 

Below is a **full README.md** for your thesis repository. You can copy and paste this directly into GitHub.

````markdown
# 🧠 Multilingual Parkinson’s Disease Detection from Speech using Self-Supervised Speech Embeddings

This repository contains the implementation plan and baseline pipeline for my master’s thesis at the **FAU Pattern Recognition Lab** under the supervision of **Dr.-Ing. Tomás Arias-Vergara**.

The project focuses on **Parkinson’s disease detection from multilingual speech signals**, using modern self-supervised speech models such as **Wav2Vec2, XLSR, and WavLM**.

The main research direction is to investigate whether speech representations can support reliable **Parkinson’s disease (PD) vs healthy control (HC)** classification across different languages, especially **Spanish and German**.

Although this project is developed in the biomedical AI domain, the methods used here are highly relevant to many real-world AI applications, including:

- speech-based health monitoring
- multilingual speech intelligence
- self-supervised learning
- medical AI systems
- cross-lingual machine learning
- voice biomarker analysis
- audio representation learning
- responsible AI for healthcare

---

# 🎯 Thesis Objective

The main goal of this thesis is to investigate whether **multilingual speech representations** can support robust Parkinson’s disease detection across languages.

In practical terms, this means answering the following question:

> Can a model trained on Parkinson’s speech from one language, such as Spanish, generalize to another language, such as German?

This is important because many speech-based disease detection systems perform well within one language, but their performance drops when tested on another language.

The reason is that the model may learn:

- language-specific pronunciation patterns
- recording condition differences
- speaker-specific characteristics
- dataset-specific bias

instead of learning the true disease-related acoustic patterns.

This thesis therefore studies whether modern pretrained speech models can extract useful disease-related embeddings from speech while reducing language dependency.

---

# 🧪 Research Motivation

Parkinson’s disease affects motor control and often influences speech production. Patients may show changes in:

- articulation
- rhythm
- loudness
- pitch variation
- voice quality
- speech rate
- syllable repetition
- phonation stability

These speech changes can be used as acoustic biomarkers for automatic PD detection.

However, speech varies strongly across languages. A Spanish speaker and a German speaker may naturally differ in rhythm, phoneme distribution, prosody, and pronunciation. This creates a major challenge for machine learning models.

A classifier trained only on Spanish speech may not perform well on German speech because it may learn Spanish-specific speech structure instead of pathology-specific cues.

This project begins by building a strong **baseline classification system** before applying more advanced methods such as **voice conversion** or cross-lingual speech transformation.

---

# 🔬 Core Research Question

The central research question is:

> Can self-supervised speech embeddings extracted from multilingual speech models support Parkinson’s disease classification across Spanish and German speech datasets?

A second long-term research question is:

> Can voice conversion or speech-domain transformation reduce language mismatch while preserving Parkinson’s disease-related acoustic information?

---

# 🗣️ Important Clarification: What Voice Conversion Means Here

In this thesis, voice conversion does **not** mean that a patient needs to speak many languages.

A German patient speaks German.  
A Spanish patient speaks Spanish.  
A Czech patient speaks Czech.

The idea is not to ask the same patient to speak Spanish, German, and Czech.

Instead, voice conversion is considered as a computational transformation step.

For example:

```text
German speech
   ↓
Speech-domain transformation / voice conversion
   ↓
Speech representation closer to Spanish domain
   ↓
Spanish-trained classifier
   ↓
PD vs HC prediction
````

The scientific question is whether such transformation can reduce language mismatch without destroying disease-related acoustic cues.

For the current first phase of the thesis, the focus is **not yet voice conversion**.

The first phase is:

```text
Speech audio
   ↓
Self-supervised speech embeddings
   ↓
t-SNE visualization
   ↓
PD vs HC classifier
   ↓
Spanish-German transfer evaluation
```

---

# 📊 Datasets

This project uses multilingual Parkinson’s disease speech datasets provided through the FAU Pattern Recognition Lab.

The current experimental focus is on:

| Language | Dataset Type                                                | Classes  |
| -------- | ----------------------------------------------------------- | -------- |
| Spanish  | Read text and monologue speech                              | PD vs HC |
| German   | Read text, sentences, words, vowel, pataka, full recordings | PD vs HC |

The first baseline experiment will use **read text speech** from Spanish and German datasets because both languages contain comparable read speech tasks.

---

# 📁 German Dataset Structure

The German dataset contains multiple speech tasks.

The observed folder structure is:

```text
Parkinson_German_Sabine_Skoda/
└── Parkinson_German/
    ├── Description-Deutsch-TobiasSubSet.xlsx
    ├── info.txt
    ├── IS2013.pdf
    ├── Protokoll_Sprechanalyse.doc
    ├── Protokoll_Sprechanalyse.pdf
    └── Recordings/
        ├── pataka/
        ├── vowel/
        ├── sentences/
        ├── words/
        ├── full/
        └── readtext/
```

The German speech protocol includes several tasks such as:

* spontaneous speech
* read text
* question-answer reading
* read sentences
* read words
* sustained vowel
* repeated vowel
* repeated syllables such as pa and ba
* pataka repetition
* complex syllable sequences

---

# 📁 Spanish Dataset Structure

The Spanish dataset currently includes:

```text
Spanish/
├── readtext_PDvsHC/
│   ├── PD/
│   └── HC/
│
└── PD_vs_HC_Monologue/
    ├── PD/
    └── HC/
```

For the first implementation phase, the project will start with:

```text
Spanish read text
German read text
```

This ensures that the speech task is comparable across languages.

---

# ⚙️ Project Workflow

The baseline pipeline follows this structure:

```text
Raw speech audio
   ↓
Audio preprocessing
   ↓
Self-supervised speech model
   ↓
Layer-wise embedding extraction
   ↓
Feature pooling
   ↓
Feature visualization
   ↓
PD vs HC classification
   ↓
Cross-language evaluation
```

---

# 🧩 Why Self-Supervised Speech Models?

Traditional speech analysis often uses handcrafted features such as:

* MFCC
* jitter
* shimmer
* pitch
* energy
* prosodic features
* openSMILE features

This thesis instead begins with modern pretrained speech models.

The planned models are:

| Model    | Purpose                                                           |
| -------- | ----------------------------------------------------------------- |
| Wav2Vec2 | General self-supervised speech representation learning            |
| XLSR     | Multilingual Wav2Vec2 model suitable for cross-lingual speech     |
| WavLM    | Speech representation model designed for robust speech processing |

These models transform raw audio into high-dimensional embeddings.

The extracted embeddings are then used as input features for machine learning classifiers.

---

# 🧠 Layer-wise Speech Embedding Extraction

Models such as Wav2Vec2 and XLSR contain multiple transformer layers.

For example, a 12-layer model produces hidden representations from layer 0 to layer 11.

Different layers may capture different types of information:

| Layer Type    | Possible Information                                |
| ------------- | --------------------------------------------------- |
| Early layers  | acoustic and phonetic structure                     |
| Middle layers | speaker and speech pattern information              |
| Later layers  | more abstract linguistic and contextual information |

For this reason, the project will compare embeddings from multiple layers.

Initial layer selection:

```text
Layer 0
Layer 4
Layer 8
Layer 11
```

Later experiments may evaluate all layers from 0 to 11.

---

# 📌 Feature Extraction Plan

For each audio file:

1. Load audio
2. Convert to mono
3. Resample to 16 kHz
4. Pass the waveform through a pretrained speech model
5. Extract hidden states from selected layers
6. Apply mean pooling over time
7. Save one fixed-size embedding vector per audio file

Example output:

```text
features/
├── xlsr_readtext_layer0.csv
├── xlsr_readtext_layer4.csv
├── xlsr_readtext_layer8.csv
└── xlsr_readtext_layer11.csv
```

Each feature file will contain:

```text
speaker_id, language, task, label, feature_0, feature_1, ..., feature_n
```

---

# 🧾 Dataset Index

A central dataset index will be created before feature extraction.

Example:

```text
metadata/
└── dataset_index_readtext.csv
```

Example columns:

| Column      | Description                        |
| ----------- | ---------------------------------- |
| file_path   | Path to audio file                 |
| speaker_id  | Anonymized speaker identifier      |
| language    | Spanish or German                  |
| task        | Read text, monologue, pataka, etc. |
| label       | PD or HC                           |
| dataset     | Spanish or German dataset source   |
| split_group | Speaker-level split identifier     |

Speaker-level splitting is important to avoid data leakage.

The same speaker must not appear in both training and testing sets.

---

# 📉 Visualization

The project will use dimensionality reduction to understand the structure of the extracted embedding space.

Planned visualization methods:

* t-SNE
* PCA
* UMAP, optional

The most important first visualization is t-SNE.

Three types of t-SNE plots will be generated:

## 1. Disease Label Visualization

```text
PD vs HC
```

This checks whether the embedding space separates Parkinson’s disease patients from healthy controls.

## 2. Language Visualization

```text
Spanish vs German
```

This checks whether the embedding space is dominated by language differences.

## 3. Combined Group Visualization

```text
Spanish PD
Spanish HC
German PD
German HC
```

This helps analyze whether the model learns disease-related information or mainly language-specific information.

---

# 🤖 Classification Task

The main classification task is:

```text
Parkinson’s disease vs healthy control
```

or:

```text
PD vs HC
```

Initial classifiers:

| Classifier          | Reason                                          |
| ------------------- | ----------------------------------------------- |
| Linear SVM          | Strong baseline for high-dimensional embeddings |
| Logistic Regression | Interpretable and efficient baseline            |
| Random Forest       | Nonlinear baseline                              |
| Extra Trees         | Ensemble baseline                               |
| XGBoost or CatBoost | Optional advanced model                         |

The first implementation will prioritize:

```text
Linear SVM
Logistic Regression
```

---

# 🌍 Classification Scenarios

The thesis baseline will evaluate five main classification scenarios.

| Train            | Test             | Purpose                                        |
| ---------------- | ---------------- | ---------------------------------------------- |
| Spanish          | Spanish          | Within-language Spanish baseline               |
| Spanish          | German           | Cross-language transfer from Spanish to German |
| German           | German           | Within-language German baseline                |
| German           | Spanish          | Cross-language transfer from German to Spanish |
| Spanish + German | Spanish + German | Multilingual mixed baseline                    |

These scenarios are the core of the first experimental phase.

They will show whether the model generalizes across languages or suffers from language mismatch.

---

# 📊 Evaluation Metrics

Models will be evaluated using:

| Metric                  | Description                                  |
| ----------------------- | -------------------------------------------- |
| UAR / Balanced Accuracy | Main metric for balanced PD vs HC evaluation |
| Accuracy                | Overall classification correctness           |
| Sensitivity             | Correct detection rate for PD patients       |
| Specificity             | Correct detection rate for healthy controls  |
| AUC                     | Area under ROC curve                         |
| Confusion Matrix        | Error analysis between PD and HC             |

The primary metric will be:

```text
UAR / Balanced Accuracy
```

This is important because biomedical datasets may have class imbalance.

---

# 🔁 Cross-Validation Strategy

The planned evaluation follows Tomás’s recommendation:

```text
Nested 10-fold cross-validation
with internal 9-fold cross-validation
```

For within-language experiments:

```text
Outer 10-fold CV
   ↓
Inner 9-fold CV for hyperparameter tuning
   ↓
Final evaluation on held-out fold
```

For cross-language experiments:

```text
Train on source language
   ↓
Tune hyperparameters using internal CV on source language
   ↓
Test on target language
```

Example:

```text
Train: Spanish
Tune: Spanish internal CV
Test: German
```

This setup ensures that the German test set remains unseen during model selection.

---

# 🏗️ Proposed Project Structure

```text
pd-speech-crosslingual/
│
├── data/
│   ├── spanish/
│   │   ├── readtext/
│   │   └── monologue/
│   │
│   └── german/
│       └── Parkinson_German/
│           └── Recordings/
│               ├── readtext/
│               ├── sentences/
│               ├── words/
│               ├── vowel/
│               ├── pataka/
│               └── full/
│
├── metadata/
│   ├── dataset_index_readtext.csv
│   ├── spanish_metadata.csv
│   └── german_metadata.csv
│
├── scripts/
│   ├── 01_create_dataset_index.py
│   ├── 02_audio_preprocessing.py
│   ├── 03_extract_embeddings.py
│   ├── 04_visualize_embeddings.py
│   ├── 05_train_classifiers.py
│   ├── 06_cross_language_evaluation.py
│   └── 07_generate_report_tables.py
│
├── src/
│   ├── audio_loader.py
│   ├── embedding_extractor.py
│   ├── dataset_builder.py
│   ├── visualization.py
│   ├── classification.py
│   ├── metrics.py
│   └── utils.py
│
├── features/
│   ├── xlsr/
│   ├── wav2vec2/
│   └── wavlm/
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── reports/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_embedding_visualization.ipynb
│   └── 03_result_analysis.ipynb
│
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

---

# 🚀 Running the Pipeline

## 1. Create a virtual environment

```bash
python -m venv .venv
```

## 2. Activate environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Create dataset index

```bash
python scripts/01_create_dataset_index.py
```

## 5. Extract speech embeddings

Example:

```bash
python scripts/03_extract_embeddings.py \
  --model xlsr \
  --layers 0 4 8 11 \
  --task readtext
```

## 6. Generate t-SNE plots

```bash
python scripts/04_visualize_embeddings.py \
  --features features/xlsr/xlsr_readtext_layer4.csv
```

## 7. Run classification

```bash
python scripts/05_train_classifiers.py \
  --features features/xlsr/xlsr_readtext_layer4.csv \
  --classifier svm
```

## 8. Run cross-language evaluation

```bash
python scripts/06_cross_language_evaluation.py \
  --features features/xlsr/xlsr_readtext_layer4.csv
```

---

# 📦 Planned Dependencies

Main Python libraries:

```text
numpy
pandas
scikit-learn
torch
torchaudio
transformers
librosa
soundfile
matplotlib
seaborn
umap-learn
tqdm
openpyxl
```

Optional:

```text
catboost
xgboost
plotly
```

---

# 📊 Expected Outputs

The pipeline will generate:

## Feature files

```text
features/xlsr/xlsr_readtext_layer0.csv
features/xlsr/xlsr_readtext_layer4.csv
features/xlsr/xlsr_readtext_layer8.csv
features/xlsr/xlsr_readtext_layer11.csv
```

## Visualization outputs

```text
outputs/figures/tsne_xlsr_layer0_label.png
outputs/figures/tsne_xlsr_layer4_language.png
outputs/figures/tsne_xlsr_layer11_combined_groups.png
```

## Classification tables

```text
outputs/tables/classification_results_readtext.csv
outputs/tables/cross_language_results.csv
outputs/tables/layer_comparison_results.csv
```

## Final report artifacts

```text
outputs/reports/baseline_summary.md
outputs/reports/thesis_experiment_log.md
```

---

# 📈 Planned Result Table

The first result table will follow this format:

| Model | Layer | Train            | Test             | UAR | Accuracy | Sensitivity | Specificity | AUC |
| ----- | ----- | ---------------- | ---------------- | --- | -------- | ----------- | ----------- | --- |
| XLSR  | 0     | Spanish          | Spanish          | TBD | TBD      | TBD         | TBD         | TBD |
| XLSR  | 0     | Spanish          | German           | TBD | TBD      | TBD         | TBD         | TBD |
| XLSR  | 0     | German           | German           | TBD | TBD      | TBD         | TBD         | TBD |
| XLSR  | 0     | German           | Spanish          | TBD | TBD      | TBD         | TBD         | TBD |
| XLSR  | 4     | Spanish          | German           | TBD | TBD      | TBD         | TBD         | TBD |
| XLSR  | 8     | German           | Spanish          | TBD | TBD      | TBD         | TBD         | TBD |
| XLSR  | 11    | Spanish + German | Spanish + German | TBD | TBD      | TBD         | TBD         | TBD |

---

# 🧪 First Experimental Milestone

The first milestone is to complete a clean baseline experiment using:

```text
Spanish readtext
German readtext
XLSR embeddings
Layer 0, 4, 8, 11
Linear SVM classifier
t-SNE visualization
Spanish-German transfer evaluation
```

The goal of this milestone is to answer:

1. Do Spanish and German embeddings form separate clusters?
2. Do PD and HC speakers show separable patterns?
3. Which XLSR layer gives the best PD vs HC classification?
4. Does Spanish to German transfer work?
5. Does German to Spanish transfer work?
6. Is the model learning disease information or mainly language information?

---

# 🧬 Future Extension: Voice Conversion

After the baseline experiments, the thesis may continue toward voice conversion or speech-domain transformation.

The future pipeline may look like this:

```text
German speech
   ↓
Voice conversion toward Spanish speech domain
   ↓
Feature extraction
   ↓
Spanish-trained PD classifier
   ↓
PD vs HC prediction
```

The goal will be to evaluate whether voice conversion can reduce language mismatch while preserving disease-related acoustic cues.

The key scientific challenge is:

```text
Reduce language mismatch
without removing Parkinson’s-related speech information
```

If the converted speech becomes too natural or too clean, it may remove important disease cues such as articulation difficulty, reduced pitch variation, tremor-like instability, or rhythm irregularity.

Therefore, the success of voice conversion will not be judged only by speech quality.

It will be judged by whether PD classification performance is preserved or improved after conversion.

---

# 🧠 Technical Skills Demonstrated

This project demonstrates applied AI and machine learning skills in several areas:

## Speech AI

* raw audio processing
* speech signal preprocessing
* speech embedding extraction
* self-supervised speech models
* multilingual speech representation learning

## Machine Learning

* supervised classification
* cross-validation
* nested hyperparameter optimization
* cross-domain evaluation
* model comparison
* performance analysis

## Biomedical AI

* Parkinson’s disease detection
* voice biomarker analysis
* patient vs control classification
* responsible handling of medical speech data

## Data Engineering

* dataset indexing
* metadata integration
* reproducible pipeline design
* structured experiment outputs
* scalable project organization

## Research Engineering

* baseline design
* experiment tracking
* result interpretation
* visualization-driven analysis
* thesis-oriented documentation

---

# 🧭 Why This Project Matters

Speech is one of the most natural and accessible human signals.

If speech-based Parkinson’s detection becomes reliable across languages, it could support:

* early screening
* remote health monitoring
* low-cost digital biomarkers
* multilingual clinical AI tools
* accessible healthcare technology

This project contributes to that direction by studying the cross-language robustness of speech-based PD detection systems.

---

# ⚠️ Data Privacy Notice

The medical speech datasets used in this thesis are provided for research purposes.

The datasets are not included in this public repository.

This repository will contain only:

* source code
* experiment configuration
* documentation
* anonymized result tables
* non-sensitive plots

No patient-identifiable audio data or private metadata will be published.

---

# 👨‍💻 Author

**Prosenjit Chowdhury**

M.Sc. Artificial Intelligence
Friedrich-Alexander University Erlangen-Nürnberg, Germany

Master’s Thesis
Pattern Recognition Lab
FAU Erlangen-Nürnberg

Working Student
SAP SE, Germany

GitHub:
[https://github.com/prosenjit-chd](https://github.com/prosenjit-chd)

---

# 📜 Supervisor

**Dr.-Ing. Tomás Arias-Vergara**
Pattern Recognition Lab
Lehrstuhl für Informatik 5
Friedrich-Alexander-Universität Erlangen-Nürnberg

---

# 📌 Current Status

Current phase:

```text
Baseline implementation phase
```

Current focus:

```text
Spanish-German readtext classification
using XLSR / Wav2Vec / WavLM embeddings
```

Next immediate steps:

1. Build dataset index
2. Extract XLSR embeddings
3. Generate t-SNE plots
4. Train baseline classifier
5. Evaluate Spanish-German transfer scenarios
6. Prepare first result table for thesis discussion

---

# ✅ Proposed Implementation Plan

## Phase 1: Dataset Preparation

Goal:

Create a clean and reproducible dataset index for Spanish and German speech files.

Tasks:

1. Organize Spanish readtext data
2. Organize German readtext data
3. Extract labels from folder structure
4. Map speaker IDs
5. Create `dataset_index_readtext.csv`
6. Verify number of PD and HC samples
7. Ensure speaker-level splitting is possible

Expected output:

```text
metadata/dataset_index_readtext.csv
```

---

## Phase 2: Audio Preprocessing

Goal:

Prepare all audio files for self-supervised speech models.

Tasks:

1. Load each audio file
2. Convert stereo to mono
3. Resample audio to 16 kHz
4. Normalize waveform amplitude
5. Handle corrupted or unreadable files
6. Save preprocessing logs

Expected output:

```text
outputs/reports/audio_preprocessing_log.csv
```

---

## Phase 3: Embedding Extraction

Goal:

Extract layer-wise embeddings from pretrained speech models.

Initial model:

```text
XLSR
```

Selected layers:

```text
0, 4, 8, 11
```

Tasks:

1. Load pretrained XLSR model
2. Feed each audio file into the model
3. Extract hidden states
4. Apply mean pooling over time
5. Save one vector per audio file
6. Repeat for selected layers

Expected output:

```text
features/xlsr/xlsr_readtext_layer0.csv
features/xlsr/xlsr_readtext_layer4.csv
features/xlsr/xlsr_readtext_layer8.csv
features/xlsr/xlsr_readtext_layer11.csv
```

---

## Phase 4: Embedding Visualization

Goal:

Understand whether embeddings capture disease information or language information.

Tasks:

1. Run t-SNE on extracted embeddings
2. Plot embeddings by PD/HC label
3. Plot embeddings by language
4. Plot embeddings by combined group
5. Compare layer-wise visual patterns

Expected output:

```text
outputs/figures/tsne_label_layer4.png
outputs/figures/tsne_language_layer4.png
outputs/figures/tsne_combined_layer4.png
```

---

## Phase 5: Baseline Classification

Goal:

Train machine learning classifiers for PD vs HC detection.

Initial classifiers:

```text
Linear SVM
Logistic Regression
```

Tasks:

1. Load embedding features
2. Standardize feature vectors
3. Apply speaker-level cross-validation
4. Train classifier
5. Tune hyperparameters using inner CV
6. Evaluate using UAR, accuracy, sensitivity, specificity, and AUC

Expected output:

```text
outputs/tables/baseline_classification_results.csv
```

---

## Phase 6: Cross-Language Evaluation

Goal:

Evaluate how well models transfer between Spanish and German speech.

Scenarios:

```text
Spanish → Spanish
Spanish → German
German → German
German → Spanish
Spanish + German → Spanish + German
```

Tasks:

1. Train on source language
2. Tune hyperparameters only on source language
3. Test on target language
4. Compare within-language and cross-language performance
5. Identify performance drops caused by language mismatch

Expected output:

```text
outputs/tables/cross_language_transfer_results.csv
```

---

## Phase 7: Model and Layer Comparison

Goal:

Find which speech model and layer are most useful for PD detection.

Models:

```text
XLSR
Wav2Vec2
WavLM
```

Tasks:

1. Repeat embedding extraction for each model
2. Compare selected layers
3. Compare classification performance
4. Compare t-SNE separability
5. Identify best model-layer combination

Expected output:

```text
outputs/tables/model_layer_comparison.csv
```

---

## Phase 8: Thesis Interpretation

Goal:

Connect experimental results to the thesis research question.

Main questions:

1. Does within-language classification perform better than cross-language classification?
2. Which language transfer direction is harder?
3. Are embeddings more language-dependent or disease-dependent?
4. Which model provides the most robust multilingual representation?
5. Is there a clear need for voice conversion or domain transformation?

Expected output:

```text
outputs/reports/baseline_experiment_summary.md
```

---

## Phase 9: Future Voice Conversion Experiment

Goal:

Evaluate whether speech-domain transformation can reduce cross-language mismatch.

Possible future workflow:

```text
German speech
   ↓
Voice conversion toward Spanish domain
   ↓
Embedding extraction
   ↓
Spanish-trained classifier
   ↓
PD vs HC prediction
```

Evaluation:

1. Compare original German test performance
2. Compare converted German test performance
3. Check whether PD cues are preserved
4. Analyze whether UAR improves or drops
5. Interpret whether voice conversion is useful for multilingual PD detection

---

# 🏁 Final Project Goal

The final goal of this thesis is to build a complete AI research pipeline that can:

1. Process multilingual Parkinson’s speech datasets
2. Extract modern self-supervised speech embeddings
3. Visualize disease and language patterns
4. Train baseline PD vs HC classifiers
5. Evaluate Spanish-German cross-language transfer
6. Identify language mismatch problems
7. Prepare the foundation for voice conversion-based improvement

This project combines speech AI, biomedical machine learning, multilingual representation learning, and responsible research engineering.

````

My suggested GitHub repository name for this README:

```text
pd-speech-crosslingual
````

And use this short GitHub description:

```text
Multilingual Parkinson’s disease detection from speech using self-supervised speech embeddings and cross-language classification.
```
