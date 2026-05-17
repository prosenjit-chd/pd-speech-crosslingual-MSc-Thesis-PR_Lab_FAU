# Voice Conversion for Multilingual Detection of Parkinson’s Disease

**Author:** Prosenjit Chowdhury
**Program:** M.Sc. in Artificial Intelligence, Friedrich-Alexander University Erlangen-Nürnberg
**Research Lab:** Pattern Recognition Lab, FAU
**Supervisor:** Dr.-Ing. Tomás Arias-Vergara

## Thesis Overview
This project is part of a Master’s thesis investigating whether Voice Conversion (VC) can reduce language mismatch in multilingual Parkinson's Disease (PD) classification while preserving disease-related acoustic cues. 

Speech-based PD detection models often suffer from poor cross-language generalization. A model trained on one language (e.g., Spanish) tends to learn language-dependent phonetic patterns rather than universal pathology cues, leading to performance drops when tested on another language (e.g., German). The ultimate goal is to apply speech-domain transformation (VC) to align these acoustic domains.

## Current Phase: Baseline Implementation
Before applying Voice Conversion, we must establish a strong **baseline speech classification pipeline**. 

The current pipeline uses self-supervised speech representation models to extract layer-wise embeddings from Spanish and German speech recordings. We evaluate these embeddings within-language and cross-language to quantify the initial language mismatch.

### Self-Supervised Models Used
1. **XLSR (XLS-R Wav2Vec2)**: `facebook/wav2vec2-large-xlsr-53` (Primary focus due to multilingual pre-training)
2. **Wav2Vec2**: `facebook/wav2vec2-base`
3. **WavLM**: `microsoft/wavlm-base`

### Dataset Overview
We currently utilize the `readtext` task from two distinct datasets:
- **Spanish Dataset**: PC-GITA (Parkinson's Disease vs Healthy Controls)
- **German Dataset**: Sabine Skoda dataset (Parkinson's Disease vs Healthy Controls)

**Data Privacy Note:** The raw medical audio datasets (`.wav`, `.mp3`, etc.) and patient metadata (`.xlsx`) contain sensitive health information. **They are strictly excluded from this repository via `.gitignore`** and must remain local to the researcher's machine.

## Pipeline Architecture
1. **Audio Preprocessing:** Load audio, convert to mono, resample to 16 kHz.
2. **Feature Extraction:** Pass through the selected Hugging Face model and extract hidden states from specific layers (e.g., 0, 4, 8, 11). Apply mean-pooling over time to generate a fixed-size feature vector.
3. **Visualization:** Generate t-SNE and PCA plots to observe clustering by disease label, language, and combined group.
4. **Classification:** Train Linear SVM and Logistic Regression models.
5. **Evaluation:** Nested 10-fold cross-validation (9-fold inner for hyperparameter tuning) evaluated on Unweighted Average Recall (UAR/Balanced Accuracy), Sensitivity, Specificity, and AUC.

### Classification Scenarios
- **Spanish → Spanish**: Train and test within Spanish data.
- **Spanish → German**: Train on Spanish, test on German.
- **German → German**: Train and test within German data.
- **German → Spanish**: Train on German, test on Spanish.
- **Spanish + German → Spanish + German**: Mixed multilingual training and testing.

## How to Run the Code

1. Ensure your local `input/` directory is populated with the `Spanish/` and `German/` folders.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the complete baseline pipeline sequentially:
   ```bash
   python scripts/06_run_full_baseline.py
   ```

Alternatively, run individual scripts:
```bash
python scripts/01_create_dataset_index.py
python scripts/02_extract_embeddings.py --model xlsr --layers 0 4 8 11
python scripts/03_visualize_embeddings.py
python scripts/04_cross_language_classification.py
python scripts/05_experiment_summary.py
```

## Directory Structure
- `configs/`: Pipeline configuration variables (`baseline_config.yaml`).
- `input/`: **(Local Only)** Raw datasets and metadata.
- `metadata/`: Generated CSV tracking paths, tasks, and labels.
- `features/`: Extracted layer-wise speech embeddings (CSV format).
- `outputs/`: 
  - `figures/`: t-SNE and PCA plots.
  - `tables/`: Detailed classification results.
  - `reports/`: Markdown summary of best performing models.
- `scripts/`: Executable pipeline scripts.
- `src/`: Core logic modules (audio, extraction, classification, metrics, visualization).