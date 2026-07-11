#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_reconstructed_full.py

Purpose:
Stage 4 evaluation script:
1. Generates metadata indices for both original and reconstructed full datasets:
   - C:\pd-speech-crosslingual\voice_conversion\metadata_full\dataset_index_original_full.csv
   - C:\pd-speech-crosslingual\voice_conversion\metadata_full\dataset_index_generated_full.csv
2. Reuses baseline FeatureExtractor to extract WavLM, Wav2Vec2, and XLSR features for layers [0, 4, 8, 11]
   and saves them under separate feature folders:
   - C:\pd-speech-crosslingual\voice_conversion\features_original_full\
   - C:\pd-speech-crosslingual\voice_conversion\features_generated_full\
3. Runs baseline classification scenarios (UAR, Accuracy, Sensitivity, Specificity, AUC) on both sets
   and saves results under separate output folders:
   - C:\pd-speech-crosslingual\voice_conversion\outputs_original_full\
   - C:\pd-speech-crosslingual\voice_conversion\outputs_generated_full\
4. Generates a comparative analysis CSV and markdown report comparing original full vs generated full:
   - C:\pd-speech-crosslingual\voice_conversion\logs_full\full_evaluation_comparison_summary.csv
   - C:\pd-speech-crosslingual\voice_conversion\logs_full\hifigan_stage4_evaluation_report.md
"""

import os
import sys
import csv
import logging
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config, setup_logging, ensure_dir, get_device, set_seed
from src.audio_utils import load_and_preprocess_audio
from src.embedding_extractor import FeatureExtractor
from src.classification import run_classification_scenarios

# Directory Constants
METADATA_DIR = PROJECT_ROOT / "voice_conversion" / "metadata_full"
FEATURES_ORIG_DIR = PROJECT_ROOT / "voice_conversion" / "features_original_full"
FEATURES_GEN_DIR = PROJECT_ROOT / "voice_conversion" / "features_generated_full"
OUTPUTS_ORIG_DIR = PROJECT_ROOT / "voice_conversion" / "outputs_original_full"
OUTPUTS_GEN_DIR = PROJECT_ROOT / "voice_conversion" / "outputs_generated_full"
LOG_DIR = PROJECT_ROOT / "voice_conversion" / "logs_full"

SELECTION_LOG_PATH = LOG_DIR / "full_selection_log.csv"
COMPARISON_SUMMARY_PATH = LOG_DIR / "full_evaluation_comparison_summary.csv"
STAGE4_LOG_PATH = LOG_DIR / "hifigan_stage4_evaluation_report.md"

def create_dataset_indexes():
    print("=" * 60)
    print("Step 1: Generating Dataset Indexes for Original vs. Generated Full Dataset")
    print("=" * 60)
    
    if not SELECTION_LOG_PATH.exists():
        print(f"ERROR: Selection log not found at {SELECTION_LOG_PATH}. Run run_full_stage.py first.")
        sys.exit(1)
        
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    orig_entries = []
    gen_entries = []
    
    with open(SELECTION_LOG_PATH, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_filename = row["new_filename"] # e.g. SP_PD_001.wav
            lang = row["language"]
            lbl = row["label"]
            orig_path = row["original_path"]
            
            orig_name = Path(orig_path).name
            speaker_id = orig_name.split("_")[0] if "_" in orig_name else orig_name.split(".")[0]
            
            # Original Preprocessed Entry (resampled & normalized input WAV)
            orig_file_path = f"voice_conversion/input_full_22050/{new_filename}"
            orig_entries.append({
                "file_path": orig_file_path,
                "speaker_id": speaker_id,
                "language": lang,
                "task": "readtext",
                "label": lbl,
                "dataset": "PCGITA" if lang == "Spanish" else "German_Sabine_Skoda",
                "source_type": "original_preprocessed"
            })
            
            # Generated HiFi-GAN Entry (output wav)
            gen_file_path = f"voice_conversion/generated_full/{new_filename.replace('.wav', '_generated.wav')}"
            gen_entries.append({
                "file_path": gen_file_path,
                "speaker_id": speaker_id,
                "language": lang,
                "task": "readtext",
                "label": lbl,
                "dataset": "PCGITA" if lang == "Spanish" else "German_Sabine_Skoda",
                "source_type": "hifigan_generated"
            })
            
    # Write Original Full Index
    orig_idx_path = METADATA_DIR / "dataset_index_original_full.csv"
    with open(orig_idx_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=orig_entries[0].keys())
        writer.writeheader()
        for e in orig_entries:
            writer.writerow(e)
            
    # Write Generated Full Index
    gen_idx_path = METADATA_DIR / "dataset_index_generated_full.csv"
    with open(gen_idx_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=gen_entries[0].keys())
        writer.writeheader()
        for e in gen_entries:
            writer.writerow(e)
            
    print(f"Created: {orig_idx_path}")
    print(f"Created: {gen_idx_path}\n")

def extract_features_for_set(index_file_path, features_out_dir, config):
    df = pd.read_csv(index_file_path)
    
    models = ["xlsr", "wav2vec2", "wavlm"]
    layers = config['model']['target_layers']
    target_sr = config['data']['target_sr'] # 16000 Hz
    device = get_device()
    
    for model_type in models:
        print(f"\n--- Extracting Features: Model={model_type.upper()} ---")
        model_out_dir = Path(features_out_dir) / model_type
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Instantiate extractor
        hf_model_name = config['model']['mapping'].get(model_type, 'facebook/wav2vec2-large-xlsr-53')
        extractor = FeatureExtractor(model_name=hf_model_name, device=device)
        
        layer_data = {layer: [] for layer in layers}
        valid_indices = []
        
        for idx, row in df.iterrows():
            rel_file_path = row['file_path']
            full_file_path = PROJECT_ROOT / rel_file_path
            
            print(f"Processing ({idx+1}/{len(df)}): {rel_file_path}")
            waveform = load_and_preprocess_audio(str(full_file_path), target_sr=target_sr)
            
            if waveform is None:
                print(f"  WARNING: Failed to load waveform for {full_file_path}")
                continue
                
            features = extractor.extract_features(waveform, target_layers=layers)
            if features is None:
                print(f"  WARNING: Failed to extract features for {full_file_path}")
                continue
                
            valid_indices.append(idx)
            for layer in layers:
                if layer in features:
                    layer_data[layer].append(features[layer])
                    
        # Save feature CSVs
        valid_df = df.iloc[valid_indices].reset_index(drop=True)
        for layer in layers:
            if len(layer_data[layer]) == 0:
                continue
            feature_matrix = np.array(layer_data[layer])
            num_feats = feature_matrix.shape[1]
            feat_cols = [f"feature_{i}" for i in range(num_feats)]
            
            feat_df = pd.DataFrame(feature_matrix, columns=feat_cols)
            final_df = pd.concat([valid_df, feat_df], axis=1)
            
            out_file = model_out_dir / f"{model_type}_readtext_layer{layer}.csv"
            final_df.to_csv(out_file, index=False)
            print(f"  Saved features for Layer {layer} to: {out_file}")

def classify_features_for_set(features_dir, outputs_dir, config):
    print(f"\n--- Running Classification Scenarios: {outputs_dir.name} ---")
    tables_dir = Path(outputs_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    classifiers = config['evaluation']['classifiers']
    
    # Retrieve cross-validation fold parameters from baseline config
    outer_folds = config['evaluation'].get('outer_folds', 10)
    inner_folds = config['evaluation'].get('inner_folds', 9)
    print(f"Using CV Splits: Outer={outer_folds}, Inner={inner_folds}")
    
    models = ["xlsr", "wav2vec2", "wavlm"]
    
    all_summary_results = []
    
    for model_type in models:
        model_feat_dir = Path(features_dir) / model_type
        if not model_feat_dir.exists():
            continue
            
        feature_files = [model_feat_dir / f for f in os.listdir(model_feat_dir) if f.endswith(".csv")]
        for feat_file in feature_files:
            print(f"Running classification on: {feat_file.name}")
            df = pd.read_csv(feat_file)
            
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            if not feature_cols:
                continue
                
            prefix = feat_file.stem
            parts = prefix.split('_')
            layer = parts[-1].replace('layer', '')
            
            results_df = run_classification_scenarios(df, feature_cols, classifiers, outer_folds, inner_folds)
            
            if not results_df.empty:
                results_df.insert(0, 'layer', layer)
                results_df.insert(0, 'model', model_type)
                
                out_file = tables_dir / f"classification_results_{prefix}.csv"
                results_df.to_csv(out_file, index=False)
                all_summary_results.append(results_df)
                
    if all_summary_results:
        final_comparison = pd.concat(all_summary_results, ignore_index=True)
        for model_type in models:
            model_comparison = final_comparison[final_comparison['model'] == model_type]
            if not model_comparison.empty:
                comp_out = tables_dir / f"model_layer_comparison_{model_type}.csv"
                model_comparison.to_csv(comp_out, index=False)
                
        return final_comparison
    return pd.DataFrame()

def generate_comparisons_and_markdown(orig_results, gen_results):
    print("=" * 60)
    print("Step 4: Compiling Comparative Report")
    print("=" * 60)
    
    if orig_results.empty or gen_results.empty:
        print("ERROR: Missing results for original or generated evaluation.")
        return
        
    orig_results = orig_results.rename(columns={
        "uar": "uar_orig", "accuracy": "acc_orig", "n_train": "n_train_orig", "n_test": "n_test_orig"
    })
    gen_results = gen_results.rename(columns={
        "uar": "uar_gen", "accuracy": "acc_gen", "n_train": "n_train_gen", "n_test": "n_test_gen"
    })
    
    merged = pd.merge(
        orig_results[["model", "layer", "train_language", "test_language", "classifier", "uar_orig", "acc_orig"]],
        gen_results[["model", "layer", "train_language", "test_language", "classifier", "uar_gen", "acc_gen"]],
        on=["model", "layer", "train_language", "test_language", "classifier"]
    )
    
    merged["uar_diff"] = merged["uar_gen"] - merged["uar_orig"]
    merged["acc_diff"] = merged["acc_gen"] - merged["acc_orig"]
    
    merged.to_csv(COMPARISON_SUMMARY_PATH, index=False)
    print(f"Comparative CSV summary saved to: {COMPARISON_SUMMARY_PATH}")
    
    md_lines = []
    md_lines.append("# HiFi-GAN Stage 4: Feature & Classification Evaluation Report\n")
    md_lines.append("## Comparative Evaluation of Reconstructed vs. Original Full Dataset\n")
    md_lines.append(f"This report compares the classification UAR and Accuracy scores for **original preprocessed** vs. **HiFi-GAN reconstructed** audios across XLSR, Wav2Vec2, and WavLM models on the full dataset.\n")
    md_lines.append("> [!NOTE]")
    md_lines.append("> Cross-validation was run using standard 10-fold outer and 9-fold inner splits as configured in baseline. Comparative results check diagnostic embedding robustness after full reconstruction.\n")
    
    md_lines.append("| Model | Layer | Scenario | Classifier | UAR Original | UAR Reconstructed | UAR Delta | Acc Original | Acc Reconstructed | Acc Delta |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    
    for _, r in merged.iterrows():
        md_lines.append(
            f"| {r['model'].upper()} | {r['layer']} | {r['train_language']}->{r['test_language']} | {r['classifier']} | "
            f"{r['uar_orig']:.4f} | {r['uar_gen']:.4f} | {r['uar_diff']:+.4f} | "
            f"{r['acc_orig']:.4f} | {r['acc_gen']:.4f} | {r['acc_diff']:+.4f} |"
        )
        
    md_lines.append("\n## Technical Conclusion")
    md_lines.append("- **Vocoding Integrity**: Near-zero delta values across all layers suggest vocoding has successfully preserved target properties needed by pre-trained feature extractors.")
    md_lines.append("- **Diagnostic Drift**: High deltas highlight potential drift or voice conversion distortion. Analyzing layers individually helps isolate which transformers are sensitive to reconstruction artifacts.")
    
    with open(STAGE4_LOG_PATH, mode="w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"Detailed Markdown evaluation report updated at: {STAGE4_LOG_PATH}\n")

def main():
    setup_logging()
    config = load_config()
    set_seed(config['random_seed'])
    
    # Task 1: Create indexes
    create_dataset_indexes()
    
    # Task 2 & 3: Run Feature Extraction
    print("\n" + "#" * 60)
    print("Starting Feature Extraction for Original Full Dataset")
    print("#" * 60)
    extract_features_for_set(
        str(METADATA_DIR / "dataset_index_original_full.csv"),
        str(FEATURES_ORIG_DIR),
        config
    )
    
    print("\n" + "#" * 60)
    print("Starting Feature Extraction for Generated Full Dataset")
    print("#" * 60)
    extract_features_for_set(
        str(METADATA_DIR / "dataset_index_generated_full.csv"),
        str(FEATURES_GEN_DIR),
        config
    )
    
    # Task 4 & 5: Run Classifications
    orig_results = classify_features_for_set(FEATURES_ORIG_DIR, OUTPUTS_ORIG_DIR, config)
    gen_results = classify_features_for_set(FEATURES_GEN_DIR, OUTPUTS_GEN_DIR, config)
    
    # Task 6: Compile comparative reports
    generate_comparisons_and_markdown(orig_results, gen_results)

if __name__ == "__main__":
    main()
