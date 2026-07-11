#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_evaluate_stage5a_classification.py

Purpose:
Runs classification evaluations for original and converted pilot data.
1. Extracts embeddings from converted audio files and saves them in features_converted_stage5/.
2. Formats original embeddings and saves them in features_original_stage5/.
3. Runs classification scenarios (SVM, Logistic Regression) on:
   - Original baseline pilot files
   - Converted pilot files (Spanish converted to German, German converted to Spanish)
4. Saves diagnostic results and generates logs_stage5/stage5a_original_vs_converted_classification_summary.csv.
"""

import os
import sys
import csv
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config, setup_logging, get_device, set_seed
from src.audio_utils import load_and_preprocess_audio
from src.embedding_extractor import FeatureExtractor
from src.classification import run_classification_scenarios

# Directory Constants
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
CONVERTED_SP_DIR = STAGE5_DIR / "converted_spanish_to_german"
CONVERTED_DE_DIR = STAGE5_DIR / "converted_german_to_spanish"
SOURCE_EMB_DIR = STAGE5_DIR / "source_embeddings"
FEATURES_ORIG_DIR = STAGE5_DIR / "features_original_stage5"
FEATURES_CONV_DIR = STAGE5_DIR / "features_converted_stage5"
OUTPUTS_ORIG_DIR = STAGE5_DIR / "outputs_original_stage5"
OUTPUTS_CONV_DIR = STAGE5_DIR / "outputs_converted_stage5"
LOG_DIR = STAGE5_DIR / "logs_stage5"

SELECTION_LOG_PATH = LOG_DIR / "stage5a_pilot_12_selection_log.csv"
COMPARISON_SUMMARY_PATH = LOG_DIR / "stage5a_original_vs_converted_classification_summary.csv"

def extract_converted_features(config, device, layers, models):
    """Extracts features from the converted audio files."""
    print("\n--- Extracting Features from Converted Audio Files ---")
    
    if not SELECTION_LOG_PATH.exists():
        print(f"ERROR: Selection log not found at {SELECTION_LOG_PATH}")
        sys.exit(1)
        
    df_selection = pd.read_csv(SELECTION_LOG_PATH)
    target_sr = config['data']['target_sr'] # 16000 Hz
    
    # Pre-build index rows for converted files
    converted_records = []
    for _, row in df_selection.iterrows():
        src_file = row["stage5_filename"]
        lang = row["language"]
        lbl = row["label"]
        group = row["group"]
        
        # Determine converted filename and path
        if lang.lower() == "spanish":
            conv_filename = src_file.replace(".wav", "_to_DE_domain.wav")
            conv_path = CONVERTED_SP_DIR / conv_filename
        else:
            conv_filename = src_file.replace(".wav", "_to_SP_domain.wav")
            conv_path = CONVERTED_DE_DIR / conv_filename
            
        # speaker_id can be resolved from original filename
        orig_name = Path(row["source_path"]).name
        speaker_id = orig_name.split("_")[0] if "_" in orig_name else orig_name.split(".")[0]
        
        converted_records.append({
            "file": conv_filename,
            "file_path": conv_path,
            "speaker_id": speaker_id,
            "language": lang,
            "label": lbl,
            "group": group
        })
        
    for model_type in models:
        print(f"Model: {model_type.upper()}")
        model_out_dir = FEATURES_CONV_DIR / model_type
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Instantiate extractor
        hf_model_name = config['model']['mapping'].get(model_type, 'facebook/wav2vec2-large-xlsr-53')
        extractor = FeatureExtractor(model_name=hf_model_name, device=device)
        
        layer_data = {layer: [] for layer in layers}
        valid_records = []
        
        for idx, rec in enumerate(converted_records):
            file_path = rec["file_path"]
            filename = rec["file"]
            print(f"  [{idx+1}/{len(converted_records)}] Extracting: {filename} ... ", end="")
            
            if not file_path.exists():
                print("FAILED (does not exist)")
                continue
                
            waveform = load_and_preprocess_audio(str(file_path), target_sr=target_sr)
            if waveform is None:
                print("FAILED (audio load)")
                continue
                
            features = extractor.extract_features(waveform, target_layers=layers)
            if features is None:
                print("FAILED (feature extraction)")
                continue
                
            valid_records.append(rec)
            for layer in layers:
                if layer in features:
                    layer_data[layer].append(features[layer])
            print("SUCCESS")
            
        if valid_records:
            valid_df = pd.DataFrame(valid_records)
            # Remove file_path as we don't want it in features file
            valid_df = valid_df.drop(columns=["file_path"])
            
            for layer in layers:
                if len(layer_data[layer]) == 0:
                    continue
                feature_matrix = np.array(layer_data[layer])
                num_feats = feature_matrix.shape[1]
                feat_cols = [f"feature_{i}" for i in range(num_feats)]
                
                feat_df = pd.DataFrame(feature_matrix, columns=feat_cols)
                final_df = pd.concat([valid_df.reset_index(drop=True), feat_df], axis=1)
                
                out_file = model_out_dir / f"{model_type}_layer{layer}_converted.csv"
                final_df.to_csv(out_file, index=False)
                print(f"  Saved Layer {layer} converted features to: {out_file.relative_to(STAGE5_DIR)}")

def prepare_original_features(layers, models):
    """Copies and reformats the original embeddings to features_original_stage5/."""
    print("\n--- Preparing Original Features ---")
    if not SOURCE_EMB_DIR.exists():
        print(f"ERROR: Source embeddings directory not found at {SOURCE_EMB_DIR}")
        sys.exit(1)
        
    for model_type in models:
        model_src_dir = SOURCE_EMB_DIR / model_type
        model_out_dir = FEATURES_ORIG_DIR / model_type
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        for layer in layers:
            src_csv = model_src_dir / f"{model_type}_layer{layer}_stage5a.csv"
            if not src_csv.exists():
                print(f"  WARNING: {src_csv.name} not found.")
                continue
                
            df = pd.read_csv(src_csv)
            # Add speaker_id to match classification requirements
            df_selection = pd.read_csv(SELECTION_LOG_PATH)
            
            speaker_map = {}
            for _, row in df_selection.iterrows():
                fname = row["stage5_filename"]
                orig_name = Path(row["source_path"]).name
                spk_id = orig_name.split("_")[0] if "_" in orig_name else orig_name.split(".")[0]
                speaker_map[fname] = spk_id
                
            df.insert(1, "speaker_id", df["file"].map(speaker_map))
            
            out_file = model_out_dir / f"{model_type}_layer{layer}_original.csv"
            df.to_csv(out_file, index=False)
            print(f"  Formatted and saved original features to: {out_file.relative_to(STAGE5_DIR)}")

def classify_features(features_dir, outputs_dir, config, tag):
    """Runs standard classification scenarios on the given features directory."""
    print(f"\n--- Classifying {tag} Features ---")
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    classifiers = config['evaluation']['classifiers']
    # Use pilot folds
    outer_folds = 3
    inner_folds = 2
    
    models = ["xlsr", "wav2vec2", "wavlm"]
    layers = [0, 4, 8, 11]
    
    all_results = []
    
    for model_type in models:
        model_feat_dir = features_dir / model_type
        if not model_feat_dir.exists():
            continue
            
        for layer in layers:
            feat_file = model_feat_dir / f"{model_type}_layer{layer}_{tag}.csv"
            if not feat_file.exists():
                continue
                
            print(f"Running classification on: {feat_file.name}")
            df = pd.read_csv(feat_file)
            
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            if not feature_cols:
                continue
                
            results_df = run_classification_scenarios(df, feature_cols, classifiers, outer_folds, inner_folds)
            
            if not results_df.empty:
                results_df.insert(0, 'layer', layer)
                results_df.insert(0, 'model', model_type)
                
                out_file = tables_dir / f"classification_results_{model_type}_layer{layer}.csv"
                results_df.to_csv(out_file, index=False)
                all_results.append(results_df)
                
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    set_seed(config['random_seed'])
    device = get_device()
    
    print("=" * 60)
    print("Stage 5A Step 6: Classification Evaluation")
    print("=" * 60)
    
    layers = config['model']['target_layers'] # [0, 4, 8, 11]
    models = ["xlsr", "wav2vec2", "wavlm"]
    
    # 1. Re-format original embeddings
    prepare_original_features(layers, models)
    
    # 2. Extract features from converted files
    extract_converted_features(config, device, layers, models)
    
    # 3. Classify original features
    orig_results = classify_features(FEATURES_ORIG_DIR, OUTPUTS_ORIG_DIR, config, "original")
    
    # 4. Classify converted features
    conv_results = classify_features(FEATURES_CONV_DIR, OUTPUTS_CONV_DIR, config, "converted")
    
    # 5. Compile comparative summary
    if not orig_results.empty and not conv_results.empty:
        # Re-key column names to distinguish original vs converted
        orig_results = orig_results.rename(columns={
            "uar": "uar_orig", "accuracy": "acc_orig", "n_train": "n_train_orig", "n_test": "n_test_orig"
        })
        conv_results = conv_results.rename(columns={
            "uar": "uar_conv", "accuracy": "acc_conv", "n_train": "n_train_conv", "n_test": "n_test_conv"
        })
        
        merged = pd.merge(
            orig_results[["model", "layer", "train_language", "test_language", "classifier", "uar_orig", "acc_orig"]],
            conv_results[["model", "layer", "train_language", "test_language", "classifier", "uar_conv", "acc_conv"]],
            on=["model", "layer", "train_language", "test_language", "classifier"]
        )
        
        merged["uar_delta"] = merged["uar_conv"] - merged["uar_orig"]
        merged["acc_delta"] = merged["acc_conv"] - merged["acc_orig"]
        
        # Add note about diagnostic classification results (User correction 4)
        merged["evaluation_type"] = "diagnostic_only"
        
        merged.to_csv(COMPARISON_SUMMARY_PATH, index=False)
        print(f"\nSaved classification comparison summary to: {COMPARISON_SUMMARY_PATH}")
        print("Step 6 Finished Successfully!\n")
    else:
        print("ERROR: Classification results are empty. Classification evaluation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
