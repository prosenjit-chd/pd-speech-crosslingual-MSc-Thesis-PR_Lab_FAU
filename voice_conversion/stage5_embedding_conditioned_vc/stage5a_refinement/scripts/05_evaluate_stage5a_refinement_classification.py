#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_evaluate_stage5a_refinement_classification.py

Purpose:
Performs crosslingual classification evaluation for the grid search settings:
1. Prepares original features in features_original_refinement/.
2. Extracts layer-wise features (0, 4, 8, 11) for all 180 converted audios using
   XLSR, Wav2Vec2, and WavLM models, saving them under features_converted_refinement/.
3. Evaluates classifiers (SVM, Logistic Regression) on each setting and saves summaries.
"""

import os
import sys
import csv
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config, setup_logging, get_device, set_seed
from src.audio_utils import load_and_preprocess_audio
from src.embedding_extractor import FeatureExtractor
from src.classification import run_classification_scenarios

# Resolve directories
REFINEMENT_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc" / "stage5a_refinement"
SOURCE_EMB_DIR = REFINEMENT_DIR / "source_embeddings_refinement"
FEATURES_ORIG_DIR = REFINEMENT_DIR / "features_original_refinement"
FEATURES_CONV_DIR = REFINEMENT_DIR / "features_converted_refinement"
OUTPUTS_ORIG_DIR = REFINEMENT_DIR / "outputs_original_refinement"
OUTPUTS_CONV_DIR = REFINEMENT_DIR / "outputs_converted_refinement"
LOGS_DIR = REFINEMENT_DIR / "logs_refinement"

SELECTION_LOG = LOGS_DIR / "stage5a_refinement_input_copy_log.csv"
GENERATION_GRID_LOG = LOGS_DIR / "stage5a_refinement_generation_grid_log.csv"
CLASSIFICATION_SUMMARY_CSV = LOGS_DIR / "stage5a_refinement_classification_summary.csv"

def format_alpha_str(alpha):
    val_int = int(round(alpha * 100))
    return f"{val_int // 100}_{val_int % 100:02d}"

def prepare_original_features(eval_models, eval_layers):
    """Formats original features with speaker_id."""
    print("Formatting original features...")
    df_selection = pd.read_csv(SELECTION_LOG)
    speaker_map = {}
    for _, row in df_selection.iterrows():
        fname = row["stage5_filename"]
        orig_name = Path(row["source_path"]).name
        spk_id = orig_name.split("_")[0] if "_" in orig_name else orig_name.split(".")[0]
        speaker_map[fname] = spk_id

    for model in eval_models:
        model_src_dir = SOURCE_EMB_DIR / model
        model_out_dir = FEATURES_ORIG_DIR / model
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        for layer in eval_layers:
            src_csv = model_src_dir / f"{model}_layer{layer}_stage5a.csv"
            if not src_csv.exists():
                continue
            df = pd.read_csv(src_csv)
            df.insert(1, "speaker_id", df["file"].map(speaker_map))
            df.to_csv(model_out_dir / f"{model}_layer{layer}_original.csv", index=False)

def extract_converted_features(config, device, eval_models, eval_layers):
    """Extracts features for all 180 converted audios."""
    print("Extracting features from 180 converted files...")
    df_grid = pd.read_csv(GENERATION_GRID_LOG)
    target_sr = config['data']['target_sr']
    
    # Pre-build speaker map
    df_selection = pd.read_csv(SELECTION_LOG)
    speaker_map = {}
    for _, row in df_selection.iterrows():
        fname = row["stage5_filename"]
        orig_name = Path(row["source_path"]).name
        spk_id = orig_name.split("_")[0] if "_" in orig_name else orig_name.split(".")[0]
        speaker_map[fname] = spk_id

    # Pre-load and cache extractors once to avoid slow reloads
    extractors = {}
    print("Pre-loading feature extraction models...")
    for model_type in eval_models:
        hf_model_name = config['model']['mapping'].get(model_type, 'facebook/wav2vec2-large-xlsr-53')
        extractors[model_type] = FeatureExtractor(model_name=hf_model_name, device=device)

    # Group files by model/layer/alpha setting
    grouped = df_grid.groupby(["condition_model", "condition_layer", "alpha"])
    total_settings = len(grouped)
    setting_idx = 0

    for (cond_model, cond_layer, alpha), group_df in grouped:
        setting_idx += 1
        alpha_str = format_alpha_str(alpha)
        cond_name = f"{cond_model}_layer{cond_layer}_alpha_{alpha_str}"
        print(f"[{setting_idx}/{total_settings}] Extracting for setting: {cond_name} ...")
        
        # Load and verify converted WAV files
        valid_records = []
        waveforms = []
        for _, row in group_df.iterrows():
            src_file = row["source_file"]
            rel_path = row["output_file"]
            wav_path = REFINEMENT_DIR / rel_path
            
            if not wav_path.exists():
                continue
                
            waveform = load_and_preprocess_audio(str(wav_path), target_sr=target_sr)
            if waveform is None:
                continue
                
            waveforms.append(waveform)
            valid_records.append({
                "file": row["source_file"], # keep original filename for index mapping
                "speaker_id": speaker_map[src_file],
                "language": row["source_language"],
                "label": row["source_file"].split("_")[1], # HC or PD from SP_HC_001.wav etc.
                "group": row["source_file"].split("_")[1]
            })

        if not valid_records:
            continue
            
        # Extract layer-wise features for each evaluation model
        for model_type in eval_models:
            model_out_dir = FEATURES_CONV_DIR / model_type
            model_out_dir.mkdir(parents=True, exist_ok=True)
            
            extractor = extractors[model_type]
            
            layer_data = {layer: [] for layer in eval_layers}
            success_indices = []
            
            for idx, w in enumerate(waveforms):
                features = extractor.extract_features(w, target_layers=eval_layers)
                if features is None:
                    continue
                success_indices.append(idx)
                for layer in eval_layers:
                    if layer in features:
                        layer_data[layer].append(features[layer])
                        
            if success_indices:
                valid_df = pd.DataFrame([valid_records[i] for i in success_indices])
                for layer in eval_layers:
                    if len(layer_data[layer]) == 0:
                        continue
                    feat_matrix = np.array(layer_data[layer])
                    feat_cols = [f"feature_{i}" for i in range(feat_matrix.shape[1])]
                    feat_df = pd.DataFrame(feat_matrix, columns=feat_cols)
                    final_df = pd.concat([valid_df.reset_index(drop=True), feat_df], axis=1)
                    
                    out_csv = model_out_dir / f"{model_type}_layer{layer}_converted_{cond_name}.csv"
                    final_df.to_csv(out_csv, index=False)

def classify_refinement_settings(config, eval_models, eval_layers):
    """Evaluates SVM and LR classification for all grid search settings."""
    print("\n--- Running Classification Scenarios for Refinement Grid ---")
    
    classifiers = config['evaluation']['classifiers']
    outer_folds = 3
    inner_folds = 2
    
    df_grid = pd.read_csv(GENERATION_GRID_LOG)
    settings = df_grid.groupby(["condition_model", "condition_layer", "alpha"]).groups.keys()
    
    summary_results = []
    
    total_runs = len(settings) * len(eval_models) * len(eval_layers)
    run_idx = 0

    for cond_model, cond_layer, alpha in settings:
        alpha_str = format_alpha_str(alpha)
        cond_name = f"{cond_model}_layer{cond_layer}_alpha_{alpha_str}"
        
        for eval_model in eval_models:
            for eval_layer in eval_layers:
                run_idx += 1
                
                orig_csv = FEATURES_ORIG_DIR / eval_model / f"{eval_model}_layer{eval_layer}_original.csv"
                conv_csv = FEATURES_CONV_DIR / eval_model / f"{eval_model}_layer{eval_layer}_converted_{cond_name}.csv"
                
                if not orig_csv.exists() or not conv_csv.exists():
                    continue
                    
                df_orig = pd.read_csv(orig_csv)
                df_conv = pd.read_csv(conv_csv)
                
                feat_cols = [c for c in df_orig.columns if c.startswith('feature_')]
                if not feat_cols:
                    continue
                
                # Baseline Classification (Original files only)
                df_baseline_res = run_classification_scenarios(df_orig, feat_cols, classifiers, outer_folds, inner_folds)
                
                # Converted Classification:
                # We want to train on original and test on converted crosslingually.
                # So we build combined DataFrames:
                # 1. For Train German -> Test Spanish Converted:
                #    Training set: German original files.
                #    Testing set: Spanish converted files.
                # 2. For Train Spanish -> Test German Converted:
                #    Training set: Spanish original files.
                #    Testing set: German converted files.
                
                # Let's build a combined DataFrame df_eval
                # We take original German files and converted Spanish files (labeled language='Spanish')
                # and original Spanish files and converted German files (labeled language='German')
                df_german_orig = df_orig[df_orig["language"].str.lower() == "german"]
                df_spanish_orig = df_orig[df_orig["language"].str.lower() == "spanish"]
                
                df_german_conv = df_conv[df_conv["language"].str.lower() == "german"]
                df_spanish_conv = df_conv[df_conv["language"].str.lower() == "spanish"]
                
                # Combine original and converted for evaluation
                # Training on original, testing on converted
                df_eval = pd.concat([df_german_orig, df_spanish_orig, df_german_conv, df_spanish_conv], ignore_index=True)
                
                # Wait, inside run_classification_scenarios:
                # If train_lang == 'German' and test_lang == 'Spanish', it takes:
                # X_train = df[df['language'] == 'German']
                # X_test = df[df['language'] == 'Spanish']
                # But here, we have BOTH original and converted German, and BOTH original and converted Spanish!
                # So we need to ensure X_train only has original files, and X_test only has converted files!
                # Let's look at how we can implement this cleanly.
                # Since run_classification_scenarios splits on language only, if we put both in the dataframe,
                # it will mix original and converted in X_train and X_test!
                # To prevent this, we should NOT use run_classification_scenarios directly for the converted test,
                # OR we should pass a custom dataframe where:
                # - 'language' == 'German' represents original German (for training)
                # - 'language' == 'Spanish' represents converted Spanish (for testing Train German -> Test Spanish)
                # Let's verify this!
                # Yes!
                # If we build:
                # DataFrame A (Train German orig -> Test Spanish converted):
                #   - rows of original German (language='German')
                #   - rows of converted Spanish (language='Spanish')
                # When we run scenario ("German", "Spanish"), it trains on German (original) and tests on Spanish (converted)!
                #
                # DataFrame B (Train Spanish orig -> Test German converted):
                #   - rows of original Spanish (language='Spanish')
                #   - rows of converted German (language='German')
                # When we run scenario ("Spanish", "German"), it trains on Spanish (original) and tests on German (converted)!
                #
                # This is brilliant! It lets us reuse run_classification_scenarios exactly, without any mixing,
                # simply by constructing these two separate evaluation dataframes for the converted cases!
                
                # Let's run baseline
                df_baseline_res['target'] = df_baseline_res['train_language'] # placeholder
                
                # Construct combined DF for German -> Spanish converted
                df_de_to_sp = pd.concat([df_german_orig, df_spanish_conv], ignore_index=True)
                df_de_to_sp_res = run_classification_scenarios(df_de_to_sp, feat_cols, classifiers, outer_folds, inner_folds)
                
                # Construct combined DF for Spanish -> German converted
                df_sp_to_de = pd.concat([df_spanish_orig, df_german_conv], ignore_index=True)
                df_sp_to_de_res = run_classification_scenarios(df_sp_to_de, feat_cols, classifiers, outer_folds, inner_folds)
                
                # Parse baseline crosslingual results
                # German -> Spanish original
                orig_de_sp = df_baseline_res[(df_baseline_res["train_language"] == "German") & (df_baseline_res["test_language"] == "Spanish")]
                # Spanish -> German original
                orig_sp_de = df_baseline_res[(df_baseline_res["train_language"] == "Spanish") & (df_baseline_res["test_language"] == "German")]
                
                # Parse converted crosslingual results
                # German -> Spanish converted
                conv_de_sp = df_de_to_sp_res[(df_de_to_sp_res["train_language"] == "German") & (df_de_to_sp_res["test_language"] == "Spanish")]
                # Spanish -> German converted
                conv_sp_de = df_sp_to_de_res[(df_sp_to_de_res["train_language"] == "Spanish") & (df_sp_to_de_res["test_language"] == "German")]

                # Merge and log results
                for scenario, orig_df, conv_df in [("German->Spanish", orig_de_sp, conv_de_sp), ("Spanish->German", orig_sp_de, conv_sp_de)]:
                    for clf in classifiers:
                        r_orig = orig_df[orig_df["classifier"] == clf]
                        r_conv = conv_df[conv_df["classifier"] == clf]
                        
                        if r_orig.empty or r_conv.empty:
                            continue
                            
                        uar_o = float(r_orig["uar"].values[0])
                        uar_c = float(r_conv["uar"].values[0])
                        acc_o = float(r_orig["accuracy"].values[0])
                        acc_c = float(r_conv["accuracy"].values[0])
                        
                        summary_results.append({
                            "condition_model": cond_model,
                            "condition_layer": cond_layer,
                            "alpha": alpha,
                            "evaluation_model": eval_model,
                            "evaluation_layer": eval_layer,
                            "scenario": scenario,
                            "classifier": clf,
                            "uar_original": uar_o,
                            "uar_converted": uar_c,
                            "uar_delta": uar_c - uar_o,
                            "accuracy_original": acc_o,
                            "accuracy_converted": acc_c,
                            "accuracy_delta": acc_c - acc_o,
                            "diagnostic_only": "diagnostic_only"
                        })
                        
                if run_idx % 20 == 0:
                    print(f"Evaluated {run_idx}/{total_runs} classification runs...")

    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv(CLASSIFICATION_SUMMARY_CSV, index=False)
        print(f"\nClassification comparison summary saved to: {CLASSIFICATION_SUMMARY_CSV}")

def main():
    setup_logging()
    config = load_config()
    set_seed(config['random_seed'])
    device = get_device()
    
    print("=" * 60)
    print("Stage 5A-Refinement Step 5: Classification Evaluation")
    print("=" * 60)
    
    eval_models = ["xlsr", "wav2vec2", "wavlm"]
    eval_layers = config['model']['target_layers'] # [0, 4, 8, 11]
    
    # 1. Format original features
    prepare_original_features(eval_models, eval_layers)
    
    # 2. Extract features from converted files
    extract_converted_features(config, device, eval_models, eval_layers)
    
    # 3. Classification grid
    classify_refinement_settings(config, eval_models, eval_layers)
    print("Step 5 Finished Successfully!\n")

if __name__ == "__main__":
    main()
