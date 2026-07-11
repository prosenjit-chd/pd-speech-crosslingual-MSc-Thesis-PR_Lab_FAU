#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_evaluate_stage5b_classification.py

Purpose:
Performs subset-level classification evaluation for Stage 5B:
1. Dynamically extracts original features (layers 0, 4, 8, 11) for all 80 original WAV files
   using XLSR, Wav2Vec2, and WavLM, saving them to features_original_stage5b/.
2. Extracts converted features (layers 0, 4, 8, 11) for all 80 converted WAV files,
   saving them to features_converted_stage5b/.
3. Trains cross-validated classifiers (Linear SVM, Logistic Regression) on both original
   and converted features.
4. Generates outputs and logs comparative metrics to logs_stage5b/.
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
STAGE5B_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc" / "stage5b_subset_80"
INPUT_DIR = STAGE5B_DIR / "input_subset_80_stage5b"
CONV_DIR = STAGE5B_DIR / "converted_audio_stage5b"
FEATURES_ORIG_DIR = STAGE5B_DIR / "features_original_stage5b"
FEATURES_CONV_DIR = STAGE5B_DIR / "features_converted_stage5b"
OUTPUTS_ORIG_DIR = STAGE5B_DIR / "outputs_original_stage5b"
OUTPUTS_CONV_DIR = STAGE5B_DIR / "outputs_converted_stage5b"
LOGS_DIR = STAGE5B_DIR / "logs_stage5b"

COPY_LOG_PATH = LOGS_DIR / "stage5b_input_copy_log.csv"
GENERATION_LOG_PATH = LOGS_DIR / "stage5b_conditioned_generation_log.csv"
CLASSIFICATION_SUMMARY_CSV = LOGS_DIR / "stage5b_original_vs_converted_classification_summary.csv"

def extract_all_features(config, device, file_list, src_dir, out_base_dir, suffix, eval_models, eval_layers, speaker_map, df_metadata):
    """Extracts features for all 80 files and saves them by model/layer."""
    target_sr = config['data']['target_sr']

    # Pre-load extractors
    extractors = {}
    print(f"Pre-loading feature extraction models for {suffix} files...")
    for model_type in eval_models:
        hf_model_name = config['model']['mapping'].get(model_type, 'facebook/wav2vec2-large-xlsr-53')
        extractors[model_type] = FeatureExtractor(model_name=hf_model_name, device=device)

    # Process each model
    for model_type in eval_models:
        model_out_dir = out_base_dir / model_type
        model_out_dir.mkdir(parents=True, exist_ok=True)
        extractor = extractors[model_type]
        
        print(f"Extracting features using {model_type} ...")
        
        # We process files sequentially
        waveforms = []
        valid_records = []
        
        for fname in file_list:
            # Map source name to determine correct output filename if suffix is converted
            if suffix == "converted":
                # Check target folder spanish_to_german or german_to_spanish
                row_meta = df_metadata[df_metadata["stage5b_filename"] == fname].iloc[0]
                lang = row_meta["language"]
                target_domain = "German" if lang.lower() == "spanish" else "Spanish"
                dir_name = "spanish_to_german" if lang.lower() == "spanish" else "german_to_spanish"
                conv_name = fname.replace(".wav", f"_to_{target_domain[:2]}_domain_xlsr_layer11_alpha_1_00.wav")
                wav_path = src_dir / dir_name / conv_name
            else:
                wav_path = src_dir / fname

            if not wav_path.exists():
                print(f"WARNING: File not found: {wav_path}")
                continue
                
            w = load_and_preprocess_audio(str(wav_path), target_sr=target_sr)
            if w is None:
                continue
                
            waveforms.append(w)
            row_meta = df_metadata[df_metadata["stage5b_filename"] == fname].iloc[0]
            valid_records.append({
                "file": fname,
                "speaker_id": speaker_map[fname],
                "language": row_meta["language"],
                "label": row_meta["label"],
                "group": row_meta["group"]
            })

        if not valid_records:
            continue
            
        # Run extraction
        layer_data = {layer: [] for layer in eval_layers}
        success_indices = []
        
        total_wavs = len(waveforms)
        for w_idx, w in enumerate(waveforms):
            features = extractor.extract_features(w, target_layers=eval_layers)
            if features is None:
                continue
            success_indices.append(w_idx)
            for layer in eval_layers:
                if layer in features:
                    layer_data[layer].append(features[layer])
                    
        # Save CSV for each layer
        if success_indices:
            valid_df = pd.DataFrame([valid_records[i] for i in success_indices])
            for layer in eval_layers:
                if len(layer_data[layer]) == 0:
                    continue
                feat_matrix = np.array(layer_data[layer])
                feat_cols = [f"feature_{i}" for i in range(feat_matrix.shape[1])]
                feat_df = pd.DataFrame(feat_matrix, columns=feat_cols)
                final_df = pd.concat([valid_df.reset_index(drop=True), feat_df], axis=1)
                
                out_csv = model_out_dir / f"{model_type}_layer{layer}_{suffix}.csv"
                final_df.to_csv(out_csv, index=False)
                print(f"Saved {model_type} L{layer} features to: {out_csv.name}")

def main():
    setup_logging()
    config = load_config()
    set_seed(config['random_seed'])
    device = get_device()

    print("=" * 60)
    print("Stage 5B Step 6: Classification Evaluation")
    print("=" * 60)

    if not COPY_LOG_PATH.exists() or not GENERATION_LOG_PATH.exists():
        print("ERROR: Input copy log or generation log not found.")
        sys.exit(1)

    df_inputs = pd.read_csv(COPY_LOG_PATH)
    file_list = df_inputs["stage5b_filename"].tolist()

    # Pre-build speaker map
    speaker_map = {}
    for _, row in df_inputs.iterrows():
        fname = row["stage5b_filename"]
        orig_name = Path(row["source_path"]).name
        spk_id = orig_name.split("_")[0] if "_" in orig_name else orig_name.split(".")[0]
        speaker_map[fname] = spk_id

    eval_models = ["xlsr", "wav2vec2", "wavlm"]
    eval_layers = config['model']['target_layers'] # [0, 4, 8, 11]

    # 1. Extract original features
    extract_all_features(
        config=config,
        device=device,
        file_list=file_list,
        src_dir=INPUT_DIR,
        out_base_dir=FEATURES_ORIG_DIR,
        suffix="original",
        eval_models=eval_models,
        eval_layers=eval_layers,
        speaker_map=speaker_map,
        df_metadata=df_inputs
    )

    # 2. Extract converted features
    extract_all_features(
        config=config,
        device=device,
        file_list=file_list,
        src_dir=CONV_DIR,
        out_base_dir=FEATURES_CONV_DIR,
        suffix="converted",
        eval_models=eval_models,
        eval_layers=eval_layers,
        speaker_map=speaker_map,
        df_metadata=df_inputs
    )

    # 3. Classification
    classifiers = config['evaluation']['classifiers']
    outer_folds = 3
    inner_folds = 2
    
    summary_results = []
    
    # Scenarios to run:
    # We want:
    # - Train German -> Test original Spanish
    # - Train German -> Test Spanish->German-domain converted
    # - Train Spanish -> Test original German
    # - Train Spanish -> Test German->Spanish-domain converted
    # Also within-language checks:
    # - Spanish -> Spanish
    # - German -> German
    # - Spanish+German -> Spanish+German

    print("\n--- Running Classifier Training & Scenarios ---")

    for eval_model in eval_models:
        for eval_layer in eval_layers:
            orig_csv = FEATURES_ORIG_DIR / eval_model / f"{eval_model}_layer{eval_layer}_original.csv"
            conv_csv = FEATURES_CONV_DIR / eval_model / f"{eval_model}_layer{eval_layer}_converted.csv"
            
            if not orig_csv.exists() or not conv_csv.exists():
                continue
                
            df_orig = pd.read_csv(orig_csv)
            df_conv = pd.read_csv(conv_csv)
            
            feat_cols = [c for c in df_orig.columns if c.startswith('feature_')]
            if not feat_cols:
                continue

            # Run baseline scenarios (Original files only)
            df_baseline_res = run_classification_scenarios(df_orig, feat_cols, classifiers, outer_folds, inner_folds)
            
            # Converted Crosslingual evaluation dataframes
            # To train on original and test on converted crosslingually:
            df_german_orig = df_orig[df_orig["language"].str.lower() == "german"]
            df_spanish_orig = df_orig[df_orig["language"].str.lower() == "spanish"]
            df_german_conv = df_conv[df_conv["language"].str.lower() == "german"]
            df_spanish_conv = df_conv[df_conv["language"].str.lower() == "spanish"]
            
            # Combined DF for German -> Spanish Converted (Train on German orig, Test on Spanish converted)
            df_de_to_sp = pd.concat([df_german_orig, df_spanish_conv], ignore_index=True)
            df_de_to_sp_res = run_classification_scenarios(df_de_to_sp, feat_cols, classifiers, outer_folds, inner_folds)
            
            # Combined DF for Spanish -> German Converted (Train on Spanish orig, Test on German converted)
            df_sp_to_de = pd.concat([df_spanish_orig, df_german_conv], ignore_index=True)
            df_sp_to_de_res = run_classification_scenarios(df_sp_to_de, feat_cols, classifiers, outer_folds, inner_folds)

            # Combined DF for Within-Language/Combined Converted Scenarios
            # (Test sets are replaced with converted features)
            # Spanish -> Spanish Converted (Train on Spanish orig, Test on Spanish converted)
            df_sp_to_sp = pd.concat([df_spanish_orig, df_spanish_conv], ignore_index=True)
            df_sp_to_sp_res = run_classification_scenarios(df_sp_to_sp, feat_cols, classifiers, outer_folds, inner_folds)

            # German -> German Converted (Train on German orig, Test on German converted)
            df_de_to_de = pd.concat([df_german_orig, df_german_conv], ignore_index=True)
            df_de_to_de_res = run_classification_scenarios(df_de_to_de, feat_cols, classifiers, outer_folds, inner_folds)

            # Combined Converted (Train on Spanish+German orig, Test on Spanish+German converted)
            df_comb_conv = pd.concat([df_orig, df_conv], ignore_index=True)
            df_comb_res = run_classification_scenarios(df_comb_conv, feat_cols, classifiers, outer_folds, inner_folds)

            # Standard scenarios to map
            scenarios_map = [
                ("German->Spanish", orig_baseline_query := {"train_language": "German", "test_language": "Spanish"}, df_de_to_sp_res, {"train_language": "German", "test_language": "Spanish"}),
                ("Spanish->German", orig_baseline_query2 := {"train_language": "Spanish", "test_language": "German"}, df_sp_to_de_res, {"train_language": "Spanish", "test_language": "German"}),
                ("Spanish->Spanish", {"train_language": "Spanish", "test_language": "Spanish"}, df_sp_to_sp_res, {"train_language": "Spanish", "test_language": "Spanish"}),
                ("German->German", {"train_language": "German", "test_language": "German"}, df_de_to_de_res, {"train_language": "German", "test_language": "German"}),
                ("Combined->Combined", {"train_language": "Spanish+German", "test_language": "Spanish+German"}, df_comb_res, {"train_language": "Spanish+German", "test_language": "Spanish+German"})
            ]

            for scen_name, q_orig, df_conv_res, q_conv in scenarios_map:
                for clf in classifiers:
                    r_orig = df_baseline_res[
                        (df_baseline_res["train_language"] == q_orig["train_language"]) &
                        (df_baseline_res["test_language"] == q_orig["test_language"]) &
                        (df_baseline_res["classifier"] == clf)
                    ]
                    
                    r_conv = df_conv_res[
                        (df_conv_res["train_language"] == q_conv["train_language"]) &
                        (df_conv_res["test_language"] == q_conv["test_language"]) &
                        (df_conv_res["classifier"] == clf)
                    ]
                    
                    if r_orig.empty or r_conv.empty:
                        continue
                        
                    uar_o = float(r_orig["uar"].values[0])
                    uar_c = float(r_conv["uar"].values[0])
                    acc_o = float(r_orig["accuracy"].values[0])
                    acc_c = float(r_conv["accuracy"].values[0])
                    
                    sens_o = float(r_orig["sensitivity"].values[0])
                    sens_c = float(r_conv["sensitivity"].values[0])
                    spec_o = float(r_orig["specificity"].values[0])
                    spec_c = float(r_conv["specificity"].values[0])
                    
                    # Read AUC if exists, else default to NaN
                    auc_o = float(r_orig["auc"].values[0]) if "auc" in r_orig.columns else np.nan
                    auc_c = float(r_conv["auc"].values[0]) if "auc" in r_conv.columns else np.nan

                    summary_results.append({
                        "evaluation_model": eval_model,
                        "evaluation_layer": eval_layer,
                        "scenario": scen_name,
                        "classifier": clf,
                        "uar_original": uar_o,
                        "uar_converted": uar_c,
                        "uar_delta": uar_c - uar_o,
                        "accuracy_original": acc_o,
                        "accuracy_converted": acc_c,
                        "accuracy_delta": acc_c - acc_o,
                        "sensitivity_original": sens_o,
                        "sensitivity_converted": sens_c,
                        "specificity_original": spec_o,
                        "specificity_converted": spec_c,
                        "auc_original": auc_o,
                        "auc_converted": auc_c
                    })

            print(f"Completed evaluation for {eval_model} L{eval_layer}.")

    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv(CLASSIFICATION_SUMMARY_CSV, index=False)
        print(f"\nClassification comparison summary saved to: {CLASSIFICATION_SUMMARY_CSV}")

    print("Step 6 Finished Successfully!\n")

if __name__ == "__main__":
    main()
