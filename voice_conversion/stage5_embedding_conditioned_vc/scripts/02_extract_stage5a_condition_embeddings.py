#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_extract_stage5a_condition_embeddings.py

Purpose:
Extracts WavLM, Wav2Vec2, and XLSR features for layers [0, 4, 8, 11]
from the 12 pilot files and saves them under:
C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc\source_embeddings\
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

# Directory Constants
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
INPUT_DIR = STAGE5_DIR / "input_pilot_12"
SOURCE_EMB_DIR = STAGE5_DIR / "source_embeddings"
SELECTION_LOG_PATH = STAGE5_DIR / "logs_stage5" / "stage5a_pilot_12_selection_log.csv"
EXTRACTION_LOG_PATH = STAGE5_DIR / "logs_stage5" / "stage5a_embedding_extraction_log.csv"

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    set_seed(config['random_seed'])
    
    print("=" * 60)
    print("Stage 5A Step 2: Extracting Source Embeddings")
    print("=" * 60)
    
    if not SELECTION_LOG_PATH.exists():
        print(f"ERROR: Selection log not found at {SELECTION_LOG_PATH}. Run step 1 script first.")
        sys.exit(1)
        
    df = pd.read_csv(SELECTION_LOG_PATH)
    models = ["xlsr", "wav2vec2", "wavlm"]
    layers = config['model']['target_layers'] # [0, 4, 8, 11]
    target_sr = config['data']['target_sr'] # 16000 Hz
    device = get_device()
    
    extraction_logs = []
    
    for model_type in models:
        print(f"\n--- Model: {model_type.upper()} ---")
        model_out_dir = SOURCE_EMB_DIR / model_type
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Instantiate extractor
        hf_model_name = config['model']['mapping'].get(model_type, 'facebook/wav2vec2-large-xlsr-53')
        print(f"Loading feature extractor for {model_type} from {hf_model_name}")
        extractor = FeatureExtractor(model_name=hf_model_name, device=device)
        
        layer_data = {layer: [] for layer in layers}
        valid_rows = []
        
        for idx, row in df.iterrows():
            filename = row['stage5_filename']
            file_path = INPUT_DIR / filename
            
            print(f"[{idx+1}/{len(df)}] Processing {filename} ... ", end="")
            
            waveform = load_and_preprocess_audio(str(file_path), target_sr=target_sr)
            if waveform is None:
                print("FAILED (audio load)")
                for l in layers:
                    extraction_logs.append({
                        "file": filename, "model": model_type, "layer": l, 
                        "status": "failed", "error": "audio load error"
                    })
                continue
                
            features = extractor.extract_features(waveform, target_layers=layers)
            if features is None:
                print("FAILED (feature extraction)")
                for l in layers:
                    extraction_logs.append({
                        "file": filename, "model": model_type, "layer": l, 
                        "status": "failed", "error": "extraction error"
                    })
                continue
                
            valid_rows.append(row)
            for layer in layers:
                if layer in features:
                    layer_data[layer].append(features[layer])
                    extraction_logs.append({
                        "file": filename, "model": model_type, "layer": layer, 
                        "status": "success", "error": ""
                    })
            print("SUCCESS")
            
        # Save features for each layer
        if valid_rows:
            valid_df = pd.DataFrame(valid_rows)
            # Standardize column naming for Step 2 requirement: file, language, label, group, feature_0...
            # Note: We rename stage5_filename to file
            valid_df = valid_df.rename(columns={"stage5_filename": "file"})
            
            for layer in layers:
                if len(layer_data[layer]) == 0:
                    continue
                feature_matrix = np.array(layer_data[layer])
                num_feats = feature_matrix.shape[1]
                feat_cols = [f"feature_{i}" for i in range(num_feats)]
                
                feat_df = pd.DataFrame(feature_matrix, columns=feat_cols)
                final_df = pd.concat([valid_df.reset_index(drop=True), feat_df], axis=1)
                
                # We only need: file, language, label, group, and feature columns
                cols_to_keep = ["file", "language", "label", "group"] + feat_cols
                final_df = final_df[cols_to_keep]
                
                out_file = model_out_dir / f"{model_type}_layer{layer}_stage5a.csv"
                final_df.to_csv(out_file, index=False)
                print(f"Saved Layer {layer} embeddings to: {out_file.relative_to(STAGE5_DIR)}")
                
    # Write extraction log
    with open(EXTRACTION_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["file", "model", "layer", "status", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for log in extraction_logs:
            writer.writerow(log)
            
    print(f"\nExtraction log written to: {EXTRACTION_LOG_PATH}")
    print("Step 2 Finished Successfully!\n")

if __name__ == "__main__":
    main()
