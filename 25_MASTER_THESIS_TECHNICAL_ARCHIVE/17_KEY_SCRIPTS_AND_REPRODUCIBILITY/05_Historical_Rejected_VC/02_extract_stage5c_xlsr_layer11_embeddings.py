#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_extract_stage5c_xlsr_layer11_embeddings.py

Purpose:
Extracts XLSR layer 11 embeddings from the 276 Stage 5C input files.
Saves features to source_embeddings_stage5c/xlsr_layer11_stage5c.csv
and logs to logs_stage5c/stage5c_embedding_extraction_log.csv.
"""

import os
import sys
import csv
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config, setup_logging, get_device, set_seed
from src.audio_utils import load_and_preprocess_audio
from src.embedding_extractor import FeatureExtractor

# Resolve directories
STAGE5C_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc" / "stage5c_full_276"
INPUT_DIR = STAGE5C_DIR / "input_full_276_stage5c"
EMB_DIR = STAGE5C_DIR / "source_embeddings_stage5c"
LOGS_DIR = STAGE5C_DIR / "logs_stage5c"

COPY_LOG_PATH = LOGS_DIR / "stage5c_input_copy_log.csv"
EMB_OUT_PATH = EMB_DIR / "xlsr_layer11_stage5c.csv"
EXTRACTION_LOG_PATH = LOGS_DIR / "stage5c_embedding_extraction_log.csv"

def main():
    setup_logging()
    config = load_config()
    set_seed(config['random_seed'])
    device = get_device()

    print("=" * 60)
    print("Stage 5C Step 2: Extracting XLSR Layer 11 Embeddings")
    print("=" * 60)
    print(f"Device: {device}")

    if not COPY_LOG_PATH.exists():
        print(f"ERROR: Copy log not found at {COPY_LOG_PATH}. Run step 1 first.")
        sys.exit(1)

    df_inputs = pd.read_csv(COPY_LOG_PATH)
    file_list = df_inputs["stage5c_filename"].tolist()

    # Load extractor (using XLSR)
    model_type = "xlsr"
    hf_model_name = config['model']['mapping'].get(model_type, 'facebook/wav2vec2-large-xlsr-53')
    target_layer = 11
    
    print(f"Loading {model_type} model: {hf_model_name}...")
    extractor = FeatureExtractor(model_name=hf_model_name, device=device)
    
    target_sr = config['data']['target_sr']
    extracted_features = []
    extraction_logs = []

    total_files = len(file_list)
    print(f"Starting extraction for {total_files} files...")

    for idx, filename in enumerate(file_list):
        wav_path = INPUT_DIR / filename
        row_info = df_inputs[df_inputs["stage5c_filename"] == filename].iloc[0]
        
        print(f"[{idx+1}/{total_files}] Processing {filename} ... ", end="")
        
        status = "failed"
        error_msg = ""
        
        if not wav_path.exists():
            error_msg = "WAV file does not exist."
            print(f"FAILED ({error_msg})")
        else:
            try:
                waveform = load_and_preprocess_audio(str(wav_path), target_sr=target_sr)
                if waveform is None:
                    error_msg = "Failed to load/preprocess audio."
                    print(f"FAILED ({error_msg})")
                else:
                    features = extractor.extract_features(waveform, target_layers=[target_layer])
                    if features is None or target_layer not in features:
                        error_msg = "Extraction failed or layer not found."
                        print(f"FAILED ({error_msg})")
                    else:
                        feat_vector = features[target_layer]
                        dim = len(feat_vector)
                        
                        # Store features
                        feature_dict = {
                            "file": filename,
                            "language": row_info["language"],
                            "label": row_info["label"],
                            "group": row_info["group"]
                        }
                        for d_idx in range(dim):
                            feature_dict[f"feature_{d_idx}"] = float(feat_vector[d_idx])
                            
                        extracted_features.append(feature_dict)
                        status = "success"
                        print(f"SUCCESS (dim={dim})")
            except Exception as e:
                error_msg = str(e)
                print(f"FAILED ({error_msg})")

        extraction_logs.append({
            "stage5c_filename": filename,
            "status": status,
            "error": error_msg
        })

    # Save CSV embeddings
    if extracted_features:
        df_emb = pd.DataFrame(extracted_features)
        df_emb.to_csv(EMB_OUT_PATH, index=False)
        print(f"Saved XLSR L11 embeddings: {EMB_OUT_PATH}")
    else:
        print("ERROR: No features extracted successfully.")
        sys.exit(1)

    # Save extraction log
    with open(EXTRACTION_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["stage5c_filename", "status", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for log in extraction_logs:
            writer.writerow(log)
    print(f"Saved extraction log: {EXTRACTION_LOG_PATH}")
    print("Step 2 Finished Successfully!\n")

if __name__ == "__main__":
    main()
