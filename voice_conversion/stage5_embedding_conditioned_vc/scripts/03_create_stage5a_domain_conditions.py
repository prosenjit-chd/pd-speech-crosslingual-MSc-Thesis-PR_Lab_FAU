#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_create_stage5a_domain_conditions.py

Purpose:
Computes averaged target-language/domain condition vectors for XLSR, Wav2Vec2, and WavLM
across layers [0, 4, 8, 11] from the extracted source embeddings:
- German-domain condition = average embedding of German pilot files
- Spanish-domain condition = average embedding of Spanish pilot files
Saves them under:
C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc\target_domain_embeddings\
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Directory Constants
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
SOURCE_EMB_DIR = STAGE5_DIR / "source_embeddings"
TARGET_EMB_DIR = STAGE5_DIR / "target_domain_embeddings"
SUMMARY_LOG_PATH = STAGE5_DIR / "logs_stage5" / "stage5a_domain_condition_summary.csv"

def main():
    print("=" * 60)
    print("Stage 5A Step 3: Creating Target-Domain Conditions")
    print("=" * 60)
    
    if not SOURCE_EMB_DIR.exists():
        print(f"ERROR: Source embeddings directory not found at {SOURCE_EMB_DIR}. Run step 2 first.")
        sys.exit(1)
        
    TARGET_EMB_DIR.mkdir(parents=True, exist_ok=True)
    
    models = ["xlsr", "wav2vec2", "wavlm"]
    layers = [0, 4, 8, 11]
    
    summary_entries = []
    
    for model in models:
        model_src_dir = SOURCE_EMB_DIR / model
        model_tgt_dir = TARGET_EMB_DIR / model
        model_tgt_dir.mkdir(parents=True, exist_ok=True)
        
        if not model_src_dir.exists():
            print(f"WARNING: Source folder for {model} does not exist. Skipping.")
            continue
            
        for layer in layers:
            src_csv = model_src_dir / f"{model}_layer{layer}_stage5a.csv"
            if not src_csv.exists():
                print(f"WARNING: Feature file {src_csv.name} not found. Skipping.")
                continue
                
            print(f"Processing: Model={model.upper()} | Layer={layer}")
            df = pd.read_csv(src_csv)
            
            # Find feature columns
            feat_cols = [col for col in df.columns if col.startswith('feature_')]
            if not feat_cols:
                print(f"  WARNING: No feature columns found in {src_csv.name}.")
                continue
                
            # Filter German and Spanish rows
            german_df = df[df['language'].str.lower() == 'german']
            spanish_df = df[df['language'].str.lower() == 'spanish']
            
            # Compute average embedding
            if not german_df.empty:
                german_avg = german_df[feat_cols].mean(axis=0).values
                # Save German condition
                de_out_csv = model_tgt_dir / f"{model}_layer{layer}_german_domain_condition.csv"
                de_df = pd.DataFrame([["german-domain"] + list(german_avg)], columns=["domain"] + feat_cols)
                de_df.to_csv(de_out_csv, index=False)
                print(f"  Saved German-domain condition to {de_out_csv.name}")
                
                summary_entries.append({
                    "model": model,
                    "layer": layer,
                    "domain": "german-domain",
                    "embedding_dimension": len(german_avg),
                    "mean_value": float(np.mean(german_avg)),
                    "std_value": float(np.std(german_avg))
                })
            else:
                print("  WARNING: No German files found in the features CSV.")
                
            if not spanish_df.empty:
                spanish_avg = spanish_df[feat_cols].mean(axis=0).values
                # Save Spanish condition
                sp_out_csv = model_tgt_dir / f"{model}_layer{layer}_spanish_domain_condition.csv"
                sp_df = pd.DataFrame([["spanish-domain"] + list(spanish_avg)], columns=["domain"] + feat_cols)
                sp_df.to_csv(sp_out_csv, index=False)
                print(f"  Saved Spanish-domain condition to {sp_out_csv.name}")
                
                summary_entries.append({
                    "model": model,
                    "layer": layer,
                    "domain": "spanish-domain",
                    "embedding_dimension": len(spanish_avg),
                    "mean_value": float(np.mean(spanish_avg)),
                    "std_value": float(np.std(spanish_avg))
                })
            else:
                print("  WARNING: No Spanish files found in the features CSV.")
                
    # Save domain condition summary
    if summary_entries:
        summary_df = pd.DataFrame(summary_entries)
        summary_df.to_csv(SUMMARY_LOG_PATH, index=False)
        print(f"\nSaved domain condition summary to: {SUMMARY_LOG_PATH}")
        print("Step 3 Finished Successfully!\n")
    else:
        print("ERROR: No domain conditions were computed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
