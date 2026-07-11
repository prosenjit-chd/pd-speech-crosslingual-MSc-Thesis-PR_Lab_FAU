#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_prepare_stage5a_refinement_embeddings.py

Purpose:
Prepares source embeddings and domain conditions for refinement:
1. Copies pre-extracted embeddings from stage5_embedding_conditioned_vc/source_embeddings/
   for XLSR L11, WavLM L8, and WavLM L11.
2. Computes the German-domain and Spanish-domain average conditions for each configuration.
3. Saves output files and creates logging summaries.
"""

import os
import sys
import csv
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# Resolve project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
REFINEMENT_DIR = STAGE5_DIR / "stage5a_refinement"

# Inputs
SRC_EMB_DIR = STAGE5_DIR / "source_embeddings"
# Outputs
DEST_EMB_DIR = REFINEMENT_DIR / "source_embeddings_refinement"
DEST_TGT_DIR = REFINEMENT_DIR / "target_domain_embeddings_refinement"
LOGS_DIR = REFINEMENT_DIR / "logs_refinement"
EMB_LOG_PATH = LOGS_DIR / "stage5a_refinement_embedding_log.csv"
DOMAIN_SUMMARY_PATH = LOGS_DIR / "stage5a_refinement_domain_condition_summary.csv"

def main():
    print("=" * 60)
    print("Stage 5A-Refinement Step 2: Preparing Embeddings and Domain Conditions")
    print("=" * 60)

    configs_to_copy = [
        ("xlsr", 11),
        ("wavlm", 8),
        ("wavlm", 11)
    ]

    embedding_logs = []
    domain_summaries = []

    for model, layer in configs_to_copy:
        src_path = SRC_EMB_DIR / model / f"{model}_layer{layer}_stage5a.csv"
        dest_model_dir = DEST_EMB_DIR / model
        dest_model_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_model_dir / f"{model}_layer{layer}_stage5a.csv"
        
        print(f"Processing Model={model.upper()} | Layer={layer} ... ", end="")

        if not src_path.exists():
            print(f"FAILED (source embedding not found at {src_path})")
            embedding_logs.append({
                "model": model,
                "layer": layer,
                "status": "failed",
                "error": "Source file not found"
            })
            continue

        try:
            # 1. Copy source embeddings
            shutil.copy2(src_path, dest_path)
            print("COPIED ... ", end="")
            
            embedding_logs.append({
                "model": model,
                "layer": layer,
                "status": "copied",
                "error": ""
            })

            # 2. Compute averages
            df = pd.read_csv(dest_path)
            feat_cols = [c for c in df.columns if c.startswith('feature_')]
            
            german_df = df[df['language'].str.lower() == 'german']
            spanish_df = df[df['language'].str.lower() == 'spanish']
            
            dest_tgt_model_dir = DEST_TGT_DIR / model
            dest_tgt_model_dir.mkdir(parents=True, exist_ok=True)
            
            if not german_df.empty:
                german_avg = german_df[feat_cols].mean(axis=0).values
                de_out = dest_tgt_model_dir / f"{model}_layer{layer}_german_domain_condition.csv"
                de_df = pd.DataFrame([["german-domain"] + list(german_avg)], columns=["domain"] + feat_cols)
                de_df.to_csv(de_out, index=False)
                
                domain_summaries.append({
                    "model": model,
                    "layer": layer,
                    "domain": "german-domain",
                    "embedding_dimension": len(german_avg),
                    "mean_value": float(np.mean(german_avg)),
                    "std_value": float(np.std(german_avg))
                })
                
            if not spanish_df.empty:
                spanish_avg = spanish_df[feat_cols].mean(axis=0).values
                sp_out = dest_tgt_model_dir / f"{model}_layer{layer}_spanish_domain_condition.csv"
                sp_df = pd.DataFrame([["spanish-domain"] + list(spanish_avg)], columns=["domain"] + feat_cols)
                sp_df.to_csv(sp_out, index=False)
                
                domain_summaries.append({
                    "model": model,
                    "layer": layer,
                    "domain": "spanish-domain",
                    "embedding_dimension": len(spanish_avg),
                    "mean_value": float(np.mean(spanish_avg)),
                    "std_value": float(np.std(spanish_avg))
                })
                
            print("AVERAGED")

        except Exception as e:
            print(f"FAILED ({e})")
            embedding_logs.append({
                "model": model,
                "layer": layer,
                "status": "failed",
                "error": str(e)
            })

    # Write logs
    with open(EMB_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "layer", "status", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for log in embedding_logs:
            writer.writerow(log)
            
    if domain_summaries:
        df_domain = pd.DataFrame(domain_summaries)
        df_domain.to_csv(DOMAIN_SUMMARY_PATH, index=False)
        print(f"Created domain condition summary: {DOMAIN_SUMMARY_PATH}")

    print("Step 2 Finished Successfully!\n")

if __name__ == "__main__":
    main()
