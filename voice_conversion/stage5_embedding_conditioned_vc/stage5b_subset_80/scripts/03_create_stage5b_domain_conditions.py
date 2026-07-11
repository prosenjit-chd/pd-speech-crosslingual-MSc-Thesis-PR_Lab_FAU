#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_create_stage5b_domain_conditions.py

Purpose:
Averages target domain conditions using XLSR layer 11 embeddings:
- Spanish-domain average (from the 40 Spanish files)
- German-domain average (from the 40 German files)
Saves results to target_domain_embeddings_stage5b/ and summarizes.
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd

# Resolve directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STAGE5B_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc" / "stage5b_subset_80"
EMB_CSV = STAGE5B_DIR / "source_embeddings_stage5b" / "xlsr_layer11_stage5b.csv"
DEST_TGT_DIR = STAGE5B_DIR / "target_domain_embeddings_stage5b"
LOGS_DIR = STAGE5B_DIR / "logs_stage5b"
SUMMARY_CSV = LOGS_DIR / "stage5b_domain_condition_summary.csv"

def main():
    print("=" * 60)
    print("Stage 5B Step 3: Computing Domain Condition Embeddings")
    print("=" * 60)

    if not EMB_CSV.exists():
        print(f"ERROR: Embeddings CSV not found at {EMB_CSV}. Run step 2 first.")
        sys.exit(1)

    df = pd.read_csv(EMB_CSV)
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    print(f"Loaded embeddings matrix: {df.shape} | Feature dimensions: {len(feat_cols)}")

    german_df = df[df["language"].str.lower() == "german"]
    spanish_df = df[df["language"].str.lower() == "spanish"]

    print(f"German files: {len(german_df)} | Spanish files: {len(spanish_df)}")

    DEST_TGT_DIR.mkdir(parents=True, exist_ok=True)
    summary_records = []

    # German domain condition
    if not german_df.empty:
        german_avg = german_df[feat_cols].mean(axis=0).values
        de_out = DEST_TGT_DIR / "xlsr_layer11_german_domain_condition.csv"
        de_df = pd.DataFrame([["german-domain"] + list(german_avg)], columns=["domain"] + feat_cols)
        de_df.to_csv(de_out, index=False)
        print(f"Saved German average target condition: {de_out}")
        
        summary_records.append({
            "condition_model": "xlsr",
            "condition_layer": 11,
            "domain": "german-domain",
            "num_files": len(german_df),
            "embedding_dimension": len(german_avg),
            "mean_embedding_value": float(np.mean(german_avg)),
            "std_embedding_value": float(np.std(german_avg))
        })

    # Spanish domain condition
    if not spanish_df.empty:
        spanish_avg = spanish_df[feat_cols].mean(axis=0).values
        sp_out = DEST_TGT_DIR / "xlsr_layer11_spanish_domain_condition.csv"
        sp_df = pd.DataFrame([["spanish-domain"] + list(spanish_avg)], columns=["domain"] + feat_cols)
        sp_df.to_csv(sp_out, index=False)
        print(f"Saved Spanish average target condition: {sp_out}")
        
        summary_records.append({
            "condition_model": "xlsr",
            "condition_layer": 11,
            "domain": "spanish-domain",
            "num_files": len(spanish_df),
            "embedding_dimension": len(spanish_avg),
            "mean_embedding_value": float(np.mean(spanish_avg)),
            "std_embedding_value": float(np.std(spanish_avg))
        })

    if summary_records:
        df_sum = pd.DataFrame(summary_records)
        df_sum.to_csv(SUMMARY_CSV, index=False)
        print(f"Saved domain condition summary log: {SUMMARY_CSV}")

    print("Step 3 Finished Successfully!\n")

if __name__ == "__main__":
    main()
