"""
Script 02: Extract Original Acoustic Features
=============================================
Extracts interpretable acoustic features for all 276 original speech recordings.
"""

import os
import sys
import logging
import time
import pandas as pd
import yaml
from tqdm import tqdm

from acoustic_feature_extractor import extract_features_from_audio

def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    output_root = config["output_root"]
    inventory_dir = os.path.join(output_root, "00_inventory")
    features_dir = os.path.join(output_root, "01_acoustic_features")
    logs_dir = os.path.join(output_root, "logs")
    os.makedirs(features_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "02_extract_original_acoustic_features.log")
    setup_logging(log_file)
    logging.info("Starting Script 02: Extract Original Acoustic Features")
    
    orig_inv_path = os.path.join(inventory_dir, "original_276_inventory.csv")
    orig_df = pd.read_csv(orig_inv_path)
    logging.info(f"Loaded original inventory: {len(orig_df)} recordings")
    
    results = []
    failures = []
    
    start_time = time.time()
    for idx, row in tqdm(orig_df.iterrows(), total=len(orig_df), desc="Extracting Original Features"):
        full_path = row["full_path"]
        feats, errs = extract_features_from_audio(full_path, target_sr=config["target_sample_rate"])
        
        record = {
            "filename": row["filename"],
            "speaker_id": row["speaker_id"],
            "language": row["language"],
            "diagnosis": row["diagnosis"],
            "dataset": row["dataset"]
        }
        if feats is not None:
            record.update(feats)
        results.append(record)
        
        if errs:
            for feat_type, msg in errs.items():
                failures.append({
                    "stage": "original",
                    "filename": row["filename"],
                    "feature_type": feat_type,
                    "error_message": msg
                })
                
    elapsed = time.time() - start_time
    logging.info(f"Feature extraction completed in {elapsed:.2f} s ({elapsed/len(orig_df):.2f} s/file)")
    
    out_df = pd.DataFrame(results)
    out_path = os.path.join(features_dir, "original_acoustic_features.csv")
    out_df.to_csv(out_path, index=False)
    logging.info(f"Saved original acoustic features: {out_path} ({len(out_df)} rows, {len(out_df.columns)} columns)")
    
    # Save any failures
    if failures:
        fail_df = pd.DataFrame(failures)
        fail_path = os.path.join(features_dir, "extraction_failures_original.csv")
        fail_df.to_csv(fail_path, index=False)
        logging.warning(f"Recorded {len(failures)} feature-level extraction warnings/failures: {fail_path}")
    else:
        logging.info("Zero extraction failures recorded.")

if __name__ == "__main__":
    main()
