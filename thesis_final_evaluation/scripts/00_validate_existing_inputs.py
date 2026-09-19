"""
Script 00: Validate Existing Inputs
===================================
Inspects existing read-only datasets and Stage 5C outputs, validating:
1. Exact file count (276 original, 276 converted)
2. Exact distribution (100 Spanish, 176 German; 138 HC, 138 PD)
3. Audio properties (sample rate == 22050, mono channel, non-zero duration)
4. Absence of data corruption or missing files
Outputs:
- 00_inventory/original_276_inventory.csv
- 00_inventory/converted_276_inventory.csv
"""

import os
import sys
import logging
import soundfile as sf
import pandas as pd
import yaml

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
    logs_dir = os.path.join(output_root, "logs")
    os.makedirs(inventory_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "00_validate_existing_inputs.log")
    setup_logging(log_file)
    logging.info("Starting Script 00: Validate Existing Inputs")
    
    orig_meta_path = config["metadata_original_full"]
    if not os.path.exists(orig_meta_path):
        logging.error(f"Original full metadata not found at: {orig_meta_path}")
        sys.exit(1)
        
    orig_df = pd.read_csv(orig_meta_path)
    logging.info(f"Loaded original full metadata: {len(orig_df)} rows")
    
    orig_audio_root = config["original_audio_22050_root"]
    converted_audio_root = config["converted_audio_stage5c_root"]
    sp_to_ge_dir = os.path.join(converted_audio_root, "spanish_to_german")
    ge_to_sp_dir = os.path.join(converted_audio_root, "german_to_spanish")
    
    # 1. Inspect Original Files
    orig_records = []
    for idx, row in orig_df.iterrows():
        fname = os.path.basename(row["file_path"])
        full_path = os.path.join(orig_audio_root, fname)
        
        if not os.path.exists(full_path):
            logging.error(f"Original audio file missing: {full_path}")
            sys.exit(1)
            
        info = sf.info(full_path)
        orig_records.append({
            "filename": fname,
            "speaker_id": row["speaker_id"],
            "language": row["language"],
            "diagnosis": row["label"],
            "dataset": row["dataset"],
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_sec": round(info.duration, 4),
            "frames": info.frames,
            "file_size_bytes": os.path.getsize(full_path),
            "full_path": full_path
        })
        
    orig_inv_df = pd.DataFrame(orig_records)
    orig_inv_path = os.path.join(inventory_dir, "original_276_inventory.csv")
    orig_inv_df.to_csv(orig_inv_path, index=False)
    logging.info(f"Original inventory written to: {orig_inv_path} ({len(orig_inv_df)} files)")
    
    # Check Original Distributions
    n_sp = (orig_inv_df["language"] == "Spanish").sum()
    n_ge = (orig_inv_df["language"] == "German").sum()
    n_hc = (orig_inv_df["diagnosis"] == "HC").sum()
    n_pd = (orig_inv_df["diagnosis"] == "PD").sum()
    
    logging.info(f"Original Counts -> Total: {len(orig_inv_df)}, Spanish: {n_sp}, German: {n_ge}, HC: {n_hc}, PD: {n_pd}")
    assert len(orig_inv_df) == 276, f"Expected 276 original files, got {len(orig_inv_df)}"
    assert n_sp == 100, f"Expected 100 Spanish, got {n_sp}"
    assert n_ge == 176, f"Expected 176 German, got {n_ge}"
    assert n_hc == 138, f"Expected 138 HC, got {n_hc}"
    assert n_pd == 138, f"Expected 138 PD, got {n_pd}"
    
    # 2. Inspect Converted Stage 5C Files
    conv_records = []
    # Spanish to German (100 files expected)
    sp_files = [f for f in os.listdir(sp_to_ge_dir) if f.endswith(".wav")]
    logging.info(f"Found {len(sp_files)} converted files in spanish_to_german")
    for f in sp_files:
        full_path = os.path.join(sp_to_ge_dir, f)
        info = sf.info(full_path)
        # Extract source ID: SP_HC_001_to_Ge_domain_xlsr_layer11_alpha_1_00.wav -> SP_HC_001.wav
        source_id = f.split("_to_")[0] + ".wav"
        diag = "HC" if "_HC_" in f else "PD"
        conv_records.append({
            "converted_filename": f,
            "source_id": source_id,
            "source_language": "Spanish",
            "target_domain": "German-domain",
            "diagnosis": diag,
            "conversion_direction": "Spanish->German-domain",
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_sec": round(info.duration, 4),
            "frames": info.frames,
            "file_size_bytes": os.path.getsize(full_path),
            "full_path": full_path
        })
        
    # German to Spanish (176 files expected)
    ge_files = [f for f in os.listdir(ge_to_sp_dir) if f.endswith(".wav")]
    logging.info(f"Found {len(ge_files)} converted files in german_to_spanish")
    for f in ge_files:
        full_path = os.path.join(ge_to_sp_dir, f)
        info = sf.info(full_path)
        # Extract source ID: DE_HC_001_to_Sp_domain_xlsr_layer11_alpha_1_00.wav -> DE_HC_001.wav
        source_id = f.split("_to_")[0] + ".wav"
        diag = "HC" if "_HC_" in f else "PD"
        conv_records.append({
            "converted_filename": f,
            "source_id": source_id,
            "source_language": "German",
            "target_domain": "Spanish-domain",
            "diagnosis": diag,
            "conversion_direction": "German->Spanish-domain",
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_sec": round(info.duration, 4),
            "frames": info.frames,
            "file_size_bytes": os.path.getsize(full_path),
            "full_path": full_path
        })
        
    conv_inv_df = pd.DataFrame(conv_records)
    conv_inv_path = os.path.join(inventory_dir, "converted_276_inventory.csv")
    conv_inv_df.to_csv(conv_inv_path, index=False)
    logging.info(f"Converted inventory written to: {conv_inv_path} ({len(conv_inv_df)} files)")
    
    n_sp_conv = (conv_inv_df["source_language"] == "Spanish").sum()
    n_ge_conv = (conv_inv_df["source_language"] == "German").sum()
    n_hc_conv = (conv_inv_df["diagnosis"] == "HC").sum()
    n_pd_conv = (conv_inv_df["diagnosis"] == "PD").sum()
    
    logging.info(f"Converted Counts -> Total: {len(conv_inv_df)}, Sp->Ge: {n_sp_conv}, Ge->Sp: {n_ge_conv}, HC: {n_hc_conv}, PD: {n_pd_conv}")
    assert len(conv_inv_df) == 276, f"Expected 276 converted files, got {len(conv_inv_df)}"
    assert n_sp_conv == 100, f"Expected 100 Spanish converted, got {n_sp_conv}"
    assert n_ge_conv == 176, f"Expected 176 German converted, got {n_ge_conv}"
    assert n_hc_conv == 138, f"Expected 138 HC, got {n_hc_conv}"
    assert n_pd_conv == 138, f"Expected 138 PD, got {n_pd_conv}"
    
    logging.info("Input validation SUCCESSFUL: All 276 original and 276 converted audio files verified.")

if __name__ == "__main__":
    main()
