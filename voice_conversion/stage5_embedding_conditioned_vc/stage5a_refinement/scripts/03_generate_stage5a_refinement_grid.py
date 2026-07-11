#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_generate_stage5a_refinement_grid.py

Purpose:
Generates the voice conversion refinement grid:
- 3 embedding conditions: xlsr_layer11, wavlm_layer8, wavlm_layer11
- 5 alphas: 0.1, 0.25, 0.5, 0.75, 1.0
Total settings: 15 settings
Total files generated: 180 audio files
"""

import os
import sys
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from scipy.io.wavfile import write
from sklearn.linear_model import Ridge

# Resolve directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
REFINEMENT_DIR = STAGE5_DIR / "stage5a_refinement"
INPUT_12_DIR = REFINEMENT_DIR / "input_pilot_12_refinement"
SOURCE_EMB_DIR = REFINEMENT_DIR / "source_embeddings_refinement"
TARGET_EMB_DIR = REFINEMENT_DIR / "target_domain_embeddings_refinement"
CONVERTED_MELS_DIR = REFINEMENT_DIR / "converted_mels_refinement"
CONVERTED_AUDIO_DIR = REFINEMENT_DIR / "converted_audio_refinement"
LOGS_DIR = REFINEMENT_DIR / "logs_refinement"
GENERATION_GRID_LOG_PATH = LOGS_DIR / "stage5a_refinement_generation_grid_log.csv"

# Add HiFi-GAN folder to path to import components
HIFIGAN_DIR = PROJECT_ROOT / "voice_conversion" / "hifi-gan"
sys.path.append(str(HIFIGAN_DIR))

try:
    from env import AttrDict
    from meldataset import mel_spectrogram, MAX_WAV_VALUE
    from models import Generator
except ImportError as e:
    print(f"ERROR: Failed to import HiFi-GAN modules: {e}")
    sys.exit(1)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath), f"Checkpoint not found at {filepath}"
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict

def format_alpha_str(alpha):
    # e.g., 0.1 -> 0_10, 0.25 -> 0_25, 0.5 -> 0_50, 0.75 -> 0_75, 1.0 -> 1_00
    val_int = int(round(alpha * 100))
    return f"{val_int // 100}_{val_int % 100:02d}"

def main():
    print("=" * 60)
    print("Stage 5A-Refinement Step 3: Generating Audio Grid")
    print("=" * 60)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load HiFi-GAN Checkpoint and Configuration
    checkpoint_file = PROJECT_ROOT / "voice_conversion" / "checkpoints" / "universal_v1" / "generator_v1"
    config_file = PROJECT_ROOT / "voice_conversion" / "checkpoints" / "universal_v1" / "config.json"

    if not checkpoint_file.exists() or not config_file.exists():
        print(f"ERROR: HiFi-GAN checkpoint or config not found. Check: {checkpoint_file} and {config_file}")
        sys.exit(1)

    with open(config_file) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)

    # Instantiate generator model
    generator = Generator(h).to(device)
    state_dict = load_checkpoint(str(checkpoint_file), device)
    generator.load_state_dict(state_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # Load input selection list (metadata)
    copy_log_path = LOGS_DIR / "stage5a_refinement_input_copy_log.csv"
    if not copy_log_path.exists():
        print(f"ERROR: Selection copy log not found at {copy_log_path}")
        sys.exit(1)
        
    df_files = pd.read_csv(copy_log_path)
    file_list = df_files["stage5_filename"].tolist()

    # 1. Pre-extract and cache log-mel spectrograms of the 12 files (mono, 22050 Hz)
    print("Caching original log-mel spectrograms...")
    original_mels = {}
    original_mel_means = {}
    
    for filename in file_list:
        wav_path = INPUT_12_DIR / filename
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
        max_abs = np.max(np.abs(y))
        if max_abs > 0:
            y = y / (max_abs + 1e-8)
        y_torch = torch.FloatTensor(y).unsqueeze(0).to(device)
        
        with torch.no_grad():
            spec = mel_spectrogram(
                y_torch,
                n_fft=h.n_fft,
                num_mels=h.num_mels,
                sampling_rate=h.sampling_rate,
                hop_size=h.hop_size,
                win_size=h.win_size,
                fmin=h.fmin,
                fmax=h.fmax,
                center=False
            )
            
        spec_np = spec.squeeze(0).cpu().numpy()
        spec_mean = np.mean(spec_np, axis=1)
        original_mels[filename] = spec_np
        original_mel_means[filename] = spec_mean

    # 2. Grid settings
    configs = [
        ("xlsr", 11),
        ("wavlm", 8),
        ("wavlm", 11)
    ]
    alphas = [0.1, 0.25, 0.5, 0.75, 1.0]

    generation_logs = []
    total_files = len(configs) * len(alphas) * len(file_list)
    processed_count = 0

    # For each conditioning model & layer
    for model_type, layer in configs:
        config_name = f"{model_type}_layer{layer}"
        
        # Load pre-copied source and domain embeddings
        src_path = SOURCE_EMB_DIR / model_type / f"{model_type}_layer{layer}_stage5a.csv"
        de_cond_path = TARGET_EMB_DIR / model_type / f"{model_type}_layer{layer}_german_domain_condition.csv"
        sp_cond_path = TARGET_EMB_DIR / model_type / f"{model_type}_layer{layer}_spanish_domain_condition.csv"
        
        if not src_path.exists() or not de_cond_path.exists() or not sp_cond_path.exists():
            print(f"WARNING: Skipping configuration {config_name} due to missing embeddings.")
            continue
            
        df_src = pd.read_csv(src_path)
        df_de = pd.read_csv(de_cond_path)
        df_sp = pd.read_csv(sp_cond_path)
        
        feat_cols = [c for c in df_src.columns if c.startswith('feature_')]
        dim_emb = len(feat_cols)
        
        # Averages
        de_domain_vector = df_de[feat_cols].values[0].astype(np.float32)
        sp_domain_vector = df_sp[feat_cols].values[0].astype(np.float32)
        
        # Map source file name to embedding and language
        embedding_dict = {}
        lang_dict = {}
        for _, row in df_src.iterrows():
            fname = row['file']
            embedding_dict[fname] = row[feat_cols].values.astype(np.float32)
            lang_dict[fname] = row['language']

        # Train Ridge Regression for this configuration (Embedding -> Mel Mean)
        X_train = np.array([embedding_dict[f] for f in file_list])
        y_train = np.array([original_mel_means[f] for f in file_list])
        reg = Ridge(alpha=10.0)
        reg.fit(X_train, y_train)

        # Loop over alphas
        for alpha in alphas:
            alpha_str = format_alpha_str(alpha)
            
            # Subdirectories for audio outputs
            sp_to_de_dir = CONVERTED_AUDIO_DIR / config_name / f"alpha_{alpha_str}" / "spanish_to_german"
            de_to_sp_dir = CONVERTED_AUDIO_DIR / config_name / f"alpha_{alpha_str}" / "german_to_spanish"
            sp_to_de_dir.mkdir(parents=True, exist_ok=True)
            de_to_sp_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- Running grid: {config_name} | alpha={alpha} ---")

            # Process files
            for filename in file_list:
                processed_count += 1
                lang = lang_dict[filename]
                x_emb = embedding_dict[filename]
                source_mel = original_mels[filename]
                
                # Determine targets
                if lang.lower() == "spanish":
                    target_domain = "German"
                    target_emb = de_domain_vector
                    out_dir = sp_to_de_dir
                    direction = "spanish_to_german"
                else:
                    target_domain = "Spanish"
                    target_emb = sp_domain_vector
                    out_dir = de_to_sp_dir
                    direction = "german_to_spanish"
                    
                out_filename = filename.replace(".wav", f"_to_{target_domain[:2]}_domain_{config_name}_alpha_{alpha_str}.wav")
                mel_out_name = filename.replace(".wav", f"_to_{target_domain[:2]}_domain_{config_name}_alpha_{alpha_str}.npy")
                
                dest_audio_path = out_dir / out_filename
                
                print(f"[{processed_count}/{total_files}] Converted: {filename} -> {target_domain}-domain ({alpha_str}) ... ", end="")
                
                try:
                    # Conversion Shift
                    pred_src_mel_mean = reg.predict(x_emb.reshape(1, -1))[0]
                    pred_tgt_mel_mean = reg.predict(target_emb.reshape(1, -1))[0]
                    delta_mel_mean = pred_tgt_mel_mean - pred_src_mel_mean
                    
                    converted_mel = source_mel + alpha * delta_mel_mean.reshape(-1, 1)
                    
                    # Save intermediate mel
                    np.save(CONVERTED_MELS_DIR / mel_out_name, converted_mel)
                    
                    # Vocode spectrogram
                    converted_mel_torch = torch.FloatTensor(converted_mel).unsqueeze(0).to(device)
                    with torch.no_grad():
                        audio_torch = generator(converted_mel_torch)
                        audio = audio_torch.squeeze().cpu().numpy()
                        
                    # Normalize audio and write
                    audio = audio * MAX_WAV_VALUE
                    audio = np.clip(audio, -MAX_WAV_VALUE, MAX_WAV_VALUE - 1)
                    audio = audio.astype(np.int16)
                    
                    write(str(dest_audio_path), h.sampling_rate, audio)
                    print("SUCCESS")
                    
                    generation_logs.append({
                        "source_file": filename,
                        "source_language": lang,
                        "target_domain": f"{target_domain}-domain",
                        "condition_model": model_type,
                        "condition_layer": layer,
                        "alpha": alpha,
                        "output_file": str(dest_audio_path.relative_to(REFINEMENT_DIR)),
                        "generation_status": "success",
                        "method_used": "prototype embedding-conditioned conversion",
                        "notes": f"Ridge mapping embedding-to-mel shift, scale alpha={alpha}"
                    })
                    
                except Exception as e:
                    print(f"FAILED ({e})")
                    generation_logs.append({
                        "source_file": filename,
                        "source_language": lang,
                        "target_domain": f"{target_domain}-domain",
                        "condition_model": model_type,
                        "condition_layer": layer,
                        "alpha": alpha,
                        "output_file": "",
                        "generation_status": f"failed ({type(e).__name__})",
                        "method_used": "prototype embedding-conditioned conversion",
                        "notes": str(e)
                    })

    # Save generation log
    with open(GENERATION_GRID_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_file", "source_language", "target_domain", 
            "condition_model", "condition_layer", "alpha", 
            "output_file", "generation_status", "method_used", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for log in generation_logs:
            writer.writerow(log)
            
    print(f"\nSaved generation grid log: {GENERATION_GRID_LOG_PATH}")
    print("Step 3 Finished Successfully!\n")

if __name__ == "__main__":
    main()
