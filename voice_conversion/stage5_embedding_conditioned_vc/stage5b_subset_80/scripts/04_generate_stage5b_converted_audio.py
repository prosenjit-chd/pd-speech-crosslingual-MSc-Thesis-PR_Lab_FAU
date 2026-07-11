#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_generate_stage5b_converted_audio.py

Purpose:
Executes prototype embedding-conditioned voice conversion on the 80 files:
- Selected configuration: XLSR Layer 11, Alpha = 1.0.
- Fits Ridge regression: Embedding -> Log-mel spectrogram mean.
- Shifts mel spectrogram: M_converted(t) = M_source(t) + 1.0 * Delta m.
- Vocodes via pretrained universal_v1 HiFi-GAN.
- Logs clipping metrics for each output.
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
STAGE5B_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc" / "stage5b_subset_80"
INPUT_DIR = STAGE5B_DIR / "input_subset_80_stage5b"
SOURCE_EMB_PATH = STAGE5B_DIR / "source_embeddings_stage5b" / "xlsr_layer11_stage5b.csv"
DE_COND_PATH = STAGE5B_DIR / "target_domain_embeddings_stage5b" / "xlsr_layer11_german_domain_condition.csv"
SP_COND_PATH = STAGE5B_DIR / "target_domain_embeddings_stage5b" / "xlsr_layer11_spanish_domain_condition.csv"
CONVERTED_MELS_DIR = STAGE5B_DIR / "converted_mels_stage5b"
CONVERTED_AUDIO_DIR = STAGE5B_DIR / "converted_audio_stage5b"
LOGS_DIR = STAGE5B_DIR / "logs_stage5b"
GENERATION_LOG_PATH = LOGS_DIR / "stage5b_conditioned_generation_log.csv"

# Add HiFi-GAN folder to path
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

def main():
    print("=" * 60)
    print("Stage 5B Step 4: Generating Converted Speech Grid")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load HiFi-GAN Checkpoint and Configuration
    checkpoint_file = PROJECT_ROOT / "voice_conversion" / "checkpoints" / "universal_v1" / "generator_v1"
    config_file = PROJECT_ROOT / "voice_conversion" / "checkpoints" / "universal_v1" / "config.json"

    if not checkpoint_file.exists() or not config_file.exists():
        print(f"ERROR: HiFi-GAN checkpoint/config not found.")
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

    # Load embeddings and domains
    if not SOURCE_EMB_PATH.exists() or not DE_COND_PATH.exists() or not SP_COND_PATH.exists():
        print("ERROR: Pre-computed embeddings or domain conditions are missing.")
        sys.exit(1)

    df_src = pd.read_csv(SOURCE_EMB_PATH)
    df_de = pd.read_csv(DE_COND_PATH)
    df_sp = pd.read_csv(SP_COND_PATH)
    
    file_list = df_src["file"].tolist()
    feat_cols = [c for c in df_src.columns if c.startswith('feature_')]
    
    de_domain_vector = df_de[feat_cols].values[0].astype(np.float32)
    sp_domain_vector = df_sp[feat_cols].values[0].astype(np.float32)
    
    embedding_dict = {}
    lang_dict = {}
    for _, row in df_src.iterrows():
        fname = row['file']
        embedding_dict[fname] = row[feat_cols].values.astype(np.float32)
        lang_dict[fname] = row['language']

    # 1. Pre-extract and cache log-mel spectrograms of the 80 files
    print("Caching original log-mel spectrograms...")
    original_mels = {}
    original_mel_means = {}
    
    for filename in file_list:
        wav_path = INPUT_DIR / filename
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
        original_mels[filename] = spec_np
        original_mel_means[filename] = np.mean(spec_np, axis=1)

    # 2. Train Ridge Regression (Embedding -> Mel Mean)
    X_train = np.array([embedding_dict[f] for f in file_list])
    y_train = np.array([original_mel_means[f] for f in file_list])
    
    print("Fitting Ridge regression model...")
    reg = Ridge(alpha=10.0)
    reg.fit(X_train, y_train)

    # Ensure output dirs exist
    sp_to_de_dir = CONVERTED_AUDIO_DIR / "spanish_to_german"
    de_to_sp_dir = CONVERTED_AUDIO_DIR / "german_to_spanish"
    sp_to_de_dir.mkdir(parents=True, exist_ok=True)
    de_to_sp_dir.mkdir(parents=True, exist_ok=True)
    CONVERTED_MELS_DIR.mkdir(parents=True, exist_ok=True)

    generation_logs = []
    total_files = len(file_list)
    
    print("Starting generation...")
    for idx, filename in enumerate(file_list):
        lang = lang_dict[filename]
        x_emb = embedding_dict[filename]
        source_mel = original_mels[filename]
        
        alpha = 1.0
        
        if lang.lower() == "spanish":
            target_domain = "German"
            target_emb = de_domain_vector
            out_dir = sp_to_de_dir
        else:
            target_domain = "Spanish"
            target_emb = sp_domain_vector
            out_dir = de_to_sp_dir
            
        out_filename = filename.replace(".wav", f"_to_{target_domain[:2]}_domain_xlsr_layer11_alpha_1_00.wav")
        mel_out_name = filename.replace(".wav", f"_to_{target_domain[:2]}_domain_xlsr_layer11_alpha_1_00.npy")
        
        dest_audio_path = out_dir / out_filename
        
        print(f"[{idx+1}/{total_files}] Converting {filename} -> {target_domain}-domain ... ", end="")
        
        try:
            # 3. Conversion Shift
            pred_src_mel_mean = reg.predict(x_emb.reshape(1, -1))[0]
            pred_tgt_mel_mean = reg.predict(target_emb.reshape(1, -1))[0]
            delta_mel_mean = pred_tgt_mel_mean - pred_src_mel_mean
            
            converted_mel = source_mel + alpha * delta_mel_mean.reshape(-1, 1)
            
            # Save intermediate mel
            np.save(CONVERTED_MELS_DIR / mel_out_name, converted_mel)
            
            # 4. Vocode
            converted_mel_torch = torch.FloatTensor(converted_mel).unsqueeze(0).to(device)
            with torch.no_grad():
                audio_torch = generator(converted_mel_torch)
                audio_float = audio_torch.squeeze().cpu().numpy()
            
            # Monitor clipping before normalization/scaling
            clipping_mask = (audio_float >= 1.0) | (audio_float <= -1.0)
            clipped_samples = int(np.sum(clipping_mask))
            max_val = float(np.max(np.abs(audio_float)))
            
            # Cast and write WAV
            audio = audio_float * MAX_WAV_VALUE
            audio = np.clip(audio, -MAX_WAV_VALUE, MAX_WAV_VALUE - 1)
            audio = audio.astype(np.int16)
            
            write(str(dest_audio_path), h.sampling_rate, audio)
            
            clipping_note = f"Clipped samples: {clipped_samples} (max peak={max_val:.4f})."
            if clipped_samples > 0:
                print(f"SUCCESS (Warning: {clipping_note})")
            else:
                print("SUCCESS")
                
            generation_logs.append({
                "source_file": filename,
                "source_language": lang,
                "target_domain": f"{target_domain}-domain",
                "condition_model": "xlsr",
                "condition_layer": 11,
                "alpha": alpha,
                "output_file": str(dest_audio_path.relative_to(STAGE5B_DIR)),
                "generation_status": "success",
                "method_used": "prototype embedding-conditioned conversion",
                "notes": clipping_note
            })
            
        except Exception as e:
            print(f"FAILED ({e})")
            generation_logs.append({
                "source_file": filename,
                "source_language": lang,
                "target_domain": f"{target_domain}-domain",
                "condition_model": "xlsr",
                "condition_layer": 11,
                "alpha": alpha,
                "output_file": "",
                "generation_status": f"failed ({type(e).__name__})",
                "method_used": "prototype embedding-conditioned conversion",
                "notes": str(e)
            })

    # Save generation log
    with open(GENERATION_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_file", "source_language", "target_domain", 
            "condition_model", "condition_layer", "alpha", 
            "output_file", "generation_status", "method_used", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for log in generation_logs:
            writer.writerow(log)
            
    print(f"\nSaved generation log: {GENERATION_LOG_PATH}")
    print("Step 4 Finished Successfully!\n")

if __name__ == "__main__":
    main()
