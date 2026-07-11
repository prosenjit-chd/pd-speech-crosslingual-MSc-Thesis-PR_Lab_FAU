#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_generate_stage5a_conditioned_audio.py

Purpose:
Performs prototype embedding-conditioned voice conversion:
1. Loads the pre-extracted XLSR layer 11 embeddings of the 12 files.
2. Extracts log-mel spectrograms from the 12 files at 22050 Hz.
3. Fits a Ridge regression mapping from XLSR layer 11 embeddings (1024-dim) to 
   time-averaged log-mel spectrograms (80-dim).
4. Performs conversion:
   - For Spanish files, shifts toward German-domain condition.
   - For German files, shifts toward Spanish-domain condition.
5. Saves intermediate log-mel spectrograms to converted_mels_stage5/.
6. Synthesizes/vocodes the converted log-mel spectrograms using the universal_v1 HiFi-GAN model.
7. Logs details as "prototype embedding-conditioned conversion".
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STAGE5_DIR = PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc"
INPUT_12_DIR = STAGE5_DIR / "input_pilot_12"
SOURCE_EMB_DIR = STAGE5_DIR / "source_embeddings"
TARGET_EMB_DIR = STAGE5_DIR / "target_domain_embeddings"
CONVERTED_MELS_DIR = STAGE5_DIR / "converted_mels_stage5"
SP_TO_DE_DIR = STAGE5_DIR / "converted_spanish_to_german"
DE_TO_SP_DIR = STAGE5_DIR / "converted_german_to_spanish"
LOG_DIR = STAGE5_DIR / "logs_stage5"
GENERATION_LOG_PATH = LOG_DIR / "stage5a_conditioned_generation_log.csv"

# Add HiFi-GAN folder to path to import components
HIFIGAN_DIR = PROJECT_ROOT / "voice_conversion" / "hifi-gan"
sys.path.append(str(HIFIGAN_DIR))

# HiFi-GAN imports (will be resolved because of sys.path append)
try:
    from env import AttrDict
    from meldataset import mel_spectrogram, MAX_WAV_VALUE
    from models import Generator
except ImportError as e:
    print(f"ERROR: Failed to import HiFi-GAN modules: {e}")
    sys.exit(1)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath), f"Checkpoint not found at {filepath}"
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def main():
    print("=" * 60)
    print("Stage 5A Step 4: Generating/Converting Conditioned Audio")
    print("=" * 60)

    # Ensure directories exist
    SP_TO_DE_DIR.mkdir(parents=True, exist_ok=True)
    DE_TO_SP_DIR.mkdir(parents=True, exist_ok=True)
    CONVERTED_MELS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load HiFi-GAN Checkpoint and Configuration
    checkpoint_file = PROJECT_ROOT / "voice_conversion" / "checkpoints" / "universal_v1" / "generator_v1"
    config_file = PROJECT_ROOT / "voice_conversion" / "checkpoints" / "universal_v1" / "config.json"

    if not checkpoint_file.exists() or not config_file.exists():
        print(f"ERROR: HiFi-GAN checkpoint or config not found. Check: {checkpoint_file} and {config_file}")
        sys.exit(1)

    with open(config_file) as f:
        json_config = json.loads(f.read())
    
    h = AttrDict(json_config)
    
    # Verify mel configuration parameters from user instruction 5
    print("\n--- Verifying HiFi-GAN Mel Configuration ---")
    print(f"  Expected: SR=22050, Mels=80, FFT=1024, Hop=256, Win=1024, fmin=0")
    print(f"  Actual:   SR={h.sampling_rate}, Mels={h.num_mels}, FFT={h.n_fft}, Hop={h.hop_size}, Win={h.win_size}, fmin={h.fmin}, fmax={h.fmax}")
    
    assert h.sampling_rate == 22050, f"Sampling rate mismatch: {h.sampling_rate}"
    assert h.num_mels == 80, f"Num mels mismatch: {h.num_mels}"
    assert h.n_fft == 1024, f"n_fft mismatch: {h.n_fft}"
    assert h.hop_size == 256, f"hop_size mismatch: {h.hop_size}"
    assert h.win_size == 1024, f"win_size mismatch: {h.win_size}"
    assert h.fmin == 0, f"fmin mismatch: {h.fmin}"
    print("HiFi-GAN config parameters successfully verified!\n")

    # Instantiate generator model
    generator = Generator(h).to(device)
    state_dict = load_checkpoint(str(checkpoint_file), device)
    generator.load_state_dict(state_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # 2. Load XLSR Layer 11 Embeddings (Source and Domain)
    xlsr_src_path = SOURCE_EMB_DIR / "xlsr" / "xlsr_layer11_stage5a.csv"
    de_cond_path = TARGET_EMB_DIR / "xlsr" / "xlsr_layer11_german_domain_condition.csv"
    sp_cond_path = TARGET_EMB_DIR / "xlsr" / "xlsr_layer11_spanish_domain_condition.csv"

    if not xlsr_src_path.exists():
        print(f"ERROR: Source XLSR layer 11 embeddings not found at {xlsr_src_path}. Run step 2 first.")
        sys.exit(1)
    if not de_cond_path.exists() or not sp_cond_path.exists():
        print(f"ERROR: Domain condition embeddings not found. Run step 3 first.")
        sys.exit(1)

    src_df = pd.read_csv(xlsr_src_path)
    de_cond_df = pd.read_csv(de_cond_path)
    sp_cond_df = pd.read_csv(sp_cond_path)

    feat_cols = [c for c in src_df.columns if c.startswith('feature_')]
    assert len(feat_cols) == 1024, f"XLSR layer 11 must have 1024 features, got {len(feat_cols)}"

    # Domain conditions
    de_domain_vector = de_cond_df[feat_cols].values[0]
    sp_domain_vector = sp_cond_df[feat_cols].values[0]

    # Map file name to embedding
    embedding_dict = {}
    metadata_dict = {}
    for _, row in src_df.iterrows():
        file_name = row['file']
        embedding_dict[file_name] = row[feat_cols].values.astype(np.float32)
        metadata_dict[file_name] = {
            "language": row['language'],
            "label": row['label'],
            "group": row['group']
        }

    # 3. Extract log-mel spectrograms and compute time-averaged mel spectrograms
    print("Extracting log-mel spectrograms for Ridge regression training...")
    mel_dict = {}
    mel_mean_dict = {}
    
    for filename in embedding_dict.keys():
        wav_path = INPUT_12_DIR / filename
        # Load audio at target sample rate 22050 Hz and convert to mono
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
        # Peak normalization
        max_abs = np.max(np.abs(y))
        if max_abs > 0:
            y = y / (max_abs + 1e-8)
            
        y_torch = torch.FloatTensor(y).unsqueeze(0).to(device)
        
        # Extract mel spectrogram using HiFi-GAN config parameters
        with torch.no_grad():
            # center=False to match HiFi-GAN training configuration
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
            
        spec_np = spec.squeeze(0).cpu().numpy() # [80, num_frames]
        spec_mean = np.mean(spec_np, axis=1) # [80]
        
        mel_dict[filename] = spec_np
        mel_mean_dict[filename] = spec_mean

    # 4. Train Ridge Regression (Embedding -> Mel Mean)
    print("\nTraining Ridge Regression (XLSR Layer 11 Embedding -> Mel Mean)...")
    X_train = np.array(list(embedding_dict.values())) # [12, 1024]
    y_train = np.array(list(mel_mean_dict.values())) # [12, 80]
    
    # alpha=10.0 regularizes the mapping to avoid overfitting on 12 samples
    reg = Ridge(alpha=10.0)
    reg.fit(X_train, y_train)
    print("Ridge regression model trained successfully.")

    # 5. Conversion and Generation
    print("\nStarting prototype embedding-conditioned voice conversion...")
    generation_logs = []
    alpha = 0.5 # conversion scale

    for filename, x_emb in embedding_dict.items():
        meta = metadata_dict[filename]
        lang = meta["language"]
        
        # Source mel
        source_mel = mel_dict[filename] # [80, num_frames]
        
        if lang.lower() == "spanish":
            target_domain = "German"
            target_emb = de_domain_vector
            out_dir = SP_TO_DE_DIR
            out_filename = filename.replace(".wav", "_to_DE_domain.wav")
            mel_out_name = filename.replace(".wav", "_to_DE_domain.npy")
        else:
            target_domain = "Spanish"
            target_emb = sp_domain_vector
            out_dir = DE_TO_SP_DIR
            out_filename = filename.replace(".wav", "_to_SP_domain.wav")
            mel_out_name = filename.replace(".wav", "_to_SP_domain.npy")
            
        print(f"Converting {filename} ({lang}) -> {target_domain}-domain ... ", end="")
        
        try:
            # Predict mean mel difference from embedding difference
            # Delta_m = W * (target_emb - x_emb)
            pred_src_mel_mean = reg.predict(x_emb.reshape(1, -1))[0]
            pred_tgt_mel_mean = reg.predict(target_emb.reshape(1, -1))[0]
            delta_mel_mean = pred_tgt_mel_mean - pred_src_mel_mean
            
            # Apply shift to source mel frames
            converted_mel = source_mel + alpha * delta_mel_mean.reshape(-1, 1)
            
            # Save intermediate mel spectrogram
            np.save(CONVERTED_MELS_DIR / mel_out_name, converted_mel)
            
            # Vocode log-mel spectrogram back to audio using generator
            converted_mel_torch = torch.FloatTensor(converted_mel).unsqueeze(0).to(device)
            with torch.no_grad():
                audio_torch = generator(converted_mel_torch)
                audio = audio_torch.squeeze().cpu().numpy()
                
            # Scale and write WAV (16-bit PCM)
            audio = audio * MAX_WAV_VALUE
            audio = np.clip(audio, -MAX_WAV_VALUE, MAX_WAV_VALUE - 1)
            audio = audio.astype(np.int16)
            
            out_file_path = out_dir / out_filename
            write(str(out_file_path), h.sampling_rate, audio)
            print("SUCCESS")
            
            generation_logs.append({
                "source_file": filename,
                "source_language": lang,
                "target_domain": f"{target_domain}-domain",
                "model_condition": "xlsr",
                "layer_condition": 11,
                "output_file": str(out_file_path.relative_to(STAGE5_DIR)),
                "generation_status": "success",
                "method_used": "prototype embedding-conditioned conversion",
                "notes": f"Ridge mapping embedding-to-mel shift, conversion scale alpha={alpha}"
            })
            
        except Exception as e:
            print(f"FAILED ({e})")
            generation_logs.append({
                "source_file": filename,
                "source_language": lang,
                "target_domain": f"{target_domain}-domain",
                "model_condition": "xlsr",
                "layer_condition": 11,
                "output_file": "",
                "generation_status": f"failed ({type(e).__name__})",
                "method_used": "prototype embedding-conditioned conversion",
                "notes": str(e)
            })

    # Save generation log
    with open(GENERATION_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_file", "source_language", "target_domain", 
            "model_condition", "layer_condition", "output_file", 
            "generation_status", "method_used", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for log in generation_logs:
            writer.writerow(log)
            
    print(f"\nGeneration log written to: {GENERATION_LOG_PATH}")
    print("Step 4 Finished Successfully!\n")

if __name__ == "__main__":
    main()
