#!/usr/bin/env python3
"""
05_final_regeneration.py
Regenerates ALL 6 Spanish -> German Pilot Files using the Selected Winning Configuration.

Implements Section 25:
- Loads winning configuration from 09_model_selection/selection_decision.json
- Runs in env_xtts (GPU accelerated)
- Generates all six files from scratch
- Saves to:
  10_final_regeneration/spanish_to_german/{safe_id}_final_to_German.wav
- Generates 10_final_regeneration/final_synthesis_manifest.csv
"""

import os
os.environ["COQUI_TOS_AGREED"] = "1"
import sys
if sys.platform == "win32":
    if os.path.exists(r"C:\Windows\System32"):
        os.add_dll_directory(r"C:\Windows\System32")
import torch
import functools
import time
import json
from pathlib import Path
import soundfile as sf
import numpy as np
import pandas as pd

# Patch torch.load for PyTorch 2.6+
_orig_torch_load = torch.load
@functools.wraps(_orig_torch_load)
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Patch torchaudio.load with soundfile to avoid TorchCodec requirement
import torchaudio
_orig_torchaudio_load = torchaudio.load
def _soundfile_load(filepath, *args, **kwargs):
    try:
        data, sr = sf.read(str(filepath))
        if data.ndim == 1:
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        else:
            tensor = torch.tensor(data.T, dtype=torch.float32)
        return tensor, sr
    except Exception:
        return _orig_torchaudio_load(filepath, *args, **kwargs)
torchaudio.load = _soundfile_load

WORKSPACE = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement")
CHUNKING_DIR = WORKSPACE / "05_chunking_search"
REF_DIR = WORKSPACE / "03_reference_audio_analysis"
FINAL_OUT_DIR = WORKSPACE / "10_final_regeneration" / "spanish_to_german"
SELECTION_FILE = WORKSPACE / "09_model_selection" / "selection_decision.json"
CANDIDATE_CONFIGS = WORKSPACE / "04_xtts_parameter_search" / "candidate_configurations.json"
PILOT_CSV = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\04_pilot_selection\pilot_12_selection.csv")

def load_chunks(strategy_name: str) -> list:
    json_path = CHUNKING_DIR / f"{strategy_name}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]

def get_reference_audio(safe_id: str, ref_type: str, ref_df: pd.DataFrame):
    row = ref_df[ref_df["safe_id"] == safe_id].iloc[0]
    if ref_type == "R1":
        return str(row["R1_path"])
    elif ref_type == "R2":
        return str(row["R2_path"])
    elif ref_type == "R3":
        return [str(row["R3_part1_path"]), str(row["R3_part2_path"])]
    else:
        raise ValueError(f"Unknown ref_type: {ref_type}")

def synthesize_chunks(tts, chunks, speaker_ref, params):
    sr = 24000
    silence_gap = np.zeros(int(0.20 * sr), dtype=np.float32)
    concatenated = []
    
    for chunk_text in chunks:
        wav_out = tts.tts(
            text=chunk_text,
            speaker_wav=speaker_ref,
            language="de",
            temperature=params["temperature"],
            repetition_penalty=params["repetition_penalty"],
            top_p=params["top_p"],
            speed=params["speed"]
        )
        wav_arr = np.array(wav_out, dtype=np.float32)
        if len(concatenated) > 0:
            concatenated.append(silence_gap)
        concatenated.append(wav_arr)
        
    full_audio = np.concatenate(concatenated)
    peak = np.max(np.abs(full_audio))
    if peak > 0:
        full_audio = full_audio * (0.95 / peak)  # Guaranteed 0.95 peak, zero clipping
        
    return full_audio, sr

def main():
    FINAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not SELECTION_FILE.exists():
        print(f"Error: {SELECTION_FILE} not found! Run candidate evaluation first.")
        sys.exit(1)
        
    with open(SELECTION_FILE, "r", encoding="utf-8") as f:
        sel_data = json.load(f)
    winner_id = sel_data["candidate_id"]
    
    with open(CANDIDATE_CONFIGS, "r", encoding="utf-8") as f:
        cands = json.load(f)
    winner_cfg = next(c for c in cands if c["id"] == winner_id)
    
    print(f"=== Regenerating ALL 6 Spanish -> German Pilot Files with Winning Config: {winner_id} ===")
    print(f"Details: {winner_cfg['description']}")
    print(f"Chunking: {winner_cfg['chunking']} | Ref: {winner_cfg['ref_type']}")
    
    chunks = load_chunks(winner_cfg["chunking"])
    ref_df = pd.read_csv(REF_DIR / "reference_audio_strategies.csv")
    
    # Target all 6 Spanish pilot files
    df_pilot = pd.read_csv(PILOT_CSV)
    df_es = df_pilot[df_pilot["language"] == "Spanish"].copy()
    
    print("=== Loading XTTS-v2 on GPU ===")
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    records = []
    
    for idx, row in df_es.iterrows():
        sid = row["safe_id"]
        base_id = sid.replace(".wav", "")
        diag = row["diagnosis"]
        out_wav_path = FINAL_OUT_DIR / f"{base_id}_final_to_German.wav"
        
        print(f"\n[{base_id}] ({diag}) -> Synthesizing {out_wav_path.name}...")
        speaker_ref = get_reference_audio(base_id, winner_cfg["ref_type"], ref_df)
        
        t0 = time.time()
        success = False
        err_msg = ""
        for attempt in range(1, 4):
            try:
                full_audio, sr = synthesize_chunks(tts, chunks, speaker_ref, winner_cfg)
                sf.write(str(out_wav_path), full_audio, sr)
                dur = len(full_audio) / sr
                elapsed = time.time() - t0
                print(f"  -> Generated in {elapsed:.2f}s! Duration: {dur:.2f}s | RTF: {elapsed/dur:.2f}")
                success = True
                break
            except Exception as e:
                err_msg = str(e)
                print(f"  [WARN] Attempt {attempt} failed: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(1.0)
                
        records.append({
            "safe_id": base_id,
            "filename": f"{base_id}_final_to_German.wav",
            "diagnosis": diag,
            "source_language": "Spanish",
            "target_language": "German",
            "selected_candidate_id": winner_id,
            "chunking_strategy": winner_cfg["chunking"],
            "reference_strategy": winner_cfg["ref_type"],
            "temperature": winner_cfg["temperature"],
            "repetition_penalty": winner_cfg["repetition_penalty"],
            "top_p": winner_cfg["top_p"],
            "speed": winner_cfg["speed"],
            "output_path": str(out_wav_path),
            "duration": round(dur, 3) if success else 0.0,
            "status": "SUCCESS" if success else "FAILED",
            "error": err_msg
        })
        
    df_manifest = pd.DataFrame(records)
    out_csv = WORKSPACE / "10_final_regeneration" / "final_synthesis_manifest.csv"
    df_manifest.to_csv(out_csv, index=False)
    print(f"\nSaved final regeneration manifest to: {out_csv}")

if __name__ == "__main__":
    main()
