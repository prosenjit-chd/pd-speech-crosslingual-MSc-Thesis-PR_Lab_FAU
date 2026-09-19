#!/usr/bin/env python3
"""
03_run_xtts_candidate_search.py
Runs Controlled XTTS-v2 Parameter Search on 4 Spanish -> German Pilot Speakers.

Search Set (Section 13):
- SP_HC_001 (HC reference baseline)
- SP_HC_003 (Problematic HC file, baseline WER=0.8913)
- SP_PD_001 (PD reference baseline)
- SP_PD_002 (Problematic PD file, baseline WER=0.9348)

Candidate Matrix (Sections 11, 12):
C01: S1_sentence | R1 (10s) | temp=0.75 | rep=2.0 | top_p=0.85 | speed=1.00
C02: S1_sentence | R1 (10s) | temp=0.65 | rep=3.5 | top_p=0.85 | speed=1.00
C03: S1_sentence | R1 (10s) | temp=0.55 | rep=5.0 | top_p=0.80 | speed=1.00
C04: S2_clause   | R1 (10s) | temp=0.65 | rep=3.5 | top_p=0.85 | speed=1.00
C05: S3_semantic_4chunk | R1 (10s) | temp=0.65 | rep=3.5 | top_p=0.85 | speed=1.00
C06: S3_semantic_4chunk | R1 (10s) | temp=0.55 | rep=4.0 | top_p=0.80 | speed=1.00
C07: S3_semantic_3chunk | R1 (10s) | temp=0.65 | rep=3.5 | top_p=0.85 | speed=1.00
C08: S3_semantic_4chunk | R2 (15s) | temp=0.65 | rep=3.5 | top_p=0.85 | speed=1.00
C09: S3_semantic_4chunk | R3 (Dual) | temp=0.65 | rep=3.5 | top_p=0.85 | speed=1.00
C10: S3_semantic_4chunk | R1 (10s) | temp=0.60 | rep=3.5 | top_p=0.85 | speed=1.05

Outputs saved to:
07_candidate_audio/{candidate_id}/{safe_id}.wav
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
CANDIDATE_AUDIO_DIR = WORKSPACE / "07_candidate_audio"
LOGS_DIR = WORKSPACE / "logs"

SEARCH_SPEAKERS = ["SP_HC_001", "SP_HC_003", "SP_PD_001", "SP_PD_002"]

CANDIDATES = [
    {
        "id": "C01",
        "description": "S1_sentence | R1 (10s) | temp=0.75 | rep=2.0 | top_p=0.85 | spd=1.00",
        "chunking": "S1_sentence",
        "ref_type": "R1",
        "temperature": 0.75,
        "repetition_penalty": 2.0,
        "top_p": 0.85,
        "speed": 1.00
    },
    {
        "id": "C02",
        "description": "S1_sentence | R1 (10s) | temp=0.65 | rep=3.5 | top_p=0.85 | spd=1.00",
        "chunking": "S1_sentence",
        "ref_type": "R1",
        "temperature": 0.65,
        "repetition_penalty": 3.5,
        "top_p": 0.85,
        "speed": 1.00
    },
    {
        "id": "C03",
        "description": "S1_sentence | R1 (10s) | temp=0.55 | rep=5.0 | top_p=0.80 | spd=1.00",
        "chunking": "S1_sentence",
        "ref_type": "R1",
        "temperature": 0.55,
        "repetition_penalty": 5.0,
        "top_p": 0.80,
        "speed": 1.00
    },
    {
        "id": "C04",
        "description": "S2_clause | R1 (10s) | temp=0.65 | rep=3.5 | top_p=0.85 | spd=1.00",
        "chunking": "S2_clause",
        "ref_type": "R1",
        "temperature": 0.65,
        "repetition_penalty": 3.5,
        "top_p": 0.85,
        "speed": 1.00
    },
    {
        "id": "C05",
        "description": "S3_semantic_4chunk | R1 (10s) | temp=0.65 | rep=3.5 | top_p=0.85 | spd=1.00",
        "chunking": "S3_semantic_4chunk",
        "ref_type": "R1",
        "temperature": 0.65,
        "repetition_penalty": 3.5,
        "top_p": 0.85,
        "speed": 1.00
    },
    {
        "id": "C06",
        "description": "S3_semantic_4chunk | R1 (10s) | temp=0.55 | rep=4.0 | top_p=0.80 | spd=1.00",
        "chunking": "S3_semantic_4chunk",
        "ref_type": "R1",
        "temperature": 0.55,
        "repetition_penalty": 4.0,
        "top_p": 0.80,
        "speed": 1.00
    },
    {
        "id": "C07",
        "description": "S3_semantic_3chunk | R1 (10s) | temp=0.65 | rep=3.5 | top_p=0.85 | spd=1.00",
        "chunking": "S3_semantic_3chunk",
        "ref_type": "R1",
        "temperature": 0.65,
        "repetition_penalty": 3.5,
        "top_p": 0.85,
        "speed": 1.00
    },
    {
        "id": "C08",
        "description": "S3_semantic_4chunk | R2 (15s) | temp=0.65 | rep=3.5 | top_p=0.85 | spd=1.00",
        "chunking": "S3_semantic_4chunk",
        "ref_type": "R2",
        "temperature": 0.65,
        "repetition_penalty": 3.5,
        "top_p": 0.85,
        "speed": 1.00
    },
    {
        "id": "C09",
        "description": "S3_semantic_4chunk | R3 (Dual) | temp=0.65 | rep=3.5 | top_p=0.85 | spd=1.00",
        "chunking": "S3_semantic_4chunk",
        "ref_type": "R3",
        "temperature": 0.65,
        "repetition_penalty": 3.5,
        "top_p": 0.85,
        "speed": 1.00
    },
    {
        "id": "C10",
        "description": "S3_semantic_4chunk | R1 (10s) | temp=0.60 | rep=3.5 | top_p=0.85 | spd=1.05",
        "chunking": "S3_semantic_4chunk",
        "ref_type": "R1",
        "temperature": 0.60,
        "repetition_penalty": 3.5,
        "top_p": 0.85,
        "speed": 1.05
    }
]

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
    silence_gap = np.zeros(int(0.20 * sr), dtype=np.float32)  # 200ms natural pause
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
        full_audio = full_audio * (0.95 / peak)  # Clean 0.95 peak normalization
        
    return full_audio, sr

def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save candidate definitions
    cand_manifest_path = WORKSPACE / "04_xtts_parameter_search" / "candidate_configurations.json"
    cand_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cand_manifest_path, "w", encoding="utf-8") as f:
        json.dump(CANDIDATES, f, indent=2)
    print(f"Saved {len(CANDIDATES)} candidate definitions to {cand_manifest_path}")
    
    ref_df = pd.read_csv(REF_DIR / "reference_audio_strategies.csv")
    
    print("=== Initializing XTTS-v2 Model on GPU ===")
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("XTTS-v2 loaded successfully.")
    
    manifest_records = []
    total_runs = len(CANDIDATES) * len(SEARCH_SPEAKERS)
    run_idx = 0
    start_all = time.time()
    
    for cand in CANDIDATES:
        cand_id = cand["id"]
        cand_dir = CANDIDATE_AUDIO_DIR / cand_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        
        chunks = load_chunks(cand["chunking"])
        print(f"\n=======================================================")
        print(f"[{cand_id}] {cand['description']}")
        print(f"  Chunking: {cand['chunking']} ({len(chunks)} chunks) | Ref: {cand['ref_type']}")
        print(f"=======================================================")
        
        for spk in SEARCH_SPEAKERS:
            run_idx += 1
            out_wav_path = cand_dir / f"{spk}_{cand_id}.wav"
            speaker_ref = get_reference_audio(spk, cand["ref_type"], ref_df)
            
            print(f"  ({run_idx}/{total_runs}) Synthesizing {spk} -> {out_wav_path.name}...")
            t0 = time.time()
            
            # Retry mechanism (Section 33)
            success = False
            err_msg = ""
            for attempt in range(1, 4):
                try:
                    full_audio, sr = synthesize_chunks(tts, chunks, speaker_ref, cand)
                    sf.write(str(out_wav_path), full_audio, sr)
                    dur = len(full_audio) / sr
                    elapsed = time.time() - t0
                    print(f"     -> Done in {elapsed:.2f}s (Audio: {dur:.2f}s, RTF: {elapsed/dur:.2f})")
                    success = True
                    break
                except Exception as e:
                    err_msg = str(e)
                    print(f"     [WARN] Attempt {attempt} failed for {spk} under {cand_id}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(1.0)
                    
            manifest_records.append({
                "candidate_id": cand_id,
                "safe_id": spk,
                "strategy": cand["chunking"],
                "ref_type": cand["ref_type"],
                "temperature": cand["temperature"],
                "repetition_penalty": cand["repetition_penalty"],
                "top_p": cand["top_p"],
                "speed": cand["speed"],
                "output_path": str(out_wav_path),
                "duration": round(dur, 3) if success else 0.0,
                "synthesis_time": round(elapsed, 2) if success else 0.0,
                "status": "SUCCESS" if success else "FAILED",
                "error": err_msg
            })
            
    df_manifest = pd.DataFrame(manifest_records)
    out_csv = WORKSPACE / "04_xtts_parameter_search" / "candidate_synthesis_manifest.csv"
    df_manifest.to_csv(out_csv, index=False)
    total_time = time.time() - start_all
    print(f"\n=== Parameter Search Synthesis Complete in {total_time/60:.2f} minutes! ===")
    print(f"Manifest saved to: {out_csv}")

if __name__ == "__main__":
    main()
