"""
Script 03: Autonomous Synthesis for 80-File True Cross-Language Validation.

Synthesizes:
- 40 Spanish -> German (using frozen C08: S3_semantic_4chunk, R2 15s clean ref, T=0.65, rep=3.5, top_p=0.85, speed=1.0, silence=0.20s, peak_norm=0.95)
- 40 German -> Spanish (using frozen validated pilot: 5-sentence, R1 10s clean ref, T=0.75, rep=2.0, top_p=0.85, speed=1.0, silence=0.25s, peak_norm=0.95)

Strict Autonomous Operation:
- Retries up to 3 times on transient errors
- Clears CUDA cache after each file
- Never retunes parameters or cherry-picks
- Saves to:
  * 05_synthesis/spanish_to_german/{safe_id}_to_German_true_crosslanguage.wav
  * 05_synthesis/german_to_spanish/{safe_id}_to_Spanish_true_crosslanguage.wav
- Records manifest to 05_synthesis/synthesis_manifest_80.csv
- Records any failures to logs/synthesis_failures.csv
"""

import os
os.environ["COQUI_TOS_AGREED"] = "1"
import sys
if sys.platform == "win32":
    if os.path.exists(r"C:\Windows\System32"):
        os.add_dll_directory(r"C:\Windows\System32")
import json
import time
import functools
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch

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

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE = PROJECT_ROOT / "true_cross_language_vc"
VAL_DIR = WORKSPACE / "21_validation_80"
SYNTH_DIR = VAL_DIR / "05_synthesis"
ES_DE_DIR = SYNTH_DIR / "spanish_to_german"
DE_ES_DIR = SYNTH_DIR / "german_to_spanish"
LOGS_DIR = VAL_DIR / "logs"

SELECT_CSV = VAL_DIR / "01_selection" / "validation_80_selection.csv"
TARGET_CSV = VAL_DIR / "03_translation" / "frozen_target_text_mapping.csv"
REF_CSV = VAL_DIR / "04_reference_audio" / "reference_audio_manifest.csv"

def synthesize_utterance(tts, chunks, speaker_ref, lang_code, params, silence_sec):
    sr = 24000
    silence_gap = np.zeros(int(silence_sec * sr), dtype=np.float32)
    concatenated = []

    for chunk_text in chunks:
        wav_out = tts.tts(
            text=chunk_text,
            speaker_wav=speaker_ref,
            language=lang_code,
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
        full_audio = full_audio * (params["peak_norm"] / peak)

    return full_audio, sr

def main():
    ES_DE_DIR.mkdir(parents=True, exist_ok=True)
    DE_ES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading Coqui XTTS-v2 Engine ===")
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("XTTS-v2 loaded successfully.")

    # Load metadata
    df_cohort = pd.read_csv(SELECT_CSV)
    df_target = pd.read_csv(TARGET_CSV)
    df_ref = pd.read_csv(REF_CSV)

    # Parameter specifications
    params_es_to_de = {
        "temperature": 0.65,
        "repetition_penalty": 3.5,
        "top_p": 0.85,
        "speed": 1.00,
        "silence_sec": 0.20,
        "peak_norm": 0.95
    }

    params_de_to_es = {
        "temperature": 0.75,
        "repetition_penalty": 2.0,
        "top_p": 0.85,
        "speed": 1.00,
        "silence_sec": 0.25,
        "peak_norm": 0.95
    }

    manifest_records = []
    failure_records = []

    total_files = len(df_cohort)
    print(f"\n=== Commencing Autonomous Synthesis for {total_files} Cohort Recordings ===")

    start_total_time = time.time()

    for idx, row in df_cohort.iterrows():
        sid = row["safe_id"]
        base_id = sid.replace(".wav", "")
        src_lang = row["language"]
        diag = row["diagnosis"]
        spk_id = row["speaker_id"]
        target_fn = row["target_filename"]

        # Get target chunks
        tgt_row = df_target[df_target["safe_id"] == sid].iloc[0]
        chunks = json.loads(tgt_row["chunks_json"])
        tgt_lang = tgt_row["target_language"]
        lang_code = "de" if tgt_lang == "German" else "es"
        params = params_es_to_de if src_lang == "Spanish" else params_de_to_es

        # Get clean reference
        ref_row = df_ref[df_ref["safe_id"] == sid].iloc[0]
        ref_audio_path = ref_row["clean_reference_path"]

        # Destination path
        out_dir = ES_DE_DIR if src_lang == "Spanish" else DE_ES_DIR
        out_path = out_dir / target_fn

        print(f"\n[{idx+1}/{total_files}] Synthesizing {sid} ({src_lang} {diag} -> {tgt_lang})...")
        print(f"  Reference: {Path(ref_audio_path).name} (dur: {ref_row['reference_duration']}s)")
        print(f"  Output: {target_fn}")

        success = False
        attempts = 0
        max_attempts = 3
        last_error = ""

        t0 = time.time()
        while attempts < max_attempts and not success:
            attempts += 1
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                audio_data, sr = synthesize_utterance(
                    tts=tts,
                    chunks=chunks,
                    speaker_ref=ref_audio_path,
                    lang_code=lang_code,
                    params=params,
                    silence_sec=params["silence_sec"]
                )

                # Save output
                sf.write(str(out_path), audio_data, sr)
                elapsed = time.time() - t0
                dur = len(audio_data) / sr
                peak = float(np.max(np.abs(audio_data)))
                rms = float(np.sqrt(np.mean(audio_data ** 2)))
                rtf = elapsed / dur if dur > 0 else 0.0

                print(f"  -> SUCCESS (attempt {attempts}): dur={dur:.2f}s, peak={peak:.3f}, RMS={rms:.3f}, RTF={rtf:.2f} in {elapsed:.1f}s")
                success = True

                manifest_records.append({
                    "safe_id": sid,
                    "speaker_id": spk_id,
                    "source_language": src_lang,
                    "target_language": tgt_lang,
                    "diagnosis": diag,
                    "target_filename": target_fn,
                    "output_path": str(out_path),
                    "duration": round(dur, 4),
                    "sample_rate": sr,
                    "peak_amplitude": round(peak, 4),
                    "rms_amplitude": round(rms, 4),
                    "synthesis_time_sec": round(elapsed, 2),
                    "rtf": round(rtf, 4),
                    "attempts": attempts,
                    "status": "SUCCESS"
                })

            except Exception as e:
                last_error = str(e)
                print(f"  -> ATTEMPT {attempts} FAILED: {last_error}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(1.0)

        if not success:
            print(f"  -> CRITICAL: {sid} FAILED after {max_attempts} attempts!")
            failure_records.append({
                "safe_id": sid,
                "speaker_id": spk_id,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "diagnosis": diag,
                "attempts": attempts,
                "last_error": last_error
            })
            manifest_records.append({
                "safe_id": sid,
                "speaker_id": spk_id,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "diagnosis": diag,
                "target_filename": target_fn,
                "output_path": "",
                "duration": 0.0,
                "sample_rate": 0,
                "peak_amplitude": 0.0,
                "rms_amplitude": 0.0,
                "synthesis_time_sec": 0.0,
                "rtf": 0.0,
                "attempts": attempts,
                "status": "FAILED"
            })

    total_time = time.time() - start_total_time
    print(f"\n==================================================")
    print(f"Synthesis Complete in {total_time/60:.2f} minutes.")
    print(f"Total files attempted: {total_files}")
    print(f"Total successful: {len(df_cohort) - len(failure_records)}")
    print(f"Total failed: {len(failure_records)}")
    print(f"==================================================")

    # Save manifest
    df_manifest = pd.DataFrame(manifest_records)
    manifest_csv = SYNTH_DIR / "synthesis_manifest_80.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"Saved synthesis manifest to: {manifest_csv}")

    # Save failures log
    df_failures = pd.DataFrame(failure_records)
    failures_csv = LOGS_DIR / "synthesis_failures.csv"
    df_failures.to_csv(failures_csv, index=False)
    print(f"Saved failures log to: {failures_csv}")

if __name__ == "__main__":
    main()
