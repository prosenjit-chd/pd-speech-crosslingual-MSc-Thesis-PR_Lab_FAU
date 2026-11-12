"""
Script 03: Autonomous Synthesis for Full 276-File True Cross-Language Evaluation.

Implements Sections 11 & 12:
- Synthesizes all 276 target-language recordings:
  * 100 Spanish -> German (using frozen C08: S3_semantic_4chunk, R2 15s clean ref, T=0.65, rep=3.5, top_p=0.85, speed=1.0, silence=0.20s, peak_norm=0.95)
  * 176 German -> Spanish (using frozen 80-validation: 5-sentence seg, R1 10s clean ref, T=0.75, rep=2.0, top_p=0.85, speed=1.0, silence=0.25s, peak_norm=0.95)
- For the 80 independent validation subjects, if identical frozen audio exists in 21_validation_80, copies and verifies it to maintain bit-for-bit consistency.
- For all remaining 196 subjects (12 pilot + 184 full cohort), generates fresh speech using Coqui XTTS-v2 on GPU.
- Automatic failure handling: retries up to 3 times, clears CUDA cache after each file, logs all retries and errors.
- Never retunes parameters per speaker.
- Saves manifest to 06_synthesis/synthesis_manifest_276.csv.
"""

import os
os.environ["COQUI_TOS_AGREED"] = "1"
import sys
if sys.platform == "win32":
    if os.path.exists(r"C:\Windows\System32"):
        os.add_dll_directory(r"C:\Windows\System32")
import json
import time
import shutil
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
EVAL_DIR = WORKSPACE / "22_full_276_evaluation"
SYNTH_DIR = EVAL_DIR / "06_synthesis"
ES_DE_DIR = SYNTH_DIR / "spanish_to_german"
DE_ES_DIR = SYNTH_DIR / "german_to_spanish"
LOGS_DIR = EVAL_DIR / "logs"

TAGS_CSV = EVAL_DIR / "02_cohort_tags" / "cohort_provenance.csv"
INV_CSV = EVAL_DIR / "01_inventory" / "full_276_inventory.csv"
TARGET_CSV = EVAL_DIR / "04_translation" / "frozen_target_translations.csv"
REF_CSV = EVAL_DIR / "05_reference_audio" / "reference_manifest_276.csv"

VAL80_SYNTH_DIR = WORKSPACE / "21_validation_80" / "05_synthesis"

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

    print("=== Loading Inventory, Provenance Tags, Translations, and References ===")
    df_inv = pd.read_csv(INV_CSV)
    df_tags = pd.read_csv(TAGS_CSV)
    df_tgt = pd.read_csv(TARGET_CSV)
    df_ref = pd.read_csv(REF_CSV)

    df_cohort = df_inv.merge(df_tags[["safe_id", "cohort_tag"]], on="safe_id")
    print(f"Total cohort to process: {len(df_cohort)} recordings.")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Load XTTS-v2 model
    tts = None
    def get_tts_model():
        nonlocal tts
        if tts is None:
            from TTS.api import TTS
            print("Loading Coqui XTTS-v2 model onto GPU...")
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
            print("XTTS-v2 successfully loaded.")
        return tts

    manifest_records = []
    failures = []
    total_start_time = time.time()

    for idx, row in df_cohort.iterrows():
        sid = row["safe_id"]
        lang = row["language"]
        diag = row["diagnosis"]
        tag = row["cohort_tag"]
        tgt_fn = row["target_filename"]
        audio_src = row["audio_path"]

        row_tgt = df_tgt[df_tgt["safe_id"] == sid].iloc[0]
        row_ref = df_ref[df_ref["safe_id"] == sid].iloc[0]

        tgt_lang = row_tgt["target_language"]
        chunks = json.loads(row_tgt["chunks_json"])
        ref_path = row_ref["reference_path"]

        if lang == "Spanish":
            out_dir = ES_DE_DIR
            lang_code = "de"
            params = {
                "temperature": 0.65,
                "repetition_penalty": 3.5,
                "top_p": 0.85,
                "speed": 1.00,
                "peak_norm": 0.95
            }
            silence_sec = 0.20
        else:
            out_dir = DE_ES_DIR
            lang_code = "es"
            params = {
                "temperature": 0.75,
                "repetition_penalty": 2.0,
                "top_p": 0.85,
                "speed": 1.00,
                "peak_norm": 0.95
            }
            silence_sec = 0.25

        out_path = out_dir / tgt_fn
        print(f"[{idx+1:03d}/{len(df_cohort):03d}] {sid} ({lang} {diag} | {tag}) -> {tgt_lang}...")

        # Check if already generated in this run or can be reused from 80-file validation
        reused = False
        if out_path.exists() and out_path.stat().st_size > 10000:
            print(f"  Existing output found: {out_path.name} ({out_path.stat().st_size} bytes)")
            reused = True
            dur = sf.info(str(out_path)).duration
            manifest_records.append({
                "safe_id": sid,
                "diagnosis": diag,
                "source_language": lang,
                "target_language": tgt_lang,
                "source_path": audio_src,
                "output_path": str(out_path),
                "cohort_tag": tag,
                "reference_audio": ref_path,
                "duration_seconds": round(dur, 4),
                "runtime": 0.0,
                "retry_count": 0,
                "final_status": "SUCCESS"
            })
            continue

        # If it's a validation_80 subject and exists in 21_validation_80, copy over for identical bit-for-bit fidelity
        if tag == "VALIDATION_80":
            val_src = (VAL80_SYNTH_DIR / ("spanish_to_german" if lang == "Spanish" else "german_to_spanish")) / tgt_fn
            if val_src.exists() and val_src.stat().st_size > 10000:
                shutil.copy2(val_src, out_path)
                dur = sf.info(str(out_path)).duration
                print(f"  Reused confirmed validation 80 audio: {tgt_fn} ({round(dur, 2)}s)")
                manifest_records.append({
                    "safe_id": sid,
                    "diagnosis": diag,
                    "source_language": lang,
                    "target_language": tgt_lang,
                    "source_path": audio_src,
                    "output_path": str(out_path),
                    "cohort_tag": tag,
                    "reference_audio": ref_path,
                    "duration_seconds": round(dur, 4),
                    "runtime": 0.0,
                    "retry_count": 0,
                    "final_status": "SUCCESS"
                })
                continue

        # Fresh synthesis required
        model = get_tts_model()
        attempt = 0
        success = False
        last_err = None
        t_start = time.time()

        while attempt < 3 and not success:
            attempt += 1
            try:
                if attempt > 1:
                    print(f"  Retry attempt {attempt} for {sid}...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                audio_data, sr = synthesize_utterance(
                    model, chunks, ref_path, lang_code, params, silence_sec
                )
                sf.write(str(out_path), audio_data, sr)
                elapsed = time.time() - t_start
                dur = len(audio_data) / sr
                rtf = elapsed / dur if dur > 0 else 0.0
                print(f"  Synthesized in {elapsed:.2f}s (Audio: {dur:.2f}s, RTF: {rtf:.2f})")
                success = True

                manifest_records.append({
                    "safe_id": sid,
                    "diagnosis": diag,
                    "source_language": lang,
                    "target_language": tgt_lang,
                    "source_path": audio_src,
                    "output_path": str(out_path),
                    "cohort_tag": tag,
                    "reference_audio": ref_path,
                    "duration_seconds": round(dur, 4),
                    "runtime": round(elapsed, 2),
                    "retry_count": attempt - 1,
                    "final_status": "SUCCESS"
                })

            except Exception as e:
                last_err = str(e)
                print(f"  Error on attempt {attempt}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(1.0)

        if not success:
            print(f"  FAILED to synthesize {sid} after 3 attempts.")
            failures.append({
                "safe_id": sid,
                "source_language": lang,
                "target_language": tgt_lang,
                "attempts": attempt,
                "last_error": last_err
            })
            manifest_records.append({
                "safe_id": sid,
                "diagnosis": diag,
                "source_language": lang,
                "target_language": tgt_lang,
                "source_path": audio_src,
                "output_path": str(out_path),
                "cohort_tag": tag,
                "reference_audio": ref_path,
                "duration_seconds": 0.0,
                "runtime": round(time.time() - t_start, 2),
                "retry_count": attempt,
                "final_status": "FAILED"
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save manifest and failures
    total_elapsed = time.time() - total_start_time
    df_man = pd.DataFrame(manifest_records)
    man_csv = SYNTH_DIR / "synthesis_manifest_276.csv"
    df_man.to_csv(man_csv, index=False)

    df_fail = pd.DataFrame(failures)
    fail_csv = LOGS_DIR / "synthesis_failures.csv"
    df_fail.to_csv(fail_csv, index=False)

    success_cnt = (df_man["final_status"] == "SUCCESS").sum()
    print(f"\n=== Synthesis Complete ===")
    print(f"Total Processed: {len(df_man)} / 276")
    print(f"Successful:      {success_cnt}")
    print(f"Failed:          {len(failures)}")
    print(f"Total Time:      {total_elapsed / 60.0:.2f} minutes")
    print(f"Saved manifest to {man_csv}")
    print(f"Saved failure log to {fail_csv}")

if __name__ == "__main__":
    main()
