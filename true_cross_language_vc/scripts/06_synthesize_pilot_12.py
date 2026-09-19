"""
Script 06: Synthesize the 12-file pilot using True Cross-Language Voice Conversion.
Uses Coqui XTTS-v2:
- Spanish sources (3 HC, 3 PD) -> Synthesize German target speech conditioned on Spanish voice.
- German sources (3 HC, 3 PD) -> Synthesize Spanish target speech conditioned on German voice.
Outputs:
- 05_target_language_synthesis/pilot_12/spanish_to_german/
- 05_target_language_synthesis/pilot_12/german_to_spanish/
- 05_target_language_synthesis/pilot_12/pilot_12_synthesis_manifest.csv
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
from pathlib import Path
import pandas as pd
import soundfile as sf

# Patch torch.load for PyTorch 2.6+ compatibility with Coqui TTS checkpoints
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
WORKSPACE_ROOT = PROJECT_ROOT / "true_cross_language_vc"
PILOT_DIR = WORKSPACE_ROOT / "05_target_language_synthesis" / "pilot_12"

def main():
    print("=== Step 1: Initializing Synthesis Engine ===")
    from TTS.api import TTS

    pilot_csv = WORKSPACE_ROOT / "04_pilot_selection" / "pilot_12_selection.csv"
    if not pilot_csv.exists():
        raise FileNotFoundError(f"Missing {pilot_csv}")

    df_pilot = pd.read_csv(pilot_csv)
    print(f"Loaded {len(df_pilot)} pilot recordings for cross-language synthesis.")

    es_to_de_dir = PILOT_DIR / "spanish_to_german"
    de_to_es_dir = PILOT_DIR / "german_to_spanish"
    es_to_de_dir.mkdir(parents=True, exist_ok=True)
    de_to_es_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 on {device.upper()}...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("TTS engine ready.")

    manifest_records = []

    for idx, row in df_pilot.iterrows():
        safe_id = row["safe_id"]
        base_id = safe_id.replace(".wav", "")
        src_lang = row["language"]
        diag = row["diagnosis"]
        src_audio = row["audio_path"]
        target_text = row["target_text"]

        if src_lang == "Spanish":
            tgt_lang_code = "de"
            tgt_lang_name = "German"
            out_fn = f"{base_id}_to_German_true_crosslanguage.wav"
            out_path = es_to_de_dir / out_fn
        else:
            tgt_lang_code = "es"
            tgt_lang_name = "Spanish"
            out_fn = f"{base_id}_to_Spanish_true_crosslanguage.wav"
            out_path = de_to_es_dir / out_fn

        print(f"\n[{idx+1}/12] Synthesizing {safe_id} ({src_lang} {diag}) -> {tgt_lang_name}...")
        t0 = time.time()
        try:
            # Generate target-language speech conditioned on source speaker's voice
            tts.tts_to_file(
                text=target_text,
                speaker_wav=src_audio,
                language=tgt_lang_code,
                file_path=str(out_path)
            )
            elapsed = time.time() - t0
            info = sf.info(str(out_path))
            data, _ = sf.read(str(out_path))
            peak = float(abs(data).max())
            rms = float((data ** 2).mean() ** 0.5)

            print(f"  -> Generated in {elapsed:.2f} s: duration={info.duration:.2f} s, peak={peak:.3f}, rms={rms:.3f}")
            status = "SUCCESS"
            error_msg = ""
        except Exception as e:
            elapsed = time.time() - t0
            status = "FAILED"
            error_msg = str(e)
            info = None
            peak, rms = 0.0, 0.0
            print(f"  -> FAILED ({elapsed:.2f} s): {error_msg}")

        manifest_records.append({
            "safe_id": safe_id,
            "source_language": src_lang,
            "target_language": tgt_lang_name,
            "target_lang_code": tgt_lang_code,
            "diagnosis": diag,
            "source_audio_path": src_audio,
            "generated_filename": out_fn,
            "generated_audio_path": str(out_path),
            "synthesis_status": status,
            "duration": round(info.duration, 4) if info else 0.0,
            "sample_rate": info.samplerate if info else 0,
            "channels": info.channels if info else 0,
            "peak_amplitude": round(peak, 4),
            "rms_amplitude": round(rms, 4),
            "elapsed_seconds": round(elapsed, 2),
            "error": error_msg
        })

    df_manifest = pd.DataFrame(manifest_records)
    manifest_csv = PILOT_DIR / "pilot_12_synthesis_manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"\nSaved synthesis manifest to {manifest_csv}.")

if __name__ == "__main__":
    main()
