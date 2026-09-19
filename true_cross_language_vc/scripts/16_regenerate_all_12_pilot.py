"""
Script 16: Regenerate ALL 12 Pilot Files using the Selected Refinement Method (Config C).
- Fixed method: Sentence-segmented synthesis + deterministic clean 10s speaker reference.
- Inter-sentence pause: 250ms natural silence gap.
- Peak normalization: 0.95 (guarantees zero digital clipping).
- Saves all 12 outputs to:
  19_pilot_refinement/04_regenerated_audio/spanish_to_german/{safe_id}_refined_to_German.wav
  19_pilot_refinement/04_regenerated_audio/german_to_spanish/{safe_id}_refined_to_Spanish.wav
- Generates refined_synthesis_manifest.csv
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
import numpy as np
import soundfile as sf

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
WORKSPACE_ROOT = PROJECT_ROOT / "true_cross_language_vc"
REFINEMENT_DIR = WORKSPACE_ROOT / "19_pilot_refinement"
OUT_DIR = REFINEMENT_DIR / "04_regenerated_audio"
CLEAN_REF_DIR = REFINEMENT_DIR / "01_reference_audio_analysis" / "clean_references"
PARAM_DIR = REFINEMENT_DIR / "03_parameter_tests"

GERMAN_SENTENCES = [
    "Gestern ging ich zum Arzt.",
    "Was fehlt Ihnen? fragte er mich.",
    "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!",
    "Haben Sie einen gebrochenen Nagel?",
    "Ja.",
    "Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab."
]

SPANISH_SENTENCES = [
    "Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México.",
    "Como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme.",
    "Solo en las cercanías de Tampico se observaron unas cinco mil tortugas.",
    "En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas.",
    "Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas."
]

def synthesize_sequence(tts, sentences, ref_wav_path, lang_code, out_wav_path):
    sr = 24000
    silence_gap = np.zeros(int(0.25 * sr), dtype=np.float32)
    concatenated = []

    for sent_text in sentences:
        wav_array = tts.tts(
            text=sent_text,
            speaker_wav=str(ref_wav_path),
            language=lang_code
        )
        wav_array = np.array(wav_array, dtype=np.float32)
        if len(concatenated) > 0:
            concatenated.append(silence_gap)
        concatenated.append(wav_array)

    full_audio = np.concatenate(concatenated)
    peak = np.max(np.abs(full_audio))
    if peak > 0:
        full_audio = full_audio * (0.95 / peak)

    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav_path), full_audio, sr)
    dur = len(full_audio) / sr
    rms = float(np.sqrt(np.mean(full_audio ** 2)))
    return dur, float(np.max(np.abs(full_audio))), rms

def main():
    print("=== Step 1: Initializing XTTS-v2 for Full 12-File Regeneration ===")
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 on {device.upper()}...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    pilot_csv = WORKSPACE_ROOT / "04_pilot_selection" / "pilot_12_selection.csv"
    df_pilot = pd.read_csv(pilot_csv)

    es_de_dir = OUT_DIR / "spanish_to_german"
    de_es_dir = OUT_DIR / "german_to_spanish"
    es_de_dir.mkdir(parents=True, exist_ok=True)
    de_es_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []

    print("\n=== Step 2: Regenerating All 12 Pilot Files with Config C ===")
    for idx, r in df_pilot.iterrows():
        sid = r["safe_id"]
        base_id = sid.replace(".wav", "")
        src_lang = r["language"]
        diag = r["diagnosis"]

        clean_ref_path = CLEAN_REF_DIR / f"{base_id}_clean_10s_ref.wav"
        if not clean_ref_path.exists():
            raise FileNotFoundError(f"Missing clean reference: {clean_ref_path}")

        if src_lang == "Spanish":
            lang_code = "de"
            tgt_lang_name = "German"
            sents = GERMAN_SENTENCES
            out_fn = f"{base_id}_refined_to_German.wav"
            out_path = es_de_dir / out_fn
        else:
            lang_code = "es"
            tgt_lang_name = "Spanish"
            sents = SPANISH_SENTENCES
            out_fn = f"{base_id}_refined_to_Spanish.wav"
            out_path = de_es_dir / out_fn

        print(f"[{idx+1}/12] Synthesizing {sid} ({src_lang} {diag} -> {tgt_lang_name})...")
        t0 = time.time()
        config_c_source = PARAM_DIR / "audio" / "config_c" / f"{base_id}_config_c.wav"
        if src_lang == "Spanish" and config_c_source.exists():
            import shutil
            shutil.copy2(config_c_source, out_path)
            data, sr = sf.read(str(out_path))
            dur = len(data) / sr
            peak = float(np.max(np.abs(data)))
            rms = float(np.sqrt(np.mean(data ** 2)))
            elapsed = 0.0
            status = "SUCCESS"
            error_msg = ""
            print(f"  -> Reused verified Config C audio in {elapsed:.1f}s: duration={dur:.2f}s, peak={peak:.3f}, rms={rms:.3f}")
        else:
            try:
                dur, peak, rms = synthesize_sequence(
                    tts=tts,
                    sentences=sents,
                    ref_wav_path=clean_ref_path,
                    lang_code=lang_code,
                    out_wav_path=out_path
                )
                elapsed = time.time() - t0
                status = "SUCCESS"
                error_msg = ""
                print(f"  -> Generated in {elapsed:.1f}s: duration={dur:.2f}s, peak={peak:.3f}, rms={rms:.3f}")
            except Exception as e:
                elapsed = time.time() - t0
                dur, peak, rms = 0.0, 0.0, 0.0
                status = "FAILED"
                error_msg = str(e)
                print(f"  -> FAILED: {error_msg}")

        manifest_records.append({
            "safe_id": sid,
            "source_language": src_lang,
            "target_language": tgt_lang_name,
            "target_lang_code": lang_code,
            "diagnosis": diag,
            "clean_reference_path": str(clean_ref_path),
            "generated_filename": out_fn,
            "generated_audio_path": str(out_path),
            "synthesis_status": status,
            "duration": round(dur, 4),
            "peak_amplitude": round(peak, 4),
            "rms_amplitude": round(rms, 4),
            "elapsed_seconds": round(elapsed, 2),
            "error": error_msg
        })

    df_manifest = pd.DataFrame(manifest_records)
    out_manifest = OUT_DIR / "refined_synthesis_manifest.csv"
    df_manifest.to_csv(out_manifest, index=False)
    print(f"\nSaved refined synthesis manifest to {out_manifest.relative_to(WORKSPACE_ROOT)}.")

if __name__ == "__main__":
    main()
