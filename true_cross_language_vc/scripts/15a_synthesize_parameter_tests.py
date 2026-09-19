"""
Script 15a: Synthesize Parameter Tests (Config B and Config C) on Spanish->German pilot files.
Runs in env_xtts (Python 3.10 with Coqui TTS).
- Config B: Sentence-segmented text + unsegmented reference audio
- Config C: Sentence-segmented text + deterministic clean 10s reference segment
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
PARAM_DIR = REFINEMENT_DIR / "03_parameter_tests"
SEGMENTS_DIR = REFINEMENT_DIR / "02_text_segmentation" / "segments"

GERMAN_SENTENCES = [
    "Gestern ging ich zum Arzt.",
    "Was fehlt Ihnen? fragte er mich.",
    "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!",
    "Haben Sie einen gebrochenen Nagel?",
    "Ja.",
    "Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab."
]

def synthesize_sentence_sequence(tts, sentences, ref_wav_path, out_wav_path, segment_save_dir=None, safe_id=""):
    """
    Synthesizes a list of sentences one by one, logs segment metadata,
    concatenates with 250ms silence pause, normalizes peak amplitude, and writes out_wav_path.
    """
    sr = 24000
    silence_gap = np.zeros(int(0.25 * sr), dtype=np.float32)
    concatenated = []
    seg_records = []

    for order, sent_text in enumerate(sentences, 1):
        wav_array = tts.tts(
            text=sent_text,
            speaker_wav=str(ref_wav_path),
            language="de"
        )
        wav_array = np.array(wav_array, dtype=np.float32)
        seg_dur = len(wav_array) / sr

        seg_path_str = ""
        if segment_save_dir is not None:
            seg_fn = f"{safe_id}_seg_{order:02d}.wav"
            seg_p = segment_save_dir / seg_fn
            sf.write(str(seg_p), wav_array, sr)
            seg_path_str = str(seg_p)

        seg_records.append({
            "segment_id": f"{safe_id}_seg_{order:02d}",
            "source_safe_id": safe_id,
            "segment_order": order,
            "target_text": sent_text,
            "generated_segment_path": seg_path_str,
            "segment_duration": round(seg_dur, 4)
        })

        if len(concatenated) > 0:
            concatenated.append(silence_gap)
        concatenated.append(wav_array)

    full_audio = np.concatenate(concatenated)
    # Peak normalize to 0.95 to prevent any digital clipping
    peak = np.max(np.abs(full_audio))
    if peak > 0:
        full_audio = full_audio * (0.95 / peak)

    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav_path), full_audio, sr)
    total_dur = len(full_audio) / sr

    return total_dur, seg_records

def main():
    print("=== Step 1: Initializing XTTS-v2 Engine for Parameter Tests ===")
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 on {device.upper()}...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    pilot_csv = WORKSPACE_ROOT / "04_pilot_selection" / "pilot_12_selection.csv"
    df_pilot = pd.read_csv(pilot_csv)
    # Filter only Spanish source speakers (ES -> DE)
    df_es = df_pilot[df_pilot["language"] == "Spanish"].copy()

    clean_ref_dir = REFINEMENT_DIR / "01_reference_audio_analysis" / "clean_references"
    config_b_dir = PARAM_DIR / "audio" / "config_b"
    config_c_dir = PARAM_DIR / "audio" / "config_c"
    config_b_dir.mkdir(parents=True, exist_ok=True)
    config_c_dir.mkdir(parents=True, exist_ok=True)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    all_seg_records = []

    print("\n=== Step 2: Running Synthesis for Config B (Sentence-Segmented + Full Reference) ===")
    for idx, r in df_es.iterrows():
        sid = r["safe_id"]
        base_id = sid.replace(".wav", "")
        diag = r["diagnosis"]
        raw_ref = Path(r["audio_path"])
        out_path = config_b_dir / f"{base_id}_config_b.wav"

        seg_save_dir = SEGMENTS_DIR / "config_b" / base_id
        seg_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [Config B] Synthesizing {sid} ({diag})...")
        t0 = time.time()
        dur, seg_recs = synthesize_sentence_sequence(
            tts=tts,
            sentences=GERMAN_SENTENCES,
            ref_wav_path=raw_ref,
            out_wav_path=out_path,
            segment_save_dir=seg_save_dir,
            safe_id=base_id
        )
        elapsed = time.time() - t0
        print(f"    -> Complete in {elapsed:.1f}s: duration={dur:.2f}s, saved to {out_path.name}")
        for s in seg_recs:
            s["config"] = "Config_B"
        all_seg_records.extend(seg_recs)

    print("\n=== Step 3: Running Synthesis for Config C (Sentence-Segmented + Clean 10s Reference) ===")
    for idx, r in df_es.iterrows():
        sid = r["safe_id"]
        base_id = sid.replace(".wav", "")
        diag = r["diagnosis"]
        clean_ref = clean_ref_dir / f"{base_id}_clean_10s_ref.wav"
        out_path = config_c_dir / f"{base_id}_config_c.wav"

        seg_save_dir = SEGMENTS_DIR / "config_c" / base_id
        seg_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [Config C] Synthesizing {sid} ({diag})...")
        t0 = time.time()
        dur, seg_recs = synthesize_sentence_sequence(
            tts=tts,
            sentences=GERMAN_SENTENCES,
            ref_wav_path=clean_ref,
            out_wav_path=out_path,
            segment_save_dir=seg_save_dir,
            safe_id=base_id
        )
        elapsed = time.time() - t0
        print(f"    -> Complete in {elapsed:.1f}s: duration={dur:.2f}s, saved to {out_path.name}")
        for s in seg_recs:
            s["config"] = "Config_C"
        all_seg_records.extend(seg_recs)

    # Save sentence_segments.csv updated with Config C segments (which represent the refined segment breakdown)
    df_segs = pd.DataFrame([s for s in all_seg_records if s["config"] == "Config_C"])
    seg_csv = REFINEMENT_DIR / "02_text_segmentation" / "sentence_segments.csv"
    df_segs.drop(columns=["config"]).to_csv(seg_csv, index=False)
    print(f"\nUpdated {seg_csv.relative_to(WORKSPACE_ROOT)} with Config C segment paths and durations.")

if __name__ == "__main__":
    main()
