"""
Script 02: Prepare Transcripts, Target Translations, and Reference Audio for 80 Cohort.

Fulfills Sections 6 & 7:
- Builds 02_transcripts/source_text_mapping.csv
- Builds 03_translation/frozen_target_text_mapping.csv
- Extracts deterministic clean reference audio for all 80 subjects:
  * Spanish -> German: R2 clean 15.0s voiced reference (or full clip if <= 15s) with 25ms fade
  * German -> Spanish: R1 clean 10.0s voiced reference with 25ms fade
- Builds 04_reference_audio/reference_audio_manifest.csv
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE = PROJECT_ROOT / "true_cross_language_vc"
VAL_DIR = WORKSPACE / "21_validation_80"

TRANS_DIR = VAL_DIR / "02_transcripts"
TARGET_DIR = VAL_DIR / "03_translation"
REF_DIR = VAL_DIR / "04_reference_audio"
REF_AUDIO_DIR = REF_DIR / "clean_references"
SELECT_CSV = VAL_DIR / "01_selection" / "validation_80_selection.csv"

# Canonical Source Texts
SPANISH_SOURCE_TEXT = (
    "Ayer fui al médico. «¿Qué le pasa?», me preguntó. Yo le dije: "
    "«¡Ay doctor, donde pongo el dedo me duele!». «¿Tiene la uña rota?». "
    "«Sí». «¡Pues ya sabemos qué es! Deje su cheque a la salida»."
)

GERMAN_SOURCE_TEXT = (
    "Von einer gewaltigen, von den Behörden geschützten Invasion wird zur Zeit "
    "die Golf- und Pazifikküste Mexikos heimgesucht: Wie alljährlich im Juni "
    "kommen Hunderttausende von Schildkröten aus dem Meer, um an Land ihre Eier abzulegen. "
    "Allein in der Nähe von Tampico wurden etwa 5000 Schildkröten beobachtet. "
    "Insgesamt wird in den kommenden Wochen mit einer Invasion von mehr als einer "
    "halben Million Schildkröten gerechnet. Die mexikanischen Behörden lassen die "
    "Legeplätze sorgfältig bewachen, um den Diebstahl von Eiern zu verhindern und "
    "ausreichend Schildkrötennachwuchs sicherzustellen."
)

# Frozen Spanish -> German Chunks (S3_semantic_4chunk from winning C08)
GERMAN_TARGET_FULL = (
    "Gestern ging ich zum Arzt. Was fehlt Ihnen? fragte er mich. "
    "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh! "
    "Haben Sie einen gebrochenen Nagel? Ja. Nun, dann wissen wir ja, was es ist! "
    "Geben Sie Ihren Scheck am Ausgang ab."
)

GERMAN_CHUNKS_S3 = [
    "Gestern ging ich zum Arzt. Was fehlt Ihnen? fragte er mich.",
    "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!",
    "Haben Sie einen gebrochenen Nagel? Ja. Nun, dann wissen wir ja, was es ist!",
    "Geben Sie Ihren Scheck am Ausgang ab."
]

# Frozen German -> Spanish Sentences (5-sentence validated pilot segmentation)
SPANISH_TARGET_FULL = (
    "Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México. "
    "Como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme. "
    "Solo en las cercanías de Tampico se observaron unas cinco mil tortugas. "
    "En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas. "
    "Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas."
)

SPANISH_SENTENCES_5 = [
    "Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México.",
    "Como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme.",
    "Solo en las cercanías de Tampico se observaron unas cinco mil tortugas.",
    "En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas.",
    "Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas."
]

def compute_audio_properties(y: np.ndarray, sr: int) -> dict:
    dur = len(y) / sr
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    is_clipped = peak >= 0.999
    
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    num_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    
    frame_rms = []
    for i in range(num_frames):
        st = i * hop_len
        frame = y[st:st + frame_len]
        frame_rms.append(np.sqrt(np.mean(frame ** 2)))
    frame_rms = np.array(frame_rms)
    
    silence_threshold = max(0.005, 0.1 * rms)
    silent_frames = np.sum(frame_rms < silence_threshold)
    silence_ratio = float(silent_frames / len(frame_rms)) if len(frame_rms) > 0 else 0.0
    voiced_ratio = float(1.0 - silence_ratio)
    
    return {
        "duration": dur,
        "peak": peak,
        "rms": rms,
        "is_clipped": is_clipped,
        "silence_ratio": silence_ratio,
        "voiced_ratio": voiced_ratio
    }

def find_best_segment(y: np.ndarray, sr: int, target_sec: float) -> tuple:
    target_samples = int(target_sec * sr)
    if len(y) <= target_samples:
        seg = y.copy()
        start_sec = 0.0
        end_sec = len(y) / sr
    else:
        hop = int(0.5 * sr)
        best_score = -1.0
        best_start = 0
        for start in range(0, len(y) - target_samples + 1, hop):
            end = start + target_samples
            seg_candidate = y[start:end]
            props = compute_audio_properties(seg_candidate, sr)
            if props["is_clipped"]:
                score = 0.0
            else:
                rms_factor = min(1.0, props["rms"] / 0.04)
                score = props["voiced_ratio"] * (1.0 - props["silence_ratio"]) * rms_factor
            if score > best_score:
                best_score = score
                best_start = start
        seg = y[best_start:best_start + target_samples].copy()
        start_sec = best_start / sr
        end_sec = (best_start + target_samples) / sr

    # Apply 25ms cosine fade in/out
    fade_len = int(0.025 * sr)
    if len(seg) > 2 * fade_len:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        seg[:fade_len] *= fade_in
        seg[-fade_len:] *= fade_out

    return seg, start_sec, end_sec

def main():
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    df_cohort = pd.read_csv(SELECT_CSV)
    print(f"Loaded cohort selection with {len(df_cohort)} recordings.")

    source_records = []
    target_records = []
    ref_records = []

    print("\n=== Processing Transcripts, Translations, and Clean References ===")
    for idx, row in df_cohort.iterrows():
        sid = row["safe_id"]
        base_id = sid.replace(".wav", "")
        lang = row["language"]
        diag = row["diagnosis"]
        spk_id = row["speaker_id"]
        orig_fn = row["original_filename"]
        audio_path = Path(row["audio_path"])

        # 1. Source Transcript Mapping
        if lang == "Spanish":
            src_text = SPANISH_SOURCE_TEXT
            tgt_lang = "German"
            tgt_text_full = GERMAN_TARGET_FULL
            chunks = GERMAN_CHUNKS_S3
            chunking_strategy = "S3_semantic_4chunk"
            ref_target_sec = 15.0
            ref_method = "R2_clean_15s_voiced_reference"
        else:
            src_text = GERMAN_SOURCE_TEXT
            tgt_lang = "Spanish"
            tgt_text_full = SPANISH_TARGET_FULL
            chunks = SPANISH_SENTENCES_5
            chunking_strategy = "sentence_segmented_5"
            ref_target_sec = 10.0
            ref_method = "R1_clean_10s_voiced_reference"

        source_records.append({
            "safe_id": sid,
            "original_filename": orig_fn,
            "speaker_id": spk_id,
            "language": lang,
            "diagnosis": diag,
            "source_text": src_text,
            "word_count": len(src_text.split())
        })

        target_records.append({
            "safe_id": sid,
            "source_language": lang,
            "target_language": tgt_lang,
            "diagnosis": diag,
            "source_text": src_text,
            "target_text": tgt_text_full,
            "chunking_strategy": chunking_strategy,
            "num_chunks": len(chunks),
            "chunks_json": json.dumps(chunks, ensure_ascii=False)
        })

        # 2. Extract Reference Audio
        y, sr = sf.read(str(audio_path))
        if y.ndim > 1:
            y = y[:, 0]
        ref_seg, st_sec, end_sec = find_best_segment(y, sr, ref_target_sec)
        ref_fn = f"{base_id}_clean_ref.wav"
        ref_path = REF_AUDIO_DIR / ref_fn
        sf.write(str(ref_path), ref_seg, sr)

        ref_props = compute_audio_properties(ref_seg, sr)
        ref_records.append({
            "safe_id": sid,
            "source_file": str(audio_path),
            "reference_start": round(st_sec, 4),
            "reference_end": round(end_sec, 4),
            "reference_duration": round(ref_props["duration"], 4),
            "selection_method": ref_method,
            "peak_amplitude": round(ref_props["peak"], 4),
            "rms_amplitude": round(ref_props["rms"], 4),
            "silence_ratio": round(ref_props["silence_ratio"], 4),
            "clean_reference_path": str(ref_path)
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(df_cohort):
            print(f"  Processed {idx + 1}/{len(df_cohort)} subjects...")

    # Save Source Text Mapping
    df_src = pd.DataFrame(source_records)
    src_csv = TRANS_DIR / "source_text_mapping.csv"
    df_src.to_csv(src_csv, index=False)
    print(f"\nSaved source text mapping to: {src_csv}")

    # Save Target Text Mapping
    df_tgt = pd.DataFrame(target_records)
    tgt_csv = TARGET_DIR / "frozen_target_text_mapping.csv"
    df_tgt.to_csv(tgt_csv, index=False)
    print(f"Saved frozen target text mapping to: {tgt_csv}")

    # Save Reference Audio Manifest
    df_ref = pd.DataFrame(ref_records)
    ref_csv = REF_DIR / "reference_audio_manifest.csv"
    df_ref.to_csv(ref_csv, index=False)
    print(f"Saved reference audio manifest to: {ref_csv}")
    print(f"All 80 clean reference WAVs stored in: {REF_AUDIO_DIR}")

if __name__ == "__main__":
    main()
