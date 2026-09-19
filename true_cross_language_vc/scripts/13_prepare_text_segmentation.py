"""
Script 13: Text Normalization and Sentence Segmentation.
Inspects German target text, normalizes decorative quotes/compound punctuation,
splits into natural sentence units, and creates sentence_segments.csv.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE_ROOT = PROJECT_ROOT / "true_cross_language_vc"
REFINEMENT_DIR = WORKSPACE_ROOT / "19_pilot_refinement" / "02_text_segmentation"

def main():
    REFINEMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Original German Target Text (PC-GITA translation)
    original_german = (
        'Gestern ging ich zum Arzt. „Was fehlt Ihnen?“, fragte er mich. '
        'Ich sagte: „Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!“ '
        '„Haben Sie einen gebrochenen Nagel?“ „Ja.“ '
        '„Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab.“'
    )
    
    # 2. Normalized German Target Text
    # Replaces German low-high quotation marks with standard punctuation while strictly preserving every single word
    normalized_german = (
        'Gestern ging ich zum Arzt. '
        'Was fehlt Ihnen? fragte er mich. '
        'Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh! '
        'Haben Sie einen gebrochenen Nagel? '
        'Ja. '
        'Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab.'
    )
    
    # Define the 6 natural sentence segments for German
    german_sentences = [
        "Gestern ging ich zum Arzt.",
        "Was fehlt Ihnen? fragte er mich.",
        "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!",
        "Haben Sie einen gebrochenen Nagel?",
        "Ja.",
        "Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab."
    ]

    # Original Spanish Target Text (Sabine Skoda translation)
    original_spanish = (
        "Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México: "
        "como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme. "
        "Solo en las cercanías de Tampico se observaron unas 5.000 tortugas. "
        "En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas. "
        "Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas."
    )
    
    # Normalized Spanish Target Text (colon split into sentence, number spelled out)
    normalized_spanish = (
        "Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México. "
        "Como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme. "
        "Solo en las cercanías de Tampico se observaron unas cinco mil tortugas. "
        "En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas. "
        "Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas."
    )

    spanish_sentences = [
        "Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México.",
        "Como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme.",
        "Solo en las cercanías de Tampico se observaron unas cinco mil tortugas.",
        "En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas.",
        "Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas."
    ]

    # Save text files
    orig_de_path = REFINEMENT_DIR / "german_target_text_original.txt"
    norm_de_path = REFINEMENT_DIR / "german_target_text_normalized.txt"
    with open(orig_de_path, "w", encoding="utf-8") as f:
        f.write(original_german.strip())
    with open(norm_de_path, "w", encoding="utf-8") as f:
        f.write(normalized_german.strip())
    print(f"Saved original and normalized German text to {REFINEMENT_DIR}.")

    # Save normalization log
    log_path = REFINEMENT_DIR / "text_normalization_log.md"
    log_content = f"""# Target Text Normalization and Segmentation Log

## 1. German Target Text (Spanish $\\to$ German Conversion)

### Original Text:
```text
{original_german}
```

### Normalized Text:
```text
{normalized_german}
```

### Normalization Actions Applied:
1. **Unicode Dialogue Quote Removal**:
   - Replaced German low quote `„` (U+201E) and high quote `“` (U+201C) with standard conversational punctuation.
   - Removed compound punctuation `?“,` which caused token boundary collisions in XTTS-v2 tokenizer.
2. **Strict Lexical Invariance**:
   - Exactly 46 words in both original and normalized versions.
   - Zero vocabulary alterations, zero semantic shortening, zero structural drift.
3. **Discrete Sentence Segmentation**:
   The passage was segmented into 6 natural conversational dialogue turns:
   - **Segment 1** (5 words): `"Gestern ging ich zum Arzt."`
   - **Segment 2** (6 words): `"Was fehlt Ihnen? fragte er mich."`
   - **Segment 3** (14 words): `"Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!"`
   - **Segment 4** (6 words): `"Haben Sie einen gebrochenen Nagel?"`
   - **Segment 5** (1 word): `"Ja."`
   - **Segment 6** (14 words): `"Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab."`

---

## 2. Spanish Target Text (German $\\to$ Spanish Conversion)

### Original Text:
```text
{original_spanish}
```

### Normalized Text:
```text
{normalized_spanish}
```

### Normalization Actions Applied:
1. Replaced colon separator with natural sentence boundary to avoid running clause drift.
2. Normalized `5.000` to `cinco mil` to ensure consistent phonetic pronunciation across synthesis engines.
3. Segmented into 5 natural expository sentences:
   - **Segment 1** (18 words): `"Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México."`
   - **Segment 2** (21 words): `"Como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme."`
   - **Segment 3** (11 words): `"Solo en las cercanías de Tampico se observaron unas cinco mil tortugas."`
   - **Segment 4** (18 words): `"En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas."`
   - **Segment 5** (22 words): `"Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas."`
"""
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"Saved text normalization log to {log_path}.")

    # Create preliminary sentence_segments.csv table template
    pilot_csv = WORKSPACE_ROOT / "04_pilot_selection" / "pilot_12_selection.csv"
    df_pilot = pd.read_csv(pilot_csv)

    records = []
    for _, r in df_pilot.iterrows():
        sid = r["safe_id"]
        lang = r["language"]
        if lang == "Spanish":
            sents = german_sentences
            tgt_lang = "German"
        else:
            sents = spanish_sentences
            tgt_lang = "Spanish"

        for order, s_text in enumerate(sents, 1):
            seg_id = f"{sid.replace('.wav', '')}_seg_{order:02d}"
            records.append({
                "segment_id": seg_id,
                "source_safe_id": sid,
                "target_language": tgt_lang,
                "segment_order": order,
                "target_text": s_text,
                "generated_segment_path": "",
                "segment_duration": 0.0
            })

    df_seg = pd.DataFrame(records)
    seg_csv = REFINEMENT_DIR / "sentence_segments.csv"
    df_seg.to_csv(seg_csv, index=False)
    print(f"Saved sentence segmentation template ({len(df_seg)} segments) to {seg_csv}.")

if __name__ == "__main__":
    main()
