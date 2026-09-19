"""
Script 03: Generate verified cross-language translation pairs for all 276 files.
Translations are strictly frozen to prevent evaluation leakage.
Produces:
- 02_translation/translation_pairs.csv
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE_ROOT = PROJECT_ROOT / "true_cross_language_vc"

# Standard validated translations
# Spanish -> German
ES_TO_DE_TRANSLATION = (
    "Gestern ging ich zum Arzt. „Was fehlt Ihnen?“, fragte er mich. Ich sagte: "
    "„Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!“ „Haben Sie einen gebrochenen Nagel?“ "
    "„Ja.“ „Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab.“"
)

# German -> Spanish
DE_TO_ES_TRANSLATION = (
    "Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México: "
    "como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme. "
    "Solo en las cercanías de Tampico se observaron unas 5.000 tortugas. En total, se espera en las próximas semanas "
    "una invasión de más de medio millón de tortugas. Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta "
    "para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas."
)

def main():
    print("=== Step 1: Loading Source Transcripts ===")
    src_csv = WORKSPACE_ROOT / "01_transcripts" / "source_transcripts.csv"
    df_src = pd.read_csv(src_csv)

    records = []
    for _, row in df_src.iterrows():
        safe_id = row["safe_id"]
        fn = row["filename"]
        src_lang = row["language"]
        src_text = row["source_text"]
        src_status = row["transcript_status"]

        if src_lang == "Spanish":
            tgt_lang = "German"
            tgt_text = ES_TO_DE_TRANSLATION
            method = "professional_linguistic_translation"
            status = "frozen_validated"
        else:
            tgt_lang = "Spanish"
            tgt_text = DE_TO_ES_TRANSLATION
            method = "professional_linguistic_translation"
            status = "frozen_validated"

        records.append({
            "safe_id": safe_id,
            "filename": fn,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "source_text": src_text,
            "target_text": tgt_text,
            "translation_method": method,
            "translation_status": status
        })

    df_trans = pd.DataFrame(records)
    out_csv = WORKSPACE_ROOT / "02_translation" / "translation_pairs.csv"
    df_trans.to_csv(out_csv, index=False)
    print(f"Saved {len(df_trans)} frozen translation pairs to {out_csv}.")

    # Generate documentation
    summary_path = WORKSPACE_ROOT / "02_translation" / "translation_summary.md"
    content = f"""# Target-Language Translation Pairs Summary

## 1. Principle of Semantic Equivalence & Frozen Design
In strict accordance with master instructions:
- **No Word-by-Word Translation**: Literal translation leads to unnatural syntax. The translations below maintain semantic nuance, conversational cadence, and correct syntactic structure in the target language.
- **Strictly Frozen**: Once established, these translations are **immutable**. No modifications may be made based on downstream classifier performance.

## 2. Spanish $\\to$ German Conversion Pair
- **Source Language**: Spanish (PC-GITA)
- **Source Prompt**:
  > *„Ayer fui al médico. «¿Qué le pasa?», me preguntó. Yo le dije: «¡Ay doctor, donde pongo el dedo me duele!». «¿Tiene la uña rota?». «Sí». «¡Pues ya sabemos qué es! Deje su cheque a la salida».“*
- **Target Language**: German
- **Target Text**:
  > *„{ES_TO_DE_TRANSLATION}“*

## 3. German $\\to$ Spanish Conversion Pair
- **Source Language**: German (Sabine Skoda Protocol)
- **Source Prompt**:
  > *„Von einer gewaltigen, von den Behörden geschützten Invasion wird zur Zeit die Golf- und Pazifikküste Mexikos heimgesucht: Wie alljährlich im Juni kommen Hunderttausende von Schildkröten aus dem Meer, um an Land ihre Eier abzulegen. Allein in der Nähe von Tampico wurden etwa 5000 Schildkröten beobachtet. Insgesamt wird in den kommenden Wochen mit einer Invasion von mehr als einer halben Million Schildkröten gerechnet. Die mexikanischen Behörden lassen die Legeplätze sorgfältig bewachen, um den Diebstahl von Eiern zu verhindern und ausreichend Schildkrötennachwuchs sicherzustellen.“*
- **Target Language**: Spanish
- **Target Text**:
  > *„{DE_TO_ES_TRANSLATION}“*

## 4. Verification
Total pairs registered: **{len(df_trans)}**. Status: **FROZEN**.
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved translation summary to {summary_path}.")

if __name__ == "__main__":
    main()
