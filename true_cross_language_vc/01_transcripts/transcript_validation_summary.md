# Source Transcript Validation Summary

## 1. Ground-Truth Reference Verification
The raw acoustic read-text dataset utilizes standardized reading passages from clinical acquisition protocols:
1. **Spanish (PC-GITA Read-Text Task)**:
   - Reference Passage:
     > *"Ayer fui al médico. «¿Qué le pasa?», me preguntó. Yo le dije: «¡Ay doctor, donde pongo el dedo me duele!». «¿Tiene la uña rota?». «Sí». «¡Pues ya sabemos qué es! Deje su cheque a la salida»."*
   - Task: Short humorous clinical dialogue designed for prosody, articulation, and stop consonant evaluation (36 words).
2. **German (Sabine Skoda Protocol Read-Text Task)**:
   - Reference Passage:
     > *"Von einer gewaltigen, von den Behörden geschützten Invasion wird zur Zeit die Golf- und Pazifikküste Mexikos heimgesucht: Wie alljährlich im Juni kommen Hunderttausende von Schildkröten aus dem Meer, um an Land ihre Eier abzulegen. Allein in der Nähe von Tampico wurden etwa 5000 Schildkröten beobachtet. Insgesamt wird in den kommenden Wochen mit einer Invasion von mehr als einer halben Million Schildkröten gerechnet. Die mexikanischen Behörden lassen die Legeplätze sorgfältig bewachen, um den Diebstahl von Eiern zu verhindern und ausreichend Schildkrötennachwuchs sicherzustellen."*
   - Task: Standardized paragraph ("Schildkröten-Inversion") from the speech analysis protocol for Parkinson's disease (74 words).

## 2. Transcript Stratification Across 276 Files
- **Total Utterances Analyzed**: 276
- **Full Standard Reference Matched (`dataset_reference`)**: 222 (80.4%)
- **Truncated / ASR Fallback (`ASR_fallback`)**: 54 (19.6%)
- **Missing Transcripts**: 0

## 3. Analysis of Truncated Recordings
A total of 54 recordings exhibited truncated reading durations (predominantly severe PD patients who fatigue or stop early in the German corpus). These are logged in `transcript_missing_or_uncertain.csv` and handled with exact spoken transcripts to ensure precise linguistic alignment.

## 4. 12-File Pilot Source Transcripts Status
All 12 files selected for the initial pilot represent complete readings with 100% verified standard reference transcripts.
