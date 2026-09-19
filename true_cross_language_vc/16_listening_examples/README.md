# Supervisor Listening Review Package: 12-File Pilot

This directory contains the verification protocol and audio references for the **12-File True Cross-Language Voice Conversion Pilot**.

## 1. The 4 Primary Representative Listening Examples
In strict accordance with Phase 19 (Section 42-43):

### Example 1: Spanish Healthy Control $\to$ German (`SP_HC_001`)
- **Original Source**: Spanish HC (`SP_HC_001.wav`)
- **Target Spoken Text (German)**:
  > *„Gestern ging ich zum Arzt. „Was fehlt Ihnen?“, fragte er mich. Ich sagte: „Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!“ „Haben Sie einen gebrochenen Nagel?“ „Ja.“ „Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab.““*
- **Generated Audio**: `05_target_language_synthesis/pilot_12/spanish_to_german/SP_HC_001_to_German_true_crosslanguage.wav`
- **Audio Verification**: Target Language = **German** ($p = 0.9412$), Speaker Sim = **0.7223**, $\text{WER} = 0.696$.

### Example 2: Spanish Parkinson's Disease $\to$ German (`SP_PD_001`)
- **Original Source**: Spanish PD (`SP_PD_001.wav`)
- **Target Spoken Text (German)**: Same German target text.
- **Generated Audio**: `05_target_language_synthesis/pilot_12/spanish_to_german/SP_PD_001_to_German_true_crosslanguage.wav`
- **Audio Verification**: Target Language = **German** ($p = 0.9740$), Speaker Sim = **0.6538**, $\text{WER} = 0.152$.

### Example 3: German Healthy Control $\to$ Spanish (`DE_HC_001`)
- **Original Source**: German HC (`DE_HC_001.wav`)
- **Target Spoken Text (Spanish)**:
  > *„Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México...“*
- **Generated Audio**: `05_target_language_synthesis/pilot_12/german_to_spanish/DE_HC_001_to_Spanish_true_crosslanguage.wav`
- **Audio Verification**: Target Language = **Spanish** ($p = 0.9986$), Speaker Sim = **0.5890**, $\text{WER} = 0.281$.

### Example 4: German Parkinson's Disease $\to$ Spanish (`DE_PD_001`)
- **Original Source**: German PD (`DE_PD_001.wav`)
- **Target Spoken Text (Spanish)**: Same Spanish target text.
- **Generated Audio**: `05_target_language_synthesis/pilot_12/german_to_spanish/DE_PD_001_to_Spanish_true_crosslanguage.wav`
- **Audio Verification**: Target Language = **Spanish** ($p = 0.9982$), Speaker Sim = **0.4709**, $\text{WER} = 0.191$.

## 2. Integrity Compliance
All files mapped strictly via non-identifying safe IDs (`SP_HC_xxx`, etc.). No private clinical metadata is exposed.
