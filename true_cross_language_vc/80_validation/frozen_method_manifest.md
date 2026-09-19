# Frozen True Cross-Language Synthesis & Evaluation Protocol Manifest

**Status:** FROZEN  
**Target:** 80-File True Cross-Language Validation  
**Execution Mode:** Fully Autonomous, Read-Only Upstream, Out-of-Sample Validation  

---

## 1. System & Architecture Specifications

### Models & Dependencies
- **Synthesis Engine:** Coqui XTTS-v2 (`tts_models/multilingual/multi-dataset/xtts_v2`, Coqui TTS 0.22.0)
- **Environment:** Isolated Python 3.10 Virtual Environment (`true_cross_language_vc/env_xtts`)
- **PyTorch Stack:** PyTorch 2.1.2+cu121 with Torchaudio 2.1.2+cu121
- **Hardware Acceleration:** NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM, CUDA 12.1 runtime)
- **Automatic Speech Recognition & LID:** OpenAI Whisper Base (`openai/whisper-base`)
- **Speaker Encoder:** SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)
- **Primary Clinical PD Classifier:** XLSR-53 (`facebook/wav2vec2-large-xlsr-53`), Layer 11 representations, Linear SVM / Logistic Regression

---

## 2. Synthesis Configurations by Direction

### Direction A: Spanish Source $\to$ German Target Speech (C08 Configuration)
- **Target Text:** Canonical professional translation of the PC-GITA reading passage ("El chiste de la uña rota").
- **Chunking Strategy:** `S3_semantic_4chunk` (4 cohesive semantic units; completely eliminates monosyllabic 1-word fragments):
  1. `Gestern ging ich zum Arzt. Was fehlt Ihnen? fragte er mich.` (11 words)
  2. `Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!` (13 words)
  3. `Haben Sie einen gebrochenen Nagel? Ja. Nun, dann wissen wir ja, was es ist!` (14 words)
  4. `Geben Sie Ihren Scheck am Ausgang ab.` (7 words)
- **Reference Audio Strategy:** `R2` deterministic clean 15-second voiced segment (sliding window scoring: maximum voiced fraction, optimal RMS, zero clipping, 25ms cosine boundary fade).
- **Inference Decoding Controls:**
  - `temperature = 0.65`
  - `repetition_penalty = 3.5`
  - `top_p = 0.85`
  - `speed = 1.00`
- **Audio Post-Processing:** Inter-chunk natural silence gap = `200ms`, peak normalization = `0.95` (guarantees zero digital clipping).

### Direction B: German Source $\to$ Spanish Target Speech (Refined Pilot Configuration)
- **Target Text:** Canonical professional translation of the German reading passage ("Die Riesenschildkröten auf den Galapagos-Inseln"):
  1. `Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México.`
  2. `Como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme.`
  3. `Solo en las cercanías de Tampico se observaron unas cinco mil tortugas.`
  4. `En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas.`
  5. `Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas.`
- **Reference Audio Strategy:** `R1` deterministic clean 10-second voiced segment (sliding window scoring, 25ms cosine boundary fade).
- **Inference Decoding Controls:**
  - `temperature = 0.75`
  - `repetition_penalty = 2.0`
  - `top_p = 0.85`
  - `speed = 1.00`
- **Audio Post-Processing:** Inter-sentence silence gap = `250ms`, peak normalization = `0.95`.

---

## 3. Evaluation & Validation Standards
1. **Audio QC:** Strict digital clipping detection (`peak >= 0.999`), NaN/Inf detection, duration bounds, silence ratio.
2. **Language ID:** Whisper first-step logit distribution over language tokens; target probability threshold $\ge 0.85$.
3. **Linguistic Fidelity:** Whisper ASR transcription compared against normalized reference text via `wer`, `cer`, `wip`.
4. **Speaker Similarity:** ECAPA-TDNN cosine similarity against original source audio. Matched pairs evaluated against full empirical distribution of different-speaker negative controls.
5. **Pathology Features:** 17 standardized clinical acoustic features extracted via uniform DSP protocol.
6. **Pathology Normalization Analysis:** Distance-to-HC-centroid statistical evaluation comparing original PD speech vs. converted PD speech.
7. **PD Classification:** Frozen baseline SSL classifiers (trained on real within-language data) tested on real target speech, original cross-language speech, and true cross-language converted speech.
