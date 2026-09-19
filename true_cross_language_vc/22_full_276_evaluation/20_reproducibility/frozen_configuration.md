# Frozen Synthesis and Evaluation Configuration Manifest
## 1. Spanish -> German Direction
- **Engine**: Coqui XTTS-v2 (v0.22.0)
- **Model Checkpoint**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Chunking Strategy**: `S3_semantic_4chunk`
- **Speaker Reference**: R2 Clean 15.0s Voiced Reference (SAD + Energy peak selection, 25ms boundary cosine fade)
- **Inference Parameters**: Temperature = 0.65, Repetition Penalty = 3.5, Top_p = 0.85, Speed = 1.00
- **Peak Normalization**: 0.95
- **Inter-chunk Silence**: 0.20s

## 2. German -> Spanish Direction
- **Engine**: Coqui XTTS-v2 (v0.22.0)
- **Model Checkpoint**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Chunking Strategy**: 5-Sentence Segmentation
- **Speaker Reference**: R1 Clean 10.0s Voiced Reference (25ms boundary cosine fade)
- **Inference Parameters**: Temperature = 0.75, Repetition Penalty = 2.0, Top_p = 0.85, Speed = 1.00
- **Peak Normalization**: 0.95
- **Inter-sentence Silence**: 0.25s

## 3. Downstream Evaluation Protocols
- **Audio QC**: Sample rate 16kHz/24kHz, Channels mono, Peak clipping threshold 0.999, Silence ratio < 0.80
- **LID Protocol**: `openai/whisper-base` first-token logits
- **ASR Protocol**: `openai/whisper-base` transcription task, jiwer WER/CER/WIP
- **Speaker Embedding**: Pretrained `speechbrain/spkrec-ecapa-voxceleb`
- **PD Classifier**: Pretrained `facebook/wav2vec2-large-xlsr-53` Layer 11, Linear SVM & Logistic Regression
