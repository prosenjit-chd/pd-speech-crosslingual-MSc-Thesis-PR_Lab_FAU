# Autonomous Spanish → German Refinement: Reproducibility Package

## 1. System & Execution Environment
```
=== ENVIRONMENT AUDIT: POST-CONFIGURATION ===
Python Version: 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]
Python Executable: C:\pd-speech-crosslingual\true_cross_language_vc\env_xtts\Scripts\python.exe
PyTorch Version: 2.1.2+cu121
Torchaudio Version: 2.1.2+cu121
CUDA Available: True
CUDA Device Count: 1
Device Name: NVIDIA GeForce RTX 4060 Laptop GPU
CUDA Version (torch): 12.1
GPU VRAM Total: 8.00 GB (8585216000 bytes)
TTS Version: 0.22.0
SoundFile Version: 0.14.0
Librosa Version: 0.10.0
Transformers Version: 4.38.2
HuggingFace Hub Version: 0.36.2
jiwer: Not installed in this env (No module named 'jiwer')
speechbrain: Not installed in this env (No module named 'speechbrain')

```

## 2. Model & Checkpoints
- **Primary Voice Conversion Model:** Coqui XTTS-v2 (`tts_models/multilingual/multi-dataset/xtts_v2`, version 0.22.0)
- **Automatic Speech Recognition & LID:** OpenAI Whisper Base (`openai/whisper-base`)
- **Speaker Encoder:** SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)

## 3. Selected Frozen Configuration
- **Candidate ID:** `C08`
- **Mean WER:** `0.3424`
- **Mean WIP:** `0.6056`
- **Mean Speaker Cosine:** `0.6753`

## 4. Deterministic DSP & Normalization Rules
- **Sampling Rates:** 24,000 Hz (XTTS-v2 native) -> 16,000 Hz (evaluation & feature extraction)
- **Peak Normalization:** Fixed at `0.95` (guarantees zero digital clipping)
- **Boundary Fading:** `25ms` cosine fade-in and fade-out on all extracted reference segments
- **Inter-chunk Silence Gap:** Fixed at `200ms` natural pause
- **Random Seed:** Fixed at `42`

## 5. Checksum Audit of Final Generated Audio
| Filename | Safe ID | Duration (s) | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `SP_HC_001_final_to_German.wav` | `SP_HC_001` | 49.41 | `1dfd5e8f4e62392ddfbb0ee8c9d44f86cf5a4c68906158682303654876c186c0` |
| `SP_HC_002_final_to_German.wav` | `SP_HC_002` | 52.70 | `8be25d7e93e5fe6ec94b51919257cf000d69c2591c0968b2ff709a424b4a3401` |
| `SP_HC_003_final_to_German.wav` | `SP_HC_003` | 46.57 | `39ab6d628275cac3f954a45fe581f643a8269873a0593041fa7f957799df646c` |
| `SP_PD_001_final_to_German.wav` | `SP_PD_001` | 48.25 | `8217e6e219e49164c3f7fb1a7fa3c2b1a467ebf715cbbc1ea0ceb6bbeaf4f772` |
| `SP_PD_002_final_to_German.wav` | `SP_PD_002` | 44.17 | `d753dba8e932f3e63a06102e9c6889f1a7454dbf33ae39583c57b43a4f7d7f23` |
| `SP_PD_003_final_to_German.wav` | `SP_PD_003` | 53.08 | `c752ae698c87d23ddad79fa38c69005486c1f47c2a16ce8cd04c6f1e5143cbc9` |
