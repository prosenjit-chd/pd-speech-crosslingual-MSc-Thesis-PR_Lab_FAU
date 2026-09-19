# Cross-Language Synthesis Model Setup Report

## 1. Model Specifications
- **Model Name**: Coqui XTTS-v2 (`tts_models/multilingual/multi-dataset/xtts_v2`)
- **Architecture**: Multilingual Autoregressive Voice-Cloning Transformer with HiFi-GAN Vocoder integration
- **Supported Languages**: German (`de`), Spanish (`es`), English (`en`), French (`fr`), and 13 other languages
- **Speaker Conditioning**: Zero-shot voice cloning conditioned on reference source speaker waveform
- **Output Sample Rate**: 24000 Hz (High-fidelity uncompressed mono PCM)
- **Framework**: Coqui TTS 0.22.0 / PyTorch 2.14.0+cpu
- **Inference Device**: CPU
- **Model Loading Latency**: 11.89 seconds
- **Deterministic Seed**: 42
- **License**: Coqui Public Model License (Research & Academic Evaluation)

## 2. Technical Differentiation from Previous Work
| Dimension | Previous Acoustic-Domain Shift (Stage 5C) | New True Cross-Language Pipeline |
| :--- | :--- | :--- |
| **Linguistic Content** | Retained original spoken language unchanged | Generated target-language linguistic syntax |
| **Phonetic Inventory** | Original source language phonemes | True target language phonemes (German / Spanish) |
| **Voice Conditioning** | Acoustic-domain latent embedding shift | Speaker-conditioned autoregressive neural synthesis |
| **WER Measurability** | Proxy consistency only (no target transcript) | True reference-based WER against target translation |
