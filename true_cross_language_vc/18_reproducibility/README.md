# Reproducibility Guide: True Cross-Language Voice Conversion Pilot

This folder documents the exact environment, dependencies, and command sequence to reproduce the 12-file pilot results.

## Environments Used
1. **TTS Environment (`env_xtts`)**:
   - Python: 3.10.11 (`true_cross_language_vc/env_xtts`)
   - Model: Coqui TTS (XTTS-v2 checkpoint)
   - Packages: `TTS`, `torch==2.6.0+cpu`, `torchaudio==2.6.0+cpu`, `transformers==4.38.2`, `soundfile`
   
2. **Analysis Environment (`.venv`)**:
   - Python: 3.12.9 (`.venv`)
   - Packages: `speechbrain`, `transformers`, `jiwer`, `scikit-learn`, `pandas`, `librosa`, `soundfile`

## Execution Sequence
To rerun the pilot pipeline from scratch:
```powershell
# 1. Prepare Inventory
& C:\pd-speech-crosslingual\.venv\Scripts\python.exe true_cross_language_vc/scripts/01_prepare_workspace_and_inventory.py

# 2. Extract Transcripts & Translation Pairs
& C:\pd-speech-crosslingual\.venv\Scripts\python.exe true_cross_language_vc/scripts/02_build_source_transcripts.py
& C:\pd-speech-crosslingual\.venv\Scripts\python.exe true_cross_language_vc/scripts/03_create_translation_pairs.py

# 3. Select Pilot 12
& C:\pd-speech-crosslingual\.venv\Scripts\python.exe true_cross_language_vc/scripts/04_select_pilot_12.py

# 4. Model Acceptance Test
& C:\pd-speech-crosslingual\true_cross_language_vc\env_xtts\Scripts\python.exe true_cross_language_vc/scripts/05_test_tts_model.py

# 5. Synthesize Pilot 12
& C:\pd-speech-crosslingual\true_cross_language_vc\env_xtts\Scripts\python.exe true_cross_language_vc/scripts/06_synthesize_pilot_12.py

# 6. Audio Validation
& C:\pd-speech-crosslingual\.venv\Scripts\python.exe true_cross_language_vc/scripts/07_validate_pilot_audio.py

# 7. Language Identification (LID)
& C:\pd-speech-crosslingual\.venv\Scripts\python.exe true_cross_language_vc/scripts/08_verify_language_id.py

# 8. Content Accuracy (ASR & WER)
& C:\pd-speech-crosslingual\.venv\Scripts\python.exe true_cross_language_vc/scripts/09_evaluate_content_accuracy.py

# 9. Speaker Similarity (ECAPA-TDNN)
& C:\pd-speech-crosslingual\.venv\Scripts\python.exe true_cross_language_vc/scripts/10_evaluate_speaker_similarity.py

# 10. Compile Final Report & Listening Checklist
& C:\pd-speech-crosslingual\.venv\Scripts\python.exe true_cross_language_vc/scripts/11_generate_pilot_report.py
```
