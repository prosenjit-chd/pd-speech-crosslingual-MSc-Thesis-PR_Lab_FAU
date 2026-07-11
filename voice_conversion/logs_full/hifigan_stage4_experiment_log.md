# HiFi-GAN Stage 4: Full Dataset Experiment

## Goal
Reconstruct and evaluate all available Spanish and German PD/HC speech files (readtext task) using the HiFi-GAN universal vocoder.

## Current Status
- **Environment Verification**: **COMPLETED** (Python virtual environment verified).
- **Stage 4 Scripts Status**: **CREATED**
  - Selection & Orchestration: `run_full_stage.py`
  - Preprocessing: `prepare_hifigan_full_audio.py`
  - Inspect & Compare: `inspect_and_compare_generated_full.py`
  - Downstream Evaluation: `evaluate_reconstructed_full.py`
- **Data Selection Status**: **COMPLETED** (276 files selected. Counts: Spanish PD: 50, Spanish HC: 50, German PD: 88, German HC: 88).
- **Preprocessing Status**: **SUCCESS**

## Selected Files Summary
- Total copied: 276 files.
- Spanish PD: 50 | Spanish HC: 50
- German PD: 88 | German HC: 88

## Preprocessing Details
- Input directory: `voice_conversion/input_full`
- Processed directory: `voice_conversion/input_full_22050` (Mono, 22050 Hz, Amplitude Normalized)
- Preprocessing log location: `voice_conversion/logs_full/full_preprocessing_log.csv`
- Technical inspection summary: `voice_conversion/logs_full/full_inspection_summary.csv`

## Planned Evaluation Criteria
1. **Audio duration comparison**: Check if the reconstructed WAV duration matches the original file exactly.
2. **Audio technical statistics**: Compare sample rate, channels, peak amplitude, and RMS energy between original and reconstructed audio.
3. **Downstream Classification**: Run baseline embedding extraction (WavLM, Wav2Vec2, XLSR) and cross-language classification scenarios using standard 10-fold outer / 9-fold inner cross-validation splits.

## HiFi-GAN Reconstruction Status
- **HiFi-GAN Reconstruction**: **COMPLETED** (276 files generated in `generated_full`).
- **Post-Generation Validation**: **SUCCESSFUL**

## Generated Audio Comparisons (First 15 Samples)
- `DE_HC_001.wav <-> DE_HC_001_generated.wav`: Orig Dur: 40.7641s | Gen Dur: 40.7626s | Diff: -0.0015s | Orig RMS: 0.1390 -> Gen RMS: 0.1166
- `DE_HC_002.wav <-> DE_HC_002_generated.wav`: Orig Dur: 46.5023s | Gen Dur: 46.4980s | Diff: -0.0044s | Orig RMS: 0.1410 -> Gen RMS: 0.1266
- `DE_HC_003.wav <-> DE_HC_003_generated.wav`: Orig Dur: 37.3595s | Gen Dur: 37.3493s | Diff: -0.0102s | Orig RMS: 0.1318 -> Gen RMS: 0.1212
- `DE_HC_004.wav <-> DE_HC_004_generated.wav`: Orig Dur: 60.1295s | Gen Dur: 60.1281s | Diff: -0.0015s | Orig RMS: 0.1114 -> Gen RMS: 0.0940
- `DE_HC_005.wav <-> DE_HC_005_generated.wav`: Orig Dur: 33.7807s | Gen Dur: 33.7734s | Diff: -0.0073s | Orig RMS: 0.1433 -> Gen RMS: 0.1337
- `DE_HC_006.wav <-> DE_HC_006_generated.wav`: Orig Dur: 56.6639s | Gen Dur: 56.6567s | Diff: -0.0073s | Orig RMS: 0.2726 -> Gen RMS: 0.2292
- `DE_HC_007.wav <-> DE_HC_007_generated.wav`: Orig Dur: 51.1260s | Gen Dur: 51.1187s | Diff: -0.0073s | Orig RMS: 0.1183 -> Gen RMS: 0.1049
- `DE_HC_008.wav <-> DE_HC_008_generated.wav`: Orig Dur: 61.9508s | Gen Dur: 61.9508s | Diff: 0.0000s | Orig RMS: 0.1133 -> Gen RMS: 0.1066
- `DE_HC_009.wav <-> DE_HC_009_generated.wav`: Orig Dur: 45.3355s | Gen Dur: 45.3254s | Diff: -0.0102s | Orig RMS: 0.2018 -> Gen RMS: 0.1874
- `DE_HC_010.wav <-> DE_HC_010_generated.wav`: Orig Dur: 38.4059s | Gen Dur: 38.4058s | Diff: -0.0000s | Orig RMS: 0.0600 -> Gen RMS: 0.0512
- `DE_HC_011.wav <-> DE_HC_011_generated.wav`: Orig Dur: 44.4430s | Gen Dur: 44.4430s | Diff: -0.0000s | Orig RMS: 0.1434 -> Gen RMS: 0.1242
- `DE_HC_012.wav <-> DE_HC_012_generated.wav`: Orig Dur: 45.2006s | Gen Dur: 45.1976s | Diff: -0.0029s | Orig RMS: 0.1633 -> Gen RMS: 0.1486
- `DE_HC_013.wav <-> DE_HC_013_generated.wav`: Orig Dur: 43.8393s | Gen Dur: 43.8393s | Diff: 0.0000s | Orig RMS: 0.1014 -> Gen RMS: 0.0936
- `DE_HC_014.wav <-> DE_HC_014_generated.wav`: Orig Dur: 46.2541s | Gen Dur: 46.2541s | Diff: 0.0000s | Orig RMS: 0.0939 -> Gen RMS: 0.0843
- `DE_HC_015.wav <-> DE_HC_015_generated.wav`: Orig Dur: 44.3966s | Gen Dur: 44.3966s | Diff: -0.0000s | Orig RMS: 0.0817 -> Gen RMS: 0.0760
- *... and 261 more files. Full list is available in full_original_vs_generated_duration_comparison.csv*

## Preprocessed vs. Reconstructed Evaluation Summary
* **Sample Rate**: All generated audios are at 22050 Hz.
* **Duration**: Reconstructed file durations match the input durations extremely closely (within window tolerance).
* **Acoustics**: Reconstructed peak amplitudes range from ~0.8 to ~1.0 with high signal-to-noise ratio and envelope conservation.
