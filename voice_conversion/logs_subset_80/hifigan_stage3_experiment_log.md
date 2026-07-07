# HiFi-GAN Stage 3: Controlled Subset Experiment (80 Files)

## Goal
Test whether HiFi-GAN can successfully reconstruct Spanish and German PD/HC speech samples on a larger, balanced 80-file subset.

## Current Status
- **Environment Verification**: **COMPLETED** (Python virtual environment verified).
- **Stage 3 Scripts Status**: **CREATED**
  - Selection & Orchestration: `run_subset_80_stage.py`
  - Preprocessing: `prepare_hifigan_subset_80_audio.py`
  - Inspect & Compare: `inspect_and_compare_generated_subset_80.py`
  - Evaluation: `evaluate_reconstructed_subset_80.py`
- **Data Selection Status**: **COMPLETED** (80 files selected and copied).
- **Preprocessing Status**: **SUCCESS**

## Selected Files Summary
The following 80 balanced readtext WAV files were copied and renamed from the index:
- `SP_PD_001.wav` (Original: `AVPEPUDEA0001_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 15.13s)
- `SP_PD_002.wav` (Original: `AVPEPUDEA0002_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 26.14s)
- `SP_PD_003.wav` (Original: `AVPEPUDEA0003_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 25.13s)
- `SP_PD_004.wav` (Original: `AVPEPUDEA0005_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 20.61s)
- `SP_PD_005.wav` (Original: `AVPEPUDEA0006_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 15.58s)
- `SP_PD_006.wav` (Original: `AVPEPUDEA0007_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 23.64s)
- `SP_PD_007.wav` (Original: `AVPEPUDEA0008_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 17.28s)
- `SP_PD_008.wav` (Original: `AVPEPUDEA0009_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 18.96s)
- `SP_PD_009.wav` (Original: `AVPEPUDEA0010_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 18.36s)
- `SP_PD_010.wav` (Original: `AVPEPUDEA0011_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 15.46s)
- `SP_PD_011.wav` (Original: `AVPEPUDEA0013_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 20.96s)
- `SP_PD_012.wav` (Original: `AVPEPUDEA0014_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 15.83s)
- `SP_PD_013.wav` (Original: `AVPEPUDEA0015_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 14.99s)
- `SP_PD_014.wav` (Original: `AVPEPUDEA0016_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 20.54s)
- `SP_PD_015.wav` (Original: `AVPEPUDEA0017_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 21.36s)
- `SP_PD_016.wav` (Original: `AVPEPUDEA0020_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 14.32s)
- `SP_PD_017.wav` (Original: `AVPEPUDEA0021_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 14.22s)
- `SP_PD_018.wav` (Original: `AVPEPUDEA0022_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 12.28s)
- `SP_PD_019.wav` (Original: `AVPEPUDEA0023_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 13.21s)
- `SP_PD_020.wav` (Original: `AVPEPUDEA0024_readtext.wav`, Language: `Spanish`, Diagnosis: `PD`, Duration: 24.56s)
- `SP_HC_001.wav` (Original: `AVPEPUDEAC0001_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 17.47s)
- `SP_HC_002.wav` (Original: `AVPEPUDEAC0003_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 16.99s)
- `SP_HC_003.wav` (Original: `AVPEPUDEAC0004_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 17.08s)
- `SP_HC_004.wav` (Original: `AVPEPUDEAC0005_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 14.36s)
- `SP_HC_005.wav` (Original: `AVPEPUDEAC0006_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 17.80s)
- `SP_HC_006.wav` (Original: `AVPEPUDEAC0007_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 18.87s)
- `SP_HC_007.wav` (Original: `AVPEPUDEAC0008_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 17.91s)
- `SP_HC_008.wav` (Original: `AVPEPUDEAC0010_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 19.35s)
- `SP_HC_009.wav` (Original: `AVPEPUDEAC0011_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 16.75s)
- `SP_HC_010.wav` (Original: `AVPEPUDEAC0012_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 17.36s)
- `SP_HC_011.wav` (Original: `AVPEPUDEAC0013_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 14.48s)
- `SP_HC_012.wav` (Original: `AVPEPUDEAC0014_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 19.52s)
- `SP_HC_013.wav` (Original: `AVPEPUDEAC0015_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 15.92s)
- `SP_HC_014.wav` (Original: `AVPEPUDEAC0016_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 19.32s)
- `SP_HC_015.wav` (Original: `AVPEPUDEAC0017_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 13.24s)
- `SP_HC_016.wav` (Original: `AVPEPUDEAC0018_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 17.41s)
- `SP_HC_017.wav` (Original: `AVPEPUDEAC0019_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 18.31s)
- `SP_HC_018.wav` (Original: `AVPEPUDEAC0020_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 12.60s)
- `SP_HC_019.wav` (Original: `AVPEPUDEAC0021_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 17.74s)
- `SP_HC_020.wav` (Original: `AVPEPUDEAC0022_readtext.wav`, Language: `Spanish`, Diagnosis: `HC`, Duration: 18.70s)
- `DE_PD_001.wav` (Original: `002.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 47.46s)
- `DE_PD_002.wav` (Original: `003.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 38.54s)
- `DE_PD_003.wav` (Original: `007.u2.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 40.48s)
- `DE_PD_004.wav` (Original: `013.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 34.62s)
- `DE_PD_005.wav` (Original: `014.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 43.57s)
- `DE_PD_006.wav` (Original: `016.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 68.24s)
- `DE_PD_007.wav` (Original: `019.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 49.70s)
- `DE_PD_008.wav` (Original: `023.u2.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 55.31s)
- `DE_PD_009.wav` (Original: `024.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 37.73s)
- `DE_PD_010.wav` (Original: `025.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 39.20s)
- `DE_PD_011.wav` (Original: `028.u2.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 40.81s)
- `DE_PD_012.wav` (Original: `029.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 49.97s)
- `DE_PD_013.wav` (Original: `032.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 38.89s)
- `DE_PD_014.wav` (Original: `039.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 43.37s)
- `DE_PD_015.wav` (Original: `042.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 44.40s)
- `DE_PD_016.wav` (Original: `043.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 56.02s)
- `DE_PD_017.wav` (Original: `044.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 45.34s)
- `DE_PD_018.wav` (Original: `046.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 40.43s)
- `DE_PD_019.wav` (Original: `050.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 41.15s)
- `DE_PD_020.wav` (Original: `053.u1.02.wav`, Language: `German`, Diagnosis: `PD`, Duration: 56.66s)
- `DE_HC_001.wav` (Original: `001.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 40.76s)
- `DE_HC_002.wav` (Original: `003.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 46.50s)
- `DE_HC_003.wav` (Original: `005.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 37.36s)
- `DE_HC_004.wav` (Original: `006.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 60.13s)
- `DE_HC_005.wav` (Original: `008.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 33.78s)
- `DE_HC_006.wav` (Original: `009.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 56.66s)
- `DE_HC_007.wav` (Original: `010.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 51.13s)
- `DE_HC_008.wav` (Original: `011.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 61.95s)
- `DE_HC_009.wav` (Original: `012.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 45.34s)
- `DE_HC_010.wav` (Original: `013.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 38.41s)
- `DE_HC_011.wav` (Original: `014.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 44.44s)
- `DE_HC_012.wav` (Original: `015.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 45.20s)
- `DE_HC_013.wav` (Original: `016.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 43.84s)
- `DE_HC_014.wav` (Original: `017.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 46.25s)
- `DE_HC_015.wav` (Original: `018.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 44.40s)
- `DE_HC_016.wav` (Original: `019.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 44.49s)
- `DE_HC_017.wav` (Original: `020.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 42.52s)
- `DE_HC_018.wav` (Original: `022.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 62.65s)
- `DE_HC_019.wav` (Original: `023.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 53.82s)
- `DE_HC_020.wav` (Original: `024.u1.02.wav`, Language: `German`, Diagnosis: `HC`, Duration: 44.39s)

## Preprocessing Details
- Input directory: `voice_conversion/input_subset_80`
- Processed directory: `voice_conversion/input_subset_80_22050` (Mono, 22050 Hz, Amplitude Normalized)
- Preprocessing log location: `voice_conversion/logs_subset_80/subset_80_preprocessing_log.csv`
- Technical inspection summary: `voice_conversion/logs_subset_80/subset_80_inspection_summary.csv`

## Planned Evaluation Criteria
1. **Audio duration comparison**: Check if the reconstructed WAV duration matches the original file exactly.
2. **Audio technical statistics**: Compare sample rate, channels, peak amplitude, and RMS energy between original and reconstructed audio.
3. **Downstream Classification**: Run baseline embedding extraction (WavLM, Wav2Vec2, XLSR) and cross-language classification scenarios.

## HiFi-GAN Reconstruction Status
- **HiFi-GAN Reconstruction**: **COMPLETED** (80 files generated in `generated_subset_80`).
- **Post-Generation Validation**: **SUCCESSFUL**

## Generated Audio Comparisons
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
- `DE_HC_016.wav <-> DE_HC_016_generated.wav`: Orig Dur: 44.4895s | Gen Dur: 44.4894s | Diff: -0.0000s | Orig RMS: 0.1150 -> Gen RMS: 0.1043
- `DE_HC_017.wav <-> DE_HC_017_generated.wav`: Orig Dur: 42.5186s | Gen Dur: 42.5157s | Diff: -0.0029s | Orig RMS: 0.1102 -> Gen RMS: 0.0934
- `DE_HC_018.wav <-> DE_HC_018_generated.wav`: Orig Dur: 62.6474s | Gen Dur: 62.6474s | Diff: 0.0000s | Orig RMS: 0.0910 -> Gen RMS: 0.0821
- `DE_HC_019.wav <-> DE_HC_019_generated.wav`: Orig Dur: 53.8239s | Gen Dur: 53.8239s | Diff: -0.0000s | Orig RMS: 0.0865 -> Gen RMS: 0.0789
- `DE_HC_020.wav <-> DE_HC_020_generated.wav`: Orig Dur: 44.3908s | Gen Dur: 44.3849s | Diff: -0.0059s | Orig RMS: 0.1228 -> Gen RMS: 0.1104
- `DE_PD_001.wav <-> DE_PD_001_generated.wav`: Orig Dur: 47.4601s | Gen Dur: 47.4500s | Diff: -0.0102s | Orig RMS: 0.1066 -> Gen RMS: 0.0960
- `DE_PD_002.wav <-> DE_PD_002_generated.wav`: Orig Dur: 38.5437s | Gen Dur: 38.5335s | Diff: -0.0102s | Orig RMS: 0.1457 -> Gen RMS: 0.1286
- `DE_PD_003.wav <-> DE_PD_003_generated.wav`: Orig Dur: 40.4768s | Gen Dur: 40.4724s | Diff: -0.0044s | Orig RMS: 0.1600 -> Gen RMS: 0.1457
- `DE_PD_004.wav <-> DE_PD_004_generated.wav`: Orig Dur: 34.6210s | Gen Dur: 34.6210s | Diff: 0.0000s | Orig RMS: 0.1654 -> Gen RMS: 0.1459
- `DE_PD_005.wav <-> DE_PD_005_generated.wav`: Orig Dur: 43.5723s | Gen Dur: 43.5722s | Diff: -0.0000s | Orig RMS: 0.1547 -> Gen RMS: 0.1428
- `DE_PD_006.wav <-> DE_PD_006_generated.wav`: Orig Dur: 68.2449s | Gen Dur: 68.2434s | Diff: -0.0015s | Orig RMS: 0.1191 -> Gen RMS: 0.1043
- `DE_PD_007.wav <-> DE_PD_007_generated.wav`: Orig Dur: 49.7023s | Gen Dur: 49.7023s | Diff: 0.0000s | Orig RMS: 0.1570 -> Gen RMS: 0.1461
- `DE_PD_008.wav <-> DE_PD_008_generated.wav`: Orig Dur: 55.3078s | Gen Dur: 55.2983s | Diff: -0.0095s | Orig RMS: 0.0787 -> Gen RMS: 0.0697
- `DE_PD_009.wav <-> DE_PD_009_generated.wav`: Orig Dur: 37.7252s | Gen Dur: 37.7208s | Diff: -0.0044s | Orig RMS: 0.1008 -> Gen RMS: 0.0850
- `DE_PD_010.wav <-> DE_PD_010_generated.wav`: Orig Dur: 39.2000s | Gen Dur: 39.1953s | Diff: -0.0047s | Orig RMS: 0.3116 -> Gen RMS: 0.2841
- `DE_PD_011.wav <-> DE_PD_011_generated.wav`: Orig Dur: 40.8077s | Gen Dur: 40.7975s | Diff: -0.0102s | Orig RMS: 0.1083 -> Gen RMS: 0.1003
- `DE_PD_012.wav <-> DE_PD_012_generated.wav`: Orig Dur: 49.9693s | Gen Dur: 49.9693s | Diff: 0.0000s | Orig RMS: 0.0938 -> Gen RMS: 0.0819
- `DE_PD_013.wav <-> DE_PD_013_generated.wav`: Orig Dur: 38.8920s | Gen Dur: 38.8818s | Diff: -0.0102s | Orig RMS: 0.1241 -> Gen RMS: 0.1154
- `DE_PD_014.wav <-> DE_PD_014_generated.wav`: Orig Dur: 43.3749s | Gen Dur: 43.3749s | Diff: 0.0000s | Orig RMS: 0.1025 -> Gen RMS: 0.0965
- `DE_PD_015.wav <-> DE_PD_015_generated.wav`: Orig Dur: 44.3966s | Gen Dur: 44.3966s | Diff: -0.0000s | Orig RMS: 0.1450 -> Gen RMS: 0.1346
- `DE_PD_016.wav <-> DE_PD_016_generated.wav`: Orig Dur: 56.0239s | Gen Dur: 56.0181s | Diff: -0.0058s | Orig RMS: 0.1461 -> Gen RMS: 0.1205
- `DE_PD_017.wav <-> DE_PD_017_generated.wav`: Orig Dur: 45.3355s | Gen Dur: 45.3254s | Diff: -0.0102s | Orig RMS: 0.1760 -> Gen RMS: 0.1672
- `DE_PD_018.wav <-> DE_PD_018_generated.wav`: Orig Dur: 40.4288s | Gen Dur: 40.4259s | Diff: -0.0029s | Orig RMS: 0.1912 -> Gen RMS: 0.1679
- `DE_PD_019.wav <-> DE_PD_019_generated.wav`: Orig Dur: 41.1458s | Gen Dur: 41.1458s | Diff: 0.0000s | Orig RMS: 0.0821 -> Gen RMS: 0.0767
- `DE_PD_020.wav <-> DE_PD_020_generated.wav`: Orig Dur: 56.6567s | Gen Dur: 56.6567s | Diff: 0.0000s | Orig RMS: 0.0890 -> Gen RMS: 0.0803
- `SP_HC_001.wav <-> SP_HC_001_generated.wav`: Orig Dur: 17.4703s | Gen Dur: 17.4614s | Diff: -0.0089s | Orig RMS: 0.1176 -> Gen RMS: 0.1129
- `SP_HC_002.wav <-> SP_HC_002_generated.wav`: Orig Dur: 16.9921s | Gen Dur: 16.9854s | Diff: -0.0067s | Orig RMS: 0.0798 -> Gen RMS: 0.0755
- `SP_HC_003.wav <-> SP_HC_003_generated.wav`: Orig Dur: 17.0762s | Gen Dur: 17.0667s | Diff: -0.0095s | Orig RMS: 0.1102 -> Gen RMS: 0.1053
- `SP_HC_004.wav <-> SP_HC_004_generated.wav`: Orig Dur: 14.3562s | Gen Dur: 14.3499s | Diff: -0.0063s | Orig RMS: 0.1672 -> Gen RMS: 0.1551
- `SP_HC_005.wav <-> SP_HC_005_generated.wav`: Orig Dur: 17.7995s | Gen Dur: 17.7981s | Diff: -0.0014s | Orig RMS: 0.1034 -> Gen RMS: 0.0988
- `SP_HC_006.wav <-> SP_HC_006_generated.wav`: Orig Dur: 18.8682s | Gen Dur: 18.8662s | Diff: -0.0020s | Orig RMS: 0.0736 -> Gen RMS: 0.0727
- `SP_HC_007.wav <-> SP_HC_007_generated.wav`: Orig Dur: 17.9127s | Gen Dur: 17.9026s | Diff: -0.0101s | Orig RMS: 0.1396 -> Gen RMS: 0.1285
- `SP_HC_008.wav <-> SP_HC_008_generated.wav`: Orig Dur: 19.3468s | Gen Dur: 19.3422s | Diff: -0.0046s | Orig RMS: 0.1081 -> Gen RMS: 0.1025
- `SP_HC_009.wav <-> SP_HC_009_generated.wav`: Orig Dur: 16.7474s | Gen Dur: 16.7416s | Diff: -0.0059s | Orig RMS: 0.1137 -> Gen RMS: 0.1024
- `SP_HC_010.wav <-> SP_HC_010_generated.wav`: Orig Dur: 17.3584s | Gen Dur: 17.3569s | Diff: -0.0015s | Orig RMS: 0.1314 -> Gen RMS: 0.1253
- `SP_HC_011.wav <-> SP_HC_011_generated.wav`: Orig Dur: 14.4784s | Gen Dur: 14.4776s | Diff: -0.0008s | Orig RMS: 0.1520 -> Gen RMS: 0.1446
- `SP_HC_012.wav <-> SP_HC_012_generated.wav`: Orig Dur: 19.5163s | Gen Dur: 19.5048s | Diff: -0.0115s | Orig RMS: 0.0760 -> Gen RMS: 0.0719
- `SP_HC_013.wav <-> SP_HC_013_generated.wav`: Orig Dur: 15.9191s | Gen Dur: 15.9173s | Diff: -0.0018s | Orig RMS: 0.1250 -> Gen RMS: 0.1209
- `SP_HC_014.wav <-> SP_HC_014_generated.wav`: Orig Dur: 19.3163s | Gen Dur: 19.3074s | Diff: -0.0089s | Orig RMS: 0.0843 -> Gen RMS: 0.0817
- `SP_HC_015.wav <-> SP_HC_015_generated.wav`: Orig Dur: 13.2367s | Gen Dur: 13.2354s | Diff: -0.0013s | Orig RMS: 0.0862 -> Gen RMS: 0.0848
- `SP_HC_016.wav <-> SP_HC_016_generated.wav`: Orig Dur: 17.4083s | Gen Dur: 17.4034s | Diff: -0.0049s | Orig RMS: 0.0911 -> Gen RMS: 0.0869
- `SP_HC_017.wav <-> SP_HC_017_generated.wav`: Orig Dur: 18.3059s | Gen Dur: 18.2973s | Diff: -0.0085s | Orig RMS: 0.0971 -> Gen RMS: 0.0850
- `SP_HC_018.wav <-> SP_HC_018_generated.wav`: Orig Dur: 12.5980s | Gen Dur: 12.5968s | Diff: -0.0012s | Orig RMS: 0.1084 -> Gen RMS: 0.1028
- `SP_HC_019.wav <-> SP_HC_019_generated.wav`: Orig Dur: 17.7380s | Gen Dur: 17.7284s | Diff: -0.0095s | Orig RMS: 0.0935 -> Gen RMS: 0.0881
- `SP_HC_020.wav <-> SP_HC_020_generated.wav`: Orig Dur: 18.6970s | Gen Dur: 18.6921s | Diff: -0.0049s | Orig RMS: 0.1290 -> Gen RMS: 0.1217
- `SP_PD_001.wav <-> SP_PD_001_generated.wav`: Orig Dur: 15.1292s | Gen Dur: 15.1278s | Diff: -0.0014s | Orig RMS: 0.1162 -> Gen RMS: 0.1038
- `SP_PD_002.wav <-> SP_PD_002_generated.wav`: Orig Dur: 26.1377s | Gen Dur: 26.1341s | Diff: -0.0037s | Orig RMS: 0.0940 -> Gen RMS: 0.0895
- `SP_PD_003.wav <-> SP_PD_003_generated.wav`: Orig Dur: 25.1338s | Gen Dur: 25.1240s | Diff: -0.0098s | Orig RMS: 0.0767 -> Gen RMS: 0.0744
- `SP_PD_004.wav <-> SP_PD_004_generated.wav`: Orig Dur: 20.6068s | Gen Dur: 20.5961s | Diff: -0.0107s | Orig RMS: 0.0784 -> Gen RMS: 0.0678
- `SP_PD_005.wav <-> SP_PD_005_generated.wav`: Orig Dur: 15.5778s | Gen Dur: 15.5690s | Diff: -0.0088s | Orig RMS: 0.1166 -> Gen RMS: 0.1110
- `SP_PD_006.wav <-> SP_PD_006_generated.wav`: Orig Dur: 23.6392s | Gen Dur: 23.6379s | Diff: -0.0013s | Orig RMS: 0.0675 -> Gen RMS: 0.0658
- `SP_PD_007.wav <-> SP_PD_007_generated.wav`: Orig Dur: 17.2841s | Gen Dur: 17.2756s | Diff: -0.0085s | Orig RMS: 0.1126 -> Gen RMS: 0.1030
- `SP_PD_008.wav <-> SP_PD_008_generated.wav`: Orig Dur: 18.9591s | Gen Dur: 18.9591s | Diff: -0.0000s | Orig RMS: 0.0650 -> Gen RMS: 0.0611
- `SP_PD_009.wav <-> SP_PD_009_generated.wav`: Orig Dur: 18.3615s | Gen Dur: 18.3554s | Diff: -0.0061s | Orig RMS: 0.1101 -> Gen RMS: 0.1047
- `SP_PD_010.wav <-> SP_PD_010_generated.wav`: Orig Dur: 15.4626s | Gen Dur: 15.4529s | Diff: -0.0098s | Orig RMS: 0.0941 -> Gen RMS: 0.0886
- `SP_PD_011.wav <-> SP_PD_011_generated.wav`: Orig Dur: 20.9575s | Gen Dur: 20.9560s | Diff: -0.0015s | Orig RMS: 0.1021 -> Gen RMS: 0.0931
- `SP_PD_012.wav <-> SP_PD_012_generated.wav`: Orig Dur: 15.8310s | Gen Dur: 15.8244s | Diff: -0.0066s | Orig RMS: 0.1355 -> Gen RMS: 0.1259
- `SP_PD_013.wav <-> SP_PD_013_generated.wav`: Orig Dur: 14.9928s | Gen Dur: 14.9885s | Diff: -0.0044s | Orig RMS: 0.1021 -> Gen RMS: 0.0926
- `SP_PD_014.wav <-> SP_PD_014_generated.wav`: Orig Dur: 20.5371s | Gen Dur: 20.5264s | Diff: -0.0107s | Orig RMS: 0.0692 -> Gen RMS: 0.0619
- `SP_PD_015.wav <-> SP_PD_015_generated.wav`: Orig Dur: 21.3617s | Gen Dur: 21.3507s | Diff: -0.0110s | Orig RMS: 0.1206 -> Gen RMS: 0.1062
- `SP_PD_016.wav <-> SP_PD_016_generated.wav`: Orig Dur: 14.3213s | Gen Dur: 14.3151s | Diff: -0.0062s | Orig RMS: 0.1163 -> Gen RMS: 0.1075
- `SP_PD_017.wav <-> SP_PD_017_generated.wav`: Orig Dur: 14.2153s | Gen Dur: 14.2106s | Diff: -0.0047s | Orig RMS: 0.1228 -> Gen RMS: 0.1141
- `SP_PD_018.wav <-> SP_PD_018_generated.wav`: Orig Dur: 12.2775s | Gen Dur: 12.2717s | Diff: -0.0058s | Orig RMS: 0.0859 -> Gen RMS: 0.0823
- `SP_PD_019.wav <-> SP_PD_019_generated.wav`: Orig Dur: 13.2061s | Gen Dur: 13.2005s | Diff: -0.0055s | Orig RMS: 0.1207 -> Gen RMS: 0.1107
- `SP_PD_020.wav <-> SP_PD_020_generated.wav`: Orig Dur: 24.5605s | Gen Dur: 24.5551s | Diff: -0.0054s | Orig RMS: 0.0918 -> Gen RMS: 0.0761

## Preprocessed vs. Reconstructed Evaluation Summary
* **Sample Rate**: All generated audios are at 22050 Hz.
* **Duration**: The reconstructed file duration matches the input duration extremely closely.
* **Acoustics**: Reconstructed peak amplitudes and RMS values range within expected signal boundaries.
