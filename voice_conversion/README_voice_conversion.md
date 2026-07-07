# Voice Conversion for Crosslingual Parkinson's Disease Detection

This folder contains the voice conversion pipeline for the Master Thesis project: **Voice Conversion for Crosslingual Detection of Parkinson's Disease Using Speech Signals**.

## Purpose of this Folder
This folder is dedicated to testing, developing, and running the voice conversion pipeline. The primary focus of the initial stage is to reconstruct/synthesize speech signals using a HiFi-GAN vocoder to ensure valid audio quality can be restored from the Parkinson's Disease (PD) and Healthy Control (HC) speech datasets.

## Supervisor Task Summary
The supervisor, Tomás, requested to start the voice conversion pipeline.
1. Use HiFi-GAN for synthesizing speech signals.
2. Explore whether speech representations/embeddings from baseline models (WavLM, Wav2Vec2, XLSR) can be used as conditioning inputs to generate new speech samples.
3. Compare original and generated/converted speech samples using the same evaluation setup as the baseline models to study classification performance and acoustic properties.

## Current Setup Status
- **Environment**: **READY** (verified Python 3.10.11 environment setup).
- **Python Version**: `3.10.11`
- **Virtual Environment Path**: `C:\pd-speech-crosslingual\voice_conversion\hifi-gan\hifigan_env`
- **Dependency Customization**: 
  The original HiFi-GAN codebase's `requirements.txt` was bypassed because it specifies outdated libraries (like `torch==1.4.0`) that are incompatible with modern CUDA architectures and Python 3.10. Modern equivalent libraries have been installed manually into `hifigan_env` to ensure Windows 10/11 and GPU support.

## Current Completed Work
* Environment verified and requirements frozen (`requirements_windows_py310_freeze.txt` generated).
* Custom `.gitignore` configured to prevent private audio, features, or checkpoints from being committed.
* Created [`prepare_hifigan_pilot_audio.py`](file:///C:/pd-speech-crosslingual/voice_conversion/scripts/prepare_hifigan_pilot_audio.py) and [`inspect_audio_folder.py`](file:///C:/pd-speech-crosslingual/voice_conversion/scripts/inspect_audio_folder.py) scripts.
* Inspected HiFi-GAN `inference.py` and `README.md` to map execution and configuration details.
* **Pilot Dataset prepared & verified**: 12 balanced files copied, renamed, preprocessed to 22050 Hz (mono, peak-normalized), and technically verified.
* **Model Configuration prepared**: `config.json` copied to the target directory.

---

## Next Steps & Inference Strategy

### 1. Download & Placement of Pre-trained Checkpoint (Pending User Action)
> [!WARNING]
> The official Google Drive link in the HiFi-GAN repository currently returns a **404 error** and is inaccessible in the browser. 
> To bypass this, download the checkpoint file from a trusted community Hugging Face mirror and save it to the project checkpoints directory:

#### Recommended Safe Mirror (Option A):
* **Source**: `huseinzol05/jik876-UNIVERSAL_V1` on Hugging Face.
* **Files**:
  * Checkpoint File: [g_02500000](https://huggingface.co/huseinzol05/jik876-UNIVERSAL_V1/resolve/main/g_02500000)
* **Instructions**:
  1. Download the `g_02500000` file from the link above.
  2. Save it into the directory: `C:\pd-speech-crosslingual\voice_conversion\checkpoints\universal_v1\`
  3. Rename it from `g_02500000` to **`generator_v1`** (no file extension).
  4. Ensure it resides in the same directory as the pre-configured `config.json`.

### 2. Run Reconstruction Inference (WAV Reconstruction)
Once the checkpoint file `C:\pd-speech-crosslingual\voice_conversion\checkpoints\universal_v1\generator_v1` is in place, execute the reconstruction in CMD:
```cmd
cd C:\pd-speech-crosslingual\voice_conversion\hifi-gan

C:\pd-speech-crosslingual\voice_conversion\hifi-gan\hifigan_env\Scripts\python.exe inference.py --checkpoint_file C:\pd-speech-crosslingual\voice_conversion\checkpoints\universal_v1\generator_v1 --input_wavs_dir C:\pd-speech-crosslingual\voice_conversion\input_pilot_22050 --output_dir C:\pd-speech-crosslingual\voice_conversion\generated
```

---

## Privacy & Security Note
> [!WARNING]
> To comply with data privacy policies and keep the GitHub repository lightweight:
> - **DO NOT push raw audio** (`voice_conversion/input_pilot/*.wav`) or **processed audio** (`voice_conversion/input_pilot_22050/*.wav`) to GitHub.
> - **DO NOT push generated audio** (`voice_conversion/generated/*.wav`) to GitHub.
> - **DO NOT push checkpoints** (`voice_conversion/checkpoints/**/*.pt`, `voice_conversion/checkpoints/**/generator_*`) to GitHub.
> - **DO NOT push raw features/embeddings** (`features/`) to GitHub.
> All these paths are ignored via the project's `.gitignore`.
