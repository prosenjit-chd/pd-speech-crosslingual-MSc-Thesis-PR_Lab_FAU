# HiFi-GAN Inference Script & README Review

This document reviews the official HiFi-GAN `inference.py` script and the accompanying repository `README.md` to establish the correct execution strategy and checkpoint configurations.

---

## 1. Technical Inspection of `inference.py`

### Required Arguments
* **`--checkpoint_file`**: **Required** (`required=True` in `argparse`). This is the path to the generator checkpoint binary (e.g. `generator_v1`).

### Optional Arguments & Default Folders
* **`--input_wavs_dir`**: Defaults to `'test_files'`. 
* **`--output_dir`**: Defaults to `'generated_files'`.

### Data Expectation (WAV vs. Mel Spectrograms)
* The script **expects raw WAV files** as input, not Mel spectrograms. 
* It reads the directory specified by `--input_wavs_dir`, loads each WAV using `load_wav` (which rescales it by `MAX_WAV_VALUE` to floating point `[-1.0, 1.0]`), converts it to a Mel-spectrogram internally via `get_mel`, runs it through the generator, and writes the reconstructed output WAV.
* *Note*: If you need to synthesize directly from pre-computed numpy Mel-spectrogram arrays, the repository provides a separate script `inference_e2e.py` which takes `--input_mels_dir` (default `test_mel_files`) and reconstructs WAVs from Mel files.

### Checkpoint & Configuration Loading
* The script loads the generator state dict using:
  ```python
  state_dict_g = load_checkpoint(a.checkpoint_file, device)
  generator.load_state_dict(state_dict_g['generator'])
  ```
* **Critical Configuration Dependency**: The script resolves the path to its hyperparameters file using:
  ```python
  config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
  ```
  This means a file named **`config.json` MUST reside in the same folder** as the generator checkpoint file. If you run:
  `--checkpoint_file voice_conversion/checkpoints/generator_v1`
  The script will look for:
  `voice_conversion/checkpoints/config.json`
  If it is missing, execution will fail with a `FileNotFoundError`.

### CUDA Usage and CPU Fallback
* **CUDA Detection**: The script automatically checks if CUDA is available:
  ```python
  if torch.cuda.is_available():
      torch.cuda.manual_seed(h.seed)
      device = torch.device('cuda')
  else:
      device = torch.device('cpu')
  ```
* **CPU Fallback Support**: Fully supported out-of-the-box. The `load_checkpoint` helper calls `torch.load(filepath, map_location=device)`. This ensures that even if the model was trained on GPU, loading it on a CPU-only environment will not crash. No code modifications or patches are required.

---

## 2. Review of HiFi-GAN `README.md`

### Pretrained Checkpoint Instructions & Download Link
* Pretrained checkpoints are hosted at:
  `https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing`

### Available Pretrained Models & Naming
* **`LJ_V1`**: Generator V1 trained on LJSpeech (single female speaker, English, 22.05 kHz).
* **`LJ_V2`**: Generator V2 trained on LJSpeech (smaller footprint version).
* **`LJ_V3`**: Generator V3 trained on LJSpeech (smallest footprint version).
* **`LJ_FT_T2_V1 / V2 / V3`**: Fine-tuned on Tacotron 2 outputs.
* **`VCTK_V1 / V2 / V3`**: Trained on VCTK (multi-speaker English, 22.05 kHz).
* **`UNIVERSAL_V1`**: Generator V1 trained on a multi-speaker/universal database.

### Recommended Checkpoint for Crosslingual Thesis
* **`UNIVERSAL_V1`** is highly recommended because it is trained on a diverse set of speakers/acoustic conditions and acts as a generic vocoder. Since our study covers both Spanish and German speakers, a speaker-independent/universal model will generalize much better than a single-speaker model (like `LJ_V1`).
* **`LJ_V1`** is also a valid alternative fallback to confirm that the pipeline behaves identically to standard literature baselines.

### Official Inference Commands
For standard reconstruction from WAV files:
```bash
python inference.py --checkpoint_file [generator checkpoint file path] --input_wavs_dir [input WAV folder] --output_dir [output WAV folder]
```

For reconstruction directly from Mel-spectrogram `.npy` arrays:
```bash
python inference_e2e.py --checkpoint_file [generator checkpoint file path] --input_mels_dir [input Mel folder] --output_dir [output WAV folder]
```
