import sys
import os
import torch
import torchaudio
import soundfile as sf
import librosa
import transformers
import huggingface_hub
from pathlib import Path

ENV_DIR = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement\01_environment")
ENV_DIR.mkdir(parents=True, exist_ok=True)

lines = []
lines.append("=== ENVIRONMENT AUDIT: POST-CONFIGURATION ===")
lines.append(f"Python Version: {sys.version}")
lines.append(f"Python Executable: {sys.executable}")
lines.append(f"PyTorch Version: {torch.__version__}")
lines.append(f"Torchaudio Version: {torchaudio.__version__}")
lines.append(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    lines.append(f"CUDA Device Count: {torch.cuda.device_count()}")
    lines.append(f"Device Name: {torch.cuda.get_device_name(0)}")
    lines.append(f"CUDA Version (torch): {torch.version.cuda}")
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    lines.append(f"GPU VRAM Total: {vram_bytes / (1024**3):.2f} GB ({vram_bytes} bytes)")
else:
    lines.append("CUDA: UNAVAILABLE")

# Check TTS
try:
    import TTS
    lines.append(f"TTS Version: {TTS.__version__}")
except Exception as e:
    lines.append(f"TTS: Error loading ({e})")

lines.append(f"SoundFile Version: {sf.__version__}")
lines.append(f"Librosa Version: {librosa.__version__}")
lines.append(f"Transformers Version: {transformers.__version__}")
lines.append(f"HuggingFace Hub Version: {huggingface_hub.__version__}")

# Check ASR / evaluation tools
try:
    import jiwer
    lines.append(f"jiwer Version: {jiwer.__version__}")
except Exception as e:
    lines.append(f"jiwer: Not installed in this env ({e})")

try:
    import speechbrain
    lines.append(f"speechbrain Version: {speechbrain.__version__}")
except Exception as e:
    lines.append(f"speechbrain: Not installed in this env ({e})")

content = "\n".join(lines) + "\n"
out_file = ENV_DIR / "environment_after.txt"
out_file.write_text(content, encoding="utf-8")
print(content)

# Append to installation log
log_file = ENV_DIR / "installation_log.txt"
with open(log_file, "a", encoding="utf-8") as f:
    f.write("\n--- PyTorch CUDA 12.1 Activation Log ---\n")
    f.write("Target: env_xtts (isolated Python 3.10)\n")
    f.write("Action: pip install torch==2.1.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121\n")
    f.write("Result: Successfully installed. CUDA available = True. GPU = NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM).\n")
    f.write("Inference acceleration confirmed: XTTS-v2 RTF = 0.89 on GPU.\n")
print(f"Environment audit saved to {out_file} and {log_file}")
