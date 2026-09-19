import os
os.environ["COQUI_TOS_AGREED"] = "1"
import sys
if sys.platform == "win32":
    if os.path.exists(r"C:\Windows\System32"):
        os.add_dll_directory(r"C:\Windows\System32")
import torch
import functools
import time
from pathlib import Path
import soundfile as sf
import numpy as np

# Patch torch.load for PyTorch 2.6+
_orig_torch_load = torch.load
@functools.wraps(_orig_torch_load)
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Patch torchaudio.load with soundfile
import torchaudio
_orig_torchaudio_load = torchaudio.load
def _soundfile_load(filepath, *args, **kwargs):
    try:
        data, sr = sf.read(str(filepath))
        if data.ndim == 1:
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        else:
            tensor = torch.tensor(data.T, dtype=torch.float32)
        return tensor, sr
    except Exception:
        return _orig_torchaudio_load(filepath, *args, **kwargs)
torchaudio.load = _soundfile_load

print("Imports complete. Torch CUDA:", torch.cuda.is_available())
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading TTS on {device}...")
t0 = time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"TTS loaded in {time.time() - t0:.2f}s")

ref_wav = r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement\03_reference_audio_analysis\clean_references\SP_HC_001_R1_10s.wav"
out_wav = r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement\04_xtts_parameter_search\test_gpu.wav"
Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

print("Synthesizing test sentence...")
t1 = time.time()
wav = tts.tts(
    text="Gestern ging ich zum Arzt.",
    speaker_wav=ref_wav,
    language="de",
    temperature=0.65,
    top_p=0.85,
    repetition_penalty=3.0,
    speed=1.0
)
synth_time = time.time() - t1
sf.write(out_wav, np.array(wav, dtype=np.float32), 24000)
dur = len(wav) / 24000
print(f"Synthesis complete in {synth_time:.2f}s! Audio duration: {dur:.2f}s. RTF: {synth_time / dur:.2f}")
