import os
import torch
import soundfile as sf
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    run_opts={"device":"cuda" if torch.cuda.is_available() else "cpu"}
)

orig_path = r"C:\pd-speech-crosslingual\input\Spanish\readtext_PDvsHC\readtext\hc\AVPEPUDEAC0001_readtext.wav"
sig_np, sr = sf.read(orig_path)
print("sf.read shape:", sig_np.shape)
sig_np = torch.tensor(sig_np, dtype=torch.float32)
if sig_np.ndim > 1: sig_np = sig_np[:, 0]

print("tensor shape:", sig_np.shape)

try:
    print("Trying encode_batch with [time]")
    emb = classifier.encode_batch(sig_np).squeeze()
    print("Success [time]!")
except Exception as e:
    print("Failed [time]:", e)

try:
    print("Trying encode_batch with [1, time]")
    emb = classifier.encode_batch(sig_np.unsqueeze(0)).squeeze()
    print("Success [1, time]!")
except Exception as e:
    print("Failed [1, time]:", e)

# Use torchaudio without torchcodec by doing sf.read then converting to torchaudio shape
try:
    sig_ta = sig_np.unsqueeze(0) # [1, time]
    sig_norm = classifier.audio_normalizer(sig_ta, sr)
    print("audio_normalizer output shape:", sig_norm.shape)
    emb = classifier.encode_batch(sig_norm).squeeze()
    print("Success after audio_normalizer!")
except Exception as e:
    print("Failed after audio_normalizer:", e)
