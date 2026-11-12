import os
import pandas as pd
import numpy as np
import soundfile as sf
import torch

base_dir = r"C:\pd-speech-crosslingual\23_final_supervisor_evaluation"
paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")

def main():
    df = pd.read_csv(paired_csv)
    # Check first 5 samples
    results = []
    
    for i in range(5):
        row = df.iloc[i]
        orig_path = row['original_audio_path']
        
        sig_np, sr = sf.read(orig_path)
        
        is_mono = (sig_np.ndim == 1) or (sig_np.shape[1] == 1)
        if sig_np.ndim > 1:
            sig_np = sig_np[:, 0]
            
        sig = torch.tensor(sig_np, dtype=torch.float32).unsqueeze(0)
        
        results.append({
            "sample_id": row['sample_id'],
            "file_type": "original",
            "sample_rate": sr,
            "is_mono": is_mono,
            "dtype": str(sig.dtype),
            "amplitude_min": float(sig.min()),
            "amplitude_max": float(sig.max()),
            "tensor_shape": str(list(sig.shape)),
            "is_zero_length": sig.shape[1] == 0
        })
        
        conv_path = row['converted_audio_path']
        sig_np, sr = sf.read(conv_path)
        
        is_mono = (sig_np.ndim == 1) or (sig_np.shape[1] == 1)
        if sig_np.ndim > 1:
            sig_np = sig_np[:, 0]
            
        sig = torch.tensor(sig_np, dtype=torch.float32).unsqueeze(0)
        
        results.append({
            "sample_id": row['sample_id'],
            "file_type": "converted",
            "sample_rate": sr,
            "is_mono": is_mono,
            "dtype": str(sig.dtype),
            "amplitude_min": float(sig.min()),
            "amplitude_max": float(sig.max()),
            "tensor_shape": str(list(sig.shape)),
            "is_zero_length": sig.shape[1] == 0
        })
        
    df_res = pd.DataFrame(results)
    print(df_res)
    
    out_dir = os.path.join(base_dir, "08_speaker_similarity_ecapa", "qc")
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, "ecapa_input_tensor_validation.csv"), index=False)
    print("Saved QC check to ecapa_input_tensor_validation.csv")

if __name__ == "__main__":
    main()
