import os
import pandas as pd
import numpy as np
import torch
import soundfile as sf
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

base_dir = r"C:\pd-speech-crosslingual\23_final_supervisor_evaluation"
paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")

def main():
    df = pd.read_csv(paired_csv)
    # deterministic subset of 10 samples
    np.random.seed(42)
    subset_idx = np.random.choice(len(df), size=10, replace=False)
    df_subset = df.iloc[subset_idx].reset_index(drop=True)
    
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    
    results = []
    
    for i in range(len(df_subset)):
        row = df_subset.iloc[i]
        orig_path = row['original_audio_path']
        conv_path = row['converted_audio_path']
        
        # Get a control path (different speaker from same dataset)
        df_control_pool = df[(df['speaker_id'] != row['speaker_id']) & (df['source_language'] == row['source_language'])]
        control_path = df_control_pool.sample(1, random_state=i).iloc[0]['converted_audio_path']
        
        def get_emb(path):
            sig_np, sr = sf.read(path)
            if sig_np.ndim > 1: sig_np = sig_np[:, 0]
            sig = torch.tensor(sig_np, dtype=torch.float32).unsqueeze(0)
            if sr != 16000:
                sig = torchaudio.functional.resample(sig, orig_freq=sr, new_freq=16000)
            return classifier.encode_batch(sig).squeeze()
            
        emb_orig = get_emb(orig_path)
        emb_conv = get_emb(conv_path)
        emb_ctrl = get_emb(control_path)
        
        sim_matched = torch.nn.functional.cosine_similarity(emb_orig.unsqueeze(0), emb_conv.unsqueeze(0)).item()
        sim_control = torch.nn.functional.cosine_similarity(emb_orig.unsqueeze(0), emb_ctrl.unsqueeze(0)).item()
        
        results.append({
            "sample_id": row['sample_id'],
            "speaker_id": row['speaker_id'],
            "direction": row['conversion_direction'],
            "matched_similarity": sim_matched,
            "control_similarity": sim_control,
            "margin": sim_matched - sim_control
        })
        
    df_res = pd.DataFrame(results)
    
    matched_mean = df_res['matched_similarity'].mean()
    control_mean = df_res['control_similarity'].mean()
    
    print("ECAPA Sanity Check Results:")
    print(df_res)
    print(f"\nMatched Mean: {matched_mean:.4f}")
    print(f"Control Mean: {control_mean:.4f}")
    
    out_dir = os.path.join(base_dir, "08_speaker_similarity_ecapa", "results")
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, "ecapa_sanity_check.csv"), index=False)
    
    with open(os.path.join(out_dir, "ecapa_sanity_check.md"), "w") as f:
        f.write("# ECAPA-TDNN Sanity Check\n\n")
        f.write(f"Matched Mean: {matched_mean:.4f}\n")
        f.write(f"Control Mean: {control_mean:.4f}\n")
        f.write(f"Margin: {matched_mean - control_mean:.4f}\n")
        f.write("\nData:\n")
        f.write(df_res.to_markdown(index=False))
        
if __name__ == "__main__":
    main()
