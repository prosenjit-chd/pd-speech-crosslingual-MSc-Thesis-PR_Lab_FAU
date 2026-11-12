import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier

base_dir = r"C:\pd-speech-crosslingual\23_final_supervisor_evaluation"
paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")

def run_ecapa():
    df = pd.read_csv(paired_csv)
    
    # Check if speechbrain is available and model can be loaded
    print("Loading SpeechBrain ECAPA-TDNN...")
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device":"cuda" if torch.cuda.is_available() else "cpu"}
        )
    except Exception as e:
        print(f"Failed to load ECAPA-TDNN: {e}")
        return

    results = []
    
    # We need to set a random seed for control selection
    rng = np.random.default_rng(42)
    
    for idx, row in df.iterrows():
        sample_id = row['sample_id']
        orig_path = row['original_audio_path']
        conv_path = row['converted_audio_path']
        speaker = row['speaker_id']
        direction = row['conversion_direction']
        
        # Original and converted embeddings
        sig_orig_np, sr_orig = sf.read(orig_path)
        if sig_orig_np.ndim > 1: sig_orig_np = sig_orig_np[:, 0]
        sig_orig = torch.tensor(sig_orig_np, dtype=torch.float32).unsqueeze(0)
        if sr_orig != 16000:
            sig_orig = torchaudio.functional.resample(sig_orig, orig_freq=sr_orig, new_freq=16000)
        emb_orig = classifier.encode_batch(sig_orig).squeeze()

        sig_conv_np, sr_conv = sf.read(conv_path)
        if sig_conv_np.ndim > 1: sig_conv_np = sig_conv_np[:, 0]
        sig_conv = torch.tensor(sig_conv_np, dtype=torch.float32).unsqueeze(0)
        if sr_conv != 16000:
            sig_conv = torchaudio.functional.resample(sig_conv, orig_freq=sr_conv, new_freq=16000)
        emb_conv = classifier.encode_batch(sig_conv).squeeze()

        cos_sim_matched = torch.nn.functional.cosine_similarity(emb_orig.unsqueeze(0), emb_conv.unsqueeze(0)).item()
        
        # Select a control: same direction, different speaker
        possible_controls = df[(df['conversion_direction'] == direction) & (df['speaker_id'] != speaker)]
        if len(possible_controls) > 0:
            control_row = possible_controls.sample(1, random_state=rng).iloc[0]
            control_path = control_row['converted_audio_path']
            sig_ctrl_np, sr_ctrl = sf.read(control_path)
            if sig_ctrl_np.ndim > 1: sig_ctrl_np = sig_ctrl_np[:, 0]
            sig_ctrl = torch.tensor(sig_ctrl_np, dtype=torch.float32).unsqueeze(0)
            if sr_ctrl != 16000:
                sig_ctrl = torchaudio.functional.resample(sig_ctrl, orig_freq=sr_ctrl, new_freq=16000)
            emb_ctrl = classifier.encode_batch(sig_ctrl).squeeze()
            cos_sim_control = torch.nn.functional.cosine_similarity(emb_orig.unsqueeze(0), emb_ctrl.unsqueeze(0)).item()
        else:
            cos_sim_control = np.nan
            
        results.append({
            'sample_id': sample_id,
            'speaker_id': speaker,
            'direction': direction,
            'matched_cosine_similarity': cos_sim_matched,
            'control_cosine_similarity': cos_sim_control
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(df)} ECAPA comparisons...")

    df_res = pd.DataFrame(results)
    out_dir = os.path.join(base_dir, "08_speaker_similarity_ecapa", "results")
    os.makedirs(out_dir, exist_ok=True)
    
    df_res.to_csv(os.path.join(out_dir, "ecapa_per_pair.csv"), index=False)
    
    summary = df_res.groupby('direction')[['matched_cosine_similarity', 'control_cosine_similarity']].agg(['mean', 'std', 'count']).reset_index()
    summary.to_csv(os.path.join(out_dir, "ecapa_direction_summary.csv"), index=False)
    print("ECAPA-TDNN completed.")

if __name__ == "__main__":
    run_ecapa()
