import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

base_dir = "C:/pd-speech-crosslingual/23_final_supervisor_evaluation"
paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")
manifest_path = os.path.join(base_dir, "09_visqol", "input_pairs", "visqol_pair_manifest.csv")
ref_dir = os.path.join(base_dir, "09_visqol", "audio_16k", "reference")
deg_dir = os.path.join(base_dir, "09_visqol", "audio_16k", "degraded")

os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
os.makedirs(ref_dir, exist_ok=True)
os.makedirs(deg_dir, exist_ok=True)

df = pd.read_csv(paired_csv)
print(f"Total pairs to process: {len(df)}")

manifest_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing ViSQOL audio"):
    pair_id = f"pair_{idx:03d}"
    
    # Paths
    ref_path = row['original_audio_path']
    deg_path = row['converted_audio_path']
    
    # Target paths
    ref_16k_path = os.path.join(ref_dir, f"{pair_id}_ref.wav")
    deg_16k_path = os.path.join(deg_dir, f"{pair_id}_deg.wav")
    
    # Resample and write Reference
    if not os.path.exists(ref_16k_path):
        y_ref, sr_ref = librosa.load(ref_path, sr=16000, mono=True)
        sf.write(ref_16k_path, y_ref, 16000, subtype='PCM_16')
        
    # Resample and write Degraded
    if not os.path.exists(deg_16k_path):
        y_deg, sr_deg = librosa.load(deg_path, sr=16000, mono=True)
        sf.write(deg_16k_path, y_deg, 16000, subtype='PCM_16')
        
    manifest_rows.append({
        'pair_id': pair_id,
        'subject_id': row['sample_id'],
        'pd_hc': row['diagnosis'],
        'source_language': row['source_language'],
        'target_language': row['target_language'],
        'direction': row['conversion_direction'],
        'reference_original_path': ref_path,
        'degraded_converted_path': deg_path,
        'reference_sr_original': row['original_sample_rate'],
        'converted_sr_original': row['converted_sample_rate'],
        'visqol_reference_16k_path': ref_16k_path,
        'visqol_degraded_16k_path': deg_16k_path
    })

df_manifest = pd.DataFrame(manifest_rows)
df_manifest.to_csv(manifest_path, index=False)
print("Manifest and audio preparation complete.")
