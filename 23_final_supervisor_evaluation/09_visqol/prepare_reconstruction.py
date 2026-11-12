import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

base_dir = "C:/pd-speech-crosslingual/23_final_supervisor_evaluation"

ref_dir_16k = os.path.join(base_dir, "09_visqol", "audio_16k", "reconstruction_reference")
deg_dir_16k = os.path.join(base_dir, "09_visqol", "audio_16k", "reconstruction_degraded")
os.makedirs(ref_dir_16k, exist_ok=True)
os.makedirs(deg_dir_16k, exist_ok=True)

def resample_audio(in_path, out_path, target_sr=16000):
    if not os.path.exists(out_path):
        y, sr = librosa.load(in_path, sr=None)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sf.write(out_path, y, target_sr, subtype='PCM_16')
    return out_path

orig_csv = "C:/pd-speech-crosslingual/voice_conversion/metadata_full/dataset_index_original_full.csv"
gen_csv = "C:/pd-speech-crosslingual/voice_conversion/metadata_full/dataset_index_generated_full.csv"

df_orig = pd.read_csv(orig_csv)
df_gen = pd.read_csv(gen_csv)

results = []
for i in tqdm(range(len(df_orig)), desc="Preparing Reconstruction 16k"):
    orig_path = os.path.join("C:/pd-speech-crosslingual", df_orig.iloc[i]['file_path'].replace('/', '\\'))
    gen_path = os.path.join("C:/pd-speech-crosslingual", df_gen.iloc[i]['file_path'].replace('/', '\\'))
    
    pair_id = f"recon_{i:03d}"
    
    ref_16k = os.path.join(ref_dir_16k, f"{pair_id}_ref.wav")
    deg_16k = os.path.join(deg_dir_16k, f"{pair_id}_deg.wav")
    
    resample_audio(orig_path, ref_16k)
    resample_audio(gen_path, deg_16k)
    
    results.append({
        'pair_id': pair_id,
        'subject_id': df_orig.iloc[i]['speaker_id'],
        'pd_hc': df_orig.iloc[i]['label'],
        'source_language': df_orig.iloc[i]['language'],
        'target_language': df_orig.iloc[i]['language'],
        'direction': f"{df_orig.iloc[i]['language']}_to_{df_orig.iloc[i]['language']}",
        'visqol_reference_16k_path': ref_16k,
        'visqol_degraded_16k_path': deg_16k
    })

df_res = pd.DataFrame(results)
manifest_csv = os.path.join(base_dir, "09_visqol", "input_pairs", "visqol_reconstruction_manifest.csv")
df_res.to_csv(manifest_csv, index=False)
print(f"Manifest written to {manifest_csv}")
