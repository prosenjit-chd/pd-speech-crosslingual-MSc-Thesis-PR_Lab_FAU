import os
import subprocess
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "C:/pd-speech-crosslingual/23_final_supervisor_evaluation"
wsl_base_dir = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation"
visqol_bin = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation/09_visqol/official_visqol_runtime/visqol/bazel-bin/visqol"
visqol_cwd = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation/09_visqol/official_visqol_runtime/visqol"

# Directories for 16k
ref_dir_16k = os.path.join(base_dir, "09_visqol", "audio_16k", "reconstruction_reference")
deg_dir_16k = os.path.join(base_dir, "09_visqol", "audio_16k", "reconstruction_degraded")
os.makedirs(ref_dir_16k, exist_ok=True)
os.makedirs(deg_dir_16k, exist_ok=True)

def wsl_path(win_path):
    return win_path.replace("C:\\", "/mnt/c/").replace("C:/", "/mnt/c/").replace("\\", "/")

def resample_audio(in_path, out_path, target_sr=16000):
    if not os.path.exists(out_path):
        y, sr = librosa.load(in_path, sr=None)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sf.write(out_path, y, target_sr, subtype='PCM_16')
    return out_path

def run_visqol(ref_wsl, deg_wsl):
    cmd = [visqol_bin, "--reference_file", ref_wsl, "--degraded_file", deg_wsl, "--use_speech_mode"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=visqol_cwd)
        for line in res.stdout.split('\n'):
            if "MOS-LQO:" in line:
                return float(line.split(":")[1].strip()), "SUCCESS", ""
        return float('nan'), "FAIL", "MOS-LQO not found in output"
    except subprocess.CalledProcessError as e:
        return float('nan'), "ERROR", str(e.stderr)

print("Loading original and reconstructed indexes...")
orig_csv = "C:/pd-speech-crosslingual/voice_conversion/metadata_full/dataset_index_original_full.csv"
gen_csv = "C:/pd-speech-crosslingual/voice_conversion/metadata_full/dataset_index_generated_full.csv"

df_orig = pd.read_csv(orig_csv)
df_gen = pd.read_csv(gen_csv)

results = []
for i in tqdm(range(len(df_orig)), desc="Reconstruction ViSQOL"):
    orig_path = os.path.join("C:/pd-speech-crosslingual", df_orig.iloc[i]['file_path'].replace('/', '\\'))
    gen_path = os.path.join("C:/pd-speech-crosslingual", df_gen.iloc[i]['file_path'].replace('/', '\\'))
    
    pair_id = f"recon_{i:03d}"
    
    ref_16k = os.path.join(ref_dir_16k, f"{pair_id}_ref.wav")
    deg_16k = os.path.join(deg_dir_16k, f"{pair_id}_deg.wav")
    
    resample_audio(orig_path, ref_16k)
    resample_audio(gen_path, deg_16k)
    
    score, status, msg = run_visqol(wsl_path(ref_16k), wsl_path(deg_16k))
    
    results.append({
        'pair_id': pair_id,
        'speaker_id': df_orig.iloc[i]['speaker_id'],
        'language': df_orig.iloc[i]['language'],
        'pd_hc': df_orig.iloc[i]['label'],
        'reference_path': orig_path,
        'degraded_path': gen_path,
        'visqol_mos_lqo': score,
        'status': status
    })

df_res = pd.DataFrame(results)
out_csv = os.path.join(base_dir, "09_visqol", "results", "visqol_reconstruction_per_pair_OFFICIAL.csv")
df_res.to_csv(out_csv, index=False)

def compute_ci(data):
    if len(data) < 2: return np.nan, np.nan
    return st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))

# Summary
print("Generating Reconstruction Summary...")
df_res['visqol_mos_lqo'] = pd.to_numeric(df_res['visqol_mos_lqo'])

summ_list = []
for (lang, pd_hc), grp in df_res.groupby(['language', 'pd_hc']):
    scores = grp['visqol_mos_lqo'].dropna()
    mean = scores.mean()
    sd = scores.std()
    median = scores.median()
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    ci_low, ci_high = compute_ci(scores)
    summ_list.append({
        'language': lang,
        'pd_hc': pd_hc,
        'count': len(scores),
        'mean': mean,
        'sd': sd,
        'median': median,
        'q1': q1,
        'q3': q3,
        'min': scores.min(),
        'max': scores.max(),
        'ci_lower': ci_low,
        'ci_upper': ci_high
    })
# Overall
scores = df_res['visqol_mos_lqo'].dropna()
ci_low, ci_high = compute_ci(scores)
summ_list.append({
    'language': 'ALL',
    'pd_hc': 'ALL',
    'count': len(scores),
    'mean': scores.mean(),
    'sd': scores.std(),
    'median': scores.median(),
    'q1': scores.quantile(0.25),
    'q3': scores.quantile(0.75),
    'min': scores.min(),
    'max': scores.max(),
    'ci_lower': ci_low,
    'ci_upper': ci_high
})
df_summ = pd.DataFrame(summ_list)
df_summ.to_csv(os.path.join(base_dir, "09_visqol", "results", "visqol_reconstruction_summary_OFFICIAL.csv"), index=False)


print("Loading Cross-language Results for comparison...")
cross_csv = os.path.join(base_dir, "09_visqol", "results", "visqol_crosslanguage_per_pair_OFFICIAL.csv")
df_cross = pd.read_csv(cross_csv)
df_cross['visqol_mos_lqo'] = pd.to_numeric(df_cross['visqol_mos_lqo'])

comp_list = []
comp_list.append({
    'setting': 'Same-Content Reconstruction',
    'count': len(df_res),
    'mean': df_res['visqol_mos_lqo'].mean(),
    'sd': df_res['visqol_mos_lqo'].std(),
    'median': df_res['visqol_mos_lqo'].median(),
    'min': df_res['visqol_mos_lqo'].min(),
    'max': df_res['visqol_mos_lqo'].max()
})
comp_list.append({
    'setting': 'Exploratory Cross-Language',
    'count': len(df_cross),
    'mean': df_cross['visqol_mos_lqo'].mean(),
    'sd': df_cross['visqol_mos_lqo'].std(),
    'median': df_cross['visqol_mos_lqo'].median(),
    'min': df_cross['visqol_mos_lqo'].min(),
    'max': df_cross['visqol_mos_lqo'].max()
})
df_comp = pd.DataFrame(comp_list)
df_comp.to_csv(os.path.join(base_dir, "09_visqol", "results", "visqol_samecontent_vs_crosslanguage_summary.csv"), index=False)

# Figure
plt.figure(figsize=(10, 6))
df_res['Setting'] = 'Same-Content Reconstruction\n(Matched Linguistic Content)'
df_cross['Setting'] = 'Exploratory Cross-Language\n(Mismatched Linguistic Content)'

df_plot = pd.concat([
    df_res[['Setting', 'visqol_mos_lqo']],
    df_cross[['Setting', 'visqol_mos_lqo']]
])

sns.violinplot(data=df_plot, x='Setting', y='visqol_mos_lqo', inner='quartile', palette='Set2')
plt.title('ViSQOL MOS-LQO: Conventional vs Exploratory Application', fontsize=14)
plt.ylabel('ViSQOL MOS-LQO Score')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "09_visqol", "figures", "visqol_samecontent_vs_crosslanguage.png"), dpi=300)

print("Done!")
