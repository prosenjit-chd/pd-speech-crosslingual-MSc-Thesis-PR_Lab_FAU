import os
import subprocess
import pandas as pd
from tqdm import tqdm
import scipy.stats as st
import numpy as np
import scipy.stats as st
import numpy as np

base_dir = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation"
manifest_path = os.path.join(base_dir, "09_visqol", "input_pairs", "visqol_reconstruction_manifest.csv")
visqol_bin = os.path.join(base_dir, "09_visqol", "official_visqol_runtime", "visqol", "bazel-bin", "visqol")
visqol_cwd = os.path.join(base_dir, "09_visqol", "official_visqol_runtime", "visqol")
results_dir = os.path.join(base_dir, "09_visqol", "results")

def wsl_path(win_path):
    return win_path.replace("C:\\", "/mnt/c/").replace("C:/", "/mnt/c/").replace("\\", "/")

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

df = pd.read_csv(manifest_path)
print("Running Same-Content Reconstruction Full Cohort...")
results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Recon ViSQOL"):
    ref_wsl = wsl_path(row['visqol_reference_16k_path'])
    deg_wsl = wsl_path(row['visqol_degraded_16k_path'])
    
    score, status, msg = run_visqol(ref_wsl, deg_wsl)
    
    results.append({
        'pair_id': row['pair_id'],
        'subject_id': row['subject_id'],
        'pd_hc': row['pd_hc'],
        'direction': row['direction'],
        'source_language': row['source_language'],
        'target_language': row['target_language'],
        'reference_path': row['visqol_reference_16k_path'],
        'degraded_path': row['visqol_degraded_16k_path'],
        'visqol_mos_lqo': score,
        'status': status,
        'error_message': msg
    })

df_res = pd.DataFrame(results)
out_file = os.path.join(results_dir, "visqol_reconstruction_per_pair_OFFICIAL.csv")
df_res.to_csv(out_file, index=False)
print(f"Evaluation complete. Saved to {out_file}")

# Stats
df_res['visqol_mos_lqo'] = pd.to_numeric(df_res['visqol_mos_lqo'])

def compute_ci(data):
    if len(data) < 2: return np.nan, np.nan
    return st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))

summ_list = []
for (lang, pd_hc), grp in df_res.groupby(['source_language', 'pd_hc']):
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
df_summ.to_csv(os.path.join(results_dir, "visqol_reconstruction_summary_OFFICIAL.csv"), index=False)

# Comparative Analysis
print("Loading Cross-language Results for comparison...")
cross_csv = os.path.join(results_dir, "visqol_crosslanguage_per_pair_OFFICIAL.csv")
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
df_comp.to_csv(os.path.join(results_dir, "visqol_samecontent_vs_crosslanguage_summary.csv"), index=False)

print("Finished Same-Content Reconstruction ViSQOL!")
