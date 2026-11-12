import os
import subprocess
import pandas as pd
from tqdm import tqdm

base_dir = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation"
manifest_path = os.path.join(base_dir, "09_visqol", "input_pairs", "visqol_pair_manifest.csv")
visqol_bin = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation/09_visqol/official_visqol_runtime/visqol/bazel-bin/visqol"
results_dir = os.path.join(base_dir, "09_visqol", "results")
wsl_manifest_path = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation/09_visqol/input_pairs/visqol_pair_manifest.csv"

def run_visqol(ref_wsl, deg_wsl):
    cmd = [visqol_bin, "--reference_file", ref_wsl, "--degraded_file", deg_wsl, "--use_speech_mode"]
    visqol_cwd = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation/09_visqol/official_visqol_runtime/visqol"
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=visqol_cwd)
        # Parse the MOS-LQO value from stdout
        for line in res.stdout.split('\n'):
            if "MOS-LQO:" in line:
                return float(line.split(":")[1].strip()), "SUCCESS", ""
        return float('nan'), "FAIL", "MOS-LQO not found in output"
    except subprocess.CalledProcessError as e:
        print(f"CMD failed: {e.cmd}")
        print(f"STDERR: {e.stderr}")
        return float('nan'), "ERROR", str(e.stderr)

def wsl_path(win_path):
    # Convert C:\... or C:/... to /mnt/c/...
    return win_path.replace("C:\\", "/mnt/c/").replace("C:/", "/mnt/c/").replace("\\", "/")

def sanity_checks(df):
    print("Running Sanity Checks...")
    sanity_results = []
    
    # Take 2 samples from Spanish and 2 from German for self-reference
    samples = df.groupby('source_language').head(2)
    for idx, row in samples.iterrows():
        ref_path = wsl_path(row['visqol_reference_16k_path'])
        score, status, msg = run_visqol(ref_path, ref_path)
        sanity_results.append({
            'test': f"Self-reference {row['source_language']}",
            'reference': ref_path,
            'degraded': ref_path,
            'score': score,
            'status': status
        })
        print(f"Self-reference {row['source_language']} ({row['pair_id']}): {score} - {status}")
        
    df_sanity = pd.DataFrame(sanity_results)
    df_sanity.to_csv(os.path.join(results_dir, "visqol_sanity_checks.csv"), index=False)
    
    # Check if sanity checks passed (scores > 4.0 typically for self-reference)
    passed = all(s['score'] > 4.0 for s in sanity_results if s['status'] == 'SUCCESS')
    if not passed or len(sanity_results) == 0:
        raise Exception("Sanity checks failed! Stopping execution.")
    print("Sanity checks passed.\n")

def run_full_cohort(df):
    print("Running Full Cohort...")
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="ViSQOL evaluation"):
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
    out_file = os.path.join(results_dir, "visqol_crosslanguage_per_pair_OFFICIAL.csv")
    df_res.to_csv(out_file, index=False)
    print(f"Evaluation complete. Saved to {out_file}")
    
    # Summary
    if df_res['status'].eq('SUCCESS').all():
        print("All pairs processed successfully.")
    else:
        fails = df_res[df_res['status'] != 'SUCCESS']
        print(f"WARNING: {len(fails)} pairs failed!")
        
    # Stats
    df_res['visqol_mos_lqo'] = pd.to_numeric(df_res['visqol_mos_lqo'])
    summ = df_res.groupby('direction')['visqol_mos_lqo'].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
    summ.to_csv(os.path.join(results_dir, "visqol_crosslanguage_summary_OFFICIAL.csv"))
    print("\nSummary by direction:")
    print(summ)

if __name__ == "__main__":
    df = pd.read_csv(wsl_manifest_path)
    sanity_checks(df)
    run_full_cohort(df)
