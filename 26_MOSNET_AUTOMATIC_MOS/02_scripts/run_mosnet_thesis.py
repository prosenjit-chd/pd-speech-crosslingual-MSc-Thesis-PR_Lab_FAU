#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis Pretrained MOSNet Automatic MOS Evaluation Pipeline
Author: MSc Thesis Candidate (PR Lab FAU)
Supervisor guidance: Dr.-Ing. Tomás Arias-Vergara

Evaluates automatic MOSNet-predicted perceptual mean opinion score (MOS)
for:
  1. Original Clinical Speech (Spanish N=100, German N=176; Total N=276)
  2. Final XTTS-v2 Cross-Language Generated Speech (ES->DE N=100, DE->ES N=176; Total N=276)
  3. HiFi-GAN Same-Content Reconstructed Speech (Spanish N=100, German N=176; Total N=276)

IMPORTANT:
- Official pretrained CNN-BLSTM checkpoint from Lo et al. (Interspeech 2019) is used.
- Mathematical architecture and pretrained weights are strictly preserved.
- No model retraining or fine-tuning is performed.
- Source audio files are processed in-place without modification.
- Output predictions are automatic MOSNet estimates, not human subjective ratings.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Compatibility Bridge for Modern SciPy (Zero modification to repo code)
# -------------------------------------------------------------------------
import scipy.signal
import scipy.signal.windows
if not hasattr(scipy.signal, 'hamming'):
    scipy.signal.hamming = scipy.signal.windows.hamming

# -------------------------------------------------------------------------
# 2. Add Cloned MOSNet Repository to Python Path
# -------------------------------------------------------------------------
REPO_DIR = r"C:\pd-speech-crosslingual\26_MOSNET_AUTOMATIC_MOS\MOSNet"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from model import CNN_BLSTM
import utils

# -------------------------------------------------------------------------
# 3. Path Configurations
# -------------------------------------------------------------------------
PROJECT_ROOT = r"C:\pd-speech-crosslingual"
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "26_MOSNET_AUTOMATIC_MOS")
MODEL_PATH = os.path.join(REPO_DIR, "pre_trained", "cnn_blstm.h5")

PAIRED_INDEX_PATH = os.path.join(
    PROJECT_ROOT, "23_final_supervisor_evaluation", "02_dataset_index", "paired_original_converted_index.csv"
)
HIFIGAN_ORIG_INDEX = os.path.join(
    PROJECT_ROOT, "voice_conversion", "metadata_full", "dataset_index_original_full.csv"
)
HIFIGAN_GEN_INDEX = os.path.join(
    PROJECT_ROOT, "voice_conversion", "metadata_full", "dataset_index_generated_full.csv"
)

OUT_ORIG_DIR = os.path.join(OUTPUT_BASE, "03_original_speech_results")
OUT_XTTS_DIR = os.path.join(OUTPUT_BASE, "04_converted_speech_results")
OUT_HIFI_DIR = os.path.join(OUTPUT_BASE, "05_hifigan_reconstruction_results")
OUT_TABLES_DIR = os.path.join(OUTPUT_BASE, "06_comparison_tables")
OUT_FIGS_DIR = os.path.join(OUTPUT_BASE, "07_figures")
OUT_AUDIT_DIR = os.path.join(OUTPUT_BASE, "09_audit")

# -------------------------------------------------------------------------
# 4. Helper: Audio Preprocessing & Spectrogram Extraction
# -------------------------------------------------------------------------
def get_audio_info(file_path):
    """Retrieve audio metadata (sample rate, channels, duration) using soundfile."""
    try:
        info = sf.info(file_path)
        return {
            'orig_sr': info.samplerate,
            'channels': info.channels,
            'duration': info.duration,
            'status': 'OK'
        }
    except Exception as e:
        return {
            'orig_sr': None,
            'channels': None,
            'duration': None,
            'status': f'ERROR_READ: {str(e)}'
        }

def compute_mosnet_prediction(model, audio_path):
    """
    Compute MOSNet prediction using official feature extraction logic.
    Spectrogram: 16 kHz STFT, N_FFT=512, hop=256, win=512, Hamming window.
    Dimension: (1, time_steps, 257).
    Returns: (predicted_mos, num_frames, elapsed_sec)
    """
    t0 = time.time()
    # utils.get_spectrograms loads audio with sr=16000 and computes STFT magnitude
    mag_sgram = utils.get_spectrograms(audio_path)
    time_steps = mag_sgram.shape[0]
    mag_sgram_batch = np.reshape(mag_sgram, (1, time_steps, utils.SGRAM_DIM))
    
    # Predict with CNN-BLSTM model
    avg_score, frame_score = model.predict(mag_sgram_batch, verbose=0, batch_size=1)
    predicted_mos = float(avg_score[0][0])
    elapsed = time.time() - t0
    return predicted_mos, time_steps, elapsed

# -------------------------------------------------------------------------
# 5. Main Execution Routine
# -------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("MOSNet Automatic MOS Evaluation Pipeline for MSc Thesis")
    print("=" * 80)
    
    # Verify model file
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MOSNet pretrained weights not found at: {MODEL_PATH}")
    print(f"[1/6] Loading MOSNet CNN-BLSTM architecture from {MODEL_PATH}...")
    mosnet_builder = CNN_BLSTM()
    model = mosnet_builder.build()
    model.load_weights(MODEL_PATH)
    print(f"      Model loaded successfully ({model.count_params():,} parameters).")
    
    # Load dataset indices
    print(f"[2/6] Loading dataset indices...")
    df_paired = pd.read_csv(PAIRED_INDEX_PATH)
    print(f"      Loaded paired index: N={len(df_paired)} pairs.")
    df_hifi_orig = pd.read_csv(HIFIGAN_ORIG_INDEX)
    df_hifi_gen = pd.read_csv(HIFIGAN_GEN_INDEX)
    print(f"      Loaded HiFi-GAN indices: Orig N={len(df_hifi_orig)}, Gen N={len(df_hifi_gen)}.")
    
    # Establish canonical mapping based on base sample ID
    df_paired['base_id'] = df_paired['sample_id'].apply(lambda x: os.path.splitext(x)[0])
    df_hifi_orig['base_id'] = df_hifi_orig['file_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df_hifi_gen['base_id'] = df_hifi_gen['file_path'].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0].replace('_generated', '')
    )
    
    # Map HiFi-GAN paths to base_id
    hifi_orig_map = dict(zip(df_hifi_orig['base_id'], df_hifi_orig['file_path']))
    hifi_gen_map = dict(zip(df_hifi_gen['base_id'], df_hifi_gen['file_path']))
    
    # Prepare lists for per-file results and preprocessing logs
    results_orig = []
    results_xtts = []
    results_hifi = []
    prep_logs = []
    
    print(f"[3/6] Running MOSNet inference across all 3 cohorts (N=276 each, Total=828 evaluations)...")
    
    for idx, row in tqdm(df_paired.iterrows(), total=len(df_paired), desc="Evaluating files"):
        base_id = row['base_id']
        speaker_id = row['speaker_id']
        src_lang = row['source_language']
        tgt_lang = row['target_language']
        diag = row['diagnosis']
        direction = row['conversion_direction'] # SP_to_GE or GE_to_SP
        
        orig_path = row['original_audio_path']
        xtts_path = row['converted_audio_path']
        
        # Relative to project root for hifigan
        hifi_rel_path = hifi_gen_map.get(base_id)
        if hifi_rel_path is not None:
            hifi_path = os.path.join(PROJECT_ROOT, hifi_rel_path.replace('/', '\\'))
        else:
            hifi_path = None
            
        # ------------------ Cohort A: Original Speech ------------------
        info_orig = get_audio_info(orig_path)
        mos_orig, frames_orig, _ = compute_mosnet_prediction(model, orig_path)
        condition_orig = f"Original {src_lang}"
        results_orig.append({
            'file_id': base_id,
            'speaker_id': speaker_id,
            'language': src_lang,
            'condition': condition_orig,
            'direction': direction,
            'diagnosis': diag,
            'duration_sec': round(info_orig['duration'], 4) if info_orig['duration'] else None,
            'mosnet_predicted_mos': round(mos_orig, 4),
            'status': 'OK' if np.isfinite(mos_orig) else 'NON_FINITE',
            'source_path': orig_path
        })
        prep_logs.append({
            'file_id': base_id,
            'cohort': 'Original',
            'condition': condition_orig,
            'input_path': orig_path,
            'original_sample_rate': info_orig['orig_sr'],
            'original_channels': info_orig['channels'],
            'duration_sec': round(info_orig['duration'], 4) if info_orig['duration'] else None,
            'conversion_performed': 'Resampled to 16000 Hz mono in memory for STFT' if info_orig['orig_sr'] != 16000 or info_orig['channels'] != 1 else 'None (native 16000 Hz mono)',
            'final_sample_rate': 16000,
            'final_channels': 1,
            'status': 'OK'
        })
        
        # ------------------ Cohort B: XTTS Generated Speech ------------------
        info_xtts = get_audio_info(xtts_path)
        mos_xtts, frames_xtts, _ = compute_mosnet_prediction(model, xtts_path)
        xtts_cond_name = f"XTTS {'ES->DE' if direction == 'SP_to_GE' else 'DE->ES'}"
        results_xtts.append({
            'file_id': base_id,
            'speaker_id': speaker_id,
            'language': tgt_lang,
            'condition': xtts_cond_name,
            'direction': direction,
            'diagnosis': diag,
            'duration_sec': round(info_xtts['duration'], 4) if info_xtts['duration'] else None,
            'mosnet_predicted_mos': round(mos_xtts, 4),
            'status': 'OK' if np.isfinite(mos_xtts) else 'NON_FINITE',
            'source_path': xtts_path
        })
        prep_logs.append({
            'file_id': base_id,
            'cohort': 'XTTS',
            'condition': xtts_cond_name,
            'input_path': xtts_path,
            'original_sample_rate': info_xtts['orig_sr'],
            'original_channels': info_xtts['channels'],
            'duration_sec': round(info_xtts['duration'], 4) if info_xtts['duration'] else None,
            'conversion_performed': 'Resampled to 16000 Hz mono in memory for STFT' if info_xtts['orig_sr'] != 16000 or info_xtts['channels'] != 1 else 'None (native 16000 Hz mono)',
            'final_sample_rate': 16000,
            'final_channels': 1,
            'status': 'OK'
        })
        
        # ------------------ Cohort C: HiFi-GAN Reconstructed Speech ------------------
        if hifi_path and os.path.exists(hifi_path):
            info_hifi = get_audio_info(hifi_path)
            mos_hifi, frames_hifi, _ = compute_mosnet_prediction(model, hifi_path)
            hifi_cond_name = f"HiFi-GAN reconstructed {src_lang}"
            results_hifi.append({
                'file_id': base_id,
                'speaker_id': speaker_id,
                'language': src_lang,
                'condition': hifi_cond_name,
                'direction': f"{src_lang}_to_{src_lang}",
                'diagnosis': diag,
                'duration_sec': round(info_hifi['duration'], 4) if info_hifi['duration'] else None,
                'mosnet_predicted_mos': round(mos_hifi, 4),
                'status': 'OK' if np.isfinite(mos_hifi) else 'NON_FINITE',
                'source_path': hifi_path
            })
            prep_logs.append({
                'file_id': base_id,
                'cohort': 'HiFi-GAN',
                'condition': hifi_cond_name,
                'input_path': hifi_path,
                'original_sample_rate': info_hifi['orig_sr'],
                'original_channels': info_hifi['channels'],
                'duration_sec': round(info_hifi['duration'], 4) if info_hifi['duration'] else None,
                'conversion_performed': 'Resampled to 16000 Hz mono in memory for STFT' if info_hifi['orig_sr'] != 16000 or info_hifi['channels'] != 1 else 'None (native 16000 Hz mono)',
                'final_sample_rate': 16000,
                'final_channels': 1,
                'status': 'OK'
            })
            
    # Save per-cohort CSVs
    print(f"[4/6] Exporting per-cohort CSVs and preprocessing audit...")
    df_out_orig = pd.DataFrame(results_orig)
    df_out_xtts = pd.DataFrame(results_xtts)
    df_out_hifi = pd.DataFrame(results_hifi)
    df_out_logs = pd.DataFrame(prep_logs)
    
    path_orig_csv = os.path.join(OUT_ORIG_DIR, "MOSNET_ORIGINAL_N276.csv")
    path_xtts_csv = os.path.join(OUT_XTTS_DIR, "MOSNET_XTTS_N276.csv")
    path_hifi_csv = os.path.join(OUT_HIFI_DIR, "MOSNET_HIFIGAN_N276.csv")
    path_logs_csv = os.path.join(OUT_AUDIT_DIR, "MOSNET_AUDIO_PREPROCESSING_LOG.csv")
    
    df_out_orig.to_csv(path_orig_csv, index=False)
    df_out_xtts.to_csv(path_xtts_csv, index=False)
    df_out_hifi.to_csv(path_hifi_csv, index=False)
    df_out_logs.to_csv(path_logs_csv, index=False)
    
    print(f"      Saved {path_orig_csv} (N={len(df_out_orig)})")
    print(f"      Saved {path_xtts_csv} (N={len(df_out_xtts)})")
    print(f"      Saved {path_hifi_csv} (N={len(df_out_hifi)})")
    print(f"      Saved {path_logs_csv} (N={len(df_out_logs)})")
    
    # -------------------------------------------------------------------------
    # 6. Condition Summary Table
    # -------------------------------------------------------------------------
    print(f"[5/6] Computing condition summary statistics and paired comparisons...")
    summary_rows = []
    
    all_evaluated = pd.concat([df_out_orig, df_out_xtts, df_out_hifi], ignore_index=True)
    
    # Conditions to summarize
    conditions_order = [
        ("Original Spanish", "SP_to_GE"),
        ("Original German", "GE_to_SP"),
        ("XTTS ES->DE", "SP_to_GE"),
        ("XTTS DE->ES", "GE_to_SP"),
        ("HiFi-GAN reconstructed Spanish", "Spanish_to_Spanish"),
        ("HiFi-GAN reconstructed German", "German_to_German"),
        ("Overall Original", "All"),
        ("Overall XTTS", "All"),
        ("Overall HiFi-GAN", "All")
    ]
    
    for cond, dir_name in conditions_order:
        if cond == "Overall Original":
            subset = df_out_orig['mosnet_predicted_mos']
        elif cond == "Overall XTTS":
            subset = df_out_xtts['mosnet_predicted_mos']
        elif cond == "Overall HiFi-GAN":
            subset = df_out_hifi['mosnet_predicted_mos']
        else:
            subset = all_evaluated[all_evaluated['condition'] == cond]['mosnet_predicted_mos']
            
        n = len(subset)
        if n == 0:
            continue
        mean_v = float(subset.mean())
        std_v = float(subset.std(ddof=1))
        median_v = float(subset.median())
        q1_v = float(subset.quantile(0.25))
        q3_v = float(subset.quantile(0.75))
        min_v = float(subset.min())
        max_v = float(subset.max())
        
        summary_rows.append({
            'condition': cond,
            'direction': dir_name,
            'n': n,
            'mean_mos': round(mean_v, 4),
            'std_mos': round(std_v, 4),
            'median_mos': round(median_v, 4),
            'q1': round(q1_v, 4),
            'q3': round(q3_v, 4),
            'min': round(min_v, 4),
            'max': round(max_v, 4)
        })
        
    df_summary = pd.DataFrame(summary_rows)
    path_summary_csv = os.path.join(OUT_TABLES_DIR, "MOSNET_SUMMARY_BY_CONDITION.csv")
    df_summary.to_csv(path_summary_csv, index=False)
    print(f"      Saved {path_summary_csv}")
    print("\nCondition Summary Table:")
    print(df_summary.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 7. Paired Comparisons and Wilcoxon Signed-Rank Statistics
    # -------------------------------------------------------------------------
    # Merge on base_id to ensure exact 1-to-1 pairing
    df_paired_full = pd.merge(
        df_out_orig[['file_id', 'language', 'direction', 'diagnosis', 'mosnet_predicted_mos']].rename(
            columns={'mosnet_predicted_mos': 'mos_orig'}
        ),
        df_out_xtts[['file_id', 'mosnet_predicted_mos']].rename(
            columns={'mosnet_predicted_mos': 'mos_xtts'}
        ),
        on='file_id'
    )
    df_paired_full = pd.merge(
        df_paired_full,
        df_out_hifi[['file_id', 'mosnet_predicted_mos']].rename(
            columns={'mosnet_predicted_mos': 'mos_hifi'}
        ),
        on='file_id'
    )
    
    paired_comparisons = [
        ("Spanish Original vs XTTS ES->DE", df_paired_full[df_paired_full['language'] == 'Spanish'], 'mos_orig', 'mos_xtts'),
        ("German Original vs XTTS DE->ES", df_paired_full[df_paired_full['language'] == 'German'], 'mos_orig', 'mos_xtts'),
        ("Overall Original vs XTTS", df_paired_full, 'mos_orig', 'mos_xtts'),
        ("Spanish Original vs HiFi-GAN Spanish", df_paired_full[df_paired_full['language'] == 'Spanish'], 'mos_orig', 'mos_hifi'),
        ("German Original vs HiFi-GAN German", df_paired_full[df_paired_full['language'] == 'German'], 'mos_orig', 'mos_hifi'),
        ("Overall Original vs HiFi-GAN Reconstructed", df_paired_full, 'mos_orig', 'mos_hifi')
    ]
    
    paired_comp_rows = []
    paired_stat_rows = []
    
    for comp_name, df_sub, col_orig, col_gen in paired_comparisons:
        n = len(df_sub)
        v_orig = df_sub[col_orig].values
        v_gen = df_sub[col_gen].values
        
        # Delta = Generated - Original
        delta = v_gen - v_orig
        abs_delta = np.abs(delta)
        
        mean_orig = float(np.mean(v_orig))
        mean_gen = float(np.mean(v_gen))
        mean_d = float(np.mean(delta))
        median_d = float(np.median(delta))
        mean_abs_d = float(np.mean(abs_delta))
        std_d = float(np.std(delta, ddof=1))
        
        paired_comp_rows.append({
            'comparison': comp_name,
            'n': n,
            'mean_original': round(mean_orig, 4),
            'mean_generated': round(mean_gen, 4),
            'mean_delta': round(mean_d, 4),
            'median_delta': round(median_d, 4),
            'mean_absolute_delta': round(mean_abs_d, 4),
            'std_delta': round(std_d, 4)
        })
        
        # Wilcoxon signed-rank test
        # Check if differences are all zero
        if np.all(delta == 0):
            stat_val = 0.0
            p_val = 1.0
            interp = "Identical distributions (all deltas zero)"
        else:
            w_res = stats.wilcoxon(v_orig, v_gen, alternative='two-sided')
            stat_val = float(w_res.statistic)
            p_val = float(w_res.pvalue)
            
            if p_val < 0.001:
                sig_text = "p < 0.001 (statistically significant difference)"
            elif p_val < 0.05:
                sig_text = f"p = {p_val:.4f} (statistically significant difference)"
            else:
                sig_text = f"p = {p_val:.4f} (no statistically significant difference detected; does not imply equivalence)"
            
            direction_note = "Generated higher" if mean_d > 0 else "Original higher"
            interp = f"{direction_note}, {sig_text}"
            
        paired_stat_rows.append({
            'comparison': comp_name,
            'n': n,
            'statistic': round(stat_val, 4),
            'p_value': f"{p_val:.4e}" if p_val < 0.0001 else round(p_val, 4),
            'interpretation': interp
        })
        
    df_paired_comp = pd.DataFrame(paired_comp_rows)
    df_paired_stat = pd.DataFrame(paired_stat_rows)
    
    path_paired_comp = os.path.join(OUT_TABLES_DIR, "MOSNET_PAIRED_COMPARISON.csv")
    path_paired_stat = os.path.join(OUT_TABLES_DIR, "MOSNET_PAIRED_STATISTICS.csv")
    
    df_paired_comp.to_csv(path_paired_comp, index=False)
    df_paired_stat.to_csv(path_paired_stat, index=False)
    
    print(f"      Saved {path_paired_comp}")
    print(f"      Saved {path_paired_stat}")
    print("\nPaired Comparison Table:")
    print(df_paired_comp.to_string(index=False))
    print("\nPaired Statistics Table:")
    print(df_paired_stat.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 8. Publication-Quality Figures (300 DPI)
    # -------------------------------------------------------------------------
    print(f"[6/6] Generating 300 DPI thesis figures...")
    
    # Styling configuration
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    
    # Figure 1: Condition Comparison Boxplot/Violin
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    conditions = [
        "Original Spanish", "Original German",
        "XTTS ES->DE", "XTTS DE->ES",
        "HiFi-GAN reconstructed Spanish", "HiFi-GAN reconstructed German"
    ]
    labels = [
        "Original\nSpanish\n(N=100)", "Original\nGerman\n(N=176)",
        "XTTS\nES→DE\n(N=100)", "XTTS\nDE→ES\n(N=176)",
        "HiFi-GAN\nSpanish\n(N=100)", "HiFi-GAN\nGerman\n(N=176)"
    ]
    data_groups = [all_evaluated[all_evaluated['condition'] == c]['mosnet_predicted_mos'].values for c in conditions]
    
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a']
    
    bp1 = ax1.boxplot(
        data_groups,
        patch_artist=True,
        labels=labels,
        widths=0.55,
        medianprops=dict(color='black', linewidth=1.8),
        whiskerprops=dict(color='#333333', linewidth=1.2),
        capprops=dict(color='#333333', linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, alpha=0.5, markerfacecolor='gray', markeredgecolor='none')
    )
    for patch, col in zip(bp1['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
        patch.set_edgecolor('#222222')
        
    # Overlay means as red diamond
    means1 = [np.mean(g) for g in data_groups]
    ax1.plot(range(1, len(data_groups) + 1), means1, 'rD', markersize=6, label='Mean', zorder=5)
    
    ax1.set_ylabel("Predicted MOS (MOSNet)", fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title("Distribution of Automatic MOSNet-Predicted MOS Across Audio Conditions", fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.set_ylim(1.0, 5.0)
    ax1.tick_params(axis='both', which='major', labelsize=10.5)
    ax1.legend(loc='lower right', frameon=True, fontsize=11)
    
    fig1_path = os.path.join(OUT_FIGS_DIR, "F_MOSNET_CONDITION_COMPARISON.png")
    fig1.tight_layout()
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"      Saved {fig1_path}")
    
    # Figure 2: Paired Original vs XTTS
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    
    # Spanish Original vs XTTS ES->DE
    sp_orig = df_paired_full[df_paired_full['language'] == 'Spanish']['mos_orig'].values
    sp_xtts = df_paired_full[df_paired_full['language'] == 'Spanish']['mos_xtts'].values
    
    bp2a = ax2a.boxplot(
        [sp_orig, sp_xtts],
        patch_artist=True,
        labels=["Original Spanish\n(Reference)", "XTTS ES→DE\n(Generated)"],
        widths=0.5,
        medianprops=dict(color='black', linewidth=1.8)
    )
    bp2a['boxes'][0].set_facecolor('#1f77b4')
    bp2a['boxes'][1].set_facecolor('#ff7f0e')
    for b in bp2a['boxes']:
        b.set_alpha(0.85)
        
    ax2a.plot([1, 2], [np.mean(sp_orig), np.mean(sp_xtts)], 'rD', markersize=6, label='Mean')
    # Connect paired samples with thin lines
    for o, g in zip(sp_orig, sp_xtts):
        ax2a.plot([1, 2], [o, g], color='gray', alpha=0.15, linewidth=0.8)
        
    ax2a.set_ylabel("Predicted MOS (MOSNet)", fontsize=12, fontweight='bold')
    ax2a.set_title(f"Spanish Cohort (N={len(sp_orig)})\nWilcoxon p = {df_paired_stat.iloc[0]['p_value']}", fontsize=12, fontweight='bold')
    ax2a.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2a.set_ylim(1.0, 5.0)
    ax2a.legend(loc='lower right', frameon=True)
    
    # German Original vs XTTS DE->ES
    ge_orig = df_paired_full[df_paired_full['language'] == 'German']['mos_orig'].values
    ge_xtts = df_paired_full[df_paired_full['language'] == 'German']['mos_xtts'].values
    
    bp2b = ax2b.boxplot(
        [ge_orig, ge_xtts],
        patch_artist=True,
        labels=["Original German\n(Reference)", "XTTS DE→ES\n(Generated)"],
        widths=0.5,
        medianprops=dict(color='black', linewidth=1.8)
    )
    bp2b['boxes'][0].set_facecolor('#aec7e8')
    bp2b['boxes'][1].set_facecolor('#ffbb78')
    for b in bp2b['boxes']:
        b.set_alpha(0.85)
        
    ax2b.plot([1, 2], [np.mean(ge_orig), np.mean(ge_xtts)], 'rD', markersize=6, label='Mean')
    for o, g in zip(ge_orig, ge_xtts):
        ax2b.plot([1, 2], [o, g], color='gray', alpha=0.12, linewidth=0.8)
        
    ax2b.set_title(f"German Cohort (N={len(ge_orig)})\nWilcoxon p = {df_paired_stat.iloc[1]['p_value']}", fontsize=12, fontweight='bold')
    ax2b.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2b.set_ylim(1.0, 5.0)
    ax2b.legend(loc='lower right', frameon=True)
    
    fig2.suptitle("Paired Comparison: Original Clinical Speech vs XTTS-v2 Cross-Language Speech", fontsize=13, fontweight='bold', y=0.98)
    fig2_path = os.path.join(OUT_FIGS_DIR, "F_MOSNET_PAIRED_ORIGINAL_VS_XTTS.png")
    fig2.tight_layout()
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"      Saved {fig2_path}")
    
    # Figure 3: Original vs HiFi-GAN Reconstructed
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    
    sp_hifi = df_paired_full[df_paired_full['language'] == 'Spanish']['mos_hifi'].values
    bp3a = ax3a.boxplot(
        [sp_orig, sp_hifi],
        patch_artist=True,
        labels=["Original Spanish\n(Reference)", "HiFi-GAN Spanish\n(Reconstructed)"],
        widths=0.5,
        medianprops=dict(color='black', linewidth=1.8)
    )
    bp3a['boxes'][0].set_facecolor('#1f77b4')
    bp3a['boxes'][1].set_facecolor('#2ca02c')
    for b in bp3a['boxes']:
        b.set_alpha(0.85)
        
    ax3a.plot([1, 2], [np.mean(sp_orig), np.mean(sp_hifi)], 'rD', markersize=6, label='Mean')
    for o, h in zip(sp_orig, sp_hifi):
        ax3a.plot([1, 2], [o, h], color='gray', alpha=0.15, linewidth=0.8)
    ax3a.set_ylabel("Predicted MOS (MOSNet)", fontsize=12, fontweight='bold')
    ax3a.set_title(f"Spanish Same-Content (N={len(sp_orig)})\nWilcoxon p = {df_paired_stat.iloc[3]['p_value']}", fontsize=12, fontweight='bold')
    ax3a.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax3a.set_ylim(1.0, 5.0)
    ax3a.legend(loc='lower right', frameon=True)
    
    ge_hifi = df_paired_full[df_paired_full['language'] == 'German']['mos_hifi'].values
    bp3b = ax3b.boxplot(
        [ge_orig, ge_hifi],
        patch_artist=True,
        labels=["Original German\n(Reference)", "HiFi-GAN German\n(Reconstructed)"],
        widths=0.5,
        medianprops=dict(color='black', linewidth=1.8)
    )
    bp3b['boxes'][0].set_facecolor('#aec7e8')
    bp3b['boxes'][1].set_facecolor('#98df8a')
    for b in bp3b['boxes']:
        b.set_alpha(0.85)
        
    ax3b.plot([1, 2], [np.mean(ge_orig), np.mean(ge_hifi)], 'rD', markersize=6, label='Mean')
    for o, h in zip(ge_orig, ge_hifi):
        ax3b.plot([1, 2], [o, h], color='gray', alpha=0.12, linewidth=0.8)
    ax3b.set_title(f"German Same-Content (N={len(ge_orig)})\nWilcoxon p = {df_paired_stat.iloc[4]['p_value']}", fontsize=12, fontweight='bold')
    ax3b.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax3b.set_ylim(1.0, 5.0)
    ax3b.legend(loc='lower right', frameon=True)
    
    fig3.suptitle("Same-Content Benchmark: Original Clinical Speech vs HiFi-GAN Vocoder Reconstruction", fontsize=13, fontweight='bold', y=0.98)
    fig3_path = os.path.join(OUT_FIGS_DIR, "F_MOSNET_HIFIGAN_COMPARISON.png")
    fig3.tight_layout()
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"      Saved {fig3_path}")
    
    print("\n" + "=" * 80)
    print("MOSNet Automatic MOS Pipeline Finished Successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()
