"""
Script 21: Setup Autonomous Refinement Workspace, Take Baseline Snapshot, and Environment Audit.
Creates all required subdirectories under 20_spanish_to_german_final_refinement/,
freezes Config C baseline snapshot for the 6 Spanish->German files, and logs environment status.
"""

from pathlib import Path
import os
import sys
import subprocess
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE_ROOT = PROJECT_ROOT / "true_cross_language_vc"
AUTONOMOUS_DIR = WORKSPACE_ROOT / "20_spanish_to_german_final_refinement"

SUBDIRS = [
    "00_baseline_snapshot",
    "01_environment",
    "02_text_analysis",
    "03_reference_audio_analysis",
    "04_xtts_parameter_search",
    "05_chunking_search",
    "06_secondary_model_test",
    "07_candidate_audio",
    "08_candidate_evaluation",
    "09_model_selection",
    "10_final_regeneration",
    "11_final_qc",
    "12_final_lid",
    "13_final_content_accuracy",
    "14_final_speaker_similarity",
    "15_pathology_sanity_check",
    "16_figures",
    "17_tables",
    "18_listening_package",
    "19_reports",
    "20_reproducibility",
    "logs",
    "scripts"
]

def main():
    print("=== Step 1: Creating 20_spanish_to_german_final_refinement Subdirectories ===")
    for sd in SUBDIRS:
        p = AUTONOMOUS_DIR / sd
        p.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {sd}")

    print("\n=== Step 2: Taking Frozen Baseline Snapshot (Config C) ===")
    ref19_dir = WORKSPACE_ROOT / "19_pilot_refinement"
    qc_csv = ref19_dir / "05_audio_qc" / "refined_audio_qc.csv"
    lid_csv = ref19_dir / "06_language_id" / "refined_language_id_results.csv"
    wer_csv = ref19_dir / "07_content_accuracy" / "refined_wer_results.csv"
    spk_csv = ref19_dir / "08_speaker_similarity" / "refined_speaker_similarity.csv"

    df_qc = pd.read_csv(qc_csv)
    df_lid = pd.read_csv(lid_csv)
    df_wer = pd.read_csv(wer_csv)
    df_spk = pd.read_csv(spk_csv)
    df_spk_matched = df_spk[df_spk["comparison_type"] == "matched_same_speaker"]

    # Filter only the 6 Spanish source files (Spanish -> German)
    es_sids = [f"SP_{diag}_{idx:03d}.wav" for diag in ["HC", "PD"] for idx in [1, 2, 3]]

    snapshot_records = []
    for sid in es_sids:
        q_row = df_qc[df_qc["safe_id"] == sid].iloc[0]
        l_row = df_lid[df_lid["safe_id"] == sid].iloc[0]
        w_row = df_wer[df_wer["safe_id"] == sid].iloc[0]
        s_row = df_spk_matched[df_spk_matched["safe_id"] == sid].iloc[0]

        snapshot_records.append({
            "safe_id": sid,
            "WER": round(float(w_row["wer"]), 4),
            "CER": round(float(w_row["cer"]), 4),
            "WIP": round(float(w_row["wip"]), 4),
            "language": l_row["predicted_code"],
            "P_target": round(float(l_row["target_language_probability"]), 4),
            "speaker_similarity": round(float(s_row["cosine_similarity"]), 4),
            "duration": round(float(q_row["duration"]), 4),
            "clipping": bool(q_row["is_clipped"])
        })

    df_snapshot = pd.DataFrame(snapshot_records)
    snapshot_csv = AUTONOMOUS_DIR / "00_baseline_snapshot" / "current_config_c_results.csv"
    df_snapshot.to_csv(snapshot_csv, index=False)
    print(f"Saved frozen baseline snapshot to {snapshot_csv.relative_to(WORKSPACE_ROOT)}:")
    print(df_snapshot.to_string(index=False))

    print("\n=== Step 3: Performing Environment Audit (Before) ===")
    env_dir = AUTONOMOUS_DIR / "01_environment"
    
    # Audit Python, PyTorch, CUDA, and Libraries
    audit_lines = []
    audit_lines.append(f"Host OS: Windows ({sys.platform})")
    audit_lines.append(f"System Python: {sys.version}")
    audit_lines.append(f"Executable: {sys.executable}")
    
    import torch
    audit_lines.append(f"PyTorch Version: {torch.__version__}")
    audit_lines.append(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        audit_lines.append(f"CUDA Version: {torch.version.cuda}")
        audit_lines.append(f"GPU Device Count: {torch.cuda.device_count()}")
        audit_lines.append(f"GPU 0 Name: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        audit_lines.append(f"GPU 0 VRAM: {mem_gb:.2f} GB")
    else:
        audit_lines.append("CUDA: Not currently active in this Python environment")

    # Check hardware via nvidia-smi
    smi_path = r"C:\Windows\System32\DriverStore\FileRepository\nvcvi.inf_amd64_d9bd6aa375e1eac3\nvidia-smi.exe"
    if os.path.exists(smi_path):
        try:
            res = subprocess.run([smi_path], capture_output=True, text=True)
            audit_lines.append("\n--- Hardware nvidia-smi Output ---")
            audit_lines.append(res.stdout)
        except Exception as e:
            audit_lines.append(f"nvidia-smi execution error: {e}")

    # Check package versions
    packages = ["transformers", "huggingface_hub", "soundfile", "librosa", "jiwer", "scipy", "speechbrain"]
    audit_lines.append("\n--- Package Versions in .venv ---")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            audit_lines.append(f"{pkg}: {ver}")
        except ImportError:
            audit_lines.append(f"{pkg}: NOT INSTALLED")

    # Check env_xtts
    xtts_py = WORKSPACE_ROOT / "env_xtts" / "Scripts" / "python.exe"
    if xtts_py.exists():
        audit_lines.append("\n--- env_xtts Audit ---")
        try:
            res = subprocess.run(
                [str(xtts_py), "-c", "import torch, TTS; print(f'PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}, TTS={TTS.__version__}')"],
                capture_output=True, text=True
            )
            audit_lines.append(res.stdout.strip())
        except Exception as e:
            audit_lines.append(f"env_xtts query error: {e}")

    env_before_file = env_dir / "environment_before.txt"
    with open(env_before_file, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))
    print(f"Saved initial environment audit to {env_before_file.relative_to(WORKSPACE_ROOT)}.")

    # Initialize installation log
    install_log = env_dir / "installation_log.txt"
    with open(install_log, "w", encoding="utf-8") as f:
        f.write("=== Autonomous Environment Installation & Upgrade Log ===\n")
    print(f"Initialized {install_log.relative_to(WORKSPACE_ROOT)}.")

if __name__ == "__main__":
    main()
