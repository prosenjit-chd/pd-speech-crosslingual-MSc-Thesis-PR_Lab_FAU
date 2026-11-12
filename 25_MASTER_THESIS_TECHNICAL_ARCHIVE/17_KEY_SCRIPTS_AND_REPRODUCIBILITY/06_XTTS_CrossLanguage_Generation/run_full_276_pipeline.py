"""
Master Pipeline Orchestrator: Autonomous Full 276-File True Cross-Language Evaluation.

Monitors synthesis completion and autonomously executes all post-synthesis stages:
1. Wait for 03_synthesize_276.py to finish (276 files).
2. Execute 04_audio_qc_lid_asr_spk_276.py (QC, Whisper LID, Whisper ASR, ECAPA Speaker Sim).
3. Execute 05_pathology_and_normalization_276.py (19 feature extraction, preservation, centroid normalization).
4. Execute 06_pd_classification_and_statistics_276.py (XLSR Layer 11, Linear SVM, Logistic Regression, FDR).
5. Execute 07_subgroup_and_error_analyses_276.py (ES->DE error tiers, asymmetry, HC vs PD quality).
6. Execute 08_figures_tables_listening_repro_276.py (Figures 01-13, Tables 01-18, Listening package, Repro).
7. Execute 09_compile_full_276_report.py (Master report and Section 35 checkpoint).
"""

import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE = PROJECT_ROOT / "true_cross_language_vc"
EVAL_DIR = WORKSPACE / "22_full_276_evaluation"
SCRIPTS_DIR = EVAL_DIR / "scripts"
SYNTH_DIR = EVAL_DIR / "06_synthesis"

PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

def run_step(step_name, script_path):
    print(f"\n{'='*70}")
    print(f"STARTING STEP: {step_name}")
    print(f"Script: {script_path.name}")
    print(f"{'='*70}")
    t0 = time.time()
    res = subprocess.run([str(PYTHON_EXE), str(script_path)], cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"ERROR: {step_name} failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"COMPLETED STEP: {step_name} in {elapsed:.2f}s")

def main():
    print("=== Master Orchestrator: Full 276 Evaluation Pipeline ===")

    # Step 1: Monitor synthesis
    man_csv = SYNTH_DIR / "synthesis_manifest_276.csv"
    while True:
        wav_files = list(SYNTH_DIR.glob("**/*.wav"))
        valid_wavs = [w for w in wav_files if w.stat().st_size > 1000]
        count = len(valid_wavs)
        if man_csv.exists() and count >= 276:
            print(f"\nSynthesis complete! Found {count}/276 valid audio files and manifest.")
            break
        print(f"[{time.strftime('%H:%M:%S')}] Synthesis progress: {count}/276 files generated. Waiting 30 seconds...")
        time.sleep(30)

    # Step 2: Audio QC, LID, ASR, Speaker Sim
    run_step("Audio QC, LID, ASR, Speaker Sim", SCRIPTS_DIR / "04_audio_qc_lid_asr_spk_276.py")

    # Step 3: Pathology Features, Preservation, Normalization
    run_step("Pathology Features & Normalization", SCRIPTS_DIR / "05_pathology_and_normalization_276.py")

    # Step 4: PD Classification & Statistics
    run_step("PD Classification & Statistics", SCRIPTS_DIR / "06_pd_classification_and_statistics_276.py")

    # Step 5: Subgroup & Error Analyses
    run_step("Subgroup & Asymmetry Analyses", SCRIPTS_DIR / "07_subgroup_and_error_analyses_276.py")

    # Step 6: Figures, Tables, Listening, Repro
    run_step("Figures, Tables, Listening & Repro", SCRIPTS_DIR / "08_figures_tables_listening_repro_276.py")

    # Step 7: Final Master Report & Checkpoint
    run_step("Master Report & Checkpoint", SCRIPTS_DIR / "09_compile_full_276_report.py")

    print("\n=======================================================")
    print("ALL FULL 276 EVALUATION PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    main()
