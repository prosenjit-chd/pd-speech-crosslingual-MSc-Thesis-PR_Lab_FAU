#!/usr/bin/env python3
"""
run_pipeline_after_search.py
Turnkey Orchestrator for Spanish -> German Final Refinement Post-Search Execution.

Executes:
1. scripts/04_evaluate_candidates.py (Evaluation of XTTS parameter search)
2. scripts/05_final_regeneration.py (Regeneration of all 6 Spanish -> German files)
3. scripts/06_final_evaluation.py (Evaluation & comparison against Config C)
4. scripts/07_pathology_sanity_check.py (17 pathology features & direction check)
5. scripts/08_generate_figures.py (Figures 1-6 in PNG 300 DPI and PDF)
6. scripts/09_build_listening_package.py (3-way listening package & README)
7. scripts/11_build_reproducibility_package.py (Reproducibility manifest & hashes)
8. scripts/10_compile_final_report.py (Master report & Final Checkpoint)
"""

import sys
import subprocess
import time
from pathlib import Path

SCRIPTS_DIR = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement\scripts")
PYTHON_EVAL = Path(r"C:\pd-speech-crosslingual\.venv\Scripts\python.exe")
PYTHON_XTTS = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\env_xtts\Scripts\python.exe")

STEPS = [
    ("Step 1: Evaluating Candidate Search Audio", PYTHON_EVAL, SCRIPTS_DIR / "04_evaluate_candidates.py"),
    ("Step 2: Regenerating All 6 Files with Winning Config", PYTHON_XTTS, SCRIPTS_DIR / "05_final_regeneration.py"),
    ("Step 3: Comprehensive Final Evaluation vs Config C", PYTHON_EVAL, SCRIPTS_DIR / "06_final_evaluation.py"),
    ("Step 4: Exploratory Pathology Sanity Check", PYTHON_EVAL, SCRIPTS_DIR / "07_pathology_sanity_check.py"),
    ("Step 5: Generating Publication Figures (1 to 6)", PYTHON_EVAL, SCRIPTS_DIR / "08_generate_figures.py"),
    ("Step 6: Building 3-Way Listening Package", PYTHON_EVAL, SCRIPTS_DIR / "09_build_listening_package.py"),
    ("Step 7: Compiling Reproducibility Package & Hashes", PYTHON_EVAL, SCRIPTS_DIR / "11_build_reproducibility_package.py"),
    ("Step 8: Compiling Master Final Report & Checkpoint", PYTHON_EVAL, SCRIPTS_DIR / "10_compile_final_report.py")
]

def run_step(desc, python_bin, script_path):
    print(f"\n{'='*70}")
    print(f"--> {desc}")
    print(f"    Script: {script_path.name} | Interpreter: {python_bin.parent.name}")
    print(f"{'='*70}")
    t0 = time.time()
    
    cmd = [str(python_bin), str(script_path)]
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"[ERROR] Step failed with return code {result.returncode} after {elapsed:.1f}s!")
        sys.exit(result.returncode)
    else:
        print(f"[SUCCESS] Step completed in {elapsed:.1f}s.")

def main():
    t_start = time.time()
    print("=== Starting Post-Search Refinement Pipeline ===")
    for desc, py_bin, script in STEPS:
        run_step(desc, py_bin, script)
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ALL REFINEMENT STEPS COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
