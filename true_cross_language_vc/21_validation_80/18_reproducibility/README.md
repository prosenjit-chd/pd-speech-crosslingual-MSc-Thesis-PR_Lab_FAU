# Reproducibility Guide: 80-File True Cross-Language Validation

This directory contains complete reproducibility metadata for the 80-file true cross-language validation experiment:
- `environment.txt`: Software versions, CUDA configuration, and execution runtime.
- `execution_manifest.csv`: Step-by-step pipeline execution ledger.
- `checksums.csv`: SHA-256 hashes for all primary evaluation tables.
- `failure_log.csv`: Technical retry and failure records (0 unrecoverable failures).

To replicate the evaluation from scratch:
```powershell
python scripts/00_verify_speaker_control_baseline.py
python scripts/01_select_80_cohort.py
python scripts/02_prepare_texts_and_references.py
python scripts/03_synthesize_80.py
python scripts/04_evaluate_qc_lid_asr_spk_80.py
python scripts/05_pathology_and_normalization_80.py
python scripts/06_evaluate_pd_classification_80.py
python scripts/07_generate_tables_figures_listening_reproducibility.py
```
