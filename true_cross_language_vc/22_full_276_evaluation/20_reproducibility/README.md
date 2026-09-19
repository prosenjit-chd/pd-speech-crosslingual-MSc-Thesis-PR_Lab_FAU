# Reproducibility Guide: Autonomous Full 276-File True Cross-Language Evaluation

This directory contains complete reproducibility metadata for the full 276-file evaluation:
- `environment.txt`: Software versions, CUDA configuration, hardware platform.
- `execution_manifest.csv`: Step-by-step pipeline execution ledger.
- `checksums.csv`: SHA-256 hashes for all primary evaluation tables.
- `failure_log.csv`: Technical retry and failure records (0 unrecoverable failures).
- `frozen_configuration.md`: Exact immutable hyperparameters for both conversion directions.

To reproduce the evaluation from scratch:
```powershell
python scripts/00_preflight_and_inventory.py
python scripts/01_cohort_provenance_tagging.py
python scripts/02_prepare_texts_and_references.py
python scripts/03_synthesize_276.py
python scripts/04_audio_qc_lid_asr_spk_276.py
python scripts/05_pathology_and_normalization_276.py
python scripts/06_pd_classification_and_statistics_276.py
python scripts/07_subgroup_and_error_analyses_276.py
python scripts/08_figures_tables_listening_repro_276.py
python scripts/09_compile_full_276_report.py
```
