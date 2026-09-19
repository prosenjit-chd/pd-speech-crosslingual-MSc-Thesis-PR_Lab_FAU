# Reproducibility and Computational Environment Summary

**Project:** Voice Conversion for Crosslingual Detection of Parkinson's Disease  
**Student:** Prosenjit Chowdhury  
**Supervisor:** Dr.-Ing. Tomás Arias-Vergara  
**FAU Pattern Recognition Lab (PRL)**  

---

## 1. System and Platform Environment
- **Operating System:** Windows 11 (Version: 10.0.22631)
- **Architecture:** AMD64 (64bit)
- **Python Interpreter:** `C:\pd-speech-crosslingual\.venv\Scripts\python.exe`
- **Python Version:** 3.12.7
- **PyTorch Version:** 2.12.0+cpu (CUDA Available: False)
- **Torchaudio Version:** 2.11.0+cpu
- **Librosa Version:** 0.11.0
- **SoundFile Version:** 0.13.1
- **Scikit-Learn Version:** 1.8.0
- **Scipy Version:** 1.17.1
- **Pandas Version:** 3.0.3
- **NumPy Version:** 2.4.5

---

## 2. Seed and Determinism Configuration
- **Global Random Seed:** `42` (Fixed across all cross-validation folds, feature scaling, and statistical bootstrap iterations)
- **Deterministic Decoding:** Greedy search (`do_sample=False`, `num_beams=1`) for Whisper-base ASR transcription.

---

## 3. Pretrained Model Identifiers
- **Speaker Verification Encoder:** `speechbrain/spkrec-ecapa-voxceleb` (192-dimensional ECAPA-TDNN embeddings)
- **Automatic Speech Recognition:** `openai/whisper-base` (74 Million parameters multilingual Transformer seq2seq)
- **Self-Supervised Learning Encoders:**
  * `facebook/wav2vec2-base-960h`
  * `facebook/wav2vec2-large-xlsr-53`
  * `microsoft/wavlm-base-plus`

---

## 4. End-to-End Execution Pipeline (Script Order)
1. `scripts/00_validate_existing_inputs.py` — Validates existing read-only inputs across Stage 5C and baselines
2. `scripts/01_build_safe_mapping.py` — Constructs 276 matched original $\leftrightarrow$ converted audio mapping
3. `scripts/02_extract_original_acoustic_features.py` — Extracts 43 interpretable acoustic features for 276 original WAVs
4. `scripts/03_extract_converted_acoustic_features.py` — Extracts 43 features for 276 converted WAVs + feature dictionary
5. `scripts/04_analyze_language_features.py` — Evaluates Question A (cross-language gap, SSL separability, ANOVA interactions)
6. `scripts/05_analyze_vc_domain_shift.py` — Evaluates Question B (feature movement, frozen language classifier shift)
7. `scripts/06_analyze_pathology_preservation.py` — Evaluates Question C (pathology effect size retention, centroid shift)
8. `scripts/09_evaluate_speaker_similarity.py` — Evaluates ECAPA-TDNN speaker cosine similarity vs. negative pairs
9. `scripts/11_evaluate_spectral_distortion.py` — Evaluates Mel-Cepstral Distortion (MCD-DTW), LSD, and Spectral Convergence
10. `scripts/12_evaluate_asr.py` — Transcribes 552 files with Whisper-base and computes consistency proxies
11. `scripts/13_synthesize_question_D.py` — Evaluates Question D master 3-way classification synthesis
12. `scripts/14_run_statistical_master_plan.py` — Runs 271 statistical tests with family-wise Benjamini-Hochberg FDR
13. `scripts/15_generate_publication_figures.py` — Renders all 12 publication-quality figures (PNG 300 DPI + vector PDF)
14. `scripts/16_generate_master_tables.py` — Generates Tables A through L and Research Question Answer Matrix
15. `scripts/17_generate_final_report.py` — Generates Sections 1–9 final evaluation report and thesis draft revision instructions
16. `scripts/18_build_reproducibility_and_deliverable.py` — Assembles final integrity check and deliverable package

---
> [!NOTE]
> All artifacts are 100% deterministic and reproducible by executing scripts 00 through 18 sequentially.
