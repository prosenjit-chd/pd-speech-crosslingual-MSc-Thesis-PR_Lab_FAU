# Spanish → German Final Refinement: Listening Evaluation Package

This package provides direct 3-way auditory and quantitative comparisons for all 6 Spanish subjects:
1. **Original Spanish Source Audio** (ground truth acoustic reference, PC-GITA reading passage)
2. **Config C Baseline German Synthesis** (initial sentence-level segmentation baseline, mean WER ~0.72)
3. **Final Optimized German Synthesis** (optimized segmentation and decoding controls)

---

## Target German Linguistic Content
> **Text:**
> *"Gestern ging ich zum Arzt. „Was fehlt Ihnen?“, fragte er mich. Ich sagte: „Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!“ „Haben Sie einen gebrochenen Nagel?“ „Ja.“ „Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab.“"*

---

## 3-Way Subject Audio & Metric Overview
| safe_id   | diagnosis   | source_audio                      | config_c_audio                   | final_audio                             |   WER_before |   WER_after | WER_change_pct   |   Sim_before |   Sim_after |
|:----------|:------------|:----------------------------------|:---------------------------------|:----------------------------------------|-------------:|------------:|:-----------------|-------------:|------------:|
| SP_HC_001 | HC          | SP_HC_001_01_original_Spanish.wav | SP_HC_001_02_config_c_German.wav | SP_HC_001_03_final_optimized_German.wav |       0.8043 |      0.8913 | -10.8%           |       0.7134 |      0.6975 |
| SP_HC_002 | HC          | SP_HC_002_01_original_Spanish.wav | SP_HC_002_02_config_c_German.wav | SP_HC_002_03_final_optimized_German.wav |       0.5217 |      0.8261 | -58.4%           |       0.6671 |      0.658  |
| SP_HC_003 | HC          | SP_HC_003_01_original_Spanish.wav | SP_HC_003_02_config_c_German.wav | SP_HC_003_03_final_optimized_German.wav |       0.8913 |      0.6957 | +21.9%           |       0.6244 |      0.6466 |
| SP_PD_001 | PD          | SP_PD_001_01_original_Spanish.wav | SP_PD_001_02_config_c_German.wav | SP_PD_001_03_final_optimized_German.wav |       0.6087 |      0.3261 | +46.4%           |       0.6921 |      0.6671 |
| SP_PD_002 | PD          | SP_PD_002_01_original_Spanish.wav | SP_PD_002_02_config_c_German.wav | SP_PD_002_03_final_optimized_German.wav |       0.9348 |      0.1739 | +81.4%           |       0.5691 |      0.5864 |
| SP_PD_003 | PD          | SP_PD_003_01_original_Spanish.wav | SP_PD_003_02_config_c_German.wav | SP_PD_003_03_final_optimized_German.wav |       0.5652 |      0.1522 | +73.1%           |       0.5677 |      0.4912 |

---

## Listening Review Guidelines
When auditioning the audio pairs in `audio/`:
1. **Content Fidelity & Phonetic Clarity:**
   - Listen to `SP_PD_003_02_config_c_German.wav` vs `SP_PD_003_03_final_optimized_German.wav`.
   - Verify that dialogue turns ("Ach Herr Doktor...", "Haben Sie einen gebrochenen Nagel? Ja.") are pronounced naturally without repetition loops, stuttering, or tail babbles.
2. **Target Language Phonology:**
   - Verify native German phonology and prosody (vowel purity, glottal stops, German cadence).
3. **Speaker Identity & Pathology Acoustic Retention:**
   - Compare with `SP_PD_003_01_original_Spanish.wav` to evaluate whether the timbre, pitch register, and pathological vocal qualities (e.g. vocal tremor, roughness, hypophonia) carry through into the German output.
