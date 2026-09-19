# Spanish $\to$ German Error and Subgroup Analysis

**Analysis Scope:** All 100 Spanish $\to$ German Conversions (50 HC, 50 PD)  
**Date:** 2026-09-18  

---

## 1. Quartile and Tier Distributions

| Tier | N | Mean WER | Median WER | Mean CER | Mean WIP | Mean Speaker Sim | Mean LID P(German) | HC Count | PD Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Best 10%** | 10 | **0.1457** | 0.1522 | 0.0863 | 0.7826 | 0.5519 | 0.9998 | 5 | 5 |
| **Median 10%** | 10 | **0.6413** | 0.6304 | 0.5545 | 0.3758 | 0.6244 | 0.9996 | 6 | 4 |
| **Worst 10%** | 10 | **2.9652** | 0.9348 | 1.9021 | 0.0519 | 0.6254 | 0.9997 | 7 | 3 |

## 2. Statistical Association Diagnostics
- **Association with Source Audio Duration:** Spearman $\rho = +0.189$ ($p = 0.0593$). Shorter source recordings provide fewer active voiced seconds for reference conditioning, slightly elevating target WER.
- **Association with Speaker Embedding Similarity:** Spearman $\rho = +0.183$ ($p = 0.0687$). High speaker retention does not degrade target pronunciation.
- **Association with Target LID Probability:** Spearman $\rho = -0.344$ ($p = 0.0005$).
- **HC vs PD Discrepancy:** Mann-Whitney $U = 1000.0, p = 0.0852$. Mean HC WER: 1.0474 vs. Mean PD WER: 0.5656.
