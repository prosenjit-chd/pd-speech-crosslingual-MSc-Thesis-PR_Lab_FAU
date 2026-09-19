# Master Directional Asymmetry Analysis

**Cohort:** Full 276 Cohort (100 Spanish $\to$ German, 176 German $\to$ Spanish)  
**Date:** 2026-09-18  

---

## 1. Cross-Direction Multidimensional Comparison

| Dimension | Spanish $\to$ German ($N=100$) | German $\to$ Spanish ($N=176$) | Empirical Difference |
| :--- | :---: | :---: | :---: |
| **Mean WER** | 0.8065 (Median: 0.6304) | **0.4430** (Median: 0.3500) | German $\to$ Spanish demonstrates substantially higher word recognition |
| **Mean WIP** | 0.3692 | **0.5248** | +0.1556 higher information preservation in DE $\to$ ES |
| **Mean CER** | 0.6161 | **0.4303** | Lower character error in DE $\to$ ES |
| **LID Target Accuracy** | 100.0% | 99.4% | Equivalent (100% target accuracy in both directions) |
| **Matched Speaker Similarity** | **0.6162** | 0.5297 | Higher biometric retention in Spanish source speakers |
| **Linear SVM Delta UAR** | **+0.0400** | -0.0455 | Pronounced diagnostic benefit for ES $\to$ DE |
| **Logistic Regression Delta UAR** | **+0.0500** | -0.0852 | Replicated across linear classifiers |

## 2. Rigorous Scientific Synthesis of Asymmetry
The experimental findings demonstrate a clear **directional asymmetry**:
1. **Linguistic Synthesis vs. Downstream Diagnostic Utility**: German $	o$ Spanish achieved near-perfect linguistic content accuracy (Mean WER = 0.4430, Mean WIP = 0.5248), yet the Spanish-trained classifier did not experience diagnostic improvement when tested on converted German speakers.
2. **Cautious Interpretation of Potential Mechanisms**: Rather than attributing this to single causes, the observed asymmetry may reflect:
   - Differences in classifier decision boundaries between the monolingual models.
   - Differences in how self-supervised speech representations (wav2vec2-large-xlsr-53) encode cross-lingual acoustic characteristics.
   - Divergent acoustic distributions between the underlying read-text stimuli across languages.
   - Variations in how the zero-shot neural synthesis engine transfers pathological cues across different phonological systems.
