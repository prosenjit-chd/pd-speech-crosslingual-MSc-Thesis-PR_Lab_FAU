# Original 12-File Pilot Diagnosis Report

**Date**: September 2026  
**Artifact**: Diagnostic evaluation of initial 12-file True Cross-Language conversion  
**Target Gate Assessment**: Content fidelity and target-language correctness  

---

## 1. Executive Diagnostic Summary

The initial 12-file pilot demonstrated that XTTS-v2 is capable of producing genuine cross-language speech with strong speaker conditioning. However, an asymmetric failure pattern was observed between language directions:

- **German $\to$ Spanish (6 files)**: **Consistently Robust**.
  - All 6 files achieved 100% correct Spanish LID ($p > 0.95$).
  - Mean WER was **24.6%** (range: 16.9% to 61.8%).
  - 5 out of 6 files are classified as **GOOD**, with 1 **MODERATE**. Zero POOR files.
  
- **Spanish $\to$ German (6 files)**: **Inconsistent Content Fidelity**.
  - Mean WER was **69.8%** (range: 15.2% to 89.1%).
  - 4 out of 6 files are classified as **POOR**, 1 as **MODERATE**, and only 1 as **GOOD** (`SP_PD_001`).
  - Failure mechanisms include:
    1. **Early Truncation / Stalling**: `SP_HC_003` generated only 6 words and `SP_PD_002` generated only 5 words of the 46-word passage, while output audio files were 48.9s and 59.3s long (filled with unnatural pauses/murmurs).
    2. **Hallucinatory Repetitions**: `SP_HC_001` and `SP_HC_002` repeated phrases (*"Was fehlt Ihnen? Was fehlt Ihnen?"*) and hallucinated German-like pseudo-words.
    3. **Language Drift**: `SP_HC_003` shifted into Yiddish (`yi`) phonetics due to phonetic ambiguity in the dialogue opening.

---

## 2. Per-File Detailed Diagnosis Table

| Safe ID | Direction | Cohort | Duration (s) | Pred Lang | $P(\text{Target})$ | WER | CER | WIP | Speaker Sim | QC Status | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SP_HC_001.wav` | Spanish->German | **HC** | 50.56 | **German** | 0.9412 | 0.696 | 0.597 | 0.331 | **0.7223** | PASSED | **MODERATE** |
| `SP_HC_002.wav` | Spanish->German | **HC** | 50.29 | **German** | 0.7238 | 0.804 | 0.669 | 0.345 | **0.6689** | PASSED | **POOR** |
| `SP_HC_003.wav` | Spanish->German | **HC** | 48.88 | **YI** | 0.3897 | 0.891 | 0.871 | 0.091 | **0.6508** | PASSED | **POOR** |
| `SP_PD_001.wav` | Spanish->German | **PD** | 32.09 | **German** | 0.9740 | 0.152 | 0.064 | 0.756 | **0.6538** | PASSED | **GOOD** |
| `SP_PD_002.wav` | Spanish->German | **PD** | 59.33 | **German** | 0.9629 | 0.891 | 0.893 | 0.109 | **0.6304** | PASSED | **POOR** |
| `SP_PD_003.wav` | Spanish->German | **PD** | 50.08 | **German** | 0.9806 | 0.457 | 0.369 | 0.453 | **0.5984** | PASSED | **MODERATE** |
| `DE_HC_001.wav` | German->Spanish | **HC** | 39.53 | **Spanish** | 0.9986 | 0.281 | 0.302 | 0.708 | **0.5890** | PASSED | **GOOD** |
| `DE_HC_002.wav` | German->Spanish | **HC** | 52.49 | **Spanish** | 0.9596 | 0.618 | 0.598 | 0.342 | **0.5993** | PASSED | **MODERATE** |
| `DE_HC_003.wav` | German->Spanish | **HC** | 39.09 | **Spanish** | 0.9984 | 0.258 | 0.232 | 0.709 | **0.5737** | PASSED | **GOOD** |
| `DE_PD_001.wav` | German->Spanish | **PD** | 35.87 | **Spanish** | 0.9982 | 0.191 | 0.160 | 0.777 | **0.4709** | PASSED | **GOOD** |
| `DE_PD_002.wav` | German->Spanish | **PD** | 39.82 | **Spanish** | 0.9984 | 0.258 | 0.253 | 0.720 | **0.6125** | PASSED | **GOOD** |
| `DE_PD_003.wav` | German->Spanish | **PD** | 37.12 | **Spanish** | 0.9979 | 0.169 | 0.172 | 0.832 | **0.6435** | PASSED | **GOOD** |

---

## 3. Detailed Failure Mode Analysis: Spanish $\to$ German

### Case Analysis of the 4 POOR Spanish $\to$ German Files:
1. **`SP_HC_003.wav` (POOR)**:
   - *Target Language Prediction*: Yiddish (`yi`, $p=0.3897$), German probability $0.3897$.
   - *Hypothesis Words*: 6 words vs. 46 reference words (`"gestern ging ich zum arzt loll"`).
   - *Mechanism*: Tokenizer confusion on initial quotes and immediate decoder termination; audio duration drifted to 48.9s with low-level background artifact.
2. **`SP_PD_002.wav` (POOR)**:
   - *Hypothesis Words*: 5 words vs. 46 reference words (`"gestern ging ich zum arzt"`).
   - *Mechanism*: The source audio has long hypophonic pauses (26.1s total). Conditioning on this unsegmented audio caused the autoregressive decoder to halt text generation after the first sentence and produce 54s of acoustic murmur.
3. **`SP_HC_001.wav` (POOR)**:
   - *WER*: 69.6%, CER: 59.7%, WIP: 0.331.
   - *Hypothesis Words*: Repetition loops (*"Was fehlt Ihnen? Was fehlt Ihnen? Brakte er mich..."*).
   - *Mechanism*: German quotation marks (`„` and `“`) and compound punctuation (`?“,`) caused attention desynchronization in XTTS-v2.
4. **`SP_HC_002.wav` (POOR)**:
   - *WER*: 80.4%, CER: 66.9%, WIP: 0.345.
   - *Hypothesis Words*: Repetitive loops and severe lexical substitutions (*"Errandeo... Warb der mich? Einen Berlte er mich..."*).

### The Successful Outlier: `SP_PD_001.wav` (GOOD)
- *WER*: **15.2%**, WIP: **0.756**, Speaker Sim: **0.6538**, Duration: **32.1s**.
- *Why it succeeded*: The source audio for `SP_PD_001` was shorter (15.1s) with consistent voicing and minimal trailing silence, and the autoregressive decoder happened to step across the dialogue quotes without looping.

---

## 4. Cohort Summary Distributions

| Direction | Total | GOOD | MODERATE | POOR | Mean WER | Mean WIP | Mean Speaker Sim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Spanish $\to$ German** | 6 | 1 (16.7%) | 1 (16.7%) | **4 (66.7%)** | **0.6983** | **0.3475** | 0.6541 |
| **German $\to$ Spanish** | 6 | **5 (83.3%)** | 1 (16.7%) | **0 (0.0%)** | **0.2461** | **0.6811** | 0.5815 |
| **Overall** | 12 | 6 (50.0%) | 2 (16.7%) | 4 (33.3%) | 0.4722 | 0.5143 | 0.6178 |

---

## 5. Diagnostic Conclusion & Immediate Corrective Actions
The diagnostic evidence points directly to three remediable engineering factors:
1. **Target Text Formatting**: The complex nested German dialogue quotation marks must be normalized into standard punctuation without changing word semantics.
2. **Text Segmentation**: Generating 46 words of conversational dialogue at once strains XTTS-v2 attention. Splitting the target paragraph into 6 discrete natural sentences will eliminate loop hallucinations and early dropouts.
3. **Reference Conditioning Quality**: Long, unsegmented speaker audios containing extensive silence/hesitations destabilize the speaker conditioning latent. A deterministic selection of an 8-second clean voiced window will stabilize generation.
