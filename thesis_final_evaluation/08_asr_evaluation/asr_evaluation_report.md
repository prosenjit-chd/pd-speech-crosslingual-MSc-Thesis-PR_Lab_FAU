# Phase 7: Open-Source Multilingual ASR & Linguistic Consistency Evaluation Report

## 1. Executive Summary & Ground-Truth Verification
Linguistic content and phonetic intelligibility preservation were evaluated across all 276 original and 276 Stage 5C converted recordings using the open-source multilingual **Whisper-base** automatic speech recognition model (`openai/whisper-base`).

- **Ground-Truth Reference Transcripts Verification**: Exact per-utterance ground-truth text transcripts are **not provided in the raw acoustic read-text corpus**. In strict adherence to scientific integrity guidelines, Word Error Rate was **not fabricated**. Instead, an **Original-vs-Converted ASR Transcript Consistency Proxy ($\text{WER}_{\text{proxy}}$ and $\text{CER}_{\text{proxy}}$)** was calculated, measuring the linguistic stability of the voice conversion system relative to the original acoustic output.
- **Mean Word Consistency Proxy ($\text{WER}_{\text{proxy}}$)**: **0.2178** (Median = **0.1415**, Std = 0.2685, IQR = 0.1934)
- **Mean Character Consistency Proxy ($\text{CER}_{\text{proxy}}$)**: **0.1035** (Median = **0.0456**, Std = 0.2054)
- **Mean Word Information Preserved (WIP)**: **0.6924**
- **PD vs. HC Consistency Comparison**: Parkinson's recordings exhibited a mean $\text{WER}_{\text{proxy}}$ of **0.2630** compared to **0.1727** for Healthy Controls (Welch's $t = 2.8253$, $p = 0.0052$, Cohen's $d = 0.3401$).

---

## 2. Overall ASR Consistency Distribution
| Metric | Unit | Mean | Median | Std | IQR | 5th Percentile | 95th Percentile | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Word Error Rate Proxy (WER Proxy)** | ratio | **0.2178** | **0.1415** | 0.2685 | 0.1934 | 0.0189 | 0.6373 | 0.0000 | 3.3710 |
| **Character Error Rate Proxy (CER Proxy)** | ratio | **0.1035** | **0.0456** | 0.2054 | 0.0805 | 0.0027 | 0.3819 | 0.0000 | 2.6327 |
| **Word Information Lost (WIL Proxy)** | ratio | **0.3076** | **0.2404** | 0.2295 | 0.2817 | 0.0374 | 0.8028 | 0.0000 | 1.0000 |
| **Word Information Preserved (WIP Proxy)** | ratio | **0.6924** | **0.7596** | 0.2295 | 0.2817 | 0.1972 | 0.9626 | 0.0000 | 1.0000 |

---

## 3. Stratified Breakdown by Language and Diagnosis
| Stratum | N Pairs | WER Proxy Mean | WER Proxy Median | CER Proxy Mean | CER Proxy Median | WIP Mean | Exact Match % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Pooled 276)** | 276 | **0.2178** | **0.1415** | **0.1035** | **0.0456** | **0.6924** | **3.26%** |
| **Spanish $\to$ German-domain** | 100 | **0.2588** | **0.1947** | **0.1079** | **0.0643** | **0.6208** | **5.00%** |
| **German $\to$ Spanish-domain** | 176 | **0.1945** | **0.1207** | **0.1010** | **0.0424** | **0.7331** | **2.27%** |
| **Healthy Controls (HC)** | 138 | **0.1727** | **0.1177** | **0.0813** | **0.0388** | **0.7411** | **4.35%** |
| **Parkinson's Disease (PD)** | 138 | **0.2630** | **0.1832** | **0.1257** | **0.0635** | **0.6438** | **2.17%** |
| **Spanish HC** | 50 | **0.1815** | **0.1258** | **0.0644** | **0.0392** | **0.7178** | **10.00%** |
| **Spanish PD** | 50 | **0.3361** | **0.3000** | **0.1513** | **0.1250** | **0.5239** | **0.00%** |
| **German HC** | 88 | **0.1676** | **0.1167** | **0.0909** | **0.0379** | **0.7543** | **1.14%** |
| **German PD** | 88 | **0.2215** | **0.1376** | **0.1111** | **0.0480** | **0.7119** | **3.41%** |

---

## 4. Experimental Setup & Decoding Parameters
- **Model Identifier**: `openai/whisper-base` (74 Million parameters, multilingual seq2seq transformer)
- **Decoding Method**: Deterministic greedy search (`do_sample=False`, `num_beams=1`)
- **Forced Language Tokens**:
  * Spanish recordings: `language="spanish"`, `task="transcribe"`
  * German recordings: `language="german"`, `task="transcribe"`
- **Sampling Rate**: 16,000 Hz resampled via bandlimited sinc interpolation
- **Total Transcribed Utterances**: 552 files (276 original + 276 converted)
- **Runtime**: 1290.80 seconds (4.68 s / audio pair)
- **Transcription Failures**: 0

---
> [!NOTE]
> **Phase 7 Status: COMPLETED (PASSED)**
> ASR evaluation is fully completed across all 276 pairs with verified methodology and zero missing data. Ready to proceed to Phase 8 (Final Question D: Master Classification Synthesis).
