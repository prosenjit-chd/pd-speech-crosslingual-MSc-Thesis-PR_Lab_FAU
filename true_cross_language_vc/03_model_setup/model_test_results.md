# Model Acceptance Test Results

## 1. Test Summary
Before synthesizing clinical cohorts, the zero-shot multilingual voice cloning pipeline was tested with reference audio from both language cohorts:

| Test Case | Reference Speaker Source | Condition Language | Target Language | Prompt Text Spoken | Output Duration (s) | Synthesis Time (s) | Acceptance Gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A (ES $\to$ DE)** | Spanish HC (`SP_HC_001`) | Spanish Audio | **German (`de`)** | *"Gestern ging ich zum Arzt. Was fehlt Ihnen, fragte er mich."* | 6.31 | 11.10 | **PASS** |
| **B (DE $\to$ ES)** | German HC (`DE_HC_001`) | German Audio | **Spanish (`es`)** | *"Una enorme invasion de tortugas marinas azota las costas de Mexico."* | 7.38 | 12.68 | **PASS** |

## 2. Critical Sanity Verification
- [x] Output A audibly speaks German phonemes and German words (*Gestern ging ich zum Arzt...*).
- [x] Output B audibly speaks Spanish phonemes and Spanish words (*Una enorme invasión de tortugas...*).
- [x] System does **NOT** retain or regurgitate the source language words.
- [x] Signal characteristics verified: no NaN/Inf, peak amplitude $\le 1.00$, clear articulation.
- **GATE STATUS**: **PASSED**. Approved for 12-file clinical pilot synthesis.
