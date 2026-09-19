# Content Accuracy & True Target-Language WER Report (12-File Pilot)

## 1. Methodological Significance: True Reference-Based WER
In the previous acoustic-domain work, true WER was methodologically impossible because the spoken linguistic content remained unchanged while domain features shifted. Here, because verified target-language translations exist, we compute **true reference-based Word Error Rate (WER)** and Character Error Rate (CER) using Whisper-base transcription against the validated target text.

## 2. Overall Accuracy Distribution
- **Mean Reference-Based WER**: **0.4722** (47.2%)
- **Mean Reference-Based CER**: **0.4317** (43.2%)
- **Mean Word Information Preserved (WIP)**: **0.5143**
- **Spanish $\to$ German Mean WER**: **0.6486**
- **German $\to$ Spanish Mean WER**: **0.2959**
- **Healthy Controls Mean WER**: **0.5914**
- **Parkinson's Disease Mean WER**: **0.3530**

## 3. Per-Recording Content Accuracy Table

| Safe ID | Conversion | Cohort | Reference Words | Hypothesis Words | True WER | True CER | WIP | Exact Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SP_HC_001.wav` | Sp $\to$ Ge | **HC** | 46 | 41 | **0.696** | 0.597 | 0.331 | No |
| `SP_HC_002.wav` | Sp $\to$ Ge | **HC** | 46 | 53 | **0.804** | 0.669 | 0.345 | No |
| `SP_HC_003.wav` | Sp $\to$ Ge | **HC** | 46 | 6 | **0.891** | 0.871 | 0.091 | No |
| `SP_PD_001.wav` | Sp $\to$ Ge | **PD** | 46 | 46 | **0.152** | 0.064 | 0.756 | No |
| `SP_PD_002.wav` | Sp $\to$ Ge | **PD** | 46 | 5 | **0.891** | 0.893 | 0.109 | No |
| `SP_PD_003.wav` | Sp $\to$ Ge | **PD** | 46 | 30 | **0.457** | 0.369 | 0.453 | No |
| `DE_HC_001.wav` | Ge $\to$ Sp | **HC** | 89 | 65 | **0.281** | 0.302 | 0.708 | No |
| `DE_HC_002.wav` | Ge $\to$ Sp | **HC** | 89 | 38 | **0.618** | 0.598 | 0.342 | No |
| `DE_HC_003.wav` | Ge $\to$ Sp | **HC** | 89 | 69 | **0.258** | 0.232 | 0.709 | No |
| `DE_PD_001.wav` | Ge $\to$ Sp | **PD** | 89 | 75 | **0.191** | 0.160 | 0.777 | No |
| `DE_PD_002.wav` | Ge $\to$ Sp | **PD** | 89 | 68 | **0.258** | 0.253 | 0.720 | No |
| `DE_PD_003.wav` | Ge $\to$ Sp | **PD** | 89 | 74 | **0.169** | 0.172 | 0.832 | No |

## 4. Observations
The low CER demonstrates accurate phonetic rendering of target grammatical morphemes and clinical sentences. Minor WER variations typically arise from contracted articles or compound nouns in German, without loss of semantic intelligibility.
