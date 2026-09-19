# Speaker Identity Preservation Report (12-File Pilot)

## 1. Methodological Interpretation Rule (Phase 10)
In strict accordance with the thesis master instruction:
- **Independent Problem Formulation**: In true cross-language voice conversion, the spoken phonetic inventory and linguistic rhythm change completely between source and target speech. Therefore, speaker similarity is evaluated independently, without expecting the high reconstruction values obtained during same-content domain shifts.
- **Control Baseline**: Matched same-speaker cross-language pairs are evaluated directly against non-matching cross-speaker pairs to establish statistical discrimination.

## 2. Speaker Verification Distribution
- **Matched Speaker Cross-Language Cosine Similarity**: **0.6178** $\pm$ 0.0617 (Range: [0.4709, 0.7223])
- **Different Speaker Control Baseline**: **0.0798** $\pm$ 0.0966
- **Speaker Discrimination Margin**: **+0.5380**
- **Healthy Controls Mean Similarity**: **0.6340**
- **Parkinson's Disease Mean Similarity**: **0.6016**

## 3. Per-Speaker Cosine Similarity Table

| Safe ID | Direction | Cohort | Matched Speaker Cosine Sim | Control Baseline | Margin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SP_HC_001.wav` | Sp $\to$ Ge | **HC** | **0.7223** | 0.1534 | +0.5689 |
| `SP_HC_002.wav` | Sp $\to$ Ge | **HC** | **0.6689** | 0.1637 | +0.5052 |
| `SP_HC_003.wav` | Sp $\to$ Ge | **HC** | **0.6508** | 0.0414 | +0.6094 |
| `SP_PD_001.wav` | Sp $\to$ Ge | **PD** | **0.6538** | 0.0858 | +0.5680 |
| `SP_PD_002.wav` | Sp $\to$ Ge | **PD** | **0.6304** | 0.2607 | +0.3697 |
| `SP_PD_003.wav` | Sp $\to$ Ge | **PD** | **0.5984** | 0.0719 | +0.5265 |
| `DE_HC_001.wav` | Ge $\to$ Sp | **HC** | **0.5890** | 0.1257 | +0.4633 |
| `DE_HC_002.wav` | Ge $\to$ Sp | **HC** | **0.5993** | -0.0255 | +0.6248 |
| `DE_HC_003.wav` | Ge $\to$ Sp | **HC** | **0.5737** | -0.1215 | +0.6952 |
| `DE_PD_001.wav` | Ge $\to$ Sp | **PD** | **0.4709** | 0.0674 | +0.4035 |
| `DE_PD_002.wav` | Ge $\to$ Sp | **PD** | **0.6125** | 0.0967 | +0.5158 |
| `DE_PD_003.wav` | Ge $\to$ Sp | **PD** | **0.6435** | 0.0381 | +0.6054 |

## 4. Conclusion
Matched-speaker cross-language representations demonstrate clear and consistent separation from cross-speaker controls across all 12 files, proving robust conditioning on the source speaker's vocal characteristics across linguistic domains.
