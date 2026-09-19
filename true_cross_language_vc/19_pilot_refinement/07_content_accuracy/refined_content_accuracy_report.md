# Refined Content Accuracy Evaluation Report

## Summary Across Subsets
- **Overall Mean WER**: **0.5799** (58.0%) | **Median**: 0.5870 | **Mean WIP**: **0.4100**
- **Spanish $\to$ German**: Mean WER = **0.7210** (72.1%) | Median = 0.7065 | Mean WIP = **0.2746**
- **German $\to$ Spanish**: Mean WER = **0.4389** (43.9%) | Median = 0.2889 | Mean WIP = **0.5454**
- **Healthy Controls (HC)**: Mean WER = **0.5881** | Mean WIP = **0.4220**
- **Parkinson's Disease (PD)**: Mean WER = **0.5718** | Mean WIP = **0.3980**

| Safe ID | Direction | Cohort | Ref Words | Hyp Words | True WER | True CER | WIP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SP_HC_001.wav` | Spanish $\to$ German | HC | 46 | 14 | **0.804** | 0.717 | **0.126** |
| `SP_HC_002.wav` | Spanish $\to$ German | HC | 46 | 38 | **0.522** | 0.425 | **0.449** |
| `SP_HC_003.wav` | Spanish $\to$ German | HC | 46 | 53 | **0.891** | 0.712 | **0.322** |
| `SP_PD_001.wav` | Spanish $\to$ German | PD | 46 | 42 | **0.609** | 0.498 | **0.377** |
| `SP_PD_002.wav` | Spanish $\to$ German | PD | 46 | 3 | **0.935** | 0.936 | **0.065** |
| `SP_PD_003.wav` | Spanish $\to$ German | PD | 46 | 31 | **0.565** | 0.408 | **0.309** |
| `DE_HC_001.wav` | German $\to$ Spanish | HC | 90 | 69 | **0.278** | 0.284 | **0.701** |
| `DE_HC_002.wav` | German $\to$ Spanish | HC | 90 | 72 | **0.222** | 0.224 | **0.756** |
| `DE_HC_003.wav` | German $\to$ Spanish | HC | 90 | 18 | **0.811** | 0.789 | **0.178** |
| `DE_PD_001.wav` | German $\to$ Spanish | PD | 90 | 74 | **0.222** | 0.215 | **0.757** |
| `DE_PD_002.wav` | German $\to$ Spanish | PD | 90 | 67 | **0.300** | 0.305 | **0.679** |
| `DE_PD_003.wav` | German $\to$ Spanish | PD | 90 | 18 | **0.800** | 0.787 | **0.200** |