# Target-Language Identification Report: 80-File Validation Cohort

## 1. Executive Summary
Evaluated using frozen independent `openai/whisper-base` decoder token logit distributions:
- **Overall Target-Language Accuracy**: **100.00%** (80/80)
- **Spanish $\to$ German Accuracy**: **100.00%** (40/40)
- **German $\to$ Spanish Accuracy**: **100.00%** (40/40)
- **Engineering Quality Gate ($\ge 90\%$)**: **PASSED**

## 2. Group-Specific LID Breakdown

| Group | N | Target Lang | Accuracy (%) | Mean $P(\text{Target})$ | Median $P(\text{Target})$ | Min $P(\text{Target})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **German HC** | 20 | Spanish | **100.0%** | 0.9911 | 0.9971 | 0.9396 |
| **German PD** | 20 | Spanish | **100.0%** | 0.9874 | 0.9966 | 0.9466 |
| **Spanish HC** | 20 | German | **100.0%** | 0.9222 | 0.9834 | 0.5606 |
| **Spanish PD** | 20 | German | **100.0%** | 0.9306 | 0.9691 | 0.6940 |
