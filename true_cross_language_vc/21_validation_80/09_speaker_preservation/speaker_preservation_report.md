# Speaker Identity Preservation Report: 80-File Validation Cohort

## 1. Executive Summary
- **Model**: Pretrained ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)
- **Matched Pairs (N=80)**:
  * **Mean Similarity**: **0.5654**
  * **Median Similarity**: **0.5698**
  * **Standard Deviation**: 0.0941
  * **5th - 95th Percentile**: [0.4207, 0.7238]
- **Different-Speaker Negative Controls (N=6320)**:
  * **Mean Similarity**: **0.1080**
  * **Median Similarity**: **0.0977**
  * **Standard Deviation**: 0.1021
  * **5th - 95th Percentile**: [-0.0394, 0.2948]
- **Discrimination Margin**:
  * **Mean Discrimination Margin**: **+0.4573**
  * **Median Discrimination Margin**: **+0.4721**
- **Hard Gate Rejection Threshold ($\ge 0.40$)**: Matched similarity comfortably exceeds negative controls across all 80 subjects.

## 2. Speaker Similarity by Direction and Group

| Group | N | Matched Mean ± SD | Matched Median | Control Mean ± SD | Discrimination Margin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **German HC** | 20 | **0.5111 ± 0.0796** | 0.5207 | 0.0950 ± 0.0963 | **+0.4161** |
| **German PD** | 20 | **0.5070 ± 0.0597** | 0.4962 | 0.0947 ± 0.0963 | **+0.4123** |
| **Spanish HC** | 20 | **0.5914 ± 0.0830** | 0.5960 | 0.1296 ± 0.1085 | **+0.4618** |
| **Spanish PD** | 20 | **0.6521 ± 0.0688** | 0.6599 | 0.1129 ± 0.1026 | **+0.5392** |
| **Subtotal (Spanish $\to$ German)** | **40** | **0.6217 ± 0.0813** | **0.6267** | **0.1213 ± 0.1059** | **+0.5005** |
| **Subtotal (German $\to$ Spanish)** | **40** | **0.5090 ± 0.0695** | **0.5009** | **0.0948 ± 0.0963** | **+0.4142** |
