# Input Validation Report: Complete Read-Text Speech Inventory

## 1. Inventory Summary
This inventory verifies the complete read-text dataset of **276 recordings** from the read-only source directory, mapped to non-identifying safe IDs for true cross-language voice conversion.

| Language | Diagnosis | Cohort Count | Dataset Source | Mean Duration (s) | Min Duration (s) | Max Duration (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Spanish** | Healthy Control (HC) | 50 | PC-GITA | 17.62 | 11.17 | 34.01 |
| **Spanish** | Parkinson's Disease (PD) | 50 | PC-GITA | 18.62 | 10.35 | 45.26 |
| **German** | Healthy Control (HC) | 88 | Sabine Skoda | 46.45 | 32.20 | 68.27 |
| **German** | Parkinson's Disease (PD) | 88 | Sabine Skoda | 45.97 | 32.18 | 120.69 |
| **Total** | All Combined | **276** | Both Datasets | **36.03** | **10.35** | **120.69** |

## 2. Integrity Verification
- **Total Files Checked**: 276
- **Missing Files**: 0
- **Channels**: All mono (channels = 1)
- **Sample Rates**: Spanish files @ 44,100 Hz, German files @ 16,000 Hz.
- **Safe ID Anonymization**: All public references use non-identifying IDs (`SP_HC_xxx`, `SP_PD_xxx`, `DE_HC_xxx`, `DE_PD_xxx`).
- **Data Protection Compliance**: Raw source datasets remain 100% read-only.
