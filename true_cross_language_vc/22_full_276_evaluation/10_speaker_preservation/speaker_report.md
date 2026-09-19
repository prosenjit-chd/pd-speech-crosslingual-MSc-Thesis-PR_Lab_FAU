# Speaker Biometric Identity Preservation Report — Full 276 Evaluation

## Metric Overview (ECAPA-TDNN Cosine Similarity)
- **Matched Pairs Evaluated**: 276
- **Different-Speaker Control Pairs (Seed 42)**: 11040
- **Matched Mean Similarity**: 0.5610 (Median: 0.5581, SD: 0.0818)
- **Control Mean Similarity**: 0.0963 (Median: 0.0875, SD: 0.0960)
- **Discrimination Margin**: +0.4647

## Stratified Speaker Preservation Breakdown

| Stratification | Matched Mean | Matched Med | Matched SD | Matched [5th, 95th] | Control Mean | Control Med | Control SD | Control [5th, 95th] | Margin |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Spanish -> German (Overall) | 0.6162 | 0.6190 | 0.0771 | [0.4911, 0.7274] | 0.1153 | 0.1063 | 0.1026 | [-0.0349, 0.3027] | +0.5009 |
| German -> Spanish (Overall) | 0.5297 | 0.5346 | 0.0666 | [0.4234, 0.6349] | 0.0855 | 0.0791 | 0.0903 | [-0.0499, 0.2478] | +0.4442 |
| Healthy Controls (HC) | 0.5504 | 0.5398 | 0.0797 | [0.4244, 0.6941] | 0.0972 | 0.0881 | 0.0975 | [-0.0452, 0.2775] | +0.4533 |
| Parkinson's Disease (PD) | 0.5716 | 0.5650 | 0.0829 | [0.4471, 0.7182] | 0.0954 | 0.0871 | 0.0945 | [-0.0454, 0.2657] | +0.4762 |
| Full Cohort (Set A) | 0.5610 | 0.5581 | 0.0818 | [0.4340, 0.7037] | 0.0963 | 0.0875 | 0.0960 | [-0.0453, 0.2724] | +0.4647 |