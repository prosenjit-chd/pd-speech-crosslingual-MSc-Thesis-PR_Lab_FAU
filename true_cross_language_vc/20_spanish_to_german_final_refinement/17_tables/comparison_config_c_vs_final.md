# Spanish -> German Refinement: Config C vs Final Selected Method

## Individual Speaker Comparison
| safe_id   | diagnosis   |   WER_before |   WER_after |   WER_rel_improvement_pct |   WIP_before |   WIP_after |   Similarity_before |   Similarity_after |
|:----------|:------------|-------------:|------------:|--------------------------:|-------------:|------------:|--------------------:|-------------------:|
| SP_HC_001 | HC          |       0.8043 |      0.8913 |                    -10.82 |       0.1258 |      0.4294 |              0.7134 |             0.6975 |
| SP_HC_002 | HC          |       0.5217 |      0.8261 |                    -58.35 |       0.4485 |      0.4854 |              0.6671 |             0.658  |
| SP_HC_003 | HC          |       0.8913 |      0.6957 |                     21.95 |       0.3216 |      0.2663 |              0.6244 |             0.6466 |
| SP_PD_001 | PD          |       0.6087 |      0.3261 |                     46.43 |       0.3773 |      0.6644 |              0.6921 |             0.6671 |
| SP_PD_002 | PD          |       0.9348 |      0.1739 |                     81.4  |       0.0652 |      0.7458 |              0.5691 |             0.5864 |
| SP_PD_003 | PD          |       0.5652 |      0.1522 |                     73.07 |       0.3093 |      0.7561 |              0.5677 |             0.4912 |

## Cohort Summary Comparison
| Metric | Config C Baseline | Final Selected Method | Absolute Delta | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **German LID Accuracy** | 100.0% | 100.0% | 0.0% | Maintained 100% |
| **Mean P(German)** | 0.9569 | 0.9637 | +0.0068 | - |
| **Mean WER** | 0.7210 | 0.5109 | -0.2101 | **+29.14%** |
| **Median WER** | 0.7065 | 0.5109 | -0.1956 | **+27.69%** |
| **Mean CER** | 0.6159 | 0.4599 | -0.1559 | **+25.32%** |
| **Mean WIP** | 0.2746 | 0.5579 | +0.2833 | **+103.16%** |
| **Mean Speaker Similarity** | 0.6390 | 0.6245 | -0.0145 | - |
| **Clipped Files** | 0/6 | 0/6 | 0 | Clean 0.95 Peak |
| **Mean Audio Duration** | 46.03s | 49.03s | +2.99s | No Trailing Loops |
