# XTTS-v2 Parameter Search & Candidate Evaluation Report

## 1. Candidate Matrix Summary

| candidate_id   | passed_hard_gates   | files_passed_gates   |   german_lid_accuracy |   mean_p_german |   mean_wer |   median_wer |   mean_cer |   mean_wip |   mean_speaker_similarity |   clipped_files | meets_early_stop   |
|:---------------|:--------------------|:---------------------|----------------------:|----------------:|-----------:|-------------:|-----------:|-----------:|--------------------------:|----------------:|:-------------------|
| C08            | True                | 4/4                  |                     1 |          0.9643 |     0.3424 |       0.2283 |     0.2232 |     0.6056 |                    0.6753 |               0 | True               |
| C06            | True                | 4/4                  |                     1 |          0.9636 |     0.4131 |       0.3587 |     0.3498 |     0.6025 |                    0.6506 |               0 | True               |
| C04            | True                | 4/4                  |                     1 |          0.9513 |     1.6033 |       0.5544 |     0.8337 |     0.4143 |                    0.6486 |               0 | False              |
| C05            | False               | 3/4                  |                     1 |          0.9439 |     0.4456 |       0.4565 |     0.3455 |     0.518  |                    0.6461 |               0 | False              |
| C02            | False               | 3/4                  |                     1 |          0.9562 |     0.5326 |       0.5653 |     0.4614 |     0.5296 |                    0.6534 |               0 | False              |
| C09            | False               | 3/4                  |                     1 |          0.904  |     0.549  |       0.6196 |     0.4496 |     0.362  |                    0.6444 |               0 | False              |
| C01            | False               | 2/4                  |                     1 |          0.9345 |     0.5543 |       0.5434 |     0.5    |     0.4918 |                    0.6479 |               0 | False              |
| C07            | False               | 1/4                  |                     1 |          0.9324 |     0.5707 |       0.6631 |     0.499  |     0.3976 |                    0.6544 |               0 | False              |
| C03            | False               | 2/4                  |                     1 |          0.9863 |     0.6685 |       0.7065 |     0.5869 |     0.3221 |                    0.6379 |               0 | False              |
| C10            | False               | 2/4                  |                     1 |          0.9017 |     1.5272 |       0.5869 |     0.9453 |     0.3907 |                    0.65   |               0 | False              |

## 2. Winning Configuration Analysis
- **Selected Candidate:** `C08`
- **Mean WER:** `0.3424` (Baseline Config C on subset: ~0.81)
- **Mean WIP:** `0.6056`
- **Mean Speaker Similarity:** `0.6753`
- **German LID Accuracy:** `100.0%` (Mean P(de) = `0.9643`)
- **Clipped Files:** `0`
- **Meets Early-Stop Rule:** `True`

## 3. Decision for Secondary Model Testing (Section 19)
> [!NOTE]
> XTTS-v2 best mean WER (0.3424) successfully broke through <= 0.60 threshold. Proceeding to freeze configuration and regenerate full 6-file cohort.