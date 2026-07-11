# Parameter Optimization & Best Setting Selection

Based on the grid search classification stability check and audio quality check, we have identified the optimal parameter set.

## Selected Optimal Configuration
- **Conditioning Model**: XLSR
- **Target Layer**: 11
- **Conditioning Scale (Alpha)**: 1.0
- **Averages**: UAR Delta: `+0.1250` | Abs UAR Delta: `0.1250`
- **Stability**: Positive Delta Runs: 5 | Negative Delta Runs: 0
- **Technical Validation Success Rate**: 100.0%

## Candidate Selection Performance Table

| Model | Layer | Alpha | Mean UAR Delta | Median UAR Delta | Mean Abs Delta | Pos Rows | Neg Rows | Audio Success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XLSR | 11 | 1.0 | +0.1250 | +0.0000 | 0.1250 | 5 | 0 | 100.0% |
| XLSR | 11 | 0.75 | +0.0972 | +0.0000 | 0.0972 | 5 | 0 | 100.0% |
| XLSR | 11 | 0.5 | +0.0278 | +0.0000 | 0.0278 | 2 | 0 | 100.0% |
| WAVLM | 8 | 0.1 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| WAVLM | 8 | 0.25 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| WAVLM | 8 | 0.5 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| WAVLM | 8 | 0.75 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| WAVLM | 8 | 1.0 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| WAVLM | 11 | 0.1 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| WAVLM | 11 | 0.25 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| WAVLM | 11 | 0.5 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| WAVLM | 11 | 0.75 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| WAVLM | 11 | 1.0 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| XLSR | 11 | 0.1 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |
| XLSR | 11 | 0.25 | +0.0139 | +0.0000 | 0.0139 | 1 | 0 | 100.0% |

## Recommendation for Stage 5B
**Decision**: `Proceed to Stage 5B 80-file subset with selected setting`
**Rationale**: The configuration `xlsr_layer11` with `alpha=1.0` shows the best preservation of downstream crosslingual PD/HC diagnostic features while yielding 100% technically validated synthetic speech.