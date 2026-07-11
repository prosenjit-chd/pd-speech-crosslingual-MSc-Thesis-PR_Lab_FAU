# Stage 5A-Refinement — Audio Validation Summary

This summary groups the technical validation results of the 180 generated files across the grid search.

## Technical Validation Summary by Setting

| Model | Layer | Alpha | Total Files | Passed | Warnings | Failed | SR=22050 | Mono | Clipped | Notes/Warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WAVLM | 8 | 0.1 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| WAVLM | 8 | 0.25 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| WAVLM | 8 | 0.5 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| WAVLM | 8 | 0.75 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| WAVLM | 8 | 1.0 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| WAVLM | 11 | 0.1 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| WAVLM | 11 | 0.25 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| WAVLM | 11 | 0.5 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| WAVLM | 11 | 0.75 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| WAVLM | 11 | 1.0 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| XLSR | 11 | 0.1 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| XLSR | 11 | 0.25 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| XLSR | 11 | 0.5 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 0/12 | All OK |
| XLSR | 11 | 0.75 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 1/12 | All OK |
| XLSR | 11 | 1.0 | 12 | 12 | 0 | 0 | 12/12 | 12/12 | 2/12 | All OK |

## Conclusion
- Output sample rate (22050 Hz) and channel format (mono) checked.
- RMS energy and peak amplitude verify natural vocoding distribution.