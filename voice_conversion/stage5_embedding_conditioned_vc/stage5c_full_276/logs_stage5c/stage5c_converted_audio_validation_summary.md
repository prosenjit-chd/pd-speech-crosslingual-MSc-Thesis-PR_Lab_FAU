# Stage 5C — Audio Validation Summary

This report summarizes the technical specifications check for all converted WAV files.

## Summary Metrics
- **Total Files Expected**: 276
- **Total Files Found**: 276
- **Success Count (specifications met, no clipping)**: 273
- **Warning Count (specifications met, with clipping)**: 3
- **Failed Count (missing/silent/wrong format)**: 0
- **Sample Rate Pass Count (22050 Hz)**: 276 / 276
- **Mono Pass Count (1 channel)**: 276 / 276
- **Average Duration Difference**: -0.0040 seconds
- **Maximum Duration Difference**: 0.0116 seconds
- **Average RMS**: 0.0966
- **RMS Range**: [0.0179, 0.3738]
- **Maximum Peak Amplitude**: 1.0000
- **Peak Amplitude Range**: [0.1644, 1.0000]
- **Number of Clipped Files**: 3 / 276 (1.1%)
- **Clipping Percentage**: 1.1%

## Clipping Warnings & Thresholds
- **Status**: PASSED
- **Message**: No warning. Clipping is within acceptable threshold (<= 5%).

## Decision Rules Output
> [!NOTE]
> **Pipeline Continue Condition Met**: All 276 converted files meet technical specifications. Proceed to classification evaluation.