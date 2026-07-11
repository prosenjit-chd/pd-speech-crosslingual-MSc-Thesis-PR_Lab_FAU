# Stage 5A-Refinement — Parameter Optimization & Grid Search Report

## 1. Goal of Stage 5A-Refinement
The goal of this refinement experiment is to systematically test 15 parameter configurations across multiple conditioning representations and scales on the same 12 pilot files. We analyze downstream classification behavior and voice conversion characteristics to determine the best parameters for the subsequent 80-file Stage 5B scale-up.

## 2. Safety Rule Confirmation
We confirm that all previous baseline, HiFi-GAN 12, 80, and 276 folders, logs, and files remain completely **read-only** and untouched. All completed Stage 5A results were kept read-only. All new refinement outputs were isolated inside:
`C:\pd-speech-crosslingual\voice_conversion\stage5_embedding_conditioned_vc\stage5a_refinement`

## 3. Why Refinement was Needed
While the Stage 5A pilot successfully verified technical audio vocoding, downstream crosslingual classification results were mixed. Since representations (such as WavLM) show different acoustic preservation capabilities compared to XLSR, a grid search over WavLM L8, WavLM L11, and XLSR L11 with alphas ranging from 0.1 to 1.0 was necessary to choose the optimal condition for scaling.

## 4. Tested Conditioning Models and Layers
- **XLSR Layer 11** (1024 dimensions)
- **WavLM Layer 8** (768 dimensions)
- **WavLM Layer 11** (768 dimensions)

## 5. Tested Alpha (Conditioning Scale) Values
- `alpha = 0.1` (weak conversion scale)
- `alpha = 0.25` (moderate conversion scale)
- `alpha = 0.5` (moderate conversion scale)
- `alpha = 0.75` (strong conversion scale)
- `alpha = 1.0` (strong conversion scale)

## 6. Audio Validation Summary
A total of 180 generated WAV files (15 settings $\times$ 12 files) were technically verified:
- **Success (specifications met)**: 180/180 (100.0%)
- **Warnings (technical deviations)**: 0
- **Failed**: 0

All generated audios matched the required vocoder configuration (22050 Hz, single channel mono). No empty waveforms or silence were found.

## 7. Classification Diagnostic Comparison
> [!IMPORTANT]
> All classification metrics reported below represent diagnostic crosslingual evaluation check values on a very small pilot sample. They serve to observe representation drift and domain shifts rather than generalizable performance.

| Model | Layer | Alpha | Mean UAR Original | Mean UAR Converted | Mean UAR Delta | Mean Acc Delta | Diagnostic Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WAVLM | 8 | 0.1 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| WAVLM | 8 | 0.25 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| WAVLM | 8 | 0.5 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| WAVLM | 8 | 0.75 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| WAVLM | 8 | 1.0 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| WAVLM | 11 | 0.1 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| WAVLM | 11 | 0.25 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| WAVLM | 11 | 0.5 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| WAVLM | 11 | 0.75 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| WAVLM | 11 | 1.0 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| XLSR | 11 | 0.1 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| XLSR | 11 | 0.25 | 0.5139 | 0.5278 | +0.0139 | +0.0139 | **DIAGNOSTIC_ONLY** |
| XLSR | 11 | 0.5 | 0.5139 | 0.5417 | +0.0278 | +0.0278 | **DIAGNOSTIC_ONLY** |
| XLSR | 11 | 0.75 | 0.5139 | 0.6111 | +0.0972 | +0.0972 | **DIAGNOSTIC_ONLY** |
| XLSR | 11 | 1.0 | 0.5139 | 0.6389 | +0.1250 | +0.1250 | **DIAGNOSTIC_ONLY** |

## 8. Best Setting Selection
### Optimal Selection Details
- **Conditioning Model**: XLSR
- **Target Layer**: 11
- **Conditioning Scale (Alpha)**: 1.0
- **Averages**: UAR Delta: `+0.1250` | Abs UAR Delta: `0.1250`
- **Stability**: Positive Delta Runs: 5 | Negative Delta Runs: 0
- **Technical Validation Success Rate**: 100.0%

### Stage 5B Recommendation
**Decision**: `Proceed to Stage 5B 80-file subset with selected setting`
**Rationale**: The configuration `xlsr_layer11` with `alpha=1.0` shows the best preservation of downstream crosslingual PD/HC diagnostic features while yielding 100% technically validated synthetic speech.


## 9. Scientific Limitations
> [!WARNING]
> **Critical Limitation Statement:**
> **“The 12-file Stage 5A experiment tests technical feasibility only. Because the embedding-to-mel mapping is trained on a very small pilot set, the results cannot be interpreted as final conversion performance.”**

- **No Language Translation**: This process is strictly acoustic. German speech was converted **toward the Spanish acoustic/domain condition**, and Spanish speech was converted **toward the German acoustic/domain condition** in log-mel feature space before vocoding.
- **Diagnostic-Only Results**: Classification results represent **diagnostic-only classification results** and not generalizable clinical performance.

## 10. Decision for Stage 5B
The selected parameter configuration and scale will be used for the Stage 5B 80-file scaling, isolating all Stage 5B runs inside `stage5b_subset_80` under the same safety constraints.

---
*Report generated automatically by `07_write_stage5a_refinement_report.py`*