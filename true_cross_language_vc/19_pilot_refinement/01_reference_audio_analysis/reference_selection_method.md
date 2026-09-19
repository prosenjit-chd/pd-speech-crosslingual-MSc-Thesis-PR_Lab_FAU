# Deterministic Speaker-Reference Selection Method

## 1. Problem Definition & Motivation
In the initial 12-file pilot, full unsegmented source recordings were supplied directly to the XTTS-v2 speaker encoder:
- Spanish source recordings ranged from 15.1s to 26.1s.
- Especially in Parkinson's disease subjects (e.g. `SP_PD_002` at 26.1s and `SP_PD_003` at 25.1s), the recordings contain prolonged pauses, hesitation breaks, and variable mic-distance effects.
- Analysis revealed that `SP_PD_002` had a full-audio silence ratio of **66.1%**, which destabilized the autoregressive synthesis attention and caused decoder stalling (only 5 words generated).

## 2. Deterministic Selection Rule
To prevent manual cherry-picking and ensure strict scientific objectivity, an automatic deterministic window selection algorithm was implemented:

1. **Target Duration Window**: Exactly $T = 10.0$ seconds.
2. **Window Scanning**: An 10.0-second rectangular window slides across the raw audio with step $\Delta t = 0.5$ seconds.
3. **Objective Window Score**:
   $$\text{Score}(w) = \text{VoicedFraction}(w) \times (1.0 - \text{SilenceRatio}(w)) \times \min\left(1.0, \frac{\text{RMS}(w)}{0.05}\right)$$
   - Disqualify any window exhibiting digital clipping ($\ge 0.999$).
   - Maximize continuous phonated speech while minimizing unvoiced hesitations and room noise.
4. **Edge Smoothing**: Apply a 25-millisecond raised cosine fade-in and fade-out to the boundary samples of the selected segment to prevent spectral click artifacts.
5. **Exact Preservation**: The selected 10-second reference audio comes directly from the original recording without any pitch modification, frequency alteration, or acoustic morphing.

## 3. Comparative Summary: Full Audio vs. Clean Reference Segment

| Safe ID | Lang | Diagnosis | Full Duration (s) | Full Silence % | Full Voiced % | Clean Window (s) | Clean Silence % | Clean Voiced % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SP_HC_001.wav` | Spanish | **HC** | 17.5 | 52.8% | 44.8% | [7.0s – 17.0s] | **47.3%** | **48.9%** |
| `SP_HC_002.wav` | Spanish | **HC** | 17.0 | 61.7% | 38.0% | [6.5s – 16.5s] | **54.1%** | **44.9%** |
| `SP_HC_003.wav` | Spanish | **HC** | 17.1 | 51.3% | 48.2% | [6.5s – 16.5s] | **46.9%** | **51.9%** |
| `SP_PD_001.wav` | Spanish | **PD** | 15.1 | 49.7% | 49.5% | [5.0s – 15.0s] | **47.5%** | **51.5%** |
| `SP_PD_002.wav` | Spanish | **PD** | 26.1 | 66.1% | 33.5% | [8.0s – 18.0s] | **62.5%** | **37.1%** |
| `SP_PD_003.wav` | Spanish | **PD** | 25.1 | 70.5% | 26.9% | [10.5s – 20.5s] | **67.5%** | **29.7%** |
| `DE_HC_001.wav` | German | **HC** | 40.8 | 31.5% | 63.3% | [27.0s – 37.0s] | **20.0%** | **75.0%** |
| `DE_HC_002.wav` | German | **HC** | 46.5 | 31.8% | 61.3% | [23.0s – 33.0s] | **15.8%** | **79.0%** |
| `DE_HC_003.wav` | German | **HC** | 37.4 | 35.0% | 60.5% | [19.5s – 29.5s] | **18.4%** | **75.0%** |
| `DE_PD_001.wav` | German | **PD** | 47.5 | 35.8% | 58.4% | [23.0s – 33.0s] | **17.2%** | **76.8%** |
| `DE_PD_002.wav` | German | **PD** | 38.5 | 37.0% | 60.4% | [22.5s – 32.5s] | **26.2%** | **70.5%** |
| `DE_PD_003.wav` | German | **PD** | 40.5 | 21.8% | 68.3% | [17.0s – 27.0s] | **13.8%** | **81.2%** |

## 4. Availability of Clean References
All 12 extracted clean reference audio segments have been generated and saved under:
`19_pilot_refinement/01_reference_audio_analysis/clean_references/`
