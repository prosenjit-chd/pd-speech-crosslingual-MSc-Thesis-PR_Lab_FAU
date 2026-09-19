# Supervisor Audio Listening Guide: 80-File Validation Cohort

This package contains **8 representative paired recordings** (2 Spanish HC, 2 Spanish PD, 2 German HC, 2 German PD) selected deterministically from the quartiles of the evaluation distribution.

## Representative Examples

| Safe ID | Group | Direction | WER | Speaker Sim | LID P(Tgt) | Original Audio File | Generated Converted Audio File |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `DE_HC_085.wav` | **German HC** | German→Spanish | 0.056 | 0.573 | 0.998 | `DE_HC_085_ORIGINAL_German_HC.wav` | `DE_HC_085_GENERATED_Spanish_HC.wav` |
| `DE_HC_028.wav` | **German HC** | German→Spanish | 0.100 | 0.477 | 0.998 | `DE_HC_028_ORIGINAL_German_HC.wav` | `DE_HC_028_GENERATED_Spanish_HC.wav` |
| `DE_PD_053.wav` | **German PD** | German→Spanish | 0.067 | 0.466 | 0.998 | `DE_PD_053_ORIGINAL_German_PD.wav` | `DE_PD_053_GENERATED_Spanish_PD.wav` |
| `DE_PD_024.wav` | **German PD** | German→Spanish | 0.267 | 0.407 | 0.986 | `DE_PD_024_ORIGINAL_German_PD.wav` | `DE_PD_024_GENERATED_Spanish_PD.wav` |
| `SP_HC_043.wav` | **Spanish HC** | Spanish→German | 0.435 | 0.639 | 0.964 | `SP_HC_043_ORIGINAL_Spanish_HC.wav` | `SP_HC_043_GENERATED_German_HC.wav` |
| `SP_HC_019.wav` | **Spanish HC** | Spanish→German | 0.696 | 0.602 | 0.998 | `SP_HC_019_ORIGINAL_Spanish_HC.wav` | `SP_HC_019_GENERATED_German_HC.wav` |
| `SP_PD_016.wav` | **Spanish PD** | Spanish→German | 0.283 | 0.490 | 0.997 | `SP_PD_016_ORIGINAL_Spanish_PD.wav` | `SP_PD_016_GENERATED_German_PD.wav` |
| `SP_PD_014.wav` | **Spanish PD** | Spanish→German | 0.609 | 0.649 | 0.978 | `SP_PD_014_ORIGINAL_Spanish_PD.wav` | `SP_PD_014_GENERATED_German_PD.wav` |

## Listening Review Protocol
1. **Target-Language Authenticity**: Does the Spanish speaker speak natural German? Does the German speaker speak natural Spanish?
2. **Speaker Preservation**: Does the vocal timbre, gender, and personal identity persist across the linguistic shift?
3. **Acoustic Hygiene**: Verify the complete absence of digital clipping, distortion, clicks, or gating artifacts.
4. **Pathological Retention**: In PD recordings, notice whether speech arrests, prosodic monotony, or hypophonia characteristics remain audible in the converted target-language domain.
