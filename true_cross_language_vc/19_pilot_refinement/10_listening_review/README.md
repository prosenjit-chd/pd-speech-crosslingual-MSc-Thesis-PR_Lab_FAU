# Manual Listening Review Protocol (3-Way Comparison)

This directory provides the structured listening review protocol for comparing:
1. **Original Source Speech** (unprocessed clinical recording)
2. **First Pilot Conversion** (unsegmented baseline XTTS synthesis)
3. **Refined Conversion** (sentence-segmented text + deterministic clean 10s speaker reference)

## Primary Representative Utterances (4 Subjects)
- **Spanish HC $\to$ German**: `SP_HC_001.wav`
- **Spanish PD $\to$ German**: `SP_PD_001.wav`
- **German HC $\to$ Spanish**: `DE_HC_001.wav`
- **German PD $\to$ Spanish**: `DE_PD_001.wav`

## Subjective Evaluation Criteria
For human clinical listening audit:
- `content_intelligible`: Scale 1–5 (1 = unintelligible, 5 = perfectly intelligible)
- `speaker_resemblance`: Scale 1–5 (1 = different person, 5 = identical speaker timbre)
- `artifacts`: Note any unnatural prosody, metallic robotic timbre, or vocoder clicks
- `pathology_characteristics_subjective`: Note whether dysarthric articulation (slurring, tremor, breathiness) is audible or regularized.
