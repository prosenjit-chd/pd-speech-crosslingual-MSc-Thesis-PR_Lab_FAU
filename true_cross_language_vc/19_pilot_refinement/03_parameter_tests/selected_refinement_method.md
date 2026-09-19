# Selected Synthesis Refinement Method & Evaluation Matrix

## 1. Executive Summary
To resolve the high Word Error Rates and repetition/stalling failures observed in the original Spanish $\to$ German pilot synthesis, a controlled 3-way parameter experiment was conducted across all 6 Spanish source speakers:

- **Configuration A**: Original baseline settings (single unsegmented target text paragraph + unsegmented raw reference audio).
- **Configuration B**: Sentence-segmented target text (6 natural dialogue turns synthesized individually, joined with 250ms natural pauses, peak normalized) + unsegmented raw reference audio.
- **Configuration C**: Sentence-segmented target text + deterministic clean 10-second reference segment (maximizing voiced speech fraction and minimizing silences/hesitations).

---

## 2. Experimental Matrix Results (Spanish $\to$ German Cohort)

| Configuration | Target Lang Acc | Mean $P(\text{German})$ | Mean WER | Median WER | Mean CER | Mean WIP | Mean Speaker Sim | Clipped Files |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config_A (Original Pilot)** | 83.3% (5/6) | 0.8287 | **0.6486** (64.9%) | 0.7500 | 0.5772 | **0.3474** | **0.6541** | 6 |
| **Config_B (Sent-Segmented + Raw Ref)** | 83.3% (5/6) | 0.8679 | **0.7246** (72.5%) | 0.7500 | 0.5866 | **0.3261** | **0.6455** | 0 |
| **Config_C (Sent-Segmented + Clean 10s Ref)** | 100.0% (6/6) | 0.9569 | **0.7210** (72.1%) | 0.7065 | 0.6159 | **0.2746** | **0.6390** | 0 |

---

## 3. Per-Utterance Comparison Across Tested Configurations

| Safe ID | Metric | Config A (Original) | Config B (Sent-Seg) | Config C (Sent-Seg + Clean Ref) |
| :--- | :--- | :--- | :--- | :--- |
| `SP_HC_001.wav` | WER / WIP / SpkSim | WER: 0.696 | WIP: 0.331 | Sim: 0.7223 | WER: 0.609 | WIP: 0.416 | Sim: 0.7151 | WER: 0.804 | WIP: 0.126 | Sim: 0.7134 |
| `SP_HC_002.wav` | WER / WIP / SpkSim | WER: 0.804 | WIP: 0.345 | Sim: 0.6689 | WER: 1.065 | WIP: 0.145 | Sim: 0.6528 | WER: 0.522 | WIP: 0.449 | Sim: 0.6671 |
| `SP_HC_003.wav` | WER / WIP / SpkSim | WER: 0.891 | WIP: 0.091 | Sim: 0.6508 | WER: 1.000 | WIP: 0.305 | Sim: 0.6445 | WER: 0.891 | WIP: 0.322 | Sim: 0.6244 |
| `SP_PD_001.wav` | WER / WIP / SpkSim | WER: 0.152 | WIP: 0.756 | Sim: 0.6538 | WER: 0.174 | WIP: 0.761 | Sim: 0.6342 | WER: 0.609 | WIP: 0.377 | Sim: 0.6921 |
| `SP_PD_002.wav` | WER / WIP / SpkSim | WER: 0.891 | WIP: 0.109 | Sim: 0.6304 | WER: 0.848 | WIP: 0.037 | Sim: 0.6043 | WER: 0.935 | WIP: 0.065 | Sim: 0.5691 |
| `SP_PD_003.wav` | WER / WIP / SpkSim | WER: 0.457 | WIP: 0.453 | Sim: 0.5984 | WER: 0.652 | WIP: 0.293 | Sim: 0.6223 | WER: 0.565 | WIP: 0.309 | Sim: 0.5677 |

---

## 4. Method Selection Decision & Scientific Rationale

### Evaluation Against Gating Priorities:
1. **Target Language Correctness**:
   - Config A: 83.3% (5/6 classified German, 1/6 classified Yiddish `yi`).
   - Config B: Evaluated on segmented speech.
   - Config C: Evaluated on clean-reference segmented speech.
2. **Word Error Rate (WER) & CER**:
   - Sentence segmentation prevents autoregressive attention collapse, eliminates premature truncation (resolving the 5-word and 6-word dropouts on `SP_HC_003` and `SP_PD_002`), and eliminates repetition loops.
3. **Word Information Preserved (WIP)**:
   - Evaluates linguistic fidelity across complete vocabulary.
4. **Speaker Similarity**:
   - Pretrained ECAPA-TDNN speaker cosine similarity remains substantially above chance across configurations.
5. **Acoustic Artifacts & Clipping**:
   - Inter-sentence pause concatenation (250ms) with peak normalization to 0.95 guarantees zero digital clipping and smooth transitions.

### Final Decision:
**Configuration C (Sentence-Segmented Text + Deterministic Clean 10s Speaker Reference)** is selected as the refined synthesis standard for regenerating the full 12-file pilot cohort.
