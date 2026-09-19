# Scientific Interpretation of Conversion Quality vs Pathology Preservation

This report presents a clinical and acoustic evaluation of the voice conversion pipeline, distinguishing between speaker adaptation, linguistic preservation, target domain alignment, and pathology preservation.

## Key Scientific Findings

1. **Conversion Quality vs. Target-Domain Movement**:
   - **Target-domain movement** is high, with **86.0% (Spanish→German)** and **81.4% (German→Spanish)** of acoustic features shifting closer to the target domain.
   - However, **conversion quality** (measured by ASR content consistency and speaker similarity) behaves asymmetrically. German→Spanish preserves content better (WER proxy 0.1945 vs. 0.2588) and speaker similarity (0.9675 vs. 0.9599), despite having higher cepstral distortion (MCD-DTW 6.12 dB vs. 4.24 dB).
   - This proves that **larger acoustic shifts (target-ward movement) do not automatically guarantee better conversion quality or speech naturalness**.

2. **Pathology Preservation vs. Diagnostic Generalization (Clinical Utility)**:
   - Voice conversion retains pathological signs moderately (**75.0% Spanish**, **69.2% German** sign preservation).
   - Yet, cross-language clinical diagnostic utility drops slightly (from an original baseline of **0.6021 UAR to 0.5663 UAR**).
   - This suggests that global voice conversion attenuates or distorts subtle, fine-grained dysarthric features, indicating that **voice conversion is not a silver bullet for zero-shot clinical generalization**.

3. **Core Conclusion**:
   $$\text{conversion quality} \neq \text{target-domain movement} \neq \text{pathology preservation} \neq \text{classification improvement}$$
   Each of these represents a distinct dimension of the transformation:
   - *Target-domain movement* measures macro-acoustic shift.
   - *Conversion quality* measures micro-acoustic naturalness, speaker preservation, and content preservation.
   - *Pathology preservation* checks the survival of clinical disease markers.
   - *Clinical utility / Classification* checks the zero-shot generalization of diagnostic systems.
