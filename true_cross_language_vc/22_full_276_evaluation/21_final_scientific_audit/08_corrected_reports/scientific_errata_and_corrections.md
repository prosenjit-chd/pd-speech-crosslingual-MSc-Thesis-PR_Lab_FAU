# Scientific Errata and Corrections

1. **Pathology Percentage Inconsistency**
   - **Original Statement**: Overall pathology direction preservation = 65.2%
   - **Corrected Statement**: Overall pathology direction preservation = 59.1%
   - **Reason**: The 65.2% was an unweighted arithmetic mean of directional percentages. The verified result computes the preservation percentage directly from the pooled statistical contrasts.
   - **Authoritative Source**: `12_pathology_preservation/pathology_preservation_comparison.csv`

2. **Normalization Interpretation Error**
   - **Original Statement**: "No statistically significant evidence of disproportionate PD-to-HC normalization was detected" while p=0.0000.
   - **Corrected Statement**: "Generated PD speech was significantly closer to the HC centroid under this feature-space distance measure, consistent with partial acoustic normalization."
   - **Reason**: If D_gen < D_orig and p < 0.05, it means distance significantly decreased.
   - **Authoritative Source**: `12_pathology_preservation/pathology_normalization_analysis.csv`

3. **Causal Claims**
   - **Original Statement**: Explanations claiming conversion "proves the language barrier was overcome" or defining the exact mechanism of the classifier's failure.
   - **Corrected Statement**: Replaced with cautious wording: "The directional asymmetry may reflect differences in synthesis fidelity, classifier decision structure, dataset characteristics, or transfer of pathology-related cues."
   - **Reason**: Removing unsupported causal explanations.
