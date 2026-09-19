# Content Quality vs Diagnostic Utility

**Observation**: 
- Spanish -> German exhibits higher WER (0.8065) and lower content fidelity. However, it showed diagnostic improvement (+0.1250 to +0.1500 UAR) in the independent 80-file validation.
- German -> Spanish exhibits much better linguistic fidelity (WER 0.4430). However, it showed no diagnostic improvement (negative deltas across all cohorts).

**Conclusion**: 
This demonstrates that linguistic fidelity alone is not sufficient to predict PD-classification benefit. The directional asymmetry may reflect differences in synthesis fidelity, classifier decision structure, dataset characteristics, or transfer of pathology-related cues.
