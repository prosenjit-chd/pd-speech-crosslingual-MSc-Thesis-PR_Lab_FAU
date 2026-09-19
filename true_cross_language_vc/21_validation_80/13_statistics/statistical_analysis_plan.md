# Statistical Analysis Plan: 80-File True Cross-Language Validation

## 1. Experimental Hypotheses
- **Primary Hypothesis $H_1$**: True cross-language voice conversion improves cross-language Parkinson's disease classification performance relative to unconverted speech:
  $$\text{UAR}(\text{Generated Cross}) > \text{UAR}(\text{Original Cross})$$
- **Secondary Hypothesis $H_2$**: True cross-language synthesis preserves pathology-relevant acoustic features without complete normalization:
  $$\text{Preservation Rate} \ge 70\%$$
- **Null Hypothesis $H_0$**: True cross-language synthesis yields no change in cross-language PD detection ($\Delta \text{UAR} \le 0$).

## 2. Statistical Methodology
1. **Primary Metric**: Unweighted Average Recall (UAR / Balanced Accuracy) across strictly balanced 20 HC / 20 PD cohorts.
2. **Confidence Intervals**: Non-parametric percentile bootstrap with $B=1,000$ resamples.
3. **Hypothesis Testing**: Empirical paired bootstrap difference distribution:
   $$p = \frac{1}{B} \sum_{b=1}^B \mathbb{I}\left(\text{UAR}_{\text{gen}}^{(b)} \le \text{UAR}_{\text{orig}}^{(b)}\right)$$
4. **Multiple Comparison Correction**: Benjamini-Hochberg False Discovery Rate (BH-FDR) applied across tested directions and classifiers ($lpha=0.05$).
5. **Pathology Feature Comparisons**: Two-sided Mann-Whitney U tests with Cliff's delta effect sizes and BH-FDR correction.
6. **Acoustic Normalization Testing**: Paired Wilcoxon signed-rank test and paired Student's t-test comparing Euclidean distances to Healthy Control centroids ($D_{\text{orig}}$ vs $D_{\text{gen}}$).
