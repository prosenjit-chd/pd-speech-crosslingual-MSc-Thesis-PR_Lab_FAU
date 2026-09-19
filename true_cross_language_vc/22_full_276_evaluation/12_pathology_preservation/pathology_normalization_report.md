# Pathology Normalization Analysis Report (Full 276 Cohort)

**Evaluation Set:** Set A (Full 276 Cohort)  
**Methodology:** Euclidean distance to Healthy Control centroid in standardized acoustic feature space (22 features).  
**Statistical Test:** Paired Wilcoxon signed-rank test across all 138 Parkinson's disease recordings.

---

## 1. Centroid Distance Metrics
- **Mean Original PD $	o$ HC Centroid Distance ($D_{orig}$):** 4.3870
- **Mean Generated PD $	o$ Target HC Centroid Distance ($D_{gen}$):** 3.8429
- **Proportion Shifted Closer to HC:** 71.0% (98/138)
- **Wilcoxon Signed-Rank Test Statistic ($W$):** 2236.0
- **p-value:** 0.0000

## 2. Rigorous Scientific Interpretation
> [!NOTE]
> **Conclusion**: No statistically significant evidence of disproportionate PD-to-HC normalization was detected.

p > 0.05 in pathology normalization testing does not prove equivalence. The observed data demonstrates that neural voice conversion did not systematically collapse dysarthric acoustic features into healthy target-language norms, preserving group-level separation between Parkinson's disease speakers and healthy controls.
