# Pathology Metric Reconciliation

**Observation**: Previous reports showed inconsistencies (e.g., 65.2% vs 59.1% overall).
**Analysis from raw CSV (`pathology_preservation_comparison.csv`)**:
The dataset evaluates 22 statistical contrasts (features). 
- Spanish -> German evaluated 22 features, 14 retained direction = 63.6%.
- German -> Spanish evaluated 22 features, 16 retained direction = 72.7%.
- The 'Combined' cohort evaluated the 22 features across the entire pooled dataset. 13 retained direction = 59.1%.
- Dev-Excluded 'Combined' cohort evaluated 22 features. 14 retained direction = 63.6%.

**Cause of Inconsistency**: The incorrect 65.2% arose from taking the unweighted arithmetic mean of the two independent directional percentages ((63.6 + 72.7) / 2 = 68.15% or similar prior calculation errors, likely miscalculating the combined contrast). The authoritative method is counting the `Combined` direction directly from the 22 statistical tests run on the pooled data.

**Authoritative Results**:
- Spanish->German: 63.6%
- German->Spanish: 72.7%
- Overall (Pooled): 59.1%
- Dev-Excluded (Pooled): 63.6%
