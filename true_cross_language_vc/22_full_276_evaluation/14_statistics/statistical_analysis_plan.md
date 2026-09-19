# Statistical Analysis Plan: Full 276-File Evaluation

## 1. Primary Diagnostic Generalization Hypothesis
- **Hypothesis $H_1$**: True cross-language voice conversion improves cross-lingual Parkinson's disease classification performance compared to testing directly on unmodified foreign-language speech:
  $$\Delta \text{UAR} = \text{UAR}_{\text{generated}} - \text{UAR}_{\text{original}} > 0$$
- **Primary Statistical Estimator**: Paired non-parametric bootstrap sampling ($B = 1,000$ iterations) with 95% percentile confidence intervals.
- **Multiple Testing Control**: Benjamini-Hochberg False Discovery Rate (FDR) control applied across all classification comparisons.
- **Decision Criterion**: Statistical significance is declared if and only if $q < 0.05$. Results where $p < 0.05$ but $q \ge 0.05$ are strictly reported as nominally significant before FDR correction.

## 2. Analysis Cohort Partitioning
- **Confirmatory Benchmark**: Independent 80-file validation (`21_validation_80`) remains the immutable primary validation result.
- **Sensitivity Cohort A**: Master Full Cohort ($N = 276$).
- **Sensitivity Cohort B**: Development-Excluded Cohort ($N = 264$, excluding 12 pilot development subjects).
