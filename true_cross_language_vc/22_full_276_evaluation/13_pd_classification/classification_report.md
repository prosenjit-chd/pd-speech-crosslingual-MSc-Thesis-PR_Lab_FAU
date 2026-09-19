# Master Classification Report (276 Full Cohort & Sensitivity Partitioning)

| Analysis Set | Direction | Classifier | Within UAR | Original Cross UAR | Generated Cross UAR | Delta UAR | 95% CI | Nominal p | FDR q |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Set A: Full Cohort | Spanish -> German | Linear SVM | 0.7330 | 0.6000 | **0.6400** | **+0.0400** | [-0.0656, +0.1457] | 0.2480 | 0.9920 |
| Set A: Full Cohort | German -> Spanish | Linear SVM | 0.8300 | 0.5909 | **0.5455** | **-0.0455** | [-0.1169, +0.0239] | 0.8850 | 1.0000 |
| Set A: Full Cohort | Spanish -> German | Logistic Regression | 0.7614 | 0.6600 | **0.7100** | **+0.0500** | [-0.0497, +0.1475] | 0.1730 | 0.9920 |
| Set A: Full Cohort | German -> Spanish | Logistic Regression | 0.8500 | 0.6193 | **0.5341** | **-0.0852** | [-0.1457, -0.0338] | 1.0000 | 1.0000 |
| Set B: Development-Excluded | Spanish -> German | Linear SVM | 0.7176 | 0.6489 | **0.6277** | **-0.0213** | [-0.1450, +0.0948] | 0.6410 | 1.0000 |
| Set B: Development-Excluded | German -> Spanish | Linear SVM | 0.7979 | 0.5529 | **0.5412** | **-0.0118** | [-0.0750, +0.0546] | 0.6210 | 1.0000 |
| Set B: Development-Excluded | Spanish -> German | Logistic Regression | 0.7588 | 0.7128 | **0.6702** | **-0.0426** | [-0.1499, +0.0487] | 0.8140 | 1.0000 |
| Set B: Development-Excluded | German -> Spanish | Logistic Regression | 0.8191 | 0.6353 | **0.5647** | **-0.0706** | [-0.1316, -0.0058] | 0.9860 | 1.0000 |