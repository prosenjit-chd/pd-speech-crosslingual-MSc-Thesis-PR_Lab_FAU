# Conversion Quality Scorecard

| Evaluation dimension | Metric | Overall | Spanish→German | German→Spanish | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Speaker preservation | ECAPA cosine similarity | 0.9648 | 0.9599 | 0.9675 | Very high speaker identity retention; German→Spanish slightly outperforms Spanish→German. |
| Spectral change | MCD-DTW (dB) | 5.4381 | 4.2387 | 6.1196 | Successful spectral envelope adaptation; Spanish→German shows less cepstral distortion. |
| Spectral change | LSD (dB) | 17.2291 | 17.6560 | 16.9865 | Moderate overall log-spectral distance change; German→Spanish has lower global distortion. |
| Spectral change | Spectral convergence | 0.7291 | 1.1228 | 0.5054 | Spanish→German exhibits larger deviation in reconstruction convergence than German→Spanish. |
| Content consistency | WER proxy | 0.2178 | 0.2588 | 0.1945 | Moderate word-level degradation; German→Spanish preserves verbal content better. |
| Content consistency | CER proxy | 0.1035 | 0.1079 | 0.1010 | Low character-level degradation (~10% error), very consistent across directions. |
| Content consistency | WIP | 0.6924 | 0.6208 | 0.7331 | Good information preservation; German→Spanish retains more lexical information. |
| Target-domain movement | % acoustic features moving target-ward | 83.7% | 86.0% | 81.4% | Robust acoustic envelope shift toward the target domain. |
| Pathology preservation | pathology sign preservation | 69.2% | 75.0% | 69.2% | Moderate preservation of pathological separation signs. |
| Clinical utility | cross-language UAR original→converted | 0.6021 → 0.5663 | 0.5992 → 0.5575 | 0.6050 → 0.5750 | Slight decrease in zero-shot cross-language diagnostic generalization; utility is slightly degraded. |
