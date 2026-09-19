### Table B: Baseline Cross-Language Classification Performance Gap Across SSL Representations

| SSL Architecture   |   Layer | Classifier          |   Within-Language UAR |   Cross-Language UAR |   Multilingual Pooled UAR |   Absolute Gap (Δ UAR) |   Relative Drop (%) |
|:-------------------|--------:|:--------------------|----------------------:|---------------------:|--------------------------:|-----------------------:|--------------------:|
| xlsr               |       4 | logistic_regression |                0.7907 |               0.572  |                    0.779  |                 0.2186 |               27.65 |
| wav2vec2           |       4 | linear_svm          |                0.763  |               0.5555 |                    0.7428 |                 0.2075 |               27.2  |
| xlsr               |       4 | linear_svm          |                0.778  |               0.5856 |                    0.7609 |                 0.1924 |               24.73 |
| wav2vec2           |      11 | linear_svm          |                0.74   |               0.5561 |                    0.7572 |                 0.1839 |               24.85 |
| xlsr               |       0 | linear_svm          |                0.7622 |               0.5784 |                    0.75   |                 0.1838 |               24.11 |
| xlsr               |       8 | linear_svm          |                0.7586 |               0.5783 |                    0.7681 |                 0.1803 |               23.77 |
| wav2vec2           |       8 | linear_svm          |                0.7266 |               0.5484 |                    0.7536 |                 0.1782 |               24.52 |
| wav2vec2           |       8 | logistic_regression |                0.7757 |               0.6019 |                    0.808  |                 0.1737 |               22.4  |
| xlsr               |       0 | logistic_regression |                0.765  |               0.5926 |                    0.7572 |                 0.1724 |               22.53 |
| wav2vec2           |       4 | logistic_regression |                0.7842 |               0.6124 |                    0.7754 |                 0.1718 |               21.91 |
| xlsr               |      11 | logistic_regression |                0.7885 |               0.6169 |                    0.779  |                 0.1716 |               21.76 |
| xlsr               |      11 | linear_svm          |                0.765  |               0.5955 |                    0.7536 |                 0.1695 |               22.16 |
| wavlm              |       4 | linear_svm          |                0.7678 |               0.5985 |                    0.7826 |                 0.1693 |               22.05 |
| wavlm              |       4 | logistic_regression |                0.775  |               0.6113 |                    0.7862 |                 0.1637 |               21.13 |
| wav2vec2           |      11 | logistic_regression |                0.75   |               0.5913 |                    0.75   |                 0.1587 |               21.17 |
| wavlm              |       8 | logistic_regression |                0.7991 |               0.6417 |                    0.7971 |                 0.1574 |               19.7  |
| wavlm              |       8 | linear_svm          |                0.7443 |               0.5875 |                    0.7754 |                 0.1568 |               21.07 |
| wav2vec2           |       0 | logistic_regression |                0.7358 |               0.5811 |                    0.7428 |                 0.1547 |               21.02 |
| wav2vec2           |       0 | linear_svm          |                0.7465 |               0.6039 |                    0.7645 |                 0.1426 |               19.1  |
| xlsr               |       8 | logistic_regression |                0.7843 |               0.6589 |                    0.7826 |                 0.1255 |               16    |
| wavlm              |      11 | linear_svm          |                0.7365 |               0.6145 |                    0.7572 |                 0.1219 |               16.56 |
| wavlm              |      11 | logistic_regression |                0.7877 |               0.6986 |                    0.7754 |                 0.0891 |               11.31 |
| wavlm              |       0 | linear_svm          |                0.7373 |               0.661  |                    0.75   |                 0.0762 |               10.34 |
| wavlm              |       0 | logistic_regression |                0.7286 |               0.6689 |                    0.7536 |                 0.0598 |                8.2  |

