### Table L: Master Question D Crosslingual Classification Synthesis Across All Scenarios

| Evaluation Scenario                                    |   N Models |   Original UAR (D.1/D.2) |   Converted UAR (D.3) |   Δ UAR |   Relative Change (%) |   Original AUC |   Converted AUC |   Paired t-stat |     p-value | Significant Change?   |
|:-------------------------------------------------------|-----------:|-------------------------:|----------------------:|--------:|----------------------:|---------------:|----------------:|----------------:|------------:|:----------------------|
| D.1 Within-Language (Spanish->Spanish, German->German) |         48 |                   0.7666 |                0.8557 |  0.0892 |                 11.63 |         0.8227 |          0.908  |         11.9169 | 8.30963e-16 | True                  |
| D.2/D.3 Cross-Language: Spanish -> German              |         24 |                   0.5992 |                0.5575 | -0.0417 |                 -6.95 |         0.6699 |          0.6395 |         -3.9638 | 0.000615363 | True                  |
| D.2/D.3 Cross-Language: German -> Spanish              |         24 |                   0.605  |                0.575  | -0.03   |                 -4.96 |         0.6975 |          0.6287 |         -3.1029 | 0.00501317  | True                  |
| D.2/D.3 Pure Cross-Language Pooled (Spanish<->German)  |         48 |                   0.6021 |                0.5663 | -0.0358 |                 -5.95 |         0.6837 |          0.6341 |         -5.0358 | 7.45107e-06 | True                  |
| Pooled-Multilingual (Combined->Combined)               |         24 |                   0.752  |                0.8211 |  0.0691 |                  9.19 |         0.8156 |          0.8801 |          8.4143 | 1.78448e-08 | True                  |
| All Scenarios Combined (120 configurations)            |        120 |                   0.6979 |                0.733  |  0.0352 |                  5.04 |         0.7657 |          0.7929 |          5.0679 | 1.49085e-06 | True                  |

