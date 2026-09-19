### Table G: Frozen Language Classifier Target Probability Shifts and Domain Migration

| Evaluation Group   |   N Samples | True Language   |   Mean German Prob (LR) |   Mean German Prob (SVM) |   LR Pred Spanish |   LR Pred German |   LR Target Class Rate (%) |   SVM Pred Spanish |   SVM Pred German |   SVM Target Class Rate (%) |
|:-------------------|------------:|:----------------|------------------------:|-------------------------:|------------------:|-----------------:|---------------------------:|-------------------:|------------------:|----------------------------:|
| Spanish_Original   |         100 | Spanish         |                  0.0055 |                   0.0565 |               100 |                0 |                       0    |                100 |                 0 |                           0 |
| Spanish_Converted  |         100 | Spanish         |                  0.0702 |                   0.1856 |               100 |                0 |                       0    |                100 |                 0 |                           0 |
| German_Original    |         176 | German          |                  0.9969 |                   0.9685 |                 0 |              176 |                       0    |                  0 |               176 |                           0 |
| German_Converted   |         176 | German          |                  0.782  |                   0.8268 |                18 |              158 |                      10.23 |                  0 |               176 |                           0 |

