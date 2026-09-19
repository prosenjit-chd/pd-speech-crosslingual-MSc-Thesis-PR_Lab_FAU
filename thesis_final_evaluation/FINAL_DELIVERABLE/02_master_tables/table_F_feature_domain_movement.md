### Table F: Top Feature-Level Distance Reductions Toward Target Domain After Voice Conversion

| Conversion Direction   | Acoustic Feature         |   Dist Before (std) |   Dist After (std) |   Mean Movement (std) |   % Utterances Closer |   Wilcoxon W |   Raw p-val |   FDR q-val | Movement Status   |
|:-----------------------|:-------------------------|--------------------:|-------------------:|----------------------:|----------------------:|-------------:|------------:|------------:|:------------------|
| Spanish->German-domain | mfcc_6_mean              |              1.6581 |             0.3364 |                1.3217 |                 98    |           20 | 7.11708e-18 | 1.85044e-16 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_1_mean              |              1.3834 |             0.1447 |                1.2387 |                 99    |            1 | 4.01616e-18 | 1.85044e-16 | CLOSER_TO_TARGET  |
| German->Spanish-domain | mfcc_1_mean              |              1.4226 |             0.2497 |                1.1729 |                 96.59 |           66 | 3.81907e-30 | 6.61971e-29 | CLOSER_TO_TARGET  |
| German->Spanish-domain | mfcc_11_mean             |              1.3389 |             0.2139 |                1.125  |                 93.18 |          174 | 2.3601e-29  | 3.06813e-28 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_11_mean             |              1.3371 |             0.2306 |                1.1065 |                 94    |           92 | 5.98888e-17 | 1.03807e-15 | CLOSER_TO_TARGET  |
| German->Spanish-domain | mfcc_6_mean              |              1.653  |             0.6202 |                1.0328 |                 93.18 |          243 | 7.4564e-29  | 7.75466e-28 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_8_mean              |              1.2859 |             0.404  |                0.882  |                 90    |          179 | 7.24546e-16 | 9.4191e-15  | CLOSER_TO_TARGET  |
| German->Spanish-domain | mfcc_8_mean              |              1.2721 |             0.4328 |                0.8393 |                 83.52 |          878 | 1.82029e-24 | 1.35222e-23 | CLOSER_TO_TARGET  |
| German->Spanish-domain | mfcc_12_mean             |              1.0277 |             0.2498 |                0.7779 |                 86.36 |          668 | 7.09677e-26 | 6.15054e-25 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_2_mean              |              1.1922 |             0.4683 |                0.7239 |                 76    |          740 | 8.38778e-10 | 3.35511e-09 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_5_mean              |              1.1484 |             0.4344 |                0.714  |                 87    |          281 | 1.20414e-14 | 1.04358e-13 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_12_mean             |              1.0139 |             0.3206 |                0.6933 |                 85    |          360 | 9.77013e-14 | 6.35058e-13 | CLOSER_TO_TARGET  |
| Spanish->German-domain | voiced_segment_rate_hz   |              1.4755 |             0.8217 |                0.6538 |                 78    |          667 | 1.67622e-10 | 8.10405e-10 | CLOSER_TO_TARGET  |
| German->Spanish-domain | mfcc_13_mean             |              0.9056 |             0.2828 |                0.6228 |                 79.55 |         1345 | 1.75885e-21 | 1.14325e-20 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_10_mean             |              0.8277 |             0.2816 |                0.5461 |                 74    |          668 | 1.71432e-10 | 8.10405e-10 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_13_mean             |              0.9235 |             0.383  |                0.5405 |                 78    |          680 | 2.24298e-10 | 9.71957e-10 | CLOSER_TO_TARGET  |
| Spanish->German-domain | spectral_rolloff_85_std  |              1.0826 |             0.571  |                0.5116 |                 71    |          981 | 1.10365e-07 | 3.826e-07   | CLOSER_TO_TARGET  |
| German->Spanish-domain | duration_voiced_sec      |              1.7856 |             1.287  |                0.4986 |                 72.16 |         3299 | 3.31835e-11 | 1.32734e-10 | CLOSER_TO_TARGET  |
| German->Spanish-domain | spectral_rolloff_85_mean |              0.7902 |             0.313  |                0.4772 |                 74.43 |         2655 | 3.37404e-14 | 1.595e-13   | CLOSER_TO_TARGET  |
| German->Spanish-domain | mfcc_9_mean              |              0.7855 |             0.3209 |                0.4647 |                 71.59 |         2530 | 7.98853e-15 | 4.15404e-14 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_9_mean              |              0.8757 |             0.4186 |                0.4571 |                 72    |          886 | 1.74638e-08 | 6.48655e-08 | CLOSER_TO_TARGET  |
| German->Spanish-domain | mfcc_5_mean              |              1.2342 |             0.7868 |                0.4473 |                 68.75 |         3719 | 1.84127e-09 | 5.98411e-09 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_3_mean              |              0.8146 |             0.4083 |                0.4063 |                 67    |         1206 | 5.75624e-06 | 1.76073e-05 | CLOSER_TO_TARGET  |
| Spanish->German-domain | mfcc_7_mean              |              0.8246 |             0.4367 |                0.3879 |                 70    |         1097 | 9.11099e-07 | 2.96107e-06 | CLOSER_TO_TARGET  |
| German->Spanish-domain | spectral_rolloff_85_std  |              1.1404 |             0.7565 |                0.3839 |                 67.05 |         3830 | 4.99789e-09 | 1.52877e-08 | CLOSER_TO_TARGET  |

