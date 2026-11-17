# Classical scale-selection baseline (LSS)

Pre-specified in PRESPECIFICATION_LOG.md (2026-11-12). Calibrated k = 0.467 on the 20 DRIVE training images (LSS_k1: k = 1). R2/P1: each image averaged over seeds 42/123/7. Acc25, paired Wilcoxon over images.

## L1. Real unseen data (primary, Holm over four)

| Case | R2 | LSS | R2 - LSS | R2 better on | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| STARE x1 | 0.499 | 0.200 | +0.299 | 100% | 5.7e-06 | **R2 better** |
| CHASE_DB1 x1 | 0.414 | 0.351 | +0.062 | 79% | 0.00068 | **R2 better** |
| HRF x1 | 0.390 | 0.301 | +0.089 | 89% | 1.7e-10 | **R2 better** |
| FIVES_test x1 | 0.224 | 0.213 | +0.011 | 49% | 0.43 | no significant difference |

## L2. Synthetic, against the true radius (primary, Holm over three)

| Case | R2 | LSS | R2 - LSS | R2 better on | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic x1 | 0.662 | 0.174 | +0.488 | 100% | 5.7e-06 | **R2 better** |
| synthetic x2 | 0.532 | 0.185 | +0.347 | 100% | 5.7e-06 | **R2 better** |
| synthetic x4 | 0.543 | 0.157 | +0.386 | 100% | 5.7e-06 | **R2 better** |

## Secondary (uncorrected) and all cases

| Case | R2 | P1 | LSS | LSS k=1 | R2 - LSS (p) | P1 - LSS (p) | LSS bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DRIVE_test x1 | 0.637 | 0.635 | 0.191 | 0.263 | +0.447 (1.9e-06) | +0.444 (1.9e-06) | 1.019 |
| DRIVE_test x2 | 0.394 | 0.431 | 0.162 | 0.253 | +0.232 (1.9e-06) | +0.269 (1.9e-06) | 1.028 |
| DRIVE_test x2 native | 0.501 | 0.399 | 0.204 | 0.249 | +0.297 (1.9e-06) | +0.195 (1.9e-06) | 1.027 |
| DRIVE_test x4 | 0.431 | 0.416 | 0.196 | 0.319 | +0.235 (1.9e-06) | +0.220 (1.9e-06) | 0.768 |
| DRIVE_test x4 native | 0.425 | 0.412 | 0.229 | 0.263 | +0.196 (1.9e-06) | +0.183 (1.9e-06) | 0.930 |
| STARE x1 | 0.499 | 0.499 | 0.200 | 0.196 | +0.299 (1.9e-06) | +0.299 (1.9e-06) | 1.009 |
| CHASE_DB1 x1 | 0.414 | 0.336 | 0.351 | 0.144 | +0.062 (0.00034) | -0.016 (0.14) | 0.815 |
| CHASE_DB1 1stHO x1 | 0.390 | 0.358 | 0.260 | 0.144 | +0.130 (7.5e-09) | +0.098 (3.2e-07) | 0.948 |
| HRF x1 | 0.390 | 0.352 | 0.301 | 0.139 | +0.089 (4.3e-11) | +0.051 (1.1e-09) | 1.121 |
| FIVES_test x1 | 0.224 | 0.125 | 0.213 | 0.357 | +0.011 (0.43) | -0.088 (2.3e-20) | 0.646 |
| synthetic x1 | 0.662 | 0.592 | 0.174 | 0.520 | +0.488 (1.9e-06) | +0.419 (1.9e-06) | 0.580 |
| synthetic x2 | 0.532 | 0.315 | 0.185 | 0.552 | +0.347 (1.9e-06) | +0.130 (0.00017) | 0.565 |
| synthetic x4 | 0.543 | 0.400 | 0.157 | 0.593 | +0.386 (1.9e-06) | +0.243 (1.9e-06) | 0.548 |
