# Calibre: R2 estimator vs width measured from predicted masks

Acc25 at GT centreline pixels; each image averaged over the available seeds; paired Wilcoxon. Mask radius uses the GT radius definition on the binarised probability map. Pre-specified primary: R2 vs maskP1 at the calibrated threshold, Holm over the unseen datasets.

## Mean Acc25 [per-seed means]

| Method | Rule (thresholds) | DRIVE_test | STARE | CHASE_DB1 | HRF |
| --- | --- | --- | --- | --- | --- |
| R2 | - | 0.637 [0.640/0.653/0.618] | 0.499 [0.508/0.529/0.460] | 0.414 [0.363/0.424/0.455] | 0.390 [0.339/0.415/0.416] |
| P1 | - | 0.635 [0.615/0.650/0.639] | 0.499 [0.535/0.465/0.498] | 0.336 [0.283/0.348/0.377] | 0.352 [0.363/0.311/0.383] |
| maskP1 | cal [0.52, 0.56] | 0.568 [0.565/0.569/0.571] | 0.482 [0.489/0.468/0.489] | 0.329 [0.325/0.322/0.341] | 0.283 [0.277/0.288/0.285] |
| maskP1 | 0.5 | 0.556 [0.552/0.563/0.554] | 0.467 [0.469/0.463/0.469] | 0.320 [0.312/0.317/0.330] | 0.273 [0.263/0.284/0.273] |
| maskR2 | cal [0.48, 0.56, 0.6] | 0.564 [0.573/0.556/0.562] | 0.488 [0.496/0.499/0.470] | 0.298 [0.312/0.288/0.293] | 0.261 [0.288/0.250/0.246] |
| maskR2 | 0.5 | 0.550 [0.557/0.562/0.530] | 0.470 [0.475/0.506/0.429] | 0.288 [0.302/0.293/0.268] | 0.251 [0.277/0.255/0.220] |
| maskUNet | cal [0.5] | 0.533 [0.533] | 0.470 [0.470] | 0.317 [0.317] | 0.315 [0.315] |
| maskUNet | 0.5 | 0.533 [0.533] | 0.470 [0.470] | 0.317 [0.317] | 0.315 [0.315] |

## R2 - maskP1 (cal) (PRIMARY)

| Dataset | Mean diff | R2 better on | p | Holm p (unseen) |
| --- | --- | --- | --- | --- |
| DRIVE_test | +0.069 | 95% | 3.8e-06 | - |
| STARE | +0.017 | 70% | 0.07 | 0.07 |
| CHASE_DB1 | +0.085 | 93% | 3.3e-06 | 6.7e-06 |
| HRF | +0.107 | 80% | 4.9e-07 | 1.5e-06 |

**Primary: R2 beats the mask baseline on 2/3 unseen datasets (CHASE_DB1, HRF).**

## R2 - maskR2 (cal)

| Dataset | Mean diff | R2 better on | p | Holm p (unseen) |
| --- | --- | --- | --- | --- |
| DRIVE_test | +0.073 | 95% | 3.8e-06 | - |
| STARE | +0.011 | 70% | 0.15 | 0.15 |
| CHASE_DB1 | +0.116 | 93% | 1.9e-07 | 3.7e-07 |
| HRF | +0.129 | 84% | 3.4e-08 | 1e-07 |

## R2 - maskUNet (cal)

| Dataset | Mean diff | R2 better on | p | Holm p (unseen) |
| --- | --- | --- | --- | --- |
| DRIVE_test | +0.104 | 100% | 1.9e-06 | - |
| STARE | +0.030 | 70% | 0.017 | 0.017 |
| CHASE_DB1 | +0.097 | 96% | 2.2e-08 | 6.7e-08 |
| HRF | +0.075 | 80% | 6.8e-07 | 1.4e-06 |

## R2 - maskP1 (0.5)

| Dataset | Mean diff | R2 better on | p | Holm p (unseen) |
| --- | --- | --- | --- | --- |
| DRIVE_test | +0.081 | 100% | 1.9e-06 | - |
| STARE | +0.032 | 85% | 0.017 | 0.017 |
| CHASE_DB1 | +0.094 | 93% | 2.8e-06 | 5.5e-06 |
| HRF | +0.117 | 82% | 8e-08 | 2.4e-07 |

## R2 - maskR2 (0.5)

| Dataset | Mean diff | R2 better on | p | Holm p (unseen) |
| --- | --- | --- | --- | --- |
| DRIVE_test | +0.088 | 100% | 1.9e-06 | - |
| STARE | +0.029 | 80% | 0.015 | 0.015 |
| CHASE_DB1 | +0.126 | 93% | 1.9e-07 | 3.7e-07 |
| HRF | +0.139 | 84% | 8e-09 | 2.4e-08 |

## R2 - maskUNet (0.5)

| Dataset | Mean diff | R2 better on | p | Holm p (unseen) |
| --- | --- | --- | --- | --- |
| DRIVE_test | +0.104 | 100% | 1.9e-06 | - |
| STARE | +0.030 | 70% | 0.017 | 0.017 |
| CHASE_DB1 | +0.097 | 96% | 2.2e-08 | 6.7e-08 |
| HRF | +0.075 | 80% | 6.8e-07 | 1.4e-06 |

## R2 - P1 (-)

| Dataset | Mean diff | R2 better on | p | Holm p (unseen) |
| --- | --- | --- | --- | --- |
| DRIVE_test | +0.003 | 45% | 0.43 | - |
| STARE | -0.000 | 55% | 0.9 | 0.9 |
| CHASE_DB1 | +0.078 | 96% | 7.5e-08 | 1.5e-07 |
| HRF | +0.038 | 91% | 2.5e-11 | 7.6e-11 |
