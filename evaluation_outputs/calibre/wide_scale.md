# Wide-scale training test: R2w / P1w

Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon.

## Primary (native FIVES_test, Holm over A and B)

| Comparison | first | second | diff | first better on | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| A. R2w - maskP1 | 0.355 | 0.349 | +0.006 | 49% | 0.93 | 0.93 |
| B. R2w - P1w | 0.355 | 0.212 | +0.143 | 92% | 1.5e-32 | 3.1e-32 |

## Secondary

C. Did widening help R2 on native FIVES?

| Comparison | first | second | diff | first better on | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| R2w - R2 (FIVES x1) | 0.355 | 0.224 | +0.131 | 98% | 1.9e-34 | - |

D. Cost check: R2w - R2 on the earlier cases (Holm over five)

| Comparison | first | second | diff | first better on | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| DRIVE_test x1 | 0.582 | 0.637 | -0.055 | 5% | 5.7e-06 | 2.3e-05 |
| DRIVE_test x2 native | 0.522 | 0.501 | +0.021 | 70% | 0.0017 | 0.0051 |
| STARE x1 | 0.455 | 0.499 | -0.044 | 5% | 3.8e-06 | 1.9e-05 |
| CHASE_DB1 x1 | 0.405 | 0.414 | -0.009 | 32% | 0.025 | 0.049 |
| HRF x1 | 0.376 | 0.390 | -0.014 | 33% | 0.31 | 0.31 |

E. R2w - P1w elsewhere (uncorrected)

| Comparison | first | second | diff | first better on | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| DRIVE_test x1 | 0.582 | 0.589 | -0.007 | 40% | 0.23 | - |
| DRIVE_test x2 native | 0.522 | 0.427 | +0.095 | 100% | 1.9e-06 | - |
| STARE x1 | 0.455 | 0.461 | -0.005 | 50% | 0.62 | - |
| CHASE_DB1 x1 | 0.405 | 0.343 | +0.062 | 79% | 5.5e-05 | - |
| HRF x1 | 0.376 | 0.337 | +0.039 | 82% | 2.1e-07 | - |
| FIVES_test x0.27 native | 0.403 | 0.381 | +0.021 | 68% | 8.3e-08 | - |

## Per seed and bias

| Model | Case | per seed (42/123/7) | median bias |
| --- | --- | --- | --- |
| R2 | FIVES_test x1 | 0.228/0.318/0.126 | 0.573 |
| R2 | DRIVE_test x1 | 0.653/0.618/0.640 | 1.014 |
| R2 | DRIVE_test x2 native | 0.503/0.528/0.471 | 0.954 |
| R2 | STARE x1 | 0.529/0.460/0.508 | 1.089 |
| R2 | CHASE_DB1 x1 | 0.424/0.455/0.363 | 0.893 |
| R2 | HRF x1 | 0.415/0.416/0.339 | 0.978 |
| R2 | FIVES_test x0.27 native | 0.418/0.476/0.355 | 0.780 |
| P1 | FIVES_test x1 | 0.178/0.148/0.049 | 0.368 |
| P1 | DRIVE_test x1 | 0.650/0.639/0.615 | 1.006 |
| P1 | DRIVE_test x2 native | 0.378/0.435/0.384 | 0.977 |
| P1 | STARE x1 | 0.465/0.498/0.535 | 1.076 |
| P1 | CHASE_DB1 x1 | 0.348/0.377/0.283 | 0.770 |
| P1 | HRF x1 | 0.311/0.383/0.363 | 0.971 |
| P1 | FIVES_test x0.27 native | 0.405/0.463/0.257 | 0.765 |
| R2w | FIVES_test x1 | 0.325/0.377/0.363 | 0.717 |
| R2w | DRIVE_test x1 | 0.557/0.571/0.617 | 1.009 |
| R2w | DRIVE_test x2 native | 0.529/0.523/0.513 | 1.018 |
| R2w | STARE x1 | 0.456/0.434/0.475 | 1.142 |
| R2w | CHASE_DB1 x1 | 0.389/0.387/0.438 | 0.929 |
| R2w | HRF x1 | 0.380/0.344/0.405 | 1.240 |
| R2w | FIVES_test x0.27 native | 0.358/0.415/0.435 | 0.792 |
| P1w | FIVES_test x1 | 0.183/0.140/0.313 | 0.561 |
| P1w | DRIVE_test x1 | 0.558/0.626/0.583 | 1.010 |
| P1w | DRIVE_test x2 native | 0.417/0.443/0.420 | 0.986 |
| P1w | STARE x1 | 0.488/0.445/0.449 | 1.108 |
| P1w | CHASE_DB1 x1 | 0.273/0.336/0.419 | 0.769 |
| P1w | HRF x1 | 0.361/0.407/0.242 | 1.326 |
| P1w | FIVES_test x0.27 native | 0.270/0.437/0.437 | 0.803 |
| maskP1 | FIVES_test x1 | 0.367/0.291/0.389 | 0.658 |

- A (R2w beats mask width on native FIVES): **NOT supported**
- B (R2w beats P1w on native FIVES): **SUPPORTED**
- Significant losses vs R2 (Holm): DRIVE_test x1, STARE x1, CHASE_DB1 x1
- Main estimator by the pre-specified rule: **R2**
