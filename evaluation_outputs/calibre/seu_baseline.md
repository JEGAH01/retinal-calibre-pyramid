# Learned scale-equivariant baseline (SEU)

Pre-specified in PRESPECIFICATION_LOG.md (2026-11-13). SEU: scale-equivariant U-Net (Sangalli et al., 2022), 488,325 parameters as R2. Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon over images.

## E1. Real unseen data (primary, Holm over four)

| Case | R2 | SEU | R2 - SEU | R2 better on | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| STARE x1 | 0.499 | 0.494 | +0.005 | 50% | 0.41 | no significant difference |
| CHASE_DB1 x1 | 0.414 | 0.395 | +0.019 | 75% | 0.0016 | **R2 better** |
| HRF x1 | 0.390 | 0.401 | -0.011 | 38% | 0.027 | **SEU better** |
| FIVES_test x1 | 0.224 | 0.207 | +0.017 | 70% | 3.6e-10 | **R2 better** |

## E2. Synthetic, against the true radius (primary, Holm over three)

| Case | R2 | SEU | R2 - SEU | R2 better on | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic x1 | 0.662 | 0.662 | -0.001 | 55% | 0.99 | no significant difference |
| synthetic x2 | 0.532 | 0.541 | -0.009 | 45% | 0.95 | no significant difference |
| synthetic x4 | 0.543 | 0.563 | -0.020 | 30% | 0.12 | no significant difference |

## Secondary (uncorrected) and all cases

| Case | R2 | P1 | SEU | R2 - SEU (p) | SEU - P1 (p) | Median bias R2 / SEU |
| --- | --- | --- | --- | --- | --- | --- |
| DRIVE_test x1 | 0.637 | 0.635 | 0.624 | +0.013 (4.8e-05) | -0.011 (0.017) | 1.016 / 1.018 |
| DRIVE_test x2 | 0.394 | 0.431 | 0.393 | +0.001 (1) | -0.039 (0.00085) | 1.194 / 1.224 |
| DRIVE_test x2 native | 0.501 | 0.399 | 0.522 | -0.022 (1.9e-05) | +0.123 (1.9e-06) | 0.949 / 0.975 |
| DRIVE_test x4 | 0.431 | 0.416 | 0.443 | -0.013 (0.00048) | +0.027 (0.021) | 0.955 / 1.000 |
| DRIVE_test x4 native | 0.425 | 0.412 | 0.426 | -0.001 (0.73) | +0.014 (0.07) | 0.986 / 1.054 |
| STARE x1 | 0.499 | 0.499 | 0.494 | +0.005 (0.41) | -0.005 (0.5) | 1.103 / 1.091 |
| CHASE_DB1 x1 | 0.414 | 0.336 | 0.395 | +0.019 (0.00053) | +0.059 (1.5e-08) | 0.889 / 0.847 |
| CHASE_DB1 1stHO x1 | 0.390 | 0.358 | 0.397 | -0.007 (0.14) | +0.039 (2.5e-07) | 0.993 / 0.936 |
| HRF x1 | 0.390 | 0.352 | 0.401 | -0.011 (0.013) | +0.049 (1.8e-09) | 0.989 / 1.057 |
| FIVES_test x1 | 0.224 | 0.125 | 0.207 | +0.017 (8.9e-11) | +0.082 (3.7e-34) | 0.547 / 0.522 |
| synthetic x1 | 0.662 | 0.592 | 0.662 | -0.001 (0.99) | +0.070 (0.00026) | 1.134 / 1.134 |
| synthetic x2 | 0.532 | 0.315 | 0.541 | -0.009 (0.47) | +0.226 (1.9e-06) | 1.203 / 1.212 |
| synthetic x4 | 0.543 | 0.400 | 0.563 | -0.020 (0.04) | +0.163 (1.9e-06) | 1.058 / 1.124 |

## SEU per seed (42/123/7)

| Case | Acc25 |
| --- | --- |
| DRIVE_test x1 | 0.638/0.594/0.640 |
| DRIVE_test x2 | 0.395/0.385/0.397 |
| DRIVE_test x2 native | 0.519/0.519/0.528 |
| DRIVE_test x4 | 0.446/0.428/0.456 |
| DRIVE_test x4 native | 0.427/0.413/0.438 |
| STARE x1 | 0.519/0.462/0.501 |
| CHASE_DB1 x1 | 0.381/0.439/0.364 |
| CHASE_DB1 1stHO x1 | 0.408/0.386/0.397 |
| HRF x1 | 0.388/0.424/0.392 |
| FIVES_test x1 | 0.208/0.236/0.176 |
| synthetic x1 | 0.763/0.546/0.678 |
| synthetic x2 | 0.577/0.517/0.529 |
| synthetic x4 | 0.569/0.561/0.558 |
