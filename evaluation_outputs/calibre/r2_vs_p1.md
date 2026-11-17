# R2 versus P1 (three seeds)

Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon. Holm over STARE, CHASE_DB1 and HRF.

| Case | R2 | P1 | diff | R2 better on | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| DRIVE_test x1 | 0.637 | 0.635 | +0.003 | 45% | 0.43 | - |
| DRIVE_test x2 native | 0.501 | 0.399 | +0.102 | 100% | 1.9e-06 | - |
| DRIVE_test x2 | 0.394 | 0.431 | -0.037 | 35% | 0.0037 | - |
| DRIVE_test x4 native | 0.425 | 0.412 | +0.012 | 60% | 0.084 | - |
| DRIVE_test x4 | 0.431 | 0.416 | +0.015 | 55% | 0.28 | - |
| STARE x1 | 0.499 | 0.499 | -0.000 | 55% | 0.9 | 0.9 |
| CHASE_DB1 x1 | 0.414 | 0.336 | +0.078 | 96% | 7.5e-08 | 1.5e-07 |
| HRF x1 | 0.390 | 0.352 | +0.038 | 91% | 2.5e-11 | 7.6e-11 |
