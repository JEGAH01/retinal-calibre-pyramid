# Weight-sharing ablation: R2 (shared) vs R2ns (per-level networks)

Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon. Holm over the pre-specified cases.

| Case | R2 | R2ns | diff | R2 better on | p | Holm p | R2ns per seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DRIVE_test x1 | 0.637 | 0.616 | +0.021 | 75% | 0.0056 | - | 0.614/0.634/0.601 |
| DRIVE_test x2 native | 0.501 | 0.433 | +0.067 | 100% | 1.9e-06 | 3.8e-06 | 0.433/0.455/0.412 |
| DRIVE_test x4 native | 0.425 | 0.402 | +0.023 | 85% | 0.00013 | - | 0.408/0.416/0.382 |
| STARE x1 | 0.499 | 0.490 | +0.009 | 55% | 0.33 | - | 0.468/0.482/0.521 |
| CHASE_DB1 x1 | 0.414 | 0.350 | +0.064 | 89% | 1e-07 | 3.1e-07 | 0.386/0.400/0.264 |
| HRF x1 | 0.390 | 0.371 | +0.019 | 71% | 1.5e-05 | 1.5e-05 | 0.371/0.382/0.360 |

- Shared better (Holm p < 0.05): CHASE_DB1 x1, HRF x1, DRIVE_test x2 native
- Non-shared better (Holm p < 0.05): none
- Interpretation, fixed in advance: shared better on all three -> weight sharing contributes to the gain; no significant difference -> the gain comes from the pyramid with scale selection, and sharing gives the same accuracy with 12x fewer parameters; non-shared better -> sharing costs accuracy (report it).
