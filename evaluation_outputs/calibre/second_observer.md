# Second-observer test

Pre-specified in PRESPECIFICATION_LOG.md (2026-11-12). Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon over images. obs1/obs2: DRIVE 1st_manual/2nd_manual, CHASE_DB1 1stHO/2ndHO. Images: DRIVE_test 20, CHASE_DB1 28. Results before this test used DRIVE obs1 and CHASE_DB1 obs2.

## Primary (Holm over O1, O2): CHASE_DB1 against obs1 (1stHO), full protocol

| Test | R2 | Other | Diff | R2 better on | p | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O1. R2 - P1 | 0.390 | 0.358 | +0.032 | 86% | 1.3e-05 | 2.6e-05 | **SUPPORTED** |
| O2. R2 - maskP1 | 0.390 | 0.356 | +0.034 | 54% | 0.099 | 0.099 | **NOT supported** |

## Full protocol, all references (R2 vs others, uncorrected; O3 = DRIVE obs2)

| Dataset | Reference | R2 | P1 | maskP1 | maskP1 @0.5 | R2-P1 (p) | R2-maskP1 (p) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DRIVE_test | obs1 1st_manual (used so far) | 0.637 | 0.635 | 0.568 | 0.556 | +0.003 (0.43) | +0.069 (3.8e-06) |
| DRIVE_test | obs2 2nd_manual | 0.656 | 0.652 | 0.574 | 0.560 | +0.004 (0.39) | +0.082 (1.9e-06) |
| CHASE_DB1 | obs1 1stHO | 0.390 | 0.358 | 0.356 | 0.344 | +0.032 (1.3e-05) | +0.034 (0.099) |
| CHASE_DB1 | obs2 2ndHO (used so far) | 0.414 | 0.336 | 0.329 | 0.320 | +0.078 (7.5e-08) | +0.085 (3.3e-06) |

## O4. Human ceiling: shared pixels (X's centreline covered by Y's mask)

| Dataset | Reference X | Coverage | Human (Y vs X) | R2 | P1 | maskP1 | R2 - human (p) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DRIVE_test | obs1 1st_manual | 76% | 0.601 | 0.593 | 0.585 | 0.562 | -0.009 (0.33) |
| DRIVE_test | obs2 2nd_manual | 77% | 0.595 | 0.621 | 0.614 | 0.576 | +0.026 (0.0083) |
| CHASE_DB1 | obs1 1stHO | 81% | 0.473 | 0.412 | 0.346 | 0.361 | -0.061 (0.0013) |
| CHASE_DB1 | obs2 2ndHO | 86% | 0.466 | 0.439 | 0.353 | 0.335 | -0.027 (0.36) |

## O5. Annotator style: median bias (estimate / reference), full protocol

| Dataset | Reference | R2 bias | P1 bias | maskP1 bias | R2 Acc25 |
| --- | --- | --- | --- | --- | --- |
| DRIVE_test | obs1 1st_manual | 1.016 | 1.003 | 1.000 | 0.637 |
| DRIVE_test | obs2 2nd_manual | 1.024 | 1.013 | 1.000 | 0.656 |
| CHASE_DB1 | obs1 1stHO | 0.993 | 0.856 | 1.216 | 0.390 |
| CHASE_DB1 | obs2 2ndHO | 0.889 | 0.794 | 1.181 | 0.414 |

## Per seed, full protocol, Acc25 (42/123/7)

| Dataset | Reference | R2 | P1 | maskP1 |
| --- | --- | --- | --- | --- |
| DRIVE_test | obs1 | 0.653/0.618/0.640 | 0.650/0.639/0.615 | 0.569/0.571/0.565 |
| DRIVE_test | obs2 | 0.677/0.634/0.656 | 0.660/0.661/0.636 | 0.575/0.576/0.570 |
| CHASE_DB1 | obs1 | 0.397/0.400/0.375 | 0.365/0.361/0.349 | 0.351/0.366/0.352 |
| CHASE_DB1 | obs2 | 0.424/0.455/0.363 | 0.348/0.377/0.283 | 0.322/0.341/0.325 |
