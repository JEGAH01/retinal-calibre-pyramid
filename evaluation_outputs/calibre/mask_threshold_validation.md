# Mask baseline with a validation-chosen threshold

Pre-specified in PRESPECIFICATION_LOG.md (2026-11-12). Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon over images.

## Thresholds

| Seed | Validation images | Validation Dice | Validation threshold | Old (DRIVE test) threshold |
| --- | --- | --- | --- | --- |
| 42 | 23.png, 35.png, 39.png, 36.png | 0.815 | 0.56 | 0.52 |
| 123 | 23.png, 35.png, 39.png, 36.png | 0.814 | 0.58 | 0.56 |
| 7 | 23.png, 35.png, 39.png, 36.png | 0.817 | 0.60 | 0.56 |

## Mask width, old vs new threshold, and R2

| Case | R2 | Mask (old, test thr.) | Mask (new, val thr.) | R2 - mask new | R2 better on | p |
| --- | --- | --- | --- | --- | --- | --- |
| DRIVE_test | 0.637 | 0.568 | 0.577 | +0.060 | 95% | 3.8e-06 |
| STARE | 0.499 | 0.482 | 0.494 | +0.005 | 65% | 0.33 |
| CHASE_DB1 | 0.414 | 0.329 | 0.335 | +0.079 | 93% | 3.3e-06 |
| CHASE_DB1 1stHO | 0.390 | 0.356 | 0.365 | +0.026 | 54% | 0.19 |
| HRF | 0.390 | 0.283 | 0.291 | +0.099 | 80% | 1.2e-06 |
| FIVES_test | 0.224 | 0.349 | 0.341 | -0.116 | 11% | 1e-30 |

## M1 (original primary): R2 - mask, Holm over STARE, CHASE_DB1 (2ndHO), HRF

| Case | Diff | Holm p | Verdict |
| --- | --- | --- | --- |
| STARE | +0.005 | 0.33 | **NOT supported** |
| CHASE_DB1 | +0.079 | 6.7e-06 | **SUPPORTED** |
| HRF | +0.099 | 3.5e-06 | **SUPPORTED** |

## M2: FIVES_test native (wide-scale family: A with B)

| Test | First | Second | Diff | First better on | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| A. R2w - mask | 0.355 | 0.341 | +0.014 | 52% | 0.16 | **NOT supported** |
| B. R2w - P1w (unchanged) | 0.355 | 0.212 | +0.143 | 92% | 3.1e-32 | **SUPPORTED** |
| R2 - mask (uncorrected) | 0.224 | 0.341 | -0.116 | 11% | 1e-30 (uncorr.) | - |

## M3: CHASE_DB1 1stHO (second-observer family: O1 with O2)

| Test | Diff | R2 better on | Holm p | Verdict |
| --- | --- | --- | --- | --- |
| O1. R2 - P1 (unchanged) | +0.032 | 86% | 2.6e-05 | **SUPPORTED** |
| O2. R2 - mask | +0.026 | 54% | 0.19 | **NOT supported** |

## Per seed, mask width at the validation threshold (42/123/7)

| Case | Acc25 |
| --- | --- |
| DRIVE_test | 0.579/0.577/0.574 |
| STARE | 0.484/0.497/0.502 |
| CHASE_DB1 | 0.329/0.344/0.332 |
| CHASE_DB1 1stHO | 0.359/0.372/0.364 |
| HRF | 0.298/0.290/0.286 |
| FIVES_test | 0.356/0.286/0.379 |
