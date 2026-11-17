# Synthetic vessels of known width

Pre-specified in PRESPECIFICATION_LOG.md (2026-11-12). 20 images, 46096 scoring pixels at x1. Acc25 per image, each image averaged over seeds 42/123/7; paired Wilcoxon over images.

## S1. Mask-derived reference / true radius (median, by true native radius)

| Factor | Reference | r 0.5-1 | r 1-1.5 | r 1.5-2 | r 2-3 | r 3-4 | All |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x1 | enlarged mask (original protocol) | 0.702 | 0.670 | 0.857 | 0.825 | 0.876 | 0.805 |
| x1 | native mask x t (native grid) | 0.702 | 0.670 | 0.857 | 0.825 | 0.876 | 0.805 |
| x2 | enlarged mask (original protocol) | 0.489 | 0.725 | 0.777 | 0.840 | 0.872 | 0.777 |
| x2 | native mask x t (native grid) | 0.702 | 0.670 | 0.857 | 0.825 | 0.876 | 0.805 |
| x4 | enlarged mask (original protocol) | 0.741 | 0.842 | 0.852 | 0.876 | 0.902 | 0.859 |
| x4 | native mask x t (native grid) | 0.702 | 0.670 | 0.857 | 0.825 | 0.876 | 0.805 |

## S2. Protocol validity: R2 - P1 Acc25 under each protocol (primary at x2)

| Factor | Δ true | Δ original | Δ native grid | mean abs(Δorig-Δtrue) | mean abs(Δnative-Δtrue) | p | Native grid better? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x2 (primary) | +0.218 | +0.165 | +0.165 | 0.064 | 0.063 | 0.35 | **NOT supported** |
| x4 | +0.143 | +0.176 | +0.150 | 0.056 | 0.049 | 0.43 | **NOT supported** |

Sign agreement with the true Δ: x2 original yes, native grid yes; x4 original yes, native grid yes

## S3. Acc25 against the true radius (Holm over x1/x2/x4)

| Factor | R2 | P1 | Mask reference | R2 - P1 | R2 better on | p | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| x1 | 0.662 | 0.592 | 0.605 | +0.069 | 80% | 0.00032 | 0.00032 | **SUPPORTED** |
| x2 | 0.532 | 0.315 | 0.573 | +0.218 | 100% | 1.9e-06 | 5.7e-06 | **SUPPORTED** |
| x4 | 0.543 | 0.400 | 0.753 | +0.143 | 90% | 9.5e-06 | 1.9e-05 | **SUPPORTED** |

## Secondary: bias (median estimate / true) and per-seed Acc25 against the truth

| Factor | R2 bias | P1 bias | R2 Acc25 (42/123/7) | P1 Acc25 (42/123/7) |
| --- | --- | --- | --- | --- |
| x1 | 1.134 | 1.168 | 0.632/0.653/0.699 | 0.511/0.586/0.681 |
| x2 | 1.203 | 1.412 | 0.539/0.525/0.533 | 0.058/0.275/0.610 |
| x4 | 1.058 | 1.315 | 0.599/0.536/0.493 | 0.227/0.391/0.582 |
