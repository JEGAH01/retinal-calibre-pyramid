---
title: "Supplementary material: Scale-selective estimation of retinal vessel calibre with a shared-weight image pyramid"
---

# Contents

This document accompanies the main paper and is referred to there as the supplementary material. Section S1 gives the per-seed values behind the seed-averaged results of the paper. Section S2 reports the additional mask-width comparisons mentioned in Section 5.2 of the paper. Section S3 reports the equivariance checks of the scale-equivariant U-Net (SEU) and of R2 with random weights, including the filter depth that was considered and rejected (Sections 3.6 and 6.2 of the paper), and Section S4 the measurements of computational cost. Nothing in this document was retrained or re-scored for it; all values are read from the saved per-image results.

# S1. Per-seed results

Every learned estimator was trained with seeds 42, 123 and 7. In the paper, each image is averaged over the three seeds before images are compared. Tables S1–S4 give, for each seed, the mean Acc25 over the images of each test set; the mean over seeds equals the value reported in the paper. Real data are scored against the annotation-implied radius and synthetic data against the true radius. Test sets on which a model was not evaluated are omitted from its table, or marked –.

**Table S1.** R2 and the matched regressor P1, Acc25 per seed.

| Test set | R2 42 | R2 123 | R2 7 | R2 mean | P1 42 | P1 123 | P1 7 | P1 mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DRIVE ×1 | 0.653 | 0.618 | 0.640 | 0.637 | 0.650 | 0.639 | 0.615 | 0.635 |
| DRIVE ×2, native grid | 0.503 | 0.528 | 0.471 | 0.501 | 0.378 | 0.435 | 0.384 | 0.399 |
| DRIVE ×2, enlarged mask | 0.396 | 0.383 | 0.403 | 0.394 | 0.393 | 0.440 | 0.460 | 0.431 |
| DRIVE ×4, native grid | 0.454 | 0.397 | 0.423 | 0.425 | 0.423 | 0.441 | 0.372 | 0.412 |
| DRIVE ×4, enlarged mask | 0.468 | 0.409 | 0.416 | 0.431 | 0.438 | 0.451 | 0.360 | 0.416 |
| STARE | 0.529 | 0.460 | 0.508 | 0.499 | 0.465 | 0.498 | 0.535 | 0.499 |
| CHASE_DB1 (2ndHO) | 0.424 | 0.455 | 0.363 | 0.414 | 0.348 | 0.377 | 0.283 | 0.336 |
| CHASE_DB1 (1stHO) | 0.397 | 0.400 | 0.375 | 0.390 | 0.365 | 0.361 | 0.349 | 0.358 |
| HRF | 0.415 | 0.416 | 0.339 | 0.390 | 0.311 | 0.383 | 0.363 | 0.352 |
| FIVES | 0.228 | 0.318 | 0.126 | 0.224 | 0.178 | 0.148 | 0.049 | 0.125 |
| FIVES ×0.27, native grid | 0.418 | 0.476 | 0.355 | 0.416 | 0.405 | 0.463 | 0.257 | 0.375 |
| Synthetic ×1 | 0.632 | 0.653 | 0.699 | 0.662 | 0.511 | 0.586 | 0.681 | 0.592 |
| Synthetic ×2 | 0.539 | 0.525 | 0.533 | 0.532 | 0.058 | 0.275 | 0.610 | 0.315 |
| Synthetic ×4 | 0.599 | 0.536 | 0.493 | 0.543 | 0.227 | 0.391 | 0.582 | 0.400 |

**Table S2.** Scale-equivariant U-Net (SEU) and the non-shared pyramid (R2ns), Acc25 per seed. R2ns was evaluated only on the test sets of the weight-sharing ablation (Section 5.3 of the paper).

| Test set | SEU 42 | SEU 123 | SEU 7 | SEU mean | R2ns 42 | R2ns 123 | R2ns 7 | R2ns mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DRIVE ×1 | 0.638 | 0.594 | 0.640 | 0.624 | 0.614 | 0.634 | 0.601 | 0.616 |
| DRIVE ×2, native grid | 0.519 | 0.519 | 0.528 | 0.522 | 0.433 | 0.455 | 0.412 | 0.433 |
| DRIVE ×2, enlarged mask | 0.395 | 0.385 | 0.397 | 0.393 | 0.440 | 0.431 | 0.484 | 0.451 |
| DRIVE ×4, native grid | 0.427 | 0.413 | 0.438 | 0.426 | 0.408 | 0.416 | 0.382 | 0.402 |
| DRIVE ×4, enlarged mask | 0.446 | 0.428 | 0.456 | 0.443 | 0.420 | 0.426 | 0.393 | 0.413 |
| STARE | 0.519 | 0.462 | 0.501 | 0.494 | 0.468 | 0.482 | 0.521 | 0.490 |
| CHASE_DB1 (2ndHO) | 0.381 | 0.439 | 0.364 | 0.395 | 0.386 | 0.400 | 0.264 | 0.350 |
| CHASE_DB1 (1stHO) | 0.408 | 0.386 | 0.397 | 0.397 | – | – | – | – |
| HRF | 0.388 | 0.424 | 0.392 | 0.401 | 0.371 | 0.382 | 0.360 | 0.371 |
| FIVES | 0.208 | 0.236 | 0.176 | 0.207 | – | – | – | – |
| Synthetic ×1 | 0.763 | 0.546 | 0.678 | 0.662 | – | – | – | – |
| Synthetic ×2 | 0.577 | 0.517 | 0.529 | 0.541 | – | – | – | – |
| Synthetic ×4 | 0.569 | 0.561 | 0.558 | 0.563 | – | – | – | – |

**Table S3.** Wide-scale training (scale factors drawn from [0.7, 8.0]; Section 5.5 of the paper), R2w and P1w, Acc25 per seed.

| Test set | R2w 42 | R2w 123 | R2w 7 | R2w mean | P1w 42 | P1w 123 | P1w 7 | P1w mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DRIVE ×1 | 0.557 | 0.571 | 0.617 | 0.582 | 0.558 | 0.626 | 0.583 | 0.589 |
| DRIVE ×2, native grid | 0.529 | 0.523 | 0.513 | 0.522 | 0.417 | 0.443 | 0.420 | 0.427 |
| DRIVE ×2, enlarged mask | 0.374 | 0.381 | 0.376 | 0.377 | 0.411 | 0.420 | 0.380 | 0.403 |
| DRIVE ×4, native grid | 0.473 | 0.447 | 0.435 | 0.452 | 0.423 | 0.419 | 0.322 | 0.388 |
| DRIVE ×4, enlarged mask | 0.464 | 0.456 | 0.449 | 0.456 | 0.445 | 0.433 | 0.362 | 0.413 |
| STARE | 0.456 | 0.434 | 0.475 | 0.455 | 0.488 | 0.445 | 0.449 | 0.461 |
| CHASE_DB1 (2ndHO) | 0.389 | 0.387 | 0.438 | 0.405 | 0.273 | 0.336 | 0.419 | 0.343 |
| HRF | 0.380 | 0.344 | 0.405 | 0.376 | 0.361 | 0.407 | 0.242 | 0.337 |
| FIVES | 0.325 | 0.377 | 0.363 | 0.355 | 0.183 | 0.140 | 0.313 | 0.212 |
| FIVES ×0.27, native grid | 0.358 | 0.415 | 0.435 | 0.403 | 0.270 | 0.437 | 0.437 | 0.381 |

**Table S4.** Mask width (transformer U-Net masks, threshold chosen per seed on the four DRIVE validation images; thresholds in brackets), Acc25 per seed.

| Test set | Seed 42 (0.56) | Seed 123 (0.58) | Seed 7 (0.60) | Mean |
| --- | --- | --- | --- | --- |
| DRIVE ×1 | 0.579 | 0.577 | 0.574 | 0.577 |
| STARE | 0.484 | 0.497 | 0.502 | 0.494 |
| CHASE_DB1 (2ndHO) | 0.329 | 0.344 | 0.332 | 0.335 |
| CHASE_DB1 (1stHO) | 0.359 | 0.372 | 0.364 | 0.365 |
| HRF | 0.298 | 0.290 | 0.286 | 0.291 |
| FIVES | 0.356 | 0.286 | 0.379 | 0.341 |

The spread over seeds is considerable for some model and test-set pairs, for example P1 on the synthetic vessels at ×2 (0.058–0.610) and R2 on FIVES (0.126–0.318). For this reason the paper bases every comparison on seed-averaged values per image rather than on single runs.

# S2. Additional mask-width comparisons

Section 5.2 of the paper compares R2 with width measured from the masks of a transformer U-Net, binarised at the threshold that maximised the mean Dice coefficient on the DRIVE validation images. Tables S5 and S6 report the same comparison for three further choices: the threshold that maximised the mean Dice coefficient on the DRIVE **test** images (used in an earlier version of the analysis, before it was replaced by the validation-chosen threshold), the fixed threshold 0.5, and masks from two other segmentation networks, a second transformer U-Net (three seeds) and a plain U-Net (one seed). Radius is computed from every mask with the definition used for the reference (Section 3.4 of the paper), and scoring is identical to that of the paper. CHASE_DB1 is scored against its second observer (2ndHO). For the plain U-Net, the test-set threshold was 0.5, so only one row is given.

**Table S5.** Mean Acc25 of mask width for each mask source and threshold rule (each image averaged over the available seeds).

| Mask source and threshold | Threshold(s) | DRIVE | STARE | CHASE_DB1 (2ndHO) | HRF |
| --- | --- | --- | --- | --- | --- |
| R2 (for reference) | – | 0.637 | 0.499 | 0.414 | 0.390 |
| Transformer U-Net (as in the paper), test-set threshold | 0.52, 0.56 | 0.568 | 0.482 | 0.329 | 0.283 |
| Transformer U-Net (as in the paper), threshold 0.5 | 0.50 | 0.556 | 0.467 | 0.320 | 0.273 |
| Second transformer U-Net, test-set threshold | 0.48, 0.56, 0.60 | 0.564 | 0.488 | 0.298 | 0.261 |
| Second transformer U-Net, threshold 0.5 | 0.50 | 0.550 | 0.470 | 0.288 | 0.251 |
| Plain U-Net (one seed), threshold 0.5 | 0.50 | 0.533 | 0.470 | 0.317 | 0.315 |

**Table S6.** R2 versus mask width for each mask source and threshold rule, Acc25. *p*: paired Wilcoxon test; Holm correction over the three unseen datasets (DRIVE is in-domain and is not corrected).

| Mask source and threshold | Test set | R2 − mask | R2 better on | *p* | Holm *p* |
| --- | --- | --- | --- | --- | --- |
| Transformer U-Net (as in the paper), test-set threshold | DRIVE | +0.069 | 95% | 3.8×10⁻⁶ | – |
|  | STARE | +0.017 | 70% | 0.070 | 0.070 |
|  | CHASE_DB1 (2ndHO) | +0.085 | 93% | 3.3×10⁻⁶ | 6.7×10⁻⁶ |
|  | HRF | +0.107 | 80% | 4.9×10⁻⁷ | 1.5×10⁻⁶ |
| Transformer U-Net (as in the paper), threshold 0.5 | DRIVE | +0.081 | 100% | 1.9×10⁻⁶ | – |
|  | STARE | +0.032 | 85% | 0.017 | 0.017 |
|  | CHASE_DB1 (2ndHO) | +0.094 | 93% | 2.8×10⁻⁶ | 5.5×10⁻⁶ |
|  | HRF | +0.117 | 82% | 8.0×10⁻⁸ | 2.4×10⁻⁷ |
| Second transformer U-Net, test-set threshold | DRIVE | +0.073 | 95% | 3.8×10⁻⁶ | – |
|  | STARE | +0.011 | 70% | 0.15 | 0.15 |
|  | CHASE_DB1 (2ndHO) | +0.116 | 93% | 1.9×10⁻⁷ | 3.7×10⁻⁷ |
|  | HRF | +0.129 | 84% | 3.4×10⁻⁸ | 1.0×10⁻⁷ |
| Second transformer U-Net, threshold 0.5 | DRIVE | +0.088 | 100% | 1.9×10⁻⁶ | – |
|  | STARE | +0.029 | 80% | 0.015 | 0.015 |
|  | CHASE_DB1 (2ndHO) | +0.126 | 93% | 1.9×10⁻⁷ | 3.7×10⁻⁷ |
|  | HRF | +0.139 | 84% | 8.0×10⁻⁹ | 2.4×10⁻⁸ |
| Plain U-Net (one seed), threshold 0.5 | DRIVE | +0.104 | 100% | 1.9×10⁻⁶ | – |
|  | STARE | +0.030 | 70% | 0.017 | 0.017 |
|  | CHASE_DB1 (2ndHO) | +0.097 | 96% | 2.2×10⁻⁸ | 6.7×10⁻⁸ |
|  | HRF | +0.075 | 80% | 6.8×10⁻⁷ | 1.4×10⁻⁶ |

R2 achieved the higher mean Acc25 against every mask source and threshold on every dataset. The differences were significant on CHASE_DB1 and HRF in every case. On STARE, the difference was significant against the plain U-Net, against both transformer U-Nets at threshold 0.5, and not significant against either transformer U-Net at its test-set threshold. With the validation-chosen threshold used in the paper (Table 4 of the paper), the STARE difference was also not significant. These results lead to the same conclusions as those in the paper.

# S3. Equivariance checks with random weights

Before SEU was trained, its implementation and its behaviour under the enlargements used in the scale tests were checked with random weights (fixed random seed 0), using the script `src/seu_checks.py`. No test data were scored.

**Implementation check.** For an exact scale-equivariant network, the responses at scale *k* + 1 of an image *E*, subsampled by 2, equal the responses at scale *k* of *E* blurred and subsampled by 2 (the semigroup action of Sangalli et al., 2022, and Worrall and Welling, 2019). On a 256-pixel crop of DRIVE test image 01, the relative error ‖*a* − *b*‖/‖*b*‖ between the two after the encoder was 0.017–0.169 over scales *k* = 0–4, which confirms that the implementation is correct. For the whole network the error was 0.113–0.406, which reflects the approximation introduced by bilinear upsampling in the decoder, as described by Sangalli et al. (2022).

**Filter depth along the scale axis.** A filter depth of two, which lets neighbouring scales interact, was also implemented (488,085 parameters). Because the coarsest scale has no coarser neighbour, the filter must be truncated there, and this truncation broke equivariance. After the encoder the error rose to 0.277–1.076 over scales, compared with 0.017–0.169 for depth one, and the error spread through the network. Depth one, which is the setting used in both experiments of Sangalli et al. (2022), was therefore adopted before any training took place. A sample-aligned upsampling, which satisfies the exact condition of the original paper, was also tried; it did not reduce the error for the whole network (0.44 against 0.41), and the bilinear upsampling of the original design was kept.

**Behaviour under the scale-test protocol.** Central crops of 256 pixels from DRIVE test images 01–03 were enlarged by factors of 2 and 4 with bilinear interpolation, as in the scale tests of the paper. The per-scale responses of the enlarged crop were area-averaged back to the native grid and compared with the responses of the native crop at the scale shifted by the enlargement (one and two octaves for SEU; two and four half-octave levels for R2). The error ‖*a* − *b*‖/‖*b*‖ was computed over the vessel band (the manual mask dilated by 3 pixels), and Table S7 gives its mean and maximum over the levels compared.

**Table S7.** Relative response error under bilinear enlargement, random weights (mean and maximum over the levels compared, three DRIVE crops).

| Model | Factor | Mean | Maximum |
| --- | --- | --- | --- |
| R2 | ×2 | 0.026 | 0.190 |
| R2 | ×4 | 0.030 | 0.180 |
| SEU (filter depth 1, used) | ×2 | 0.277 | 0.487 |
| SEU (filter depth 1, used) | ×4 | 0.393 | 0.606 |
| SEU (filter depth 2, rejected) | ×2 | 0.702 | 0.946 |
| SEU (filter depth 2, rejected) | ×4 | 0.762 | 0.987 |

Before training, R2 was thus much closer to exact equivariance under the enlargements used in the scale tests than SEU. As discussed in Section 6.2 of the paper, this difference did not translate into a difference in accuracy after training. Evaluating the scales of SEU one at a time gave outputs identical to joint evaluation (maximum absolute difference 0.0), which allowed whole images to be processed within the memory of the GPU.

# S4. Computational cost

Cost was measured in 32-bit floating point on a laptop GPU (NVIDIA RTX 3060, 6 GB), with untrained weights, which do not affect cost. A training step uses a batch of two 256-pixel patches (forward and backward pass); inference processes one whole image. Peaks above 6 GB spill into shared system memory under Windows, which slows SEU on HRF.

**Table S8.** Time and peak GPU memory. The first value is the measurement reported in the paper; the value in brackets is a repeat measurement on the same machine.

| Model | Training step | Peak | FIVES image (2048×2048) | Peak | HRF image (3504×2336) | Peak |
| --- | --- | --- | --- | --- | --- | --- |
| R2 | 0.25 s (0.09 s) | 0.5 GB | 0.4 s (0.4 s) | 2.2 GB | 4.9 s (6.3 s) | 4.3 GB |
| SEU | 0.23 s (0.20 s) | 1.6 GB | 5.3 s (6.6 s) | 4.0 GB | 18.8 s (23.6 s) | 7.9 GB |

Timings on a laptop GPU vary between runs, but in both measurements R2 was several times faster than SEU at inference. Every SEU layer operates on six full-resolution scales, whereas the levels of the R2 pyramid shrink with the image.

# References

Sangalli, M., Blusseau, S., Velasco-Forero, S., Angulo, J., 2022. Scale equivariant U-Net. In: Proc. British Machine Vision Conference (BMVC). https://doi.org/10.48550/arXiv.2210.04508.

Worrall, D.E., Welling, M., 2019. Deep scale-spaces: equivariance over scale. In: Advances in Neural Information Processing Systems 32 (NeurIPS). https://doi.org/10.48550/arXiv.1905.11697.
