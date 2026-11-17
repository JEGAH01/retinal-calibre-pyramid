# Pre-specification log (excerpt)

This file is an excerpt of the authors' dated working log. Each comparison in the paper was written down here,
with its primary test and decision rule, before the corresponding training or scoring was run; the result entry
follows. Entries are copied unchanged and in their original order, with two exceptions:

- two sentences about a separate segmentation study are replaced by a marked note;
- entries about experiments that are not part of the paper are not included.

Names used in the log:

| Log name | Meaning |
| --- | --- |
| R2, P1, R2ns, R2w, P1w, SEU | The estimators of the paper (`checkpoints/sge_<name>_seed<s>/sge.pth`) |
| MFC-P1-warm, `MFC_P1_warm_*` | The transformer U-Net whose masks give the mask-width baseline in the paper |
| MFC-R2-warm, `MFC_R2_warm_*` | The second transformer U-Net (alternative mask source, Supplementary Tables S5–S6) |
| ExternalUNet | The plain U-Net mask source (one seed, Supplementary Tables S5–S6) |
| `evaluate_topology_v2.resolve_datasets` | `src.common.resolve_datasets` in this repository |
| Table 3, "the paper" | Refer to the manuscript draft at the time of writing |

---

## Standalone estimator over three seeds (2026-11-10)

Standalone R2 (pyramid of shared U-Nets) vs P1 (plain U-Net), calibre Acc25, seeds 42/123/7, seed-averaged per image, paired Wilcoxon, Holm over the unseen sets.

| Case | R2 | P1 | diff | R2 better on | Holm p |
| --- | --- | --- | --- | --- | --- |
| DRIVE_test x1 | 0.637 | 0.635 | +0.003 | 45% | (p=0.43) |
| DRIVE_test x2 | 0.394 | 0.431 | -0.037 | 35% | (p=0.0037) |
| DRIVE_test x4 | 0.431 | 0.416 | +0.015 | 55% | (p=0.28) |
| STARE | 0.499 | 0.499 | -0.000 | 55% | 0.9 |
| CHASE_DB1 | 0.414 | 0.336 | +0.078 | 96% | 1.5e-07 |
| HRF | 0.390 | 0.352 | +0.038 | 91% | 7.6e-11 |

Per seed: CHASE favours R2 at all three seeds; HRF at seeds 42 and 123 (seed 7: P1 0.363 vs R2 0.339); STARE favours R2 only at seed 42.

What this means:
- The seed-42 standalone result (R2 ahead on all three unseen sets, HRF +0.104) was also partly a favourable draw. The three-seed effect is smaller: CHASE +0.078 and HRF +0.038 hold; STARE vanishes.
- The scale-equivariance story is not supported. On DRIVE upscaled x2, R2 is significantly *worse*. The gain on CHASE/HRF is real, but we can't credit it to equivariance.
- *[Sentence about a separate segmentation study omitted from this excerpt.]*
- Any further architectural change must be fixed in advance and run at three seeds from the start. Seed-42 pilots have misled twice now.

## DRIVE x2 loss traced to the ground-truth target, not the estimator (2026-11-10)

Diagnosis, using masks only (no model): the radius target `max(EDT - 0.5, 0.5)` is not scale-covariant for thin vessels. The median native DRIVE radius is 0.91 px. At the same place, the target computed on the x2-enlarged, re-thresholded mask is 0.90x of 2x the native target; at x4 it is 0.99x. A correctly scaling estimator therefore looks ~10% too wide at x2 only. This matches the pattern: R2 lost at x2 (bias 1.19) but not at x4.

Fix (evaluation only, same for every model): `src/test_sge_temperature.py --native-grid`. It area-averages the predicted log-radius map back to the native grid, divides by t, and scores against the native targets, so every factor uses the same ground truth. The original protocol is kept and still reported. Output: `evaluation_outputs/calibre/sge_drive_nativegrid.md`.

Three seeds, seed-averaged per image, paired Wilcoxon (20 images):

| Case | R2 | P1 | diff | R2 better on | p |
| --- | --- | --- | --- | --- | --- |
| x1 | 0.637 | 0.635 | +0.003 | 45% | 0.43 |
| x2 (original target) | 0.394 | 0.431 | -0.037 | 35% | 0.0037 |
| **x2 native grid** | **0.501** | **0.399** | **+0.102** | **100%** | **1.9e-06** |
| x4 (original target) | 0.431 | 0.416 | +0.015 | 55% | 0.28 |
| x4 native grid | 0.425 | 0.412 | +0.012 | 60% | 0.084 |

R2 wins x2 native at every seed (0.503/0.528/0.471 vs 0.378/0.435/0.384).

Caveats that must go in the thesis/paper:
- The native-grid protocol was introduced after seeing the x2 result. It is justified by a model-independent ground-truth check (above), but both protocols must be reported.
- x2 lies inside the training scale augmentation (0.7-3.2), so it tests scale consistency, not extrapolation. The out-of-range factor (x4) shows no significant difference.
- STARE is kept and reported as a null result (R2 = P1). Removing it after seeing the results would be selective reporting.
- *[Sentence about a separate segmentation study omitted from this excerpt.]*

## PRE-SPECIFIED: mask-width baseline (written 2026-11-11, before computing it)

Question: does the R2 estimator measure calibre better than simply measuring width from a segmentation network's predicted mask?

- Baseline: binarise the saved probability map at the run's DRIVE-calibrated threshold (the same rule as dice_cal), then compute radius with the same definition as the ground truth (`max(EDT - 0.5, 0.5)`, propagated from the nearest predicted-skeleton pixel). Score it at GT centreline pixels with Acc25, exactly like the estimators. Threshold 0.5 is reported as a secondary result.
- Segmentation sources: MFC-P1-warm (seeds 42/123/7), MFC-R2-warm (seeds 42/123/7), ExternalUNet (seed 42 only).
- **Primary comparison:** standalone R2 (3 seeds) vs the mask from MFC-P1-warm (3 seeds). Each image is averaged over seeds; paired Wilcoxon on STARE, CHASE_DB1 and HRF; Holm over the three. DRIVE_test x1 is reported but not tested for the claim (the threshold was calibrated on it).
- R2 "beats the mask baseline" on a dataset if the mean difference is > 0 with Holm p < 0.05. Every dataset is reported, whatever the result.

## RESULT: mask-width baseline (2026-11-11)

Source: `evaluation_outputs/calibre/mask_baseline.md`, `src/mask_calibre_baseline.py`. Sanity check: the GT mask used as the prediction gives Acc25 = 1.0.

Primary, R2 (3 seeds) minus the mask from MFC-P1-warm at its calibrated threshold (3 seeds), Acc25:

| Dataset | R2 | Mask | Diff | R2 better on | Holm p |
| --- | --- | --- | --- | --- | --- |
| DRIVE_test (not tested) | 0.637 | 0.568 | +0.069 | 95% | (p=3.8e-6) |
| STARE | 0.499 | 0.482 | +0.017 | 70% | 0.07 |
| CHASE_DB1 | 0.414 | 0.329 | **+0.085** | 93% | 6.7e-6 |
| HRF | 0.390 | 0.283 | **+0.107** | 80% | 1.5e-6 |

**R2 beats the mask baseline on 2/3 unseen datasets (CHASE_DB1, HRF), as pre-specified. STARE is positive but not significant.**

Secondary:
- Against the other mask sources (MFC-R2-warm, ExternalUNet) and at threshold 0.5, R2 is ahead everywhere. STARE is significant in 4 of these 5 secondary comparisons (Holm p 0.015-0.017); the exception is MFC-R2-warm at the calibrated threshold (p=0.15). Counting the primary comparison too, STARE is significant in 4 of 6.
- The standalone P1 regressor also beats the masks. The order on the unseen sets is R2 > P1 > mask: learned per-pixel regression beats mask-based width, and the pyramid adds a further gain on CHASE_DB1 and HRF.

## PRE-SPECIFIED: weight-sharing ablation (written 2026-11-11, before training)

Question: does the R2 gain come from **sharing one U-Net across pyramid levels**, or from the pyramid with scale selection alone?

- Variant R2ns: `config_sge_R2ns_seed{42,123,7}.yaml`. Each is identical to `config_sge_R2_pyrunet_seed*.yaml` except `share_weights: false`, which gives an independent LevelUNet per level (5,859,900 params vs 488,325). The equivariance loss is kept, so equivariance is encouraged but not built in. Training is the same: `src.pretrain_sge`, 5 epochs.
- Runs: `run_sharing_ablation.ps1` (train, then calibre with `--native-grid`, then the analysis).
- Test (`python -m src.sharing_ablation_analysis`): Acc25, each image averaged over seeds, paired Wilcoxon, R2 − R2ns. Holm over the three cases where R2 beat P1: **CHASE_DB1 ×1, HRF ×1, DRIVE_test ×2 native**. The other cases are reported uncorrected. The analysis was checked with P1 standing in for R2ns, and it reproduced the known R2 − P1 table exactly.
- Interpretation, fixed now:
  - Shared better on all three: weight sharing contributes to the gain.
  - No significant difference: the gain comes from the pyramid with scale selection, and sharing reaches the same accuracy with 12× fewer parameters.
  - Non-shared better: sharing costs accuracy. Report it.
- Confound to report: R2ns has 12× more parameters. A small R2ns would need its own pre-specification.

## RESULT: weight-sharing ablation (2026-11-11): sharing contributes to the gain

Run `run_sharing_ablation.ps1`. The training loss and DRIVE-validation radius error of R2ns look normal at every seed, so this is not a failed run. Output: `evaluation_outputs/calibre/sharing_ablation.md`.

| Case | R2 | R2ns | diff | R2 better on | p | Holm p | R2ns per seed (42/123/7) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CHASE_DB1 ×1 | 0.414 | 0.350 | **+0.064** | 89% | 1e-07 | **3.1e-07** | 0.386/0.400/0.264 |
| HRF ×1 | 0.390 | 0.371 | **+0.019** | 71% | 1.5e-05 | **1.5e-05** | 0.371/0.382/0.360 |
| DRIVE_test ×2 native | 0.501 | 0.433 | **+0.067** | 100% | 1.9e-06 | **3.8e-06** | 0.433/0.455/0.412 |
| DRIVE_test ×1 | 0.637 | 0.616 | +0.021 | 75% | 0.0056 | - | 0.614/0.634/0.601 |
| DRIVE_test ×4 native | 0.425 | 0.402 | +0.023 | 85% | 0.00013 | - | 0.408/0.416/0.382 |
| STARE ×1 | 0.499 | 0.490 | +0.009 | 55% | 0.33 | - | 0.468/0.482/0.521 |

**Verdict, by the rule fixed in advance: shared better on all three cases, so weight sharing contributes to the gain.** No case favours R2ns. Sharing gives better calibre with 12× fewer parameters (488k vs 5.86M).

Descriptive only (not pre-specified, no test): R2ns lies between P1 and R2 on CHASE_DB1 (0.336 < 0.350 < 0.414), HRF (0.352 < 0.371 < 0.390) and DRIVE ×2 native (0.399 < 0.433 < 0.501). That is consistent with both the pyramid and the sharing helping, but the R2ns − P1 comparison was not pre-specified and must not be reported as a test. R2ns is less stable across seeds (CHASE_DB1 seed 7: 0.264) and over-estimates radius on STARE and HRF (median bias up to 1.20), while R2's biases are closer to 1. Reading: without sharing, each level learns its own radius calibration, and the levels disagree on unseen data.

## PRE-SPECIFIED: FIVES calibre test (written 2026-11-11, before scoring)

Data: FIVES_test, 200 images at 2048×2048. It has never been used to train any estimator or MFC model. An earlier single-seed segmentation pilot trained on FIVES_train only. It is opt-in in `evaluate_topology_v2.resolve_datasets` and `test_sge_temperature` (`--cases FIVES_test`), so default runs are unchanged.

**Disclosure.** Before writing this, one smoke-test image (001) was scored with seed-42 R2 and P1 to check the pipeline:
- R2: Acc25 0.137, bias 0.51.
- P1: Acc25 0.124, bias 0.36.
- Median GT radius 6.2 px, against 0.9 px on DRIVE.

FIVES annotations are wide in pixels. The field of view is 2013 px (DRIVE 540, HRF 3265), and the vessels are also drawn thicker relative to the field of view. So native FIVES is a strong scale-extrapolation case, and both estimators may score low in absolute terms. No other FIVES result was seen.

Tests (`run_fives_calibre.ps1`, then `python -m src.fives_calibre_analysis`):
- **Primary:** standalone R2 vs P1 at native resolution (the same protocol as every other dataset). Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon over 200 images. Supported if the mean difference is > 0 with p < 0.05.
- **Secondary:**
  - FIVES brought to the DRIVE field-of-view scale (×0.27 = 540/2013, INTER_AREA downscale, native-grid scoring), R2 vs P1.
  - R2 and P1 vs width from MFC-P1-warm masks (3 seeds, each seed's DRIVE-calibrated threshold).
  - The same comparisons per seed.
  - Median bias.
- Every result is reported whatever it shows. FIVES is not dropped if it goes against R2.
- The analysis was checked on synthetic inputs (the expected differences were recovered).

## RESULT: FIVES calibre test (2026-11-12): primary SUPPORTED; R2 loses to mask width at native FIVES scale

Run `run_fives_calibre.ps1`. Output: `evaluation_outputs/calibre/fives_calibre.md`. 200 images per model and seed, as checked by the script.

| Method | Acc25 | per seed (42/123/7) | median bias |
| --- | --- | --- | --- |
| R2 | 0.224 | 0.228/0.318/0.126 | 0.573 |
| P1 | 0.125 | 0.178/0.148/0.049 | 0.368 |
| maskP1 (DRIVE-calibrated thresholds 0.52/0.56/0.56) | 0.349 | 0.367/0.291/0.389 | 0.658 |

| Comparison | Mean diff | first better on | p |
| --- | --- | --- | --- |
| **R2 − P1 (PRIMARY)** | **+0.099** | **98%** | **2e-34** |
| R2 − maskP1 | −0.125 | 10% | 2.9e-31 |
| P1 − maskP1 | −0.224 | 2% | 2.7e-34 |

- **Primary, by the rule fixed in advance: SUPPORTED.** R2 beats P1 at native FIVES resolution on 98% of images. Every seed agrees: +0.051, +0.170 and +0.077, all with p < 1e-20.
- **Secondary, DRIVE field-of-view scale (×0.27, native-grid):** R2 0.416 vs P1 0.375, a difference of +0.041 on 85% of images, p = 1e-24. Also supported.
- **Secondary, mask width: R2 LOSES on native FIVES** (−0.125, mask better on 90% of images). This must be reported. It reverses the order seen on CHASE_DB1 and HRF.
  - Likely reason: both estimators under-estimate badly (median bias R2 0.57, P1 0.37). FIVES vessels have a median radius of about 6 px, far beyond the DRIVE training radii of about 1 px. The training scale augmentation (0.7–3.2) does not reach them, and the 12-level pyramid at that size relies on its coarse levels.
  - A mask measures the width it segments, whatever the scale, so it does not extrapolate the way the estimators must.
  - This is a stated scope limit, not something to tune away after the fact. Any fix (wider scale augmentation, or training on FIVES_train) would be a new experiment with its own pre-specification.
- The mask baseline was not computed at ×0.27 (not pre-specified), so we cannot say whether R2 beats mask width once FIVES is brought to DRIVE scale.
- Absolute Acc25 on native FIVES is low for every method, and seed spread is large (R2 0.13–0.32). Report the per-seed values.

## PRE-SPECIFIED: wide-scale training test (written 2026-11-12, before any training)

Motivation: on native FIVES, both estimators under-estimate radius (median bias R2 0.57, P1 0.37), and width measured from masks beats R2. The working explanation is that FIVES radii lie outside the training scale range.

Choice of range, using training data only. Radii at the skeleton were measured on DRIVE_train (20 masks) and FIVES_train (30 masks). FIVES_test was not used.
- DRIVE_train radius p10/50/90/99: 0.5 / 0.9 / 2.5 / 3.7 px.
- FIVES_train radius p10/50/90/99: 2.3 / 5.5 / 10.5 / 15.5 px.
- FIVES is about 4–6× DRIVE, while the current augmentation stops at ×3.2.
- The new range is **[0.7, 8.0]**, log-uniform, still with probability 0.8. ×8 covers the FIVES p99 and stays within the pyramid's top level (about 22.6 px).

Models:
- `config_sge_R2w_seed{42,123,7}.yaml` and `config_sge_P1w_seed{42,123,7}.yaml`. Each is identical to the R2 or P1 config except `max_scale: 8.0`, and diff shows exactly one changed line.
- Training is the same as before: `src.pretrain_sge`, 5 epochs.
- A 40-step timing smoke test showed no slowdown. Its scores were not used.
- Runs: `run_wide_scale.ps1`. Analysis: `python -m src.wide_scale_analysis`, checked by feeding it the existing R2/P1 results under the new names; it reproduced the known tables exactly.

Tests: Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon.
- **Primary (Holm over A and B), native FIVES_test:**
  - A. R2w − maskP1 > 0. Does wide training close the gap to mask width? maskP1 is the existing per-image result in `fives_mask_per_image.csv`.
  - B. R2w − P1w > 0. Does the architecture advantage hold when both models get the same wide training?
- **Secondary:**
  - C. R2w − R2 on native FIVES.
  - D. Cost check: R2w − R2 on DRIVE_test ×1, DRIVE_test ×2 native, STARE, CHASE_DB1 and HRF, with Holm over the five. Any significant loss is a cost and is reported.
  - E. R2w − P1w on those five cases and on FIVES ×0.27 native, uncorrected.
  - Per-seed values and median bias.
- **Model-selection rule, fixed now:** R2w replaces R2 as the paper's main estimator only if A and B are both supported and D shows no significant loss. Otherwise R2 stays the main estimator, and R2w is reported as a separate experiment whatever it shows.
- Disclosure: this test was designed after seeing the FIVES result. It is a follow-up to a known failure, not an independent confirmation. FIVES_test is reused, so a pass shows the fix works on that set, not on new data.

## RESULT: wide-scale training test (2026-11-12): B supported, A not supported, costs elsewhere, so R2 stays the main estimator

Run `run_wide_scale.ps1`. Output: `evaluation_outputs/calibre/wide_scale.md`.

Primary, native FIVES_test, Holm over the two:

| Comparison | first | second | diff | first better on | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| A. R2w − maskP1 | 0.355 | 0.349 | +0.006 | 49% | 0.93 | **NOT supported** (a tie) |
| B. R2w − P1w | 0.355 | 0.212 | **+0.143** | 92% | 3.1e-32 | **SUPPORTED** |

Secondary:
- **C.** R2w − R2 on native FIVES: +0.131, better on 98% of images, p = 2e-34. Wide training fixes most of the FIVES failure:
  - median bias goes from 0.57 to 0.72;
  - every seed improves (0.325/0.377/0.363 against 0.228/0.318/0.126);
  - the gap to mask width goes from −0.125 to a tie.
- **D. Cost check, R2w − R2 (Holm over five):**
  - DRIVE ×1: −0.055 (Holm 2.3e-5), a significant loss.
  - STARE: −0.044 (1.9e-5), a significant loss.
  - CHASE_DB1: −0.009 (0.049), a significant loss.
  - HRF: −0.014 (0.31), no significant change.
  - DRIVE ×2 native: +0.021 (0.005), a significant gain.
  - Wide training spreads the training samples over a larger range of scales, so fewer of them are near ×1. That costs accuracy on the original resolutions.
- **E. R2w − P1w, uncorrected:**
  - Better on CHASE_DB1 (+0.062), HRF (+0.039), DRIVE ×2 native (+0.095) and FIVES ×0.27 native (+0.021).
  - Tied on DRIVE ×1 (−0.007) and STARE (−0.005).
  - **This repeats the R2 > P1 pattern under a second, independent training regime.**
- Seed spread: P1w is unstable (HRF 0.24–0.41, FIVES 0.14–0.31). R2w is tighter.

**Model selection, by the rule fixed in advance:** A failed and D shows significant losses, so **R2 stays the main estimator**. R2w is reported as a separate experiment.

What this shows:
1. The shared pyramid beats the matched U-Net under both training regimes and at every scale tested. B is the largest effect seen in the project.
2. The FIVES under-estimation was a training-range problem, and wide training fixes most of it.
3. No single training range is best everywhere. Extending the range trades accuracy at ×1 for accuracy at large radii.
4. Even when trained wide, the estimator only ties mask width on native FIVES; it does not beat it.

## PRE-SPECIFIED: second-observer test (written 2026-11-12, before scoring)

Reviewer concern: the reference radius comes from one annotator's mask, so the estimator may only reproduce that annotator's drawing style.

Data found while preparing this test:
- DRIVE_test has a second observer (`datasets/DRIVE/test/2nd_manual`, 20 images). All results so far use `1st_manual`, which is also what the models were trained on.
- **CHASE_DB1: every result so far used the SECOND observer (`*_2ndHO.png`), not the conventional first.** `prepare_datasets.py` paired each image with the `2ndHO` file. This was checked pixel by pixel: all 28 processed masks equal `2ndHO` (Dice 1.000), not `1stHO` (Dice about 0.75–0.82). The paper must state this. The usual reference, `1stHO`, is therefore the unused observer here.
- STARE's second observer (`labels-vk`) is not downloaded, so STARE is not part of this test.

Models: R2 and P1 (`checkpoints/sge_{R2,P1}_seed{42,123,7}/sge.pth`), and mask width from the MFC-P1-warm probability maps (3 seeds) at the DRIVE-calibrated threshold, exactly as in Table 3, plus threshold 0.5. Nothing is retrained.

Per image and per reference observer X, with the other observer called Y:
- **Full protocol:** each method scored against X at X's own centreline pixels (Acc25, error, bias), exactly as in the paper.
- **Shared-pixel protocol:** the pixels of X's centreline covered by Y's mask. On those pixels, Y's radius field is scored against X as a "human" method, together with R2, P1 and mask width. This gives the human agreement on the same pixels.
- Each image is averaged over seeds; paired Wilcoxon over images.

Tests:
- **Primary (Holm over the two), CHASE_DB1 with the first observer `1stHO` (the one not used so far), full protocol:**
  - O1. R2 − P1 > 0.
  - O2. R2 − maskP1 (calibrated threshold) > 0.
  - Supported if the difference is positive with Holm p < 0.05.
- **Secondary, uncorrected:**
  - O3. The same two comparisons on DRIVE_test with `2nd_manual`. In domain, R2 ≈ P1 is expected, as with `1st_manual`.
  - O4. Human ceiling: on the shared pixels, the Acc25 of Y vs X (human) next to R2, P1 and mask width vs X, for both observers on both datasets. R2 − human is reported with a Wilcoxon p. Descriptive: is the estimator's error within the disagreement between two experts?
  - O5. Annotator style: R2's Acc25 and bias against each observer in the full protocol. If R2 had learned `1st_manual`'s style, it would agree better with DRIVE's first observer than with its second, beyond what P1 and the human comparison show. Descriptive, no test.

Script: `src/second_observer.py`. Output: `evaluation_outputs/calibre/second_observer.md`.

## PRE-SPECIFIED: synthetic vessels of known width (written 2026-11-12, before scoring)

Reviewer concerns:
1. The reference is not the true width.
2. The native-grid protocol was chosen after seeing the ×2 results.

Synthetic images are the only setting where the true radius is known exactly.

Construction (`src/synthetic_calibre.py`, fixed seed 2026):
- **Backgrounds:** the 20 DRIVE_test images with their vessels removed by inpainting: OpenCV Telea inpainting, applied to the `1st_manual` mask dilated by 2 px. The models never trained on these images.
- **Vessels:** 20 smooth random curves per image inside the field of view. Each curve has a constant true radius r drawn log-uniformly from [0.5, 4.0] px, which covers the DRIVE_train range (p99 3.7 px).
- **Rendering:** each pixel is supersampled 8×8, and its coverage by the tube of radius r is computed. The coverage is blurred with a Gaussian PSF of σ = 0.5 px. The background is darkened by coverage × depth, where the per-channel depth is the median centreline/inpainted-background intensity ratio measured on DRIVE_test vessels with reference radius ≥ 2 px. Thin vessels therefore get lower contrast only through the blur, as in a camera.
- **"Annotation" mask:** pixels whose centre lies within r of the centreline. This mimics a perfect annotator on the pixel grid.
- **Scoring pixels:** the true centreline, sampled every 0.25 px and rounded to pixels. Pixels are excluded if they lie within 2·r_max + 2 px of another curve (crossings) or within 3 px of a curve end or of the FOV border.
- **Enlargement:** ×2 and ×4 are made exactly as in the real scale test, with cv2 INTER_LINEAR on the image and on the mask, re-thresholded at 0.5. The true radius at factor t is t·r, with centreline positions mapped as (x + 0.5)·t − 0.5.

Tests:
- **S1. Reference quality, no model:** at ×1, ×2 and ×4, the median ratio of each mask-derived reference to the true radius, by true-radius bin. References: (a) the native mask; (b) the enlarged, re-thresholded mask (original protocol); (c) the native mask times t (native-grid protocol). Descriptive. This checks the claimed ~10% artefact of the original protocol at ×2 against the exact truth.
- **S2. Protocol validity, primary for the protocol question:** per image at ×2, the Acc25 difference R2 − P1 is computed under the true radius (Δ_true), the original protocol (Δ_orig) and the native-grid protocol (Δ_native). Paired Wilcoxon on |Δ_native − Δ_true| against |Δ_orig − Δ_true| over the 20 images. Native-grid is supported as the better protocol if its error is smaller with p < 0.05. Also reported: whether each protocol's mean Δ has the same sign as the true mean Δ, and the same at ×4.
- **S3. Accuracy against the true radius (Holm over ×1, ×2, ×4):** R2 − P1 > 0 in Acc25 against the true radius, paired Wilcoxon over the 20 images, each image averaged over the three seeds. Secondary: mean Acc25 per method and factor, and the mask-derived reference (a) scored as if it were an estimator (its Acc25 against the truth).
- Disclosure: the synthetic vessels are simpler than real ones (constant radius, no central reflex, no pathology). The absolute Acc25 values are not comparable with the real datasets; the test is about relative behaviour and about the references.

## RESULT: second-observer test (2026-11-12): O1 supported, O2 NOT supported

Run `python -m src.second_observer`. Output: `evaluation_outputs/calibre/second_observer.md`. Check: the rows for the observers used so far reproduce the paper exactly (DRIVE obs1 R2/P1/mask 0.637/0.635/0.568; CHASE_DB1 2ndHO 0.414/0.336/0.329).

Primary, CHASE_DB1 against the first observer `1stHO`, Holm over two:

| Test | R2 | Other | Diff | R2 better on | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| O1. R2 − P1 | 0.390 | 0.358 | +0.032 | 86% | 2.6e-5 | **SUPPORTED** |
| O2. R2 − maskP1 | 0.390 | 0.356 | +0.034 | 54% | 0.099 | **NOT supported** |

- **R2 > P1 holds under both CHASE_DB1 observers**, but the gain is less than half as large with `1stHO` (+0.032 against +0.078).
- **R2 > mask width on CHASE_DB1 depends on the observer.** It is significant with `2ndHO` (+0.085) but not with `1stHO` (+0.034, p = 0.099, better on only 54% of images). The paper's claim "more accurate than mask width on CHASE_DB1" must be qualified.
- **O3, DRIVE `2nd_manual`:** the same pattern as `1st_manual`. R2 ≈ P1 (+0.004, p = 0.39), and R2 > mask width (+0.082, p = 1.9e-6).
- **O4, human ceiling** (shared pixels; human = the other observer's radius):
  - DRIVE: R2 is at human level (−0.009, p = 0.33 against obs1; +0.026, p = 0.008 against obs2).
  - CHASE_DB1: R2 is below human agreement against `1stHO` (0.412 vs 0.473, p = 0.001) and not significantly different against `2ndHO` (0.439 vs 0.466, p = 0.36).
  - P1 and mask width are well below human on CHASE_DB1 (0.33–0.36).
- **O5, annotator style:** no sign that R2 copies its training annotator. On DRIVE it agrees slightly *better* with `2nd_manual` (0.656) than with `1st_manual`, the observer it was trained on (0.637), with bias 1.02 against both. Mask width over-estimates on CHASE_DB1 (bias 1.18–1.22); R2's bias is 0.99 against `1stHO` and 0.89 against `2ndHO`.

## RESULT: synthetic known-width test (2026-11-12): S3 supported at all factors; S2 (native grid better) NOT supported

Run `python -m src.synthetic_calibre`, run twice with identical output. Output: `evaluation_outputs/calibre/synthetic_calibre.md`. 20 images, 46,096 scoring pixels at ×1.

- **S1, reference quality (no model).** Even a perfect pixel-grid annotation, scored with the paper's radius definition max(EDT − 0.5, 0.5), **under-estimates the true radius**: median ratio 0.805 overall, 0.67–0.70 for r < 1.5 px, 0.83–0.88 for r ≥ 1.5 px.
  - The original ×2 protocol is worse for the thinnest vessels (0.489 against 0.702 for r 0.5–1 px), slightly better for r 1–1.5, and similar above. Overall 0.777 against 0.805.
  - At ×4 the enlarged-mask reference is closer to the truth (0.859).
  - So the scale artefact exists, but it is concentrated in vessels below 1 px. The larger, scale-independent problem is a downward bias of about 20% in the reference itself.
- **S2, protocol validity (primary at ×2): NOT supported.** The R2 − P1 difference against the truth is +0.218. Both protocols give +0.165, and their errors are equal (0.064 vs 0.063, p = 0.35). Both have the correct sign at ×2 and ×4. **On synthetic data the native-grid protocol is not better than the original one.** Unlike real DRIVE ×2, where the original protocol reversed the result, here neither protocol does. The synthetic radii are spread more evenly (median 1.46 px against 0.91 px on DRIVE), so fewer pixels sit in the bin where the artefact is large.
- **S3, accuracy against the true radius (Holm over three): SUPPORTED at every factor.**

| Factor | R2 | P1 | Perfect-annotation reference | R2 − P1 | R2 better on | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| ×1 | 0.662 | 0.592 | 0.605 | +0.069 | 80% | 3.2e-4 |
| ×2 | 0.532 | 0.315 | 0.573 | +0.218 | 100% | 5.7e-6 |
| ×4 | 0.543 | 0.400 | 0.753 | +0.143 | 90% | 1.9e-5 |

- **Bias against the truth:** R2 1.13 / 1.20 / 1.06, P1 1.17 / 1.41 / 1.32. Both over-estimate synthetic vessels, and P1 much more so after enlargement.
- **Seed stability:** P1 is very unstable at ×2 (0.058 / 0.275 / 0.610). R2 is stable (0.539 / 0.525 / 0.533).
- Caveat, as pre-specified: synthetic vessels are simpler than real ones, so absolute values are not comparable with real datasets.

What this changes:
1. The scale advantage of R2 over P1 is now shown **against an exact truth**, with no reference-protocol choice involved, at ×1, ×2 and ×4. This is stronger evidence than the real-DRIVE ×2 test.
2. The claim that the native-grid protocol is the better protocol is **not supported**. It should be dropped as a contribution and kept only as a sensitivity analysis.
3. A new model-free finding: the distance-transform radius reference is biased low by about 20%, and by about 30% for vessels thinner than 1.5 px, even with a perfect annotation.
4. The CHASE_DB1 mask-width claim depends on the observer. R2 vs P1 holds with both observers.

## PRE-SPECIFIED: mask baseline with a validation-chosen threshold (written 2026-11-12, before computing)

Reviewer Weakness 5: the mask baseline's binarisation threshold was chosen to maximise mean Dice on the DRIVE **test** set (0.52 / 0.56 / 0.56 for seeds 42 / 123 / 7). The test set must not select a baseline's settings.

Fix:
- **Threshold:** for each seed, the threshold that maximises mean Dice on the segmentation network's own **validation** images. These are DRIVE training images 23, 35, 36 and 39, from `split_topology_samples` with validation_ratio 0.2 and split_seed 42, the same for all three seeds; the network never trained on them. The rule is the same as before: grid 0.20–0.80 in steps of 0.02, whole image, `dice_at`. Same networks (`checkpoints/mfc_P1_warm_seed{s}/best_model.pth`), native inference.
- **Re-score** mask width (maskP1) at that threshold from the saved probability maps on DRIVE_test, STARE, CHASE_DB1 (`2ndHO`, and `1stHO` as in the second-observer test), HRF and FIVES_test. The radius definition and Acc25 are unchanged.

Tests (the earlier pre-specified tests, repeated with the corrected baseline; estimator results unchanged):
- **M1 (the original primary):** R2 − maskP1, Holm over STARE, CHASE_DB1 (`2ndHO`) and HRF.
- **M2:** FIVES_test native, R2 − maskP1 and R2w − maskP1 (wide-scale test A), with the same Holm family as before (A with B; B is unchanged).
- **M3:** CHASE_DB1 `1stHO`, R2 − maskP1 (second-observer test O2), with the same Holm family as before (O1 with O2).
- **DRIVE_test:** R2 − maskP1, now a proper held-out comparison.

Decision, fixed now: the paper reports the validation-threshold mask baseline as **the** mask baseline in every table and in the text, whatever the result. The test-set-threshold numbers move to the supplementary material, and the limitation about the threshold is removed. If a seed's validation threshold equals its test threshold, its numbers do not change.

Script: `src/mask_threshold_validation.py`. Output: `evaluation_outputs/calibre/mask_threshold_validation.md`.

## RESULT: mask baseline with a validation-chosen threshold (2026-11-12): every verdict unchanged

Run `python -m src.mask_threshold_validation`, run twice with identical output. Output: `evaluation_outputs/calibre/mask_threshold_validation.md`.

- **Validation thresholds** (images 23, 35, 36, 39): 0.56 / 0.58 / 0.60 for seeds 42 / 123 / 7, against the old test-set thresholds 0.52 / 0.56 / 0.56. Validation Dice is 0.814–0.817.
- **Mask width changes little:** +0.006 to +0.012 on every dataset except FIVES (−0.008).

| Case | R2 | Mask (old) | Mask (new) | R2 − mask (new) | R2 better on | p |
| --- | --- | --- | --- | --- | --- | --- |
| DRIVE_test | 0.637 | 0.568 | 0.577 | +0.060 | 95% | 3.8e-6 |
| DRIVE_test, 2nd_manual | 0.656 | 0.574 | 0.584 | +0.072 | 100% | 1.9e-6 |
| STARE | 0.499 | 0.482 | 0.494 | +0.005 | 65% | 0.33 |
| CHASE_DB1 (2ndHO) | 0.414 | 0.329 | 0.335 | +0.079 | 93% | 3.3e-6 |
| CHASE_DB1 (1stHO) | 0.390 | 0.356 | 0.365 | +0.026 | 54% | 0.19 |
| HRF | 0.390 | 0.283 | 0.291 | +0.099 | 80% | 1.2e-6 |
| FIVES_test | 0.224 | 0.349 | 0.341 | −0.116 | 11% | 1e-30 |

The DRIVE `2nd_manual` row was computed afterwards, because the decision rule requires the new baseline in every table, including Table 8. It uses the same thresholds and rule (`mask_threshold_validation_drive_obs2.csv`).

Verdicts, all unchanged:
- **M1:** STARE not supported (Holm 0.33); CHASE_DB1 `2ndHO` supported (+0.079, Holm 6.7e-6); HRF supported (+0.099, Holm 3.5e-6).
- **M2:** wide-scale test A still NOT supported (R2w 0.355 vs mask 0.341, +0.014, 52% of images, Holm 0.16), and B unchanged. R2 vs mask on FIVES is −0.116.
- **M3:** O2 still NOT supported (+0.026, 54%, Holm 0.19), and O1 unchanged.

By the pre-specified decision, the paper now uses these numbers as the mask baseline everywhere, the old ones move to the supplementary material, and the threshold limitation is removed.

## PRE-SPECIFIED: classical scale-selection baseline (written 2026-11-12, before computing)

Reviewer concern: P1 shows that the pyramid helps against a single-scale regressor, but not that the learned design beats an established scale-aware method. The paper cites automatic scale selection (Lindeberg, 1998) as the idea behind R2, so the classical method is the natural baseline. It needs no training.

Method (`src/scale_selection_baseline.py`), "LSS":
- **Input:** green channel of the RGB image, inverted so that vessels are bright ridges, as float. No CLAHE.
- **Scales:** σ from 0.5 px to 22.6 px in steps of 2^(1/4) (23 scales), the same span as R2's pyramid.
- **Ridge strength per scale:** the γ-normalised Hessian response σ² · max(−λ_min, 0), where λ_min is the smaller eigenvalue of the Hessian of the Gaussian-smoothed image. With this normalisation (γ = 1 in t = σ²), a bar of half-width r gives its maximum response at σ = r.
- **Selected scale σ\* per pixel:** the arg-max over scales, refined by a parabola in ln σ through the maximum and its two neighbours.
- **Radius:** r̂ = k · σ\*. One constant k is calibrated on all 20 DRIVE **training** images (the images the learned models used) as exp(median of ln(r_ref / σ\*)) over their centreline pixels. A secondary, uncalibrated variant uses k = 1, the theoretical value for a bar.
- **Scoring:** exactly as for R2/P1. Acc25 at the reference centreline pixels. DRIVE ×2 and ×4 are scored with both protocols (enlarged mask, native grid). Synthetic images use the true radius as in `src/synthetic_calibre.py`. Deterministic, so there are no seeds.

Tests (R2 averaged over its three seeds, paired Wilcoxon over images):
- **L1, primary, real unseen data (Holm over four):** R2 − LSS on STARE, CHASE_DB1 (`2ndHO`), HRF and FIVES_test (native).
- **L2, primary, synthetic truth (Holm over three):** R2 − LSS against the true radius at ×1, ×2 and ×4.
- **Secondary, uncorrected:**
  - R2 − LSS on DRIVE_test ×1, on ×2 and ×4 under both protocols, and on CHASE_DB1 `1stHO`.
  - The uncalibrated LSS (k = 1).
  - LSS bias.
  - P1 − LSS, descriptive.
- **Decision, fixed now:** LSS is added to the paper as a baseline in every table where R2 and P1 appear, whatever the result. If LSS is better than R2 anywhere, the paper says so. LSS is scale-covariant by construction, so it may do well in the scale tests; that is part of what this test checks.

Output: `evaluation_outputs/calibre/scale_selection_baseline.md`.

## RESULT: classical scale-selection baseline (2026-11-13): L1 supported on 3 of 4, L2 supported; the uncalibrated variant beats R2 on FIVES

Run by the user (`python -m src.scale_selection_baseline`, twice, identical). Calibrated k = 0.467 on the 20 DRIVE training images. Output: `evaluation_outputs/calibre/scale_selection_baseline{.md,_per_image.csv}`.

**L1 (real unseen data, Holm over four):** R2 better on three of four.
| Case | R2 | LSS | R2 − LSS | R2 better on | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| STARE | 0.499 | 0.200 | +0.299 | 100% | 5.7e-06 | R2 better |
| CHASE_DB1 (2ndHO) | 0.414 | 0.351 | +0.062 | 79% | 6.8e-04 | R2 better |
| HRF | 0.390 | 0.301 | +0.089 | 89% | 1.7e-10 | R2 better |
| FIVES_test | 0.224 | 0.213 | +0.011 | 49% | 0.43 | no significant difference |

**L2 (synthetic, against the true radius, Holm over three):** supported at every factor. ×1 +0.488, ×2 +0.347, ×4 +0.386 (all 100% of images, Holm 5.7e-06).

**Secondary (uncorrected):**
- R2 − LSS is positive on every DRIVE case under both protocols (+0.196 to +0.447, all p = 1.9e-06) and on CHASE_DB1 `1stHO` (+0.130, p = 7.5e-09). P1 − LSS is positive everywhere except CHASE_DB1 2ndHO (−0.016, p = 0.14) and FIVES (−0.088, p = 2.3e-20).
- **Uncalibrated LSS (k = 1):** much worse than R2 on DRIVE, STARE, CHASE_DB1 and HRF (it doubles the radius there, median bias 1.7–2.4). **But it beats R2 on FIVES_test** (0.357 vs 0.224, R2 better on 14% of images, p = 9.4e-27) and on synthetic ×4 (0.593 vs 0.543, −0.050, p = 0.019). It is level at synthetic ×2 (0.552 vs 0.532, p = 0.28), and R2 still wins at synthetic ×1 (+0.142).
- Calibrated LSS bias: about 1.0 on DRIVE, STARE and HRF, 0.82 on CHASE_DB1 2ndHO, **0.65 on FIVES and 0.55–0.58 on the synthetic truth**.

**Not pre-specified (computed afterwards, descriptive):** on FIVES, R2w − LSS_k1 = −0.002 (0.355 vs 0.357, p = 0.76), R2w − LSS = +0.142 (p = 7e-20), P1w − LSS = −0.001 (p = 0.58).

**Interpretation.**
- The calibrated classical method loses to R2 on every DRIVE-convention dataset and against the truth, so the learned estimator adds more than scale selection does.
- The ratio between the selected scale and the radius is not a property of the method alone. It is about 2.1 against DRIVE-style annotations and about 1.2–1.4 against the synthetic truth and FIVES. One scalar, set by the annotation convention and the image blur, decides which LSS variant wins. This matches S1 (the annotation-implied radius is about 20% below the true width) and the FIVES widening finding.
- On FIVES, a classical method with the right scalar matches the target-scale-informed R2w and beats R2 by a wide margin. So R2's FIVES deficit is a calibration-convention problem, not a lack of scale coverage, and the paper must say plainly that a training-free method beats R2 there.
- Per the pre-specified decision, LSS (both variants) goes into every table with R2 and P1.

## PRE-SPECIFIED: learned scale-equivariant baseline SEU (written 2026-11-13, before any SEU training)

Reviewer concern: R2 should be compared with a standard scale-equivariant network, not only with a single-scale regressor and a classical method. Plan and reasoning: `PLAN_scale_equivariant_baseline.md`.

**Model** (`models/scale_equivariant_unet.py`, estimator type `se_unet`). An independent PyTorch implementation of the scale-equivariant U-Net of Sangalli et al. (2022, BMVC). The authors' code is TensorFlow with no licence, so it was read but not copied.
- Gaussian scale-space lifting, 6 octave scales, σ_k² = 0.25(4^k − 1).
- Semigroup scale cross-correlation: 3×3 filters, dilation 2^k at scale k, filter depth 1 along scale, as in both of the paper's experiments.
- Strided subsampling, bilinear upsampling.
- The P1/R2 U-Net layout and widths, so it has **488,325 parameters, the same as R2**.
- Readout: R2's soft-argmax over scales, not the paper's max projection, with a ±0.5-octave correction. Radius span 0.35–22.6 px, as for R2.
- In evaluation the scales run one at a time. This gives identical output (difference 0.0) and lets whole images run without tiling, as for R2.

**Design changes made during the pre-training checks** (random weights, no data scores; `python -m src.seu_checks`):
- **Filter depth 2 (scales interacting) was dropped.** The missing coarser scale at the top of the scale axis broke equivariance and spread through the U-Net. Under the exact semigroup action the error after the encoder was 0.28–1.08 per scale with depth 2, against 0.02–0.17 with depth 1. Under the scale-test protocol it was 0.70 (×2) and 0.76 (×4), against 0.28 and 0.39 with depth 1.
- **Sample-aligned upsampling was tried and dropped.** It is the condition in the paper's eq. 5, but it did not reduce the error (whole network 0.44 vs 0.41), so the paper's own bilinear upsampling is kept.
- **Tiling was dropped.** Per-scale evaluation fits FIVES (4.0 GB) and HRF (7.9 GB peak, spilling into shared memory) whole.

**Check results (final model, random weights):**
- Exact semigroup action: after the encoder, error 0.02–0.17 per scale, so the implementation is correct. The whole network has 0.11–0.41, the upsampling approximation the paper describes (its Section 4.2 and Appendix B).
- Scale-test protocol (bilinear ×2/×4 enlargement): mean response error **SEU 0.28 / 0.39**, against **R2 0.026 / 0.030**. So before training, R2 is much closer to equivariant under the enlargements used in our scale tests.
- Cost (fp32, RTX 3060 laptop, 6 GB):
  | | Train step | FIVES image | HRF image |
  | --- | --- | --- | --- |
  | SEU | 0.23 s, 1.6 GB | 5.3 s, 4.0 GB | 18.8 s, 7.9 GB |
  | R2 | 0.25 s, 0.5 GB | 0.4 s, 2.2 GB | 4.9 s, 4.3 GB |

**Training:** `config_sge_SEU_seed{42,123,7}.yaml`, each identical to `config_sge_R2_pyrunet_seed*.yaml` except the estimator line. `src.pretrain_sge`, 5 epochs, default learning rate, as R2. The equivariance loss (t = 2^±0.5, weight 0.1) is kept exactly as for R2. No tuning. If a seed diverges (NaN, or validation radius accuracy at t = 1 below that before training), that is reported, and one re-run with learning rate 1e-3 is allowed for all three seeds, judged on DRIVE validation patches only.

**Tests** (Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon; scored by the existing pipeline exactly as R2/P1):
- **E1, primary, real unseen data (Holm over four):** R2 − SEU on STARE, CHASE_DB1 (`2ndHO`), HRF and FIVES_test (native).
- **E2, primary, synthetic truth (Holm over three):** R2 − SEU against the true radius at ×1, ×2, ×4.
- **Secondary, uncorrected:**
  - SEU − P1 on the same cases.
  - R2 − SEU on DRIVE ×1, and on ×2/×4 under both protocols.
  - CHASE_DB1 `1stHO`.
  - Bias.
  - Spread over seeds.
  - Inference time.
- **Decision, fixed now:**
  - SEU goes into every table with R2 and P1, whatever the result.
  - **R2 better on E1 and E2:** contribution 1 stands.
  - **Level (no significant difference):** the paper says the shared pyramid matches a scale-equivariant U-Net of the same size at lower inference cost, and contribution 1 is reworded to that.
  - **SEU better anywhere:** the paper says so plainly, and contribution 1 is reframed as one scale-equivariant design among others.

## RESULT: learned scale-equivariant baseline SEU (2026-11-16): mixed. R2 better on CHASE_DB1 and FIVES, SEU better on HRF, level on STARE and the synthetic truth

Run by the user (`run_seu_baseline.ps1`, 2026-11-15/16). Output: `evaluation_outputs/calibre/seu_baseline{.md,_per_image.csv}`, `sge_SEU_s{42,123,7}.md`.

**Training record.**
- All three seeds trained without divergence (validation radius error at t = 1: 0.211 / 0.200 / 0.207 against 0.51–0.75 before training), so the learning-rate rerun allowed by the pre-specification was not used.
- **Seed 123 was trained twice with the identical command and config.** The second run overwrote the first checkpoint, and the second run is the one scored. The two runs differ only through GPU non-determinism (validation error at t = 1: 0.191 first, 0.200 second). No test score was computed for the first run, so the choice could not have been influenced by results.
- Scoring used the same pipeline as R2/P1, including the 256 px tiles with 64 px overlap that every learned estimator goes through in `tiled_radius`. (The "no tiling" in the pre-specification refers to the model's own optional internal tiling, which was not needed.)

**E1 (real unseen data, Holm over four):**
| Case | R2 | SEU | R2 − SEU | R2 better on | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| STARE | 0.499 | 0.494 | +0.005 | 50% | 0.41 | no significant difference |
| CHASE_DB1 (2ndHO) | 0.414 | 0.395 | +0.019 | 75% | 0.0016 | R2 better |
| HRF | 0.390 | 0.401 | −0.011 | 38% | 0.027 | **SEU better** |
| FIVES_test | 0.224 | 0.207 | +0.017 | 70% | 3.6e-10 | R2 better |

**E2 (synthetic, against the true radius, Holm over three):** no significant difference at any factor. ×1 −0.001 (0.99), ×2 −0.009 (0.95), ×4 −0.020 (0.12). SEU's means are equal or slightly higher.

**Secondary (uncorrected):**
- R2 − SEU: DRIVE ×1 +0.013 (p = 4.8e-05). DRIVE ×2, native grid −0.022 (1.9e-05); ×2, enlarged mask +0.001 (1.0). DRIVE ×4, native grid −0.001 (0.73); ×4, enlarged mask −0.013 (4.8e-04).
- **CHASE_DB1 `1stHO`: −0.007 (p = 0.14).** So, as with mask width, R2's CHASE_DB1 advantage over SEU holds against one observer only.
- DRIVE 2nd observer: SEU 0.641 vs R2 0.656. FIVES ×0.27, native grid: SEU 0.377 vs R2 0.416 (+0.040, R2 better on 88% of images, p = 4e-28).
- **SEU − P1:** positive on CHASE_DB1 under both observers (+0.059, +0.039), HRF (+0.049), FIVES (+0.082), DRIVE ×2 native grid (+0.123), DRIVE ×4 (+0.014 to +0.027) and synthetic at every factor (+0.070, +0.226, +0.163). All p < 0.03 except DRIVE ×4 native grid (0.07). Level on STARE. Negative on DRIVE ×1 (−0.011, p = 0.017) and on DRIVE ×2 enlarged mask (−0.039).
- Median bias, R2 / SEU: about the same everywhere (HRF 0.99 / 1.06, CHASE_DB1 2ndHO 0.89 / 0.85, FIVES 0.55 / 0.52, synthetic ×4 1.06 / 1.12).
- **Spread over seeds (max − min of the per-seed means):** mixed. SEU varies less than R2 on most real cases (FIVES 0.060 vs 0.192, HRF 0.036 vs 0.077, DRIVE ×2/×4 native grid 0.009–0.025 vs 0.057), but more on synthetic ×1 (0.217 vs 0.067) and ×2 (0.060 vs 0.014). On FIVES ×0.27, native grid, the per-seed medians were 0.310 / 0.493 / 0.326.
- Inference cost (`src.seu_checks`, whole image, fp32, RTX 3060 laptop): FIVES 2048² SEU 5.3 s vs R2 0.4 s; HRF SEU 18.8 s vs R2 4.9 s. In the tiled scoring run, SEU took 15–18 s per HRF image and 8.6–11.7 s per native FIVES image.

**Decision (as fixed in advance).** SEU is better on one primary case (HRF), so the "SEU better anywhere" rule applies:
- The paper says so plainly.
- Contribution 1 is reframed as one scale-equivariant design among others.
- SEU goes into every table with R2 and P1.

**Interpretation.**
- A scale-equivariant U-Net of the same size and readout recovers most of R2's advantage over P1. So the gain comes from building scale structure into the estimator, not from the pyramid specifically.
- Between the two designs, the results are close and split:
  - R2 wins clearly only on native FIVES and, against one observer, on CHASE_DB1.
  - SEU wins narrowly on HRF.
  - Against the true radius they are level.
- R2's practical advantages:
  - It is 4–13× faster at inference.
  - It covers the scale range in half-octave steps, where SEU must use octaves.
- Neither design is consistently more stable across seeds.
- Before training, R2 was ten times closer to exactly equivariant under the bilinear enlargements of the scale tests (0.026 vs 0.28). After training this difference did not show up as an accuracy difference on the scale tests.
