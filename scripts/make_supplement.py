"""Build the supplementary-material Markdown for the CMIG calibre paper from the saved per-image results.
Nothing is retrained or re-scored; every number is read from evaluation_outputs/calibre.

    python manuscript/make_supplement.py
Writes manuscript/CMIG_calibre_supplementary.md (build the Word file with build_docx.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "evaluation_outputs" / "calibre"
SEEDS = (42, 123, 7)
OUT = ROOT / "evaluation_outputs" / "supplementary_tables.md"

CASES = {  # result-file case name -> label used in the paper
    "DRIVE_test x1": "DRIVE ×1",
    "DRIVE_test x2 native": "DRIVE ×2, native grid",
    "DRIVE_test x2": "DRIVE ×2, enlarged mask",
    "DRIVE_test x4 native": "DRIVE ×4, native grid",
    "DRIVE_test x4": "DRIVE ×4, enlarged mask",
    "STARE x1": "STARE",
    "CHASE_DB1 x1": "CHASE_DB1 (2ndHO)",
    "CHASE_DB1 1stHO x1": "CHASE_DB1 (1stHO)",
    "HRF x1": "HRF",
    "FIVES_test x1": "FIVES",
    "FIVES_test x0.27 native": "FIVES ×0.27, native grid",
    "synthetic x1": "Synthetic ×1",
    "synthetic x2": "Synthetic ×2",
    "synthetic x4": "Synthetic ×4",
}


def f3(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pfmt(p):
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m}×10{str(int(e)).translate(sup)}"


def holm(p):
    out, running = {}, 0.0
    for k, key in enumerate(sorted(p, key=p.get)):
        running = max(running, min(1.0, (len(p) - k) * p[key]))
        out[key] = running
    return out


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def per_seed():
    """case -> model -> {seed: mean Acc25 over images}."""
    res = {}
    seu = pd.read_csv(CAL / "seu_baseline_per_image.csv")
    for (case, model, seed), g in seu.groupby(["case", "model", "seed"]):
        res.setdefault(case, {}).setdefault(model, {})[seed] = g.acc25.mean()
    for s in SEEDS:
        for name, prefix in ((f"sge_fives_s{s}_per_image.csv", ("R2", "P1")), (f"sge_R2ns_s{s}_per_image.csv", ("R2ns",)),
                             (f"sge_wide_s{s}_per_image.csv", ("R2w", "P1w"))):
            d = pd.read_csv(CAL / name)
            for (case, est), g in d.groupby(["case", "estimator"]):
                model = est.rsplit("_s", 1)[0]
                if model in prefix and (case not in res or model not in res[case] or s not in res[case][model]):
                    res.setdefault(case, {}).setdefault(model, {})[s] = g.acc25.mean()
    mask = pd.read_csv(CAL / "mask_threshold_validation_per_image.csv")
    name = {"DRIVE_test": "DRIVE_test x1", "STARE": "STARE x1", "CHASE_DB1": "CHASE_DB1 x1",
            "CHASE_DB1 1stHO": "CHASE_DB1 1stHO x1", "HRF": "HRF x1", "FIVES_test": "FIVES_test x1"}
    thr = {}
    for (case, seed), g in mask.groupby(["case", "seed"]):
        res.setdefault(name[case], {}).setdefault("Mask width", {})[seed] = g.acc25.mean()
        thr[seed] = g.threshold.iloc[0]
    return res, thr


def seed_rows(res, models):
    rows = []
    for case, label in CASES.items():
        if case not in res or not any(m in res[case] for m in models):
            continue
        row = [label]
        for m in models:
            v = res[case].get(m, {})
            vals = [v.get(s) for s in SEEDS]
            row += [f3(x) for x in vals] + [f3(np.mean(vals)) if all(x is not None for x in vals) else "–"]
        rows.append(row)
    return rows


def mask_alternatives():
    """R2 (seed-averaged) against width from each mask source and threshold rule, as in mask_baseline.py."""
    r2 = pd.read_csv(CAL / "seu_baseline_per_image.csv")
    r2 = r2[r2.model == "R2"].groupby(["case", "i"]).acc25.mean()
    m = pd.read_csv(CAL / "mask_baseline_per_image.csv")
    dsets = {"DRIVE_test": "DRIVE_test x1", "STARE": "STARE x1", "CHASE_DB1": "CHASE_DB1 x1", "HRF": "HRF x1"}
    sources = [("maskP1", "cal", "Transformer U-Net (as in the paper), test-set threshold"),
               ("maskP1", "0.5", "Transformer U-Net (as in the paper), threshold 0.5"),
               ("maskR2", "cal", "Second transformer U-Net, test-set threshold"),
               ("maskR2", "0.5", "Second transformer U-Net, threshold 0.5"),
               ("maskUNet", "cal", "Plain U-Net (one seed), threshold 0.5")]
    means, tests = [], []
    for model, rule, label in sources:
        g = m[(m.model == model) & (m.rule == rule)]
        thr = sorted(set(g.threshold))
        row = [label, ", ".join(f"{t:.2f}" for t in thr)]
        res = {}
        for d, case in dsets.items():
            per = g[g.dataset == d].groupby("i").acc25.mean()
            row.append(f3(per.mean()))
            x = r2.loc[case].loc[per.index].to_numpy()
            diff = x - per.to_numpy()
            res[d] = (diff.mean(), (diff > 0).mean(), wilcoxon(x, per.to_numpy()).pvalue)
        means.append(row)
        adj = holm({d: res[d][2] for d in ("STARE", "CHASE_DB1", "HRF")})
        for d in dsets:
            mdiff, better, p = res[d]
            tests.append([label if d == "DRIVE_test" else "", d.replace("_test", "").replace("CHASE_DB1", "CHASE_DB1 (2ndHO)"),
                          f"{mdiff:+.3f}", f"{better:.0%}", pfmt(p), pfmt(adj[d]) if d in adj else "–"])
    return means, tests


def main():
    res, thr = per_seed()
    R2 = {c: np.mean([res[c]["R2"][s] for s in SEEDS]) for c in res if "R2" in res[c] and len(res[c]["R2"]) == 3}
    # consistency checks against values reported in the paper
    for case, want in (("CHASE_DB1 x1", 0.414), ("HRF x1", 0.390), ("FIVES_test x1", 0.224), ("synthetic x2", 0.532),
                       ("FIVES_test x0.27 native", 0.416)):
        assert abs(R2[case] - want) < 0.0006, (case, R2[case], want)

    hdr = ["Test set"] + [f"{m} {s}" if s != "mean" else f"{m} mean" for m in ("R2", "P1") for s in (*SEEDS, "mean")]
    t1 = table(hdr, seed_rows(res, ("R2", "P1")))
    hdr = ["Test set"] + [f"{m} {s}" if s != "mean" else f"{m} mean" for m in ("SEU", "R2ns") for s in (*SEEDS, "mean")]
    t2 = table(hdr, seed_rows(res, ("SEU", "R2ns")))
    hdr = ["Test set"] + [f"{m} {s}" if s != "mean" else f"{m} mean" for m in ("R2w", "P1w") for s in (*SEEDS, "mean")]
    t3 = table(hdr, seed_rows(res, ("R2w", "P1w")))
    hdr = ["Test set"] + [f"Seed {s} ({thr[s]:.2f})" for s in SEEDS] + ["Mean"]
    t4 = table(hdr, [[r[0]] + r[1:] for r in seed_rows(res, ("Mask width",))])
    means, tests = mask_alternatives()
    t5 = table(["Mask source and threshold", "Threshold(s)", "DRIVE", "STARE", "CHASE_DB1 (2ndHO)", "HRF"],
               [["R2 (for reference)", "–"] + [f3(R2[c]) for c in ("DRIVE_test x1", "STARE x1", "CHASE_DB1 x1", "HRF x1")]] + means)
    t6 = table(["Mask source and threshold", "Test set", "R2 − mask", "R2 better on", "*p*", "Holm *p*"], tests)

    text = TEMPLATE.format(T1=t1, T2=t2, T3=t3, T4=t4, T5=t5, T6=t6)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")


TEMPLATE = """---
title: "Supplementary material: Scale-selective estimation of retinal vessel calibre with a shared-weight image pyramid"
---

# Contents

This document accompanies the main paper and is referred to there as the supplementary material. Section S1 gives the per-seed values behind the seed-averaged results of the paper. Section S2 reports the additional mask-width comparisons mentioned in Section 5.2 of the paper. Section S3 reports the equivariance checks of the scale-equivariant U-Net (SEU) and of R2 with random weights, including the filter depth that was considered and rejected (Sections 3.6 and 6.2 of the paper), and Section S4 the measurements of computational cost. Nothing in this document was retrained or re-scored for it; all values are read from the saved per-image results.

# S1. Per-seed results

Every learned estimator was trained with seeds 42, 123 and 7. In the paper, each image is averaged over the three seeds before images are compared. Tables S1–S4 give, for each seed, the mean Acc25 over the images of each test set; the mean over seeds equals the value reported in the paper. Real data are scored against the annotation-implied radius and synthetic data against the true radius. Test sets on which a model was not evaluated are omitted from its table, or marked –.

**Table S1.** R2 and the matched regressor P1, Acc25 per seed.

{T1}

**Table S2.** Scale-equivariant U-Net (SEU) and the non-shared pyramid (R2ns), Acc25 per seed. R2ns was evaluated only on the test sets of the weight-sharing ablation (Section 5.3 of the paper).

{T2}

**Table S3.** Wide-scale training (scale factors drawn from [0.7, 8.0]; Section 5.5 of the paper), R2w and P1w, Acc25 per seed.

{T3}

**Table S4.** Mask width (transformer U-Net masks, threshold chosen per seed on the four DRIVE validation images; thresholds in brackets), Acc25 per seed.

{T4}

The spread over seeds is considerable for some model and test-set pairs, for example P1 on the synthetic vessels at ×2 (0.058–0.610) and R2 on FIVES (0.126–0.318). For this reason the paper bases every comparison on seed-averaged values per image rather than on single runs.

# S2. Additional mask-width comparisons

Section 5.2 of the paper compares R2 with width measured from the masks of a transformer U-Net, binarised at the threshold that maximised the mean Dice coefficient on the DRIVE validation images. Tables S5 and S6 report the same comparison for three further choices: the threshold that maximised the mean Dice coefficient on the DRIVE **test** images (used in an earlier version of the analysis, before it was replaced by the validation-chosen threshold), the fixed threshold 0.5, and masks from two other segmentation networks, a second transformer U-Net (three seeds) and a plain U-Net (one seed). Radius is computed from every mask with the definition used for the reference (Section 3.4 of the paper), and scoring is identical to that of the paper. CHASE_DB1 is scored against its second observer (2ndHO). For the plain U-Net, the test-set threshold was 0.5, so only one row is given.

**Table S5.** Mean Acc25 of mask width for each mask source and threshold rule (each image averaged over the available seeds).

{T5}

**Table S6.** R2 versus mask width for each mask source and threshold rule, Acc25. *p*: paired Wilcoxon test; Holm correction over the three unseen datasets (DRIVE is in-domain and is not corrected).

{T6}

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
"""

if __name__ == "__main__":
    main()
