"""FIVES calibre test (pre-specified in PRESPECIFICATION_LOG.md, 2026-11-11, before scoring).

FIVES_test: 200 images, 2048x2048, never used to train any estimator or segmentation network.
Primary: standalone R2 vs P1, Acc25 per image averaged over seeds 42/123/7, paired Wilcoxon.
Secondary: R2 vs P1 with FIVES brought to DRIVE field-of-view scale (x0.27, native-grid
scoring); R2 and P1 vs width measured from the transformer U-Net masks (each seed's DRIVE-calibrated
threshold, the same rule as src.mask_calibre_baseline), and the same comparisons per seed.

Inputs (scripts/reproduce.ps1):
  evaluation_outputs/calibre/sge_fives_s{seed}_per_image.csv          (R2_s{seed}, P1_s{seed})
  evaluation_outputs/csr_probs/MFC_P1_warm_s{seed}_FIVES/FIVES_test/predictions/probability/*.png

    python -m src.fives_calibre_analysis
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from skimage.morphology import skeletonize
from tqdm import tqdm

from models.medial_frame import medial_geometry_targets
from src import common as ev
from src.common import read_gray
from src.common import THRESHOLDS, dice_at
from src.mask_calibre_baseline import mask_radius, scores
from src.test_sge_temperature import GEO

ROOT = Path(__file__).resolve().parents[1]
CSR = ROOT / "evaluation_outputs" / "csr_probs"
CAL = ROOT / "evaluation_outputs" / "calibre"
SEEDS = [42, 123, 7]
DRIVE_RUN = {42: "MFC_P1_warm_native", 123: "MFC_P1_warm_s123_native", 7: "MFC_P1_warm_s7_native"}


def pairs(name: str):
    spec = ev.resolve_datasets(ev.load_config(GEO[0]), [name])[name]
    return ev.pair_files(spec.image_dir, spec.mask_dir)


def drive_threshold(seed: int) -> float:
    """Same rule as dice_cal: maximise mean Dice on DRIVE_test with this run's saved probabilities."""
    run = CSR / DRIVE_RUN[seed] / "DRIVE_test" / "predictions" / "probability"
    data = [(read_gray(run / f"{i + 1:03d}.png").astype(np.float32) / 255.0, ev.load_binary_mask(p.mask_path) > 0.5)
            for i, p in enumerate(pairs("DRIVE_test"))]
    return float(max(THRESHOLDS, key=lambda t: np.mean([dice_at(pr, g, t) for pr, g in data])))


def mask_scores() -> pd.DataFrame:
    out = CAL / "fives_mask_per_image.csv"
    if out.exists():
        return pd.read_csv(out)
    fives = pairs("FIVES_test")
    gt, rows = {}, []
    for seed in SEEDS:
        t_cal = drive_threshold(seed)
        prob_dir = CSR / f"MFC_P1_warm_s{seed}_FIVES" / "FIVES_test" / "predictions" / "probability"
        for i, pair in enumerate(tqdm(fives, desc=f"FIVES mask seed {seed} t={t_cal:.2f}")):
            if i not in gt:
                vessel = ev.load_binary_mask(pair.mask_path) > 0.5
                skel = skeletonize(vessel)
                gt[i] = (skel, medial_geometry_targets(vessel)["radius"][skel])
            skel, true = gt[i]
            prob = read_gray(prob_dir / f"{i + 1:03d}.png").astype(np.float32) / 255.0
            if prob.shape != skel.shape:
                raise ValueError(f"seed {seed} image {i + 1:03d}: probability {prob.shape} vs mask {skel.shape}")
            rows.append({"i": i, "model": "maskP1", "seed": seed, "threshold": t_cal,
                         **scores(mask_radius(prob >= t_cal)[skel] / true)})
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    return table


def estimator_scores(case: str = "FIVES_test x1") -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        t = pd.read_csv(CAL / f"sge_fives_s{seed}_per_image.csv")
        t = t[t.case == case].copy()
        t["model"] = t.estimator.str.split("_").str[0]
        t["i"] = t.groupby("estimator").cumcount()
        t["seed"] = seed
        frames.append(t[["i", "model", "seed", "acc25", "error", "bias"]])
    return pd.concat(frames)


def paired(a: pd.Series, b: pd.Series):
    d = a - b
    return d.mean(), (d > 0).mean(), (wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0)


def main() -> None:
    data = pd.concat([estimator_scores(), mask_scores()])
    n = data.groupby(["model", "seed"]).size()
    if not (n == 200).all():
        raise ValueError(f"Expected 200 images per model and seed, got:\n{n}")
    avg = data.groupby(["model", "i"]).acc25.mean()
    seed_means = data.groupby(["model", "seed"]).acc25.mean()
    bias = data.groupby(["model"]).bias.median()
    lines = ["# FIVES_test calibre (200 images, 2048x2048, unseen)", "",
             "Acc25 at GT centreline pixels, each image averaged over seeds 42/123/7, paired Wilcoxon.", "",
             "| Method | Acc25 | per seed (42/123/7) | median bias |", "| --- | --- | --- | --- |"]
    for m in ("R2", "P1", "maskP1"):
        lines.append(f"| {m} | {avg.loc[m].mean():.3f} | "
                     + "/".join(f"{seed_means.loc[(m, s)]:.3f}" for s in SEEDS) + f" | {bias.loc[m]:.3f} |")
    lines += ["", "| Comparison | Mean diff | first better on | p |", "| --- | --- | --- | --- |"]
    verdict = None
    for a, b in (("R2", "P1"), ("R2", "maskP1"), ("P1", "maskP1")):
        diff, win, p = paired(avg.loc[a], avg.loc[b])
        tag = " (PRIMARY)" if (a, b) == ("R2", "P1") else ""
        lines.append(f"| {a} - {b}{tag} | {diff:+.3f} | {win:.0%} | {p:.2g} |")
        if tag:
            verdict = diff > 0 and p < 0.05
    lines += ["", "## Per seed (R2 - P1, mean diff / p)", ""]
    for s in SEEDS:
        a = data[(data.model == "R2") & (data.seed == s)].set_index("i").acc25
        b = data[(data.model == "P1") & (data.seed == s)].set_index("i").acc25
        diff, _, p = paired(a, b.loc[a.index])
        lines.append(f"- seed {s}: {diff:+.3f} / p={p:.2g}")
    small = estimator_scores("FIVES_test x0.27 native")
    if len(small):
        sa = small.groupby(["model", "i"]).acc25.mean()
        diff, win, p = paired(sa.loc["R2"], sa.loc["P1"])
        sb = small.groupby("model").bias.median()
        lines += ["", "## Secondary: FIVES at DRIVE field-of-view scale (x0.27, native-grid scoring)", "",
                  f"- R2 {sa.loc['R2'].mean():.3f} (bias {sb.loc['R2']:.3f}) vs P1 {sa.loc['P1'].mean():.3f} "
                  f"(bias {sb.loc['P1']:.3f}): diff {diff:+.3f}, R2 better on {win:.0%}, p={p:.2g}"]
    lines += ["", f"**Primary (R2 > P1 on FIVES, p < 0.05): {'SUPPORTED' if verdict else 'NOT supported'}.**"]
    out = CAL / "fives_calibre.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
