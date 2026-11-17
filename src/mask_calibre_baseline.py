"""Mask-width calibre baseline: radius measured from a segmentation network's predicted mask.

For each saved run in evaluation_outputs/csr_probs, the probability map is binarised at the
run's DRIVE-calibrated threshold (maximises mean Dice on DRIVE_test over the same grid as
dice_cal) and at 0.5. Radius is computed with the ground-truth definition
(max(EDT - 0.5, 0.5), propagated from the nearest skeleton pixel of the PREDICTED mask) and
scored at ground-truth centreline pixels with the same Acc25 / error / bias as
src.test_sge_temperature. Images are in pair_files order (001..N), matching that script's CSVs.
Pre-specified comparison: PRESPECIFICATION_LOG.md (2026-11-11).

    python -m src.mask_calibre_baseline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
from skimage.morphology import skeletonize
from tqdm import tqdm

from models.medial_frame import medial_geometry_targets
from src import common as ev
from src.common import read_gray
from src.common import THRESHOLDS, dice_at
from src.test_sge_temperature import GEO

ROOT = Path(__file__).resolve().parents[1]
CSR = ROOT / "evaluation_outputs" / "csr_probs"
CAL = ROOT / "evaluation_outputs" / "calibre"
DATASETS = ["DRIVE_test", "STARE", "CHASE_DB1", "HRF"]
UNSEEN = ["STARE", "CHASE_DB1", "HRF"]
RUNS = {"MFC_P1_warm_native": ("maskP1", 42), "MFC_P1_warm_s123_native": ("maskP1", 123),
        "MFC_P1_warm_s7_native": ("maskP1", 7), "MFC_R2_warm_native": ("maskR2", 42),
        "MFC_R2_warm_s123_native": ("maskR2", 123), "MFC_R2_warm_s7_native": ("maskR2", 7),
        "ExternalUNet_native": ("maskUNet", 42)}
ESTIMATOR_FILES = {42: "sge_R2_standalone_per_image.csv", 123: "sge_standalone_s123_per_image.csv",
                   7: "sge_standalone_s7_per_image.csv"}


def mask_radius(mask: np.ndarray) -> np.ndarray:
    """Radius part of models.medial_frame.medial_geometry_targets (identical definition)."""
    mask = mask.astype(bool)
    if mask.sum() < 3:
        return np.ones(mask.shape, np.float32)
    skeleton = skeletonize(mask)
    if skeleton.sum() == 0:
        return np.ones(mask.shape, np.float32)
    radius = np.maximum(ndi.distance_transform_edt(mask) - 0.5, 0.5)
    _, (iy, ix) = ndi.distance_transform_edt(~skeleton, return_indices=True)
    return radius[iy, ix].astype(np.float32)


def scores(ratio: np.ndarray) -> dict:
    return {"acc25": float(np.mean(np.abs(ratio - 1) <= 0.25)),
            "error": float(np.median(np.abs(ratio - 1))), "bias": float(np.median(ratio))}


def compute(runs) -> pd.DataFrame:
    config = ev.load_config(GEO[0])
    pairs = {}
    for d in DATASETS:
        spec = ev.resolve_datasets(config, [d])[d]
        pairs[d] = ev.pair_files(spec.image_dir, spec.mask_dir)
    gt, rows = {}, []
    for run in runs:
        family, seed = RUNS[run]

        def prob_path(d: str, i: int) -> Path:
            return CSR / run / d / "predictions" / "probability" / f"{i + 1:03d}.png"

        drive = [(read_gray(prob_path("DRIVE_test", i)).astype(np.float32) / 255.0,
                  ev.load_binary_mask(p.mask_path) > 0.5) for i, p in enumerate(pairs["DRIVE_test"])]
        t_cal = float(max(THRESHOLDS, key=lambda t: np.mean([dice_at(pr, g, t) for pr, g in drive])))
        for d in DATASETS:
            for i, pair in enumerate(tqdm(pairs[d], desc=f"{run} {d} t={t_cal:.2f}")):
                if (d, i) not in gt:
                    vessel = ev.load_binary_mask(pair.mask_path) > 0.5
                    skel = skeletonize(vessel)
                    gt[(d, i)] = (skel, medial_geometry_targets(vessel)["radius"][skel])
                skel, true = gt[(d, i)]
                prob = read_gray(prob_path(d, i)).astype(np.float32) / 255.0
                if prob.shape != skel.shape:
                    raise ValueError(f"{run} {d} {i + 1:03d}: probability {prob.shape} vs mask {skel.shape}")
                for rule, t in (("cal", t_cal), ("0.5", 0.5)):
                    ratio = mask_radius(prob >= t)[skel] / true
                    rows.append({"dataset": d, "i": i, "run": run, "model": family, "seed": seed,
                                 "rule": rule, "threshold": t, **scores(ratio)})
    return pd.DataFrame(rows)


def estimators() -> pd.DataFrame:
    frames = []
    for seed, name in ESTIMATOR_FILES.items():
        t = pd.read_csv(CAL / name)
        t = t[t.estimator.isin(["R2", "P1"]) & t.case.str.endswith(" x1")].copy()
        t["i"] = t.groupby(["case", "estimator"]).cumcount()
        t["dataset"] = t.case.str.replace(" x1", "", regex=False)
        t["model"], t["seed"], t["rule"], t["threshold"] = t.estimator, seed, "-", np.nan
        frames.append(t[["dataset", "i", "model", "seed", "rule", "threshold", "acc25", "error", "bias"]])
    return pd.concat(frames)


def holm(p: dict) -> dict:
    out, running = {}, 0.0
    for k, key in enumerate(sorted(p, key=p.get)):
        running = max(running, min(1.0, (len(p) - k) * p[key]))
        out[key] = running
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--output", default="mask_baseline.md")
    a = ap.parse_args()
    per_csv = CAL / (Path(a.output).stem + "_per_image.csv")
    masks = pd.read_csv(per_csv) if per_csv.exists() else compute(a.runs)
    masks.to_csv(per_csv, index=False)
    est = estimators()

    lines = ["# Calibre: R2 estimator vs width measured from predicted masks", "",
             "Acc25 at GT centreline pixels; each image averaged over the available seeds; paired Wilcoxon. "
             "Mask radius uses the GT radius definition on the binarised probability map. "
             "Pre-specified primary: R2 vs maskP1 at the calibrated threshold, Holm over the unseen datasets.", "",
             "## Mean Acc25 [per-seed means]", "",
             "| Method | Rule (thresholds) | " + " | ".join(DATASETS) + " |",
             "| --- | --- | " + " | ".join("---" for _ in DATASETS) + " |"]
    table = {(m, "-"): est[est.model == m] for m in ("R2", "P1")}
    for m in ("maskP1", "maskR2", "maskUNet"):
        for rule in ("cal", "0.5"):
            table[(m, rule)] = masks[(masks.model == m) & (masks.rule.astype(str) == rule)]
    avg = {k: v.groupby(["dataset", "i"]).acc25.mean() for k, v in table.items()}
    for (m, rule), v in table.items():
        cells = []
        for d in DATASETS:
            s = v[v.dataset == d].groupby("seed").acc25.mean()
            cells.append(f"{s.mean():.3f} [" + "/".join(f"{x:.3f}" for x in s.values) + "]")
        thr = sorted(set(np.round(v.threshold.dropna(), 2))) if rule == "cal" else []
        lines.append(f"| {m} | {rule}{' ' + str(thr) if thr else ''} | " + " | ".join(cells) + " |")

    for other, rule in [("maskP1", "cal"), ("maskR2", "cal"), ("maskUNet", "cal"),
                        ("maskP1", "0.5"), ("maskR2", "0.5"), ("maskUNet", "0.5"), ("P1", "-")]:
        res = {}
        for d in DATASETS:
            x, y = avg[("R2", "-")].loc[d], avg[(other, rule)].loc[d]
            diff = x - y
            res[d] = (diff.mean(), (diff > 0).mean(), wilcoxon(x, y).pvalue if np.any(diff != 0) else 1.0)
        adj = holm({d: res[d][2] for d in UNSEEN})
        primary = (other, rule) == ("maskP1", "cal")
        lines += ["", f"## R2 - {other} ({rule}){' (PRIMARY)' if primary else ''}", "",
                  "| Dataset | Mean diff | R2 better on | p | Holm p (unseen) |", "| --- | --- | --- | --- | --- |"]
        for d in DATASETS:
            mean, win, p = res[d]
            lines.append(f"| {d} | {mean:+.3f} | {win:.0%} | {p:.2g} | " + (f"{adj[d]:.2g} |" if d in adj else "- |"))
        if primary:
            beat = [d for d in UNSEEN if res[d][0] > 0 and adj[d] < 0.05]
            lines += ["", f"**Primary: R2 beats the mask baseline on {len(beat)}/3 unseen datasets "
                          f"({', '.join(beat) if beat else 'none'}).**"]
    out = CAL / a.output
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
