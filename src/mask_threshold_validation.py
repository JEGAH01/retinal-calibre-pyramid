"""Mask-width baseline with a validation-chosen threshold (pre-specified in PRESPECIFICATION_LOG.md,
2026-11-12, before computing).

The earlier mask baseline binarised the transformer U-Net probability maps at the threshold that maximised
mean Dice on the DRIVE TEST set. Here the threshold maximises mean Dice on the segmentation network's
own validation images (DRIVE training 23, 35, 36, 39: split_topology_samples, ratio 0.2, split_seed 42),
same grid and rule. Mask width is re-scored on every dataset; the estimators' results are unchanged.

    python -m src.mask_threshold_validation            # GPU for the 4 x 3 validation maps, then CPU
    python -m src.mask_threshold_validation --analyse-only
Output: evaluation_outputs/calibre/mask_threshold_validation{.md,_per_image.csv}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from skimage.morphology import skeletonize
from tqdm import tqdm

from models.medial_frame import medial_geometry_targets
from src import common as ev
from src.common import read_gray
from src.common import THRESHOLDS, dice_at
from src.common import Segmenter, load_model
from src.fives_calibre_analysis import pairs
from src.mask_calibre_baseline import holm, mask_radius, scores
from src.second_observer import observer_files
from src.topology_datasets import load_topology_samples, split_topology_samples
from src.wide_scale_analysis import load as load_estimators

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "evaluation_outputs" / "calibre"
CSR = ROOT / "evaluation_outputs" / "csr_probs"
SEEDS = (42, 123, 7)
RUNS = {42: "MFC_P1_warm_native", 123: "MFC_P1_warm_s123_native", 7: "MFC_P1_warm_s7_native"}
CASES = ["DRIVE_test", "STARE", "CHASE_DB1", "CHASE_DB1 1stHO", "HRF", "FIVES_test"]
UNSEEN = ["STARE", "CHASE_DB1", "HRF"]
OUT_NAME = "mask_threshold_validation"


def validation_samples(config: dict):
    d, top = config["dataset"], config["topology"]
    samples = load_topology_samples(image_dir=d["train_dataset"]["image_dir"], mask_dir=d["train_dataset"]["mask_dir"],
                                    topology_root=top["target_root"])
    _, val = split_topology_samples(samples, float(d.get("validation_ratio", 0.2)), int(d.get("split_seed", 42)))
    return val


def validation_threshold(seed: int) -> tuple[float, list[str], float]:
    """Threshold maximising mean Dice on the validation images; probability maps are cached."""
    cfg = "configs/data.yaml"  # DRIVE training split (same for every seed)
    config = ev.load_config(cfg)
    val = validation_samples(config)
    cache = CSR / f"MFC_P1_warm_s{seed}_DRIVEval"
    data, segment = [], None
    for s in val:
        path = cache / f"{s.image_path.stem}.png"
        if not path.exists():
            if segment is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                c, model = load_model(cfg, f"checkpoints/mfc_P1_warm_seed{seed}/best_model.pth", device)
                segment = Segmenter(c, model, device, 2, bool(c.get("hardware", {}).get("mixed_precision", True)))
            image = ev.load_rgb_image(s.image_path)
            ev.save_gray(path, segment(image, 1.0, image.shape[:2]))
        data.append((read_gray(path).astype(np.float32) / 255.0, ev.load_binary_mask(s.mask_path) > 0.5))
    t = float(max(THRESHOLDS, key=lambda x: np.mean([dice_at(p, g, x) for p, g in data])))
    return t, [s.image_path.name for s in val], float(np.mean([dice_at(p, g, t) for p, g in data]))


def reference(case: str):
    """(probability-map dataset name, list of reference masks) in pair_files order."""
    if case == "CHASE_DB1 1stHO":
        ps = pairs("CHASE_DB1")
        return "CHASE_DB1", [f[0] for f in observer_files("CHASE_DB1", ps)]
    return case, [p.mask_path for p in pairs(case)]


def prob_path(seed: int, dataset: str, i: int) -> Path:
    run = f"MFC_P1_warm_s{seed}_FIVES" if dataset == "FIVES_test" else RUNS[seed]
    return CSR / run / dataset / "predictions" / "probability" / f"{i + 1:03d}.png"


def compute(thr: dict[int, float]) -> pd.DataFrame:
    rows = []
    for case in CASES:
        dataset, masks = reference(case)
        for i, m in enumerate(tqdm(masks, desc=case)):
            vessel = ev.load_binary_mask(m) > 0.5
            skel = skeletonize(vessel)
            true = medial_geometry_targets(vessel)["radius"][skel]
            for s in SEEDS:
                prob = read_gray(prob_path(s, dataset, i)).astype(np.float32) / 255.0
                rows.append({"case": case, "i": i, "seed": s, "threshold": thr[s],
                             **scores(mask_radius(prob >= thr[s])[skel] / true)})
    return pd.DataFrame(rows)


def old_mask() -> pd.DataFrame:
    """Mask baseline at the test-set threshold, as reported so far."""
    a = pd.read_csv(CAL / "mask_baseline_per_image.csv")
    a = a[(a.model == "maskP1") & (a.rule.astype(str) == "cal")].rename(columns={"dataset": "case"})
    b = pd.read_csv(CAL / "fives_mask_per_image.csv").assign(case="FIVES_test")
    c = pd.read_csv(CAL / "second_observer_per_image.csv")
    c = c[(c.dataset == "CHASE_DB1") & (c.reference == "obs1") & (c.protocol == "full") & (c.method == "maskP1")]
    c = c.assign(case="CHASE_DB1 1stHO")
    return pd.concat([x[["case", "i", "seed", "acc25"]] for x in (a, b, c)])


def estimators() -> pd.DataFrame:
    e = load_estimators()
    e = e[e.case.str.endswith(" x1") & e.model.isin(["R2", "P1", "R2w", "P1w"])].copy()
    e["case"] = e.case.str.replace(" x1", "", regex=False)
    so = pd.read_csv(CAL / "second_observer_per_image.csv")
    so = so[(so.dataset == "CHASE_DB1") & (so.reference == "obs1") & (so.protocol == "full") & so.method.isin(["R2", "P1"])]
    so = so.assign(case="CHASE_DB1 1stHO", model=so.method)
    return pd.concat([e[["case", "model", "seed", "i", "acc25"]], so[["case", "model", "seed", "i", "acc25"]]])


def test(x: pd.Series, y: pd.Series):
    d = x - y
    return x.mean(), y.mean(), d.mean(), (d > 0).mean(), (wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0)


def analyse(new: pd.DataFrame, thr: dict, val_info: dict) -> str:
    est = estimators().groupby(["case", "model", "i"]).acc25.mean()
    new_avg = new.groupby(["case", "i"]).acc25.mean()
    old_avg = old_mask().groupby(["case", "i"]).acc25.mean()
    m = lambda model, case: est.loc[(case, model)]
    L = ["# Mask baseline with a validation-chosen threshold", "",
         "Pre-specified in PRESPECIFICATION_LOG.md (2026-11-12). Acc25, each image averaged over seeds 42/123/7, "
         "paired Wilcoxon over images.", "",
         "## Thresholds", "", "| Seed | Validation images | Validation Dice | Validation threshold | Old (DRIVE test) threshold |",
         "| --- | --- | --- | --- | --- |"]
    old_thr = {42: 0.52, 123: 0.56, 7: 0.56}
    for s in SEEDS:
        names, dice = val_info[s]
        L.append(f"| {s} | {', '.join(names)} | {dice:.3f} | {thr[s]:.2f} | {old_thr[s]:.2f} |")

    L += ["", "## Mask width, old vs new threshold, and R2", "",
          "| Case | R2 | Mask (old, test thr.) | Mask (new, val thr.) | R2 - mask new | R2 better on | p |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    res = {}
    for case in CASES:
        r = test(m("R2", case), new_avg.loc[case])
        res[case] = r
        L.append(f"| {case} | {r[0]:.3f} | {old_avg.loc[case].mean():.3f} | {r[1]:.3f} | {r[2]:+.3f} | {r[3]:.0%} | {r[4]:.2g} |")

    adj = holm({c: res[c][4] for c in UNSEEN})
    L += ["", "## M1 (original primary): R2 - mask, Holm over STARE, CHASE_DB1 (2ndHO), HRF", "",
          "| Case | Diff | Holm p | Verdict |", "| --- | --- | --- | --- |"]
    for c in UNSEEN:
        ok = res[c][2] > 0 and adj[c] < 0.05
        L.append(f"| {c} | {res[c][2]:+.3f} | {adj[c]:.2g} | {'**SUPPORTED**' if ok else '**NOT supported**'} |")

    a = test(m("R2w", "FIVES_test"), new_avg.loc["FIVES_test"])
    b = test(m("R2w", "FIVES_test"), m("P1w", "FIVES_test"))
    adj2 = holm({"A": a[4], "B": b[4]})
    L += ["", "## M2: FIVES_test native (wide-scale family: A with B)", "",
          "| Test | First | Second | Diff | First better on | Holm p | Verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in (("A. R2w - mask", a), ("B. R2w - P1w (unchanged)", b)):
        ok = r[2] > 0 and adj2[k[0]] < 0.05
        L.append(f"| {k} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:+.3f} | {r[3]:.0%} | {adj2[k[0]]:.2g} | {'**SUPPORTED**' if ok else '**NOT supported**'} |")
    f = res["FIVES_test"]
    L.append(f"| R2 - mask (uncorrected) | {f[0]:.3f} | {f[1]:.3f} | {f[2]:+.3f} | {f[3]:.0%} | {f[4]:.2g} (uncorr.) | - |")

    o1 = test(m("R2", "CHASE_DB1 1stHO"), m("P1", "CHASE_DB1 1stHO"))
    o2 = res["CHASE_DB1 1stHO"]
    adj3 = holm({"O1": o1[4], "O2": o2[4]})
    L += ["", "## M3: CHASE_DB1 1stHO (second-observer family: O1 with O2)", "",
          "| Test | Diff | R2 better on | Holm p | Verdict |", "| --- | --- | --- | --- | --- |"]
    for k, r in (("O1. R2 - P1 (unchanged)", o1), ("O2. R2 - mask", o2)):
        ok = r[2] > 0 and adj3[k[:2]] < 0.05
        L.append(f"| {k} | {r[2]:+.3f} | {r[3]:.0%} | {adj3[k[:2]]:.2g} | {'**SUPPORTED**' if ok else '**NOT supported**'} |")

    L += ["", "## Per seed, mask width at the validation threshold (42/123/7)", "", "| Case | Acc25 |", "| --- | --- |"]
    for case in CASES:
        d = new[new.case == case]
        L.append(f"| {case} | " + "/".join(f"{d[d.seed == s].acc25.mean():.3f}" for s in SEEDS) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--output", default=OUT_NAME)
    a = ap.parse_args()
    val_info, thr = {}, {}
    for s in SEEDS:
        t, names, dice = validation_threshold(s)
        thr[s], val_info[s] = t, (names, dice)
        print(f"seed {s}: validation threshold {t:.2f} (Dice {dice:.3f} on {', '.join(names)})", flush=True)
    csv = CAL / f"{a.output}_per_image.csv"
    if a.analyse_only:
        new = pd.read_csv(csv)
    else:
        new = compute(thr)
        new.to_csv(csv, index=False)
    report = analyse(new, thr, val_info)
    (CAL / f"{a.output}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
