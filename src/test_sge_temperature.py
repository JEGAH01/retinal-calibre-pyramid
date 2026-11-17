"""Score SGE geometry estimators on vessel calibre, optionally at several readout temperatures.

Originally: does soft-argmax averaging over pyramid levels compress the SGE radius?

The SGE reads radius as a softmax-weighted average of level radii. When the
level distribution is spread, the average is pulled toward the middle. Lower
softmax temperature at inference moves the readout toward the single most
likely level. No retraining. Temperature must be chosen on DRIVE only.

Scores at ground-truth centreline pixels (as src.calibre_accuracy): Acc25
(share within 25% of true radius), median |est/true - 1| and bias; plus
DRIVE_test enlarged x2 and x4.

Enlarged cases are scored two ways. "x2"/"x4": against radius targets recomputed
from the enlarged, re-thresholded mask (the original protocol). That target is
not scale-covariant for thin vessels (radius = EDT - 0.5 on the pixel grid): on
DRIVE_test the x2 target is ~10% below 2x the native target at the same place,
~1% at x4. "x2 native"/"x4 native" (--native-grid): the predicted log-radius map
is area-averaged back to the native grid and divided by t, then scored against
the native targets, so every factor uses the same ground truth.

    python -m src.test_sge_temperature --temperatures 1 --native-grid --output sge_s42.md \
        --estimators R2=checkpoints/sge_R2_seed42/sge.pth P1=checkpoints/sge_P1_seed42/sge.pth
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.morphology import skeletonize
from tqdm import tqdm

from models.medial_frame import medial_geometry_targets
from src import common as ev
import argparse

from src.common import Geometry, load_geometry_model

ROOT = Path(__file__).resolve().parents[1]
GEO = ("configs/data.yaml", None)  # dataset locations and preprocessing
TEMPERATURES = [1.0, 0.3, 0.1, 0.03]
CASES = [("DRIVE_test", 1.0), ("DRIVE_test", 2.0), ("DRIVE_test", 4.0), ("STARE", 1.0), ("CHASE_DB1", 1.0), ("HRF", 1.0)]
# Only with --cases FIVES_test (2026-11-11). 0.27 = DRIVE/FIVES field-of-view width (540/2013 px),
# i.e. FIVES brought to DRIVE scale; score it with --native-grid.
OPT_IN_CASES = [("FIVES_test", 1.0), ("FIVES_test", 0.27)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimators", nargs="+", required=True, help="name=checkpoints/.../sge.pth")
    ap.add_argument("--temperatures", type=float, nargs="+", default=TEMPERATURES)
    ap.add_argument("--output", default="sge_temperature.md")
    ap.add_argument("--cases", nargs="+", default=None, help="Restrict to these datasets, e.g. DRIVE_test")
    ap.add_argument("--native-grid", action="store_true", help="Also score enlarged cases on the native grid")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for item in a.estimators:
        name, path = item.split("=", 1)
        path, _, cfg = path.partition("@")
        if not cfg:
            cfg = GEO[0] if not path.endswith("sge.pth") else None
        c, m = load_geometry_model(cfg, path, device)
        loaded[name] = (Geometry(c, m, device), m.geometry_estimator)
    config = ev.load_config(GEO[0])
    rows = []
    selected = [c for c in CASES if a.cases is None or c[0] in a.cases]
    selected += [c for c in OPT_IN_CASES if a.cases is not None and c[0] in a.cases]
    for dataset, t in selected:
        spec = ev.resolve_datasets(config, [dataset])[dataset]
        for pair in tqdm(ev.pair_files(spec.image_dir, spec.mask_dir), desc=f"{dataset} x{t:g}"):
            image = ev.load_rgb_image(pair.image_path)
            vessel = ev.load_binary_mask(pair.mask_path) > 0.5
            native_hw = vessel.shape
            native_skel = skeletonize(vessel)
            native_true = medial_geometry_targets(vessel)["radius"][native_skel]
            if t != 1.0:
                size = (round(image.shape[1] * t), round(image.shape[0] * t))
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if t < 1 else cv2.INTER_LINEAR)
                vessel = cv2.resize(vessel.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0.5
            skel = skeletonize(vessel)
            true = medial_geometry_targets(vessel)["radius"][skel]
            for name, (geometry, estimator) in loaded.items():
              for temperature in a.temperatures:
                estimator.temperature = temperature
                radius_map = geometry(image)[0]
                scored = [(f"{dataset} x{t:g}", radius_map[skel] / true)]
                if a.native_grid and t != 1.0:
                    log_native = cv2.resize(np.log(radius_map).astype(np.float32), (native_hw[1], native_hw[0]),
                                            interpolation=cv2.INTER_AREA if t > 1 else cv2.INTER_LINEAR)
                    scored.append((f"{dataset} x{t:g} native", (np.exp(log_native) / t)[native_skel] / native_true))
                for case, ratio in scored:
                    rows.append({"case": case, "estimator": name, "temperature": temperature,
                                 "acc25": float(np.mean(np.abs(ratio - 1) <= 0.25)),
                                 "error": float(np.median(np.abs(ratio - 1))), "bias": float(np.median(ratio))})
    per = pd.DataFrame(rows)
    agg = per.groupby(["case", "estimator", "temperature"], sort=False)[["acc25", "error", "bias"]].median()
    lines = ["# SGE calibre scores", "",
             "Per image, then median over images. Centreline pixels of the manual mask.", "",
             "| Case | Estimator | Temperature | Acc25 | Error | Bias |", "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {c} | {e} | {tt:g} | {v.acc25:.3f} | {v.error:.3f} | {v.bias:.3f} |" for (c, e, tt), v in agg.iterrows()]
    per.to_csv(ROOT / "evaluation_outputs" / "calibre" / (Path(a.output).stem + "_per_image.csv"), index=False)
    out = ROOT / "evaluation_outputs" / "calibre" / a.output
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
