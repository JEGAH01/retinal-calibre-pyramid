"""Classical scale-selection calibre baseline, "LSS" (pre-specified in PRESPECIFICATION_LOG.md,
2026-11-12, before computing). No training; one constant calibrated on the DRIVE training images.

Per pixel, the scale sigma* maximising the gamma-normalised ridge strength
    R(sigma) = sigma^2 * max(-lambda_min(Hessian of G_sigma * I), 0)
of the inverted green channel I (Lindeberg, 1998), over sigma = 0.5 * 2^(k/4), k = 0..22
(0.5-22.6 px, the span of R2's pyramid), refined by a parabola in ln sigma. A bar of half-width r
peaks at sigma = r. Radius = k_cal * sigma*, k_cal = exp(median ln(r_ref / sigma*)) over the
centreline pixels of the 20 DRIVE training images; "LSS_k1" uses k = 1.

Scored exactly like R2/P1 (src.test_sge_temperature, src.synthetic_calibre, src.second_observer).

    python -m src.scale_selection_baseline              # CPU, ~30-45 min (FIVES and HRF dominate)
    python -m src.scale_selection_baseline --analyse-only
Output: evaluation_outputs/calibre/scale_selection_baseline{.md,_per_image.csv,_k.json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from skimage.morphology import skeletonize
from tqdm import tqdm

from models.medial_frame import medial_geometry_targets
from src import common as ev
from src import synthetic_calibre as syn
from src.fives_calibre_analysis import pairs
from src.mask_calibre_baseline import holm, scores
from src.second_observer import observer_files
from src.wide_scale_analysis import load as load_estimators

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "evaluation_outputs" / "calibre"
NAME = "scale_selection_baseline"
SIGMA0, STEP, N_SCALES = 0.5, np.log(2.0) / 4, 23
D2 = np.array([[1.0, -2.0, 1.0]], np.float32)
DXY = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], np.float32) / 4
SEEDS = (42, 123, 7)


def selected_sigma(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel sigma* (px) of the gamma-normalised ridge strength, parabola-refined in ln sigma."""
    img = 255.0 - rgb[..., 1].astype(np.float32)
    best = np.full(img.shape, -np.inf, np.float32)
    idx = np.zeros(img.shape, np.int16)
    left = np.full(img.shape, np.nan, np.float32)
    right = np.full(img.shape, np.nan, np.float32)
    prev = None
    for k in range(N_SCALES):
        s = SIGMA0 * np.exp(k * STEP)
        g = cv2.GaussianBlur(img, (0, 0), s, borderType=cv2.BORDER_REFLECT)
        lxx = cv2.filter2D(g, -1, D2, borderType=cv2.BORDER_REFLECT)
        lyy = cv2.filter2D(g, -1, D2.T, borderType=cv2.BORDER_REFLECT)
        lxy = cv2.filter2D(g, -1, DXY, borderType=cv2.BORDER_REFLECT)
        lam_min = 0.5 * (lxx + lyy) - np.sqrt(0.25 * (lxx - lyy) ** 2 + lxy ** 2)
        resp = (s * s) * np.maximum(-lam_min, 0.0)
        new = resp > best
        fill = (idx == k - 1) & ~new
        right[fill] = resp[fill]
        left[new] = prev[new] if prev is not None else np.nan
        right[new] = np.nan
        best[new], idx[new] = resp[new], k
        prev = resp
    den = left - 2 * best + right
    ok = np.isfinite(left) & np.isfinite(right) & (den < 0)
    off = np.zeros(img.shape, np.float32)
    off[ok] = np.clip(0.5 * (left[ok] - right[ok]) / den[ok], -0.5, 0.5)
    return (SIGMA0 * np.exp((idx + off) * STEP)).astype(np.float32)


def calibrate() -> float:
    d = ROOT / "datasets" / "DRIVE" / "training"
    logs = []
    for img_path in tqdm(sorted((d / "image").glob("*.png")), desc="calibrate on DRIVE training"):
        vessel = ev.load_binary_mask(d / "1st_manual" / img_path.name) > 0.5
        skel = skeletonize(vessel)
        ref = medial_geometry_targets(vessel)["radius"][skel]
        sig = selected_sigma(ev.load_rgb_image(img_path))[skel]
        logs.append(np.log(ref / sig))
    return float(np.exp(np.median(np.concatenate(logs))))


def rows_for(case: str, i: int, ratio_sigma: np.ndarray, k_cal: float) -> list[dict]:
    """ratio_sigma = sigma*/r_ref at the scored pixels; both variants from it."""
    return [{"case": case, "i": i, "method": "LSS", **scores(k_cal * ratio_sigma)},
            {"case": case, "i": i, "method": "LSS_k1", **scores(ratio_sigma)}]


def compute(k_cal: float, limit: int | None = None) -> pd.DataFrame:
    rows = []
    for name in ("DRIVE_test", "STARE", "CHASE_DB1", "HRF", "FIVES_test"):
        ps = pairs(name)
        first = [f[0] for f in observer_files("CHASE_DB1", ps)] if name == "CHASE_DB1" else None
        for i, pair in enumerate(tqdm(ps[:limit], desc=name)):
            image = ev.load_rgb_image(pair.image_path)
            vessel = ev.load_binary_mask(pair.mask_path) > 0.5
            skel = skeletonize(vessel)
            true = medial_geometry_targets(vessel)["radius"][skel]
            sig = selected_sigma(image)
            rows += rows_for(f"{name} x1", i, sig[skel] / true, k_cal)
            if first is not None:
                v1 = ev.load_binary_mask(first[i]) > 0.5
                s1 = skeletonize(v1)
                rows += rows_for("CHASE_DB1 1stHO x1", i, sig[s1] / medial_geometry_targets(v1)["radius"][s1], k_cal)
            if name != "DRIVE_test":
                continue
            h, w = vessel.shape
            for t in (2.0, 4.0):
                size = (round(w * t), round(h * t))
                big = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
                vt = cv2.resize(vessel.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0.5
                st = skeletonize(vt)
                sig_t = selected_sigma(big)
                rows += rows_for(f"DRIVE_test x{t:g}", i, sig_t[st] / medial_geometry_targets(vt)["radius"][st], k_cal)
                native = np.exp(cv2.resize(np.log(sig_t), (w, h), interpolation=cv2.INTER_AREA)) / t
                rows += rows_for(f"DRIVE_test x{t:g} native", i, native[skel] / true, k_cal)
    names = sorted(p.name[:-10] for p in syn.OUT.glob("*_image.png"))
    for i, nm in enumerate(tqdm(names[:limit], desc="synthetic")):
        img0 = cv2.cvtColor(cv2.imread(str(syn.OUT / f"{nm}_image.png")), cv2.COLOR_BGR2RGB)
        points = np.load(syn.OUT / f"{nm}_points.npy")
        h0, w0 = img0.shape[:2]
        for t in syn.FACTORS:
            img = img0 if t == 1.0 else cv2.resize(img0, (round(w0 * t), round(h0 * t)), interpolation=cv2.INTER_LINEAR)
            y, x, r = syn.pixels_at(points, t, img.shape[:2])
            rows += rows_for(f"synthetic x{t:g}", i, selected_sigma(img)[y, x] / r, k_cal)
    return pd.DataFrame(rows)


def learned() -> pd.DataFrame:
    """R2 and P1 per image (seed-averaged later) under the same case names."""
    e = load_estimators()
    e = e[e.model.isin(["R2", "P1"])][["case", "model", "seed", "i", "acc25", "bias"]]
    g = pd.read_csv(CAL / "sge_drive_nativegrid_per_image.csv")
    g["i"] = g.groupby(["case", "estimator"]).cumcount()
    g = g[g.case.isin(["DRIVE_test x2", "DRIVE_test x4"])]
    g = g.assign(model=g.estimator.str.split("_s").str[0], seed=g.estimator.str.split("_s").str[1].astype(int))
    so = pd.read_csv(CAL / "second_observer_per_image.csv")
    so = so[(so.dataset == "CHASE_DB1") & (so.reference == "obs1") & (so.protocol == "full") & so.method.isin(["R2", "P1"])]
    so = so.assign(case="CHASE_DB1 1stHO x1", model=so.method)
    sy = pd.read_csv(CAL / "synthetic_calibre_per_image.csv")
    sy = sy[(sy.protocol == "truth") & sy.method.isin(["R2", "P1"])]
    sy = sy.assign(case="synthetic x" + sy.factor.map(lambda f: f"{f:g}"), i=sy.image, model=sy.method)
    cols = ["case", "model", "seed", "i", "acc25", "bias"]
    return pd.concat([e[cols], g[cols], so[cols], sy[cols]])


def analyse(t: pd.DataFrame, k_cal: float) -> str:
    est = learned().groupby(["case", "model", "i"]).acc25.mean()
    lss = t.set_index(["case", "method", "i"]).acc25.sort_index()

    def cmp(case, a="R2", b="LSS"):
        x = est.loc[(case, a)] if a in ("R2", "P1") else lss.loc[(case, a)]
        y = lss.loc[(case, b)]
        d = x - y
        return x.mean(), y.mean(), d.mean(), (d > 0).mean(), (wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0)

    L = ["# Classical scale-selection baseline (LSS)", "",
         f"Pre-specified in PRESPECIFICATION_LOG.md (2026-11-12). Calibrated k = {k_cal:.3f} on the 20 DRIVE training images "
         "(LSS_k1: k = 1). R2/P1: each image averaged over seeds 42/123/7. Acc25, paired Wilcoxon over images.", ""]
    for title, cases in (("L1. Real unseen data (primary, Holm over four)", ["STARE x1", "CHASE_DB1 x1", "HRF x1", "FIVES_test x1"]),
                         ("L2. Synthetic, against the true radius (primary, Holm over three)", ["synthetic x1", "synthetic x2", "synthetic x4"])):
        res = {c: cmp(c) for c in cases}
        adj = holm({c: r[4] for c, r in res.items()})
        L += [f"## {title}", "", "| Case | R2 | LSS | R2 - LSS | R2 better on | Holm p | Verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for c, (a, b, d, w, _) in res.items():
            ok = d > 0 and adj[c] < 0.05
            worse = d < 0 and adj[c] < 0.05
            v = "**R2 better**" if ok else ("**LSS better**" if worse else "no significant difference")
            L.append(f"| {c} | {a:.3f} | {b:.3f} | {d:+.3f} | {w:.0%} | {adj[c]:.2g} | {v} |")
        L.append("")
    L += ["## Secondary (uncorrected) and all cases", "",
          "| Case | R2 | P1 | LSS | LSS k=1 | R2 - LSS (p) | P1 - LSS (p) | LSS bias |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    order = ["DRIVE_test x1", "DRIVE_test x2", "DRIVE_test x2 native", "DRIVE_test x4", "DRIVE_test x4 native", "STARE x1",
             "CHASE_DB1 x1", "CHASE_DB1 1stHO x1", "HRF x1", "FIVES_test x1", "synthetic x1", "synthetic x2", "synthetic x4"]
    for c in order:
        r, p = cmp(c), cmp(c, "P1")
        k1 = lss.loc[(c, "LSS_k1")].mean()
        bias = t[(t.case == c) & (t.method == "LSS")].bias.median()
        L.append(f"| {c} | {r[0]:.3f} | {p[0]:.3f} | {r[1]:.3f} | {k1:.3f} | {r[2]:+.3f} ({r[4]:.2g}) | {p[2]:+.3f} ({p[4]:.2g}) | {bias:.3f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    kfile, csv = CAL / f"{NAME}_k.json", CAL / f"{NAME}_per_image.csv"
    if a.analyse_only:
        k_cal, t = json.loads(kfile.read_text())["k"], pd.read_csv(csv)
    else:
        k_cal = calibrate()
        kfile.write_text(json.dumps({"k": k_cal}))
        print(f"calibrated k = {k_cal:.3f}", flush=True)
        t = compute(k_cal)
        t.to_csv(csv, index=False)
    report = analyse(t, k_cal)
    (CAL / f"{NAME}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
