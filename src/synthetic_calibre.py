"""Synthetic vessels of known width (pre-specified in PRESPECIFICATION_LOG.md, 2026-11-12).

The only setting where the true radius is known exactly. Used for:
  S1 reference quality (no model): mask-derived radius vs the true radius, at x1/x2/x4
  S2 protocol validity: which scale-test protocol (original vs native grid) reproduces the
     R2 - P1 difference measured against the true radius
  S3 accuracy against the true radius: R2 vs P1, Holm over x1/x2/x4

Construction (fixed seed 2026):
  backgrounds  the 20 DRIVE_test images, vessels removed (Telea inpainting of 1st_manual dilated 2 px)
  vessels      20 smooth random curves per image in the FOV, constant true radius r ~ logU[0.5, 4.0] px
  rendering    pixel coverage of the tube (9x9 supersampling, exact distance to the curve), Gaussian PSF
               sigma 0.5 px, background darkened by coverage x per-channel depth; the depth is the median
               centreline/inpainted-background ratio on DRIVE_test vessels with reference radius >= 2 px
  annotation   pixels whose centre lies within r of the curve (a perfect annotator on the pixel grid)
  scoring      true centreline sampled every 0.25 px, rounded to pixels; excluded within 2*r_max + 2 px of
               another curve, within 3 px of a curve end or of the FOV border
  enlargement  x2/x4 exactly as the real scale test (cv2 INTER_LINEAR on image and mask, mask > 0.5);
               true radius t*r, positions (x + 0.5) t - 0.5

    python -m src.synthetic_calibre                 # build images + score (GPU, ~10-20 min)
    python -m src.synthetic_calibre --build-only    # only the images (CPU), for inspection
    python -m src.synthetic_calibre --analyse-only  # tables from the saved CSVs
Output: evaluation_outputs/calibre/synthetic/ (images, masks, points) and synthetic_calibre{.md,_*.csv}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from skimage.morphology import skeletonize
from tqdm import tqdm

from src import common as ev
from src.common import Geometry, load_geometry_model
from src.mask_calibre_baseline import holm, mask_radius

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "evaluation_outputs" / "calibre"
OUT = CAL / "synthetic"
DRIVE = ROOT / "datasets_processed" / "DRIVE_test"
SEED, N_CURVES, R_MIN, R_MAX = 2026, 20, 0.5, 4.0
SS, PSF_SIGMA, STEP = 9, 0.5, 0.25
FACTORS = (1.0, 2.0, 4.0)
SEEDS = (42, 123, 7)
BINS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.01]


# ---------------------------------------------------------------- construction

def backgrounds():
    names = sorted(p.name for p in (DRIVE / "images").glob("*.png"))
    for name in names:
        bgr = cv2.imread(str(DRIVE / "images" / name), cv2.IMREAD_COLOR)
        vessel = cv2.imread(str(DRIVE / "masks" / name), cv2.IMREAD_GRAYSCALE) > 127
        fov = cv2.imread(str(DRIVE / "fov_masks" / name), cv2.IMREAD_GRAYSCALE) > 127
        hole = cv2.dilate(vessel.astype(np.uint8), np.ones((5, 5), np.uint8))  # 2 px each side
        clean = cv2.inpaint(bgr, hole, 5, cv2.INPAINT_TELEA)
        yield name, bgr, clean, vessel, fov


def vessel_depth() -> np.ndarray:
    """Per-channel (BGR) relative darkening at wide-vessel centrelines: 1 - median(orig / background)."""
    ratios = []
    for _, bgr, clean, vessel, fov in backgrounds():
        skel = skeletonize(vessel) & fov
        sel = skel & (mask_radius(vessel) >= 2.0)
        ratios.append(bgr[sel].astype(np.float64) / np.maximum(clean[sel].astype(np.float64), 1.0))
    return np.clip(1.0 - np.median(np.concatenate(ratios), axis=0), 0.0, 0.9)


def random_curve(rng, fov_inner: np.ndarray, r: float):
    """Smooth random polyline (pixel-centre coords, x right / y down), sampled every 0.01 px, inside the FOV."""
    h, w = fov_inner.shape
    ys, xs = np.nonzero(fov_inner)
    for _ in range(200):
        k = rng.integers(len(xs))
        length = rng.uniform(80, 220)
        n = int(length / 0.01)
        knots = np.cumsum(rng.normal(0, 0.06, int(length) + 2))  # heading random walk per px
        heading = rng.uniform(0, 2 * np.pi) + np.interp(np.arange(n) * 0.01, np.arange(len(knots)), knots)
        pts = np.stack([xs[k] + np.cumsum(np.cos(heading)) * 0.01, ys[k] + np.cumsum(np.sin(heading)) * 0.01], 1)
        ix, iy = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
        if ix.min() < 0 or iy.min() < 0 or ix.max() >= w or iy.max() >= h:
            continue
        if fov_inner[iy, ix].all():
            return pts
    raise RuntimeError("could not place a curve")


def render(name, clean, fov, depth, rng):
    h, w = fov.shape
    fov_inner = cv2.erode(fov.astype(np.uint8), np.ones((2 * int(R_MAX) + 13,) * 2, np.uint8)) > 0
    cover = np.zeros((h, w), np.float64)
    annot = np.zeros((h, w), bool)
    curves = []
    sub = (np.arange(SS) + 0.5) / SS - 0.5
    for c in range(N_CURVES):
        r = float(np.exp(rng.uniform(np.log(R_MIN), np.log(R_MAX))))
        pts = random_curve(rng, fov_inner, r)
        tree = cKDTree(pts)
        x0, y0 = np.floor(pts.min(0) - r - 2).astype(int)
        x1, y1 = np.ceil(pts.max(0) + r + 2).astype(int)
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        d_centre = tree.query(np.stack([gx.ravel(), gy.ravel()], 1))[0].reshape(gx.shape)
        annot[y0:y1 + 1, x0:x1 + 1] |= d_centre <= r
        near = d_centre <= r + 1.0
        py, px = gy[near], gx[near]
        sx = (px[:, None, None] + sub[None, None, :]).repeat(SS, 1)
        sy = (py[:, None, None] + sub[None, :, None]).repeat(SS, 2)
        d = tree.query(np.stack([sx.ravel(), sy.ravel()], 1))[0].reshape(len(px), SS * SS)
        frac = (d <= r).mean(1)
        cover[py, px] = np.maximum(cover[py, px], frac)
        curves.append((pts, r))
    blurred = cv2.GaussianBlur(cover.astype(np.float32), (0, 0), PSF_SIGMA)
    img = clean.astype(np.float32) * (1.0 - blurred[..., None] * depth[None, None, :].astype(np.float32))
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return img, annot, curves


def scoring_points(curves, fov):
    """Continuous centreline points (every 0.25 px) with their true radius, after the exclusions."""
    trees = [cKDTree(p) for p, _ in curves]
    border = cv2.erode(fov.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    out = []
    for k, (pts, r) in enumerate(curves):
        s = pts[:: int(STEP / 0.01)]
        arc = np.arange(len(s)) * STEP
        keep = (arc >= 3.0) & (arc <= arc[-1] - 3.0)
        for j, t in enumerate(trees):
            if j != k:
                keep &= t.query(s)[0] > 2 * R_MAX + 2
        ix, iy = np.round(s[:, 0]).astype(int), np.round(s[:, 1]).astype(int)
        keep &= border[iy, ix]
        out.append(np.column_stack([s[keep], np.full(keep.sum(), r), np.full(keep.sum(), k)]))
    return np.concatenate(out)  # x, y, r, curve


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    depth = vessel_depth()
    np.save(OUT / "depth_bgr.npy", depth)
    print("vessel depth (B, G, R):", np.round(depth, 3))
    for i, (name, _, clean, _, fov) in enumerate(tqdm(list(backgrounds()), desc="render")):
        rng = np.random.default_rng(SEED + i)
        img, annot, curves = render(name, clean, fov, depth, rng)
        cv2.imwrite(str(OUT / f"{name[:-4]}_image.png"), img)
        cv2.imwrite(str(OUT / f"{name[:-4]}_annotation.png"), annot.astype(np.uint8) * 255)
        np.save(OUT / f"{name[:-4]}_points.npy", scoring_points(curves, fov))


# ---------------------------------------------------------------- scoring

def pixels_at(points: np.ndarray, t: float, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique scoring pixels at factor t (pixels hit by two curves are dropped) and true radius t*r."""
    x = np.round((points[:, 0] + 0.5) * t - 0.5).astype(int)
    y = np.round((points[:, 1] + 0.5) * t - 0.5).astype(int)
    ok = (x >= 0) & (y >= 0) & (x < shape[1]) & (y < shape[0])
    df = pd.DataFrame({"x": x[ok], "y": y[ok], "r": points[ok, 2] * t, "c": points[ok, 3]})
    g = df.groupby(["y", "x"]).agg(r=("r", "first"), n=("c", "nunique")).reset_index()
    g = g[g.n == 1]
    return g.y.to_numpy(), g.x.to_numpy(), g.r.to_numpy()


def acc(ratio: np.ndarray) -> dict:
    return {"acc25": float(np.mean(np.abs(ratio - 1) <= 0.25)), "error": float(np.median(np.abs(ratio - 1))),
            "bias": float(np.median(ratio)), "n": int(ratio.size)}


def score(limit: int | None = None, model_names=("R2", "P1")) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for s in SEEDS:
        for m in model_names:
            c, net = load_geometry_model(None, str(ROOT / f"checkpoints/sge_{m}_seed{s}/sge.pth"), device)
            models[(m, s)] = Geometry(c, net, device)
    rows, refs = [], []
    names = sorted(p.name[:-10] for p in OUT.glob("*_image.png"))[:limit]
    for i, name in enumerate(tqdm(names, desc="score")):
        img0 = cv2.cvtColor(cv2.imread(str(OUT / f"{name}_image.png")), cv2.COLOR_BGR2RGB)
        ann0 = cv2.imread(str(OUT / f"{name}_annotation.png"), cv2.IMREAD_GRAYSCALE) > 127
        points = np.load(OUT / f"{name}_points.npy")
        h0, w0 = ann0.shape
        ref0 = mask_radius(ann0)
        skel0 = skeletonize(ann0)
        y0, x0, r0 = pixels_at(points, 1.0, ann0.shape)
        for t in FACTORS:
            if t == 1.0:
                img, ann = img0, ann0
            else:
                size = (round(w0 * t), round(h0 * t))
                img = cv2.resize(img0, size, interpolation=cv2.INTER_LINEAR)
                ann = cv2.resize(ann0.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0.5
            y, x, r = pixels_at(points, t, ann.shape)
            ref_t = mask_radius(ann)
            skel_t = skeletonize(ann)
            # S1: references against the truth, per true-radius bin (native r = r / t)
            for label, ratio in (("enlarged mask (original protocol)", ref_t[y, x] / r),
                                 ("native mask x t (native grid)", ref0[y0, x0] / r0)):
                refs.append(pd.DataFrame({"image": i, "factor": t, "reference": label, "r_native": r / t if "enlarged" in label else r0,
                                          "ratio": ratio}))
            for (m, s), g in models.items():
                est = g(img)[0]
                base = {"image": i, "factor": t, "method": m, "seed": s}
                rows.append({**base, "protocol": "truth", **acc(est[y, x] / r)})
                rows.append({**base, "protocol": "original", **acc(est[skel_t] / ref_t[skel_t])})
                if t != 1.0:
                    log_native = cv2.resize(np.log(est).astype(np.float32), (w0, h0), interpolation=cv2.INTER_AREA)
                    native = np.exp(log_native) / t
                else:
                    native = est
                rows.append({**base, "protocol": "native grid", **acc(native[skel0] / ref0[skel0])})
            rows.append({"image": i, "factor": t, "method": "mask reference", "seed": 0, "protocol": "truth",
                         **acc(ref_t[y, x] / r)})
    return pd.DataFrame(rows), pd.concat(refs, ignore_index=True)


# ---------------------------------------------------------------- analysis

def img_mean(t: pd.DataFrame, factor, protocol, method) -> pd.Series:
    d = t[(t.factor == factor) & (t.protocol == protocol) & (t.method == method)]
    return d.groupby("image").acc25.mean()


def analyse(t: pd.DataFrame, refs: pd.DataFrame) -> str:
    n = t.image.nunique()
    pts = int(t[(t.factor == 1.0) & (t.method == "mask reference")].n.sum())
    lines = ["# Synthetic vessels of known width", "",
             f"Pre-specified in PRESPECIFICATION_LOG.md (2026-11-12). {n} images, {pts} scoring pixels at x1. "
             "Acc25 per image, each image averaged over seeds 42/123/7; paired Wilcoxon over images.", ""]

    lines += ["## S1. Mask-derived reference / true radius (median, by true native radius)", "",
              "| Factor | Reference | " + " | ".join(f"r {BINS[k]:g}-{min(BINS[k + 1], 4):g}" for k in range(len(BINS) - 1)) + " | All |",
              "| --- | --- |" + " --- |" * len(BINS)]
    for (f, ref), d in refs.groupby(["factor", "reference"], sort=False):
        b = pd.cut(d.r_native, BINS, right=False)
        cells = [f"{v:.3f}" for v in d.groupby(b, observed=False).ratio.median()]
        lines.append(f"| x{f:g} | {ref} | " + " | ".join(cells) + f" | {d.ratio.median():.3f} |")

    lines += ["", "## S2. Protocol validity: R2 - P1 Acc25 under each protocol (primary at x2)", "",
              "| Factor | Δ true | Δ original | Δ native grid | mean abs(Δorig-Δtrue) | mean abs(Δnative-Δtrue) | p | Native grid better? |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in (2.0, 4.0):
        dt = img_mean(t, f, "truth", "R2") - img_mean(t, f, "truth", "P1")
        do = img_mean(t, f, "original", "R2") - img_mean(t, f, "original", "P1")
        dn = img_mean(t, f, "native grid", "R2") - img_mean(t, f, "native grid", "P1")
        eo, en = (do - dt).abs(), (dn - dt).abs()
        p = wilcoxon(en, eo).pvalue if np.any(en != eo) else float("nan")
        better = en.mean() < eo.mean() and p < 0.05
        tag = " (primary)" if f == 2.0 else ""
        lines.append(f"| x{f:g}{tag} | {dt.mean():+.3f} | {do.mean():+.3f} | {dn.mean():+.3f} | {eo.mean():.3f} | {en.mean():.3f} | "
                     f"{p:.2g} | {'**SUPPORTED**' if better else '**NOT supported**'} |")
    lines.append("")
    lines.append("Sign agreement with the true Δ: " + "; ".join(
        f"x{f:g} original {'yes' if np.sign((img_mean(t, f, 'original', 'R2') - img_mean(t, f, 'original', 'P1')).mean()) == np.sign((img_mean(t, f, 'truth', 'R2') - img_mean(t, f, 'truth', 'P1')).mean()) else 'NO'}, "
        f"native grid {'yes' if np.sign((img_mean(t, f, 'native grid', 'R2') - img_mean(t, f, 'native grid', 'P1')).mean()) == np.sign((img_mean(t, f, 'truth', 'R2') - img_mean(t, f, 'truth', 'P1')).mean()) else 'NO'}"
        for f in (2.0, 4.0)))

    lines += ["", "## S3. Acc25 against the true radius (Holm over x1/x2/x4)", "",
              "| Factor | R2 | P1 | Mask reference | R2 - P1 | R2 better on | p | Holm p | Verdict |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    res = {}
    for f in FACTORS:
        a, b = img_mean(t, f, "truth", "R2"), img_mean(t, f, "truth", "P1")
        res[f] = (a.mean(), b.mean(), img_mean(t, f, "truth", "mask reference").mean(), (a - b).mean(), (a > b).mean(),
                  wilcoxon(a, b).pvalue if np.any(a != b) else 1.0)
    adj = holm({f: v[5] for f, v in res.items()})
    for f, (a, b, m, d, w, p) in res.items():
        ok = d > 0 and adj[f] < 0.05
        lines.append(f"| x{f:g} | {a:.3f} | {b:.3f} | {m:.3f} | {d:+.3f} | {w:.0%} | {p:.2g} | {adj[f]:.2g} | "
                     f"{'**SUPPORTED**' if ok else '**NOT supported**'} |")

    lines += ["", "## Secondary: bias (median estimate / true) and per-seed Acc25 against the truth", "",
              "| Factor | R2 bias | P1 bias | R2 Acc25 (42/123/7) | P1 Acc25 (42/123/7) |", "| --- | --- | --- | --- | --- |"]
    for f in FACTORS:
        d = t[(t.factor == f) & (t.protocol == "truth")]
        bias = {m: d[d.method == m].groupby("image").bias.mean().median() for m in ("R2", "P1")}
        seeds = {m: "/".join(f"{d[(d.method == m) & (d.seed == s)].acc25.mean():.3f}" for s in SEEDS) for m in ("R2", "P1")}
        lines.append(f"| x{f:g} | {bias['R2']:.3f} | {bias['P1']:.3f} | {seeds['R2']} | {seeds['P1']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Score only the first N images (smoke test; not saved)")
    ap.add_argument("--models", nargs="+", default=["R2", "P1"],
                    help="Checkpoints checkpoints/sge_{name}_seed{seed}/sge.pth. Other than R2 P1: per-image CSV "
                         "synthetic_calibre_{names}_per_image.csv only, no analysis (e.g. --models SEU)")
    a = ap.parse_args()
    rows_csv, refs_csv = CAL / "synthetic_calibre_per_image.csv", CAL / "synthetic_calibre_references.csv.gz"
    extra = a.models != ["R2", "P1"]
    if extra:
        rows_csv = CAL / f"synthetic_calibre_{'_'.join(a.models)}_per_image.csv"
    if a.analyse_only:
        t, refs = pd.read_csv(rows_csv), pd.read_csv(refs_csv)
    else:
        if not (OUT / "020_points.npy").exists():
            build()
        if a.build_only:
            return
        t, refs = score(a.limit, a.models)
        if a.limit:
            print(t.groupby(["factor", "protocol", "method"]).n.sum().to_string())
            return
        if extra:
            t.to_csv(rows_csv, index=False)
            print(f"saved {rows_csv}")
            return
        t.to_csv(rows_csv, index=False)
        refs.to_csv(refs_csv, index=False)
    report = analyse(t, refs)
    (CAL / "synthetic_calibre.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
