"""Figures 2-4 of the paper, from the released weights and per-image results.

Figure 2  Estimated / reference radius on the centreline of one unseen image per dataset (CHASE_DB1, HRF,
          FIVES): R2, P1 and mask width (transformer U-Net, validation threshold), seed 42. The image shown
          is the one whose seed-42 R2 Acc25 is the median of its dataset (lower median), not a chosen
          example. Each panel's Acc25 is recomputed here and checked against the saved per-image CSVs.
Figure 3  Bias of the reference radius on synthetic vessels (Table 7).
Figure 4  Acc25 against the true radius of the synthetic vessels at x1, x2 and x4 (Table 8).

    python -m src.fig_results            # needs the datasets and evaluation_outputs/csr_probs (Figure 2)
    python -m src.fig_results --only 3 4 # from the CSVs alone
Output: evaluation_outputs/figures/fig{2,3,4}_*.{png,pdf}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "evaluation_outputs" / "calibre"
OUT = ROOT / "evaluation_outputs" / "figures"
SEEDS = (42, 123, 7)

# Reference categorical palette, fixed order (series 1-5); neutral for the reference-only line.
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
NEUTRAL, INK, MUTED = "#6b6b6b", "#1f1f1f", "#6b6b6b"
# Diverging: under-estimate blue, over-estimate orange, neutral grey at ratio 1.
DIVERGING = LinearSegmentedColormap.from_list("ratio", ["#184f95", "#6da7ec", "#d9d9d6", "#f2a27f", "#b8461c"])
SEQUENTIAL = LinearSegmentedColormap.from_list("radius", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "stix"})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- Figure 2
CASES = [("CHASE_DB1", "CHASE_DB1 x1", "sge_R2_standalone_per_image.csv", "R2", "P1"),
         ("HRF", "HRF x1", "sge_R2_standalone_per_image.csv", "R2", "P1"),
         ("FIVES_test", "FIVES_test x1", "sge_fives_s42_per_image.csv", "R2_s42", "P1_s42")]
LABEL = {"CHASE_DB1": "CHASE_DB1", "HRF": "HRF", "FIVES_test": "FIVES"}


def per_image(file: str, case: str, estimator: str) -> pd.Series:
    t = pd.read_csv(CAL / file)
    t = t[(t.case == case) & (t.estimator == estimator)].reset_index(drop=True)
    return t.acc25


def crop_box(vessel: np.ndarray, side: int) -> tuple[int, int]:
    """Top-left corner of the side x side window with the most vessel pixels (deterministic)."""
    density = ndi.uniform_filter(vessel.astype(np.float32), size=side, mode="constant")
    h, w = vessel.shape
    half = side // 2
    density[:half, :] = density[h - half:, :] = -1
    density[:, :half] = density[:, w - half:] = -1
    cy, cx = np.unravel_index(np.argmax(density), density.shape)
    return cy - half, cx - half


def thicken(skel: np.ndarray, values: np.ndarray, width: float) -> np.ndarray:
    """Spread centreline values to pixels within `width` of the centreline (display only)."""
    dist, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
    out = values[iy, ix].astype(np.float32)
    out[dist > width] = np.nan
    return out


def figure2() -> None:
    from models.medial_frame import medial_geometry_targets
    from src.common import Geometry, load_binary_mask, load_geometry_model, load_rgb_image, read_gray
    from src.mask_calibre_baseline import mask_radius, scores
    from src.mask_threshold_validation import pairs, prob_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geo = {}
    for name in ("R2", "P1"):
        c, m = load_geometry_model(None, str(ROOT / f"checkpoints/sge_{name}_seed42/sge.pth"), device)
        geo[name] = Geometry(c, m, device)
    mask_csv = pd.read_csv(CAL / "mask_threshold_validation_per_image.csv")
    threshold = float(mask_csv[mask_csv.seed == 42].threshold.iloc[0])

    rows = []
    for dataset, case, file, r2_name, p1_name in CASES:
        r2_saved = per_image(file, case, r2_name)
        i = int(np.argsort(r2_saved.values, kind="stable")[(len(r2_saved) - 1) // 2])
        pair = pairs(dataset)[i]
        image = load_rgb_image(pair.image_path)
        vessel = load_binary_mask(pair.mask_path) > 0.5
        skel = skeletonize(vessel)
        true = medial_geometry_targets(vessel)["radius"]
        prob = read_gray(prob_path(42, dataset, i)).astype(np.float32) / 255.0
        fields = {"R2": geo["R2"](image)[0], "P1": geo["P1"](image)[0], "Mask width": mask_radius(prob >= threshold)}
        acc = {k: scores(v[skel] / true[skel])["acc25"] for k, v in fields.items()}
        saved = {"R2": r2_saved[i], "P1": per_image(file, case, p1_name)[i],
                 "Mask width": mask_csv[(mask_csv.case == dataset) & (mask_csv.i == i) & (mask_csv.seed == 42)].acc25.item()}
        for k in acc:
            assert abs(acc[k] - saved[k]) < 1e-9, (dataset, i, k, acc[k], saved[k])
        rows.append((dataset, i, pair.image_path.name, image, vessel, skel, true, fields, acc))
        print(f"{dataset}: image {i + 1} ({pair.image_path.name}), Acc25 " +
              ", ".join(f"{k} {v:.3f}" for k, v in acc.items()) + " (matches the saved CSVs)")

    fig = plt.figure(figsize=(7.2, 4.9))
    grid = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.07], wspace=0.06, hspace=0.32,
                            left=0.075, right=0.93, top=0.93, bottom=0.03)
    norm = LogNorm(0.5, 2.0)
    titles = ["Image", "Reference radius", "R2", "P1", "Mask width"]
    for r, (dataset, i, fname, image, vessel, skel, true, fields, acc) in enumerate(rows):
        side = int(round(0.3 * min(vessel.shape)))
        y0, x0 = crop_box(vessel, side)
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))
        width = max(1.0, side / 180)
        gray = image[sl].mean(axis=2) / 255.0
        ref_disp = thicken(skel[sl], np.where(skel, true, 0)[sl], width)
        vmax = float(np.nanpercentile(true[skel], 98))
        for c in range(5):
            ax = fig.add_subplot(grid[r, c])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.imshow(image[sl])
                ax.set_ylabel(f"{LABEL[dataset]}\nimage {i + 1}", color=INK, fontsize=8)
            else:
                ax.imshow(gray, cmap="gray", vmin=0, vmax=1, alpha=0.45)
                if c == 1:
                    im_ref = ax.imshow(ref_disp, cmap=SEQUENTIAL, vmin=0.5, vmax=vmax, interpolation="nearest")
                else:
                    name = titles[c]
                    ratio = np.where(skel, fields[name] / np.maximum(true, 1e-6), 1.0)
                    im_rat = ax.imshow(thicken(skel[sl], ratio[sl], width), cmap=DIVERGING, norm=norm,
                                       interpolation="nearest")
                    ax.text(0.03, 0.04, f"Acc25 {acc[name]:.2f}", transform=ax.transAxes, fontsize=7, color=INK,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            if r == 0:
                ax.set_title(titles[c], fontsize=8.5, color=INK)
        if r == 0:
            cax = fig.add_subplot(grid[0:2, 5])
            cb = fig.colorbar(im_rat, cax=cax, ticks=[0.5, 0.75, 1.0, 1.33, 2.0])
            cb.ax.set_yticklabels(["0.5", "0.75", "1", "1.33", "2"])
            cb.ax.minorticks_off()
            cb.set_label("estimated / reference radius", fontsize=7.5)
            tr = cb.ax.get_yaxis_transform()  # x in axes units, y in data units
            cb.ax.plot([-0.35, -0.35], [0.75, 1.25], transform=tr, color=INK, lw=0.8, clip_on=False)
            cb.ax.text(-0.55, 1.0, "±25%", transform=tr, rotation=90, ha="center", va="center",
                       fontsize=6, color=INK, clip_on=False)
            cb.outline.set_visible(False)
        if r == 2:
            cax = fig.add_subplot(grid[2, 5])
            cb = fig.colorbar(im_ref, cax=cax)
            cb.set_label("reference radius (px)", fontsize=7.5)
            cb.outline.set_visible(False)
    save(fig, "fig2_radius_maps")


# ---------------------------------------------------------------- Figure 3
BINS = [0.5, 1, 1.5, 2, 3, 4]
BIN_LABELS = ["0.5–1", "1–1.5", "1.5–2", "2–3", "3–4"]


def figure3() -> None:
    r = pd.read_csv(CAL / "synthetic_calibre_references.csv.gz")
    series = [("Native mask (×1, and native grid at any factor)", "native mask x t (native grid)", 1.0, BLUE, "o"),
              ("Enlarged mask, ×2", "enlarged mask (original protocol)", 2.0, ORANGE, "s"),
              ("Enlarged mask, ×4", "enlarged mask (original protocol)", 4.0, AQUA, "^")]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(BIN_LABELS))
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(len(x) - 1.02, 1.005, "unbiased", ha="right", va="bottom", fontsize=7, color=MUTED)
    for label, ref, factor, colour, marker in series:
        g = r[(r.reference == ref) & (r.factor == factor)]
        med = g.groupby(pd.cut(g.r_native, BINS, right=False), observed=True).ratio.median().values
        ax.plot(x, med, color=colour, lw=2, marker=marker, ms=5, mec="white", mew=0.8, label=label, zorder=3)
    ax.set_xticks(x, BIN_LABELS)
    ax.set_xlabel("true radius at ×1 (px)")
    ax.set_ylabel("reference / true radius (median)")
    ax.set_ylim(0.4, 1.06)
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig3_reference_bias")


# ---------------------------------------------------------------- Figure 4
def figure4() -> None:
    syn = pd.concat([pd.read_csv(CAL / "synthetic_calibre_per_image.csv"),
                     pd.read_csv(CAL / "synthetic_calibre_SEU_per_image.csv")])
    syn = syn[syn.protocol == "truth"]
    lss = pd.read_csv(CAL / "scale_selection_baseline_per_image.csv")
    lss = lss[lss.case.str.startswith("synthetic")].assign(factor=lambda t: t.case.str[-1].astype(float))
    factors = [1.0, 2.0, 4.0]
    fig = plt.figure(figsize=(7.2, 2.7))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 2.1], wspace=0.22, left=0.02, right=0.80, top=0.9, bottom=0.17)
    ax0 = fig.add_subplot(grid[0])
    img = plt.imread(CAL / "synthetic" / "001_image.png")
    ax0.imshow(img)
    ax0.set_anchor("N")  # top-align with panel (b)
    ax0.set_xticks([]); ax0.set_yticks([])
    for s in ax0.spines.values():
        s.set_visible(False)
    ax0.set_title("(a) Synthetic image (example)", fontsize=8.5, color=INK)
    ax = fig.add_subplot(grid[1])
    ax.set_title("(b) Acc25 against the true radius", fontsize=8.5, color=INK)
    x = np.log2(factors)
    offsets = {"R2": -0.06, "SEU": 0.0, "P1": 0.06}
    for name, colour, marker in [("R2", BLUE, "o"), ("P1", ORANGE, "s"), ("SEU", AQUA, "D")]:
        t = syn[syn.method == name]
        mean = t.groupby("factor").acc25.mean().reindex(factors).values
        per_seed = t.groupby(["factor", "seed"]).acc25.mean().unstack().reindex(factors)
        lo, hi = per_seed.min(axis=1).values, per_seed.max(axis=1).values
        xs = x + offsets[name]
        ax.vlines(xs, lo, hi, color=colour, lw=1.2, alpha=0.6)
        ax.plot(xs, mean, color=colour, lw=2, marker=marker, ms=5, mec="white", mew=0.8, label=name, zorder=3)
    for name, label, colour, marker in [("LSS", "LSS", YELLOW, "v"), ("LSS_k1", "LSS (k = 1)", VIOLET, "^")]:
        mean = lss[lss.method == name].groupby("factor").acc25.mean().reindex(factors).values
        ax.plot(x, mean, color=colour, lw=2, marker=marker, ms=5, mec="white", mew=0.8, label=label, zorder=3)
    ref = syn[syn.method == "mask reference"].groupby("factor").acc25.mean().reindex(factors).values
    ax.plot(x, ref, color=NEUTRAL, lw=1.4, ls=(0, (4, 2)), marker="o", ms=4, mfc="white", label="Perfect annotation")
    ax.set_xticks(x, ["×1", "×2", "×4"])
    ax.set_xlim(-0.3, 2.3)
    ax.set_xlabel("enlargement factor")
    ax.set_ylabel("Acc25")
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig4_synthetic_accuracy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", type=int, default=[2, 3, 4])
    a = ap.parse_args()
    if 2 in a.only:
        figure2()
    if 3 in a.only:
        figure3()
    if 4 in a.only:
        figure4()


if __name__ == "__main__":
    main()
