"""Figure 1: the R2 estimator, drawn with its real outputs (seed 42, DRIVE test image 01, CPU).

    python -m src.fig_architecture
Writes evaluation_outputs/figures/fig1_architecture.{png,pdf}.
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import common as ev
from src.common import load_geometry_model
from src.mask_calibre_baseline import mask_radius
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, ACCENT, SECOND = "#2b2b29", "#6b6b66", "#2a78d6", "#eb6834"


def run():
    rgb = cv2.cvtColor(cv2.imread(str(ROOT / "datasets/DRIVE/test/images/01.png")), cv2.COLOR_BGR2RGB)
    manual = cv2.imread(str(ROOT / "datasets/DRIVE/test/1st_manual/01.png"), cv2.IMREAD_GRAYSCALE) > 127
    c, model = load_geometry_model(None, str(ROOT / "checkpoints/sge_R2_seed42/sge.pth"), torch.device("cpu"))
    clahe = ev.CLAHEEnhancement() if c.get("preprocessing", {}).get("clahe", {}).get("enabled", True) else None
    x = clahe(rgb) if clahe is not None else rgb
    t = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        f = model.geometry_estimator(t)
    radius = torch.exp(f.log_radius)[0, 0].numpy()
    probs = f.level_probabilities[0].numpy()
    return rgb, manual, radius, probs


def pick(manual, radius_ref, target):
    sk = skeletonize(manual)
    ys, xs = np.nonzero(sk)
    r = radius_ref[ys, xs]
    i = np.argmin(np.abs(r - target) + 0.001 * np.hypot(ys - 292, xs - 282) * (r > 0))
    return ys[i], xs[i]


def arrow(fig, a, b):
    fig.patches.append(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10,
                                       color=MUTED, lw=1.0))


def main() -> None:
    rgb, manual, radius, probs = run()
    ref = mask_radius(manual)
    thin, wide = pick(manual, ref, 1.0), pick(manual, ref, 3.5)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8})
    W, H = 7.2, 3.0
    fig = plt.figure(figsize=(W, H))
    asp = W / H  # a square image of figure-width w has figure-height w * asp

    # (a) input
    ax = fig.add_axes([0.0, 0.22, 0.15, 0.15 * asp]); ax.imshow(rgb); ax.set_axis_off()
    ax.set_title("(a) Image $I$", fontsize=8.5, color=INK)
    for (y, xx), col in ((thin, ACCENT), (wide, SECOND)):
        ax.plot(xx, y, "o", ms=6, mfc="none", mec=col, mew=1.5)

    # (b) pyramid: levels 0, 2, 4, 6 at their relative sizes, bottom-aligned
    x0, base = 0.175, 0.30
    for lev in (0, 2, 4, 6):
        w = 0.10 * 2 ** (-lev / 2)
        a = fig.add_axes([x0, base, w, w * asp])
        a.imshow(cv2.resize(rgb, None, fx=2 ** (-lev / 2), fy=2 ** (-lev / 2), interpolation=cv2.INTER_AREA))
        a.set_axis_off()
        fig.text(x0 + w / 2, base - 0.025, f"$\ell$={lev}", ha="center", va="top", fontsize=7, color=MUTED)
        x0 += w + 0.012
    fig.text(x0, base + 0.02, "…", fontsize=10, color=MUTED)
    fig.text(0.175, 0.86, "(b) Pyramid, 12 levels", fontsize=8.5, color=INK)
    fig.text(0.175, 0.15, "level $\ell$: $I$ resized by $2^{-\ell/2}$", fontsize=7.5, color=MUTED)

    # (c) shared U-Net
    arrow(fig, (0.395, 0.50), (0.425, 0.50))
    fig.patches.append(FancyBboxPatch((0.432, 0.36), 0.115, 0.30, boxstyle="round,pad=0.008,rounding_size=0.015",
                                      transform=fig.transFigure, fc="#eaf2fc", ec=ACCENT, lw=1.2))
    fig.text(0.4895, 0.585, "U-Net $g$", ha="center", fontsize=9, color=INK, weight="bold")
    fig.text(0.4895, 0.49, "same weights\nat every level", ha="center", va="center", fontsize=7.5, color=INK)
    fig.text(0.4895, 0.395, "488,325 params", ha="center", fontsize=7, color=MUTED)
    fig.text(0.432, 0.86, "(c) Shared network", fontsize=8.5, color=INK)
    fig.text(0.432, 0.31, "per level: scale logit $s_\ell$,\ncorrection $c_\ell$, orientation,\nband; resized to $I$",
             fontsize=7, color=MUTED, va="top")

    # (d) softmax over levels, real values at the two marked pixels
    arrow(fig, (0.555, 0.50), (0.585, 0.50))
    ax = fig.add_axes([0.635, 0.30, 0.16, 0.50])
    lv = np.arange(probs.shape[0])
    ax.bar(lv - 0.2, probs[:, thin[0], thin[1]], width=0.4, color=ACCENT, label=f"thin (ref. {ref[thin]:.1f} px)")
    ax.bar(lv + 0.2, probs[:, wide[0], wide[1]], width=0.4, color=SECOND, label=f"wide (ref. {ref[wide]:.1f} px)")
    ax.set_xlabel("level $\ell$", labelpad=1); ax.set_ylabel("$p_\ell$", labelpad=1)
    ax.set_xticks([0, 3, 6, 9, 11]); ax.set_ylim(0, 1)
    ax.tick_params(length=2, labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1)
    fig.text(0.60, 0.86, "(d) Softmax over levels", fontsize=8.5, color=INK)
    fig.text(0.432, 0.05, r"Readout: $\ln\hat r = \ln r_0 + \sum_\ell p_\ell\,[\,\ell\,\frac{1}{2}\ln 2"
             r" + \frac{1}{4}\ln 2\,\tanh c_\ell\,]$, $r_0 = 0.5$ px", fontsize=8, color=INK)

    # (e) radius field, shown inside the manual vessel mask
    arrow(fig, (0.81, 0.50), (0.84, 0.50))
    ax = fig.add_axes([0.85, 0.22, 0.15, 0.15 * asp])
    ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cmap="gray", vmin=0, vmax=600)
    im = ax.imshow(np.ma.masked_where(~manual, radius), cmap="Blues", vmin=0.5, vmax=4.5)
    ax.set_axis_off()
    ax.set_title("(e) Radius $\hat r$", fontsize=8.5, color=INK)
    cb = fig.add_axes([0.87, 0.14, 0.11, 0.025])
    fig.colorbar(im, cax=cb, orientation="horizontal", ticks=[1, 2, 3, 4]).ax.tick_params(labelsize=7, length=2)
    fig.text(0.925, 0.03, "px", ha="center", fontsize=7, color=MUTED)

    out = ROOT / "evaluation_outputs" / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig1_architecture.png", dpi=600, facecolor="white")
    fig.savefig(out / "fig1_architecture.pdf", facecolor="white")
    print(f"saved {out / 'fig1_architecture.png'}; thin r_hat {radius[thin]:.2f}, wide r_hat {radius[wide]:.2f}")


if __name__ == "__main__":
    main()
