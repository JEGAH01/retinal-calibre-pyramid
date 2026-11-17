"""Learned scale-equivariant baseline SEU against R2 (pre-specified in PRESPECIFICATION_LOG.md, 2026-11-13,
before any SEU training).

Primary:
  E1. R2 - SEU on STARE, CHASE_DB1 (2ndHO), HRF, FIVES_test (native), Holm over four.
  E2. R2 - SEU against the true radius of the synthetic vessels at x1, x2, x4, Holm over three.
Secondary (uncorrected): SEU - P1 on every case; R2 - SEU on DRIVE x1, x2/x4 under both protocols and on
CHASE_DB1 1stHO; bias; per-seed values.

Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon over images.

Inputs (SEU), produced by scripts/reproduce.ps1:
  evaluation_outputs/calibre/sge_SEU_s{seed}_per_image.csv      (src.test_sge_temperature --native-grid)
  evaluation_outputs/calibre/synthetic_calibre_SEU_per_image.csv (src.synthetic_calibre --models SEU)
  evaluation_outputs/calibre/second_observer_SEU_per_image.csv   (src.second_observer --models SEU)
R2 and P1: the same per-image results as in the paper (src.scale_selection_baseline.learned).

    python -m src.seu_baseline_analysis
Output: evaluation_outputs/calibre/seu_baseline{.md,_per_image.csv}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.mask_calibre_baseline import holm
from src.scale_selection_baseline import learned
from src.wide_scale_analysis import CAL, SEEDS, indexed

E1 = ["STARE x1", "CHASE_DB1 x1", "HRF x1", "FIVES_test x1"]
E2 = ["synthetic x1", "synthetic x2", "synthetic x4"]
ORDER = ["DRIVE_test x1", "DRIVE_test x2", "DRIVE_test x2 native", "DRIVE_test x4", "DRIVE_test x4 native",
         "STARE x1", "CHASE_DB1 x1", "CHASE_DB1 1stHO x1", "HRF x1", "FIVES_test x1",
         "synthetic x1", "synthetic x2", "synthetic x4"]


def seu() -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        t = indexed(pd.read_csv(CAL / f"sge_SEU_s{s}_per_image.csv"))
        frames.append(t[t.case.isin(ORDER)].assign(model="SEU", seed=s))
    so = pd.read_csv(CAL / "second_observer_SEU_per_image.csv")
    so = so[(so.dataset == "CHASE_DB1") & (so.reference == "obs1") & (so.protocol == "full") & (so.method == "SEU")]
    frames.append(so.assign(case="CHASE_DB1 1stHO x1", model="SEU"))
    sy = pd.read_csv(CAL / "synthetic_calibre_SEU_per_image.csv")
    sy = sy[(sy.protocol == "truth") & (sy.method == "SEU")]
    frames.append(sy.assign(case="synthetic x" + sy.factor.map(lambda f: f"{f:g}"), i=sy.image, model="SEU"))
    data = pd.concat(frames)[["case", "model", "seed", "i", "acc25", "bias"]]
    n = data.groupby(["case", "seed"]).size().unstack("seed")
    if n.isna().any().any() or (n.nunique(axis=1) > 1).any() or set(n.index) != set(ORDER):
        raise ValueError(f"SEU results incomplete or uneven across seeds:\n{n}")
    return data


def test(x: pd.Series, y: pd.Series):
    y = y.loc[x.index]
    d = x - y
    return x.mean(), y.mean(), d.mean(), (d > 0).mean(), (wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0)


def main() -> None:
    data = pd.concat([learned()[["case", "model", "seed", "i", "acc25", "bias"]], seu()])
    data = data[data.case.isin(ORDER) & data.model.isin(["R2", "P1", "SEU"])]
    data.to_csv(CAL / "seu_baseline_per_image.csv", index=False)
    avg = data.groupby(["model", "case", "i"]).acc25.mean()
    bias = data.groupby(["model", "case", "i"]).bias.mean()
    m = lambda model, case: avg.loc[(model, case)]

    L = ["# Learned scale-equivariant baseline (SEU)", "",
         "Pre-specified in PRESPECIFICATION_LOG.md (2026-11-13). SEU: scale-equivariant U-Net (Sangalli et al., 2022), "
         "488,325 parameters as R2. Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon over images.", ""]
    for title, cases in (("E1. Real unseen data (primary, Holm over four)", E1),
                         ("E2. Synthetic, against the true radius (primary, Holm over three)", E2)):
        res = {c: test(m("R2", c), m("SEU", c)) for c in cases}
        adj = holm({c: r[4] for c, r in res.items()})
        L += [f"## {title}", "", "| Case | R2 | SEU | R2 - SEU | R2 better on | Holm p | Verdict |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for c, (a, b, d, w, _) in res.items():
            v = ("**R2 better**" if d > 0 and adj[c] < 0.05 else
                 "**SEU better**" if d < 0 and adj[c] < 0.05 else "no significant difference")
            L.append(f"| {c} | {a:.3f} | {b:.3f} | {d:+.3f} | {w:.0%} | {adj[c]:.2g} | {v} |")
        L.append("")

    L += ["## Secondary (uncorrected) and all cases", "",
          "| Case | R2 | P1 | SEU | R2 - SEU (p) | SEU - P1 (p) | Median bias R2 / SEU |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in ORDER:
        r, q = test(m("R2", c), m("SEU", c)), test(m("SEU", c), m("P1", c))
        L.append(f"| {c} | {r[0]:.3f} | {q[1]:.3f} | {r[1]:.3f} | {r[2]:+.3f} ({r[4]:.2g}) | {q[2]:+.3f} ({q[4]:.2g}) | "
                 f"{bias.loc[('R2', c)].median():.3f} / {bias.loc[('SEU', c)].median():.3f} |")

    L += ["", "## SEU per seed (42/123/7)", "", "| Case | Acc25 |", "| --- | --- |"]
    d = data[data.model == "SEU"]
    for c in ORDER:
        L.append(f"| {c} | " + "/".join(f"{d[(d.case == c) & (d.seed == s)].acc25.mean():.3f}" for s in SEEDS) + " |")
    report = "\n".join(L) + "\n"
    (CAL / "seu_baseline.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
