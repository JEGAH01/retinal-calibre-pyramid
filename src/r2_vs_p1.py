"""R2 versus P1, three seeds (Table 3 of the paper; PRESPECIFICATION_LOG.md, 2026-11-10).

Acc25 per image averaged over seeds 42/123/7, then the mean over images. Paired Wilcoxon test per case;
Holm correction over the three unseen datasets (STARE, CHASE_DB1, HRF). DRIVE is in-domain and its
enlarged cases are reported with both scale-test protocols, uncorrected.

Inputs (src.test_sge_temperature):
  evaluation_outputs/calibre/sge_R2_standalone_per_image.csv   seed 42  (R2, P1)
  evaluation_outputs/calibre/sge_standalone_s{123,7}_per_image.csv      (R2, P1)
  evaluation_outputs/calibre/sge_drive_nativegrid_per_image.csv         (R2_s*, P1_s*; --native-grid)

    python -m src.r2_vs_p1
Output: evaluation_outputs/calibre/r2_vs_p1.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.mask_calibre_baseline import holm
from src.wide_scale_analysis import CAL, SEEDS, indexed

STANDALONE = {42: "sge_R2_standalone_per_image.csv", 123: "sge_standalone_s123_per_image.csv",
              7: "sge_standalone_s7_per_image.csv"}
CASES = ["DRIVE_test x1", "DRIVE_test x2 native", "DRIVE_test x2", "DRIVE_test x4 native", "DRIVE_test x4",
         "STARE x1", "CHASE_DB1 x1", "HRF x1"]
UNSEEN = ["STARE x1", "CHASE_DB1 x1", "HRF x1"]


def load() -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        t = indexed(pd.read_csv(CAL / STANDALONE[s]))
        t = t[t.estimator.isin(["R2", "P1"])]
        frames.append(t.assign(model=t.estimator, seed=s))
    t = indexed(pd.read_csv(CAL / "sge_drive_nativegrid_per_image.csv"))
    t = t[t.case.str.endswith("native")]
    frames.append(t.assign(model=t.estimator.str.split("_s").str[0], seed=t.estimator.str.split("_s").str[1].astype(int)))
    return pd.concat(frames)


def main() -> None:
    data = load()
    data = data[data.case.isin(CASES)]
    assert data.groupby(["model", "case", "i"]).seed.nunique().eq(len(SEEDS)).all(), "missing seeds"
    avg = data.groupby(["model", "case", "i"]).acc25.mean()
    res = {}
    for c in CASES:
        a, b = avg.loc[("R2", c)], avg.loc[("P1", c)]
        d = a - b
        res[c] = (a.mean(), b.mean(), d.mean(), (d > 0).mean(), wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0)
    adj = holm({c: res[c][4] for c in UNSEEN})
    lines = ["# R2 versus P1 (three seeds)", "",
             "Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon. Holm over STARE, CHASE_DB1 and HRF.", "",
             "| Case | R2 | P1 | diff | R2 better on | p | Holm p |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in CASES:
        a, b, d, w, p = res[c]
        lines.append(f"| {c} | {a:.3f} | {b:.3f} | {d:+.3f} | {w:.0%} | {p:.2g} | "
                     + (f"{adj[c]:.2g}" if c in adj else "-") + " |")
    out = CAL / "r2_vs_p1.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
