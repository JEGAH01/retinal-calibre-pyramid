"""Weight-sharing ablation: standalone R2 (shared LevelUNet) vs R2ns (one LevelUNet per level).

Pre-specified (PRESPECIFICATION_LOG.md, 2026-11-11, before training R2ns): Acc25, each image
averaged over seeds 42/123/7, paired Wilcoxon, Holm over the three cases where R2 beat P1
(CHASE_DB1 x1, HRF x1, DRIVE_test x2 native). All other cases reported uncorrected.

Inputs: R2 x1 cases from the standalone CSVs, R2 x2/x4 native from sge_drive_nativegrid,
R2ns from sge_R2ns_s{seed}_per_image.csv (scripts/reproduce.ps1).

    python -m src.sharing_ablation_analysis
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "evaluation_outputs" / "calibre"
SEEDS = [42, 123, 7]
PRIMARY = ["CHASE_DB1 x1", "HRF x1", "DRIVE_test x2 native"]
CASES = ["DRIVE_test x1", "DRIVE_test x2 native", "DRIVE_test x4 native", "STARE x1", "CHASE_DB1 x1", "HRF x1"]
STANDALONE = {42: "sge_R2_standalone_per_image.csv", 123: "sge_standalone_s123_per_image.csv",
              7: "sge_standalone_s7_per_image.csv"}


def indexed(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["i"] = t.groupby(["case", "estimator"]).cumcount()
    return t


def load() -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        t = indexed(pd.read_csv(CAL / STANDALONE[s]))
        t = t[(t.estimator == "R2") & t.case.str.endswith(" x1")]
        frames.append(t.assign(model="R2", seed=s))
    t = indexed(pd.read_csv(CAL / "sge_drive_nativegrid_per_image.csv"))
    t = t[t.estimator.str.startswith("R2_") & t.case.str.endswith("native")]
    frames.append(t.assign(model="R2", seed=t.estimator.str.split("_s").str[1].astype(int)))
    for s in SEEDS:
        t = indexed(pd.read_csv(CAL / f"sge_R2ns_s{s}_per_image.csv"))
        frames.append(t.assign(model="R2ns", seed=s))
    return pd.concat(frames)


def main() -> None:
    data = load()
    data = data[data.case.isin(CASES)]
    avg = data.groupby(["model", "case", "i"]).acc25.mean()
    per_seed = data.groupby(["model", "case", "seed"]).acc25.mean()
    res = {}
    for c in CASES:
        a, b = avg.loc[("R2", c)], avg.loc[("R2ns", c)]
        d = a - b
        res[c] = (a.mean(), b.mean(), d.mean(), (d > 0).mean(), wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0)
    order = sorted(PRIMARY, key=lambda c: res[c][4])
    adj, running = {}, 0.0
    for k, c in enumerate(order):
        running = max(running, min(1.0, (len(order) - k) * res[c][4]))
        adj[c] = running
    lines = ["# Weight-sharing ablation: R2 (shared) vs R2ns (per-level networks)", "",
             "Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon. Holm over the pre-specified cases.", "",
             "| Case | R2 | R2ns | diff | R2 better on | p | Holm p | R2ns per seed |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in CASES:
        a, b, d, w, p = res[c]
        seeds = "/".join(f"{per_seed.loc[('R2ns', c, s)]:.3f}" for s in SEEDS)
        lines.append(f"| {c} | {a:.3f} | {b:.3f} | {d:+.3f} | {w:.0%} | {p:.2g} | "
                     + (f"{adj[c]:.2g}" if c in adj else "-") + f" | {seeds} |")
    better = [c for c in PRIMARY if res[c][2] > 0 and adj[c] < 0.05]
    worse = [c for c in PRIMARY if res[c][2] < 0 and adj[c] < 0.05]
    lines += ["", f"- Shared better (Holm p < 0.05): {', '.join(better) or 'none'}",
              f"- Non-shared better (Holm p < 0.05): {', '.join(worse) or 'none'}",
              "- Interpretation, fixed in advance: shared better on all three -> weight sharing contributes to the gain; "
              "no significant difference -> the gain comes from the pyramid with scale selection, and sharing gives the same "
              "accuracy with 12x fewer parameters; non-shared better -> sharing costs accuracy (report it)."]
    out = CAL / "sharing_ablation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
