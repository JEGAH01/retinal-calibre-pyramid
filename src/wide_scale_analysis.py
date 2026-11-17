"""Wide-scale training test (pre-specified in PRESPECIFICATION_LOG.md, 2026-11-12, before training).

R2w and P1w are R2 and P1 retrained with scale augmentation widened from [0.7, 3.2] to [0.7, MAX]
(configs/sge_{R2w,P1w}_seed{42,123,7}.yaml; nothing else changed). Acc25, each image averaged
over seeds 42/123/7, paired Wilcoxon.

Primary (Holm over the two):
  A. R2w - maskP1 on native FIVES_test > 0   (does wide training close the gap to mask width?)
  B. R2w - P1w   on native FIVES_test > 0    (does the architecture advantage hold under the same wide training?)
Secondary:
  C. R2w - R2 on native FIVES_test (did widening help R2?).
  D. Cost check, R2w - R2 on DRIVE_test x1, DRIVE_test x2 native, STARE x1, CHASE_DB1 x1, HRF x1,
     Holm over the five; a significant loss is a cost and is reported.
  E. R2w - P1w on the same five cases and on FIVES_test x0.27 native (uncorrected).
Model-selection rule: R2w replaces R2 as the main estimator only if A and B are supported and D shows
no significant loss; otherwise R2 stays the main estimator and R2w is reported as a separate experiment.

Inputs:
  evaluation_outputs/calibre/sge_wide_s{seed}_per_image.csv   (R2w_s{seed}, P1w_s{seed}; scripts/reproduce.ps1)
  R2/P1 originals: standalone CSVs, sge_drive_nativegrid, sge_fives_s{seed}; maskP1: fives_mask_per_image.csv

    python -m src.wide_scale_analysis
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "evaluation_outputs" / "calibre"
SEEDS = [42, 123, 7]
STANDALONE = {42: "sge_R2_standalone_per_image.csv", 123: "sge_standalone_s123_per_image.csv",
              7: "sge_standalone_s7_per_image.csv"}
FIVES = "FIVES_test x1"
COST = ["DRIVE_test x1", "DRIVE_test x2 native", "STARE x1", "CHASE_DB1 x1", "HRF x1"]
SMALL = "FIVES_test x0.27 native"


def indexed(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["i"] = t.groupby(["case", "estimator"]).cumcount()
    return t


def load() -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        t = indexed(pd.read_csv(CAL / STANDALONE[s]))
        t = t[t.estimator.isin(["R2", "P1"]) & t.case.str.endswith(" x1")]
        frames.append(t.assign(model=t.estimator, seed=s))
    t = indexed(pd.read_csv(CAL / "sge_drive_nativegrid_per_image.csv"))
    t = t[t.case.str.endswith("native")]
    frames.append(t.assign(model=t.estimator.str.split("_s").str[0], seed=t.estimator.str.split("_s").str[1].astype(int)))
    for s in SEEDS:
        for name in (f"sge_fives_s{s}_per_image.csv", f"sge_wide_s{s}_per_image.csv"):
            t = indexed(pd.read_csv(CAL / name))
            frames.append(t.assign(model=t.estimator.str.split("_s").str[0], seed=s))
    m = pd.read_csv(CAL / "fives_mask_per_image.csv")
    frames.append(m.assign(case=FIVES, estimator="maskP1"))
    data = pd.concat(frames)[["case", "model", "seed", "i", "acc25", "bias"]]
    data = data.drop_duplicates(["case", "model", "seed", "i"])
    n = data.groupby(["case", "model", "seed"]).size().unstack("seed")
    if n.isna().any().any() or (n.nunique(axis=1) > 1).any():
        raise ValueError(f"Uneven image counts across seeds:\n{n}")
    return data


def test(avg: pd.Series, case: str, a: str, b: str):
    x, y = avg.loc[(a, case)], avg.loc[(b, case)]
    y = y.loc[x.index]
    d = x - y
    return x.mean(), y.mean(), d.mean(), (d > 0).mean(), (wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0)


def holm(ps: dict) -> dict:
    out, running = {}, 0.0
    for k, (key, p) in enumerate(sorted(ps.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (len(ps) - k) * p))
        out[key] = running
    return out


def row(label, r, adj=None):
    a, b, d, w, p = r
    return f"| {label} | {a:.3f} | {b:.3f} | {d:+.3f} | {w:.0%} | {p:.2g} | " + ("-" if adj is None else f"{adj:.2g}") + " |"


def main() -> None:
    data = load()
    avg = data.groupby(["model", "case", "i"]).acc25.mean()
    per_seed = data.groupby(["model", "case", "seed"]).acc25.mean()
    bias = data.groupby(["model", "case"]).bias.median()
    head = ["| Comparison | first | second | diff | first better on | p | Holm p |", "| --- | --- | --- | --- | --- | --- | --- |"]
    lines = ["# Wide-scale training test: R2w / P1w", "",
             "Acc25, each image averaged over seeds 42/123/7, paired Wilcoxon.", "", "## Primary (native FIVES_test, Holm over A and B)", ""] + head
    prim = {"A": test(avg, FIVES, "R2w", "maskP1"), "B": test(avg, FIVES, "R2w", "P1w")}
    padj = holm({k: v[4] for k, v in prim.items()})
    lines += [row("A. R2w - maskP1", prim["A"], padj["A"]), row("B. R2w - P1w", prim["B"], padj["B"])]
    ok = {k: prim[k][2] > 0 and padj[k] < 0.05 for k in prim}

    lines += ["", "## Secondary", "", "C. Did widening help R2 on native FIVES?", ""] + head
    lines.append(row("R2w - R2 (FIVES x1)", test(avg, FIVES, "R2w", "R2")))
    cost = {c: test(avg, c, "R2w", "R2") for c in COST}
    cadj = holm({c: r[4] for c, r in cost.items()})
    lines += ["", "D. Cost check: R2w - R2 on the earlier cases (Holm over five)", ""] + head
    lines += [row(c, cost[c], cadj[c]) for c in COST]
    losses = [c for c in COST if cost[c][2] < 0 and cadj[c] < 0.05]
    lines += ["", "E. R2w - P1w elsewhere (uncorrected)", ""] + head
    lines += [row(c, test(avg, c, "R2w", "P1w")) for c in COST + [SMALL]]

    lines += ["", "## Per seed and bias", "", "| Model | Case | per seed (42/123/7) | median bias |", "| --- | --- | --- | --- |"]
    for m in ("R2", "P1", "R2w", "P1w", "maskP1"):
        for c in [FIVES] + COST + [SMALL]:
            if (m, c, SEEDS[0]) in per_seed.index:
                seeds = "/".join(f"{per_seed.loc[(m, c, s)]:.3f}" for s in SEEDS)
                lines.append(f"| {m} | {c} | {seeds} | {bias.loc[(m, c)]:.3f} |")

    main_model = "R2w" if ok["A"] and ok["B"] and not losses else "R2"
    lines += ["", f"- A (R2w beats mask width on native FIVES): **{'SUPPORTED' if ok['A'] else 'NOT supported'}**",
              f"- B (R2w beats P1w on native FIVES): **{'SUPPORTED' if ok['B'] else 'NOT supported'}**",
              f"- Significant losses vs R2 (Holm): {', '.join(losses) or 'none'}",
              f"- Main estimator by the pre-specified rule: **{main_model}**"]
    out = CAL / "wide_scale.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
