"""Second-observer test (pre-specified in PRESPECIFICATION_LOG.md, 2026-11-12, before scoring).

Is the estimator measuring vessels, or reproducing one annotator's drawing style?
Every method is scored against BOTH manual annotations of DRIVE_test and CHASE_DB1.

  DRIVE_test : obs1 = test/1st_manual (used so far, and for training), obs2 = test/2nd_manual
  CHASE_DB1  : obs2 = *_2ndHO.png (used so far), obs1 = *_1stHO.png (conventional reference)

Methods: R2 and P1 (standalone, seeds 42/123/7), mask width from the transformer U-Net probability
maps at the DRIVE-calibrated threshold (as Table 3) and at 0.5, and, on shared pixels, the other
human observer. Radius definition everywhere: max(EDT - 0.5, 0.5) of the mask, propagated from
the nearest skeleton pixel (src.mask_calibre_baseline.mask_radius).

Protocols, per image and reference observer X (other observer Y):
  full   : X's centreline pixels (the paper's protocol)
  shared : X's centreline pixels covered by Y's mask; the human method is Y's radius field

    python -m src.second_observer                  # inference + analysis (GPU, a few minutes)
    python -m src.second_observer --limit 2        # smoke test
    python -m src.second_observer --analyse-only   # re-run the tables from the saved CSV
Output: evaluation_outputs/calibre/second_observer{.md,_per_image.csv}
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

from src import common as ev
from src.common import read_gray
from src.common import Geometry, load_geometry_model
from src.mask_calibre_baseline import holm, mask_radius, scores
from src.test_sge_temperature import GEO

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "evaluation_outputs" / "calibre"
CSR = ROOT / "evaluation_outputs" / "csr_probs"
SEEDS = (42, 123, 7)
MASK_RUNS = {42: "MFC_P1_warm_native", 123: "MFC_P1_warm_s123_native", 7: "MFC_P1_warm_s7_native"}
OBSERVER_NAMES = {("DRIVE_test", "obs1"): "1st_manual", ("DRIVE_test", "obs2"): "2nd_manual",
                  ("CHASE_DB1", "obs1"): "1stHO", ("CHASE_DB1", "obs2"): "2ndHO"}
USED_SO_FAR = {"DRIVE_test": "obs1", "CHASE_DB1": "obs2"}


def load(path: Path) -> np.ndarray:
    return ev.load_binary_mask(path) > 0.5


def observer_files(dataset: str, pairs) -> list[tuple[Path, Path]]:
    """(obs1, obs2) per processed image, in pair_files order; checks the processed mask is the one used."""
    if dataset == "DRIVE_test":
        d = ROOT / "datasets" / "DRIVE" / "test"
        files = [(d / "1st_manual" / f"{i + 1:02d}.png", d / "2nd_manual" / f"{i + 1:02d}.png") for i in range(len(pairs))]
    else:
        d = ROOT / "datasets" / "CHASE_DB1"
        second = sorted(d.glob("*_2ndHO.png"))
        files = [(p.with_name(p.name.replace("_2ndHO", "_1stHO")), p) for p in second]
    if len(files) != len(pairs):
        raise ValueError(f"{dataset}: {len(files)} observer files vs {len(pairs)} images")
    used = 0 if USED_SO_FAR[dataset] == "obs1" else 1
    for pair, f in zip(pairs, files):
        if not np.array_equal(load(pair.mask_path), load(f[used])):
            raise ValueError(f"{dataset}: {pair.mask_path.name} is not {f[used].name}")
    return files


def thresholds() -> dict[int, float]:
    t = pd.read_csv(CAL / "mask_baseline_per_image.csv")
    t = t[(t.model == "maskP1") & (t.rule.astype(str) == "cal")]
    return {s: float(t[t.seed == s].threshold.iloc[0]) for s in SEEDS}


def compute(limit: int | None, model_names=("R2", "P1")) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for s in SEEDS:
        for m in model_names:
            c, net = load_geometry_model(None, str(ROOT / f"checkpoints/sge_{m}_seed{s}/sge.pth"), device)
            models[(m, s)] = Geometry(c, net, device)
    thr = thresholds()
    config = ev.load_config(GEO[0])
    rows = []
    for dataset in ("DRIVE_test", "CHASE_DB1"):
        spec = ev.resolve_datasets(config, [dataset])[dataset]
        pairs = ev.pair_files(spec.image_dir, spec.mask_dir)
        files = observer_files(dataset, pairs)
        for i, (pair, (f1, f2)) in enumerate(tqdm(list(zip(pairs, files))[:limit], desc=dataset)):
            image = ev.load_rgb_image(pair.image_path)
            obs = {"obs1": load(f1), "obs2": load(f2)}
            field = {k: mask_radius(v) for k, v in obs.items()}
            est = {}
            for (m, s), g in models.items():
                est[(m, s)] = g(image)[0]
            for s in SEEDS:
                prob = read_gray(CSR / MASK_RUNS[s] / dataset / "predictions" / "probability" / f"{i + 1:03d}.png") / 255.0
                est[("maskP1", s)] = mask_radius(prob >= thr[s])
                est[("maskP1_0.5", s)] = mask_radius(prob >= 0.5)
            for x, y in (("obs1", "obs2"), ("obs2", "obs1")):
                skel = skeletonize(obs[x])
                true = field[x][skel]
                shared = obs[y][skel]
                base = {"dataset": dataset, "i": i, "reference": x}
                for (m, s), r in est.items():
                    ratio = r[skel] / true
                    rows.append({**base, "protocol": "full", "method": m, "seed": s, "coverage": 1.0, **scores(ratio)})
                    rows.append({**base, "protocol": "shared", "method": m, "seed": s,
                                 "coverage": float(shared.mean()), **scores(ratio[shared])})
                rows.append({**base, "protocol": "shared", "method": "human", "seed": 0,
                             "coverage": float(shared.mean()), **scores(field[y][skel][shared] / true[shared])})
    return pd.DataFrame(rows)


def per_image(t: pd.DataFrame, dataset: str, ref: str, protocol: str, method: str) -> pd.Series:
    d = t[(t.dataset == dataset) & (t.reference == ref) & (t.protocol == protocol) & (t.method == method)]
    return d.groupby("i").acc25.mean()


def compare(t, dataset, ref, protocol, a, b):
    x, y = per_image(t, dataset, ref, protocol, a), per_image(t, dataset, ref, protocol, b)
    diff = x - y
    p = wilcoxon(x, y).pvalue if len(diff) > 1 and np.any(diff != 0) else float("nan")
    return x.mean(), y.mean(), diff.mean(), (diff > 0).mean(), p


def analyse(t: pd.DataFrame) -> str:
    n = {d: t[t.dataset == d].i.nunique() for d in ("DRIVE_test", "CHASE_DB1")}
    lines = ["# Second-observer test", "",
             "Pre-specified in PRESPECIFICATION_LOG.md (2026-11-12). Acc25, each image averaged over seeds 42/123/7, "
             "paired Wilcoxon over images. obs1/obs2: DRIVE 1st_manual/2nd_manual, CHASE_DB1 1stHO/2ndHO. "
             f"Images: DRIVE_test {n['DRIVE_test']}, CHASE_DB1 {n['CHASE_DB1']}. "
             "Results before this test used DRIVE obs1 and CHASE_DB1 obs2.", ""]

    lines += ["## Primary (Holm over O1, O2): CHASE_DB1 against obs1 (1stHO), full protocol", "",
              "| Test | R2 | Other | Diff | R2 better on | p | Holm p | Verdict |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    prim = {"O1. R2 - P1": compare(t, "CHASE_DB1", "obs1", "full", "R2", "P1"),
            "O2. R2 - maskP1": compare(t, "CHASE_DB1", "obs1", "full", "R2", "maskP1")}
    adj = holm({k: v[4] for k, v in prim.items()})
    for k, (a, b, d, w, p) in prim.items():
        ok = d > 0 and adj[k] < 0.05
        lines.append(f"| {k} | {a:.3f} | {b:.3f} | {d:+.3f} | {w:.0%} | {p:.2g} | {adj[k]:.2g} | "
                     f"{'**SUPPORTED**' if ok else '**NOT supported**'} |")

    lines += ["", "## Full protocol, all references (R2 vs others, uncorrected; O3 = DRIVE obs2)", "",
              "| Dataset | Reference | R2 | P1 | maskP1 | maskP1 @0.5 | R2-P1 (p) | R2-maskP1 (p) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for d in ("DRIVE_test", "CHASE_DB1"):
        for ref in ("obs1", "obs2"):
            m = {k: per_image(t, d, ref, "full", k).mean() for k in ("R2", "P1", "maskP1", "maskP1_0.5")}
            c1, c2 = compare(t, d, ref, "full", "R2", "P1"), compare(t, d, ref, "full", "R2", "maskP1")
            tag = " (used so far)" if USED_SO_FAR[d] == ref else ""
            lines.append(f"| {d} | {ref} {OBSERVER_NAMES[(d, ref)]}{tag} | {m['R2']:.3f} | {m['P1']:.3f} | {m['maskP1']:.3f} | "
                         f"{m['maskP1_0.5']:.3f} | {c1[2]:+.3f} ({c1[4]:.2g}) | {c2[2]:+.3f} ({c2[4]:.2g}) |")

    lines += ["", "## O4. Human ceiling: shared pixels (X's centreline covered by Y's mask)", "",
              "| Dataset | Reference X | Coverage | Human (Y vs X) | R2 | P1 | maskP1 | R2 - human (p) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for d in ("DRIVE_test", "CHASE_DB1"):
        for ref in ("obs1", "obs2"):
            cov = t[(t.dataset == d) & (t.reference == ref) & (t.method == "human")].coverage.mean()
            m = {k: per_image(t, d, ref, "shared", k).mean() for k in ("human", "R2", "P1", "maskP1")}
            c = compare(t, d, ref, "shared", "R2", "human")
            lines.append(f"| {d} | {ref} {OBSERVER_NAMES[(d, ref)]} | {cov:.0%} | {m['human']:.3f} | {m['R2']:.3f} | "
                         f"{m['P1']:.3f} | {m['maskP1']:.3f} | {c[2]:+.3f} ({c[4]:.2g}) |")

    lines += ["", "## O5. Annotator style: median bias (estimate / reference), full protocol", "",
              "| Dataset | Reference | R2 bias | P1 bias | maskP1 bias | R2 Acc25 |", "| --- | --- | --- | --- | --- | --- |"]
    for d in ("DRIVE_test", "CHASE_DB1"):
        for ref in ("obs1", "obs2"):
            sel = t[(t.dataset == d) & (t.reference == ref) & (t.protocol == "full")]
            b = {k: sel[sel.method == k].groupby("i").bias.mean().median() for k in ("R2", "P1", "maskP1")}
            lines.append(f"| {d} | {ref} {OBSERVER_NAMES[(d, ref)]} | {b['R2']:.3f} | {b['P1']:.3f} | {b['maskP1']:.3f} | "
                         f"{per_image(t, d, ref, 'full', 'R2').mean():.3f} |")

    lines += ["", "## Per seed, full protocol, Acc25 (42/123/7)", "", "| Dataset | Reference | R2 | P1 | maskP1 |", "| --- | --- | --- | --- | --- |"]
    for d in ("DRIVE_test", "CHASE_DB1"):
        for ref in ("obs1", "obs2"):
            sel = t[(t.dataset == d) & (t.reference == ref) & (t.protocol == "full")]
            cells = ["/".join(f"{sel[(sel.method == k) & (sel.seed == s)].acc25.mean():.3f}" for s in SEEDS) for k in ("R2", "P1", "maskP1")]
            lines.append(f"| {d} | {ref} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Images per dataset (smoke test)")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--output", default="second_observer")
    ap.add_argument("--models", nargs="+", default=["R2", "P1"],
                    help="Checkpoints checkpoints/sge_{name}_seed{seed}/sge.pth. Other than R2 P1: per-image CSV "
                         "second_observer_{names}_per_image.csv only, no analysis (e.g. --models SEU)")
    a = ap.parse_args()
    if a.models != ["R2", "P1"]:
        csv = CAL / f"{a.output}_{'_'.join(a.models)}_per_image.csv"
        t = compute(a.limit, a.models)
        if a.limit:
            print(t.groupby(["dataset", "reference", "protocol", "method"]).size().to_string())
            return
        t.to_csv(csv, index=False)
        print(f"saved {csv}")
        return
    csv = CAL / f"{a.output}_per_image.csv"
    t = pd.read_csv(csv) if a.analyse_only else compute(a.limit)
    if not a.analyse_only:
        t.to_csv(csv, index=False)
    report = analyse(t)
    (CAL / f"{a.output}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
