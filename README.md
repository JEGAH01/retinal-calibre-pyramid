# Scale-selective estimation of retinal vessel calibre with a shared-weight image pyramid

Code, trained weights, per-image results and the pre-specification log for the paper

> J. Aliyu, L. Hanye. *Scale-selective estimation of retinal vessel calibre with a shared-weight image pyramid.*
> Submitted to Computerized Medical Imaging and Graphics.

R2 estimates the local vessel radius at every pixel of a fundus image. A small U-Net is applied with shared weights
to every level of a 12-level half-octave image pyramid, and a soft-argmax over levels reads out the radius. It is
trained only on the 20 DRIVE training images and scored, without retraining, on STARE, CHASE_DB1, HRF, FIVES,
enlarged DRIVE images and synthetic vessels of known width.

## Contents

| Path | What it is |
| --- | --- |
| `models/medial_frame.py` | R2 (`ScaleEquivariantGeometryEstimator`, `type: pyramid_unet`), R2ns (`share_weights: false`), P1 (`PlainGeometryEstimator`), the geometry targets and losses |
| `models/scale_equivariant_unet.py` | SEU, the scale-equivariant U-Net baseline (independent implementation of Sangalli et al., 2022) |
| `src/pretrain_sge.py` | Training of one estimator |
| `src/test_sge_temperature.py` | Calibre scoring (Acc25, error, bias per image) |
| `src/*_analysis.py`, `src/r2_vs_p1.py`, `src/mask_*.py`, `src/second_observer.py`, `src/synthetic_calibre.py`, `src/scale_selection_baseline.py`, `src/seu_checks.py` | The pre-specified comparisons (see the table below) |
| `src/prepare_datasets.py`, `src/generate_topology_targets.py`, `src/datasets.py`, `src/topology_datasets.py` | Data preparation and the training loader |
| `configs/` | One config per estimator and seed (`sge_<estimator>_seed<s>.yaml`), and `data.yaml` |
| `checkpoints/sge_<estimator>_seed<s>/sge.pth` | The 18 trained estimators of the paper (R2, P1, R2ns, R2w, P1w, SEU; seeds 42, 123, 7) |
| `evaluation_outputs/calibre/` | Per-image scores and analysis outputs behind every table |
| `PRESPECIFICATION_LOG.md` | Dated log: each comparison and its decision rule was written down before it was run |
| `scripts/reproduce.ps1` | Runs everything in order |
| `scripts/make_supplement.py` | Builds the supplementary tables from the per-image CSVs |

| Paper | Script | Output in `evaluation_outputs/calibre/` |
| --- | --- | --- |
| Table 3 | `src.r2_vs_p1` | `r2_vs_p1.md` |
| Table 4, S4 | `src.mask_threshold_validation` | `mask_threshold_validation.md` |
| Table 5 | `src.sharing_ablation_analysis` | `sharing_ablation.md` |
| Table 6, S3 | `src.fives_calibre_analysis`, `src.wide_scale_analysis` | `fives_calibre.md`, `wide_scale.md` |
| Tables 7–8 | `src.synthetic_calibre` | `synthetic_calibre.md` |
| Table 9 | `src.second_observer` | `second_observer.md` |
| Table 10 | `src.scale_selection_baseline` | `scale_selection_baseline.md` |
| Table 11 | `src.seu_baseline_analysis` | `seu_baseline.md` |
| Tables S5–S6 | `src.mask_calibre_baseline` | `mask_baseline.md` |
| Table S7 | `src.seu_checks` | `seu_checks.txt` (printed) |
| Tables S1–S8 | `scripts/make_supplement.py` | `../supplementary_tables.md` |
| Figure 1 | `src.fig_architecture` | `../figures/fig1_architecture.*` |
| Figures 2–4 | `src.fig_results` (Figure 2 needs the datasets and the mask probability maps; `--only 3 4` runs from the CSVs alone) | `../figures/fig{2,3,4}_*` |

## Installation

Python 3.10 and PyTorch 2.5 (CUDA optional; every script also runs on the CPU, slowly):

```
pip install -r requirements.txt
```

## Data

The datasets are not redistributed here. Download them from their providers:
[DRIVE](https://drive.grand-challenge.org/), [STARE](https://cecas.clemson.edu/~ahoover/stare/),
[CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/),
[HRF](https://www5.cs.fau.de/research/data/fundus-images/) and
[FIVES](https://doi.org/10.6084/m9.figshare.19688169). Place them as follows (DRIVE as PNG files):

```
datasets/
  DRIVE/training/{image,1st_manual,masks}/21.png ... 40.png
  DRIVE/test/{images,1st_manual,2nd_manual,masks}/01.png ... 20.png
  STARE/{train,val}/{input,label}/
  CHASE_DB1/Image_01L.jpg, Image_01L_1stHO.png, Image_01L_2ndHO.png, ...
  HRF/{images,manual1,mask}/
  FIVES/{train,test}/{Original,Ground truth}/
```

Then, from the repository root:

```
python -m src.prepare_datasets
python -m src.generate_topology_targets --config configs/data.yaml --dataset DRIVE_train
```

`prepare_datasets` writes the test sets to `datasets_processed/`. For CHASE_DB1 it pairs each image with the
**second** observer (`*_2ndHO.png`); the paper's results refer to this observer unless stated otherwise, and the
first observer is used in the second-observer test (Table 9).

## Usage

Score R2 and P1 (seed 42) on the four standard test sets:

```
python -m src.test_sge_temperature --temperatures 1 --output my_scores.md \
    --estimators R2=checkpoints/sge_R2_seed42/sge.pth P1=checkpoints/sge_P1_seed42/sge.pth
```

Radius map for one image, in Python:

```python
import torch
from src.common import Geometry, load_geometry_model, load_rgb_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config, model = load_geometry_model(None, "checkpoints/sge_R2_seed42/sge.pth", device)
radius, band = Geometry(config, model, device)(load_rgb_image("datasets/DRIVE/test/images/01.png"))
```

Train one estimator (20–30 minutes on one GPU):

```
python -m src.pretrain_sge --config configs/sge_R2_seed42.yaml --epochs 5 --output checkpoints/sge_R2_seed42_retrained
```

Reproduce all tables (Windows PowerShell; skips steps whose outputs exist):

```
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1             # analyses from the released scores
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Rescore    # also re-score every estimator
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Train      # also retrain the 18 estimators
```

## Mask-width baseline

The mask-width baseline measures the radius from the masks of segmentation networks (two transformer U-Nets and a
plain U-Net). These networks are not part of this repository. Their probability maps on the test sets are
provided as a separate archive, `mask_probabilities.zip` (see the release page); unzip it into
`evaluation_outputs/csr_probs/`. In the code and the log the networks appear under their internal run names
(`MFC_P1_warm_*`: the transformer U-Net of the paper; `MFC_R2_warm_*`: the second transformer U-Net;
`ExternalUNet_*`: the plain U-Net). Without the archive, the mask-width scores can still be re-analysed from
the released per-image CSVs.

## Notes on reproducibility

- The analyses re-run from the released per-image CSVs reproduce the released outputs exactly, and re-scoring
  the released weights reproduces the per-image scores.
- Training is not bit-for-bit reproducible: the random patch sampling and augmentation of the training loader
  are not seeded. Retrained estimators are therefore close to, but not identical with, the released ones.
- The configs and the copies of them stored in `sge.pth` contain only the settings the estimators use; the
  weights are unchanged.
- `PRESPECIFICATION_LOG.md` is an excerpt of the authors' working log. Entries are copied unchanged; entries
  about experiments outside the paper are not included, and two sentences about a separate segmentation study
  are replaced by a marked note.
- The seed-42 score file `sge_R2_standalone_per_image.csv` is limited to R2 and P1.

## Licence

The code is released under the MIT licence (`LICENSE`). The datasets keep their own licences.
