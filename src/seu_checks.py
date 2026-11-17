"""Pre-training checks for the SEU baseline (PRESPECIFICATION_LOG.md, step 2). Random weights,
no data scores.

1. Implementation check, exact semigroup action (Sangalli et al. 2022; Worrall and Welling 2019). A 256 px
   DRIVE crop E and x = (E blurred by the scale-1 Gaussian) subsampled by 2. For a correct implementation the
   scale-(k+1) responses of E, subsampled by 2, equal the scale-k responses of x up to the approximations the
   paper names (bilinear upsampling, borders). Reported after the encoder and for the whole network.
2. Equivariance under the scale-test protocol, SEU against R2. 256 px central crops of DRIVE test 01-03 are
   enlarged x2 and x4 (bilinear, as in the scale tests). Per-scale raw responses of the enlarged crop,
   area-averaged back to the native grid, against those of the native crop shifted by the matching number of
   scale steps (SEU: 1 and 2 octaves; R2: 2 and 4 half-octave levels). Error ||a - b|| / ||b|| over the vessel
   band (manual mask dilated by 3 px), mean and max over the levels compared. Filter depth 2 is included
   to document why depth 1 was chosen.
3. Per-scale evaluation is identical to joint evaluation (depth 1).
4. GPU memory and time: one training step (batch 2, 256 px, forward + backward) and inference on one
   2048x2048 (FIVES) and one 3504x2336 (HRF) image.

    python -m src.seu_checks
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

from models.medial_frame import build_geometry_estimator
from models.scale_equivariant_unet import gaussian_kernel1d

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT / "datasets" / "DRIVE" / "test"
R2_CFG = {"type": "pyramid_unet", "hidden_channels": 16, "levels": 12, "level_step": 0.5, "base_radius": 0.5}
SEU_CFG = {"type": "se_unet"}  # filter depth 1, P1 widths
SEU2_CFG = {"type": "se_unet", "scale_depth": 2, "widths": [16, 32, 52, 52, 52]}  # 488,085 parameters


def load(i: int):
    img = np.asarray(Image.open(DRIVE / "images" / f"{i:02d}.png").convert("RGB"), np.float32) / 255.0
    mask = np.asarray(Image.open(DRIVE / "1st_manual" / f"{i:02d}.png")) > 127
    img, mask = img[164:420, 154:410], mask[164:420, 154:410]  # 256 px central crop
    return (torch.from_numpy(img).permute(2, 0, 1)[None] - 0.5) / 0.25, mask


def enlarge(x, t):
    h, w = x.shape[-2:]
    return F.interpolate(x, size=(h * t, w * t), mode="bilinear", align_corners=False)


def blur(x, s):
    g = gaussian_kernel1d(s)
    r = g.numel() // 2
    b, c, h, w = x.shape
    y = x.reshape(b * c, 1, h, w)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1))
    return y.reshape(b, c, h, w)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@torch.no_grad()
def semigroup(cfg):
    torch.manual_seed(0)
    m = build_geometry_estimator(cfg).train()  # joint evaluation of all scales (GroupNorm is per scale)
    E, _ = load(1)
    x = blur(E, 0.5 * math.sqrt(3))[..., ::2, ::2]
    a, b = m.lifting(x), m.lifting(E)
    for block in m.encoder:
        a, b = block(a), block(b)
    enc = [rel(b[0, k + 1][..., ::2, ::2], a[0, k]) for k in range(a.shape[1] - 1)]
    ra, rb = m.level_responses(x), m.level_responses(E)
    full = [rel(rb[0, k + 1][..., ::2, ::2], ra[0, k]) for k in range(ra.shape[1] - 1)]
    return enc, full


@torch.no_grad()
def protocol(m, x, mask, t, shift):
    band = torch.from_numpy(ndi.binary_dilation(mask, iterations=3))
    a = m.level_responses(x)
    y = m.level_responses(enlarge(x, t))
    b = F.interpolate(y.flatten(0, 1), size=x.shape[-2:], mode="area").unflatten(0, y.shape[:2])
    return [rel(b[0, k + shift][:, band], a[0, k][:, band]) for k in range(a.shape[1] - shift)]


def main() -> None:
    print("1. Exact semigroup action (implementation check), relative error per scale k = 0..4")
    for name, cfg in (("SEU", SEU_CFG), ("SEU depth 2", SEU2_CFG)):
        enc, full = semigroup(cfg)
        print(f"   {name}: after encoder {[round(e, 3) for e in enc]}; whole network {[round(e, 3) for e in full]}")

    print("\n2. Equivariance under the scale-test protocol (random weights, 256 px crops of DRIVE test 01-03)")
    print("| Model | Factor | Response error, mean over levels | max |")
    print("| --- | --- | --- | --- |")
    for name, cfg in (("SEU", SEU_CFG), ("SEU depth 2", SEU2_CFG), ("R2", R2_CFG)):
        torch.manual_seed(0)
        m = build_geometry_estimator(cfg).eval()
        for t in (2, 4):
            shift = int(round(math.log2(t) / m.level_step))
            errs = [e for i in (1, 2, 3) for e in protocol(m, *load(i), t, shift)]
            print(f"| {name} | x{t} | {np.mean(errs):.3f} | {np.max(errs):.3f} |", flush=True)

    m = build_geometry_estimator(SEU_CFG)
    x = torch.randn(1, 3, 200, 232)
    with torch.no_grad():
        joint = m.train().level_responses(x)
        separate = m.eval().level_responses(x)
    print(f"\n3. Per-scale against joint evaluation: max |difference| = {(joint - separate).abs().max().item():.1e}")

    if not torch.cuda.is_available():
        print("no GPU; skipping 4")
        return
    print("\n4. GPU memory and time (fp32; peaks above 6144 MiB spill into shared memory on Windows)")
    dev = torch.device("cuda")
    print("| Model | Train step, 2x256 px | peak | FIVES 2048x2048 | peak | HRF 3504x2336 | peak |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, cfg in (("SEU", SEU_CFG), ("R2", R2_CFG)):
        m = build_geometry_estimator(cfg).to(dev)
        opt = torch.optim.AdamW(m.parameters(), 1e-4)
        xb = torch.randn(2, 3, 256, 256, device=dev)
        m.train()
        for i in range(4):
            if i == 1:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            loss = m(xb).log_radius.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        cells = [f"{(time.time() - t0) / 3:.3f} s", f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"]
        m.eval()
        for h, w in ((2048, 2048), (2336, 3504)):
            xi = torch.randn(1, 3, h, w, device=dev)
            try:
                with torch.no_grad():
                    m(xi); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
                    m(xi); torch.cuda.synchronize()
                cells += [f"{time.time() - t0:.1f} s", f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"]
            except torch.cuda.OutOfMemoryError:
                cells += ["out of memory", "-"]
            del xi
            torch.cuda.empty_cache()
        print(f"| {name} | " + " | ".join(cells) + " |", flush=True)
        del m, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
