"""Scale-equivariant U-Net radius estimator ("SEU"), a learned scale-equivariant baseline for R2.

Independent PyTorch implementation of the scale-equivariant U-Net of Sangalli et al. (2022, BMVC,
arXiv:2210.04508), which builds on the semigroup scale cross-correlation of Worrall and Welling (2019).
No code was taken from the authors' repository (TensorFlow, no licence). Design fixed in
PRESPECIFICATION_LOG.md (2026-11-13):

* Lifting: Gaussian scale space on the full grid, S = 6 octave scales (base 2),
  sigma_k^2 = sigma_0^2 (4^k - 1), sigma_0 = 0.5 px (k = 0 is the image itself).
* Scale cross-correlation: output scale k is a 3x3 convolution with dilation 2^k that reads the input
  scales k .. k+K_s-1 (coarser scales beyond the last are replaced by the last one, as in the reference
  code). Strided convolutions subsample, bilinear upsampling per scale upsamples (paper, Sections 4.1-4.2).
* Backbone: the P1 / LevelUNet layout and widths (16, 32, 64, 64, 64; GroupNorm(4), GELU, stride-2,
  concatenated skips). Filter depth 1 along scale, as in both experiments of the paper, so SEU has exactly
  R2's parameters (488,325). Depth 2 (scales interacting) was tried and lost equivariance through
  truncation at the coarsest scale (``scale_depth`` option, review log 2026-11-13).
* Readout: not the paper's max projection (scale invariant), but R2's soft-argmax over scales of
  (k ln 2 + bounded correction), with the correction bounded to +-0.5 octave. Output is a MedialFrameField
  exactly as for R2 and P1.

In eval mode with filter depth 1 the scales are computed one at a time (identical output, about 1/S of the
memory), so whole images are processed as for R2. Overlapping tiles (``tile`` > 0) remain available as a
fallback and are off by default.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.medial_frame import LN2, MedialFrameField


def gaussian_kernel1d(sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


class GaussianLifting(nn.Module):
    """Image [B,C,H,W] -> scale space [B,S,C,H,W]."""

    def __init__(self, number_of_scales: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.sigmas = [sigma0 * math.sqrt(4.0 ** k - 1.0) for k in range(number_of_scales)]
        for k, s in enumerate(self.sigmas):
            if k:
                self.register_buffer(f"kernel{k}", gaussian_kernel1d(s), persistent=False)

    def forward(self, image: torch.Tensor, scales=None) -> torch.Tensor:
        b, c, h, w = image.shape
        out = []
        for k in (range(len(self.sigmas)) if scales is None else scales):
            if k == 0:
                out.append(image)
                continue
            g = getattr(self, f"kernel{k}").to(image.dtype)
            r = g.numel() // 2
            x = image.reshape(b * c, 1, h, w)
            # reflect padding needs pad < size; fall back to replicate for tiny inputs
            mode = "reflect" if r < min(h, w) else "replicate"
            x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), g.view(1, 1, 1, -1))
            x = F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), g.view(1, 1, -1, 1))
            out.append(x.reshape(b, c, h, w))
        return torch.stack(out, dim=1)


class ScaleConv(nn.Module):
    """Semigroup scale cross-correlation, 3x3 spatial, depth ``scale_depth`` along scale."""

    def __init__(self, in_channels: int, out_channels: int, scale_depth: int = 1, stride: int = 1) -> None:
        super().__init__()
        self.scale_depth, self.stride = int(scale_depth), int(stride)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels * self.scale_depth, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # nn.Conv2d default
        self.k0 = 0  # index of the first scale in the input (set when scales are processed separately)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,S,C,H,W] -> [B,S,O,H',W']
        s = x.shape[1]
        out = []
        for k in range(s):
            idx = [min(k + l, s - 1) for l in range(self.scale_depth)]
            xk = x[:, idx].flatten(1, 2)
            d = 2 ** (self.k0 + k)
            out.append(F.conv2d(xk, self.weight, stride=self.stride, padding=d, dilation=d))
        return torch.stack(out, dim=1)


class ScaleGroupNorm(nn.Module):
    """GroupNorm applied per sample and per scale (commutes with shifts of the scale index)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape[:2]
        return self.norm(x.flatten(0, 1)).unflatten(0, (b, s))


def _scale_block(in_channels: int, out_channels: int, stride: int = 1, scale_depth: int = 1) -> nn.Sequential:
    return nn.Sequential(
        ScaleConv(in_channels, out_channels, 1, stride), ScaleGroupNorm(out_channels), nn.GELU(),
        ScaleConv(out_channels, out_channels, scale_depth), ScaleGroupNorm(out_channels), nn.GELU(),
    )


class ScaleEquivariantUNetEstimator(nn.Module):
    """SEU: scale-equivariant U-Net with R2's soft-argmax radius readout over octave scales."""

    def __init__(self, in_channels: int = 3, widths=(16, 32, 64, 64, 64), number_of_scales: int = 6,
                 base_radius: float = 0.5, sigma0: float = 0.5, temperature: float = 1.0,
                 tile: int = 0, margin: int = 256, scale_depth: int = 1) -> None:
        super().__init__()
        self.number_of_levels = int(number_of_scales)
        self.level_step = 1.0  # octaves
        self.log_base_radius = math.log(float(base_radius))
        self.temperature = float(temperature)
        self.share_weights = True
        self.tile, self.margin = int(tile), int(margin)
        self.lifting = GaussianLifting(self.number_of_levels, sigma0)
        c = [int(w) for w in widths]
        d = int(scale_depth)
        self.encoder = nn.ModuleList([_scale_block(in_channels, c[0], 1, d)] +
                                     [_scale_block(c[i], c[i + 1], 2, d) for i in range(4)])
        self.decoder = nn.ModuleList([_scale_block(c[i + 1] + c[i], c[i], 1, d) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], 5, kernel_size=1)  # shared across scales
        self.scale_depth = d

    # ---- network -------------------------------------------------------------------------
    def level_responses(self, image: torch.Tensor) -> torch.Tensor:
        """Raw per-scale outputs [B,S,5,H,W]. With filter depth 1 the scales never interact, so in eval
        mode they are computed one at a time: identical output, about 1/S of the memory."""
        if self.training or self.scale_depth != 1:
            return self._responses(image, None)
        return torch.cat([self._responses(image, [k]) for k in range(self.number_of_levels)], dim=1)

    def _responses(self, image: torch.Tensor, scales) -> torch.Tensor:
        for mod in self.modules():
            if isinstance(mod, ScaleConv):
                mod.k0 = 0 if scales is None else scales[0]
        height, width = image.shape[-2:]
        pad_h, pad_w = (-height) % 16, (-width) % 16
        x = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else image
        x = self.lifting(x, scales)
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = skips.pop()
        for block in self.decoder:
            skip = skips.pop()
            b, s = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), size=skip.shape[-2:], mode="bilinear",
                              align_corners=False).unflatten(0, (b, s))
            x = block(torch.cat([x, skip], dim=2))
        b, s = x.shape[:2]
        out = self.head(x.flatten(0, 1)).unflatten(0, (b, s))
        for mod in self.modules():
            if isinstance(mod, ScaleConv):
                mod.k0 = 0
        return out[..., :height, :width]

    def _tiled_responses(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        t, m = self.tile, self.margin
        out = None
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                ya, xa = max(0, y0 - m), max(0, x0 - m)
                yb, xb = min(h, y1 + m), min(w, x1 + m)
                r = self.level_responses(image[..., ya:yb, xa:xb])
                if out is None:
                    out = image.new_zeros(image.shape[0], r.shape[1], r.shape[2], h, w)
                out[..., y0:y1, x0:x1] = r[..., y0 - ya:y1 - ya, x0 - xa:x1 - xa]
        return out

    def forward(self, image: torch.Tensor) -> MedialFrameField:
        image = image.float()
        h, w = image.shape[-2:]
        big = (not self.training) and self.tile > 0 and (h > self.tile + 2 * self.margin or w > self.tile + 2 * self.margin)
        stacked = self._tiled_responses(image) if big else self.level_responses(image)
        scale_logits = stacked[:, :, 0]
        correction = 0.5 * self.level_step * LN2 * torch.tanh(stacked[:, :, 1])
        orientation = torch.tanh(stacked[:, :, 2:4])
        band_logits = stacked[:, :, 4]
        probabilities = torch.softmax(scale_logits / self.temperature, dim=1)
        level_log_radius = torch.arange(self.number_of_levels, device=image.device,
                                        dtype=image.dtype).view(1, -1, 1, 1) * (self.level_step * LN2)
        log_radius = self.log_base_radius + (probabilities * (level_log_radius + correction)).sum(dim=1, keepdim=True)
        return MedialFrameField(
            log_radius=log_radius,
            vector=(probabilities.unsqueeze(2) * orientation).sum(dim=1),
            band_logit=(probabilities * band_logits).sum(dim=1, keepdim=True),
            level_probabilities=probabilities,
        )
