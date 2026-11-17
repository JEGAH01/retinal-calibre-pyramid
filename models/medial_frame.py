"""Vessel-calibre estimators (R2, R2ns, P1) and their geometry targets.

ScaleEquivariantGeometryEstimator (R2; R2ns with ``share_weights: false``)
    A small network ``g`` is applied with *shared weights* to every level of a
    half-octave image pyramid. Per pixel it predicts, for each level, a
    scale-selection logit, a bounded sub-level radius correction, a
    doubled-angle orientation vector and a vessel-band logit. A soft-argmax
    over levels gives a dense medial frame field:

        log r(x)      local vessel radius (pixels, input resolution)
        v(x)          doubled-angle tangent vector (cos 2t, sin 2t) * coherence
        band(x)       probability that x lies within ~2 radii of a centreline

    Because ``g`` is shared across levels, rescaling the image by 2**(k*step)
    shifts the level responses by k and multiplies r by exactly that factor
    (pyramid-aligned factors, inside the pyramid range). R2 uses LevelUNet as
    ``g`` (``type: pyramid_unet``).

PlainGeometryEstimator (P1)
    The matched single-scale regressor (``type: plain``).

Geometry supervision (``medial_geometry_targets``) is computed from the
(augmented) ground-truth mask: medial radius propagated from the nearest
skeleton pixel, skeleton tangent from a structure tensor, and the band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

LN2 = math.log(2.0)


# ============================================================
# GEOMETRY FIELD CONTAINER
# ============================================================


@dataclass
class MedialFrameField:
    """Dense medial frame predicted at input resolution."""

    log_radius: torch.Tensor   # [B,1,H,W] natural log of radius in input pixels
    vector: torch.Tensor       # [B,2,H,W] doubled-angle tangent vector, |v| in [0,1]
    band_logit: torch.Tensor   # [B,1,H,W]
    level_probabilities: torch.Tensor  # [B,L,H,W]

    def canonical(self) -> "MedialFrameField":
        """Replace the radius outside the vessel band by an image-level
        band-weighted mean (global canonicalisation in the background)."""
        band = torch.sigmoid(self.band_logit)
        weight_sum = band.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        global_log_radius = (band * self.log_radius).sum(
            dim=(2, 3), keepdim=True
        ) / weight_sum
        log_rho = band * self.log_radius + (1.0 - band) * global_log_radius
        return MedialFrameField(
            log_radius=log_rho,
            vector=self.vector * band,
            band_logit=self.band_logit,
            level_probabilities=self.level_probabilities,
        )

    def at_size(self, size: Sequence[int]) -> Dict[str, torch.Tensor]:
        """Average the (canonical) frame onto a feature grid of ``size``.

        Returns log_rho [B,1,h,w], cos/sin of theta [B,1,h,w] and
        confidence c in [0,1] [B,1,h,w].
        """
        size = (int(size[0]), int(size[1]))
        same = tuple(self.log_radius.shape[-2:]) == size
        if same:
            log_rho, vec = self.log_radius, self.vector
        else:
            log_rho = F.interpolate(self.log_radius, size=size, mode="area")
            vec = F.interpolate(self.vector, size=size, mode="area")
        magnitude = torch.linalg.vector_norm(vec, dim=1, keepdim=True)
        confidence = magnitude.clamp(0.0, 1.0)
        double_angle = torch.atan2(vec[:, 1:2], vec[:, 0:1])
        theta = 0.5 * double_angle
        return {
            "log_rho": log_rho,
            "cos": torch.cos(theta),
            "sin": torch.sin(theta),
            "confidence": confidence,
        }


# ============================================================
# SCALE-EQUIVARIANT GEOMETRY ESTIMATOR
# ============================================================


def _level_network(in_channels: int, hidden: int, depth: int) -> nn.Sequential:
    layers = []
    channels = in_channels
    for _ in range(depth - 1):
        layers += [
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=hidden),
            nn.GELU(),
        ]
        channels = hidden
    # 5 outputs: scale logit, radius correction, 2 orientation, band logit
    layers.append(nn.Conv2d(channels, 5, kernel_size=1))
    return nn.Sequential(*layers)


class ScaleEquivariantGeometryEstimator(nn.Module):
    """Pyramid-shared CNN predicting a dense medial frame field.

    Parameters
    ----------
    number_of_levels:
        Pyramid levels.  With ``level_step=0.5`` (half octave) and
        ``base_radius=1.0``, 11 levels cover radii from 1 px to 32 px.
    share_weights:
        True (default) gives scale equivariance by construction.  False
        builds an independent network per level (ablation M4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 16,
        depth: int = 5,
        number_of_levels: int = 11,
        level_step: float = 0.5,
        base_radius: float = 1.0,
        share_weights: bool = True,
        temperature: float = 1.0,
        level_network: str = "cnn",
    ) -> None:
        super().__init__()
        if number_of_levels < 2:
            raise ValueError("number_of_levels must be at least 2.")
        self.number_of_levels = int(number_of_levels)
        self.level_step = float(level_step)
        self.log_base_radius = math.log(float(base_radius))
        self.share_weights = bool(share_weights)
        self.temperature = float(temperature)
        if level_network == "unet":
            make = lambda: LevelUNet(in_channels, hidden_channels)  # noqa: E731
        elif level_network == "cnn":
            make = lambda: _level_network(in_channels, hidden_channels, depth)  # noqa: E731
        else:
            raise ValueError(f"Unknown level_network: {level_network}")
        if self.share_weights:
            self.level_network = make()
        else:
            self.level_networks = nn.ModuleList(make() for _ in range(self.number_of_levels))

    def _network(self, level: int) -> nn.Module:
        return self.level_network if self.share_weights else self.level_networks[level]

    def level_responses(self, image: torch.Tensor) -> torch.Tensor:
        """Raw per-level outputs [B, L, 5, H, W] at input resolution."""
        height, width = image.shape[-2:]
        image = image.float()
        outputs = []
        for level in range(self.number_of_levels):
            factor = 2.0 ** (-level * self.level_step)
            if level == 0:
                level_image = image
            else:
                level_size = (
                    max(4, int(round(height * factor))),
                    max(4, int(round(width * factor))),
                )
                level_image = F.interpolate(
                    image, size=level_size, mode="bilinear",
                    align_corners=False, antialias=True,
                )
            response = self._network(level)(level_image)
            if level != 0:
                response = F.interpolate(
                    response, size=(height, width), mode="bilinear",
                    align_corners=False,
                )
            outputs.append(response)
        return torch.stack(outputs, dim=1)

    def forward(self, image: torch.Tensor) -> MedialFrameField:
        image = image.float()
        stacked = self.level_responses(image)  # [B,L,5,H,W]
        scale_logits = stacked[:, :, 0]
        correction = 0.5 * self.level_step * LN2 * torch.tanh(stacked[:, :, 1])
        orientation = torch.tanh(stacked[:, :, 2:4])  # [B,L,2,H,W]
        band_logits = stacked[:, :, 4]

        probabilities = torch.softmax(scale_logits / self.temperature, dim=1)
        level_log_radius = torch.arange(
            self.number_of_levels, device=image.device, dtype=image.dtype
        ).view(1, -1, 1, 1) * (self.level_step * LN2)
        log_radius = self.log_base_radius + (
            probabilities * (level_log_radius + correction)
        ).sum(dim=1, keepdim=True)
        vector = (probabilities.unsqueeze(2) * orientation).sum(dim=1)
        band_logit = (probabilities * band_logits).sum(dim=1, keepdim=True)
        return MedialFrameField(
            log_radius=log_radius,
            vector=vector,
            band_logit=band_logit,
            level_probabilities=probabilities,
        )


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(4, out_channels), nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels), nn.GELU(),
    )


class PlainGeometryEstimator(nn.Module):
    """Ablation baseline for the SGE: a conventional radius regressor.

    A small U-Net on the input image at native resolution (no image pyramid,
    no weight sharing across scales, so no scale equivariance by construction),
    regressing log radius, orientation and band directly, in the spirit of
    per-pixel width regressors such as Li et al. (MICCAI 2021) and radius heads
    such as MARVEL. Four stride-2 stages give a receptive field of several
    hundred pixels, so wide vessels are visible. Same output field and losses
    as the SGE, so it is a drop-in replacement.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 16, base_radius: float = 1.0) -> None:
        super().__init__()
        c = [hidden_channels, 2 * hidden_channels, 4 * hidden_channels, 4 * hidden_channels, 4 * hidden_channels]
        self.encoder = nn.ModuleList([_conv_block(in_channels, c[0])] +
                                     [_conv_block(c[i], c[i + 1], stride=2) for i in range(4)])
        self.decoder = nn.ModuleList([_conv_block(c[i + 1] + c[i], c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], 4, kernel_size=1)  # log-radius offset, 2 orientation, band logit
        self.log_base_radius = math.log(float(base_radius))
        self.temperature = 1.0  # unused; kept for interface parity with the SGE
        self.number_of_levels = 1

    def forward(self, image: torch.Tensor) -> MedialFrameField:
        image = image.float()
        height, width = image.shape[-2:]
        pad_h, pad_w = (-height) % 16, (-width) % 16
        x = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else image
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = skips.pop()
        for block in self.decoder:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        out = self.head(x)[..., :height, :width]
        return MedialFrameField(
            log_radius=self.log_base_radius + out[:, 0:1],
            vector=torch.tanh(out[:, 1:3]),
            band_logit=out[:, 3:4],
            level_probabilities=torch.ones_like(out[:, 0:1]),
        )


class LevelUNet(nn.Module):
    """Per-level network for the R2 estimator: the PlainGeometryEstimator U-Net
    (same widths and depth) with 5 raw outputs as expected by the SGE (scale
    logit, radius correction, 2 orientation, band logit). Replicate padding to
    a multiple of 16, since deep pyramid levels can be smaller than 16 px."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 16) -> None:
        super().__init__()
        c = [hidden_channels, 2 * hidden_channels, 4 * hidden_channels, 4 * hidden_channels, 4 * hidden_channels]
        self.encoder = nn.ModuleList([_conv_block(in_channels, c[0])] +
                                     [_conv_block(c[i], c[i + 1], stride=2) for i in range(4)])
        self.decoder = nn.ModuleList([_conv_block(c[i + 1] + c[i], c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], 5, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_h, pad_w = (-height) % 16, (-width) % 16
        x = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else image
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = skips.pop()
        for block in self.decoder:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)[..., :height, :width]


def build_geometry_estimator(estimator_config) -> nn.Module:
    """SGE (default, ``type: pyramid``), the plain ablation (``type: plain``), or
    R2 (``type: pyramid_unet``): the SGE pyramid with the plain U-Net as level network."""
    cfg = estimator_config or {}
    kind = cfg.get("type", "pyramid")
    if kind == "plain":
        return PlainGeometryEstimator(hidden_channels=int(cfg.get("hidden_channels", 16)),
                                      base_radius=float(cfg.get("base_radius", 1.0)))
    if kind == "se_unet":  # scale-equivariant U-Net baseline (SEU)
        from models.scale_equivariant_unet import ScaleEquivariantUNetEstimator
        return ScaleEquivariantUNetEstimator(widths=tuple(cfg.get("widths", (16, 32, 64, 64, 64))),
                                             scale_depth=int(cfg.get("scale_depth", 1)),
                                             number_of_scales=int(cfg.get("scales", 6)),
                                             base_radius=float(cfg.get("base_radius", 0.5)),
                                             tile=int(cfg.get("tile", 0)), margin=int(cfg.get("margin", 256)))
    if kind not in ("pyramid", "pyramid_unet"):
        raise ValueError(f"Unknown estimator type: {kind}")
    return ScaleEquivariantGeometryEstimator(
        hidden_channels=int(cfg.get("hidden_channels", 16)), depth=int(cfg.get("depth", 5)),
        number_of_levels=int(cfg.get("levels", 11)), level_step=float(cfg.get("level_step", 0.5)),
        base_radius=float(cfg.get("base_radius", 1.0)), share_weights=bool(cfg.get("share_weights", True)),
        level_network="unet" if kind == "pyramid_unet" else "cnn")


# ============================================================
# GEOMETRY TARGETS AND LOSSES
# ============================================================


def medial_geometry_targets(
    mask: np.ndarray,
    band_radii: float = 2.0,
    orientation_sigma: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Medial geometry targets from one binary mask [H,W].

    radius:      medial radius of the nearest centreline point (px), >= 0.5
    vector:      doubled-angle tangent (cos 2t, sin 2t) at the nearest
                 centreline point  [2,H,W]
    coherence:   structure-tensor anisotropy (0 at junctions, 1 on lines)
    band:        pixels within band_radii * radius (+1 px) of a centreline
    """
    mask = mask.astype(bool)
    height, width = mask.shape
    empty = {
        "radius": np.ones((height, width), np.float32),
        "vector": np.zeros((2, height, width), np.float32),
        "coherence": np.zeros((height, width), np.float32),
        "band": np.zeros((height, width), np.float32),
    }
    if mask.sum() < 3:
        return empty
    distance = ndi.distance_transform_edt(mask)
    skeleton = skeletonize(mask)
    if skeleton.sum() == 0:
        return empty
    distance_to_skeleton, (iy, ix) = ndi.distance_transform_edt(
        ~skeleton, return_indices=True
    )
    skeleton_radius = np.maximum(distance - 0.5, 0.5)
    radius = skeleton_radius[iy, ix]
    band = distance_to_skeleton <= band_radii * radius + 1.0

    smooth = ndi.gaussian_filter(skeleton.astype(np.float32), 1.0)
    gy, gx = np.gradient(smooth)
    jxx = ndi.gaussian_filter(gx * gx, orientation_sigma)
    jyy = ndi.gaussian_filter(gy * gy, orientation_sigma)
    jxy = ndi.gaussian_filter(gx * gy, orientation_sigma)
    # dominant gradient direction is the normal; tangent = normal + 90 deg,
    # so the doubled tangent angle is the doubled normal angle + 180 deg.
    double_normal = np.arctan2(2.0 * jxy, jxx - jyy)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (jxx + jyy + 1e-12)
    vector = np.stack((-np.cos(double_normal), -np.sin(double_normal)))
    vector = vector[:, iy, ix]
    coherence = coherence[iy, ix]
    return {
        "radius": radius.astype(np.float32),
        "vector": vector.astype(np.float32),
        "coherence": np.clip(coherence, 0.0, 1.0).astype(np.float32),
        "band": band.astype(np.float32),
    }


def batch_geometry_targets(mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute geometry targets for a [B,1,H,W] mask tensor on the CPU."""
    masks = (mask.detach().float().cpu().numpy()[:, 0] > 0.5)
    per_sample = [medial_geometry_targets(m) for m in masks]
    device = mask.device
    return {
        "radius": torch.from_numpy(np.stack([t["radius"] for t in per_sample]))[:, None].to(device),
        "vector": torch.from_numpy(np.stack([t["vector"] for t in per_sample])).to(device),
        "coherence": torch.from_numpy(np.stack([t["coherence"] for t in per_sample]))[:, None].to(device),
        "band": torch.from_numpy(np.stack([t["band"] for t in per_sample]))[:, None].to(device),
    }


def geometry_losses(
    field: MedialFrameField,
    targets: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Radius (log-L1), orientation (coherence-weighted 1-cos 2dtheta) and
    band (BCE) losses."""
    band = targets["band"]
    denominator = band.sum().clamp_min(1.0)
    radius_loss = (
        (field.log_radius.float() - torch.log(targets["radius"].clamp_min(0.5))).abs() * band
    ).sum() / denominator

    predicted = field.vector.float()
    predicted_unit = predicted / torch.linalg.vector_norm(predicted, dim=1, keepdim=True).clamp_min(1e-6)
    cosine = (predicted_unit * targets["vector"]).sum(dim=1, keepdim=True)
    weight = band * targets["coherence"]
    orientation_loss = ((1.0 - cosine) * weight).sum() / weight.sum().clamp_min(1.0)

    band_loss = F.binary_cross_entropy_with_logits(field.band_logit.float(), band)
    return {
        "geometry_radius_loss": radius_loss,
        "geometry_orientation_loss": orientation_loss,
        "geometry_band_loss": band_loss,
    }


def equivariance_loss(
    estimator: ScaleEquivariantGeometryEstimator,
    image: torch.Tensor,
    field: MedialFrameField,
) -> torch.Tensor:
    """Scale-consistency of the radius field under a random pyramid-aligned
    rescale t in {2^-0.5, 2^0.5}: ln r(S_t I)(t x) - ln t == ln r(I)(x)."""
    exponent = -1.0 if torch.rand(()) < 0.5 else 1.0
    t = 2.0 ** (0.5 * exponent)
    height, width = image.shape[-2:]
    size = (max(8, int(round(height * t))), max(8, int(round(width * t))))
    scaled = F.interpolate(image.float(), size=size, mode="bilinear", align_corners=False, antialias=True)
    scaled_field = estimator(scaled)
    back = F.interpolate(scaled_field.log_radius, size=(height, width), mode="bilinear", align_corners=False)
    weight = torch.sigmoid(field.band_logit.detach())
    difference = (back - math.log(t) - field.log_radius).abs()
    return (difference * weight).sum() / weight.sum().clamp_min(1.0)
