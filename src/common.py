"""Shared helpers for the calibre scripts.

Every function below is copied verbatim from the authors' training and
evaluation code (only the segmentation-network parts are left out), so the
scripts behave exactly as they did when the paper's results were produced.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
import yaml

from src.datasets import (
    CLAHEEnhancement,
    load_binary_mask,
    load_rgb_image,
    normalize_filename,
)

__all__ = [
    "CLAHEEnhancement", "load_binary_mask", "load_rgb_image", "load_config", "resolve_datasets",
    "pair_files", "save_gray", "write_csv", "normalize_rgb_patch", "tiled_radius", "StandaloneSGE",
    "load_geometry_model", "Geometry", "read_gray", "THRESHOLDS", "dice_at", "load_model", "Segmenter",
]


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    ".ppm", ".gif", ".bmp",
}


IMAGENET_MEAN = np.asarray(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)


IMAGENET_STD = np.asarray(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


PathLike = str | Path


def load_config(path: PathLike) -> Dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping."
        )

    return config


@dataclass(frozen=True)
class DatasetSpecification:
    name: str
    image_dir: Path
    mask_dir: Path


@dataclass(frozen=True)
class SamplePair:
    key: str
    image_path: Path
    mask_path: Path


def supported_files(
    directory: PathLike,
) -> List[Path]:
    directory = Path(directory)

    if not directory.is_dir():
        return []

    return sorted(
        [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ],
        key=lambda path: path.name.lower(),
    )


def pair_files(
    image_dir: PathLike,
    mask_dir: PathLike,
) -> List[SamplePair]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir.resolve()}"
        )

    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Mask directory not found: {mask_dir.resolve()}"
        )

    images: Dict[str, Path] = {}
    masks: Dict[str, Path] = {}

    for path in supported_files(image_dir):
        key = normalize_filename(path.name)

        if key in images:
            raise RuntimeError(
                f"Duplicate normalized image key: {key}"
            )

        images[key] = path

    for path in supported_files(mask_dir):
        key = normalize_filename(path.name)

        if key in masks:
            raise RuntimeError(
                f"Duplicate normalized mask key: {key}"
            )

        masks[key] = path

    keys = sorted(
        set(images).intersection(masks)
    )

    if not keys:
        raise RuntimeError(
            "No image-mask pairs were found after "
            "filename normalization."
        )

    return [
        SamplePair(
            key=key,
            image_path=images[key],
            mask_path=masks[key],
        )
        for key in keys
    ]


def add_dataset_if_available(
    result: Dict[str, DatasetSpecification],
    name: str,
    image_dir: PathLike,
    mask_dir: PathLike,
) -> None:
    image_dir = Path(str(image_dir))
    mask_dir = Path(str(mask_dir))

    if image_dir.is_dir() and mask_dir.is_dir():
        result[name] = DatasetSpecification(
            name=name,
            image_dir=image_dir,
            mask_dir=mask_dir,
        )


def resolve_datasets(
    config: Mapping[str, Any],
    requested: Optional[Sequence[str]],
) -> Dict[str, DatasetSpecification]:
    result: Dict[str, DatasetSpecification] = {}
    dataset_config = config.get("dataset", {})

    test_dataset = dataset_config.get("test_dataset")

    if isinstance(test_dataset, Mapping):
        add_dataset_if_available(
            result=result,
            name=str(
                test_dataset.get("name", "DRIVE_test")
            ),
            image_dir=test_dataset.get(
                "image_dir",
                "datasets_processed/DRIVE_test/images",
            ),
            mask_dir=test_dataset.get(
                "mask_dir",
                "datasets_processed/DRIVE_test/masks",
            ),
        )

    external = dataset_config.get(
        "external_datasets",
        {},
    )

    if isinstance(external, Mapping):
        for name, entry in external.items():
            if not isinstance(entry, Mapping):
                continue

            add_dataset_if_available(
                result=result,
                name=str(name),
                image_dir=entry.get("image_dir", ""),
                mask_dir=entry.get("mask_dir", ""),
            )

    defaults = {
        "DRIVE_test": (
            "datasets_processed/DRIVE_test/images",
            "datasets_processed/DRIVE_test/masks",
        ),
        "STARE": (
            "datasets_processed/STARE/images",
            "datasets_processed/STARE/masks",
        ),
        "CHASE_DB1": (
            "datasets_processed/CHASE_DB1/images",
            "datasets_processed/CHASE_DB1/masks",
        ),
        "HRF": (
            "datasets_processed/HRF/images",
            "datasets_processed/HRF/masks",
        ),
    }

    for name, (image_dir, mask_dir) in defaults.items():
        if name not in result:
            add_dataset_if_available(
                result=result,
                name=name,
                image_dir=image_dir,
                mask_dir=mask_dir,
            )

    if requested:
        requested_lower = {
            name.lower()
            for name in requested
        }

        # Opt-in datasets: only added when requested by name, so default
        # evaluations (requested=None) are unchanged. FIVES test split
        # (200 images, 2048x2048) for the calibre test, 2026-11-11.
        opt_in = {
            "FIVES_test": (
                "datasets_processed/FIVES_test/images",
                "datasets_processed/FIVES_test/masks",
            ),
        }
        for name, (image_dir, mask_dir) in opt_in.items():
            if name.lower() in requested_lower and name not in result:
                add_dataset_if_available(
                    result=result,
                    name=name,
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                )

        result = {
            name: specification
            for name, specification in result.items()
            if name.lower() in requested_lower
        }

    if not result:
        raise RuntimeError(
            "No requested evaluation datasets were found."
        )

    return result


def normalize_rgb_patch(
    patch_rgb: np.ndarray,
) -> torch.Tensor:
    patch = patch_rgb.astype(
        np.float32
    ) / 255.0

    patch = (
        patch - IMAGENET_MEAN
    ) / IMAGENET_STD

    patch = np.transpose(
        patch,
        (2, 0, 1),
    )

    return torch.from_numpy(
        np.ascontiguousarray(patch)
    ).float()


def save_gray(
    path: Path,
    array: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if array.dtype == bool:
        output = array.astype(np.uint8) * 255

    elif np.issubdtype(
        array.dtype,
        np.floating,
    ):
        output = (
            np.clip(array, 0.0, 1.0)
            * 255.0
        ).round().astype(np.uint8)

    else:
        output = np.clip(
            array,
            0,
            255,
        ).astype(np.uint8)

    if not cv2.imwrite(
        str(path),
        output,
    ):
        raise IOError(
            f"Could not save: {path.resolve()}"
        )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def tiled_radius(model, image_rgb: np.ndarray, device, tile: int = 256, overlap: int = 64) -> np.ndarray:
    """Predict the radius map in overlapping tiles of the training patch
    size (256), the regime used in training and patch evaluation."""
    height, width = image_rgb.shape[:2]
    out = np.zeros((height, width), np.float32)
    band_out = np.zeros((height, width), np.float32)
    weight = np.zeros((height, width), np.float32)
    step = tile - overlap
    ys = list(range(0, max(1, height - overlap), step))
    xs = list(range(0, max(1, width - overlap), step))
    for y in ys:
        for x in xs:
            y0, x0 = min(y, max(0, height - tile)), min(x, max(0, width - tile))
            crop = image_rgb[y0:y0 + tile, x0:x0 + tile]
            if crop.shape[0] < tile or crop.shape[1] < tile:
                crop = np.pad(crop, ((0, tile - crop.shape[0]), (0, tile - crop.shape[1]), (0, 0)), mode="reflect")
            tensor = normalize_rgb_patch(crop)[None].to(device)
            geometry = model.predict_geometry(tensor)
            radius = geometry["radius"][0, 0].float().cpu().numpy()
            band = geometry["band"][0, 0].float().cpu().numpy()
            h, w = min(tile, height - y0), min(tile, width - x0)
            radius = radius[:h, :w]
            band = band[:h, :w]
            # down-weight tile borders
            wy = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1).astype(np.float32)
            wx = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1).astype(np.float32)
            wmap = np.minimum.outer(wy, wx).clip(1, overlap)
            out[y0:y0 + h, x0:x0 + w] += radius * wmap
            band_out[y0:y0 + h, x0:x0 + w] += band * wmap
            weight[y0:y0 + h, x0:x0 + w] += wmap
    weight = np.maximum(weight, 1e-6)
    return out / weight, band_out / weight


class StandaloneSGE(torch.nn.Module):
    """A geometry estimator saved by src/pretrain_sge.py (sge.pth), with a
    predict_geometry interface (radius and band maps)."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        from models.medial_frame import build_geometry_estimator
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = saved["config"]
        self.geometry_estimator = build_geometry_estimator(
            saved["config"].get("model", {}).get("medial_frame", {}).get("estimator", {}))
        self.geometry_estimator.load_state_dict(saved["estimator_state"], strict=True)

    def predict_geometry(self, x: torch.Tensor):
        with torch.autocast(device_type=x.device.type, enabled=False):
            field = self.geometry_estimator(x.float())
        return {"radius": torch.exp(field.log_radius), "band": torch.sigmoid(field.band_logit)}


def load_geometry_model(config_path, checkpoint_path: str, device):
    """Standalone estimator (sge.pth; config ignored)."""
    if str(checkpoint_path).endswith("sge.pth"):
        model = StandaloneSGE(checkpoint_path).to(device).eval()
        return model.config, model
    return load_model(config_path, checkpoint_path, device)


class Geometry:
    def __init__(self, config, model, device):
        self.model = model
        self.device = device
        self.clahe = CLAHEEnhancement() if bool(config.get("preprocessing", {}).get("clahe", {}).get("enabled", True)) else None

    def __call__(self, image_rgb: np.ndarray):
        image = self.clahe(image_rgb) if self.clahe is not None else image_rgb
        with torch.no_grad():
            radius, band = tiled_radius(self.model, image, self.device)
        return np.maximum(radius, 0.25), np.clip(band, 0.0, 1.0)


def read_gray(path: Path) -> np.ndarray:
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:  # e.g. .gif masks
        from PIL import Image
        image = np.array(Image.open(path).convert("L"))
    return image


THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.02), 2)


def dice_at(prob, gt, t):
    p = prob >= t
    inter = np.logical_and(p, gt).sum()
    return (2 * inter + 1e-6) / (p.sum() + gt.sum() + 1e-6)


SEGMENTATION_MISSING = (
    "The segmentation networks behind the mask-width baseline are not part of this release. "
    "Use the released probability maps (see README, 'Mask-width baseline')."
)


def load_model(config_path, checkpoint_path, device):
    raise RuntimeError(SEGMENTATION_MISSING)


class Segmenter:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(SEGMENTATION_MISSING)
