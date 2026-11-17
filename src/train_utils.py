"""Training helpers for src.pretrain_sge, copied verbatim from the authors' training code."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
import yaml

from src.topology_datasets import (
    build_topology_dataloaders,
    build_topology_patch_dataloaders,
)


def load_config(path: str = "config_topology.yaml") -> Dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")
    return config


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_training_loaders(config: Mapping) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, str]:
    dataset = config["dataset"]
    train_dataset = dataset["train_dataset"]
    topology = config["topology"]
    patch = config.get("patch_training", {})
    preprocessing = config["preprocessing"]
    loader = config["dataloader"]

    common = dict(
        image_dir=str(train_dataset["image_dir"]).strip(),
        mask_dir=str(train_dataset["mask_dir"]).strip(),
        topology_root=str(topology["target_root"]).strip(),
        batch_size=int(loader["batch_size"]),
        validation_ratio=float(dataset.get("validation_ratio", 0.2)),
        num_workers=int(loader.get("num_workers", 0)),
        clahe=bool(preprocessing.get("clahe", {}).get("enabled", True)),
        seed=int(dataset.get("split_seed", config["seed"]["value"])),
        pin_memory=bool(loader.get("pin_memory", True)),
        persistent_workers=bool(loader.get("persistent_workers", False)),
        drop_last=bool(loader.get("drop_last", False)),
    )
    if common["num_workers"] == 0:
        common["persistent_workers"] = False

    if bool(patch.get("enabled", True)):
        patch_size = int(patch.get("patch_size", 256))
        input_size = int(patch.get("model_input_size", 256))
        train_loader, val_loader = build_topology_patch_dataloaders(
            **common,
            patch_size=patch_size,
            model_input_size=input_size,
            patches_per_image=int(patch.get("patches_per_image", 100)),
            validation_stride=int(patch.get("validation_stride", patch_size // 2)),
            vessel_center_probability=float(patch.get("vessel_center_probability", 0.60)),
            hard_negative_probability=float(patch.get("hard_negative_probability", 0.25)),
            random_probability=float(patch.get("random_probability", 0.15)),
            minimum_vessel_fraction=float(patch.get("minimum_vessel_fraction", 0.01)),
            hard_negative_max_vessel_fraction=float(patch.get("hard_negative_max_vessel_fraction", 0.005)),
            hard_negative_candidates=int(patch.get("hard_negative_candidates", 20)),
            max_sampling_attempts=int(patch.get("max_sampling_attempts", 20)),
            include_empty_validation_patches=bool(patch.get("include_empty_validation_patches", True)),
            scale_augmentation_enabled=bool(patch.get("scale_augmentation", {}).get("enabled", False)),
            scale_augmentation_probability=float(patch.get("scale_augmentation", {}).get("probability", 0.0)),
            scale_augmentation_min=float(patch.get("scale_augmentation", {}).get("min_scale", 1.0)),
            scale_augmentation_max=float(patch.get("scale_augmentation", {}).get("max_scale", 1.0)),
            scale_augmentation_distribution=str(patch.get("scale_augmentation", {}).get("distribution", "uniform")),
            regenerate_topology_targets=bool(patch.get("scale_augmentation", {}).get("regenerate_topology_targets", False)),
        )
        return train_loader, val_loader, f"topology patch training ({patch_size} -> {input_size})"

    image_size = int(preprocessing["image_size"]["height"])
    train_loader, val_loader = build_topology_dataloaders(**common, image_size=image_size)
    return train_loader, val_loader, f"topology whole-image training ({image_size}x{image_size})"
