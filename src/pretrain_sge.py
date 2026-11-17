"""
Train one calibre estimator (R2, R2ns, P1, R2w, P1w or SEU; the type is set in the config).

Trains ONLY the ScaleEquivariantGeometryEstimator (R2: 488,325 parameters)
on DRIVE training patches with the radius / orientation / band / equivariance
losses, then reports radius accuracy on the DRIVE validation images at
several synthetic rescale factors, including factors outside the training
scale range.  This tests the core claim (scale extrapolation of the
estimator) in minutes, before any full-model training.

The paper's estimators were trained for 5 epochs with the default learning rate.

Usage (repository root):
    python -m src.pretrain_sge --config configs/sge_R2_seed42.yaml \
        --epochs 5 --output checkpoints/sge_R2_seed42
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from models.medial_frame import (
    ScaleEquivariantGeometryEstimator,
    batch_geometry_targets,
    build_geometry_estimator,
    equivariance_loss,
    geometry_losses,
    medial_geometry_targets,
)
from src.train_utils import build_training_loaders, load_config, seed_everything

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_estimator(config):
    """SGE by default; model.medial_frame.estimator.type: plain gives the non-equivariant ablation."""
    return build_geometry_estimator(config.get("model", {}).get("medial_frame", {}).get("estimator", {}))


def radius_accuracy(estimator, loader, device, factors) -> dict:
    """Median |ln r_hat - ln r*| on vessel pixels and the predicted/GT
    median radius ratio for whole validation patches rescaled by t."""
    estimator.eval()
    results = {}
    for t in factors:
        errors, ratios = [], []
        for batch in loader:
            images = batch["image"].float()
            masks = batch["mask"].float()
            if t != 1.0:
                size = (int(round(images.shape[-2] * t)), int(round(images.shape[-1] * t)))
                images = torch.nn.functional.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=t < 1)
                masks = (torch.nn.functional.interpolate(masks, size=size, mode="bilinear", align_corners=False) > 0.5).float()
            with torch.no_grad():
                radius = torch.exp(estimator(images.to(device)).log_radius).cpu().numpy()[:, 0]
            for r_hat, m in zip(radius, masks.numpy()[:, 0] > 0.5):
                if m.sum() < 50:
                    continue
                gt = medial_geometry_targets(m)["radius"]
                errors.append(float(np.median(np.abs(np.log(np.maximum(r_hat[m], 1e-3)) - np.log(gt[m])))))
                ratios.append(float(np.median(r_hat[m]) / np.median(gt[m])))
        results[f"x{t:g}"] = {
            "median_abs_log_error": float(np.median(errors)),
            "median_pred_over_gt": float(np.median(ratios)),
            "patches": len(errors),
        }
    estimator.train()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many steps (0 = full epochs).")
    parser.add_argument("--equivariance-weight", type=float, default=None)
    parser.add_argument("--factors", type=float, nargs="+", default=[0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    seed_everything(int(config["seed"]["value"]), False)
    device = torch.device(arguments.device)
    train_loader, val_loader, mode = build_training_loaders(config)
    estimator = build_estimator(config).to(device)
    loss_cfg = config.get("model", {}).get("medial_frame", {}).get("loss", {})
    w_r = float(loss_cfg.get("radius_weight", 0.5))
    w_o = float(loss_cfg.get("orientation_weight", 0.2))
    w_b = float(loss_cfg.get("band_weight", 0.2))
    w_e = float(loss_cfg.get("equivariance_weight", 0.1)) if arguments.equivariance_weight is None else arguments.equivariance_weight
    optimizer = AdamW(estimator.parameters(), lr=arguments.lr, weight_decay=1e-4)

    print(f"SGE pretraining | {mode} | params {sum(p.numel() for p in estimator.parameters()):,} | device {device}")
    print("before training:", json.dumps(radius_accuracy(estimator, val_loader, device, [1.0])))
    step = 0
    for epoch in range(arguments.epochs):
        running = {}
        for batch in tqdm(train_loader, desc=f"SGE epoch {epoch + 1}"):
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().to(device)
            field = estimator(images)
            terms = geometry_losses(field, batch_geometry_targets(masks))
            loss = w_r * terms["geometry_radius_loss"] + w_o * terms["geometry_orientation_loss"] + w_b * terms["geometry_band_loss"]
            if w_e > 0:
                terms["geometry_equivariance_loss"] = equivariance_loss(estimator, images, field)
                loss = loss + w_e * terms["geometry_equivariance_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1.0)
            optimizer.step()
            for k, v in terms.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            step += 1
            if arguments.max_steps and step >= arguments.max_steps:
                break
        n = max(1, min(len(train_loader), step))
        print(f"epoch {epoch + 1}: " + ", ".join(f"{k}={v / n:.4f}" for k, v in running.items()))
        if arguments.max_steps and step >= arguments.max_steps:
            break

    accuracy = radius_accuracy(estimator, val_loader, device, arguments.factors)
    print("radius accuracy on DRIVE validation patches by rescale factor:")
    for key, value in accuracy.items():
        print(f"  {key:>6}: median |ln r_hat - ln r*| = {value['median_abs_log_error']:.3f}"
              f"   pred/GT = {value['median_pred_over_gt']:.3f}   (n={value['patches']})")
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"estimator_state": estimator.state_dict(), "config": dict(config), "steps": step}, output / "sge.pth")
    with open(output / "radius_accuracy.json", "w", encoding="utf-8") as stream:
        json.dump({"steps": step, "accuracy": accuracy}, stream, indent=2)
    print("saved", output / "sge.pth")


if __name__ == "__main__":
    main()
