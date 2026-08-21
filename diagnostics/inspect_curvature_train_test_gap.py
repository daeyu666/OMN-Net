"""E22: train-test curvature identifiability gap diagnostic for OMN-Net.

E16 showed a strong LR-HSI-derived curvature admissible subspace, while E17/
E17-b realized only a small fraction of that oracle capacity on the held-out
center test region.  E22 determines whether the bottleneck is primarily:

1) fitting/identifiability: the fixed curvature predictor is also weak on the
   training region; or
2) generalization: the predictor fits the training region substantially better
   than the held-out test region.

No network is trained here.  One fixed rank-6 E17/E17-b checkpoint is evaluated
on exactly the same curvature target used during training.

For each split the script reports:
* Stage-2 / predicted / GT-pixel-scalar / curvature-oracle / full-Pcomp metrics;
* the exact normalized SmoothL1 curvature objective used by E17/E17-b;
* curvature capture, Pcomp capture, amplitude ratio and cosine alignment;
* target/predicted curvature RMS and the fraction of PSNR oracle gain realized.

The training region is evaluated twice:
* train_native: deterministic, augmentation disabled.  This is the primary
  measure of whether the learned mapping fits the actual training scene region.
* train_augmented: one deterministic seeded augmentation view per patch,
  matching the transformation family seen during training.  It is secondary.

GT is used only to form the same supervised curvature target and diagnostic
oracles; no parameter is updated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import parse_args
from data_loader import build_datasets
from metrics import MetricAverager, calc_metrics
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.local_curvature_extrapolation import LocalCurvatureExtrapolationNet
from models.local_curvature_extrapolation_e17b import LocalCurvatureExtrapolationE17BNet
from train_local_curvature_extrapolation import build_targets
from utils import ensure_dir, get_device, load_checkpoint, move_to_device, set_seed


def _has_option(arguments: List[str], option: str) -> bool:
    return any(x == option or x.startswith(option + "=") for x in arguments)


def parse_specific_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--foundation_checkpoint",
        type=str,
        default="./checkpoints/RAPD-Net/basis_for_stage2.pth",
    )
    p.add_argument(
        "--local_checkpoint",
        type=str,
        default=(
            "./checkpoints/local_null_manifold/PaviaU/"
            "local_null_best_psnr.pth"
        ),
    )
    p.add_argument(
        "--curvature_checkpoint",
        type=str,
        default=(
            "./checkpoints/local_curvature_extrapolation_e17b/PaviaU/"
            "curvature_e17b_best_psnr.pth"
        ),
    )
    p.add_argument(
        "--curvature_variant",
        type=str,
        choices=["e17", "e17b"],
        default="e17b",
    )

    p.add_argument("--anchor_ridge_ratio", type=float, default=1e-3)
    p.add_argument("--projector_tolerance", type=float, default=1e-6)
    p.add_argument("--tangent_dimension", type=int, default=4)
    p.add_argument("--tangent_kernel_size", type=int, default=5)
    p.add_argument("--tangent_dilation", type=int, default=2)
    p.add_argument("--tangent_chunk_pixels", type=int, default=2048)
    p.add_argument("--proposal_amplitude_multiplier", type=float, default=8.0)
    p.add_argument("--proposal_predictor_hidden", type=int, default=96)
    p.add_argument("--proposal_predictor_blocks", type=int, default=4)

    p.add_argument("--diagnostic_image_size", type=int, default=128)
    p.add_argument("--curvature_rank", type=int, default=6)
    p.add_argument("--curvature_svd_chunk_pixels", type=int, default=2048)
    p.add_argument("--curvature_svd_tolerance", type=float, default=1e-5)
    p.add_argument("--curvature_abs_tolerance", type=float, default=1e-9)
    p.add_argument(
        "--curvature_proposal_amplitude_multiplier", type=float, default=8.0
    )
    p.add_argument("--curvature_predictor_hidden", type=int, default=96)
    p.add_argument("--curvature_predictor_blocks", type=int, default=4)
    p.add_argument("--curvature_loss_beta", type=float, default=0.25)

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"
    if cfg.curvature_rank < 1 or cfg.curvature_rank > 8:
        raise ValueError("curvature_rank must be in [1,8]")
    if cfg.curvature_loss_beta <= 0:
        raise ValueError("curvature_loss_beta must be positive")
    cfg.image_size = cfg.diagnostic_image_size
    return cfg


def _read_checkpoint_metadata(path: str, device: torch.device) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Curvature checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    return {
        "epoch": int(state.get("epoch", 0)),
        "best_metric": float(state.get("best_metric", 0.0)),
        "extra": state.get("extra", {}) or {},
    }


def build_model(cfg, info, device):
    foundation, _ = load_foundation_checkpoint(
        cfg.foundation_checkpoint, info["n_bands"], device
    )
    local_model = LocalNullManifoldNet(
        foundation=foundation,
        spectral_response=build_spectral_response(info).to(device),
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        projector_tolerance=cfg.projector_tolerance,
        tangent_dimension=cfg.tangent_dimension,
        tangent_kernel_size=cfg.tangent_kernel_size,
        tangent_dilation=cfg.tangent_dilation,
        tangent_chunk_pixels=cfg.tangent_chunk_pixels,
        proposal_amplitude_multiplier=cfg.proposal_amplitude_multiplier,
        predictor_hidden_channels=cfg.proposal_predictor_hidden,
        predictor_blocks=cfg.proposal_predictor_blocks,
    ).to(device)
    local_epoch, local_best = load_checkpoint(
        local_model,
        cfg.local_checkpoint,
        optimizer=None,
        strict=True,
        map_location=device,
        load_optimizer=False,
    )
    local_model.eval()
    for parameter in local_model.parameters():
        parameter.requires_grad_(False)

    model_cls = (
        LocalCurvatureExtrapolationNet
        if cfg.curvature_variant == "e17"
        else LocalCurvatureExtrapolationE17BNet
    )
    model = model_cls(
        local_model=local_model,
        curvature_rank=cfg.curvature_rank,
        curvature_svd_chunk_pixels=cfg.curvature_svd_chunk_pixels,
        curvature_svd_tolerance=cfg.curvature_svd_tolerance,
        curvature_abs_tolerance=cfg.curvature_abs_tolerance,
        proposal_amplitude_multiplier=cfg.curvature_proposal_amplitude_multiplier,
        predictor_hidden_channels=cfg.curvature_predictor_hidden,
        predictor_blocks=cfg.curvature_predictor_blocks,
    ).to(device)

    metadata = _read_checkpoint_metadata(cfg.curvature_checkpoint, device)
    checkpoint_rank = metadata["extra"].get("curvature_rank")
    if checkpoint_rank is not None and int(checkpoint_rank) != int(cfg.curvature_rank):
        raise ValueError(
            "Curvature checkpoint rank mismatch: "
            f"checkpoint rank={checkpoint_rank}, requested rank={cfg.curvature_rank}."
        )
    role = str(metadata["extra"].get("model_role", ""))
    if cfg.curvature_variant == "e17b" and role and "e17b" not in role:
        raise ValueError(
            f"Checkpoint role '{role}' does not look like E17-b."
        )
    if cfg.curvature_variant == "e17" and role and "e17b" in role:
        raise ValueError(
            f"Checkpoint role '{role}' is E17-b but --curvature_variant=e17."
        )

    curvature_epoch, curvature_best = load_checkpoint(
        model,
        cfg.curvature_checkpoint,
        optimizer=None,
        strict=True,
        map_location=device,
        load_optimizer=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return {
        "model": model,
        "local_epoch": int(local_epoch),
        "local_best": float(local_best),
        "curvature_epoch": int(curvature_epoch),
        "curvature_best": float(curvature_best),
        "checkpoint_metadata": metadata,
    }


def _make_loader(dataset, batch_size: int, pin_memory: bool = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


def pixel_scalar_oracle(prediction: torch.Tensor, target: torch.Tensor):
    pred64 = prediction.double()
    target64 = target.double()
    dot = (pred64 * target64).sum(dim=1, keepdim=True)
    energy = pred64.square().sum(dim=1, keepdim=True)
    alpha = dot / energy.clamp_min(1e-30)
    return alpha.to(prediction.dtype) * prediction, alpha


@torch.no_grad()
def evaluate_split(model, loader, cfg, device):
    model.eval()
    meters = {
        name: MetricAverager()
        for name in [
            "stage2",
            "pred",
            "pixel_scalar_oracle",
            "curvature_oracle",
            "full_pcomp",
        ]
    }

    loss_sum = 0.0
    loss_weight = 0.0
    pixel_count = 0.0

    curvature_energy = 0.0
    pred_energy = 0.0
    pred_curvature_dot = 0.0
    pred_curvature_error = 0.0
    pcomp_energy = 0.0
    pred_pcomp_error = 0.0
    oracle_pcomp_error = 0.0

    target_abs_sum = 0.0
    pred_abs_sum = 0.0
    alpha_sum = 0.0
    alpha_sq_sum = 0.0
    alpha_count = 0.0

    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        targets = build_targets(model, out, batch["gt"])

        pred = out["curvature_residual"]
        target = targets["curvature"]
        scale = out["coefficient_scale"].view(1, -1, 1, 1)

        loss = F.smooth_l1_loss(
            pred / scale,
            target / scale,
            beta=cfg.curvature_loss_beta,
            reduction="mean",
        )
        weight = float(pred.numel())
        loss_sum += float(loss.item()) * weight
        loss_weight += weight

        pixel_scalar_residual, alpha = pixel_scalar_oracle(pred, target)
        pixel_scalar_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + pixel_scalar_residual,
            basis=out["basis"],
        )
        curvature_oracle_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + target,
            basis=out["basis"],
        )
        full_pcomp_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + targets["pcomp"],
            basis=out["basis"],
        )

        meters["stage2"].update(
            calc_metrics(out["reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
        )
        meters["pred"].update(
            calc_metrics(
                out["curvature_reconstructed_hsi"], batch["gt"], cfg.scale_ratio
            )
        )
        meters["pixel_scalar_oracle"].update(
            calc_metrics(pixel_scalar_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["curvature_oracle"].update(
            calc_metrics(curvature_oracle_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["full_pcomp"].update(
            calc_metrics(full_pcomp_hsi, batch["gt"], cfg.scale_ratio)
        )

        pred64 = pred.double()
        target64 = target.double()
        pcomp64 = targets["pcomp"].double()

        curvature_energy += float(target64.square().sum().item())
        pred_energy += float(pred64.square().sum().item())
        pred_curvature_dot += float((pred64 * target64).sum().item())
        pred_curvature_error += float(
            (pred64 - target64).square().sum().item()
        )
        pcomp_energy += float(pcomp64.square().sum().item())
        pred_pcomp_error += float((pred64 - pcomp64).square().sum().item())
        oracle_pcomp_error += float(
            (target64 - pcomp64).square().sum().item()
        )

        target_abs_sum += float(target64.abs().sum().item())
        pred_abs_sum += float(pred64.abs().sum().item())
        pixel_count += float(pred.size(0) * pred.size(2) * pred.size(3))

        nonzero = pred64.square().sum(dim=1, keepdim=True) > 1e-30
        valid_alpha = nonzero.to(alpha.dtype)
        alpha_sum += float((alpha * valid_alpha).sum().item())
        alpha_sq_sum += float((alpha.square() * valid_alpha).sum().item())
        alpha_count += float(valid_alpha.sum().item())

    result: Dict[str, float] = {}
    for name, meter in meters.items():
        for key, value in meter.average().items():
            result[f"{name}_{key.lower()}"] = float(value)

    curvature_energy_safe = max(curvature_energy, 1e-30)
    pred_energy_safe = max(pred_energy, 1e-30)
    pcomp_energy_safe = max(pcomp_energy, 1e-30)
    coeff_count = max(loss_weight, 1.0)

    result["normalized_smooth_l1"] = loss_sum / coeff_count
    result["pred_curvature_capture"] = (
        1.0 - pred_curvature_error / curvature_energy_safe
    )
    result["pred_pcomp_capture"] = 1.0 - pred_pcomp_error / pcomp_energy_safe
    result["oracle_pcomp_capture"] = 1.0 - oracle_pcomp_error / pcomp_energy_safe
    result["curvature_amplitude_ratio"] = math.sqrt(
        pred_energy_safe / curvature_energy_safe
    )
    result["curvature_cosine"] = pred_curvature_dot / math.sqrt(
        pred_energy_safe * curvature_energy_safe
    )
    result["target_curvature_rms"] = math.sqrt(curvature_energy_safe / coeff_count)
    result["pred_curvature_rms"] = math.sqrt(pred_energy_safe / coeff_count)
    result["target_curvature_mean_abs"] = target_abs_sum / coeff_count
    result["pred_curvature_mean_abs"] = pred_abs_sum / coeff_count
    result["pixel_count"] = pixel_count
    result["mean_pixel_optimal_scalar"] = (
        alpha_sum / max(alpha_count, 1.0)
    )
    result["rms_pixel_optimal_scalar"] = math.sqrt(
        alpha_sq_sum / max(alpha_count, 1.0)
    )

    stage2_psnr = result["stage2_psnr"]
    pred_psnr = result["pred_psnr"]
    oracle_psnr = result["curvature_oracle_psnr"]
    oracle_gain = oracle_psnr - stage2_psnr
    pred_gain = pred_psnr - stage2_psnr
    result["pred_psnr_gain_over_stage2"] = pred_gain
    result["oracle_psnr_gain_over_stage2"] = oracle_gain
    result["psnr_oracle_gain_realization"] = (
        pred_gain / oracle_gain if abs(oracle_gain) > 1e-12 else 0.0
    )
    return result


def _gap(train_result: Dict[str, float], test_result: Dict[str, float]):
    keys = [
        "normalized_smooth_l1",
        "pred_curvature_capture",
        "pred_pcomp_capture",
        "curvature_amplitude_ratio",
        "curvature_cosine",
        "pred_psnr_gain_over_stage2",
        "psnr_oracle_gain_realization",
        "pixel_scalar_oracle_psnr",
    ]
    out = {}
    for key in keys:
        train_value = float(train_result[key])
        test_value = float(test_result[key])
        out[f"{key}_train_minus_test"] = train_value - test_value
        if abs(train_value) > 1e-30:
            out[f"{key}_test_over_train"] = test_value / train_value
    train_loss = max(float(train_result["normalized_smooth_l1"]), 1e-30)
    out["normalized_smooth_l1_test_over_train"] = (
        float(test_result["normalized_smooth_l1"]) / train_loss
    )
    return out


def _print_split(name: str, r: Dict[str, float]):
    print(
        f"{name} | "
        f"Stage2={r['stage2_psnr']:.4f} "
        f"Pred={r['pred_psnr']:.4f} "
        f"PixelScalar={r['pixel_scalar_oracle_psnr']:.4f} "
        f"CurvOracle={r['curvature_oracle_psnr']:.4f} "
        f"FullPcomp={r['full_pcomp_psnr']:.4f}"
    )
    print(
        f"{name} identifiability | "
        f"loss={r['normalized_smooth_l1']:.6f} "
        f"CurvCap={100.0*r['pred_curvature_capture']:.2f}% "
        f"PcompCap={100.0*r['pred_pcomp_capture']:.2f}% "
        f"Amp={r['curvature_amplitude_ratio']:.3f} "
        f"Cos={r['curvature_cosine']:.3f} "
        f"PSNRgain={r['pred_psnr_gain_over_stage2']:+.4f} "
        f"OracleRealize={100.0*r['psnr_oracle_gain_realization']:.2f}%"
    )


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    train_set, test_set, info = build_datasets(cfg)
    model_info = build_model(cfg, info, device)
    model = model_info["model"]

    # Deterministic native training patches: primary fitting diagnostic.
    train_set.augment = False
    train_native_loader = _make_loader(
        train_set, batch_size=max(1, int(cfg.batch_size))
    )
    test_loader = _make_loader(test_set, batch_size=1)

    train_native = evaluate_split(model, train_native_loader, cfg, device)
    test = evaluate_split(model, test_loader, cfg, device)

    # One deterministic augmented view per training patch, matching the training
    # augmentation family while keeping E22 reproducible.
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    train_set.augment = True
    train_augmented_loader = _make_loader(
        train_set, batch_size=max(1, int(cfg.batch_size))
    )
    train_augmented = evaluate_split(
        model, train_augmented_loader, cfg, device
    )
    train_set.augment = False

    native_gap = _gap(train_native, test)
    augmented_gap = _gap(train_augmented, test)

    print(
        "E22 curvature train-test gap | "
        f"variant={cfg.curvature_variant} rank={cfg.curvature_rank} "
        f"checkpoint_epoch={model_info['curvature_epoch']} "
        f"checkpoint_best={model_info['curvature_best']:.4f}"
    )
    _print_split("Train-native", train_native)
    _print_split("Train-aug", train_augmented)
    _print_split("Test", test)
    print(
        "Native->Test gap | "
        f"loss x{native_gap['normalized_smooth_l1_test_over_train']:.3f} | "
        f"Cos {train_native['curvature_cosine']:.3f}->{test['curvature_cosine']:.3f} | "
        f"CurvCap {100.0*train_native['pred_curvature_capture']:.2f}%"
        f"->{100.0*test['pred_curvature_capture']:.2f}% | "
        f"OracleRealize {100.0*train_native['psnr_oracle_gain_realization']:.2f}%"
        f"->{100.0*test['psnr_oracle_gain_realization']:.2f}%"
    )

    output = {
        "experiment": "E22 curvature train-test identifiability gap",
        "dataset": cfg.dataset,
        "curvature_variant": cfg.curvature_variant,
        "curvature_rank": cfg.curvature_rank,
        "foundation_checkpoint": cfg.foundation_checkpoint,
        "local_checkpoint": cfg.local_checkpoint,
        "curvature_checkpoint": cfg.curvature_checkpoint,
        "local_checkpoint_epoch": model_info["local_epoch"],
        "local_checkpoint_best": model_info["local_best"],
        "curvature_checkpoint_epoch": model_info["curvature_epoch"],
        "curvature_checkpoint_best": model_info["curvature_best"],
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "train_native": train_native,
        "train_augmented": train_augmented,
        "test": test,
        "native_to_test_gap": native_gap,
        "augmented_to_test_gap": augmented_gap,
    }

    out_dir = os.path.join(
        cfg.output_root,
        "diagnostics",
        "curvature_train_test_gap",
        cfg.dataset,
    )
    ensure_dir(out_dir)
    out_path = os.path.join(
        out_dir,
        f"curvature_train_test_gap_{cfg.curvature_variant}.json",
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
