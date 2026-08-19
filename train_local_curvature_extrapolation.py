"""E17: train LR-HSI curvature-authorized complement extrapolation.

E16 showed that LR-HSI second-order curvature spans a useful part of the true
Stage-2 P_comp residual.  E17 asks only whether the amplitude of that fixed
curvature subspace is predictable from legal HR-MSI / frozen Stage-2 context.

The predictor emits a global coefficient proposal.  The actual correction is
P_curv r, never a free P_comp residual.  Training supervision is the GT
projection onto the LR-HSI-derived curvature subspace; no curvature direction
comes from GT.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from metrics import MetricAverager, calc_metrics
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.local_curvature_extrapolation import (
    LocalCurvatureExtrapolationNet,
    project_to_curvature,
)
from models.nonlocal_complement import (
    flatten_spatial,
    flatten_tangent,
    project_complement_vectors,
    unflatten_spatial,
)
from utils import (
    AverageMeter,
    CSVLogger,
    ensure_dir,
    get_device,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


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
        default="./checkpoints/local_null_manifold/PaviaU/local_null_best_psnr.pth",
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

    p.add_argument("--curvature_rank", type=int, default=6)
    p.add_argument("--curvature_svd_chunk_pixels", type=int, default=2048)
    p.add_argument("--curvature_svd_tolerance", type=float, default=1e-5)
    p.add_argument("--curvature_abs_tolerance", type=float, default=1e-9)
    p.add_argument("--curvature_proposal_amplitude_multiplier", type=float, default=8.0)
    p.add_argument("--curvature_predictor_hidden", type=int, default=96)
    p.add_argument("--curvature_predictor_blocks", type=int, default=4)
    p.add_argument("--curvature_loss_beta", type=float, default=0.25)
    p.add_argument("--curvature_grad_clip", type=float, default=1.0)
    p.add_argument("--curvature_eval_interval", type=int, default=10)

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
    if cfg.curvature_loss_beta <= 0 or cfg.curvature_eval_interval < 1:
        raise ValueError("invalid curvature loss/eval settings")
    return cfg


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

    model = LocalCurvatureExtrapolationNet(
        local_model=local_model,
        curvature_rank=cfg.curvature_rank,
        curvature_svd_chunk_pixels=cfg.curvature_svd_chunk_pixels,
        curvature_svd_tolerance=cfg.curvature_svd_tolerance,
        curvature_abs_tolerance=cfg.curvature_abs_tolerance,
        proposal_amplitude_multiplier=cfg.curvature_proposal_amplitude_multiplier,
        predictor_hidden_channels=cfg.curvature_predictor_hidden,
        predictor_blocks=cfg.curvature_predictor_blocks,
    ).to(device)
    return model, foundation, local_epoch, local_best


@torch.no_grad()
def build_targets(model, out: Dict[str, torch.Tensor], gt: torch.Tensor):
    gt_coeff = model.local_model.foundation.encode(gt, basis=out["basis"])
    remaining = gt_coeff - out["corrected_coefficients"]
    remaining_flat = flatten_spatial(remaining)
    tangent_flat = flatten_tangent(out["tangent_basis"])
    targets = []
    for b in range(remaining.size(0)):
        targets.append(
            project_complement_vectors(
                remaining_flat[b],
                tangent_flat[b],
                model.local_model.geometry.null_projector,
            )
        )
    pcomp_flat = torch.stack(targets, dim=0)
    pcomp = unflatten_spatial(
        pcomp_flat, remaining.size(2), remaining.size(3)
    )
    curvature = project_to_curvature(out["curvature_basis"], pcomp)
    return {
        "gt_coefficients": gt_coeff,
        "remaining": remaining,
        "pcomp": pcomp,
        "curvature": curvature,
    }


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    meter = AverageMeter()
    saturation = AverageMeter()
    valid_rank = AverageMeter()
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        targets = build_targets(model, out, batch["gt"])
        scale = out["coefficient_scale"].view(1, -1, 1, 1)
        loss = F.smooth_l1_loss(
            out["curvature_residual"] / scale,
            targets["curvature"] / scale,
            beta=cfg.curvature_loss_beta,
        )
        loss.backward()
        if cfg.curvature_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), cfg.curvature_grad_clip
            )
        optimizer.step()

        n = batch["gt"].size(0)
        meter.update(loss.detach().item(), n)
        saturation.update(
            (out["normalized_curvature_proposal"].detach().abs() > 0.98)
            .float().mean().item(),
            n,
        )
        valid_rank.update(
            out["curvature_valid_mask"].float().sum(dim=1).mean().item(), n
        )
    return {
        "loss": meter.avg,
        "saturation": saturation.avg,
        "valid_rank": valid_rank.avg,
    }


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    meters = {
        name: MetricAverager()
        for name in ["stage2", "pred", "curvature_oracle", "full_pcomp"]
    }
    pcomp_energy = 0.0
    pred_pcomp_error = 0.0
    oracle_pcomp_error = 0.0
    curvature_energy = 0.0
    pred_curvature_error = 0.0
    valid_rank = AverageMeter()
    saturation = AverageMeter()

    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        targets = build_targets(model, out, batch["gt"])

        oracle_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + targets["curvature"],
            basis=out["basis"],
        )
        full_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + targets["pcomp"],
            basis=out["basis"],
        )
        meters["stage2"].update(
            calc_metrics(out["reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
        )
        meters["pred"].update(
            calc_metrics(out["curvature_reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
        )
        meters["curvature_oracle"].update(
            calc_metrics(oracle_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["full_pcomp"].update(
            calc_metrics(full_hsi, batch["gt"], cfg.scale_ratio)
        )

        pcomp_energy += float(targets["pcomp"].double().square().sum().item())
        pred_pcomp_error += float(
            (out["curvature_residual"].double() - targets["pcomp"].double())
            .square().sum().item()
        )
        oracle_pcomp_error += float(
            (targets["curvature"].double() - targets["pcomp"].double())
            .square().sum().item()
        )
        curvature_energy += float(
            targets["curvature"].double().square().sum().item()
        )
        pred_curvature_error += float(
            (out["curvature_residual"].double() - targets["curvature"].double())
            .square().sum().item()
        )
        n = batch["gt"].size(0)
        valid_rank.update(
            out["curvature_valid_mask"].float().sum(dim=1).mean().item(), n
        )
        saturation.update(
            (out["normalized_curvature_proposal"].abs() > 0.98)
            .float().mean().item(), n
        )

    result = {}
    for name, meter in meters.items():
        for key, value in meter.average().items():
            result[f"{name}_{key.lower()}"] = value
    pcomp_energy = max(pcomp_energy, 1e-30)
    curvature_energy = max(curvature_energy, 1e-30)
    result["pred_pcomp_capture"] = 1.0 - pred_pcomp_error / pcomp_energy
    result["oracle_pcomp_capture"] = 1.0 - oracle_pcomp_error / pcomp_energy
    result["pred_curvature_capture"] = 1.0 - pred_curvature_error / curvature_energy
    result["mean_valid_curvature_rank"] = valid_rank.avg
    result["proposal_saturation"] = saturation.avg
    return result


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    model, foundation, local_epoch, local_best = build_model(cfg, info, device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05
    )

    root = "local_curvature_extrapolation"
    ckpt_dir = os.path.join(cfg.checkpoint_root, root, cfg.dataset)
    out_dir = os.path.join(cfg.output_root, root, cfg.dataset)
    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)
    log_path = os.path.join(cfg.log_root, root, f"{cfg.dataset}.log")
    csv_path = os.path.join(cfg.log_root, root, f"{cfg.dataset}.csv")
    logger = CSVLogger(
        csv_path,
        [
            "epoch", "lr", "loss", "psnr", "sam", "stage2_psnr",
            "oracle_psnr", "pred_pcomp_capture", "oracle_pcomp_capture",
            "pred_curvature_capture", "valid_rank", "saturation",
        ],
    )

    start = evaluate(model, test_loader, cfg, device)
    write_log(
        log_path,
        "E17 start | "
        f"Stage2 epoch={local_epoch} best={local_best:.4f} | "
        f"rank={cfg.curvature_rank} | "
        f"Stage2={start['stage2_psnr']:.4f} "
        f"Pred={start['pred_psnr']:.4f} "
        f"Oracle={start['curvature_oracle_psnr']:.4f} "
        f"FullPcomp={start['full_pcomp_psnr']:.4f} | "
        f"oracle_capture={100.0*start['oracle_pcomp_capture']:.2f}%"
    )

    best_psnr = start["pred_psnr"]
    best_epoch = 0
    best_path = os.path.join(ckpt_dir, "curvature_best_psnr.pth")
    save_checkpoint(
        model, optimizer, 0, best_psnr, best_path,
        extra={
            "model_role": "local_curvature_extrapolation",
            "dataset": cfg.dataset,
            "curvature_rank": cfg.curvature_rank,
            "local_checkpoint": cfg.local_checkpoint,
            "supervision": "GT projection onto LR-HSI-derived curvature subspace",
        },
    )

    last_eval = start
    for epoch in range(1, cfg.epochs + 1):
        stat = train_epoch(model, train_loader, optimizer, cfg, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        should_eval = (
            epoch % cfg.curvature_eval_interval == 0 or epoch == cfg.epochs
        )
        if should_eval:
            last_eval = evaluate(model, test_loader, cfg, device)
            write_log(
                log_path,
                f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
                f"loss={stat['loss']:.6f} | "
                f"PSNR={last_eval['pred_psnr']:.4f} "
                f"SAM={last_eval['pred_sam']:.4f} "
                f"Stage2={last_eval['stage2_psnr']:.4f} "
                f"Oracle={last_eval['curvature_oracle_psnr']:.4f} | "
                f"PcompCap={100.0*last_eval['pred_pcomp_capture']:.2f}% "
                f"CurvCap={100.0*last_eval['pred_curvature_capture']:.2f}% "
                f"validR={last_eval['mean_valid_curvature_rank']:.2f} "
                f"sat={100.0*last_eval['proposal_saturation']:.2f}%"
            )
            if last_eval["pred_psnr"] > best_psnr:
                best_psnr = last_eval["pred_psnr"]
                best_epoch = epoch
                save_checkpoint(
                    model, optimizer, epoch, best_psnr, best_path,
                    extra={
                        "model_role": "local_curvature_extrapolation",
                        "dataset": cfg.dataset,
                        "curvature_rank": cfg.curvature_rank,
                        "local_checkpoint": cfg.local_checkpoint,
                        "supervision": "GT projection onto LR-HSI-derived curvature subspace",
                    },
                )
        else:
            write_log(
                log_path,
                f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
                f"loss={stat['loss']:.6f} validR={stat['valid_rank']:.2f} "
                f"sat={100.0*stat['saturation']:.2f}%"
            )

        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": stat["loss"],
            "valid_rank": stat["valid_rank"],
            "saturation": stat["saturation"],
        }
        if should_eval:
            row.update({
                "psnr": last_eval["pred_psnr"],
                "sam": last_eval["pred_sam"],
                "stage2_psnr": last_eval["stage2_psnr"],
                "oracle_psnr": last_eval["curvature_oracle_psnr"],
                "pred_pcomp_capture": last_eval["pred_pcomp_capture"],
                "oracle_pcomp_capture": last_eval["oracle_pcomp_capture"],
                "pred_curvature_capture": last_eval["pred_curvature_capture"],
            })
        logger.write(row)

    summary = {
        "best_epoch": best_epoch,
        "best_psnr": best_psnr,
        "checkpoint": best_path,
        "curvature_rank": cfg.curvature_rank,
        "stage2_psnr": start["stage2_psnr"],
        "curvature_oracle_psnr": start["curvature_oracle_psnr"],
        "full_pcomp_psnr": start["full_pcomp_psnr"],
        "last_eval": last_eval,
    }
    with open(
        os.path.join(out_dir, "training_summary.json"),
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"Best E17 PSNR={best_psnr:.4f} at epoch {best_epoch} | "
        f"Stage2={start['stage2_psnr']:.4f} "
        f"Oracle={start['curvature_oracle_psnr']:.4f}"
    )


if __name__ == "__main__":
    main()
