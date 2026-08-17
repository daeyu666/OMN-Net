"""Train the validated OMN-Net local null-manifold extrapolation module."""
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
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models import (
    FixedSpatialDegradation,
    LocalNullManifoldNet,
    build_spectral_response,
    load_foundation_checkpoint,
)
from utils import (
    AverageMeter,
    CSVLogger,
    ensure_dir,
    get_device,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


def first_spectral_difference(x):
    return x[:, 1:] - x[:, :-1]


def second_spectral_difference(x):
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


def _has_option(arguments: List[str], option: str) -> bool:
    return any(x == option or x.startswith(option + "=") for x in arguments)


def parse_specific_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--foundation_checkpoint",
        type=str,
        default="./checkpoints/spectral_foundation/PaviaU/"
        "foundation_for_local_null.pth",
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
    p.add_argument("--proposal_grad_clip", type=float, default=1.0)
    p.add_argument("--local_lambda_l1", type=float, default=1.0)
    p.add_argument("--local_lambda_sam", type=float, default=0.3)
    p.add_argument("--local_lambda_sgrad1", type=float, default=0.1)
    p.add_argument("--local_lambda_sgrad2", type=float, default=0.05)
    p.add_argument("--local_lambda_residual", type=float, default=0.8)
    p.add_argument("--local_lambda_lr_hsi", type=float, default=0.2)
    p.add_argument("--local_lambda_lr_null", type=float, default=0.1)
    p.add_argument("--diagnose_only", action="store_true")
    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"
    default = (
        "./checkpoints/spectral_foundation/PaviaU/"
        "foundation_for_local_null.pth"
    )
    if cfg.dataset != "PaviaU" and cfg.foundation_checkpoint == default:
        cfg.foundation_checkpoint = os.path.join(
            cfg.checkpoint_root,
            "spectral_foundation",
            cfg.dataset,
            "foundation_for_local_null.pth",
        )
    return cfg


@torch.no_grad()
def build_targets(model, out: Dict[str, torch.Tensor], gt: torch.Tensor):
    c_gt = model.foundation.encode(gt, basis=out["basis"])
    target_null = model.geometry.project_null(c_gt)
    missing = target_null - out["null_seed_coefficients"]
    t = out["tangent_basis"]
    coordinates = torch.einsum("nrdhw,nrhw->ndhw", t, missing)
    target = torch.einsum("nrdhw,ndhw->nrhw", t, coordinates)
    target = model.geometry.project_null(target)
    return {
        "coefficients": c_gt,
        "target_null": target_null,
        "missing": missing,
        "tangent_target": target,
    }


def compute_losses(model, out, batch, hsi_deg, coeff_deg, sam_loss, cfg):
    gt = batch["gt"]
    lr_hsi = batch["lr_hsi"]
    pred = out["reconstructed_hsi"]
    targets = build_targets(model, out, gt)
    scale = out["coefficient_scale"].view(1, -1, 1, 1)

    hsi_l1 = F.l1_loss(pred, gt)
    sam = sam_loss(pred, gt)
    sgrad1 = F.l1_loss(
        first_spectral_difference(pred), first_spectral_difference(gt)
    )
    sgrad2 = F.l1_loss(
        second_spectral_difference(pred), second_spectral_difference(gt)
    )
    residual = F.smooth_l1_loss(
        out["tangent_residual"] / scale,
        targets["tangent_target"] / scale,
        beta=0.25,
    )
    lr_hsi_loss = F.l1_loss(
        hsi_deg(pred, target_size=lr_hsi.shape[-2:]), lr_hsi
    )

    corrected_null = (
        out["null_seed_coefficients"] + out["tangent_residual"]
    )
    lr_target_null = model.geometry.project_null(out["lr_coefficients"])
    degraded_null = coeff_deg(
        corrected_null, target_size=lr_target_null.shape[-2:]
    )
    lr_null_loss = F.smooth_l1_loss(
        degraded_null / scale, lr_target_null / scale, beta=0.25
    )

    total = (
        cfg.local_lambda_l1 * hsi_l1
        + cfg.local_lambda_sam * sam
        + cfg.local_lambda_sgrad1 * sgrad1
        + cfg.local_lambda_sgrad2 * sgrad2
        + cfg.local_lambda_residual * residual
        + cfg.local_lambda_lr_hsi * lr_hsi_loss
        + cfg.local_lambda_lr_null * lr_null_loss
    )
    return {
        "total": total,
        "hsi_l1": hsi_l1,
        "sam": sam,
        "sgrad1": sgrad1,
        "sgrad2": sgrad2,
        "residual": residual,
        "lr_hsi": lr_hsi_loss,
        "lr_null": lr_null_loss,
    }, targets


def diagnostics(out):
    return {
        "rho_tan": float(
            out["tangent_projection_energy_ratio"].detach().item()
        ),
        "rho_off": float(out["off_tangent_energy_ratio"].detach().item()),
        "sat": float(out["proposal_saturation_ratio"].detach().item()),
    }


def train_epoch(model, loader, optimizer, hsi_deg, coeff_deg, sam_loss, cfg, device):
    model.train()
    model.foundation.eval()
    meters = {
        key: AverageMeter()
        for key in ["total", "residual", "rho_tan", "rho_off", "sat"]
    }
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        losses, _ = compute_losses(
            model, out, batch, hsi_deg, coeff_deg, sam_loss, cfg
        )
        losses["total"].backward()
        if cfg.proposal_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.proposal_predictor.parameters(),
                cfg.proposal_grad_clip,
            )
        optimizer.step()
        n = batch["lr_hsi"].size(0)
        diag = diagnostics(out)
        meters["total"].update(losses["total"].detach().item(), n)
        meters["residual"].update(losses["residual"].detach().item(), n)
        for key in ["rho_tan", "rho_off", "sat"]:
            meters[key].update(diag[key], n)
    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def evaluate(model, loader, hsi_deg, coeff_deg, sam_loss, cfg, device):
    model.eval()
    metrics = {
        key: MetricAverager()
        for key in ["local", "anchor", "tangent_oracle", "basis_oracle"]
    }
    missing_energy = 0.0
    error_energy = 0.0
    diag_meters = {
        key: AverageMeter() for key in ["rho_tan", "rho_off", "sat"]
    }

    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        _, targets = compute_losses(
            model, out, batch, hsi_deg, coeff_deg, sam_loss, cfg
        )
        tangent_oracle = model.foundation.decode(
            out["anchor_coefficients"] + targets["tangent_target"],
            basis=out["basis"],
        )
        basis_oracle = model.foundation.decode(
            targets["coefficients"], basis=out["basis"]
        )
        for name, prediction in {
            "local": out["reconstructed_hsi"],
            "anchor": out["anchor_hsi"],
            "tangent_oracle": tangent_oracle,
            "basis_oracle": basis_oracle,
        }.items():
            metrics[name].update(
                calc_metrics(prediction, batch["gt"], cfg.scale_ratio)
            )

        missing = targets["missing"].double()
        remaining = (
            targets["missing"] - out["tangent_residual"]
        ).double()
        missing_energy += float(missing.square().sum().item())
        error_energy += float(remaining.square().sum().item())
        n = batch["lr_hsi"].size(0)
        diag = diagnostics(out)
        for key in diag_meters:
            diag_meters[key].update(diag[key], n)

    result = {}
    for prefix, meter in metrics.items():
        for key, value in meter.average().items():
            result[f"{prefix}_{key.lower()}"] = value
    for key, meter in diag_meters.items():
        result[key] = meter.avg
    missing_energy = max(missing_energy, 1e-30)
    result["missing_mse_capture"] = 1.0 - error_energy / missing_energy
    result["null_rrmse"] = math.sqrt(error_energy / missing_energy)
    return result


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    foundation, _ = load_foundation_checkpoint(
        cfg.foundation_checkpoint, info["n_bands"], device
    )
    model = LocalNullManifoldNet(
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

    optimizer = torch.optim.AdamW(
        model.proposal_predictor.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05
    )
    hsi_deg = FixedSpatialDegradation(info["n_bands"]).to(device)
    coeff_deg = FixedSpatialDegradation(foundation.basis_rank).to(device)
    sam_loss = SAMLoss()

    root = "local_null_manifold"
    ckpt_dir = os.path.join(cfg.checkpoint_root, root, cfg.dataset)
    out_dir = os.path.join(cfg.output_root, root, cfg.dataset)
    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)
    log_path = os.path.join(cfg.log_root, root, f"{cfg.dataset}.log")
    logger = CSVLogger(
        os.path.join(cfg.log_root, root, f"{cfg.dataset}.csv"),
        [
            "epoch",
            "lr",
            "psnr",
            "sam",
            "anchor_psnr",
            "oracle_psnr",
            "capture",
            "rho_tan",
            "rho_off",
            "sat",
        ],
    )

    start = evaluate(
        model, test_loader, hsi_deg, coeff_deg, sam_loss, cfg, device
    )
    write_log(
        log_path,
        "Local-null start | "
        f"PSNR={start['local_psnr']:.4f} "
        f"SAM={start['local_sam']:.4f} | "
        f"anchor={start['anchor_psnr']:.4f}/"
        f"{start['anchor_sam']:.4f} | "
        f"tangent_oracle={start['tangent_oracle_psnr']:.4f} | "
        f"basis_oracle={start['basis_oracle_psnr']:.4f}",
    )
    if cfg.diagnose_only:
        with open(
            os.path.join(out_dir, "diagnose_only.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(start, handle, indent=2, ensure_ascii=False)
        return

    best_psnr = start["local_psnr"]
    best_epoch = 0
    best_path = os.path.join(ckpt_dir, "local_null_best_psnr.pth")
    save_checkpoint(
        model,
        optimizer,
        0,
        best_psnr,
        best_path,
        extra={
            "model_role": "local_null_manifold",
            "dataset": cfg.dataset,
            "tangent_dimension": cfg.tangent_dimension,
        },
    )

    for epoch in range(1, cfg.epochs + 1):
        train_epoch(
            model,
            train_loader,
            optimizer,
            hsi_deg,
            coeff_deg,
            sam_loss,
            cfg,
            device,
        )
        val = evaluate(
            model, test_loader, hsi_deg, coeff_deg, sam_loss, cfg, device
        )
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        logger.write(
            {
                "epoch": epoch,
                "lr": lr,
                "psnr": val["local_psnr"],
                "sam": val["local_sam"],
                "anchor_psnr": val["anchor_psnr"],
                "oracle_psnr": val["tangent_oracle_psnr"],
                "capture": val["missing_mse_capture"],
                "rho_tan": val["rho_tan"],
                "rho_off": val["rho_off"],
                "sat": val["sat"],
            }
        )
        write_log(
            log_path,
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"PSNR={val['local_psnr']:.4f} SAM={val['local_sam']:.4f} | "
            f"capture={100.0*val['missing_mse_capture']:.2f}% | "
            f"rho_tan={val['rho_tan']:.4f} "
            f"off={val['rho_off']:.4f} sat={100.0*val['sat']:.2f}%",
        )
        if val["local_psnr"] > best_psnr:
            best_psnr = val["local_psnr"]
            best_epoch = epoch
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_psnr,
                best_path,
                extra={
                    "model_role": "local_null_manifold",
                    "dataset": cfg.dataset,
                    "tangent_dimension": cfg.tangent_dimension,
                },
            )

    with open(
        os.path.join(out_dir, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_psnr": best_psnr,
                "checkpoint": best_path,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Best local-null PSNR={best_psnr:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
