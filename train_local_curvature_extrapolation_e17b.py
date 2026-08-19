"""E17-b: train signed-curvature-aware LR-HSI complement extrapolation.

Single-variable change from E17:
    expose the eight signed P_comp-projected LR-HSI curvature vectors to the
    global proposal predictor.

Everything else remains unchanged: frozen Stage-2, rank-6 curvature
authorization, 32-D global proposal, P_curv projection, and SmoothL1
supervision against the GT residual projected onto the LR-HSI-derived curvature
subspace.
"""
from __future__ import annotations

import json
import math
import os

import torch

from data_loader import build_loaders
from metrics import MetricAverager, calc_metrics
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.local_curvature_extrapolation_e17b import LocalCurvatureExtrapolationE17BNet
from train_local_curvature_extrapolation import (
    build_targets,
    parse_specific_args,
    train_epoch,
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

    model = LocalCurvatureExtrapolationE17BNet(
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
def evaluate(model, loader, cfg, device):
    model.eval()
    meters = {
        name: MetricAverager()
        for name in [
            "stage2",
            "pred",
            "curvature_oracle",
            "full_pcomp",
            "sample_scalar_oracle",
            "pixel_scalar_oracle",
        ]
    }
    pcomp_energy = 0.0
    pred_pcomp_error = 0.0
    oracle_pcomp_error = 0.0
    curvature_energy = 0.0
    pred_curvature_energy = 0.0
    pred_curvature_dot = 0.0
    pred_curvature_error = 0.0
    valid_rank = AverageMeter()
    saturation = AverageMeter()
    sample_alpha = AverageMeter()

    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        targets = build_targets(model, out, batch["gt"])

        pred = out["curvature_residual"]
        target = targets["curvature"]

        oracle_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + target,
            basis=out["basis"],
        )
        full_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + targets["pcomp"],
            basis=out["basis"],
        )

        batch_dot = (pred.double() * target.double()).sum()
        batch_pred_energy = pred.double().square().sum()
        alpha = float(
            (batch_dot / batch_pred_energy.clamp_min(1e-30)).item()
        )
        sample_scalar_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + alpha * pred,
            basis=out["basis"],
        )

        pixel_dot = (pred.double() * target.double()).sum(dim=1, keepdim=True)
        pixel_pred_energy = pred.double().square().sum(dim=1, keepdim=True)
        pixel_alpha = pixel_dot / pixel_pred_energy.clamp_min(1e-30)
        pixel_scalar_residual = pixel_alpha.to(pred.dtype) * pred
        pixel_scalar_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + pixel_scalar_residual,
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
        meters["curvature_oracle"].update(
            calc_metrics(oracle_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["full_pcomp"].update(
            calc_metrics(full_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["sample_scalar_oracle"].update(
            calc_metrics(sample_scalar_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["pixel_scalar_oracle"].update(
            calc_metrics(pixel_scalar_hsi, batch["gt"], cfg.scale_ratio)
        )

        pcomp_energy += float(targets["pcomp"].double().square().sum().item())
        pred_pcomp_error += float(
            (pred.double() - targets["pcomp"].double()).square().sum().item()
        )
        oracle_pcomp_error += float(
            (target.double() - targets["pcomp"].double()).square().sum().item()
        )
        curvature_energy += float(target.double().square().sum().item())
        pred_curvature_energy += float(pred.double().square().sum().item())
        pred_curvature_dot += float(
            (pred.double() * target.double()).sum().item()
        )
        pred_curvature_error += float(
            (pred.double() - target.double()).square().sum().item()
        )

        n = batch["gt"].size(0)
        sample_alpha.update(alpha, n)
        valid_rank.update(
            out["curvature_valid_mask"].float().sum(dim=1).mean().item(), n
        )
        saturation.update(
            (out["normalized_curvature_proposal"].abs() > 0.98)
            .float().mean().item(),
            n,
        )

    result = {}
    for name, meter in meters.items():
        for key, value in meter.average().items():
            result[f"{name}_{key.lower()}"] = value

    pcomp_energy = max(pcomp_energy, 1e-30)
    curvature_energy = max(curvature_energy, 1e-30)
    result["pred_pcomp_capture"] = 1.0 - pred_pcomp_error / pcomp_energy
    result["oracle_pcomp_capture"] = 1.0 - oracle_pcomp_error / pcomp_energy
    result["pred_curvature_capture"] = (
        1.0 - pred_curvature_error / curvature_energy
    )
    result["pred_curvature_amplitude_ratio"] = math.sqrt(
        pred_curvature_energy / curvature_energy
    )
    if pred_curvature_energy > 1e-30:
        result["pred_curvature_cosine"] = pred_curvature_dot / math.sqrt(
            pred_curvature_energy * curvature_energy
        )
    else:
        result["pred_curvature_cosine"] = 0.0
    result["mean_sample_optimal_scalar"] = sample_alpha.avg
    result["mean_valid_curvature_rank"] = valid_rank.avg
    result["proposal_saturation"] = saturation.avg
    return result


def _checkpoint_extra(cfg):
    return {
        "model_role": "local_curvature_extrapolation_e17b",
        "experiment": "E17-b signed projected curvature bank",
        "dataset": cfg.dataset,
        "curvature_rank": cfg.curvature_rank,
        "local_checkpoint": cfg.local_checkpoint,
        "supervision": "GT projection onto LR-HSI-derived curvature subspace",
        "single_variable_change": (
            "8 signed P_comp-projected LR-HSI curvature vectors added to predictor input"
        ),
    }


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

    root = "local_curvature_extrapolation_e17b"
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
            "oracle_psnr", "sample_scalar_psnr", "pixel_scalar_psnr",
            "pred_pcomp_capture", "oracle_pcomp_capture",
            "pred_curvature_capture", "curvature_amp_ratio",
            "curvature_cosine", "sample_optimal_scalar", "valid_rank",
            "saturation",
        ],
    )

    start = evaluate(model, test_loader, cfg, device)
    write_log(
        log_path,
        "E17-b start | signed_curvature_bank=ON | "
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
    best_path = os.path.join(ckpt_dir, "curvature_e17b_best_psnr.pth")
    save_checkpoint(
        model, optimizer, 0, best_psnr, best_path, extra=_checkpoint_extra(cfg)
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
                f"Scalar={last_eval['sample_scalar_oracle_psnr']:.4f} "
                f"PixelScalar={last_eval['pixel_scalar_oracle_psnr']:.4f} | "
                f"Amp={last_eval['pred_curvature_amplitude_ratio']:.3f} "
                f"Cos={last_eval['pred_curvature_cosine']:.3f} "
                f"alpha={last_eval['mean_sample_optimal_scalar']:.3f} | "
                f"PcompCap={100.0*last_eval['pred_pcomp_capture']:.2f}% "
                f"CurvCap={100.0*last_eval['pred_curvature_capture']:.2f}% "
                f"validR={last_eval['mean_valid_curvature_rank']:.2f} "
                f"sat={100.0*last_eval['proposal_saturation']:.2f}%"
            )
            if last_eval["pred_psnr"] > best_psnr:
                best_psnr = last_eval["pred_psnr"]
                best_epoch = epoch
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_psnr,
                    best_path,
                    extra=_checkpoint_extra(cfg),
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
                "sample_scalar_psnr": last_eval["sample_scalar_oracle_psnr"],
                "pixel_scalar_psnr": last_eval["pixel_scalar_oracle_psnr"],
                "pred_pcomp_capture": last_eval["pred_pcomp_capture"],
                "oracle_pcomp_capture": last_eval["oracle_pcomp_capture"],
                "pred_curvature_capture": last_eval["pred_curvature_capture"],
                "curvature_amp_ratio": last_eval[
                    "pred_curvature_amplitude_ratio"
                ],
                "curvature_cosine": last_eval["pred_curvature_cosine"],
                "sample_optimal_scalar": last_eval[
                    "mean_sample_optimal_scalar"
                ],
            })
        logger.write(row)

    summary = {
        "experiment": "E17-b signed projected curvature bank",
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
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"Best E17-b PSNR={best_psnr:.4f} at epoch {best_epoch} | "
        f"Stage2={start['stage2_psnr']:.4f} "
        f"Oracle={start['curvature_oracle_psnr']:.4f}"
    )


if __name__ == "__main__":
    main()
