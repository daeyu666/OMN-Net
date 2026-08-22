"""E30: train half-resolution curvature latent-field recovery.

E30 follows the positive E29 stride-2 oracle.  The predictor never emits an HR
curvature residual directly.  It emits a spatially compressed latent field z at
H/2 x W/2; the only writeback path is

    z -> bilinear upsample -> P_curv -> Delta C.

Two controlled variants share the same latent predictor capacity and loss:
* msi_only (E30-A): predictor input is only PixelUnshuffle(2) HR-MSI.
* fusion   (E30-B): add legal LR-HSI / frozen Stage-2 state descriptors.

Important supervision choice:
E29's least-squares latent z is not used as a regression label because latent
coordinates need not be unique after Upsample + P_curv.  The unique supervised
quantity is the authorized GT residual t_curv itself.  This avoids repeating the
non-unique-support supervision failure seen in earlier experiments.
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
from models.curvature_latent_field import CurvatureLatentFieldNet
from train_local_curvature_extrapolation import build_targets
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
        default=(
            "./checkpoints/local_null_manifold/PaviaU/"
            "local_null_best_psnr.pth"
        ),
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

    p.add_argument(
        "--e30_variant",
        type=str,
        choices=["msi_only", "fusion"],
        default="fusion",
    )
    p.add_argument("--latent_stride", type=int, default=2)
    p.add_argument("--latent_amplitude_multiplier", type=float, default=8.0)
    p.add_argument("--latent_predictor_hidden", type=int, default=96)
    p.add_argument("--latent_predictor_blocks", type=int, default=4)

    p.add_argument("--curvature_rank", type=int, default=6)
    p.add_argument("--curvature_svd_chunk_pixels", type=int, default=2048)
    p.add_argument("--curvature_svd_tolerance", type=float, default=1e-5)
    p.add_argument("--curvature_abs_tolerance", type=float, default=1e-9)
    p.add_argument("--e30_loss_beta", type=float, default=0.25)
    p.add_argument("--e30_grad_clip", type=float, default=1.0)
    p.add_argument("--e30_eval_interval", type=int, default=10)
    p.add_argument(
        "--e29_stride2_reference_psnr",
        type=float,
        default=0.0,
        help=(
            "Optional reporting-only E29 stride-2 oracle PSNR.  It never enters "
            "training or checkpoint selection."
        ),
    )

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    if cfg.latent_stride < 1:
        raise ValueError("latent_stride must be positive")
    if cfg.latent_amplitude_multiplier <= 0:
        raise ValueError("latent_amplitude_multiplier must be positive")
    if cfg.latent_predictor_hidden < 1 or cfg.latent_predictor_blocks < 1:
        raise ValueError("latent predictor size must be positive")
    if cfg.curvature_rank < 1 or cfg.curvature_rank > 8:
        raise ValueError("curvature_rank must be in [1,8]")
    if cfg.curvature_svd_chunk_pixels < 1:
        raise ValueError("curvature_svd_chunk_pixels must be positive")
    if cfg.curvature_svd_tolerance <= 0 or cfg.curvature_abs_tolerance <= 0:
        raise ValueError("curvature SVD tolerances must be positive")
    if cfg.e30_loss_beta <= 0 or cfg.e30_eval_interval < 1:
        raise ValueError("invalid E30 loss/eval settings")
    if cfg.e29_stride2_reference_psnr < 0:
        raise ValueError("E29 reference PSNR cannot be negative")
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

    model = CurvatureLatentFieldNet(
        local_model=local_model,
        variant=cfg.e30_variant,
        latent_stride=cfg.latent_stride,
        curvature_rank=cfg.curvature_rank,
        curvature_svd_chunk_pixels=cfg.curvature_svd_chunk_pixels,
        curvature_svd_tolerance=cfg.curvature_svd_tolerance,
        curvature_abs_tolerance=cfg.curvature_abs_tolerance,
        latent_amplitude_multiplier=cfg.latent_amplitude_multiplier,
        predictor_hidden_channels=cfg.latent_predictor_hidden,
        predictor_blocks=cfg.latent_predictor_blocks,
    ).to(device)
    return model, foundation, local_epoch, local_best


def _normalized_latent_tv(latent: torch.Tensor, scale: torch.Tensor) -> float:
    normalized = latent / scale.view(1, -1, 1, 1)
    values = []
    if normalized.size(-1) > 1:
        values.append(
            (normalized[:, :, :, 1:] - normalized[:, :, :, :-1]).abs().mean()
        )
    if normalized.size(-2) > 1:
        values.append(
            (normalized[:, :, 1:, :] - normalized[:, :, :-1, :]).abs().mean()
        )
    if not values:
        return 0.0
    return float(torch.stack(values).mean().detach().item())


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    loss_meter = AverageMeter()
    valid_rank = AverageMeter()
    saturation = AverageMeter()
    latent_tv = AverageMeter()

    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        targets = build_targets(model, out, batch["gt"])

        scale = out["coefficient_scale"].view(1, -1, 1, 1)
        loss = F.smooth_l1_loss(
            out["curvature_residual"] / scale,
            targets["curvature"] / scale,
            beta=cfg.e30_loss_beta,
        )
        loss.backward()
        if cfg.e30_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), cfg.e30_grad_clip
            )
        optimizer.step()

        n = batch["gt"].size(0)
        loss_meter.update(float(loss.detach().item()), n)
        valid_rank.update(
            out["curvature_valid_mask"].float().sum(dim=1).mean().item(), n
        )
        saturation.update(
            (out["normalized_curvature_latent"].detach().abs() > 0.98)
            .float()
            .mean()
            .item(),
            n,
        )
        latent_tv.update(
            _normalized_latent_tv(
                out["curvature_latent"].detach(),
                out["coefficient_scale"].detach(),
            ),
            n,
        )

    return {
        "loss": loss_meter.avg,
        "valid_rank": valid_rank.avg,
        "saturation": saturation.avg,
        "latent_tv": latent_tv.avg,
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
    pred_curvature_energy = 0.0
    pred_curvature_dot = 0.0
    pred_curvature_error = 0.0
    valid_rank = AverageMeter()
    saturation = AverageMeter()
    latent_tv = AverageMeter()

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

        pcomp64 = targets["pcomp"].double()
        pred64 = pred.double()
        target64 = target.double()
        pcomp_energy += float(pcomp64.square().sum().item())
        pred_pcomp_error += float((pred64 - pcomp64).square().sum().item())
        oracle_pcomp_error += float((target64 - pcomp64).square().sum().item())

        curvature_energy += float(target64.square().sum().item())
        pred_curvature_energy += float(pred64.square().sum().item())
        pred_curvature_dot += float((pred64 * target64).sum().item())
        pred_curvature_error += float((pred64 - target64).square().sum().item())

        n = batch["gt"].size(0)
        valid_rank.update(
            out["curvature_valid_mask"].float().sum(dim=1).mean().item(), n
        )
        saturation.update(
            (out["normalized_curvature_latent"].abs() > 0.98)
            .float()
            .mean()
            .item(),
            n,
        )
        latent_tv.update(
            _normalized_latent_tv(
                out["curvature_latent"], out["coefficient_scale"]
            ),
            n,
        )

    result: Dict[str, float] = {}
    for name, meter in meters.items():
        for key, value in meter.average().items():
            result[f"{name}_{key.lower()}"] = value

    pcomp_energy = max(pcomp_energy, 1e-30)
    curvature_energy = max(curvature_energy, 1e-30)
    result["pred_pcomp_capture"] = 1.0 - pred_pcomp_error / pcomp_energy
    result["oracle_pcomp_capture"] = 1.0 - oracle_pcomp_error / pcomp_energy
    result["pred_curvature_capture"] = 1.0 - pred_curvature_error / curvature_energy
    result["pred_curvature_amplitude_ratio"] = math.sqrt(
        pred_curvature_energy / curvature_energy
    )
    if pred_curvature_energy > 1e-30:
        result["pred_curvature_cosine"] = pred_curvature_dot / math.sqrt(
            pred_curvature_energy * curvature_energy
        )
    else:
        result["pred_curvature_cosine"] = 0.0
    result["mean_valid_curvature_rank"] = valid_rank.avg
    result["latent_saturation"] = saturation.avg
    result["normalized_latent_tv"] = latent_tv.avg

    if cfg.e29_stride2_reference_psnr > 0:
        denom = max(
            cfg.e29_stride2_reference_psnr - result["stage2_psnr"], 1e-12
        )
        result["e29_stride2_gain_realization"] = (
            result["pred_psnr"] - result["stage2_psnr"]
        ) / denom
    else:
        result["e29_stride2_gain_realization"] = 0.0
    return result


def _experiment_name(variant: str) -> str:
    return "E30-A" if variant == "msi_only" else "E30-B"


def _root_name(variant: str) -> str:
    return (
        "curvature_latent_field_e30a"
        if variant == "msi_only"
        else "curvature_latent_field_e30b"
    )


def _checkpoint_extra(cfg):
    name = _experiment_name(cfg.e30_variant)
    return {
        "model_role": "curvature_latent_field",
        "experiment": name,
        "e30_variant": cfg.e30_variant,
        "dataset": cfg.dataset,
        "latent_stride": cfg.latent_stride,
        "curvature_rank": cfg.curvature_rank,
        "local_checkpoint": cfg.local_checkpoint,
        "supervision": (
            "authorized curvature residual; no direct GT latent-coordinate regression"
        ),
        "information_boundary": (
            "latent -> bilinear upsample -> LR-HSI-derived P_curv -> writeback"
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

    experiment = _experiment_name(cfg.e30_variant)
    root = _root_name(cfg.e30_variant)
    ckpt_dir = os.path.join(cfg.checkpoint_root, root, cfg.dataset)
    out_dir = os.path.join(cfg.output_root, root, cfg.dataset)
    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)
    log_path = os.path.join(cfg.log_root, root, f"{cfg.dataset}.log")
    csv_path = os.path.join(cfg.log_root, root, f"{cfg.dataset}.csv")

    logger = CSVLogger(
        csv_path,
        [
            "epoch",
            "lr",
            "loss",
            "psnr",
            "sam",
            "stage2_psnr",
            "curvature_oracle_psnr",
            "full_pcomp_psnr",
            "pred_pcomp_capture",
            "pred_curvature_capture",
            "curvature_amp_ratio",
            "curvature_cosine",
            "valid_rank",
            "latent_saturation",
            "latent_tv",
            "e29_gain_realization",
        ],
    )

    start = evaluate(model, test_loader, cfg, device)
    reference_text = (
        f" E29s2={cfg.e29_stride2_reference_psnr:.4f}"
        if cfg.e29_stride2_reference_psnr > 0
        else ""
    )
    write_log(
        log_path,
        f"{experiment} start | variant={cfg.e30_variant} "
        f"latent_stride={cfg.latent_stride} | "
        f"Stage2 epoch={local_epoch} best={local_best:.4f} | "
        f"Stage2={start['stage2_psnr']:.4f} "
        f"Pred={start['pred_psnr']:.4f} "
        f"CurvOracle={start['curvature_oracle_psnr']:.4f} "
        f"FullPcomp={start['full_pcomp_psnr']:.4f}{reference_text}"
    )

    best_psnr = start["pred_psnr"]
    best_epoch = 0
    best_path = os.path.join(ckpt_dir, "latent_best_psnr.pth")
    save_checkpoint(
        model,
        optimizer,
        0,
        best_psnr,
        best_path,
        extra=_checkpoint_extra(cfg),
    )

    last_eval = start
    for epoch in range(1, cfg.epochs + 1):
        stat = train_epoch(model, train_loader, optimizer, cfg, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        should_eval = epoch % cfg.e30_eval_interval == 0 or epoch == cfg.epochs

        if should_eval:
            last_eval = evaluate(model, test_loader, cfg, device)
            ref = ""
            if cfg.e29_stride2_reference_psnr > 0:
                ref = (
                    f" E29Realize="
                    f"{100.0*last_eval['e29_stride2_gain_realization']:.2f}%"
                )
            write_log(
                log_path,
                f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
                f"loss={stat['loss']:.6f} | "
                f"PSNR={last_eval['pred_psnr']:.4f} "
                f"SAM={last_eval['pred_sam']:.4f} "
                f"Stage2={last_eval['stage2_psnr']:.4f} "
                f"CurvOracle={last_eval['curvature_oracle_psnr']:.4f} | "
                f"Amp={last_eval['pred_curvature_amplitude_ratio']:.3f} "
                f"Cos={last_eval['pred_curvature_cosine']:.3f} "
                f"PcompCap={100.0*last_eval['pred_pcomp_capture']:.2f}% "
                f"CurvCap={100.0*last_eval['pred_curvature_capture']:.2f}% "
                f"validR={last_eval['mean_valid_curvature_rank']:.2f} "
                f"sat={100.0*last_eval['latent_saturation']:.2f}% "
                f"latentTV={last_eval['normalized_latent_tv']:.4f}{ref}"
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
                f"loss={stat['loss']:.6f} "
                f"validR={stat['valid_rank']:.2f} "
                f"sat={100.0*stat['saturation']:.2f}% "
                f"latentTV={stat['latent_tv']:.4f}"
            )

        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": stat["loss"],
            "valid_rank": stat["valid_rank"],
            "latent_saturation": stat["saturation"],
            "latent_tv": stat["latent_tv"],
        }
        if should_eval:
            row.update(
                {
                    "psnr": last_eval["pred_psnr"],
                    "sam": last_eval["pred_sam"],
                    "stage2_psnr": last_eval["stage2_psnr"],
                    "curvature_oracle_psnr": last_eval[
                        "curvature_oracle_psnr"
                    ],
                    "full_pcomp_psnr": last_eval["full_pcomp_psnr"],
                    "pred_pcomp_capture": last_eval["pred_pcomp_capture"],
                    "pred_curvature_capture": last_eval[
                        "pred_curvature_capture"
                    ],
                    "curvature_amp_ratio": last_eval[
                        "pred_curvature_amplitude_ratio"
                    ],
                    "curvature_cosine": last_eval["pred_curvature_cosine"],
                    "e29_gain_realization": last_eval[
                        "e29_stride2_gain_realization"
                    ],
                }
            )
        logger.write(row)

    summary = {
        "experiment": experiment,
        "variant": cfg.e30_variant,
        "best_epoch": int(best_epoch),
        "best_psnr": float(best_psnr),
        "checkpoint": best_path,
        "latent_stride": int(cfg.latent_stride),
        "curvature_rank": int(cfg.curvature_rank),
        "stage2_psnr": float(start["stage2_psnr"]),
        "curvature_oracle_psnr": float(start["curvature_oracle_psnr"]),
        "full_pcomp_psnr": float(start["full_pcomp_psnr"]),
        "e29_stride2_reference_psnr": float(
            cfg.e29_stride2_reference_psnr
        ),
        "last_eval": last_eval,
    }
    with open(
        os.path.join(out_dir, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        f"{experiment} done. Best PSNR={best_psnr:.4f} at epoch {best_epoch}. "
        f"Checkpoint: {best_path}"
    )


if __name__ == "__main__":
    main()
