"""Train OMN-Net innovation point 2 with sparse-simplex LR-HSI arbitration.

Stage-1 and Stage-2 are frozen. The only supervision in this first experiment is
P_comp residual reconstruction. No HSI L1/SAM auxiliary loss, uncertainty gate,
trust radius, prototype consensus, or direction-consensus rule is used.
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
from models import (
    LocalNullManifoldNet,
    SparseSimplexComplementNet,
    build_spectral_response,
    flatten_spatial,
    flatten_tangent,
    load_foundation_checkpoint,
    project_complement_vectors,
    unflatten_spatial,
)
from utils import (
    AverageMeter,
    CSVLogger,
    count_parameters,
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
        default="./checkpoints/local_null_manifold/PaviaU/"
        "local_null_best_psnr.pth",
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

    p.add_argument("--stage3_patch_size", type=int, default=128)
    p.add_argument("--stage3_stride", type=int, default=64)
    p.add_argument("--stage3_batch_size", type=int, default=1)
    p.add_argument("--nonlocal_top_k", type=int, default=690)
    p.add_argument("--nonlocal_exclusion_radius_lr", type=int, default=1)
    p.add_argument("--nonlocal_query_chunk_pixels", type=int, default=64)
    p.add_argument("--stage3_context_channels", type=int, default=64)
    p.add_argument("--stage3_context_blocks", type=int, default=2)
    p.add_argument("--stage3_scorer_hidden", type=int, default=96)
    p.add_argument("--stage3_scorer_blocks", type=int, default=2)
    p.add_argument("--stage3_sparsemax_temperature", type=float, default=1.0)
    p.add_argument("--stage3_loss_beta", type=float, default=0.25)
    p.add_argument("--stage3_grad_clip", type=float, default=1.0)
    p.add_argument("--diagnose_only", action="store_true")

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    # Stage3 must expose at least K=690 LR states. With scale=4, a 128x128 HR
    # patch yields a 32x32 LR-HSI memory containing 1024 observed states.
    cfg.patch_size = cfg.stage3_patch_size
    cfg.stride = cfg.stage3_stride
    cfg.batch_size = cfg.stage3_batch_size
    return cfg


def build_complement_target(
    model: SparseSimplexComplementNet,
    out: Dict[str, torch.Tensor],
    gt: torch.Tensor,
) -> torch.Tensor:
    foundation = model.local_model.foundation
    geometry = model.local_model.geometry
    gt_coefficients = foundation.encode(gt, basis=out["basis"])
    gt_null = geometry.project_null(gt_coefficients)
    missing = gt_null - out["null_seed_coefficients"]
    missing_flat = flatten_spatial(missing)
    tangent_flat = flatten_tangent(out["tangent_basis"])
    targets = []
    for b in range(missing.size(0)):
        targets.append(
            project_complement_vectors(
                missing_flat[b],
                tangent_flat[b],
                geometry.null_projector,
            )
        )
    target_flat = torch.stack(targets, dim=0)
    return unflatten_spatial(target_flat, gt.size(2), gt.size(3))


def compute_loss(
    model: SparseSimplexComplementNet,
    out: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    beta: float,
):
    target = build_complement_target(model, out, gt)
    scale = out["coefficient_scale"].view(1, -1, 1, 1)
    loss = F.smooth_l1_loss(
        out["complement_residual"] / scale,
        target / scale,
        beta=beta,
    )
    return loss, target


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    meters = {
        key: AverageMeter()
        for key in ["loss", "active", "max_weight", "probe_norm"]
    }
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        loss, _ = compute_loss(
            model, out, batch["gt"], cfg.stage3_loss_beta
        )
        loss.backward()
        if cfg.stage3_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), cfg.stage3_grad_clip
            )
        optimizer.step()
        n = batch["lr_hsi"].size(0)
        meters["loss"].update(loss.detach().item(), n)
        meters["active"].update(
            out["active_candidates_mean"].item(), n
        )
        meters["max_weight"].update(out["max_weight_mean"].item(), n)
        meters["probe_norm"].update(out["probe_norm_mean"].item(), n)
    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    metrics = {
        name: MetricAverager()
        for name in ["stage3", "stage2", "stage2_gt_comp"]
    }
    comp_energy = 0.0
    comp_error = 0.0
    diag = {
        key: AverageMeter()
        for key in ["active", "max_weight", "probe_norm"]
    }

    for batch in loader:
        batch = move_to_device(batch, device)
        gt = batch["gt"]
        out = model(batch["lr_hsi"], batch["hr_msi"])
        target = build_complement_target(model, out, gt)
        gt_comp_coefficients = out["stage2_coefficients"] + target
        gt_comp_hsi = model.local_model.foundation.decode(
            gt_comp_coefficients, basis=out["basis"]
        )
        for name, prediction in {
            "stage3": out["reconstructed_hsi"],
            "stage2": out["stage2_hsi"],
            "stage2_gt_comp": gt_comp_hsi,
        }.items():
            metrics[name].update(
                calc_metrics(prediction, gt, cfg.scale_ratio)
            )

        comp_energy += float(target.double().square().sum().item())
        comp_error += float(
            (
                out["complement_residual"].double() - target.double()
            ).square().sum().item()
        )
        n = gt.size(0)
        diag["active"].update(out["active_candidates_mean"].item(), n)
        diag["max_weight"].update(out["max_weight_mean"].item(), n)
        diag["probe_norm"].update(out["probe_norm_mean"].item(), n)

    result = {}
    for prefix, meter in metrics.items():
        for key, value in meter.average().items():
            result[f"{prefix}_{key.lower()}"] = value
    comp_energy = max(comp_energy, 1e-30)
    result["pcomp_capture"] = 1.0 - comp_error / comp_energy
    result["pcomp_rrmse"] = math.sqrt(comp_error / comp_energy)
    for key, meter in diag.items():
        result[key] = meter.avg
    return result


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
    model = SparseSimplexComplementNet(
        local_model=local_model,
        top_k=cfg.nonlocal_top_k,
        exclusion_radius_lr=cfg.nonlocal_exclusion_radius_lr,
        query_chunk_pixels=cfg.nonlocal_query_chunk_pixels,
        context_channels=cfg.stage3_context_channels,
        context_blocks=cfg.stage3_context_blocks,
        scorer_hidden=cfg.stage3_scorer_hidden,
        scorer_blocks=cfg.stage3_scorer_blocks,
        sparsemax_temperature=cfg.stage3_sparsemax_temperature,
    ).to(device)
    return model, local_epoch, local_best


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    lr_memory_h = cfg.stage3_patch_size // cfg.scale_ratio
    lr_memory_count = lr_memory_h * lr_memory_h
    max_excluded = (2 * cfg.nonlocal_exclusion_radius_lr + 1) ** 2
    if lr_memory_count - max_excluded < cfg.nonlocal_top_k:
        raise ValueError(
            f"Stage3 patch gives only {lr_memory_count} LR states; "
            f"K={cfg.nonlocal_top_k} cannot be retained after local exclusion."
        )

    model, local_epoch, local_best = build_model(cfg, info, device)
    trainable = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05
    )

    root = "sparse_simplex_complement"
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
            "stage2_psnr",
            "gt_comp_psnr",
            "capture",
            "active",
            "max_weight",
            "probe_norm",
        ],
    )

    start_epoch = 0
    if cfg.resume:
        start_epoch, _ = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            strict=True,
            map_location=device,
            load_optimizer=True,
        )

    start = evaluate(model, test_loader, cfg, device)
    write_log(
        log_path,
        "Sparse-simplex start | "
        f"Stage2 checkpoint epoch={local_epoch} best={local_best:.4f} | "
        f"trainable={count_parameters(model):.3f}M | "
        f"K={cfg.nonlocal_top_k} patch={cfg.stage3_patch_size} | "
        f"PSNR={start['stage3_psnr']:.4f} "
        f"SAM={start['stage3_sam']:.4f} | "
        f"Stage2={start['stage2_psnr']:.4f} | "
        f"GT P_comp={start['stage2_gt_comp_psnr']:.4f} | "
        f"capture={100.0*start['pcomp_capture']:.2f}% | "
        f"active={start['active']:.2f}",
    )

    if cfg.diagnose_only:
        with open(
            os.path.join(out_dir, "diagnose_only.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(start, handle, indent=2, ensure_ascii=False)
        return

    best_psnr = start["stage3_psnr"]
    best_epoch = start_epoch
    best_path = os.path.join(ckpt_dir, "sparse_simplex_best_psnr.pth")
    save_checkpoint(
        model,
        optimizer,
        start_epoch,
        best_psnr,
        best_path,
        extra={
            "model_role": "sparse_simplex_complement",
            "dataset": cfg.dataset,
            "local_checkpoint": cfg.local_checkpoint,
            "top_k": cfg.nonlocal_top_k,
            "stage3_patch_size": cfg.stage3_patch_size,
            "loss": "P_comp SmoothL1 only",
        },
    )

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        train_stat = train_epoch(model, train_loader, optimizer, cfg, device)
        val = evaluate(model, test_loader, cfg, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        logger.write(
            {
                "epoch": epoch,
                "lr": lr,
                "psnr": val["stage3_psnr"],
                "sam": val["stage3_sam"],
                "stage2_psnr": val["stage2_psnr"],
                "gt_comp_psnr": val["stage2_gt_comp_psnr"],
                "capture": val["pcomp_capture"],
                "active": val["active"],
                "max_weight": val["max_weight"],
                "probe_norm": val["probe_norm"],
            }
        )
        write_log(
            log_path,
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"train_loss={train_stat['loss']:.6f} | "
            f"PSNR={val['stage3_psnr']:.4f} SAM={val['stage3_sam']:.4f} | "
            f"capture={100.0*val['pcomp_capture']:.2f}% | "
            f"active={val['active']:.2f} maxw={val['max_weight']:.4f} | "
            f"probe={val['probe_norm']:.5f}",
        )

        if val["stage3_psnr"] > best_psnr:
            best_psnr = val["stage3_psnr"]
            best_epoch = epoch
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_psnr,
                best_path,
                extra={
                    "model_role": "sparse_simplex_complement",
                    "dataset": cfg.dataset,
                    "local_checkpoint": cfg.local_checkpoint,
                    "top_k": cfg.nonlocal_top_k,
                    "stage3_patch_size": cfg.stage3_patch_size,
                    "loss": "P_comp SmoothL1 only",
                },
            )

    summary = {
        "best_epoch": best_epoch,
        "best_psnr": best_psnr,
        "checkpoint": best_path,
        "local_epoch": int(local_epoch),
        "local_best": float(local_best),
        "top_k": cfg.nonlocal_top_k,
        "patch_size": cfg.stage3_patch_size,
        "trainable_parameters_m": count_parameters(model),
    }
    with open(
        os.path.join(out_dir, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"Best sparse-simplex PSNR={best_psnr:.4f} at epoch {best_epoch}"
    )


if __name__ == "__main__":
    main()
