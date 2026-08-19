"""E14: LR-HSI memory-spatial-context candidate-rankability probe.

Relative to E13-v2 this experiment changes one variable only: every recalled
candidate receives a learned descriptor of its 5x5 LR-HSI coefficient
neighborhood. The descriptor is selection-only. Candidate residual values,
observable K=690 recall, query encoder, GT Frank-Wolfe teacher, and Top-M convex
oracle evaluation are otherwise unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch

from config import parse_args
from data_loader import build_loaders
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.memory_context_distillation import MemoryContextSupportRanker
from train_candidate_support_distillation import _has_option, _parse_int_list
from train_candidate_support_distillation_v2 import (
    evaluate,
    format_eval,
    train_epoch,
)
from utils import (
    CSVLogger,
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_log,
)


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

    p.add_argument("--rank_patch_size", type=int, default=128)
    p.add_argument("--rank_stride", type=int, default=64)
    p.add_argument("--rank_batch_size", type=int, default=1)
    p.add_argument("--rank_train_query_pixels", type=int, default=256)
    p.add_argument("--nonlocal_top_k", type=int, default=690)
    p.add_argument("--nonlocal_exclusion_radius_lr", type=int, default=1)
    p.add_argument("--nonlocal_query_chunk_pixels", type=int, default=64)
    p.add_argument("--rank_context_channels", type=int, default=64)
    p.add_argument("--rank_context_blocks", type=int, default=2)
    p.add_argument("--rank_scorer_hidden", type=int, default=96)
    p.add_argument("--rank_scorer_blocks", type=int, default=2)
    p.add_argument("--rank_scorer_init_std", type=float, default=1e-3)

    # E14's only new model capacity.
    p.add_argument("--memory_context_hidden", type=int, default=64)
    p.add_argument("--memory_context_channels", type=int, default=64)

    p.add_argument("--teacher_fw_iterations", type=int, default=30)
    p.add_argument("--distill_temperature", type=float, default=1.0)
    p.add_argument("--rank_top_ms", type=str, default="32,64,128")
    p.add_argument("--rank_eval_interval", type=int, default=10)
    p.add_argument("--rank_grad_clip", type=float, default=1.0)
    p.add_argument("--wide_reference_capture", type=float, default=0.5253)
    p.add_argument("--diagnose_only", action="store_true")

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    if cfg.distill_temperature <= 0:
        raise ValueError("distill_temperature must be positive")
    if cfg.rank_scorer_init_std <= 0:
        raise ValueError("rank_scorer_init_std must be positive")
    if cfg.memory_context_hidden < 1 or cfg.memory_context_channels < 1:
        raise ValueError("memory context channels must be positive")
    if cfg.teacher_fw_iterations < 0 or cfg.rank_eval_interval < 1:
        raise ValueError("invalid teacher/eval settings")

    cfg.rank_top_ms = _parse_int_list(cfg.rank_top_ms)
    cfg.patch_size = cfg.rank_patch_size
    cfg.stride = cfg.rank_stride
    cfg.batch_size = cfg.rank_batch_size
    return cfg


def build_model(cfg, info, device):
    foundation, _ = load_foundation_checkpoint(
        cfg.foundation_checkpoint,
        info["n_bands"],
        device,
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

    model = MemoryContextSupportRanker(
        local_model=local_model,
        top_k=cfg.nonlocal_top_k,
        exclusion_radius_lr=cfg.nonlocal_exclusion_radius_lr,
        query_chunk_pixels=cfg.nonlocal_query_chunk_pixels,
        context_channels=cfg.rank_context_channels,
        context_blocks=cfg.rank_context_blocks,
        scorer_hidden=cfg.rank_scorer_hidden,
        scorer_blocks=cfg.rank_scorer_blocks,
        sparsemax_temperature=1.0,
        scorer_init_std=cfg.rank_scorer_init_std,
        memory_hidden_channels=cfg.memory_context_hidden,
        memory_context_channels=cfg.memory_context_channels,
    ).to(device)
    return model, local_epoch, local_best


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    lr_h = cfg.rank_patch_size // cfg.scale_ratio
    memory_count = lr_h * lr_h
    max_excluded = (2 * cfg.nonlocal_exclusion_radius_lr + 1) ** 2
    if memory_count - max_excluded < cfg.nonlocal_top_k:
        raise ValueError("rank patch does not expose enough LR states")
    if max(cfg.rank_top_ms) > cfg.nonlocal_top_k:
        raise ValueError("rank_top_ms cannot exceed K")

    model, local_epoch, local_best = build_model(cfg, info, device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=cfg.lr * 0.05,
    )

    root = "candidate_memory_context_distillation"
    ckpt_dir = os.path.join(cfg.checkpoint_root, root, cfg.dataset)
    out_dir = os.path.join(cfg.output_root, root, cfg.dataset)
    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)
    log_path = os.path.join(cfg.log_root, root, f"{cfg.dataset}.log")

    csv_fields = [
        "epoch",
        "lr",
        "loss",
        "teacher_active",
        "logit_std",
        "top1_changed",
    ]
    for m in cfg.rank_top_ms:
        csv_fields.extend(
            [
                f"train_pred_mass_{m}",
                f"train_obs_mass_{m}",
                f"pred_capture_{m}",
                f"obs_capture_{m}",
                f"pred_psnr_{m}",
                f"obs_psnr_{m}",
                f"outside_rate_{m}",
            ]
        )
    logger = CSVLogger(
        os.path.join(cfg.log_root, root, f"{cfg.dataset}.csv"),
        csv_fields,
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
        "E14 memory-context start | "
        f"Stage2 epoch={local_epoch} best={local_best:.4f} | "
        f"trainable={count_parameters(model):.3f}M K={cfg.nonlocal_top_k} | "
        f"memoryRF=5x5 memoryC={cfg.memory_context_channels} | "
        + format_eval(start, cfg.rank_top_ms),
    )

    if cfg.diagnose_only:
        with open(
            os.path.join(out_dir, "diagnose_only.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(start, handle, indent=2, ensure_ascii=False)
        return

    primary_m = 64 if 64 in cfg.rank_top_ms else cfg.rank_top_ms[0]
    best_capture = start[f"pred_{primary_m}_capture"]
    best_epoch = start_epoch
    best_path = os.path.join(
        ckpt_dir,
        "memory_context_ranker_best_topm_capture.pth",
    )
    save_checkpoint(
        model,
        optimizer,
        start_epoch,
        best_capture,
        best_path,
        extra={
            "model_role": "memory_context_support_distillation_ranker",
            "top_k": cfg.nonlocal_top_k,
            "primary_m": primary_m,
            "teacher_fw_iterations": cfg.teacher_fw_iterations,
            "memory_receptive_field": "5x5 LR",
            "memory_context_channels": cfg.memory_context_channels,
            "selection_only": True,
        },
    )

    last_eval = start
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        train_stat = train_epoch(
            model,
            train_loader,
            optimizer,
            cfg,
            device,
        )
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        should_eval = (
            epoch % cfg.rank_eval_interval == 0 or epoch == cfg.epochs
        )

        if should_eval:
            last_eval = evaluate(model, test_loader, cfg, device)
            write_log(
                log_path,
                f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
                f"loss={train_stat['loss']:.5f} "
                f"teacher_active={train_stat['teacher_active']:.2f} "
                f"train_logit_std={train_stat['logit_std']:.4f} | "
                + format_eval(last_eval, cfg.rank_top_ms),
            )
            current = last_eval[f"pred_{primary_m}_capture"]
            if current > best_capture:
                best_capture = current
                best_epoch = epoch
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_capture,
                    best_path,
                    extra={
                        "model_role": "memory_context_support_distillation_ranker",
                        "top_k": cfg.nonlocal_top_k,
                        "primary_m": primary_m,
                        "teacher_fw_iterations": cfg.teacher_fw_iterations,
                        "memory_receptive_field": "5x5 LR",
                        "memory_context_channels": cfg.memory_context_channels,
                        "selection_only": True,
                    },
                )
        else:
            write_log(
                log_path,
                f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
                f"loss={train_stat['loss']:.5f} "
                f"teacher_active={train_stat['teacher_active']:.2f} "
                f"logit_std={train_stat['logit_std']:.4f} | "
                + " ".join(
                    [
                        f"mass{m}={100.0*train_stat[f'pred_mass_{m}']:.1f}%/"
                        f"{100.0*train_stat[f'obs_mass_{m}']:.1f}%"
                        for m in cfg.rank_top_ms
                    ]
                ),
            )

        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": train_stat["loss"],
            "teacher_active": train_stat["teacher_active"],
            "logit_std": train_stat["logit_std"],
        }
        if should_eval:
            row["top1_changed"] = last_eval["top1_changed"]
        for m in cfg.rank_top_ms:
            row[f"train_pred_mass_{m}"] = train_stat[f"pred_mass_{m}"]
            row[f"train_obs_mass_{m}"] = train_stat[f"obs_mass_{m}"]
            if should_eval:
                row[f"pred_capture_{m}"] = last_eval[f"pred_{m}_capture"]
                row[f"obs_capture_{m}"] = last_eval[f"obs_{m}_capture"]
                row[f"pred_psnr_{m}"] = last_eval[f"pred_{m}_psnr"]
                row[f"obs_psnr_{m}"] = last_eval[f"obs_{m}_psnr"]
                row[f"outside_rate_{m}"] = last_eval[f"outside_rate_{m}"]
        logger.write(row)

    summary = {
        "best_epoch": best_epoch,
        "primary_m": primary_m,
        "best_predicted_capture": best_capture,
        "wide_reference_capture": cfg.wide_reference_capture,
        "memory_context": {
            "receptive_field_lr": "5x5",
            "hidden_channels": cfg.memory_context_hidden,
            "output_channels": cfg.memory_context_channels,
            "selection_only": True,
        },
        "checkpoint": best_path,
        "last_eval": last_eval,
    }
    with open(
        os.path.join(out_dir, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        f"Best E14 Top-{primary_m} capture="
        f"{100.0*best_capture:.2f}% at epoch {best_epoch}"
    )


if __name__ == "__main__":
    main()
