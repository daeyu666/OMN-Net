"""E13-v2: pure learned-logit GT-support distillation rankability probe.

This experiment removes the fixed negative observable-distance term from the
final candidate logits. Observable distance remains an input feature, but the
student must learn the ranking itself. The purpose is to distinguish a real
feature-identifiability failure from the original E13's fixed-prior domination.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.support_distillation_v2 import PureLearnedSupportRanker
from train_candidate_support_distillation import (
    _has_option,
    _parse_int_list,
    build_full_complement_target,
    build_sampled_complement_target,
    frank_wolfe_teacher_weights,
    observable_teacher_mass,
    oracle_residuals_from_ranked_indices,
    sample_query_indices,
    teacher_mass_at_top_m,
)
from metrics import MetricAverager, calc_metrics
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
    model = PureLearnedSupportRanker(
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
    ).to(device)
    return model, local_epoch, local_best


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    meters = {
        "loss": AverageMeter(),
        "teacher_active": AverageMeter(),
        "logit_std": AverageMeter(),
    }
    student_mass = {m: AverageMeter() for m in cfg.rank_top_ms}
    observable_mass = {m: AverageMeter() for m in cfg.rank_top_ms}

    for batch in loader:
        batch = move_to_device(batch, device)
        n = batch["gt"].size(0)
        q_count = batch["gt"].size(2) * batch["gt"].size(3)
        query_indices = sample_query_indices(
            n,
            q_count,
            cfg.rank_train_query_pixels,
            device,
        )

        optimizer.zero_grad(set_to_none=True)
        out = model.score_queries(
            batch["lr_hsi"],
            batch["hr_msi"],
            query_indices=query_indices,
            return_candidate_details=True,
        )
        target = build_sampled_complement_target(
            model,
            out,
            batch["gt"],
            query_indices,
        )
        teacher = frank_wolfe_teacher_weights(
            out["candidate_residuals_flat"],
            target,
            cfg.teacher_fw_iterations,
        )
        log_probability = F.log_softmax(
            out["candidate_logits_flat"] / cfg.distill_temperature,
            dim=2,
        )
        loss = -(teacher * log_probability).sum(dim=2).mean()
        loss.backward()
        if cfg.rank_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(),
                cfg.rank_grad_clip,
            )
        optimizer.step()

        active = (teacher > 1e-6).to(teacher.dtype).sum(dim=2).mean()
        pred_mass = teacher_mass_at_top_m(
            out["candidate_logits_flat"].detach(),
            teacher,
            cfg.rank_top_ms,
        )
        obs_mass = observable_teacher_mass(teacher, cfg.rank_top_ms)
        meters["loss"].update(loss.detach().item(), n)
        meters["teacher_active"].update(active.detach().item(), n)
        meters["logit_std"].update(out["learned_logit_std"].item(), n)
        for m in cfg.rank_top_ms:
            student_mass[m].update(pred_mass[m], n)
            observable_mass[m].update(obs_mass[m], n)

    result = {key: meter.avg for key, meter in meters.items()}
    for m in cfg.rank_top_ms:
        result[f"pred_mass_{m}"] = student_mass[m].avg
        result[f"obs_mass_{m}"] = observable_mass[m].avg
    return result


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    max_m = max(cfg.rank_top_ms)
    metrics = {"stage2": MetricAverager()}
    for prefix in ["pred", "obs"]:
        for m in cfg.rank_top_ms:
            metrics[f"{prefix}_{m}"] = MetricAverager()

    target_energy = 0.0
    errors = {
        f"{prefix}_{m}": 0.0
        for prefix in ["pred", "obs"]
        for m in cfg.rank_top_ms
    }
    outside = {m: AverageMeter() for m in cfg.rank_top_ms}
    top1_changed = AverageMeter()
    logit_std = AverageMeter()

    for batch in loader:
        batch = move_to_device(batch, device)
        gt = batch["gt"]
        out = model.score_queries(
            batch["lr_hsi"],
            batch["hr_msi"],
            rank_top_m=max_m,
        )
        target = build_full_complement_target(model, out, gt)
        pred_residuals = oracle_residuals_from_ranked_indices(
            model,
            out,
            target,
            out["ranked_candidate_indices_flat"],
            cfg.rank_top_ms,
            cfg.teacher_fw_iterations,
            cfg.nonlocal_query_chunk_pixels,
        )
        obs_residuals = oracle_residuals_from_ranked_indices(
            model,
            out,
            target,
            out["observable_candidate_indices_flat"],
            cfg.rank_top_ms,
            cfg.teacher_fw_iterations,
            cfg.nonlocal_query_chunk_pixels,
        )

        metrics["stage2"].update(
            calc_metrics(out["stage2_hsi"], gt, cfg.scale_ratio)
        )
        target_energy += float(target.double().square().sum().item())
        positions = out["ranked_candidate_positions_flat"]
        top1_changed.update(
            (positions[:, :, 0] != 0).float().mean().item(),
            gt.size(0),
        )
        logit_std.update(out["learned_logit_std"].item(), gt.size(0))
        for m in cfg.rank_top_ms:
            mm = min(int(m), positions.size(2))
            outside[m].update(
                (positions[:, :, :mm] >= mm).float().mean().item(),
                gt.size(0),
            )

        for prefix, residual_dict in [
            ("pred", pred_residuals),
            ("obs", obs_residuals),
        ]:
            for m, residual in residual_dict.items():
                prediction = model.local_model.foundation.decode(
                    out["stage2_coefficients"] + residual,
                    basis=out["basis"],
                )
                metrics[f"{prefix}_{m}"].update(
                    calc_metrics(prediction, gt, cfg.scale_ratio)
                )
                errors[f"{prefix}_{m}"] += float(
                    (residual.double() - target.double()).square().sum().item()
                )

    target_energy = max(target_energy, 1e-30)
    result: Dict[str, float] = {
        "top1_changed": top1_changed.avg,
        "logit_std": logit_std.avg,
    }
    for m in cfg.rank_top_ms:
        result[f"outside_rate_{m}"] = outside[m].avg
    for name, meter in metrics.items():
        for key, value in meter.average().items():
            result[f"{name}_{key.lower()}"] = value
    for name, error in errors.items():
        capture = 1.0 - error / target_energy
        result[f"{name}_capture"] = capture
        if cfg.wide_reference_capture > 0:
            result[f"{name}_wide_retention"] = capture / cfg.wide_reference_capture
    return result


def format_eval(result: Dict[str, float], top_ms: Sequence[int]) -> str:
    parts = [
        f"Stage2={result['stage2_psnr']:.4f}",
        f"logit_std={result['logit_std']:.4f}",
        f"top1_changed={100.0*result['top1_changed']:.1f}%",
    ]
    for m in top_ms:
        parts.append(
            f"M{m}: pred={result[f'pred_{m}_psnr']:.4f}/"
            f"{100.0*result[f'pred_{m}_capture']:.2f}% "
            f"obs={result[f'obs_{m}_psnr']:.4f}/"
            f"{100.0*result[f'obs_{m}_capture']:.2f}% "
            f"outside={100.0*result[f'outside_rate_{m}']:.1f}%"
        )
    return " | ".join(parts)


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

    root = "candidate_support_distillation_v2"
    ckpt_dir = os.path.join(cfg.checkpoint_root, root, cfg.dataset)
    out_dir = os.path.join(cfg.output_root, root, cfg.dataset)
    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)
    log_path = os.path.join(cfg.log_root, root, f"{cfg.dataset}.log")

    csv_fields = [
        "epoch", "lr", "loss", "teacher_active", "logit_std",
        "top1_changed",
    ]
    for m in cfg.rank_top_ms:
        csv_fields.extend([
            f"train_pred_mass_{m}", f"train_obs_mass_{m}",
            f"pred_capture_{m}", f"obs_capture_{m}",
            f"pred_psnr_{m}", f"obs_psnr_{m}", f"outside_rate_{m}",
        ])
    logger = CSVLogger(
        os.path.join(cfg.log_root, root, f"{cfg.dataset}.csv"),
        csv_fields,
    )

    start = evaluate(model, test_loader, cfg, device)
    write_log(
        log_path,
        "E13-v2 start | "
        f"Stage2 epoch={local_epoch} best={local_best:.4f} | "
        f"trainable={count_parameters(model):.3f}M K={cfg.nonlocal_top_k} | "
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
    best_epoch = 0
    best_path = os.path.join(ckpt_dir, "ranker_v2_best_topm_capture.pth")
    save_checkpoint(
        model, optimizer, 0, best_capture, best_path,
        extra={
            "model_role": "pure_learned_support_distillation_ranker",
            "top_k": cfg.nonlocal_top_k,
            "primary_m": primary_m,
            "teacher_fw_iterations": cfg.teacher_fw_iterations,
        },
    )

    last_eval = start
    for epoch in range(1, cfg.epochs + 1):
        train_stat = train_epoch(model, train_loader, optimizer, cfg, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        should_eval = epoch % cfg.rank_eval_interval == 0 or epoch == cfg.epochs

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
                    model, optimizer, epoch, best_capture, best_path,
                    extra={
                        "model_role": "pure_learned_support_distillation_ranker",
                        "top_k": cfg.nonlocal_top_k,
                        "primary_m": primary_m,
                        "teacher_fw_iterations": cfg.teacher_fw_iterations,
                    },
                )
        else:
            write_log(
                log_path,
                f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
                f"loss={train_stat['loss']:.5f} "
                f"teacher_active={train_stat['teacher_active']:.2f} "
                f"logit_std={train_stat['logit_std']:.4f} | "
                + " ".join([
                    f"mass{m}={100.0*train_stat[f'pred_mass_{m}']:.1f}%/"
                    f"{100.0*train_stat[f'obs_mass_{m}']:.1f}%"
                    for m in cfg.rank_top_ms
                ]),
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
        f"Best E13-v2 Top-{primary_m} capture="
        f"{100.0*best_capture:.2f}% at epoch {best_epoch}"
    )


if __name__ == "__main__":
    main()
