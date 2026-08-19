"""E13: GT Frank-Wolfe support distillation / candidate-rankability probe.

Purpose
-------
Determine whether the current query/candidate information is sufficient to
identify useful LR-HSI states inside the validated K=690 recall pool.

Stage-1 and Stage-2 are frozen. The student predicts logits over all K real
candidates. A GT-only Frank-Wolfe teacher constructs a sparse simplex target
from the same candidates. Training uses only distribution distillation; there
is no sparsemax, complement reconstruction loss, HSI loss, prototype rule, or
trust-region mechanism.

Success is judged by how much convex-oracle P_comp capacity is retained after
the learned ranking is truncated to Top-32/64/128 candidates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from metrics import MetricAverager, calc_metrics
from models import (
    LocalNullManifoldNet,
    build_spectral_response,
    flatten_spatial,
    flatten_tangent,
    gather_complement_candidates,
    load_foundation_checkpoint,
    project_complement_vectors,
    unflatten_spatial,
)
from models.support_distillation import CandidateSupportRanker
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


def _parse_int_list(text: str) -> List[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or values[0] < 1:
        raise ValueError("rank_top_ms must contain positive integers")
    return values


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

    p.add_argument("--teacher_fw_iterations", type=int, default=30)
    p.add_argument("--distill_temperature", type=float, default=1.0)
    p.add_argument("--rank_top_ms", type=str, default="32,64,128")
    p.add_argument("--rank_eval_interval", type=int, default=10)
    p.add_argument("--rank_grad_clip", type=float, default=1.0)
    p.add_argument(
        "--wide_reference_capture",
        type=float,
        default=0.5253,
        help="Validated K=690 convex capture for retention reporting only.",
    )
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
    if cfg.teacher_fw_iterations < 0:
        raise ValueError("teacher_fw_iterations cannot be negative")
    if cfg.rank_eval_interval < 1:
        raise ValueError("rank_eval_interval must be positive")

    cfg.rank_top_ms = _parse_int_list(cfg.rank_top_ms)
    cfg.patch_size = cfg.rank_patch_size
    cfg.stride = cfg.rank_stride
    cfg.batch_size = cfg.rank_batch_size
    return cfg


def sample_query_indices(
    batch_size: int,
    query_count: int,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    count = min(max(int(sample_count), 1), query_count)
    return torch.stack(
        [
            torch.randperm(query_count, device=device)[:count]
            for _ in range(batch_size)
        ],
        dim=0,
    )


@torch.no_grad()
def build_full_complement_target(
    model: CandidateSupportRanker,
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
    return unflatten_spatial(
        torch.stack(targets, dim=0),
        gt.size(2),
        gt.size(3),
    )


@torch.no_grad()
def build_sampled_complement_target(
    model: CandidateSupportRanker,
    out: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    foundation = model.local_model.foundation
    geometry = model.local_model.geometry
    gt_coefficients = foundation.encode(gt, basis=out["basis"])
    gt_null = geometry.project_null(gt_coefficients)
    missing_flat = flatten_spatial(
        gt_null - out["null_seed_coefficients"]
    )
    tangent_flat = flatten_tangent(out["tangent_basis"])
    rank = missing_flat.size(2)
    dim = tangent_flat.size(3)
    targets = []
    for b in range(missing_flat.size(0)):
        selected_missing = torch.gather(
            missing_flat[b],
            0,
            query_indices[b].unsqueeze(-1).expand(-1, rank),
        )
        selected_tangent = torch.gather(
            tangent_flat[b],
            0,
            query_indices[b].view(-1, 1, 1).expand(-1, rank, dim),
        )
        targets.append(
            project_complement_vectors(
                selected_missing,
                selected_tangent,
                geometry.null_projector,
            )
        )
    return torch.stack(targets, dim=0)


@torch.no_grad()
def frank_wolfe_teacher_weights(
    candidates: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Approximate the GT simplex oracle and return its candidate weights.

    candidates: [N,Q,K,R]
    target: [N,Q,R]
    """
    if candidates.ndim != 4 or target.ndim != 3:
        raise ValueError("invalid teacher tensor shapes")
    if candidates.shape[:2] != target.shape[:2]:
        raise ValueError("teacher query shapes differ")
    if candidates.size(3) != target.size(2):
        raise ValueError("teacher coefficient ranks differ")

    n, q, k, rank = candidates.shape
    c = candidates.reshape(n * q, k, rank)
    t = target.reshape(n * q, rank)
    rows = torch.arange(c.size(0), device=c.device)

    error = (c - t.unsqueeze(1)).square().sum(dim=2)
    first = error.argmin(dim=1)
    weights = c.new_zeros(c.size(0), k)
    weights[rows, first] = 1.0
    current = c[rows, first].clone()

    for _ in range(max(int(iterations), 0)):
        residual = current - t
        linear = torch.einsum("qkr,qr->qk", c, residual)
        vertex_index = linear.argmin(dim=1)
        vertex = c[rows, vertex_index]
        direction = vertex - current
        numerator = -(residual * direction).sum(dim=1)
        denominator = direction.square().sum(dim=1).clamp_min(1e-20)
        gamma = (numerator / denominator).clamp(0.0, 1.0)

        weights = weights * (1.0 - gamma.unsqueeze(1))
        weights[rows, vertex_index] += gamma
        current = current + gamma.unsqueeze(1) * direction

    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights.reshape(n, q, k)


def teacher_mass_at_top_m(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    top_ms: Sequence[int],
) -> Dict[int, float]:
    result = {}
    for m in top_ms:
        mm = min(int(m), logits.size(2))
        order = torch.topk(logits, k=mm, dim=2).indices
        mass = torch.gather(teacher, 2, order).sum(dim=2).mean()
        result[int(m)] = float(mass.detach().item())
    return result


def observable_teacher_mass(
    teacher: torch.Tensor,
    top_ms: Sequence[int],
) -> Dict[int, float]:
    return {
        int(m): float(
            teacher[:, :, : min(int(m), teacher.size(2))]
            .sum(dim=2)
            .mean()
            .detach()
            .item()
        )
        for m in top_ms
    }


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    meters = {
        "loss": AverageMeter(),
        "teacher_active": AverageMeter(),
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
        for m in cfg.rank_top_ms:
            student_mass[m].update(pred_mass[m], n)
            observable_mass[m].update(obs_mass[m], n)

    result = {key: meter.avg for key, meter in meters.items()}
    for m in cfg.rank_top_ms:
        result[f"pred_mass_{m}"] = student_mass[m].avg
        result[f"obs_mass_{m}"] = observable_mass[m].avg
    return result


def _fw_residual(
    candidates: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    error = (candidates - target.unsqueeze(1)).square().sum(dim=2)
    rows = torch.arange(candidates.size(0), device=candidates.device)
    current = candidates[rows, error.argmin(dim=1)].clone()
    for _ in range(max(int(iterations), 0)):
        residual = current - target
        linear = torch.einsum("qkr,qr->qk", candidates, residual)
        vertex = candidates[rows, linear.argmin(dim=1)]
        direction = vertex - current
        numerator = -(residual * direction).sum(dim=1)
        denominator = direction.square().sum(dim=1).clamp_min(1e-20)
        gamma = (numerator / denominator).clamp(0.0, 1.0)
        current = current + gamma.unsqueeze(1) * direction
    return current


@torch.no_grad()
def oracle_residuals_from_ranked_indices(
    model: CandidateSupportRanker,
    out: Dict[str, torch.Tensor],
    target: torch.Tensor,
    ranked_indices: torch.Tensor,
    top_ms: Sequence[int],
    iterations: int,
    chunk_pixels: int,
) -> Dict[int, torch.Tensor]:
    """Evaluate all Top-M prefixes while gathering max-M candidates only once."""
    n, q_count, max_m = ranked_indices.shape
    _, rank, height, width = out["null_seed_coefficients"].shape
    if q_count != height * width:
        raise ValueError("ranked index count does not match HR field")

    geometry = model.local_model.geometry
    memory_null = geometry.project_null(out["lr_coefficients"])
    tangent_residual = out["stage2_coefficients"] - out["anchor_coefficients"]
    local_state = out["null_seed_coefficients"] + tangent_residual
    memory_flat = flatten_spatial(memory_null)
    local_flat = flatten_spatial(local_state)
    tangent_flat = flatten_tangent(out["tangent_basis"])
    target_flat = flatten_spatial(target)

    residuals = {
        int(m): local_state.new_zeros(n, q_count, rank)
        for m in top_ms
    }
    for b in range(n):
        for start in range(0, q_count, chunk_pixels):
            stop = min(start + chunk_pixels, q_count)
            idx = ranked_indices[b, start:stop]
            candidates = gather_complement_candidates(
                memory_flat[b],
                local_flat[b, start:stop],
                tangent_flat[b, start:stop],
                idx,
                geometry.null_projector,
            )
            target_chunk = target_flat[b, start:stop]
            for m in top_ms:
                mm = min(int(m), max_m)
                residuals[int(m)][b, start:stop] = _fw_residual(
                    candidates[:, :mm],
                    target_chunk,
                    iterations,
                )

    return {
        m: unflatten_spatial(value, height, width)
        for m, value in residuals.items()
    }


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
        energy = float(target.double().square().sum().item())
        target_energy += energy
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
    result = {}
    for name, meter in metrics.items():
        for key, value in meter.average().items():
            result[f"{name}_{key.lower()}"] = value
    for name, error in errors.items():
        capture = 1.0 - error / target_energy
        result[f"{name}_capture"] = capture
        result[f"{name}_rrmse"] = math.sqrt(max(error / target_energy, 0.0))
        if cfg.wide_reference_capture > 0:
            result[f"{name}_wide_retention"] = (
                capture / cfg.wide_reference_capture
            )
    return result


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
    model = CandidateSupportRanker(
        local_model=local_model,
        top_k=cfg.nonlocal_top_k,
        exclusion_radius_lr=cfg.nonlocal_exclusion_radius_lr,
        query_chunk_pixels=cfg.nonlocal_query_chunk_pixels,
        context_channels=cfg.rank_context_channels,
        context_blocks=cfg.rank_context_blocks,
        scorer_hidden=cfg.rank_scorer_hidden,
        scorer_blocks=cfg.rank_scorer_blocks,
        sparsemax_temperature=1.0,
    ).to(device)
    return model, local_epoch, local_best


def _format_eval(result: Dict[str, float], top_ms: Sequence[int]) -> str:
    parts = [f"Stage2={result['stage2_psnr']:.4f}"]
    for m in top_ms:
        parts.append(
            f"M{m}: pred={result[f'pred_{m}_psnr']:.4f}/"
            f"{100.0*result[f'pred_{m}_capture']:.2f}% "
            f"obs={result[f'obs_{m}_psnr']:.4f}/"
            f"{100.0*result[f'obs_{m}_capture']:.2f}%"
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
        raise ValueError(
            f"rank patch exposes only {memory_count} LR states; "
            f"K={cfg.nonlocal_top_k} is unavailable after exclusion"
        )
    if max(cfg.rank_top_ms) > cfg.nonlocal_top_k:
        raise ValueError("rank_top_ms cannot exceed nonlocal_top_k")

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

    root = "candidate_support_distillation"
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
        "E13 rankability start | "
        f"Stage2 epoch={local_epoch} best={local_best:.4f} | "
        f"trainable={count_parameters(model):.3f}M | "
        f"K={cfg.nonlocal_top_k} teacherFW={cfg.teacher_fw_iterations} | "
        + _format_eval(start, cfg.rank_top_ms),
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
    best_path = os.path.join(ckpt_dir, "ranker_best_topm_capture.pth")
    save_checkpoint(
        model,
        optimizer,
        start_epoch,
        best_capture,
        best_path,
        extra={
            "model_role": "candidate_support_distillation_ranker",
            "dataset": cfg.dataset,
            "local_checkpoint": cfg.local_checkpoint,
            "top_k": cfg.nonlocal_top_k,
            "primary_m": primary_m,
            "teacher_fw_iterations": cfg.teacher_fw_iterations,
            "supervision": "GT Frank-Wolfe simplex distillation only",
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
                f"teacher_active={train_stat['teacher_active']:.2f} | "
                + _format_eval(last_eval, cfg.rank_top_ms),
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
                        "model_role": "candidate_support_distillation_ranker",
                        "dataset": cfg.dataset,
                        "local_checkpoint": cfg.local_checkpoint,
                        "top_k": cfg.nonlocal_top_k,
                        "primary_m": primary_m,
                        "teacher_fw_iterations": cfg.teacher_fw_iterations,
                        "supervision": "GT Frank-Wolfe simplex distillation only",
                    },
                )
        else:
            write_log(
                log_path,
                f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
                f"loss={train_stat['loss']:.5f} "
                f"teacher_active={train_stat['teacher_active']:.2f} | "
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
        }
        for m in cfg.rank_top_ms:
            row[f"train_pred_mass_{m}"] = train_stat[f"pred_mass_{m}"]
            row[f"train_obs_mass_{m}"] = train_stat[f"obs_mass_{m}"]
            if should_eval:
                row[f"pred_capture_{m}"] = last_eval[f"pred_{m}_capture"]
                row[f"obs_capture_{m}"] = last_eval[f"obs_{m}_capture"]
                row[f"pred_psnr_{m}"] = last_eval[f"pred_{m}_psnr"]
                row[f"obs_psnr_{m}"] = last_eval[f"obs_{m}_psnr"]
        logger.write(row)

    summary = {
        "best_epoch": best_epoch,
        "primary_m": primary_m,
        "best_predicted_capture": best_capture,
        "wide_reference_capture": cfg.wide_reference_capture,
        "best_wide_retention": (
            best_capture / cfg.wide_reference_capture
            if cfg.wide_reference_capture > 0
            else None
        ),
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
        f"Best predicted Top-{primary_m} capture="
        f"{100.0*best_capture:.2f}% at epoch {best_epoch}"
    )


if __name__ == "__main__":
    main()
