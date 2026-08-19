"""E15: no-training degradation-closed global complement transport diagnostic.

This experiment abandons per-pixel candidate identity prediction.  For each HR
query, a simplex over the validated K=690 LR-HSI states is optimized jointly
across the whole HR field.  Candidate values remain LR-HSI-only and the actual
correction is always projected through the query's P_comp operator.

The optimization uses NO GT:

    L = L_LR_close + lambda_g L_MSI_guide + lambda_o L_observable_cost

where L_LR_close uses the same fixed 5x5 Gaussian + bicubic degradation as
Stage-2 training.  GT is used only after/between fixed optimization steps to
report PSNR/SAM and P_comp capture; it never changes the optimization path.

Before optimization the script also measures "closure visibility": how much of
the GT P_comp residual survives the physical LR degradation and whether that
degraded residual aligns with the Stage-2 LR-null closure error.  If visibility
is weak, LR closure cannot identify the missing HR complement even in
principle, which is an important negative result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import parse_args
from data_loader import build_loaders
from metrics import calc_metrics
from models import (
    FixedSpatialDegradation,
    LocalNullManifoldNet,
    build_spectral_response,
    flatten_spatial,
    flatten_tangent,
    load_foundation_checkpoint,
    project_complement_vectors,
    unflatten_spatial,
)
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

    p.add_argument("--diagnostic_image_size", type=int, default=128)
    p.add_argument("--nonlocal_top_k", type=int, default=690)
    p.add_argument("--nonlocal_exclusion_radius_lr", type=int, default=1)
    p.add_argument("--nonlocal_query_chunk_pixels", type=int, default=64)
    p.add_argument("--transport_steps", type=int, default=100)
    p.add_argument("--transport_eval_interval", type=int, default=10)
    p.add_argument("--transport_lr", type=float, default=0.1)
    p.add_argument("--transport_init_temperature", type=float, default=1.0)
    p.add_argument("--transport_lambda_guide", type=float, default=0.1)
    p.add_argument("--transport_lambda_observable", type=float, default=0.05)
    p.add_argument("--transport_grad_clip", type=float, default=5.0)
    p.add_argument("--guide_scale_ratio", type=float, default=1.0)
    p.add_argument("--loss_scale_floor", type=float, default=1e-8)

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"
    if cfg.transport_steps < 1 or cfg.transport_eval_interval < 1:
        raise ValueError("transport steps/eval interval must be positive")
    if cfg.transport_lr <= 0 or cfg.transport_init_temperature <= 0:
        raise ValueError("transport lr/temperature must be positive")
    if cfg.transport_lambda_guide < 0 or cfg.transport_lambda_observable < 0:
        raise ValueError("transport loss weights must be non-negative")
    if cfg.guide_scale_ratio <= 0 or cfg.loss_scale_floor <= 0:
        raise ValueError("guide/loss scales must be positive")
    cfg.image_size = cfg.diagnostic_image_size
    return cfg


def build_local_model(cfg, info, device):
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
    local_epoch, local_best = load_checkpoint(
        model,
        cfg.local_checkpoint,
        optimizer=None,
        strict=True,
        map_location=device,
        load_optimizer=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, foundation, local_epoch, local_best


@torch.no_grad()
def retrieve_all_topk(
    query_observable: torch.Tensor,
    memory_observable: torch.Tensor,
    top_k: int,
    exclusion_radius_lr: int,
    chunk_pixels: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query_observable.size(0) != 1 or memory_observable.size(0) != 1:
        raise ValueError("E15 currently expects batch size 1")
    _, channels, height, width = query_observable.shape
    _, _, memory_h, memory_w = memory_observable.shape
    memory_count = memory_h * memory_w
    max_excluded = (2 * exclusion_radius_lr + 1) ** 2
    actual_k = min(int(top_k), memory_count - max_excluded)
    if actual_k < 1:
        raise ValueError("no candidate remains after local exclusion")

    query = flatten_spatial(query_observable)[0].float()
    memory = flatten_spatial(memory_observable)[0].float()
    mean = memory.mean(dim=0, keepdim=True)
    std = memory.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    query_std = (query - mean) / std
    memory_std = (memory - mean) / std

    q_count = height * width
    memory_linear = torch.arange(memory_count, device=query_observable.device)
    memory_y = torch.div(memory_linear, memory_w, rounding_mode="floor")
    memory_x = memory_linear.remainder(memory_w)
    query_linear = torch.arange(q_count, device=query_observable.device)
    query_y = torch.div(query_linear, width, rounding_mode="floor")
    query_x = query_linear.remainder(width)
    query_lr_y = torch.floor(
        (query_y.float() + 0.5) * memory_h / height
    ).long().clamp_(0, memory_h - 1)
    query_lr_x = torch.floor(
        (query_x.float() + 0.5) * memory_w / width
    ).long().clamp_(0, memory_w - 1)

    idx_chunks = []
    dist_chunks = []
    for start in range(0, q_count, chunk_pixels):
        stop = min(start + chunk_pixels, q_count)
        distances = torch.cdist(query_std[start:stop], memory_std, p=2).square()
        distances = distances / max(channels, 1)
        cy = query_lr_y[start:stop].unsqueeze(1)
        cx = query_lr_x[start:stop].unsqueeze(1)
        local_mask = (
            (memory_y.unsqueeze(0) - cy).abs() <= exclusion_radius_lr
        ) & (
            (memory_x.unsqueeze(0) - cx).abs() <= exclusion_radius_lr
        )
        distances = distances.masked_fill(local_mask, float("inf"))
        finite = torch.isfinite(distances).sum(dim=1)
        if int(finite.min().item()) < actual_k:
            raise RuntimeError("not enough finite candidates")
        top_dist, top_idx = torch.topk(
            distances, k=actual_k, dim=1, largest=False, sorted=True
        )
        idx_chunks.append(top_idx)
        dist_chunks.append(top_dist)

    indices = torch.cat(idx_chunks, dim=0)
    distances = torch.cat(dist_chunks, dim=0)
    centered = distances - distances[:, :1]
    distance_scale = centered.median(dim=1, keepdim=True).values.clamp_min(eps)
    normalized = centered / distance_scale
    return indices, distances, normalized


@torch.no_grad()
def build_guide_weights(hr_msi: torch.Tensor, scale_ratio: float):
    dx = (hr_msi[:, :, :, 1:] - hr_msi[:, :, :, :-1]).square().mean(dim=1)
    dy = (hr_msi[:, :, 1:, :] - hr_msi[:, :, :-1, :]).square().mean(dim=1)
    values = torch.cat([dx.reshape(-1), dy.reshape(-1)], dim=0)
    positive = values[values > 0]
    if positive.numel() > 0:
        scale = positive.median()
    else:
        scale = values.new_tensor(1.0)
    scale = (scale * float(scale_ratio)).clamp_min(1e-12)
    wx = torch.exp(-dx / scale).unsqueeze(1)
    wy = torch.exp(-dy / scale).unsqueeze(1)
    return wx, wy, scale


def transport_forward(
    logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    normalized_distances: torch.Tensor,
    memory_null_flat: torch.Tensor,
    local_flat: torch.Tensor,
    tangent_flat: torch.Tensor,
    null_projector: torch.Tensor,
    local_state: torch.Tensor,
    lr_target_null: torch.Tensor,
    coefficient_scale: torch.Tensor,
    coeff_degradation: FixedSpatialDegradation,
    guide_x: torch.Tensor,
    guide_y: torch.Tensor,
):
    q_count, k = logits.shape
    memory_count = memory_null_flat.size(0)
    weights = torch.softmax(logits, dim=1)

    # P_comp is linear, so projecting the convexly transported LR state is
    # exactly equivalent to convexly mixing all query-specific P_comp residuals.
    full_weights = logits.new_zeros(q_count, memory_count).scatter(
        1, candidate_indices, weights
    )
    transported_memory = full_weights @ memory_null_flat
    delta_flat = project_complement_vectors(
        transported_memory - local_flat,
        tangent_flat,
        null_projector,
    )
    n, _, height, width = local_state.shape
    delta = unflatten_spatial(delta_flat.unsqueeze(0), height, width)
    corrected_null = local_state + delta

    scale = coefficient_scale.view(1, -1, 1, 1)
    degraded = coeff_degradation(
        corrected_null, target_size=lr_target_null.shape[-2:]
    )
    closure = ((degraded - lr_target_null) / scale).square().mean()

    delta_normalized = delta / scale
    dx = delta_normalized[:, :, :, 1:] - delta_normalized[:, :, :, :-1]
    dy = delta_normalized[:, :, 1:, :] - delta_normalized[:, :, :-1, :]
    guide = 0.5 * (
        (guide_x * dx.square()).mean() + (guide_y * dy.square()).mean()
    )
    observable = (weights * normalized_distances).sum(dim=1).mean()
    return {
        "weights": weights,
        "delta": delta,
        "corrected_null": corrected_null,
        "closure": closure,
        "guide": guide,
        "observable": observable,
    }


@torch.no_grad()
def closure_visibility(
    target_comp: torch.Tensor,
    local_state: torch.Tensor,
    lr_target_null: torch.Tensor,
    coefficient_scale: torch.Tensor,
    degradation: FixedSpatialDegradation,
) -> Dict[str, float]:
    scale = coefficient_scale.view(1, -1, 1, 1)
    stage2_degraded = degradation(
        local_state, target_size=lr_target_null.shape[-2:]
    )
    degraded_gt = degradation(
        target_comp, target_size=lr_target_null.shape[-2:]
    )
    closure_residual = (lr_target_null - stage2_degraded) / scale
    degraded_gt_n = degraded_gt / scale

    e_stage2 = closure_residual.square().mean()
    e_with_gt = (
        (stage2_degraded + degraded_gt - lr_target_null) / scale
    ).square().mean()
    high_energy = (target_comp / scale).square().mean().clamp_min(1e-30)
    low_energy = degraded_gt_n.square().mean()
    dot = (closure_residual * degraded_gt_n).sum()
    cosine = dot / (
        closure_residual.square().sum().sqrt()
        * degraded_gt_n.square().sum().sqrt()
    ).clamp_min(1e-30)
    improvement = (e_stage2 - e_with_gt) / e_stage2.clamp_min(1e-30)
    return {
        "stage2_lr_closure_mse": float(e_stage2.item()),
        "gt_comp_lr_closure_mse": float(e_with_gt.item()),
        "gt_comp_closure_improvement": float(improvement.item()),
        "gt_comp_degradation_energy_ratio": float((low_energy / high_energy).item()),
        "gt_comp_closure_alignment_cosine": float(cosine.item()),
    }


@torch.no_grad()
def evaluate_state(
    model: LocalNullManifoldNet,
    out: Dict[str, torch.Tensor],
    state: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    target_comp: torch.Tensor,
    scale_ratio: int,
) -> Dict[str, float]:
    prediction = model.foundation.decode(
        out["stage2_coefficients"] + state["delta"],
        basis=out["basis"],
    )
    metrics = calc_metrics(prediction, gt, scale_ratio)
    stage2_metrics = calc_metrics(out["stage2_hsi"], gt, scale_ratio)
    target_energy = target_comp.double().square().sum().item()
    error = (state["delta"].double() - target_comp.double()).square().sum().item()
    target_energy = max(float(target_energy), 1e-30)
    weights = state["weights"]
    effective = 1.0 / weights.square().sum(dim=1).clamp_min(1e-12)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
    return {
        "psnr": float(metrics["PSNR"]),
        "sam": float(metrics["SAM"]),
        "rmse": float(metrics["RMSE"]),
        "stage2_psnr": float(stage2_metrics["PSNR"]),
        "stage2_sam": float(stage2_metrics["SAM"]),
        "pcomp_capture": float(1.0 - error / target_energy),
        "closure": float(state["closure"].item()),
        "guide": float(state["guide"].item()),
        "observable": float(state["observable"].item()),
        "effective_support": float(effective.mean().item()),
        "max_weight": float(weights.max(dim=1).values.mean().item()),
        "entropy": float(entropy.mean().item()),
    }


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    model, foundation, local_epoch, local_best = build_local_model(
        cfg, info, device
    )
    coeff_degradation = FixedSpatialDegradation(foundation.basis_rank).to(device)

    all_reports = []
    for batch_index, batch in enumerate(test_loader):
        batch = move_to_device(batch, device)
        if batch["gt"].size(0) != 1:
            raise ValueError("E15 currently requires diagnostic batch size 1")
        with torch.no_grad():
            out = model(batch["lr_hsi"], batch["hr_msi"])
            geometry = model.geometry
            reduced = geometry.reduced_response.to(out["lr_coefficients"])
            query_observable = torch.einsum(
                "mr,nrhw->nmhw", reduced, out["anchor_coefficients"]
            )
            memory_observable = torch.einsum(
                "mr,nrhw->nmhw", reduced, out["lr_coefficients"]
            )
            candidate_indices, _, normalized_distances = retrieve_all_topk(
                query_observable,
                memory_observable,
                cfg.nonlocal_top_k,
                cfg.nonlocal_exclusion_radius_lr,
                cfg.nonlocal_query_chunk_pixels,
            )

            memory_null = geometry.project_null(out["lr_coefficients"])
            local_state = out["null_seed_coefficients"] + out["tangent_residual"]
            lr_target_null = geometry.project_null(out["lr_coefficients"])
            memory_null_flat = flatten_spatial(memory_null)[0]
            local_flat = flatten_spatial(local_state)[0]
            tangent_flat = flatten_tangent(out["tangent_basis"])[0]

            gt_coeff = foundation.encode(batch["gt"], basis=out["basis"])
            gt_null = geometry.project_null(gt_coeff)
            missing = gt_null - out["null_seed_coefficients"]
            target_comp_flat = project_complement_vectors(
                flatten_spatial(missing)[0],
                tangent_flat,
                geometry.null_projector,
            )
            _, _, height, width = local_state.shape
            target_comp = unflatten_spatial(
                target_comp_flat.unsqueeze(0), height, width
            )
            visibility = closure_visibility(
                target_comp,
                local_state,
                lr_target_null,
                out["coefficient_scale"],
                coeff_degradation,
            )
            guide_x, guide_y, guide_scale = build_guide_weights(
                batch["hr_msi"], cfg.guide_scale_ratio
            )

        init_logits = (
            -normalized_distances / float(cfg.transport_init_temperature)
        ).to(out["stage2_coefficients"].dtype)
        logits = torch.nn.Parameter(init_logits.clone())
        optimizer = torch.optim.Adam([logits], lr=cfg.transport_lr)

        with torch.no_grad():
            initial_state = transport_forward(
                logits,
                candidate_indices,
                normalized_distances,
                memory_null_flat,
                local_flat,
                tangent_flat,
                geometry.null_projector,
                local_state,
                lr_target_null,
                out["coefficient_scale"],
                coeff_degradation,
                guide_x,
                guide_y,
            )
            term_scales = {
                "closure": max(
                    float(initial_state["closure"].item()), cfg.loss_scale_floor
                ),
                "guide": max(
                    float(initial_state["guide"].item()), cfg.loss_scale_floor
                ),
                "observable": max(
                    float(initial_state["observable"].item()), cfg.loss_scale_floor
                ),
            }
            history = [
                {
                    "step": 0,
                    **evaluate_state(
                        model,
                        out,
                        initial_state,
                        batch["gt"],
                        target_comp,
                        cfg.scale_ratio,
                    ),
                }
            ]

        for step in range(1, cfg.transport_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            state = transport_forward(
                logits,
                candidate_indices,
                normalized_distances,
                memory_null_flat,
                local_flat,
                tangent_flat,
                geometry.null_projector,
                local_state,
                lr_target_null,
                out["coefficient_scale"],
                coeff_degradation,
                guide_x,
                guide_y,
            )
            loss = (
                state["closure"] / term_scales["closure"]
                + cfg.transport_lambda_guide
                * state["guide"] / term_scales["guide"]
                + cfg.transport_lambda_observable
                * state["observable"] / term_scales["observable"]
            )
            loss.backward()
            if cfg.transport_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([logits], cfg.transport_grad_clip)
            optimizer.step()
            with torch.no_grad():
                # Softmax is shift invariant; centering removes the free gauge.
                logits.sub_(logits.mean(dim=1, keepdim=True))

            if step % cfg.transport_eval_interval == 0 or step == cfg.transport_steps:
                with torch.no_grad():
                    eval_state = transport_forward(
                        logits,
                        candidate_indices,
                        normalized_distances,
                        memory_null_flat,
                        local_flat,
                        tangent_flat,
                        geometry.null_projector,
                        local_state,
                        lr_target_null,
                        out["coefficient_scale"],
                        coeff_degradation,
                        guide_x,
                        guide_y,
                    )
                    record = {
                        "step": step,
                        "optimization_loss": float(loss.detach().item()),
                        **evaluate_state(
                            model,
                            out,
                            eval_state,
                            batch["gt"],
                            target_comp,
                            cfg.scale_ratio,
                        ),
                    }
                    history.append(record)
                    print(
                        f"batch={batch_index} step={step:03d} "
                        f"PSNR={record['psnr']:.4f} "
                        f"Stage2={record['stage2_psnr']:.4f} "
                        f"capture={100.0*record['pcomp_capture']:.2f}% "
                        f"close={record['closure']:.6e} "
                        f"Neff={record['effective_support']:.1f} "
                        f"maxw={record['max_weight']:.4f}"
                    )

        all_reports.append(
            {
                "batch_index": batch_index,
                "candidate_k": int(candidate_indices.size(1)),
                "memory_states": int(memory_null_flat.size(0)),
                "guide_scale": float(guide_scale.item()),
                "term_scales": term_scales,
                "closure_visibility": visibility,
                "history": history,
            }
        )

    summary = {
        "local_checkpoint_epoch": int(local_epoch),
        "local_checkpoint_best": float(local_best),
        "transport_steps": int(cfg.transport_steps),
        "transport_lr": float(cfg.transport_lr),
        "lambda_guide": float(cfg.transport_lambda_guide),
        "lambda_observable": float(cfg.transport_lambda_observable),
        "note": "GT is evaluation-only; optimization uses LR closure, MSI guide and observable cost.",
        "batches": all_reports,
    }
    out_dir = os.path.join(cfg.output_root, "diagnostics", cfg.dataset)
    ensure_dir(out_dir)
    output_path = os.path.join(
        out_dir, "degradation_closed_global_transport.json"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("=== E15 closure visibility ===")
    for report in all_reports:
        v = report["closure_visibility"]
        print(
            f"batch={report['batch_index']} "
            f"visibility={v['gt_comp_degradation_energy_ratio']:.6e} "
            f"alignment={v['gt_comp_closure_alignment_cosine']:.4f} "
            f"closure_improvement={100.0*v['gt_comp_closure_improvement']:.2f}%"
        )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
