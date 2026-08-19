"""No-training oracle for OMN-Net innovation point 2.

The diagnostic asks four progressively stricter questions about the tangent
complement P_comp:

1. How much GT missing-null energy remains outside the local tangent space?
2. Can any *non-local LR-HSI null state* represent that complement residual?
3. Can an MSI-observable retrieval key place useful states inside a top-K list?
4. Without GT selection, does deterministic soft aggregation already help the
   validated Stage-2 reconstruction?

GT is used only to construct evaluation targets and oracle choices. Retrieval
keys and memory values are built entirely from the observed LR-HSI / HR-MSI
inputs and the frozen Stage-1/Stage-2 geometry.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import parse_args
from data_loader import build_loaders
from metrics import MetricAverager, calc_metrics
from models import (
    LocalNullManifoldNet,
    ObservableKeyedComplementMemory,
    build_spectral_response,
    gather_complement_candidates,
    load_foundation_checkpoint,
    project_complement_vectors,
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

    p.add_argument("--nonlocal_top_k", type=int, default=32)
    p.add_argument("--nonlocal_exclusion_radius_lr", type=int, default=1)
    p.add_argument("--nonlocal_query_chunk_pixels", type=int, default=128)
    p.add_argument("--nonlocal_temperature_ratio", type=float, default=1.0)
    p.add_argument("--nonlocal_convex_iterations", type=int, default=30)
    p.add_argument("--global_convex_iterations", type=int, default=20)

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    if cfg.dataset != "PaviaU":
        if cfg.foundation_checkpoint.endswith(
            "checkpoints/RAPD-Net/basis_for_stage2.pth"
        ):
            cfg.foundation_checkpoint = os.path.join(
                cfg.checkpoint_root,
                "spectral_foundation",
                cfg.dataset,
                "foundation_for_local_null.pth",
            )
        if "local_null_manifold/PaviaU" in cfg.local_checkpoint:
            cfg.local_checkpoint = os.path.join(
                cfg.checkpoint_root,
                "local_null_manifold",
                cfg.dataset,
                "local_null_best_psnr.pth",
            )
    return cfg


def tangent_project(
    tangent_basis: torch.Tensor, vectors: torch.Tensor
) -> torch.Tensor:
    coordinates = torch.einsum(
        "nrdhw,nrhw->ndhw", tangent_basis, vectors
    )
    return torch.einsum(
        "nrdhw,ndhw->nrhw", tangent_basis, coordinates
    )


def flatten_field(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))


def flatten_tangent(x: torch.Tensor) -> torch.Tensor:
    return (
        x.permute(0, 3, 4, 1, 2)
        .reshape(x.size(0), x.size(3) * x.size(4), x.size(1), x.size(2))
        .contiguous()
    )


def unflatten_field(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return (
        x.reshape(x.size(0), height, width, x.size(2))
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def frank_wolfe_convex_hull(
    candidates: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Approximate the closest point in a candidate convex hull.

    candidates: [Q,K,R], target: [Q,R], valid_mask: optional [Q,K].
    The method uses exact line search in each Frank-Wolfe step and requires no
    gradients or learnable parameters.
    """
    if iterations <= 0:
        error = (candidates - target.unsqueeze(1)).square().sum(dim=2)
        if valid_mask is not None:
            error = error.masked_fill(~valid_mask, float("inf"))
        index = error.argmin(dim=1)
        return candidates[
            torch.arange(candidates.size(0), device=candidates.device), index
        ]

    error = (candidates - target.unsqueeze(1)).square().sum(dim=2)
    if valid_mask is not None:
        error = error.masked_fill(~valid_mask, float("inf"))
    index = error.argmin(dim=1)
    rows = torch.arange(candidates.size(0), device=candidates.device)
    current = candidates[rows, index].clone()

    for _ in range(iterations):
        residual = current - target
        linear = torch.einsum("qkr,qr->qk", candidates, residual)
        if valid_mask is not None:
            linear = linear.masked_fill(~valid_mask, float("inf"))
        vertex_index = linear.argmin(dim=1)
        vertex = candidates[rows, vertex_index]
        direction = vertex - current
        numerator = -(residual * direction).sum(dim=1)
        denominator = direction.square().sum(dim=1).clamp_min(1e-20)
        gamma = (numerator / denominator).clamp(0.0, 1.0)
        current = current + gamma.unsqueeze(1) * direction
    return current


def retrieved_oracles(
    memory_null: torch.Tensor,
    local_state: torch.Tensor,
    tangent_basis: torch.Tensor,
    target_comp: torch.Tensor,
    topk_indices: torch.Tensor,
    null_projector: torch.Tensor,
    chunk_pixels: int,
    convex_iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n, rank, height, width = local_state.shape
    memory_flat = flatten_field(memory_null)
    local_flat = flatten_field(local_state)
    target_flat = flatten_field(target_comp)
    tangent_flat = flatten_tangent(tangent_basis)
    indices_flat = (
        topk_indices.permute(0, 2, 3, 1)
        .reshape(n, height * width, topk_indices.size(1))
        .contiguous()
    )

    hard = local_state.new_zeros(n, height * width, rank)
    convex = local_state.new_zeros(n, height * width, rank)
    for batch_index in range(n):
        memory = memory_flat[batch_index]
        for start in range(0, height * width, chunk_pixels):
            stop = min(start + chunk_pixels, height * width)
            candidates = gather_complement_candidates(
                memory,
                local_flat[batch_index, start:stop],
                tangent_flat[batch_index, start:stop],
                indices_flat[batch_index, start:stop],
                null_projector,
            )
            target = target_flat[batch_index, start:stop]
            error = (candidates - target.unsqueeze(1)).square().sum(dim=2)
            best_index = error.argmin(dim=1)
            rows = torch.arange(candidates.size(0), device=candidates.device)
            hard[batch_index, start:stop] = candidates[rows, best_index]
            convex[batch_index, start:stop] = frank_wolfe_convex_hull(
                candidates, target, convex_iterations
            )
    return (
        unflatten_field(hard, height, width),
        unflatten_field(convex, height, width),
    )


def global_oracles(
    memory_null: torch.Tensor,
    local_state: torch.Tensor,
    tangent_basis: torch.Tensor,
    target_comp: torch.Tensor,
    null_projector: torch.Tensor,
    exclusion_radius_lr: int,
    chunk_pixels: int,
    convex_iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GT choice over all non-local LR-HSI states; values remain LR-HSI-only."""
    n, rank, height, width = local_state.shape
    _, _, memory_h, memory_w = memory_null.shape
    memory_count = memory_h * memory_w
    memory_flat = flatten_field(memory_null)
    local_flat = flatten_field(local_state)
    target_flat = flatten_field(target_comp)
    tangent_flat = flatten_tangent(tangent_basis)

    memory_linear = torch.arange(memory_count, device=local_state.device)
    memory_y = torch.div(memory_linear, memory_w, rounding_mode="floor")
    memory_x = memory_linear.remainder(memory_w)
    query_count = height * width
    query_linear = torch.arange(query_count, device=local_state.device)
    query_y = torch.div(query_linear, width, rounding_mode="floor")
    query_x = query_linear.remainder(width)
    query_lr_y = torch.floor(
        (query_y.to(torch.float32) + 0.5) * memory_h / height
    ).to(torch.long).clamp_(0, memory_h - 1)
    query_lr_x = torch.floor(
        (query_x.to(torch.float32) + 0.5) * memory_w / width
    ).to(torch.long).clamp_(0, memory_w - 1)

    hard = local_state.new_zeros(n, query_count, rank)
    convex = local_state.new_zeros(n, query_count, rank)
    all_memory_indices = torch.arange(memory_count, device=local_state.device)

    for batch_index in range(n):
        memory = memory_flat[batch_index]
        for start in range(0, query_count, chunk_pixels):
            stop = min(start + chunk_pixels, query_count)
            q = stop - start
            indices = all_memory_indices.unsqueeze(0).expand(q, -1)
            candidates = gather_complement_candidates(
                memory,
                local_flat[batch_index, start:stop],
                tangent_flat[batch_index, start:stop],
                indices,
                null_projector,
            )
            cy = query_lr_y[start:stop].unsqueeze(1)
            cx = query_lr_x[start:stop].unsqueeze(1)
            valid = ~(
                (
                    (memory_y.unsqueeze(0) - cy).abs()
                    <= exclusion_radius_lr
                )
                & (
                    (memory_x.unsqueeze(0) - cx).abs()
                    <= exclusion_radius_lr
                )
            )
            target = target_flat[batch_index, start:stop]
            error = (candidates - target.unsqueeze(1)).square().sum(dim=2)
            error = error.masked_fill(~valid, float("inf"))
            best_index = error.argmin(dim=1)
            rows = torch.arange(q, device=local_state.device)
            hard[batch_index, start:stop] = candidates[rows, best_index]
            convex[batch_index, start:stop] = frank_wolfe_convex_hull(
                candidates,
                target,
                convex_iterations,
                valid_mask=valid,
            )
    return (
        unflatten_field(hard, height, width),
        unflatten_field(convex, height, width),
    )


def update_metric(
    meters: Dict[str, MetricAverager],
    name: str,
    prediction: torch.Tensor,
    gt: torch.Tensor,
    scale_ratio: int,
):
    meters[name].update(calc_metrics(prediction, gt, scale_ratio))


def add_error_energy(
    accumulators: Dict[str, float],
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
):
    accumulators[name] = accumulators.get(name, 0.0) + float(
        (prediction.double() - target.double()).square().sum().item()
    )


@torch.no_grad()
def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)

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

    memory_module = ObservableKeyedComplementMemory(
        top_k=cfg.nonlocal_top_k,
        exclusion_radius_lr=cfg.nonlocal_exclusion_radius_lr,
        query_chunk_pixels=cfg.nonlocal_query_chunk_pixels,
        temperature_ratio=cfg.nonlocal_temperature_ratio,
    ).to(device)

    metric_names = [
        "anchor",
        "stage2",
        "tangent_oracle",
        "stage2_gt_complement",
        "retrieval_soft",
        "retrieved_hard_oracle",
        "retrieved_convex_oracle",
        "global_hard_oracle",
        "global_convex_oracle",
        "full_null_oracle",
        "basis_oracle",
    ]
    meters = {name: MetricAverager() for name in metric_names}
    comp_errors: Dict[str, float] = {}
    missing_energy = 0.0
    tangent_energy = 0.0
    complement_energy = 0.0
    decomposition_error_max = 0.0

    retrieval_sum = {
        "top1_key_distance": 0.0,
        "effective_support": 0.0,
        "consensus_variance": 0.0,
        "max_weight": 0.0,
    }
    retrieval_pixels = 0

    for batch in test_loader:
        batch = move_to_device(batch, device)
        gt = batch["gt"]
        out = model(batch["lr_hsi"], batch["hr_msi"])
        basis = out["basis"]
        geometry = model.geometry

        gt_coeff = foundation.encode(gt, basis=basis)
        gt_null = geometry.project_null(gt_coeff)
        missing = gt_null - out["null_seed_coefficients"]
        tangent_target = tangent_project(out["tangent_basis"], missing)
        tangent_target = geometry.project_null(tangent_target)
        complement_target = project_complement_vectors(
            flatten_field(missing)[0],
            flatten_tangent(out["tangent_basis"])[0],
            geometry.null_projector,
        )
        complement_target = unflatten_field(
            complement_target.unsqueeze(0), gt.shape[-2], gt.shape[-1]
        )
        decomposition_error_max = max(
            decomposition_error_max,
            float(
                (
                    missing - tangent_target - complement_target
                ).abs().max().item()
            ),
        )

        lr_observable_key = torch.einsum(
            "mr,nrhw->nmhw",
            geometry.reduced_response.to(out["lr_coefficients"]),
            out["lr_coefficients"],
        )
        hr_observable_key = torch.einsum(
            "mr,nrhw->nmhw",
            geometry.reduced_response.to(out["anchor_coefficients"]),
            out["anchor_coefficients"],
        )
        memory_null = geometry.project_null(out["lr_coefficients"])
        local_null_state = (
            out["null_seed_coefficients"] + out["tangent_residual"]
        )

        retrieval = memory_module(
            query_observable=hr_observable_key,
            memory_observable=lr_observable_key,
            memory_null=memory_null,
            local_null_state=local_null_state,
            tangent_basis=out["tangent_basis"],
            null_projector=geometry.null_projector,
        )
        soft_comp = retrieval["complement_residual"]
        retrieved_hard, retrieved_convex = retrieved_oracles(
            memory_null=memory_null,
            local_state=local_null_state,
            tangent_basis=out["tangent_basis"],
            target_comp=complement_target,
            topk_indices=retrieval["topk_indices"],
            null_projector=geometry.null_projector,
            chunk_pixels=cfg.nonlocal_query_chunk_pixels,
            convex_iterations=cfg.nonlocal_convex_iterations,
        )
        global_hard, global_convex = global_oracles(
            memory_null=memory_null,
            local_state=local_null_state,
            tangent_basis=out["tangent_basis"],
            target_comp=complement_target,
            null_projector=geometry.null_projector,
            exclusion_radius_lr=cfg.nonlocal_exclusion_radius_lr,
            chunk_pixels=cfg.nonlocal_query_chunk_pixels,
            convex_iterations=cfg.global_convex_iterations,
        )

        stage2_gt_comp_coeff = out["corrected_coefficients"] + complement_target
        tangent_oracle_coeff = out["anchor_coefficients"] + tangent_target
        full_null_coeff = out["anchor_coefficients"] + missing

        coefficient_predictions = {
            "retrieval_soft": out["corrected_coefficients"] + soft_comp,
            "retrieved_hard_oracle": (
                out["corrected_coefficients"] + retrieved_hard
            ),
            "retrieved_convex_oracle": (
                out["corrected_coefficients"] + retrieved_convex
            ),
            "global_hard_oracle": out["corrected_coefficients"] + global_hard,
            "global_convex_oracle": (
                out["corrected_coefficients"] + global_convex
            ),
        }

        update_metric(
            meters, "anchor", out["anchor_hsi"], gt, cfg.scale_ratio
        )
        update_metric(
            meters, "stage2", out["reconstructed_hsi"], gt, cfg.scale_ratio
        )
        update_metric(
            meters,
            "tangent_oracle",
            foundation.decode(tangent_oracle_coeff, basis=basis),
            gt,
            cfg.scale_ratio,
        )
        update_metric(
            meters,
            "stage2_gt_complement",
            foundation.decode(stage2_gt_comp_coeff, basis=basis),
            gt,
            cfg.scale_ratio,
        )
        for name, coefficients in coefficient_predictions.items():
            update_metric(
                meters,
                name,
                foundation.decode(coefficients, basis=basis),
                gt,
                cfg.scale_ratio,
            )
        update_metric(
            meters,
            "full_null_oracle",
            foundation.decode(full_null_coeff, basis=basis),
            gt,
            cfg.scale_ratio,
        )
        update_metric(
            meters,
            "basis_oracle",
            foundation.decode(gt_coeff, basis=basis),
            gt,
            cfg.scale_ratio,
        )

        missing_energy += float(missing.double().square().sum().item())
        tangent_energy += float(tangent_target.double().square().sum().item())
        complement_energy += float(
            complement_target.double().square().sum().item()
        )
        for name, residual in {
            "retrieval_soft": soft_comp,
            "retrieved_hard_oracle": retrieved_hard,
            "retrieved_convex_oracle": retrieved_convex,
            "global_hard_oracle": global_hard,
            "global_convex_oracle": global_convex,
        }.items():
            add_error_energy(comp_errors, name, residual, complement_target)

        pixels = gt.size(0) * gt.size(2) * gt.size(3)
        retrieval_pixels += pixels
        retrieval_sum["top1_key_distance"] += float(
            retrieval["topk_distances"][:, 0].double().sum().item()
        )
        retrieval_sum["effective_support"] += float(
            retrieval["effective_support"].double().sum().item()
        )
        retrieval_sum["consensus_variance"] += float(
            retrieval["consensus_variance"].double().sum().item()
        )
        retrieval_sum["max_weight"] += float(
            retrieval["topk_weights"][:, 0].double().sum().item()
        )

    missing_energy = max(missing_energy, 1e-30)
    complement_energy = max(complement_energy, 1e-30)
    result = {
        "checkpoint": {
            "foundation": cfg.foundation_checkpoint,
            "local": cfg.local_checkpoint,
            "local_epoch": int(local_epoch),
            "local_best_metric": float(local_best),
        },
        "settings": {
            "tangent_dimension": cfg.tangent_dimension,
            "tangent_kernel_size": cfg.tangent_kernel_size,
            "tangent_dilation": cfg.tangent_dilation,
            "nonlocal_top_k": cfg.nonlocal_top_k,
            "nonlocal_exclusion_radius_lr": cfg.nonlocal_exclusion_radius_lr,
            "nonlocal_temperature_ratio": cfg.nonlocal_temperature_ratio,
            "nonlocal_convex_iterations": cfg.nonlocal_convex_iterations,
            "global_convex_iterations": cfg.global_convex_iterations,
            "key": "S @ coefficient (MSI-observable reduced-response key)",
            "value": "P_null C_lr from observed LR-HSI",
            "candidate_residual": (
                "P_null (I - P_tan) P_null "
                "(C_null_LR(q) - C_null_stage2(p))"
            ),
        },
        "metrics": {name: meters[name].average() for name in metric_names},
        "energy": {
            "missing_null_total": missing_energy,
            "tangent_target_fraction": tangent_energy / missing_energy,
            "complement_target_fraction": complement_energy / missing_energy,
            "decomposition_max_abs_error": decomposition_error_max,
        },
        "complement_recovery": {},
        "retrieval": {
            key: value / max(retrieval_pixels, 1)
            for key, value in retrieval_sum.items()
        },
    }
    for name, error in comp_errors.items():
        result["complement_recovery"][name] = {
            "capture": 1.0 - error / complement_energy,
            "relative_rmse": math.sqrt(error / complement_energy),
        }

    print("=" * 100)
    print("OMN-Net Tangent-Complement Non-local Spectral Recurrence Oracle")
    print("=" * 100)
    print(
        f"Loaded Stage2 checkpoint: epoch={local_epoch}, "
        f"stored_best={local_best:.4f}"
    )
    print(
        f"Anchor              : PSNR={result['metrics']['anchor']['PSNR']:.4f} "
        f"SAM={result['metrics']['anchor']['SAM']:.4f}"
    )
    print(
        f"Stage2 actual       : PSNR={result['metrics']['stage2']['PSNR']:.4f} "
        f"SAM={result['metrics']['stage2']['SAM']:.4f}"
    )
    print(
        f"Tangent oracle      : PSNR={result['metrics']['tangent_oracle']['PSNR']:.4f}"
    )
    print(
        f"Stage2 + GT P_comp  : PSNR="
        f"{result['metrics']['stage2_gt_complement']['PSNR']:.4f}"
    )
    for name, label in [
        ("global_hard_oracle", "Global hard LR-state oracle"),
        ("global_convex_oracle", "Global convex LR-state oracle"),
        ("retrieved_hard_oracle", "TopK key hard oracle"),
        ("retrieved_convex_oracle", "TopK key convex oracle"),
        ("retrieval_soft", "Observable-key soft retrieval"),
    ]:
        metric = result["metrics"][name]
        recovery = result["complement_recovery"][name]
        print(
            f"{label:<28}: PSNR={metric['PSNR']:.4f} "
            f"SAM={metric['SAM']:.4f} | "
            f"P_comp capture={100.0 * recovery['capture']:.2f}%"
        )
    print(
        f"Full null oracle     : PSNR="
        f"{result['metrics']['full_null_oracle']['PSNR']:.4f}"
    )
    print(
        f"P_comp energy share  : "
        f"{100.0 * result['energy']['complement_target_fraction']:.2f}%"
    )
    print(
        f"Retrieval support    : Neff="
        f"{result['retrieval']['effective_support']:.2f}, "
        f"consensus_var={result['retrieval']['consensus_variance']:.6e}"
    )

    out_dir = os.path.join(
        cfg.output_root,
        "diagnostics",
        "nonlocal_complement",
        cfg.dataset,
    )
    ensure_dir(out_dir)
    output_path = os.path.join(out_dir, "nonlocal_complement_oracle.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
