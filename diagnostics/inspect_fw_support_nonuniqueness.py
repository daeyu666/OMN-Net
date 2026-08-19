"""Diagnose whether GT Frank-Wolfe support is a stable learning target.

The K=690 candidate convex hull can contain many different sparse decompositions
of nearly the same P_comp residual.  E13/E14 distilled one Frank-Wolfe support
as if candidate identity were unique.  This script holds the target and
candidate set fixed, changes only the Frank-Wolfe initialization, and measures:

* pairwise support Jaccard;
* pairwise weighted support overlap;
* pairwise reconstructed-residual disagreement;
* oracle RRMSE/objective spread;
* active/union/intersection support sizes.

GT is used only to define the oracle P_comp target.  This is a diagnostic, not
an inference path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import parse_args
from data_loader import build_loaders
from models import (
    LocalNullManifoldNet,
    build_spectral_response,
    flatten_spatial,
    flatten_tangent,
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
    p.add_argument("--diagnostic_queries", type=int, default=512)
    p.add_argument("--nonlocal_top_k", type=int, default=690)
    p.add_argument("--nonlocal_exclusion_radius_lr", type=int, default=1)
    p.add_argument("--nonlocal_query_chunk_pixels", type=int, default=64)
    p.add_argument("--fw_repeats", type=int, default=10)
    p.add_argument("--fw_iterations", type=int, default=30)
    p.add_argument("--fw_init_pool", type=int, default=32)
    p.add_argument("--support_tolerance", type=float, default=1e-6)

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"
    if cfg.diagnostic_queries < 1 or cfg.fw_repeats < 2:
        raise ValueError("diagnostic_queries must be positive and repeats >= 2")
    if cfg.fw_iterations < 0 or cfg.fw_init_pool < 1:
        raise ValueError("invalid Frank-Wolfe settings")
    if cfg.support_tolerance <= 0:
        raise ValueError("support_tolerance must be positive")
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
def retrieve_selected_topk(
    query_observable: torch.Tensor,
    memory_observable: torch.Tensor,
    selected: torch.Tensor,
    top_k: int,
    exclusion_radius_lr: int,
    chunk_pixels: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if query_observable.size(0) != 1 or memory_observable.size(0) != 1:
        raise ValueError("support diagnostic currently expects batch size 1")
    _, channels, height, width = query_observable.shape
    _, _, memory_h, memory_w = memory_observable.shape
    memory_count = memory_h * memory_w
    max_excluded = (2 * exclusion_radius_lr + 1) ** 2
    actual_k = min(int(top_k), memory_count - max_excluded)
    if actual_k < 1:
        raise ValueError("no candidate remains after local exclusion")

    q_all = flatten_spatial(query_observable)[0].float()
    memory = flatten_spatial(memory_observable)[0].float()
    mean = memory.mean(dim=0, keepdim=True)
    std = memory.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    q_selected = (q_all[selected] - mean) / std
    memory_std = (memory - mean) / std

    memory_linear = torch.arange(memory_count, device=query_observable.device)
    memory_y = torch.div(memory_linear, memory_w, rounding_mode="floor")
    memory_x = memory_linear.remainder(memory_w)
    query_y = torch.div(selected, width, rounding_mode="floor")
    query_x = selected.remainder(width)
    query_lr_y = torch.floor(
        (query_y.float() + 0.5) * memory_h / height
    ).long().clamp_(0, memory_h - 1)
    query_lr_x = torch.floor(
        (query_x.float() + 0.5) * memory_w / width
    ).long().clamp_(0, memory_w - 1)

    index_chunks = []
    distance_chunks = []
    for start in range(0, selected.numel(), chunk_pixels):
        stop = min(start + chunk_pixels, selected.numel())
        distances = torch.cdist(q_selected[start:stop], memory_std, p=2).square()
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
        index_chunks.append(top_idx)
        distance_chunks.append(top_dist)
    return torch.cat(index_chunks, dim=0), torch.cat(distance_chunks, dim=0)


@torch.no_grad()
def randomized_fw(
    candidates: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
    init_pool: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, _ = candidates.shape
    rows = torch.arange(q, device=candidates.device)
    error = (candidates - target.unsqueeze(1)).square().sum(dim=2)
    pool_k = min(max(int(init_pool), 1), k)
    pool = torch.topk(error, k=pool_k, dim=1, largest=False, sorted=False).indices
    generator = torch.Generator(device=candidates.device)
    generator.manual_seed(int(seed))
    chosen_column = torch.randint(
        0, pool_k, (q,), device=candidates.device, generator=generator
    )
    initial = pool[rows, chosen_column]

    weights = candidates.new_zeros(q, k)
    weights[rows, initial] = 1.0
    current = candidates[rows, initial].clone()

    for _ in range(max(int(iterations), 0)):
        residual = current - target
        linear = torch.einsum("qkr,qr->qk", candidates, residual)
        vertex_index = linear.argmin(dim=1)
        vertex = candidates[rows, vertex_index]
        direction = vertex - current
        numerator = -(residual * direction).sum(dim=1)
        denominator = direction.square().sum(dim=1).clamp_min(1e-20)
        gamma = (numerator / denominator).clamp(0.0, 1.0)
        weights = weights * (1.0 - gamma.unsqueeze(1))
        weights[rows, vertex_index] += gamma
        current = current + gamma.unsqueeze(1) * direction

    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return current, weights


def summarize_stability(
    residuals: List[torch.Tensor],
    weights: List[torch.Tensor],
    target: torch.Tensor,
    tolerance: float,
) -> Dict[str, float]:
    target_energy = target.double().square().sum().item()
    target_energy = max(float(target_energy), 1e-30)

    run_rrmse = []
    run_active = []
    supports = []
    for residual, weight in zip(residuals, weights):
        error = (residual.double() - target.double()).square().sum().item()
        run_rrmse.append(math.sqrt(max(float(error) / target_energy, 0.0)))
        support = weight > tolerance
        supports.append(support)
        run_active.append(float(support.float().sum(dim=1).mean().item()))

    jaccards = []
    weighted_overlaps = []
    output_rrmse = []
    for i, j in combinations(range(len(weights)), 2):
        si, sj = supports[i], supports[j]
        intersection = (si & sj).sum(dim=1).float()
        union = (si | sj).sum(dim=1).float().clamp_min(1.0)
        jaccards.append(float((intersection / union).mean().item()))
        weighted_overlaps.append(
            float(torch.minimum(weights[i], weights[j]).sum(dim=1).mean().item())
        )
        disagreement = (
            residuals[i].double() - residuals[j].double()
        ).square().sum().item()
        output_rrmse.append(
            math.sqrt(max(float(disagreement) / target_energy, 0.0))
        )

    support_stack = torch.stack(supports, dim=0)
    union_size = support_stack.any(dim=0).float().sum(dim=1).mean().item()
    intersection_size = support_stack.all(dim=0).float().sum(dim=1).mean().item()

    rrmse_tensor = torch.tensor(run_rrmse, dtype=torch.float64)
    active_tensor = torch.tensor(run_active, dtype=torch.float64)
    return {
        "oracle_rrmse_mean": float(rrmse_tensor.mean().item()),
        "oracle_rrmse_std": float(rrmse_tensor.std(unbiased=False).item()),
        "oracle_rrmse_min": float(rrmse_tensor.min().item()),
        "oracle_rrmse_max": float(rrmse_tensor.max().item()),
        "active_support_mean": float(active_tensor.mean().item()),
        "active_support_std": float(active_tensor.std(unbiased=False).item()),
        "support_union_size_mean": float(union_size),
        "support_intersection_size_mean": float(intersection_size),
        "pairwise_jaccard_mean": float(sum(jaccards) / len(jaccards)),
        "pairwise_jaccard_min": float(min(jaccards)),
        "pairwise_weight_overlap_mean": float(
            sum(weighted_overlaps) / len(weighted_overlaps)
        ),
        "pairwise_output_rrmse_mean": float(
            sum(output_rrmse) / len(output_rrmse)
        ),
        "pairwise_output_rrmse_max": float(max(output_rrmse)),
    }


@torch.no_grad()
def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    model, foundation, local_epoch, local_best = build_local_model(
        cfg, info, device
    )

    aggregate: Dict[str, List[float]] = {}
    batch_reports = []
    for batch_index, batch in enumerate(test_loader):
        batch = move_to_device(batch, device)
        if batch["gt"].size(0) != 1:
            raise ValueError("use diagnostic batch size 1")
        out = model(batch["lr_hsi"], batch["hr_msi"])
        geometry = model.geometry
        gt_coeff = foundation.encode(batch["gt"], basis=out["basis"])
        gt_null = geometry.project_null(gt_coeff)
        missing = gt_null - out["null_seed_coefficients"]
        target_flat_all = project_complement_vectors(
            flatten_spatial(missing)[0],
            flatten_tangent(out["tangent_basis"])[0],
            geometry.null_projector,
        )

        _, _, height, width = out["anchor_coefficients"].shape
        q_count = height * width
        sample_count = min(int(cfg.diagnostic_queries), q_count)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(cfg.seed) + 1009 * batch_index)
        selected = torch.randperm(
            q_count, generator=generator, device=device
        )[:sample_count]

        reduced = geometry.reduced_response.to(out["lr_coefficients"])
        query_observable = torch.einsum(
            "mr,nrhw->nmhw", reduced, out["anchor_coefficients"]
        )
        memory_observable = torch.einsum(
            "mr,nrhw->nmhw", reduced, out["lr_coefficients"]
        )
        top_idx, _ = retrieve_selected_topk(
            query_observable,
            memory_observable,
            selected,
            cfg.nonlocal_top_k,
            cfg.nonlocal_exclusion_radius_lr,
            cfg.nonlocal_query_chunk_pixels,
        )

        memory_null = geometry.project_null(out["lr_coefficients"])
        local_state = out["null_seed_coefficients"] + out["tangent_residual"]
        candidates = gather_complement_candidates(
            flatten_spatial(memory_null)[0],
            flatten_spatial(local_state)[0, selected],
            flatten_tangent(out["tangent_basis"])[0, selected],
            top_idx,
            geometry.null_projector,
        )
        target = target_flat_all[selected]

        residual_runs = []
        weight_runs = []
        for repeat in range(cfg.fw_repeats):
            residual, weights = randomized_fw(
                candidates,
                target,
                cfg.fw_iterations,
                cfg.fw_init_pool,
                seed=int(cfg.seed) + 7919 * (repeat + 1) + batch_index,
            )
            residual_runs.append(residual)
            weight_runs.append(weights)

        report = summarize_stability(
            residual_runs,
            weight_runs,
            target,
            cfg.support_tolerance,
        )
        report.update(
            {
                "batch_index": batch_index,
                "queries": sample_count,
                "top_k": int(top_idx.size(1)),
            }
        )
        batch_reports.append(report)
        for key, value in report.items():
            if isinstance(value, float):
                aggregate.setdefault(key, []).append(value)

    summary = {
        "local_checkpoint_epoch": int(local_epoch),
        "local_checkpoint_best": float(local_best),
        "fw_repeats": int(cfg.fw_repeats),
        "fw_iterations": int(cfg.fw_iterations),
        "fw_init_pool": int(cfg.fw_init_pool),
        "support_tolerance": float(cfg.support_tolerance),
        "batches": batch_reports,
        "mean": {
            key: float(sum(values) / len(values))
            for key, values in aggregate.items()
        },
    }

    out_dir = os.path.join(cfg.output_root, "diagnostics", cfg.dataset)
    ensure_dir(out_dir)
    output_path = os.path.join(out_dir, "fw_support_nonuniqueness.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    mean = summary["mean"]
    print("=== Frank-Wolfe support non-uniqueness diagnostic ===")
    print(
        f"queries/batch={cfg.diagnostic_queries} K={cfg.nonlocal_top_k} "
        f"repeats={cfg.fw_repeats} iterations={cfg.fw_iterations}"
    )
    print(
        f"support Jaccard={mean['pairwise_jaccard_mean']:.4f} "
        f"(min-pair={mean['pairwise_jaccard_min']:.4f}) | "
        f"weight overlap={mean['pairwise_weight_overlap_mean']:.4f}"
    )
    print(
        f"oracle RRMSE={mean['oracle_rrmse_mean']:.6f} "
        f"+/-{mean['oracle_rrmse_std']:.6f} | "
        f"pairwise output RRMSE={mean['pairwise_output_rrmse_mean']:.6f}"
    )
    print(
        f"active={mean['active_support_mean']:.2f} | "
        f"union={mean['support_union_size_mean']:.2f} | "
        f"intersection={mean['support_intersection_size_mean']:.2f}"
    )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
