"""No-training prototype-consensus diagnostic for OMN-Net innovation point 2.

K is fixed as the validated high-recall observable pool. LR-HSI null states are
clustered into deterministic scene spectral prototypes. For each HR query, the
prototype that is most enriched inside its observable-compatible pool relative
to the prototype's scene-wide prior is selected. Only candidates belonging to
that winning prototype can contribute P_comp values.

GT is used only for metrics and oracle selection inside the already selected
prototype; it never affects retrieval, clustering, prototype scoring, or the
non-GT residual.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List

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
from models.complement_consensus import SpectralPrototypeConsensus
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

    p.add_argument("--nonlocal_top_k", type=int, default=690)
    p.add_argument("--nonlocal_exclusion_radius_lr", type=int, default=1)
    p.add_argument("--nonlocal_query_chunk_pixels", type=int, default=64)
    p.add_argument("--nonlocal_temperature_ratio", type=float, default=1.0)

    p.add_argument("--consensus_prototypes", type=int, default=32)
    p.add_argument("--consensus_kmeans_iterations", type=int, default=20)
    p.add_argument("--consensus_prior_exponent", type=float, default=0.5)
    p.add_argument("--consensus_min_candidates", type=int, default=4)
    p.add_argument("--consensus_query_chunk_pixels", type=int, default=64)
    p.add_argument("--oracle_convex_iterations", type=int, default=30)

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"
    return cfg


def flatten_field(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))


def flatten_tangent_field(x: torch.Tensor) -> torch.Tensor:
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


def tangent_project(tangent_basis: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    coordinates = torch.einsum("nrdhw,nrhw->ndhw", tangent_basis, vectors)
    return torch.einsum("nrdhw,ndhw->nrhw", tangent_basis, coordinates)


def masked_frank_wolfe(
    candidates: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    error = (candidates - target.unsqueeze(1)).square().sum(dim=2)
    error = error.masked_fill(~valid, float("inf"))
    rows = torch.arange(candidates.size(0), device=candidates.device)
    current = candidates[rows, error.argmin(dim=1)].clone()
    for _ in range(max(iterations, 0)):
        residual = current - target
        linear = torch.einsum("qkr,qr->qk", candidates, residual)
        linear = linear.masked_fill(~valid, float("inf"))
        vertex = candidates[rows, linear.argmin(dim=1)]
        direction = vertex - current
        numerator = -(residual * direction).sum(dim=1)
        denominator = direction.square().sum(dim=1).clamp_min(1e-20)
        gamma = (numerator / denominator).clamp(0.0, 1.0)
        current = current + gamma.unsqueeze(1) * direction
    return current


def winner_oracles(
    memory_null: torch.Tensor,
    local_state: torch.Tensor,
    tangent_basis: torch.Tensor,
    target_comp: torch.Tensor,
    topk_indices: torch.Tensor,
    winner_mask: torch.Tensor,
    null_projector: torch.Tensor,
    chunk_pixels: int,
    convex_iterations: int,
):
    n, rank, height, width = local_state.shape
    k = topk_indices.size(1)
    memory_flat = flatten_field(memory_null)
    local_flat = flatten_field(local_state)
    tangent_flat = flatten_tangent_field(tangent_basis)
    target_flat = flatten_field(target_comp)
    indices_flat = (
        topk_indices.permute(0, 2, 3, 1)
        .reshape(n, height * width, k)
        .contiguous()
    )
    mask_flat = (
        winner_mask.permute(0, 2, 3, 1)
        .reshape(n, height * width, k)
        .contiguous()
    )
    hard = local_state.new_zeros(n, height * width, rank)
    convex = local_state.new_zeros(n, height * width, rank)

    for b in range(n):
        memory = memory_flat[b]
        for start in range(0, height * width, chunk_pixels):
            stop = min(start + chunk_pixels, height * width)
            indices = indices_flat[b, start:stop]
            valid = mask_flat[b, start:stop]
            candidates = gather_complement_candidates(
                memory,
                local_flat[b, start:stop],
                tangent_flat[b, start:stop],
                indices,
                null_projector,
            )
            target = target_flat[b, start:stop]
            error = (candidates - target.unsqueeze(1)).square().sum(dim=2)
            error = error.masked_fill(~valid, float("inf"))
            rows = torch.arange(candidates.size(0), device=candidates.device)
            hard[b, start:stop] = candidates[rows, error.argmin(dim=1)]
            convex[b, start:stop] = masked_frank_wolfe(
                candidates,
                target,
                valid,
                convex_iterations,
            )
    return unflatten_field(hard, height, width), unflatten_field(convex, height, width)


def update_metric(meters, name, prediction, gt, scale_ratio):
    meters[name].update(calc_metrics(prediction, gt, scale_ratio))


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

    recall = ObservableKeyedComplementMemory(
        top_k=cfg.nonlocal_top_k,
        exclusion_radius_lr=cfg.nonlocal_exclusion_radius_lr,
        query_chunk_pixels=cfg.nonlocal_query_chunk_pixels,
        temperature_ratio=cfg.nonlocal_temperature_ratio,
    ).to(device)
    consensus = SpectralPrototypeConsensus(
        n_prototypes=cfg.consensus_prototypes,
        kmeans_iterations=cfg.consensus_kmeans_iterations,
        prior_exponent=cfg.consensus_prior_exponent,
        min_cluster_candidates=cfg.consensus_min_candidates,
        query_chunk_pixels=cfg.consensus_query_chunk_pixels,
    ).to(device)

    names = [
        "stage2",
        "stage2_gt_comp",
        "consensus_uniform",
        "consensus_soft",
        "consensus_hard_oracle",
        "consensus_convex_oracle",
    ]
    meters = {name: MetricAverager() for name in names}
    comp_error: Dict[str, float] = {}
    comp_energy = 0.0
    stat_sum = {
        "winner_mass": 0.0,
        "winner_prior": 0.0,
        "winner_enrichment": 0.0,
        "winner_support": 0.0,
    }
    stat_pixels = 0

    for batch in test_loader:
        batch = move_to_device(batch, device)
        gt = batch["gt"]
        out = model(batch["lr_hsi"], batch["hr_msi"])
        basis = out["basis"]
        geometry = model.geometry

        gt_coeff = foundation.encode(gt, basis=basis)
        gt_null = geometry.project_null(gt_coeff)
        missing = gt_null - out["null_seed_coefficients"]
        comp_flat = project_complement_vectors(
            flatten_field(missing)[0],
            flatten_tangent_field(out["tangent_basis"])[0],
            geometry.null_projector,
        )
        comp_target = unflatten_field(
            comp_flat.unsqueeze(0), gt.shape[-2], gt.shape[-1]
        )

        lr_key = torch.einsum(
            "mr,nrhw->nmhw",
            geometry.reduced_response.to(out["lr_coefficients"]),
            out["lr_coefficients"],
        )
        hr_key = torch.einsum(
            "mr,nrhw->nmhw",
            geometry.reduced_response.to(out["anchor_coefficients"]),
            out["anchor_coefficients"],
        )
        memory_null = geometry.project_null(out["lr_coefficients"])
        local_state = out["null_seed_coefficients"] + out["tangent_residual"]

        retrieved = recall(
            query_observable=hr_key,
            memory_observable=lr_key,
            memory_null=memory_null,
            local_null_state=local_state,
            tangent_basis=out["tangent_basis"],
            null_projector=geometry.null_projector,
        )
        selected = consensus(
            topk_indices=retrieved["topk_indices"],
            topk_observable_weights=retrieved["topk_weights"],
            memory_null=memory_null,
            local_null_state=local_state,
            tangent_basis=out["tangent_basis"],
            null_projector=geometry.null_projector,
        )
        hard, convex = winner_oracles(
            memory_null=memory_null,
            local_state=local_state,
            tangent_basis=out["tangent_basis"],
            target_comp=comp_target,
            topk_indices=retrieved["topk_indices"],
            winner_mask=selected["winner_mask"],
            null_projector=geometry.null_projector,
            chunk_pixels=cfg.consensus_query_chunk_pixels,
            convex_iterations=cfg.oracle_convex_iterations,
        )

        residuals = {
            "consensus_uniform": selected["uniform_residual"],
            "consensus_soft": selected["soft_residual"],
            "consensus_hard_oracle": hard,
            "consensus_convex_oracle": convex,
        }
        update_metric(
            meters, "stage2", out["reconstructed_hsi"], gt, cfg.scale_ratio
        )
        update_metric(
            meters,
            "stage2_gt_comp",
            foundation.decode(out["corrected_coefficients"] + comp_target, basis=basis),
            gt,
            cfg.scale_ratio,
        )
        for name, residual in residuals.items():
            prediction = foundation.decode(
                out["corrected_coefficients"] + residual, basis=basis
            )
            update_metric(meters, name, prediction, gt, cfg.scale_ratio)
            comp_error[name] = comp_error.get(name, 0.0) + float(
                (residual.double() - comp_target.double()).square().sum().item()
            )

        comp_energy += float(comp_target.double().square().sum().item())
        pixels = gt.size(0) * gt.size(2) * gt.size(3)
        stat_pixels += pixels
        for key in stat_sum:
            stat_sum[key] += float(selected[key].double().sum().item())

    comp_energy = max(comp_energy, 1e-30)
    result = {
        "checkpoint": {
            "local_epoch": int(local_epoch),
            "local_best_metric": float(local_best),
        },
        "settings": {
            "nonlocal_top_k": cfg.nonlocal_top_k,
            "consensus_prototypes": cfg.consensus_prototypes,
            "consensus_kmeans_iterations": cfg.consensus_kmeans_iterations,
            "consensus_prior_exponent": cfg.consensus_prior_exponent,
            "consensus_min_candidates": cfg.consensus_min_candidates,
            "oracle_convex_iterations": cfg.oracle_convex_iterations,
        },
        "metrics": {name: meters[name].average() for name in names},
        "complement_recovery": {},
        "consensus": {
            key: value / max(stat_pixels, 1) for key, value in stat_sum.items()
        },
    }
    for name, error in comp_error.items():
        result["complement_recovery"][name] = {
            "capture": 1.0 - error / comp_energy,
            "relative_rmse": math.sqrt(error / comp_energy),
        }

    print("=" * 104)
    print("OMN-Net P_comp Spectral Prototype Consensus Diagnostic")
    print("=" * 104)
    print(
        f"Stage2 checkpoint: epoch={local_epoch}, stored_best={local_best:.4f} | "
        f"K={cfg.nonlocal_top_k}, prototypes={cfg.consensus_prototypes}"
    )
    print(
        f"Stage2              : PSNR={result['metrics']['stage2']['PSNR']:.4f} "
        f"SAM={result['metrics']['stage2']['SAM']:.4f}"
    )
    print(
        f"Stage2 + GT P_comp  : PSNR={result['metrics']['stage2_gt_comp']['PSNR']:.4f}"
    )
    labels = [
        ("consensus_uniform", "Prototype consensus uniform"),
        ("consensus_soft", "Prototype consensus obs-soft"),
        ("consensus_hard_oracle", "Winning-prototype hard oracle"),
        ("consensus_convex_oracle", "Winning-prototype convex oracle"),
    ]
    for name, label in labels:
        metric = result["metrics"][name]
        recovery = result["complement_recovery"][name]
        print(
            f"{label:<32}: PSNR={metric['PSNR']:.4f} "
            f"SAM={metric['SAM']:.4f} | "
            f"P_comp capture={100.0 * recovery['capture']:.2f}%"
        )
    print(
        "Consensus stats     : "
        f"support={result['consensus']['winner_support']:.2f}, "
        f"mass={result['consensus']['winner_mass']:.4f}, "
        f"prior={result['consensus']['winner_prior']:.4f}, "
        f"enrichment={result['consensus']['winner_enrichment']:.2f}"
    )

    out_dir = os.path.join(
        cfg.output_root, "diagnostics", "nonlocal_complement", cfg.dataset
    )
    ensure_dir(out_dir)
    output_path = os.path.join(out_dir, "complement_prototype_consensus.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
