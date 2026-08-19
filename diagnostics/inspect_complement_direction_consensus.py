"""No-training P_comp direction-consensus diagnostic for OMN-Net.

K is fixed as the validated wide observable recall pool. This experiment asks
whether useful LR-HSI candidates repeat the same *signed tangent-complement
correction direction*. Two priors are compared without changing the candidate
values: uniform support inside the K pool and the existing observable retrieval
weights. GT is used only for metrics and oracle selection.
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
from models.complement_direction_consensus import ComplementDirectionConsensus
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
    p.add_argument("--consensus_top_m", type=int, default=32)
    p.add_argument("--consensus_temperature_ratio", type=float, default=1.0)
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


def frank_wolfe(candidates: torch.Tensor, target: torch.Tensor, iterations: int) -> torch.Tensor:
    error = (candidates - target.unsqueeze(1)).square().sum(dim=2)
    rows = torch.arange(candidates.size(0), device=candidates.device)
    current = candidates[rows, error.argmin(dim=1)].clone()
    for _ in range(max(iterations, 0)):
        residual = current - target
        linear = torch.einsum("qkr,qr->qk", candidates, residual)
        vertex = candidates[rows, linear.argmin(dim=1)]
        direction = vertex - current
        numerator = -(residual * direction).sum(dim=1)
        denominator = direction.square().sum(dim=1).clamp_min(1e-20)
        gamma = (numerator / denominator).clamp(0.0, 1.0)
        current = current + gamma.unsqueeze(1) * direction
    return current


def candidate_oracles(
    memory_null: torch.Tensor,
    local_state: torch.Tensor,
    tangent_basis: torch.Tensor,
    target_comp: torch.Tensor,
    candidate_indices: torch.Tensor,
    null_projector: torch.Tensor,
    chunk_pixels: int,
    convex_iterations: int,
):
    n, rank, height, width = local_state.shape
    k = candidate_indices.size(1)
    memory_flat = flatten_field(memory_null)
    local_flat = flatten_field(local_state)
    tangent_flat = flatten_tangent_field(tangent_basis)
    target_flat = flatten_field(target_comp)
    indices_flat = (
        candidate_indices.permute(0, 2, 3, 1)
        .reshape(n, height * width, k)
        .contiguous()
    )
    hard = local_state.new_zeros(n, height * width, rank)
    convex = local_state.new_zeros(n, height * width, rank)
    for b in range(n):
        for start in range(0, height * width, chunk_pixels):
            stop = min(start + chunk_pixels, height * width)
            candidates = gather_complement_candidates(
                memory_flat[b],
                local_flat[b, start:stop],
                tangent_flat[b, start:stop],
                indices_flat[b, start:stop],
                null_projector,
            )
            target = target_flat[b, start:stop]
            error = (candidates - target.unsqueeze(1)).square().sum(dim=2)
            rows = torch.arange(candidates.size(0), device=candidates.device)
            hard[b, start:stop] = candidates[rows, error.argmin(dim=1)]
            convex[b, start:stop] = frank_wolfe(candidates, target, convex_iterations)
    return unflatten_field(hard, height, width), unflatten_field(convex, height, width)


def residual_from_indices(
    memory_null: torch.Tensor,
    local_state: torch.Tensor,
    tangent_basis: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_weights: torch.Tensor,
    null_projector: torch.Tensor,
    chunk_pixels: int,
) -> torch.Tensor:
    n, rank, height, width = local_state.shape
    k = candidate_indices.size(1)
    memory_flat = flatten_field(memory_null)
    local_flat = flatten_field(local_state)
    tangent_flat = flatten_tangent_field(tangent_basis)
    indices_flat = (
        candidate_indices.permute(0, 2, 3, 1)
        .reshape(n, height * width, k)
        .contiguous()
    )
    weights_flat = (
        candidate_weights.permute(0, 2, 3, 1)
        .reshape(n, height * width, k)
        .contiguous()
    )
    out = local_state.new_zeros(n, height * width, rank)
    for b in range(n):
        for start in range(0, height * width, chunk_pixels):
            stop = min(start + chunk_pixels, height * width)
            candidates = gather_complement_candidates(
                memory_flat[b],
                local_flat[b, start:stop],
                tangent_flat[b, start:stop],
                indices_flat[b, start:stop],
                null_projector,
            )
            out[b, start:stop] = torch.sum(
                weights_flat[b, start:stop].unsqueeze(-1) * candidates, dim=1
            )
    return unflatten_field(out, height, width)


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
    consensus_uniform = ComplementDirectionConsensus(
        top_m=cfg.consensus_top_m,
        prior_mode="uniform",
        temperature_ratio=cfg.consensus_temperature_ratio,
        query_chunk_pixels=cfg.nonlocal_query_chunk_pixels,
    ).to(device)
    consensus_observable = ComplementDirectionConsensus(
        top_m=cfg.consensus_top_m,
        prior_mode="observable",
        temperature_ratio=cfg.consensus_temperature_ratio,
        query_chunk_pixels=cfg.nonlocal_query_chunk_pixels,
    ).to(device)

    names = [
        "stage2",
        "stage2_gt_comp",
        "wide_hard_oracle",
        "wide_convex_oracle",
        "observable_m_hard_oracle",
        "observable_m_convex_oracle",
        "uniform_top1",
        "uniform_soft",
        "uniform_hard_oracle",
        "uniform_convex_oracle",
        "observable_direction_top1",
        "observable_direction_soft",
        "observable_direction_hard_oracle",
        "observable_direction_convex_oracle",
    ]
    meters = {name: MetricAverager() for name in names}
    comp_error: Dict[str, float] = {}
    comp_energy = 0.0
    diagnostic_sum = {
        "uniform_resultant_norm": 0.0,
        "uniform_alignment": 0.0,
        "observable_resultant_norm": 0.0,
        "observable_alignment": 0.0,
    }
    diagnostic_pixels = 0

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
        uniform = consensus_uniform(
            topk_indices=retrieved["topk_indices"],
            topk_observable_weights=retrieved["topk_weights"],
            memory_null=memory_null,
            local_null_state=local_state,
            tangent_basis=out["tangent_basis"],
            null_projector=geometry.null_projector,
        )
        obs_direction = consensus_observable(
            topk_indices=retrieved["topk_indices"],
            topk_observable_weights=retrieved["topk_weights"],
            memory_null=memory_null,
            local_null_state=local_state,
            tangent_basis=out["tangent_basis"],
            null_projector=geometry.null_projector,
        )

        m = min(cfg.consensus_top_m, retrieved["topk_indices"].size(1))
        obs_m_indices = retrieved["topk_indices"][:, :m]

        wide_hard, wide_convex = candidate_oracles(
            memory_null, local_state, out["tangent_basis"], comp_target,
            retrieved["topk_indices"], geometry.null_projector,
            cfg.nonlocal_query_chunk_pixels, cfg.oracle_convex_iterations,
        )
        obs_m_hard, obs_m_convex = candidate_oracles(
            memory_null, local_state, out["tangent_basis"], comp_target,
            obs_m_indices, geometry.null_projector,
            cfg.nonlocal_query_chunk_pixels, cfg.oracle_convex_iterations,
        )
        uniform_hard, uniform_convex = candidate_oracles(
            memory_null, local_state, out["tangent_basis"], comp_target,
            uniform["selected_indices"], geometry.null_projector,
            cfg.nonlocal_query_chunk_pixels, cfg.oracle_convex_iterations,
        )
        obs_dir_hard, obs_dir_convex = candidate_oracles(
            memory_null, local_state, out["tangent_basis"], comp_target,
            obs_direction["selected_indices"], geometry.null_projector,
            cfg.nonlocal_query_chunk_pixels, cfg.oracle_convex_iterations,
        )

        residuals = {
            "wide_hard_oracle": wide_hard,
            "wide_convex_oracle": wide_convex,
            "observable_m_hard_oracle": obs_m_hard,
            "observable_m_convex_oracle": obs_m_convex,
            "uniform_top1": uniform["top1_residual"],
            "uniform_soft": uniform["soft_residual"],
            "uniform_hard_oracle": uniform_hard,
            "uniform_convex_oracle": uniform_convex,
            "observable_direction_top1": obs_direction["top1_residual"],
            "observable_direction_soft": obs_direction["soft_residual"],
            "observable_direction_hard_oracle": obs_dir_hard,
            "observable_direction_convex_oracle": obs_dir_convex,
        }

        update_metric(meters, "stage2", out["reconstructed_hsi"], gt, cfg.scale_ratio)
        gt_comp_hsi = foundation.decode(
            out["corrected_coefficients"] + comp_target, basis=basis
        )
        update_metric(meters, "stage2_gt_comp", gt_comp_hsi, gt, cfg.scale_ratio)
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
        diagnostic_pixels += pixels
        diagnostic_sum["uniform_resultant_norm"] += float(
            uniform["resultant_norm"].double().sum().item()
        )
        diagnostic_sum["uniform_alignment"] += float(
            uniform["selected_alignment"].double().sum().item()
        )
        diagnostic_sum["observable_resultant_norm"] += float(
            obs_direction["resultant_norm"].double().sum().item()
        )
        diagnostic_sum["observable_alignment"] += float(
            obs_direction["selected_alignment"].double().sum().item()
        )

    comp_energy = max(comp_energy, 1e-30)
    result = {
        "checkpoint": {
            "local_epoch": int(local_epoch),
            "local_best_metric": float(local_best),
        },
        "settings": {
            "K": cfg.nonlocal_top_k,
            "M": cfg.consensus_top_m,
            "consensus_temperature_ratio": cfg.consensus_temperature_ratio,
            "oracle_convex_iterations": cfg.oracle_convex_iterations,
        },
        "metrics": {name: meters[name].average() for name in names},
        "complement_recovery": {},
        "direction_diagnostics": {
            key: value / max(diagnostic_pixels, 1)
            for key, value in diagnostic_sum.items()
        },
    }
    for name, error in comp_error.items():
        result["complement_recovery"][name] = {
            "capture": 1.0 - error / comp_energy,
            "relative_rmse": math.sqrt(error / comp_energy),
        }

    print("=" * 100)
    print("OMN-Net P_comp Direction Consensus")
    print("=" * 100)
    print(
        f"Stage2 checkpoint: epoch={local_epoch}, stored_best={local_best:.4f} | "
        f"K={cfg.nonlocal_top_k}, M={cfg.consensus_top_m}"
    )
    print(
        f"Stage2                    : PSNR={result['metrics']['stage2']['PSNR']:.4f} "
        f"SAM={result['metrics']['stage2']['SAM']:.4f}"
    )
    print(
        f"Stage2 + GT P_comp        : PSNR={result['metrics']['stage2_gt_comp']['PSNR']:.4f}"
    )

    rows = [
        ("wide_hard_oracle", "Wide-K hard oracle"),
        ("wide_convex_oracle", "Wide-K convex oracle"),
        ("observable_m_hard_oracle", "Observable Top-M hard oracle"),
        ("observable_m_convex_oracle", "Observable Top-M convex oracle"),
        ("uniform_top1", "Direction-uniform top-1"),
        ("uniform_soft", "Direction-uniform soft"),
        ("uniform_hard_oracle", "Direction-uniform hard oracle"),
        ("uniform_convex_oracle", "Direction-uniform convex oracle"),
        ("observable_direction_top1", "Direction-obs top-1"),
        ("observable_direction_soft", "Direction-obs soft"),
        ("observable_direction_hard_oracle", "Direction-obs hard oracle"),
        ("observable_direction_convex_oracle", "Direction-obs convex oracle"),
    ]
    for name, label in rows:
        metric = result["metrics"][name]
        recovery = result["complement_recovery"][name]
        print(
            f"{label:<32}: PSNR={metric['PSNR']:.4f} SAM={metric['SAM']:.4f} | "
            f"P_comp capture={100.0 * recovery['capture']:.2f}%"
        )

    diag = result["direction_diagnostics"]
    print(
        "Direction coherence             : "
        f"uniform |g|={diag['uniform_resultant_norm']:.4f}, "
        f"sel_align={diag['uniform_alignment']:.4f}; "
        f"observable |g|={diag['observable_resultant_norm']:.4f}, "
        f"sel_align={diag['observable_alignment']:.4f}"
    )

    out_dir = os.path.join(
        cfg.output_root, "diagnostics", "nonlocal_complement", cfg.dataset
    )
    ensure_dir(out_dir)
    output_path = os.path.join(out_dir, "complement_direction_consensus.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
