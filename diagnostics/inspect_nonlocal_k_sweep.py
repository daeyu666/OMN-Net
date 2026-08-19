"""Sweep the smallest observable-recall K for OMN-Net innovation point 2.

This is a no-training diagnostic. It retrieves the largest requested candidate
set once, reuses each prefix K, and measures how much of the global LR-HSI
convex complement-recovery ceiling is retained.

The primary selection rule is the smallest K reaching 95% of the global convex
P_comp capture. The script also reports 90% and 99% thresholds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

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
from diagnostics.inspect_nonlocal_complement_oracle import (
    flatten_field,
    flatten_tangent,
    frank_wolfe_convex_hull,
    global_oracles,
    unflatten_field,
)
from utils import ensure_dir, get_device, load_checkpoint, move_to_device, set_seed


def _has_option(arguments: List[str], option: str) -> bool:
    return any(x == option or x.startswith(option + "=") for x in arguments)


def _parse_int_list(text: str) -> List[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or values[0] < 1:
        raise ValueError("nonlocal_top_ks must contain positive integers")
    return values


def _parse_float_list(text: str) -> List[float]:
    values = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if not values or values[0] <= 0.0 or values[-1] > 1.0:
        raise ValueError("retention thresholds must lie in (0, 1]")
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
    p.add_argument(
        "--nonlocal_top_ks",
        type=str,
        default="16,32,48,64,96,128,192,256",
    )
    p.add_argument("--nonlocal_exclusion_radius_lr", type=int, default=1)
    p.add_argument("--nonlocal_query_chunk_pixels", type=int, default=128)
    p.add_argument("--nonlocal_convex_iterations", type=int, default=30)
    p.add_argument("--global_convex_iterations", type=int, default=20)
    p.add_argument(
        "--retention_thresholds",
        type=str,
        default="0.90,0.95,0.99",
    )

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    cfg.nonlocal_top_ks = _parse_int_list(cfg.nonlocal_top_ks)
    cfg.retention_thresholds = _parse_float_list(cfg.retention_thresholds)

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


def sweep_retrieved_oracles(
    memory_null: torch.Tensor,
    local_state: torch.Tensor,
    tangent_basis: torch.Tensor,
    target_comp: torch.Tensor,
    max_topk_indices: torch.Tensor,
    null_projector: torch.Tensor,
    k_values: List[int],
    chunk_pixels: int,
    convex_iterations: int,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Evaluate all K prefixes while projecting max-K candidates only once."""
    n, rank, height, width = local_state.shape
    memory_flat = flatten_field(memory_null)
    local_flat = flatten_field(local_state)
    target_flat = flatten_field(target_comp)
    tangent_flat = flatten_tangent(tangent_basis)
    indices_flat = (
        max_topk_indices.permute(0, 2, 3, 1)
        .reshape(n, height * width, max_topk_indices.size(1))
        .contiguous()
    )

    available_k = max_topk_indices.size(1)
    valid_ks = [k for k in k_values if k <= available_k]
    if not valid_ks:
        raise RuntimeError(
            f"No requested K is <= available retrieval count {available_k}"
        )

    hard = {
        k: local_state.new_zeros(n, height * width, rank) for k in valid_ks
    }
    convex = {
        k: local_state.new_zeros(n, height * width, rank) for k in valid_ks
    }

    for batch_index in range(n):
        memory = memory_flat[batch_index]
        for start in range(0, height * width, chunk_pixels):
            stop = min(start + chunk_pixels, height * width)
            max_candidates = gather_complement_candidates(
                memory,
                local_flat[batch_index, start:stop],
                tangent_flat[batch_index, start:stop],
                indices_flat[batch_index, start:stop],
                null_projector,
            )
            target = target_flat[batch_index, start:stop]
            rows = torch.arange(
                max_candidates.size(0), device=max_candidates.device
            )
            for k in valid_ks:
                candidates = max_candidates[:, :k]
                error = (
                    candidates - target.unsqueeze(1)
                ).square().sum(dim=2)
                best_index = error.argmin(dim=1)
                hard[k][batch_index, start:stop] = candidates[
                    rows, best_index
                ]
                convex[k][batch_index, start:stop] = frank_wolfe_convex_hull(
                    candidates,
                    target,
                    convex_iterations,
                )

    return {
        k: (
            unflatten_field(hard[k], height, width),
            unflatten_field(convex[k], height, width),
        )
        for k in valid_ks
    }


def _capture(error: float, target_energy: float) -> float:
    return 1.0 - error / max(target_energy, 1e-30)


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

    max_requested_k = max(cfg.nonlocal_top_ks)
    memory_module = ObservableKeyedComplementMemory(
        top_k=max_requested_k,
        exclusion_radius_lr=cfg.nonlocal_exclusion_radius_lr,
        query_chunk_pixels=cfg.nonlocal_query_chunk_pixels,
        temperature_ratio=1.0,
    ).to(device)

    base_meters = {
        name: MetricAverager()
        for name in [
            "stage2",
            "stage2_gt_complement",
            "global_convex_oracle",
        ]
    }
    sweep_meters: Dict[int, Dict[str, MetricAverager]] = {
        k: {"hard": MetricAverager(), "convex": MetricAverager()}
        for k in cfg.nonlocal_top_ks
    }
    hard_errors = {k: 0.0 for k in cfg.nonlocal_top_ks}
    convex_errors = {k: 0.0 for k in cfg.nonlocal_top_ks}
    global_convex_error = 0.0
    complement_energy = 0.0
    actual_available_k = None

    for batch in test_loader:
        batch = move_to_device(batch, device)
        gt = batch["gt"]
        out = model(batch["lr_hsi"], batch["hr_msi"])
        basis = out["basis"]
        geometry = model.geometry

        gt_coeff = foundation.encode(gt, basis=basis)
        gt_null = geometry.project_null(gt_coeff)
        missing = gt_null - out["null_seed_coefficients"]
        complement_flat = project_complement_vectors(
            flatten_field(missing)[0],
            flatten_tangent(out["tangent_basis"])[0],
            geometry.null_projector,
        )
        complement_target = unflatten_field(
            complement_flat.unsqueeze(0), gt.shape[-2], gt.shape[-1]
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
        actual_available_k = retrieval["topk_indices"].size(1)
        active_ks = [
            k for k in cfg.nonlocal_top_ks if k <= actual_available_k
        ]
        sweep = sweep_retrieved_oracles(
            memory_null=memory_null,
            local_state=local_null_state,
            tangent_basis=out["tangent_basis"],
            target_comp=complement_target,
            max_topk_indices=retrieval["topk_indices"],
            null_projector=geometry.null_projector,
            k_values=active_ks,
            chunk_pixels=cfg.nonlocal_query_chunk_pixels,
            convex_iterations=cfg.nonlocal_convex_iterations,
        )

        _, global_convex = global_oracles(
            memory_null=memory_null,
            local_state=local_null_state,
            tangent_basis=out["tangent_basis"],
            target_comp=complement_target,
            null_projector=geometry.null_projector,
            exclusion_radius_lr=cfg.nonlocal_exclusion_radius_lr,
            chunk_pixels=cfg.nonlocal_query_chunk_pixels,
            convex_iterations=cfg.global_convex_iterations,
        )

        stage2_gt_comp = out["corrected_coefficients"] + complement_target
        global_convex_coeff = out["corrected_coefficients"] + global_convex

        base_meters["stage2"].update(
            calc_metrics(out["reconstructed_hsi"], gt, cfg.scale_ratio)
        )
        base_meters["stage2_gt_complement"].update(
            calc_metrics(
                foundation.decode(stage2_gt_comp, basis=basis),
                gt,
                cfg.scale_ratio,
            )
        )
        base_meters["global_convex_oracle"].update(
            calc_metrics(
                foundation.decode(global_convex_coeff, basis=basis),
                gt,
                cfg.scale_ratio,
            )
        )

        complement_energy += float(
            complement_target.double().square().sum().item()
        )
        global_convex_error += float(
            (
                global_convex.double() - complement_target.double()
            ).square().sum().item()
        )

        for k in active_ks:
            hard, convex = sweep[k]
            hard_coeff = out["corrected_coefficients"] + hard
            convex_coeff = out["corrected_coefficients"] + convex
            sweep_meters[k]["hard"].update(
                calc_metrics(
                    foundation.decode(hard_coeff, basis=basis),
                    gt,
                    cfg.scale_ratio,
                )
            )
            sweep_meters[k]["convex"].update(
                calc_metrics(
                    foundation.decode(convex_coeff, basis=basis),
                    gt,
                    cfg.scale_ratio,
                )
            )
            hard_errors[k] += float(
                (
                    hard.double() - complement_target.double()
                ).square().sum().item()
            )
            convex_errors[k] += float(
                (
                    convex.double() - complement_target.double()
                ).square().sum().item()
            )

    complement_energy = max(complement_energy, 1e-30)
    global_capture = _capture(global_convex_error, complement_energy)
    active_ks = [
        k for k in cfg.nonlocal_top_ks
        if actual_available_k is not None and k <= actual_available_k
    ]

    rows = []
    previous_convex_capture = None
    previous_convex_psnr = None
    for k in active_ks:
        hard_metric = sweep_meters[k]["hard"].average()
        convex_metric = sweep_meters[k]["convex"].average()
        hard_capture = _capture(hard_errors[k], complement_energy)
        convex_capture = _capture(convex_errors[k], complement_energy)
        retention = (
            convex_capture / global_capture
            if global_capture > 0 else float("nan")
        )
        rows.append(
            {
                "k": k,
                "hard": {**hard_metric, "capture": hard_capture},
                "convex": {
                    **convex_metric,
                    "capture": convex_capture,
                    "global_capture_retention": retention,
                    "delta_capture_vs_previous_k": (
                        None
                        if previous_convex_capture is None
                        else convex_capture - previous_convex_capture
                    ),
                    "delta_psnr_vs_previous_k": (
                        None
                        if previous_convex_psnr is None
                        else convex_metric["PSNR"] - previous_convex_psnr
                    ),
                },
            }
        )
        previous_convex_capture = convex_capture
        previous_convex_psnr = convex_metric["PSNR"]

    smallest = {}
    for threshold in cfg.retention_thresholds:
        chosen = next(
            (
                row["k"]
                for row in rows
                if row["convex"]["global_capture_retention"] >= threshold
            ),
            None,
        )
        smallest[f"{threshold:.2f}"] = chosen

    recommended_k = smallest.get("0.95")
    if recommended_k is None:
        recommended_k = rows[-1]["k"] if rows else None

    result = {
        "checkpoint": {
            "foundation": cfg.foundation_checkpoint,
            "local": cfg.local_checkpoint,
            "local_epoch": int(local_epoch),
            "local_best_metric": float(local_best),
        },
        "settings": {
            "requested_k_values": cfg.nonlocal_top_ks,
            "actual_available_k": actual_available_k,
            "exclusion_radius_lr": cfg.nonlocal_exclusion_radius_lr,
            "nonlocal_convex_iterations": cfg.nonlocal_convex_iterations,
            "global_convex_iterations": cfg.global_convex_iterations,
            "selection_rule": (
                "smallest K reaching 95% of global convex P_comp capture"
            ),
        },
        "baseline": {
            "stage2": base_meters["stage2"].average(),
            "stage2_gt_complement": (
                base_meters["stage2_gt_complement"].average()
            ),
            "global_convex_oracle": {
                **base_meters["global_convex_oracle"].average(),
                "capture": global_capture,
            },
        },
        "k_sweep": rows,
        "smallest_k_by_global_capture_retention": smallest,
        "recommended_k_95pct": recommended_k,
    }

    print("=" * 104)
    print("OMN-Net Observable-Key Wide-Recall K Sweep")
    print("=" * 104)
    print(
        f"Loaded Stage2 checkpoint: epoch={local_epoch}, "
        f"stored_best={local_best:.4f}"
    )
    print(
        f"Stage2 actual       : "
        f"PSNR={result['baseline']['stage2']['PSNR']:.4f} "
        f"SAM={result['baseline']['stage2']['SAM']:.4f}"
    )
    print(
        f"Stage2 + GT P_comp  : "
        f"PSNR={result['baseline']['stage2_gt_complement']['PSNR']:.4f}"
    )
    print(
        f"Global convex oracle: "
        f"PSNR={result['baseline']['global_convex_oracle']['PSNR']:.4f} | "
        f"P_comp capture={100.0 * global_capture:.2f}%"
    )
    print("-" * 104)
    print(
        f"{'K':>5} | {'Hard PSNR':>10} | {'Hard cap':>9} | "
        f"{'Convex PSNR':>11} | {'Convex cap':>10} | "
        f"{'Global retained':>15}"
    )
    print("-" * 104)
    for row in rows:
        print(
            f"{row['k']:5d} | "
            f"{row['hard']['PSNR']:10.4f} | "
            f"{100.0 * row['hard']['capture']:8.2f}% | "
            f"{row['convex']['PSNR']:11.4f} | "
            f"{100.0 * row['convex']['capture']:9.2f}% | "
            f"{100.0 * row['convex']['global_capture_retention']:14.2f}%"
        )
    print("-" * 104)
    for threshold in cfg.retention_thresholds:
        key = f"{threshold:.2f}"
        print(
            f"Smallest K retaining {100.0 * threshold:.0f}% "
            f"of Global Convex capture: {smallest[key]}"
        )
    print(f"Recommended minimum K (95% rule): {recommended_k}")

    out_dir = os.path.join(
        cfg.output_root,
        "diagnostics",
        "nonlocal_complement",
        cfg.dataset,
    )
    ensure_dir(out_dir)
    output_path = os.path.join(out_dir, "nonlocal_k_sweep.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
