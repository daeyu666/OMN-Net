"""E21: binary-acceptance and [0,1] trust-region oracle for OMN-Net.

E16-E20 show that the LR-HSI-derived curvature subspace is a strong admissible
space, while the exact HR residual inside that space is only partially
identifiable. E21 asks a narrower question: if the curvature predictor is kept
fixed, how much can be recovered by deciding only whether / how strongly its
already-predicted residual should be written back?

No network is trained in this script.

Two GT-only upper bounds are reported:
1) Binary acceptance oracle:
       m*(p) = 1 if ||e(p)-r(p)||^2 < ||e(p)||^2 else 0
   where e is the Stage-2 remaining coefficient residual and r is the fixed
   predicted curvature residual.
2) [0,1] trust-region oracle:
       alpha*(p) = clip(<r,e> / (||r||^2 + eps), 0, 1)
       r_safe(p) = alpha*(p) r(p)
   This oracle may only keep, shrink or reject the prediction. It may never
   amplify it above 1 or reverse its sign.

Because the spectral basis is orthonormal and r lies in P_curv, coefficient-
space decisions are equivalent to minimizing the reconstruction error along
that authorized direction; all components orthogonal to r are constant with
respect to the scalar decision.

GT is used only to compute the oracle mask/scalar and metrics. It never changes
P_curv, the predicted direction, or the predicted unconstrained amplitude.
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
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.local_curvature_extrapolation import LocalCurvatureExtrapolationNet
from models.local_curvature_extrapolation_e17b import LocalCurvatureExtrapolationE17BNet
from train_local_curvature_extrapolation import build_targets
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
        default=(
            "./checkpoints/local_null_manifold/PaviaU/"
            "local_null_best_psnr.pth"
        ),
    )
    p.add_argument(
        "--curvature_checkpoint",
        type=str,
        default=(
            "./checkpoints/local_curvature_extrapolation_e17b/PaviaU/"
            "curvature_e17b_best_psnr.pth"
        ),
    )
    p.add_argument(
        "--curvature_variant",
        type=str,
        choices=["e17", "e17b"],
        default="e17b",
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
    p.add_argument("--curvature_rank", type=int, default=6)
    p.add_argument("--curvature_svd_chunk_pixels", type=int, default=2048)
    p.add_argument("--curvature_svd_tolerance", type=float, default=1e-5)
    p.add_argument("--curvature_abs_tolerance", type=float, default=1e-9)
    p.add_argument("--curvature_proposal_amplitude_multiplier", type=float, default=8.0)
    p.add_argument("--curvature_predictor_hidden", type=int, default=96)
    p.add_argument("--curvature_predictor_blocks", type=int, default=4)

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"
    if cfg.curvature_rank < 1 or cfg.curvature_rank > 8:
        raise ValueError("curvature_rank must be in [1,8]")
    cfg.image_size = cfg.diagnostic_image_size
    return cfg


def _read_checkpoint_metadata(path: str, device: torch.device) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Curvature checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    return {
        "epoch": int(state.get("epoch", 0)),
        "best_metric": float(state.get("best_metric", 0.0)),
        "extra": state.get("extra", {}) or {},
    }


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
    for parameter in local_model.parameters():
        parameter.requires_grad_(False)

    model_cls = (
        LocalCurvatureExtrapolationNet
        if cfg.curvature_variant == "e17"
        else LocalCurvatureExtrapolationE17BNet
    )
    model = model_cls(
        local_model=local_model,
        curvature_rank=cfg.curvature_rank,
        curvature_svd_chunk_pixels=cfg.curvature_svd_chunk_pixels,
        curvature_svd_tolerance=cfg.curvature_svd_tolerance,
        curvature_abs_tolerance=cfg.curvature_abs_tolerance,
        proposal_amplitude_multiplier=cfg.curvature_proposal_amplitude_multiplier,
        predictor_hidden_channels=cfg.curvature_predictor_hidden,
        predictor_blocks=cfg.curvature_predictor_blocks,
    ).to(device)

    metadata = _read_checkpoint_metadata(cfg.curvature_checkpoint, device)
    checkpoint_rank = metadata["extra"].get("curvature_rank")
    if checkpoint_rank is not None and int(checkpoint_rank) != int(cfg.curvature_rank):
        raise ValueError(
            "Curvature checkpoint rank mismatch: "
            f"checkpoint rank={checkpoint_rank}, requested rank={cfg.curvature_rank}. "
            "The original E17 output path may have been overwritten by the r1/r2 "
            "ablation. Point --curvature_checkpoint to a preserved rank-6 checkpoint "
            "or use the separate E17-b rank-6 checkpoint."
        )
    role = str(metadata["extra"].get("model_role", ""))
    if cfg.curvature_variant == "e17b" and role and "e17b" not in role:
        raise ValueError(
            f"Checkpoint role '{role}' does not look like E17-b; choose the matching "
            "--curvature_variant or checkpoint."
        )
    if cfg.curvature_variant == "e17" and role and "e17b" in role:
        raise ValueError(
            f"Checkpoint role '{role}' is E17-b but --curvature_variant=e17."
        )

    curvature_epoch, curvature_best = load_checkpoint(
        model,
        cfg.curvature_checkpoint,
        optimizer=None,
        strict=True,
        map_location=device,
        load_optimizer=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return (
        model,
        local_epoch,
        local_best,
        curvature_epoch,
        curvature_best,
        metadata,
    )


def binary_acceptance_oracle(
    remaining: torch.Tensor,
    prediction: torch.Tensor,
):
    """Per-pixel keep/reject oracle; prediction direction/amplitude unchanged."""
    rem = remaining.double()
    pred = prediction.double()
    base_error = rem.square().sum(dim=1, keepdim=True)
    pred_error = (rem - pred).square().sum(dim=1, keepdim=True)
    accept = pred_error < base_error
    residual = accept.to(prediction.dtype) * prediction
    return residual, accept, base_error, pred_error


def bounded_trust_oracle(
    remaining: torch.Tensor,
    prediction: torch.Tensor,
):
    """Per-pixel scalar oracle with alpha constrained to [0,1]."""
    rem = remaining.double()
    pred = prediction.double()
    dot = (pred * rem).sum(dim=1, keepdim=True)
    energy = pred.square().sum(dim=1, keepdim=True)
    alpha_unclipped = dot / energy.clamp_min(1e-30)
    alpha = alpha_unclipped.clamp(0.0, 1.0)
    residual = alpha.to(prediction.dtype) * prediction
    return residual, alpha, alpha_unclipped


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    meters = {
        name: MetricAverager()
        for name in [
            "stage2",
            "pred",
            "binary_accept",
            "bounded_trust",
            "curvature_oracle",
            "full_pcomp",
        ]
    }

    pixel_count = 0.0
    accept_count = 0.0
    trust_zero_count = 0.0
    trust_one_count = 0.0
    trust_interior_count = 0.0
    trust_alpha_sum = 0.0
    trust_alpha_sq_sum = 0.0
    unconstrained_negative_count = 0.0
    unconstrained_over_one_count = 0.0

    pred_energy = 0.0
    binary_energy = 0.0
    trust_energy = 0.0
    target_curvature_energy = 0.0
    pred_target_dot = 0.0
    trust_target_dot = 0.0

    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        targets = build_targets(model, out, batch["gt"])

        pred = out["curvature_residual"]
        remaining = targets["remaining"]

        binary_residual, accept, base_error, pred_error = binary_acceptance_oracle(
            remaining, pred
        )
        trust_residual, alpha, alpha_unclipped = bounded_trust_oracle(
            remaining, pred
        )

        binary_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + binary_residual,
            basis=out["basis"],
        )
        trust_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + trust_residual,
            basis=out["basis"],
        )
        curvature_oracle_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + targets["curvature"],
            basis=out["basis"],
        )
        full_pcomp_hsi = model.local_model.foundation.decode(
            out["corrected_coefficients"] + targets["pcomp"],
            basis=out["basis"],
        )

        meters["stage2"].update(
            calc_metrics(out["reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
        )
        meters["pred"].update(
            calc_metrics(
                out["curvature_reconstructed_hsi"], batch["gt"], cfg.scale_ratio
            )
        )
        meters["binary_accept"].update(
            calc_metrics(binary_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["bounded_trust"].update(
            calc_metrics(trust_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["curvature_oracle"].update(
            calc_metrics(curvature_oracle_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["full_pcomp"].update(
            calc_metrics(full_pcomp_hsi, batch["gt"], cfg.scale_ratio)
        )

        count = float(alpha.numel())
        pixel_count += count
        accept_count += float(accept.sum().item())
        trust_zero_count += float((alpha <= 1e-12).sum().item())
        trust_one_count += float((alpha >= 1.0 - 1e-12).sum().item())
        trust_interior_count += float(
            ((alpha > 1e-12) & (alpha < 1.0 - 1e-12)).sum().item()
        )
        trust_alpha_sum += float(alpha.sum().item())
        trust_alpha_sq_sum += float(alpha.square().sum().item())
        unconstrained_negative_count += float((alpha_unclipped < 0.0).sum().item())
        unconstrained_over_one_count += float((alpha_unclipped > 1.0).sum().item())

        pred64 = pred.double()
        binary64 = binary_residual.double()
        trust64 = trust_residual.double()
        target64 = targets["curvature"].double()
        pred_energy += float(pred64.square().sum().item())
        binary_energy += float(binary64.square().sum().item())
        trust_energy += float(trust64.square().sum().item())
        target_curvature_energy += float(target64.square().sum().item())
        pred_target_dot += float((pred64 * target64).sum().item())
        trust_target_dot += float((trust64 * target64).sum().item())

    result: Dict[str, float] = {}
    for name, meter in meters.items():
        for key, value in meter.average().items():
            result[f"{name}_{key.lower()}"] = value

    denom = max(pixel_count, 1.0)
    result["binary_accept_rate"] = accept_count / denom
    result["trust_zero_rate"] = trust_zero_count / denom
    result["trust_one_rate"] = trust_one_count / denom
    result["trust_interior_rate"] = trust_interior_count / denom
    result["trust_alpha_mean"] = trust_alpha_sum / denom
    result["trust_alpha_rms"] = math.sqrt(trust_alpha_sq_sum / denom)
    result["unclipped_negative_rate"] = unconstrained_negative_count / denom
    result["unclipped_over_one_rate"] = unconstrained_over_one_count / denom

    target_energy = max(target_curvature_energy, 1e-30)
    result["pred_curvature_amplitude_ratio"] = math.sqrt(
        pred_energy / target_energy
    )
    result["trust_curvature_amplitude_ratio"] = math.sqrt(
        trust_energy / target_energy
    )
    if pred_energy > 1e-30:
        result["pred_curvature_cosine"] = pred_target_dot / math.sqrt(
            pred_energy * target_energy
        )
    else:
        result["pred_curvature_cosine"] = 0.0
    if trust_energy > 1e-30:
        result["trust_curvature_cosine"] = trust_target_dot / math.sqrt(
            trust_energy * target_energy
        )
    else:
        result["trust_curvature_cosine"] = 0.0
    result["binary_retained_pred_energy"] = binary_energy / max(pred_energy, 1e-30)
    result["trust_retained_pred_energy"] = trust_energy / max(pred_energy, 1e-30)
    return result


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    (
        model,
        local_epoch,
        local_best,
        curvature_epoch,
        curvature_best,
        metadata,
    ) = build_model(cfg, info, device)

    result = evaluate(model, test_loader, cfg, device)

    print(
        "E21 curvature trust oracle | "
        f"variant={cfg.curvature_variant} rank={cfg.curvature_rank} | "
        f"Stage2={result['stage2_psnr']:.4f} "
        f"Pred={result['pred_psnr']:.4f} "
        f"Binary={result['binary_accept_psnr']:.4f} "
        f"Trust01={result['bounded_trust_psnr']:.4f} "
        f"CurvOracle={result['curvature_oracle_psnr']:.4f} "
        f"FullPcomp={result['full_pcomp_psnr']:.4f}"
    )
    print(
        "SAM | "
        f"Stage2={result['stage2_sam']:.4f} "
        f"Pred={result['pred_sam']:.4f} "
        f"Binary={result['binary_accept_sam']:.4f} "
        f"Trust01={result['bounded_trust_sam']:.4f} "
        f"CurvOracle={result['curvature_oracle_sam']:.4f}"
    )
    print(
        "Trust stats | "
        f"accept={100.0*result['binary_accept_rate']:.2f}% "
        f"alpha0={100.0*result['trust_zero_rate']:.2f}% "
        f"alpha(0,1)={100.0*result['trust_interior_rate']:.2f}% "
        f"alpha1={100.0*result['trust_one_rate']:.2f}% "
        f"alpha_mean={result['trust_alpha_mean']:.3f} "
        f"alpha_rms={result['trust_alpha_rms']:.3f}"
    )
    print(
        "Unclipped scalar | "
        f"negative={100.0*result['unclipped_negative_rate']:.2f}% "
        f">1={100.0*result['unclipped_over_one_rate']:.2f}% | "
        f"PredAmp={result['pred_curvature_amplitude_ratio']:.3f} "
        f"PredCos={result['pred_curvature_cosine']:.3f} "
        f"TrustAmp={result['trust_curvature_amplitude_ratio']:.3f} "
        f"TrustCos={result['trust_curvature_cosine']:.3f}"
    )
    print(
        "Retained prediction energy | "
        f"Binary={100.0*result['binary_retained_pred_energy']:.2f}% "
        f"Trust01={100.0*result['trust_retained_pred_energy']:.2f}%"
    )

    out_dir = os.path.join(
        cfg.output_root,
        "diagnostics",
        "curvature_trust_oracle",
        cfg.dataset,
    )
    ensure_dir(out_dir)
    payload = {
        "experiment": "E21 curvature acceptance/trust oracle",
        "dataset": cfg.dataset,
        "curvature_variant": cfg.curvature_variant,
        "curvature_rank": cfg.curvature_rank,
        "curvature_checkpoint": cfg.curvature_checkpoint,
        "checkpoint_epoch": curvature_epoch,
        "checkpoint_best_metric": curvature_best,
        "checkpoint_extra": metadata["extra"],
        "local_epoch": local_epoch,
        "local_best_metric": local_best,
        "rule": {
            "binary": "keep prediction iff per-pixel squared error decreases",
            "trust_01": "alpha=clip(<pred,remaining>/||pred||^2,0,1)",
            "restriction": "keep/shrink/reject only; no amplification or sign reversal",
        },
        "metrics": result,
    }
    path = os.path.join(out_dir, f"curvature_trust_oracle_{cfg.curvature_variant}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
