"""E18: no-training observable-curvature witness diagnostic for OMN-Net.

E16 showed that LR-HSI second-order curvature spans a useful rank-6 subspace
inside the Stage-2 tangent complement. E17/E17-b showed that a generic proposal
predictor does not reliably select the correct direction inside that subspace.

E18 asks a narrower physical question: can the observable part of the same
LR-HSI curvature event act as a witness for its hidden P_comp companion?

Information boundary:
* the paired LR curvature bank is computed only from observed LR-HSI;
* its observable component is obtained with P_obs;
* its hidden companion is obtained with P_null (I-P_tan) P_null;
* the HR query witness is computed from the analytical-anchor observable field;
* HR-MSI never generates a hidden coefficient direction;
* the final correction is projected back to the LR-HSI-derived P_curv subspace;
* GT is used only for oracle/diagnostic metrics and never for witness weights.

Primary deterministic rule, direction by direction:
    g_i = <eta_obs_i, kappa_obs_i> /
          (||kappa_obs_i||^2 + ridge)
    r_witness = P_curv sum_i g_i kappa_hidden_i

The LR radius-1/radius-2 curvature events are compared with HR observable
curvatures measured at radius scale_ratio / 2*scale_ratio respectively, so the
spatial displacement represents the same scene-scale neighborhood.
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
from metrics import MetricAverager, calc_metrics
from models import (
    LocalNullManifoldNet,
    build_spectral_response,
    flatten_spatial,
    flatten_tangent,
    load_foundation_checkpoint,
    project_complement_vectors,
    unflatten_spatial,
)
from models.local_curvature_extrapolation import (
    build_curvature_basis,
    build_lr_curvature_bank,
    map_lr_bank_to_hr,
    project_to_curvature,
)
from utils import ensure_dir, get_device, load_checkpoint, move_to_device, set_seed


CURVATURE_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
    (0, 2),
    (2, 0),
    (2, 2),
    (2, -2),
)


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
    p.add_argument("--witness_ridge_ratio", type=float, default=1e-3)
    p.add_argument("--witness_relative_tolerance", type=float, default=1e-5)
    p.add_argument("--witness_abs_tolerance", type=float, default=1e-12)

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
    if cfg.curvature_svd_chunk_pixels < 1:
        raise ValueError("curvature_svd_chunk_pixels must be positive")
    if cfg.witness_ridge_ratio < 0:
        raise ValueError("witness_ridge_ratio must be non-negative")
    if cfg.witness_relative_tolerance <= 0 or cfg.witness_abs_tolerance <= 0:
        raise ValueError("witness tolerances must be positive")

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


def _shift_reflect(x: torch.Tensor, dy: int, dx: int, pad: int) -> torch.Tensor:
    """Reflect-padded integer shift preserving [N,C,H,W]."""
    h, w = x.shape[-2:]
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y0 = pad + int(dy)
    x0 = pad + int(dx)
    return padded[:, :, y0:y0 + h, x0:x0 + w]


def build_hr_matched_curvature_bank(
    coefficients: torch.Tensor,
    scale_ratio: int,
) -> torch.Tensor:
    """HR curvature bank [N,R,8,H,W] at LR-matched physical displacements.

    The numerator uses HR offsets scale_ratio*delta. The denominator remains
    ||delta||^2 in LR-pixel units, so LR and HR banks describe curvature over
    the same scene-scale displacement rather than differing by scale_ratio^2.
    """
    if coefficients.ndim != 4:
        raise ValueError("coefficients must be [N,R,H,W]")
    scale = int(scale_ratio)
    if scale < 1 or abs(float(scale_ratio) - scale) > 1e-8:
        raise ValueError("E18 currently requires an integer scale_ratio")
    max_radius = 2 * scale
    if min(coefficients.shape[-2:]) <= max_radius:
        raise ValueError("HR field is too small for matched curvature offsets")

    vectors = []
    for dy_lr, dx_lr in CURVATURE_OFFSETS:
        dy = dy_lr * scale
        dx = dx_lr * scale
        positive = _shift_reflect(coefficients, dy, dx, max_radius)
        negative = _shift_reflect(coefficients, -dy, -dx, max_radius)
        denominator = float(dy_lr * dy_lr + dx_lr * dx_lr)
        vectors.append(
            (positive + negative - 2.0 * coefficients) / denominator
        )
    return torch.stack(vectors, dim=2)


def bank_to_query_layout(bank: torch.Tensor) -> torch.Tensor:
    """[N,R,V,H,W] -> [N,Q,V,R]."""
    if bank.ndim != 5:
        raise ValueError("bank must be [N,R,V,H,W]")
    n, rank, vectors, h, w = bank.shape
    return (
        bank.permute(0, 3, 4, 2, 1)
        .reshape(n, h * w, vectors, rank)
        .contiguous()
    )


def query_to_coefficients(
    vectors: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """[N,Q,R] -> [N,R,H,W]."""
    return unflatten_spatial(vectors, height, width)


def project_query_vectors(
    projector: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Project [N,Q,V,R] with a global coefficient projector [R,R]."""
    if vectors.ndim != 4:
        raise ValueError("vectors must be [N,Q,V,R]")
    return torch.einsum("rs,nqvs->nqvr", projector.to(vectors), vectors)


def build_paired_witness_banks(
    model: LocalNullManifoldNet,
    stage2: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mapped LR observable witnesses and hidden companions.

    Both originate from the same full LR-HSI second-difference events.
    Shapes are [N,Q,8,R].
    """
    geometry = model.geometry
    lr_full_bank = build_lr_curvature_bank(stage2["lr_coefficients"])
    _, _, h, w = stage2["corrected_coefficients"].shape
    mapped_full = map_lr_bank_to_hr(lr_full_bank, h, w)
    lr_observable = project_query_vectors(
        geometry.observable_projector,
        mapped_full,
    )

    tangent = flatten_tangent(stage2["tangent_basis"])
    hidden_batches = []
    for b in range(mapped_full.size(0)):
        hidden_batches.append(
            project_complement_vectors(
                mapped_full[b],
                tangent[b],
                geometry.null_projector,
            )
        )
    lr_hidden = torch.stack(hidden_batches, dim=0)
    return lr_observable.detach(), lr_hidden.detach()


def build_witness_residual(
    model: LocalNullManifoldNet,
    stage2: Dict[str, torch.Tensor],
    curvature_basis: torch.Tensor,
    scale_ratio: int,
    ridge_ratio: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> Dict[str, torch.Tensor]:
    """Construct the deterministic E18 curvature-complement correction."""
    geometry = model.geometry
    lr_observable, lr_hidden = build_paired_witness_banks(model, stage2)

    # The analytical anchor contains the high-resolution coefficient field whose
    # observable component is fixed by HR-MSI through the Stage-1 SRF anchor.
    hr_observable_coefficients = geometry.project_observable(
        stage2["anchor_coefficients"]
    )
    hr_observable_bank = build_hr_matched_curvature_bank(
        hr_observable_coefficients,
        scale_ratio=scale_ratio,
    )
    hr_observable = bank_to_query_layout(hr_observable_bank)

    lr_energy = lr_observable.double().square().sum(dim=-1)
    hr_energy = hr_observable.double().square().sum(dim=-1)
    dot = (lr_observable.double() * hr_observable.double()).sum(dim=-1)

    max_lr_energy = lr_energy.max(dim=-1, keepdim=True).values
    validity_threshold = torch.maximum(
        max_lr_energy * float(relative_tolerance),
        lr_energy.new_full(max_lr_energy.shape, float(absolute_tolerance)),
    )
    valid = lr_energy > validity_threshold

    mean_lr_energy = (
        (lr_energy * valid.to(lr_energy.dtype)).sum(dim=-1, keepdim=True)
        / valid.to(lr_energy.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
    )
    ridge = float(ridge_ratio) * mean_lr_energy
    gains = dot / (lr_energy + ridge).clamp_min(float(absolute_tolerance))
    gains = torch.where(valid, gains, torch.zeros_like(gains))

    # Observable-only direction agreement is a diagnostic, not a gate.
    cosine = dot / torch.sqrt(
        lr_energy.clamp_min(float(absolute_tolerance))
        * hr_energy.clamp_min(float(absolute_tolerance))
    )
    cosine = torch.where(valid, cosine, torch.zeros_like(cosine))

    raw_query_residual = torch.einsum(
        "nqv,nqvr->nqr",
        gains.to(lr_hidden.dtype),
        lr_hidden,
    )
    _, _, h, w = stage2["corrected_coefficients"].shape
    raw_residual = query_to_coefficients(raw_query_residual, h, w)

    # Rank-6 authorization remains identical to E16/E17/E17-b.
    witness_residual = project_to_curvature(curvature_basis, raw_residual)

    return {
        "lr_observable_bank": lr_observable,
        "lr_hidden_bank": lr_hidden,
        "hr_observable_bank": hr_observable,
        "witness_gains": gains.to(lr_hidden.dtype),
        "witness_valid_mask": valid,
        "witness_observable_cosine": cosine.to(lr_hidden.dtype),
        "raw_witness_residual": raw_residual,
        "witness_residual": witness_residual,
    }


@torch.no_grad()
def build_gt_targets(
    model: LocalNullManifoldNet,
    stage2: Dict[str, torch.Tensor],
    curvature_basis: torch.Tensor,
    gt: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    gt_coefficients = model.foundation.encode(gt, basis=stage2["basis"])
    remaining = gt_coefficients - stage2["corrected_coefficients"]
    remaining_flat = flatten_spatial(remaining)
    tangent_flat = flatten_tangent(stage2["tangent_basis"])

    pcomp_batches = []
    for b in range(remaining.size(0)):
        pcomp_batches.append(
            project_complement_vectors(
                remaining_flat[b],
                tangent_flat[b],
                model.geometry.null_projector,
            )
        )
    pcomp_flat = torch.stack(pcomp_batches, dim=0)
    pcomp = unflatten_spatial(
        pcomp_flat,
        remaining.size(2),
        remaining.size(3),
    )
    curvature = project_to_curvature(curvature_basis, pcomp)
    return {
        "gt_coefficients": gt_coefficients,
        "remaining": remaining,
        "pcomp": pcomp,
        "curvature": curvature,
    }


def pixel_scalar_oracle(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GT best scalar per pixel, preserving only prediction direction."""
    pred64 = prediction.double()
    target64 = target.double()
    dot = (pred64 * target64).sum(dim=1, keepdim=True)
    energy = pred64.square().sum(dim=1, keepdim=True)
    alpha = dot / energy.clamp_min(1e-30)
    return (alpha.to(prediction.dtype) * prediction), alpha


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    meters = {
        name: MetricAverager()
        for name in [
            "stage2",
            "witness",
            "pixel_scalar_oracle",
            "curvature_oracle",
            "full_pcomp",
        ]
    }

    pcomp_energy = 0.0
    witness_pcomp_error = 0.0
    oracle_pcomp_error = 0.0
    curvature_energy = 0.0
    witness_curvature_energy = 0.0
    witness_curvature_dot = 0.0
    witness_curvature_error = 0.0
    witness_gain_abs_sum = 0.0
    witness_gain_square_sum = 0.0
    witness_gain_count = 0.0
    witness_valid_count = 0.0
    witness_total_count = 0.0
    witness_cosine_sum = 0.0
    pixel_alpha_abs_sum = 0.0
    pixel_alpha_count = 0.0
    valid_rank_sum = 0.0
    valid_rank_count = 0.0

    for batch in loader:
        batch = move_to_device(batch, device)
        stage2 = model(batch["lr_hsi"], batch["hr_msi"])
        curvature_basis, curvature_singular, curvature_valid = (
            build_curvature_basis(
                model,
                stage2,
                curvature_rank=cfg.curvature_rank,
                chunk_pixels=cfg.curvature_svd_chunk_pixels,
                relative_tolerance=cfg.curvature_svd_tolerance,
                absolute_tolerance=cfg.curvature_abs_tolerance,
            )
        )
        witness = build_witness_residual(
            model,
            stage2,
            curvature_basis,
            scale_ratio=cfg.scale_ratio,
            ridge_ratio=cfg.witness_ridge_ratio,
            relative_tolerance=cfg.witness_relative_tolerance,
            absolute_tolerance=cfg.witness_abs_tolerance,
        )
        targets = build_gt_targets(
            model,
            stage2,
            curvature_basis,
            batch["gt"],
        )

        pred = witness["witness_residual"]
        target = targets["curvature"]
        corrected = stage2["corrected_coefficients"] + pred
        witness_hsi = model.foundation.decode(corrected, basis=stage2["basis"])

        scalar_residual, pixel_alpha = pixel_scalar_oracle(pred, target)
        pixel_scalar_hsi = model.foundation.decode(
            stage2["corrected_coefficients"] + scalar_residual,
            basis=stage2["basis"],
        )
        curvature_oracle_hsi = model.foundation.decode(
            stage2["corrected_coefficients"] + target,
            basis=stage2["basis"],
        )
        full_pcomp_hsi = model.foundation.decode(
            stage2["corrected_coefficients"] + targets["pcomp"],
            basis=stage2["basis"],
        )

        meters["stage2"].update(
            calc_metrics(stage2["reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
        )
        meters["witness"].update(
            calc_metrics(witness_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["pixel_scalar_oracle"].update(
            calc_metrics(pixel_scalar_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["curvature_oracle"].update(
            calc_metrics(curvature_oracle_hsi, batch["gt"], cfg.scale_ratio)
        )
        meters["full_pcomp"].update(
            calc_metrics(full_pcomp_hsi, batch["gt"], cfg.scale_ratio)
        )

        pcomp_energy += float(targets["pcomp"].double().square().sum().item())
        witness_pcomp_error += float(
            (pred.double() - targets["pcomp"].double()).square().sum().item()
        )
        oracle_pcomp_error += float(
            (target.double() - targets["pcomp"].double()).square().sum().item()
        )
        curvature_energy += float(target.double().square().sum().item())
        witness_curvature_energy += float(pred.double().square().sum().item())
        witness_curvature_dot += float(
            (pred.double() * target.double()).sum().item()
        )
        witness_curvature_error += float(
            (pred.double() - target.double()).square().sum().item()
        )

        valid = witness["witness_valid_mask"]
        gains = witness["witness_gains"].double()
        obs_cos = witness["witness_observable_cosine"].double()
        valid_f = valid.to(gains.dtype)
        witness_gain_abs_sum += float((gains.abs() * valid_f).sum().item())
        witness_gain_square_sum += float((gains.square() * valid_f).sum().item())
        witness_gain_count += float(valid_f.sum().item())
        witness_valid_count += float(valid_f.sum().item())
        witness_total_count += float(valid.numel())
        witness_cosine_sum += float((obs_cos * valid_f).sum().item())

        nonzero_pred = pred.double().square().sum(dim=1, keepdim=True) > 1e-30
        alpha_valid = nonzero_pred.to(pixel_alpha.dtype)
        pixel_alpha_abs_sum += float(
            (pixel_alpha.abs() * alpha_valid).sum().item()
        )
        pixel_alpha_count += float(alpha_valid.sum().item())

        valid_rank_sum += float(
            curvature_valid.float().sum(dim=1).double().sum().item()
        )
        valid_rank_count += float(
            curvature_valid.size(0)
            * curvature_valid.size(2)
            * curvature_valid.size(3)
        )

    result: Dict[str, float] = {}
    for name, meter in meters.items():
        for key, value in meter.average().items():
            result[f"{name}_{key.lower()}"] = float(value)

    pcomp_energy = max(pcomp_energy, 1e-30)
    curvature_energy = max(curvature_energy, 1e-30)
    witness_curvature_energy_safe = max(witness_curvature_energy, 1e-30)

    result["witness_pcomp_capture"] = 1.0 - witness_pcomp_error / pcomp_energy
    result["oracle_pcomp_capture"] = 1.0 - oracle_pcomp_error / pcomp_energy
    result["witness_curvature_capture"] = (
        1.0 - witness_curvature_error / curvature_energy
    )
    result["witness_curvature_amplitude_ratio"] = math.sqrt(
        witness_curvature_energy_safe / curvature_energy
    )
    result["witness_curvature_cosine"] = witness_curvature_dot / math.sqrt(
        witness_curvature_energy_safe * curvature_energy
    )
    result["mean_abs_witness_gain"] = witness_gain_abs_sum / max(
        witness_gain_count, 1.0
    )
    result["rms_witness_gain"] = math.sqrt(
        witness_gain_square_sum / max(witness_gain_count, 1.0)
    )
    result["mean_observable_witness_cosine"] = witness_cosine_sum / max(
        witness_gain_count, 1.0
    )
    result["valid_witness_fraction"] = witness_valid_count / max(
        witness_total_count, 1.0
    )
    result["mean_abs_pixel_optimal_scalar"] = pixel_alpha_abs_sum / max(
        pixel_alpha_count, 1.0
    )
    result["mean_valid_curvature_rank"] = valid_rank_sum / max(
        valid_rank_count, 1.0
    )
    return result


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    model, foundation, local_epoch, local_best = build_local_model(
        cfg, info, device
    )

    result = evaluate(model, test_loader, cfg, device)
    result.update({
        "experiment": "E18 observable curvature witness",
        "dataset": cfg.dataset,
        "foundation_checkpoint": cfg.foundation_checkpoint,
        "local_checkpoint": cfg.local_checkpoint,
        "local_checkpoint_epoch": int(local_epoch),
        "local_checkpoint_best": float(local_best),
        "curvature_rank": int(cfg.curvature_rank),
        "scale_ratio": int(cfg.scale_ratio),
        "witness_ridge_ratio": float(cfg.witness_ridge_ratio),
        "witness_relative_tolerance": float(cfg.witness_relative_tolerance),
        "witness_abs_tolerance": float(cfg.witness_abs_tolerance),
        "gt_usage": "metrics/oracles only; never witness construction",
        "witness_source": (
            "HR analytical-anchor observable curvature matched to paired "
            "LR-HSI observable/hidden curvature events"
        ),
    })

    out_dir = os.path.join(
        cfg.output_root,
        "diagnostics",
        "observable_curvature_witness",
        cfg.dataset,
    )
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "observable_curvature_witness.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print("E18 Observable Curvature Witness")
    print(
        f"Stage2={result['stage2_psnr']:.4f} dB / "
        f"{result['stage2_sam']:.4f} deg"
    )
    print(
        f"Witness={result['witness_psnr']:.4f} dB / "
        f"{result['witness_sam']:.4f} deg"
    )
    print(
        f"Witness + GT pixel scalar={result['pixel_scalar_oracle_psnr']:.4f} dB / "
        f"{result['pixel_scalar_oracle_sam']:.4f} deg"
    )
    print(
        f"Curvature oracle={result['curvature_oracle_psnr']:.4f} dB | "
        f"Full Pcomp={result['full_pcomp_psnr']:.4f} dB"
    )
    print(
        f"CurvCap={100.0 * result['witness_curvature_capture']:.2f}% | "
        f"PcompCap={100.0 * result['witness_pcomp_capture']:.2f}% | "
        f"Amp={result['witness_curvature_amplitude_ratio']:.3f} | "
        f"Cos={result['witness_curvature_cosine']:.3f}"
    )
    print(
        f"ObsWitnessCos={result['mean_observable_witness_cosine']:.3f} | "
        f"valid={100.0 * result['valid_witness_fraction']:.2f}% | "
        f"|g|mean={result['mean_abs_witness_gain']:.3f} | "
        f"g_rms={result['rms_witness_gain']:.3f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
