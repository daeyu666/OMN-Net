"""E28: field-level degradation-closed curvature inversion for OMN-Net.

Zero-training diagnostic for the post-E21 main line.  Instead of predicting a
curvature correction independently at every HR pixel, solve one global linear
inverse problem over the LR-HSI-derived curvature authorization field:

    Delta C(p) = U_curv(p) a(p)
    A(a) = D(U_curv a)

D is the fixed 5x5 Gaussian (sigma=2) + bicubic spatial degradation already
used by OMN-Net physical diagnostics.

E28-A measures whether the true HR curvature target is identifiable from its
own degraded closure, not merely how much raw energy survives downsampling.
E28-B uses b_GT=D(t_curv) as a GT-only target-closure inversion oracle.
E28-C uses only the legal observed LR closure
    b_LR=C_LR-D(C_Stage2).

No predictor checkpoint is needed and no network parameter is updated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import torch

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
    load_foundation_checkpoint,
)
from models.local_curvature_extrapolation import (
    build_curvature_basis,
    project_to_curvature,
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

    # lambda in A^T(AA^T + lambda I)^(-1)b.
    p.add_argument("--inversion_ridge", type=float, default=1e-6)
    p.add_argument("--inversion_iterations", type=int, default=250)
    p.add_argument("--inversion_tolerance", type=float, default=1e-7)
    p.add_argument("--inversion_log_interval", type=int, default=25)
    p.add_argument(
        "--output_json",
        type=str,
        default="./outputs/diagnostics/e28_curvature_field_inversion.json",
    )

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
    if cfg.curvature_svd_tolerance <= 0 or cfg.curvature_abs_tolerance <= 0:
        raise ValueError("curvature SVD tolerances must be positive")
    if cfg.inversion_ridge < 0:
        raise ValueError("inversion_ridge must be non-negative")
    if cfg.inversion_iterations < 1:
        raise ValueError("inversion_iterations must be positive")
    if cfg.inversion_tolerance <= 0:
        raise ValueError("inversion_tolerance must be positive")
    if cfg.inversion_log_interval < 1:
        raise ValueError("inversion_log_interval must be positive")

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


def basis_forward(
    curvature_basis: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """U_curv a: [N,R,D,H,W] x [N,D,H,W] -> [N,R,H,W]."""
    return torch.einsum("nrdhw,ndhw->nrhw", curvature_basis, coordinates)


def basis_adjoint(
    curvature_basis: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """U_curv^T x: [N,R,D,H,W] x [N,R,H,W] -> [N,D,H,W]."""
    return torch.einsum("nrdhw,nrhw->ndhw", curvature_basis, coefficients)


def degradation_adjoint(
    degradation: FixedSpatialDegradation,
    low_field: torch.Tensor,
    high_size: Tuple[int, int],
) -> torch.Tensor:
    """Autograd adjoint of the fixed torch degradation operator."""
    with torch.enable_grad():
        high = torch.zeros(
            low_field.size(0),
            degradation.channels,
            int(high_size[0]),
            int(high_size[1]),
            device=low_field.device,
            dtype=low_field.dtype,
            requires_grad=True,
        )
        degraded = degradation(high, target_size=low_field.shape[-2:])
        grad = torch.autograd.grad(
            outputs=degraded,
            inputs=high,
            grad_outputs=low_field,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
    return grad.detach()


def apply_a(
    curvature_basis: torch.Tensor,
    coordinates: torch.Tensor,
    degradation: FixedSpatialDegradation,
    low_size: Tuple[int, int],
) -> torch.Tensor:
    """A(a)=D(U_curv a)."""
    return degradation(
        basis_forward(curvature_basis, coordinates),
        target_size=low_size,
    )


def apply_at(
    curvature_basis: torch.Tensor,
    low_field: torch.Tensor,
    degradation: FixedSpatialDegradation,
    high_size: Tuple[int, int],
) -> torch.Tensor:
    """A^T(y)=U_curv^T D^T(y)."""
    high_adjoint = degradation_adjoint(
        degradation,
        low_field,
        high_size,
    )
    return basis_adjoint(curvature_basis, high_adjoint)


def _dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.double() * y.double()).sum()


def _norm(x: torch.Tensor) -> float:
    return math.sqrt(max(float(_dot(x, x).item()), 0.0))


def _rms(x: torch.Tensor) -> float:
    return math.sqrt(max(float(x.double().square().mean().item()), 0.0))


def _cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    denom = _norm(x) * _norm(y)
    if denom <= 1e-30:
        return 0.0
    return float(_dot(x, y).item()) / denom


def _capture(target: torch.Tensor, estimate: torch.Tensor) -> float:
    energy = float(_dot(target, target).item())
    if energy <= 1e-30:
        return 0.0
    error = float(_dot(target - estimate, target - estimate).item())
    return 1.0 - error / energy


def solve_lr_space_cg(
    curvature_basis: torch.Tensor,
    rhs: torch.Tensor,
    degradation: FixedSpatialDegradation,
    high_size: Tuple[int, int],
    ridge: float,
    max_iterations: int,
    tolerance: float,
    log_interval: int,
    label: str,
):
    """Solve y=(AA^T+lambda I)^-1 b by CG, then a=A^T y."""
    low_size = tuple(int(x) for x in rhs.shape[-2:])

    def normal_lr(y: torch.Tensor) -> torch.Tensor:
        at_y = apply_at(
            curvature_basis,
            y,
            degradation,
            high_size=high_size,
        )
        value = apply_a(
            curvature_basis,
            at_y,
            degradation,
            low_size=low_size,
        )
        if ridge > 0:
            value = value + float(ridge) * y
        return value

    y = torch.zeros_like(rhs)
    residual = rhs.detach().clone()
    direction = residual.clone()
    rr = _dot(residual, residual)
    rr0 = max(float(rr.item()), 1e-30)
    converged = False
    stop_reason = "max_iterations"
    completed = 0

    for iteration in range(1, max_iterations + 1):
        md = normal_lr(direction)
        denom = _dot(direction, md)
        denom_value = float(denom.item())
        if not math.isfinite(denom_value) or denom_value <= 1e-30:
            stop_reason = "non_positive_curvature"
            break

        alpha = (rr / denom).to(direction.dtype)
        y = y + alpha * direction
        residual = residual - alpha * md
        rr_new = _dot(residual, residual)
        completed = iteration
        relative = math.sqrt(max(float(rr_new.item()), 0.0) / rr0)

        if iteration == 1 or iteration % log_interval == 0:
            print(
                f"[E28-{label}] CG {iteration:04d} "
                f"linear_residual={relative:.6e}"
            )

        if relative <= tolerance:
            rr = rr_new
            converged = True
            stop_reason = "tolerance"
            break
        if not math.isfinite(float(rr_new.item())):
            rr = rr_new
            stop_reason = "non_finite_residual"
            break

        beta = (rr_new / rr.clamp_min(1e-300)).to(direction.dtype)
        direction = residual + beta * direction
        rr = rr_new

    coordinates = apply_at(
        curvature_basis,
        y,
        degradation,
        high_size=high_size,
    )
    high_field = basis_forward(curvature_basis, coordinates)
    fitted_rhs = apply_a(
        curvature_basis,
        coordinates,
        degradation,
        low_size=low_size,
    )
    linear_residual = _norm(rhs - normal_lr(y)) / max(_norm(rhs), 1e-30)
    data_residual = _norm(rhs - fitted_rhs) / max(_norm(rhs), 1e-30)

    info = {
        "iterations": int(completed),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "linear_relative_residual": float(linear_residual),
        "data_relative_residual": float(data_residual),
        "rhs_range_capture": float(_capture(rhs, fitted_rhs)),
        "coordinate_rms": float(_rms(coordinates)),
        "field_rms": float(_rms(high_field)),
    }
    return coordinates.detach(), high_field.detach(), fitted_rhs.detach(), info


def check_adjoint(
    curvature_basis: torch.Tensor,
    degradation: FixedSpatialDegradation,
    low_size: Tuple[int, int],
    seed: int,
) -> Dict[str, float]:
    """Verify numerically that <Aa,y>=<a,A^Ty>."""
    generator = torch.Generator(device=curvature_basis.device)
    generator.manual_seed(int(seed) + 2801)
    n, _, d, h, w = curvature_basis.shape
    a = torch.randn(
        n,
        d,
        h,
        w,
        generator=generator,
        device=curvature_basis.device,
        dtype=curvature_basis.dtype,
    )
    y = torch.randn(
        n,
        degradation.channels,
        int(low_size[0]),
        int(low_size[1]),
        generator=generator,
        device=curvature_basis.device,
        dtype=curvature_basis.dtype,
    )
    aa = apply_a(curvature_basis, a, degradation, low_size)
    aty = apply_at(curvature_basis, y, degradation, (h, w))
    lhs = float(_dot(aa, y).item())
    rhs = float(_dot(a, aty).item())
    relative = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    return {"lhs": lhs, "rhs": rhs, "relative_error": float(relative)}


def residual_metrics(
    foundation,
    out: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    target_curvature: torch.Tensor,
    residual: torch.Tensor,
    scale_ratio: int,
) -> Dict[str, float]:
    reconstructed = foundation.decode(
        out["corrected_coefficients"] + residual,
        basis=out["basis"],
    )
    metrics = calc_metrics(reconstructed, gt, scale_ratio)
    return {
        "psnr": float(metrics["PSNR"]),
        "sam": float(metrics["SAM"]),
        "rmse": float(metrics["RMSE"]),
        "cos": float(_cosine(residual, target_curvature)),
        "curv_cap": float(_capture(target_curvature, residual)),
        "residual_rms": float(_rms(residual)),
    }


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    model, foundation, local_epoch, local_best = build_local_model(
        cfg, info, device
    )

    degradation = FixedSpatialDegradation(
        channels=model.basis_rank,
        kernel_size=5,
        sigma=2.0,
    ).to(device)
    degradation.eval()

    if len(test_loader) != 1:
        print(
            f"[E28] warning: expected one test batch, got {len(test_loader)}; "
            "evaluating the first batch only."
        )

    batch = move_to_device(next(iter(test_loader)), device)

    with torch.no_grad():
        out = model(batch["lr_hsi"], batch["hr_msi"])
        curvature_basis, curvature_singular, curvature_valid = build_curvature_basis(
            model,
            out,
            curvature_rank=cfg.curvature_rank,
            chunk_pixels=cfg.curvature_svd_chunk_pixels,
            relative_tolerance=cfg.curvature_svd_tolerance,
            absolute_tolerance=cfg.curvature_abs_tolerance,
        )
        gt_coefficients = foundation.encode(batch["gt"], basis=out["basis"])
        remaining = gt_coefficients - out["corrected_coefficients"]
        target_curvature = project_to_curvature(curvature_basis, remaining)

        low_size = tuple(int(x) for x in out["lr_coefficients"].shape[-2:])
        high_size = tuple(
            int(x) for x in out["corrected_coefficients"].shape[-2:]
        )

        # E28-A/B: GT only defines the degraded right-hand side.
        b_gt = degradation(target_curvature, target_size=low_size)

        # E28-C: entirely legal observed LR-HSI closure.
        degraded_stage2 = degradation(
            out["corrected_coefficients"], target_size=low_size
        )
        b_lr = out["lr_coefficients"] - degraded_stage2

        # Diagnostic only: quantify torch-D vs actual dataset LR generation.
        degraded_gt_coeff = degradation(gt_coefficients, target_size=low_size)

        stage2_raw = calc_metrics(
            out["reconstructed_hsi"], batch["gt"], cfg.scale_ratio
        )
        curvature_oracle_hsi = foundation.decode(
            out["corrected_coefficients"] + target_curvature,
            basis=out["basis"],
        )
        oracle_raw = calc_metrics(
            curvature_oracle_hsi, batch["gt"], cfg.scale_ratio
        )

    print("\n[E28] checking A/A^T adjoint consistency...")
    adjoint_diag = check_adjoint(
        curvature_basis,
        degradation,
        low_size=low_size,
        seed=cfg.seed,
    )
    print(
        "[E28] adjoint relative error = "
        f"{adjoint_diag['relative_error']:.3e}"
    )

    print("\n[E28-B] solving GT target-closure inversion...")
    _, field_b, fit_b, solver_b = solve_lr_space_cg(
        curvature_basis,
        b_gt,
        degradation,
        high_size,
        cfg.inversion_ridge,
        cfg.inversion_iterations,
        cfg.inversion_tolerance,
        cfg.inversion_log_interval,
        "B",
    )

    print("\n[E28-C] solving legal LR-closure inversion...")
    _, field_c, fit_c, solver_c = solve_lr_space_cg(
        curvature_basis,
        b_lr,
        degradation,
        high_size,
        cfg.inversion_ridge,
        cfg.inversion_iterations,
        cfg.inversion_tolerance,
        cfg.inversion_log_interval,
        "C",
    )

    with torch.no_grad():
        stage2 = {
            "psnr": float(stage2_raw["PSNR"]),
            "sam": float(stage2_raw["SAM"]),
            "rmse": float(stage2_raw["RMSE"]),
        }
        curvature_oracle = {
            "psnr": float(oracle_raw["PSNR"]),
            "sam": float(oracle_raw["SAM"]),
            "rmse": float(oracle_raw["RMSE"]),
        }
        e28b = residual_metrics(
            foundation,
            out,
            batch["gt"],
            target_curvature,
            field_b,
            cfg.scale_ratio,
        )
        e28c = residual_metrics(
            foundation,
            out,
            batch["gt"],
            target_curvature,
            field_c,
            cfg.scale_ratio,
        )

        oracle_gain = max(curvature_oracle["psnr"] - stage2["psnr"], 1e-30)
        e28b["oracle_realize"] = (e28b["psnr"] - stage2["psnr"]) / oracle_gain
        e28c["oracle_realize"] = (e28c["psnr"] - stage2["psnr"]) / oracle_gain

        target_rms = max(_rms(target_curvature), 1e-30)
        bgt_rms = _rms(b_gt)
        blr_rms = _rms(b_lr)
        degradation_error = degraded_gt_coeff - out["lr_coefficients"]
        lr_norm = max(_norm(out["lr_coefficients"]), 1e-30)

        visibility = {
            "target_curvature_rms": float(target_rms),
            "degraded_target_rms": float(bgt_rms),
            "degraded_target_rms_ratio": float(bgt_rms / target_rms),
            "degraded_target_mean_square_ratio": float(
                b_gt.double().square().mean().item()
                / max(
                    float(target_curvature.double().square().mean().item()),
                    1e-30,
                )
            ),
            "closure_visibility": float(_capture(target_curvature, field_b)),
            "minimum_norm_field_ratio": float(
                _norm(field_b) / max(_norm(target_curvature), 1e-30)
            ),
            "b_gt_range_capture": float(_capture(b_gt, fit_b)),
        }

        closure_alignment = {
            "b_lr_b_gt_cosine": float(_cosine(b_lr, b_gt)),
            "b_lr_rms": float(blr_rms),
            "b_gt_rms": float(bgt_rms),
            "b_lr_to_b_gt_rms_ratio": float(
                blr_rms / max(bgt_rms, 1e-30)
            ),
            "b_lr_minus_b_gt_relative_l2": float(
                _norm(b_lr - b_gt) / max(_norm(b_gt), 1e-30)
            ),
            "b_lr_range_capture": float(_capture(b_lr, fit_c)),
        }

        degradation_consistency = {
            "dataset_lr_coeff_relative_l2_error": float(
                _norm(degradation_error) / lr_norm
            ),
            "dataset_lr_coeff_cosine": float(
                _cosine(degraded_gt_coeff, out["lr_coefficients"])
            ),
            "dataset_lr_coeff_error_rms": float(_rms(degradation_error)),
            "dataset_lr_coeff_rms": float(_rms(out["lr_coefficients"])),
        }

        valid_rank = curvature_valid.float().sum(dim=1)
        curvature_geometry = {
            "requested_rank": int(cfg.curvature_rank),
            "mean_valid_rank": float(valid_rank.mean().item()),
            "full_rank_pixel_fraction": float(
                (valid_rank >= float(cfg.curvature_rank)).float().mean().item()
            ),
            "mean_leading_singular": float(
                curvature_singular[:, 0].mean().item()
            ),
        }

    result = {
        "experiment": "E28_curvature_field_degradation_inversion",
        "dataset": cfg.dataset,
        "scale_ratio": int(cfg.scale_ratio),
        "diagnostic_image_size": int(cfg.diagnostic_image_size),
        "msi_mode": cfg.msi_mode,
        "srf_band_set": cfg.srf_band_set,
        "foundation_checkpoint": cfg.foundation_checkpoint,
        "local_checkpoint": cfg.local_checkpoint,
        "local_checkpoint_epoch": int(local_epoch),
        "local_checkpoint_best": float(local_best),
        "inversion": {
            "ridge": float(cfg.inversion_ridge),
            "max_iterations": int(cfg.inversion_iterations),
            "tolerance": float(cfg.inversion_tolerance),
        },
        "curvature_geometry": curvature_geometry,
        "adjoint_check": adjoint_diag,
        "degradation_consistency": degradation_consistency,
        "stage2": stage2,
        "curvature_oracle": curvature_oracle,
        "e28a_visibility": visibility,
        "e28b_target_closure": {**e28b, **solver_b},
        "closure_alignment": closure_alignment,
        "e28c_legal_lr_closure": {**e28c, **solver_c},
    }

    print("\n" + "=" * 78)
    print("E28 FIELD-LEVEL DEGRADATION-CLOSED CURVATURE INVERSION")
    print("=" * 78)
    print(
        f"Stage2       : PSNR={stage2['psnr']:.4f} "
        f"SAM={stage2['sam']:.4f}"
    )
    print(
        f"CurvOracle   : PSNR={curvature_oracle['psnr']:.4f} "
        f"SAM={curvature_oracle['sam']:.4f}"
    )
    print(
        "E28-A        : "
        f"ClosureVisibility={visibility['closure_visibility'] * 100.0:.2f}% "
        f"bGT-RangeCap={visibility['b_gt_range_capture'] * 100.0:.2f}% "
        f"D-RMSratio={visibility['degraded_target_rms_ratio']:.4f}"
    )
    print(
        "E28-B GTclose: "
        f"PSNR={e28b['psnr']:.4f} SAM={e28b['sam']:.4f} "
        f"Cos={e28b['cos']:.4f} CurvCap={e28b['curv_cap'] * 100.0:.2f}% "
        f"OracleRealize={e28b['oracle_realize'] * 100.0:.2f}%"
    )
    print(
        "LR alignment : "
        f"cos(bLR,bGT)={closure_alignment['b_lr_b_gt_cosine']:.4f} "
        f"bLR-RangeCap={closure_alignment['b_lr_range_capture'] * 100.0:.2f}% "
        "LR-D-consistency="
        f"{degradation_consistency['dataset_lr_coeff_cosine']:.6f}"
    )
    print(
        "E28-C legal  : "
        f"PSNR={e28c['psnr']:.4f} SAM={e28c['sam']:.4f} "
        f"Cos={e28c['cos']:.4f} CurvCap={e28c['curv_cap'] * 100.0:.2f}% "
        f"OracleRealize={e28c['oracle_realize'] * 100.0:.2f}%"
    )
    print(
        "CG-B/C fit   : "
        f"{solver_b['rhs_range_capture'] * 100.0:.2f}% / "
        f"{solver_c['rhs_range_capture'] * 100.0:.2f}%"
    )
    print("=" * 78)

    # Do not confuse a solver failure with an information-limit result.
    if solver_b["rhs_range_capture"] < 0.99:
        verdict = (
            "INCONCLUSIVE: E28-B has not fitted b_GT sufficiently. Increase "
            "--inversion_iterations or reduce --inversion_ridge before judging "
            "the degradation-closure hypothesis."
        )
    elif e28b["psnr"] < 45.7:
        verdict = (
            "KILL pure degradation-closed curvature inversion: even the GT "
            "target-closure minimum-norm inverse stays below 45.7 dB."
        )
    elif e28b["psnr"] < 46.0:
        verdict = (
            "BORDERLINE: useful headroom exists but pure closure does not by "
            "itself reach 46 dB. Inspect bLR/bGT alignment and the HR nullspace."
        )
    else:
        verdict = (
            "KEEP: field-level degradation closure has enough intrinsic capacity "
            "to reach 46 dB; legal E28-C regularization is worth developing."
        )
    result["verdict"] = verdict
    print("[E28 verdict]", verdict)

    output_dir = os.path.dirname(cfg.output_json)
    if output_dir:
        ensure_dir(output_dir)
    with open(cfg.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"[E28] saved: {cfg.output_json}")


if __name__ == "__main__":
    main()
