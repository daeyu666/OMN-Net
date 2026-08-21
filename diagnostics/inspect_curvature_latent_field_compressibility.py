"""E29: spatial compressibility oracle for the curvature-authorized residual field.

Post-E28 question
-----------------
E28 showed that the physical LR degradation closure D(U_curv a) is too weak to
identify the HR curvature field, even when its exact GT degraded right-hand side
is supplied.  The next question is therefore not whether LR closure can invert
all HR degrees of freedom, but whether the GT curvature target itself can be
represented by a much lower-resolution *continuous latent coefficient field*.

This script is a zero-training GT-only oracle diagnostic.  For a chosen spatial
stride s, define a coarse latent field

    z_s in R^{R x H/s x W/s}

where R is the global coefficient rank (32 for the current PaviaU experiments).
The field is bilinearly upsampled to HR and only its locally authorized
curvature component is allowed to modify Stage-2:

    z_hr = Upsample(z_s)
    t_hat(p) = P_curv(p) z_hr(p)

No free P_comp component is ever introduced.  The script solves the linear
least-squares oracle

    min_z || P_curv Upsample(z) - t_curv^GT ||_2^2 + lambda ||z||_2^2

with conjugate gradients on the normal equations.  GT is used only as the
oracle target and for metrics.

Why stride=4 matters on the standard 128x128 / scale-4 setting:
    128 / 4 = 32,
so the latent spatial grid has exactly the LR-HSI spatial resolution.  If the
stride-4 oracle retains >=46 dB, then the missing curvature field is spatially
compressible to LR-grid degrees of freedom even though direct LR degradation
closure cannot identify them.  If stride-2 is strong but stride-4 is weak, the
remaining information lives mainly between LR and HR spatial scales and a
future HR-MSI-guided lifting mechanism is justified.  If even stride-2 is weak,
coarse-field recovery should be deprioritized.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import parse_args
from data_loader import build_loaders
from metrics import calc_metrics
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.local_curvature_extrapolation import build_curvature_basis, project_to_curvature
from utils import ensure_dir, get_device, load_checkpoint, move_to_device, set_seed


def _has_option(arguments: List[str], option: str) -> bool:
    return any(x == option or x.startswith(option + "=") for x in arguments)


def _parse_int_list(text: str) -> List[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError("latent strides must be positive integers")
        values.append(value)
    values = sorted(set(values))
    if not values:
        raise ValueError("at least one latent stride is required")
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

    p.add_argument("--latent_strides", type=str, default="1,2,4,8,16")
    p.add_argument("--latent_ridge", type=float, default=1e-6)
    p.add_argument("--cg_iterations", type=int, default=300)
    p.add_argument("--cg_tolerance", type=float, default=1e-7)
    p.add_argument("--cg_log_interval", type=int, default=50)
    p.add_argument(
        "--output_json",
        type=str,
        default="./outputs/diagnostics/e29_curvature_latent_compressibility.json",
    )

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    cfg.latent_strides = _parse_int_list(cfg.latent_strides)
    if cfg.curvature_rank < 1 or cfg.curvature_rank > 8:
        raise ValueError("curvature_rank must be in [1,8]")
    if cfg.curvature_svd_chunk_pixels < 1:
        raise ValueError("curvature_svd_chunk_pixels must be positive")
    if cfg.curvature_svd_tolerance <= 0 or cfg.curvature_abs_tolerance <= 0:
        raise ValueError("curvature SVD tolerances must be positive")
    if cfg.latent_ridge < 0:
        raise ValueError("latent_ridge must be non-negative")
    if cfg.cg_iterations < 1:
        raise ValueError("cg_iterations must be positive")
    if cfg.cg_tolerance <= 0:
        raise ValueError("cg_tolerance must be positive")
    if cfg.cg_log_interval < 1:
        raise ValueError("cg_log_interval must be positive")

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


def _dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.double() * y.double()).sum()


def _norm(x: torch.Tensor) -> float:
    return math.sqrt(max(float(_dot(x, x).item()), 0.0))


def upsample_latent(latent: torch.Tensor, hr_size: Tuple[int, int]) -> torch.Tensor:
    if tuple(latent.shape[-2:]) == tuple(hr_size):
        return latent
    return F.interpolate(
        latent,
        size=hr_size,
        mode="bilinear",
        align_corners=False,
    )


def upsample_adjoint(
    high_field: torch.Tensor,
    latent_shape: Tuple[int, int, int, int],
) -> torch.Tensor:
    """Exact autograd adjoint of the bilinear upsampling used above."""
    if tuple(latent_shape[-2:]) == tuple(high_field.shape[-2:]):
        return high_field

    with torch.enable_grad():
        latent = torch.zeros(
            latent_shape,
            dtype=high_field.dtype,
            device=high_field.device,
            requires_grad=True,
        )
        high = upsample_latent(latent, high_field.shape[-2:])
        grad = torch.autograd.grad(
            outputs=high,
            inputs=latent,
            grad_outputs=high_field,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
    return grad.detach()


def apply_b(
    latent: torch.Tensor,
    curvature_basis: torch.Tensor,
    hr_size: Tuple[int, int],
) -> torch.Tensor:
    """B(z) = P_curv Upsample(z)."""
    high = upsample_latent(latent, hr_size)
    return project_to_curvature(curvature_basis, high)


def apply_bt(
    high_field: torch.Tensor,
    curvature_basis: torch.Tensor,
    latent_shape: Tuple[int, int, int, int],
) -> torch.Tensor:
    """B^T(y) = Upsample^T P_curv y."""
    projected = project_to_curvature(curvature_basis, high_field)
    return upsample_adjoint(projected, latent_shape)


def check_adjoint(
    curvature_basis: torch.Tensor,
    latent_shape: Tuple[int, int, int, int],
    hr_size: Tuple[int, int],
) -> float:
    latent = torch.randn(latent_shape, device=curvature_basis.device, dtype=curvature_basis.dtype)
    high = torch.randn(
        curvature_basis.size(0),
        curvature_basis.size(1),
        hr_size[0],
        hr_size[1],
        device=curvature_basis.device,
        dtype=curvature_basis.dtype,
    )
    lhs = _dot(apply_b(latent, curvature_basis, hr_size), high)
    rhs = _dot(latent, apply_bt(high, curvature_basis, latent_shape))
    denom = max(abs(float(lhs.item())), abs(float(rhs.item())), 1e-30)
    return abs(float(lhs.item() - rhs.item())) / denom


def conjugate_gradient(
    normal_operator,
    rhs: torch.Tensor,
    iterations: int,
    tolerance: float,
    log_interval: int,
    label: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CG solve for a symmetric positive-definite normal equation."""
    x = torch.zeros_like(rhs)
    r = rhs - normal_operator(x)
    p = r.clone()
    rr = _dot(r, r)
    rhs_norm = max(_norm(rhs), 1e-30)
    initial_rel = math.sqrt(max(float(rr.item()), 0.0)) / rhs_norm
    final_rel = initial_rel
    steps = 0
    breakdown = False

    if initial_rel <= tolerance:
        return x, {
            "iterations": 0.0,
            "initial_relative_residual": initial_rel,
            "final_relative_residual": initial_rel,
            "breakdown": 0.0,
        }

    for step in range(1, int(iterations) + 1):
        ap = normal_operator(p)
        denom = _dot(p, ap)
        denom_value = float(denom.item())
        if not math.isfinite(denom_value) or denom_value <= 0.0:
            breakdown = True
            break

        alpha = rr / denom
        x = x + alpha.to(x.dtype) * p
        r = r - alpha.to(r.dtype) * ap
        rr_new = _dot(r, r)
        final_rel = math.sqrt(max(float(rr_new.item()), 0.0)) / rhs_norm
        steps = step

        if step == 1 or step % int(log_interval) == 0 or final_rel <= tolerance:
            print(f"[{label}] CG {step:4d}: relative residual={final_rel:.3e}")

        if final_rel <= tolerance:
            rr = rr_new
            break

        beta = rr_new / rr.clamp_min(1e-300)
        p = r + beta.to(p.dtype) * p
        rr = rr_new

    return x, {
        "iterations": float(steps),
        "initial_relative_residual": float(initial_rel),
        "final_relative_residual": float(final_rel),
        "breakdown": float(breakdown),
    }


def solve_latent_oracle(
    target: torch.Tensor,
    curvature_basis: torch.Tensor,
    stride: int,
    ridge: float,
    iterations: int,
    tolerance: float,
    log_interval: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    n, coeff_rank, hr_h, hr_w = target.shape
    if hr_h % stride != 0 or hr_w % stride != 0:
        raise ValueError(
            f"stride={stride} does not divide HR size {(hr_h, hr_w)} exactly"
        )
    coarse_h = hr_h // stride
    coarse_w = hr_w // stride
    latent_shape = (n, coeff_rank, coarse_h, coarse_w)

    adjoint_error = check_adjoint(
        curvature_basis,
        latent_shape=latent_shape,
        hr_size=(hr_h, hr_w),
    )

    rhs = apply_bt(target, curvature_basis, latent_shape)

    def normal(z: torch.Tensor) -> torch.Tensor:
        bz = apply_b(z, curvature_basis, (hr_h, hr_w))
        result = apply_bt(bz, curvature_basis, latent_shape)
        if ridge > 0:
            result = result + float(ridge) * z
        return result

    latent, cg = conjugate_gradient(
        normal,
        rhs,
        iterations=iterations,
        tolerance=tolerance,
        log_interval=log_interval,
        label=f"s{stride}",
    )
    prediction = apply_b(latent, curvature_basis, (hr_h, hr_w))

    info = {
        **cg,
        "adjoint_relative_error": float(adjoint_error),
        "latent_height": float(coarse_h),
        "latent_width": float(coarse_w),
        "latent_spatial_ratio": float((coarse_h * coarse_w) / (hr_h * hr_w)),
        "latent_scalar_dof": float(coeff_rank * coarse_h * coarse_w),
        "authorized_hr_scalar_dof": float(
            curvature_basis.size(2) * hr_h * hr_w
        ),
    }
    info["latent_to_authorized_dof_ratio"] = (
        info["latent_scalar_dof"] / max(info["authorized_hr_scalar_dof"], 1.0)
    )
    return latent, prediction, info


def field_stats(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred64 = prediction.double()
    target64 = target.double()
    pred_energy = float(pred64.square().sum().item())
    target_energy = float(target64.square().sum().item())
    error_energy = float((pred64 - target64).square().sum().item())
    dot = float((pred64 * target64).sum().item())

    target_energy_safe = max(target_energy, 1e-30)
    pred_energy_safe = max(pred_energy, 1e-30)
    amplitude = math.sqrt(pred_energy_safe / target_energy_safe)
    cosine = dot / math.sqrt(pred_energy_safe * target_energy_safe)
    capture = 1.0 - error_energy / target_energy_safe

    return {
        "amplitude_ratio": float(amplitude),
        "cosine": float(cosine),
        "curvature_capture": float(capture),
        "target_energy": float(target_energy),
        "prediction_energy": float(pred_energy),
        "error_energy": float(error_energy),
    }


def _metrics_to_plain(metrics: Dict[str, float]) -> Dict[str, float]:
    return {str(k).lower(): float(v) for k, v in metrics.items()}


def _decision(results: Dict[int, Dict[str, float]]) -> str:
    s1 = results.get(1)
    if s1 is None:
        return "INCONCLUSIVE: include stride=1 as the solver/oracle sanity check."
    if s1["field_curvature_capture"] < 0.995:
        return (
            "INCONCLUSIVE: stride-1 did not recover >=99.5% of the curvature "
            "target; increase CG accuracy before interpreting spatial compression."
        )

    s4 = results.get(4)
    s2 = results.get(2)
    if s4 is not None and s4["psnr"] >= 46.0:
        return (
            "STRONG POSITIVE: the curvature target remains >=46 dB with an "
            "LR-grid (stride-4) continuous latent field.  Next test should ask "
            "how LR-HSI/HR-MSI can identify that coarse latent field without GT."
        )
    if s4 is not None and s4["psnr"] >= 45.7:
        return (
            "MODERATE: stride-4 retains substantial curvature information but "
            "does not independently cross 46 dB.  Keep only if a legal spatial "
            "lifting cue has a clearly measurable >=0.3 dB ceiling."
        )
    if s2 is not None and s2["psnr"] >= 46.0:
        return (
            "HIGH-FREQUENCY POSITIVE: stride-2 is >=46 dB but stride-4 is weak. "
            "The missing field is compressible to half-resolution, not LR "
            "resolution; a future route must explicitly use HR-MSI to lift "
            "LR spectral geometry into this intermediate spatial scale."
        )
    if s2 is not None and s2["psnr"] >= 45.7:
        return (
            "WEAK/MODERATE: only the half-resolution latent field has useful "
            "headroom.  Do not build a large model yet; first diagnose whether "
            "HR-MSI can legally identify the missing half-scale latent modes."
        )
    return (
        "NEGATIVE: even a GT-optimized stride-2 continuous latent field cannot "
        "reach 45.7 dB.  Spatially compressed latent-field recovery is unlikely "
        "to be the route to the 46 dB target."
    )


@torch.no_grad()
def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    model, foundation, local_epoch, local_best = build_local_model(
        cfg, info, device
    )

    if len(test_loader) != 1:
        print(
            f"[warning] expected one deterministic test image, got {len(test_loader)}; "
            "results will be reported per sample and the final JSON stores the last one."
        )

    final_payload = None
    for batch_index, batch in enumerate(test_loader):
        batch = move_to_device(batch, device)
        out = model(batch["lr_hsi"], batch["hr_msi"])

        curvature_basis, curvature_singular, curvature_valid = build_curvature_basis(
            model,
            out,
            curvature_rank=cfg.curvature_rank,
            chunk_pixels=cfg.curvature_svd_chunk_pixels,
            relative_tolerance=cfg.curvature_svd_tolerance,
            absolute_tolerance=cfg.curvature_abs_tolerance,
        )

        gt_coeff = foundation.encode(batch["gt"], basis=out["basis"])
        remaining = gt_coeff - out["corrected_coefficients"]
        target_curvature = project_to_curvature(curvature_basis, remaining)

        stage2_hsi = out["reconstructed_hsi"]
        curvature_oracle_hsi = foundation.decode(
            out["corrected_coefficients"] + target_curvature,
            basis=out["basis"],
        )
        stage2_metrics = _metrics_to_plain(
            calc_metrics(stage2_hsi, batch["gt"], cfg.scale_ratio)
        )
        oracle_metrics = _metrics_to_plain(
            calc_metrics(curvature_oracle_hsi, batch["gt"], cfg.scale_ratio)
        )

        mean_valid_rank = float(
            curvature_valid.float().sum(dim=1).mean().item()
        )
        mean_singular = float(curvature_singular.mean().item())

        print("\n=== E29 CURVATURE LATENT-FIELD SPATIAL COMPRESSIBILITY ===")
        print(
            f"Stage2     : PSNR={stage2_metrics['psnr']:.4f} "
            f"SAM={stage2_metrics['sam']:.4f}"
        )
        print(
            f"CurvOracle : PSNR={oracle_metrics['psnr']:.4f} "
            f"SAM={oracle_metrics['sam']:.4f}"
        )
        print(
            f"Curvature rank requested={cfg.curvature_rank}, "
            f"mean valid={mean_valid_rank:.3f}, mean singular={mean_singular:.6e}"
        )
        print(
            "\nstride | latent grid | PSNR    SAM     CurvCap   Cos      Amp      "
            "CGrel      AdjErr     DOFratio"
        )
        print("-" * 102)

        results: Dict[int, Dict[str, float]] = {}
        hr_h, hr_w = target_curvature.shape[-2:]
        stage2_psnr = stage2_metrics["psnr"]
        oracle_gain = max(oracle_metrics["psnr"] - stage2_psnr, 1e-12)

        for stride in cfg.latent_strides:
            _, prediction, solver = solve_latent_oracle(
                target_curvature,
                curvature_basis,
                stride=int(stride),
                ridge=cfg.latent_ridge,
                iterations=cfg.cg_iterations,
                tolerance=cfg.cg_tolerance,
                log_interval=cfg.cg_log_interval,
            )
            reconstructed = foundation.decode(
                out["corrected_coefficients"] + prediction,
                basis=out["basis"],
            )
            metrics = _metrics_to_plain(
                calc_metrics(reconstructed, batch["gt"], cfg.scale_ratio)
            )
            field = field_stats(prediction, target_curvature)
            oracle_realize = (metrics["psnr"] - stage2_psnr) / oracle_gain

            record = {
                **metrics,
                **{f"field_{k}": v for k, v in field.items()},
                **{f"solver_{k}": v for k, v in solver.items()},
                "oracle_gain_realization": float(oracle_realize),
                "stride": float(stride),
                "hr_height": float(hr_h),
                "hr_width": float(hr_w),
            }
            results[int(stride)] = record

            print(
                f"{stride:>6d} | "
                f"{int(solver['latent_height']):>3d}x{int(solver['latent_width']):<3d} | "
                f"{metrics['psnr']:>7.4f} "
                f"{metrics['sam']:>7.4f} "
                f"{100.0 * field['curvature_capture']:>7.2f}% "
                f"{field['cosine']:>7.4f} "
                f"{field['amplitude_ratio']:>7.4f} "
                f"{solver['final_relative_residual']:>9.2e} "
                f"{solver['adjoint_relative_error']:>9.2e} "
                f"{100.0 * solver['latent_to_authorized_dof_ratio']:>7.2f}%"
            )

        decision = _decision(results)
        print("\nDecision:")
        print(decision)

        if 4 in results:
            r4 = results[4]
            print(
                "\nStride-4 focus (LR-grid on 128/4 setting): "
                f"PSNR={r4['psnr']:.4f}, "
                f"CurvCap={100.0*r4['field_curvature_capture']:.2f}%, "
                f"OracleRealize={100.0*r4['oracle_gain_realization']:.2f}%"
            )
        if 2 in results:
            r2 = results[2]
            print(
                "Stride-2 focus (half-resolution latent): "
                f"PSNR={r2['psnr']:.4f}, "
                f"CurvCap={100.0*r2['field_curvature_capture']:.2f}%, "
                f"OracleRealize={100.0*r2['oracle_gain_realization']:.2f}%"
            )

        final_payload = {
            "experiment": "E29_curvature_latent_field_spatial_compressibility",
            "batch_index": int(batch_index),
            "dataset": cfg.dataset,
            "scale_ratio": int(cfg.scale_ratio),
            "image_size": int(cfg.diagnostic_image_size),
            "curvature_rank": int(cfg.curvature_rank),
            "latent_strides": [int(x) for x in cfg.latent_strides],
            "latent_ridge": float(cfg.latent_ridge),
            "stage2": stage2_metrics,
            "curvature_oracle": oracle_metrics,
            "mean_valid_curvature_rank": mean_valid_rank,
            "mean_curvature_singular_value": mean_singular,
            "local_checkpoint_epoch": int(local_epoch),
            "local_checkpoint_best": float(local_best),
            "results": {str(k): v for k, v in results.items()},
            "decision": decision,
        }

    if final_payload is None:
        raise RuntimeError("test loader produced no batches")

    ensure_dir(os.path.dirname(cfg.output_json) or ".")
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {cfg.output_json}")


if __name__ == "__main__":
    main()
