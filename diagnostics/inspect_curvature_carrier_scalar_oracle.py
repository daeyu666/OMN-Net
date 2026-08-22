"""E31: LR-HSI curvature-carrier x mid-scale scalar-field oracle.

E29 showed that the rank-6 curvature target can be represented well by an
arbitrary 32-D latent field on the H/2 x W/2 grid, but E30 showed that directly
regressing that 32-D latent is not identifiable from legal inputs.  E31 removes
free spectral latent directions entirely.

Eight signed second-order curvature carriers are constructed only from the
observed LR-HSI null-coefficient field.  They are bilinearly lifted from the LR
spatial grid to the H/2 latent grid.  The only unknowns are eight scalar fields:

    z_mid(p) = sum_i alpha_i(p) q_i^LR->mid(p)
    t_hat    = P_curv Up_2(z_mid)

GT is used only to solve the linear least-squares oracle over alpha and to report
metrics.  No network is trained, HR-MSI never creates a spectral direction, and
all final corrections remain strictly inside P_curv.
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
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.local_curvature_extrapolation import (
    build_curvature_basis,
    build_lr_curvature_bank,
    project_to_curvature,
)
from utils import ensure_dir, get_device, load_checkpoint, move_to_device, set_seed


CURVATURE_VECTOR_COUNT = 8


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

    # H/2 latent grid for the current E29-positive route.
    p.add_argument("--latent_stride", type=int, default=2)
    p.add_argument("--carrier_rms_floor", type=float, default=1e-8)
    p.add_argument("--oracle_ridge", type=float, default=1e-6)
    p.add_argument("--cg_iterations", type=int, default=800)
    p.add_argument("--cg_tolerance", type=float, default=1e-7)
    p.add_argument("--cg_log_interval", type=int, default=100)
    p.add_argument("--e29_stride2_reference_psnr", type=float, default=46.2169)
    p.add_argument(
        "--output_json",
        type=str,
        default="./outputs/diagnostics/e31_curvature_carrier_scalar_oracle.json",
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
    if cfg.latent_stride < 1:
        raise ValueError("latent_stride must be positive")
    if cfg.carrier_rms_floor <= 0:
        raise ValueError("carrier_rms_floor must be positive")
    if cfg.oracle_ridge < 0:
        raise ValueError("oracle_ridge must be non-negative")
    if cfg.cg_iterations < 1 or cfg.cg_log_interval < 1:
        raise ValueError("CG iterations/log interval must be positive")
    if cfg.cg_tolerance <= 0:
        raise ValueError("cg_tolerance must be positive")

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


def _metrics_plain(metrics: Dict[str, float]) -> Dict[str, float]:
    return {str(k).lower(): float(v) for k, v in metrics.items()}


def build_midscale_carriers(
    model: LocalNullManifoldNet,
    stage2: Dict[str, torch.Tensor],
    latent_size: Tuple[int, int],
    rms_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LR-HSI-only carriers [N,8,R,Hmid,Wmid], plus raw per-carrier RMS.

    The bank is created in the observed LR null coefficient field using the same
    eight fixed signed second differences as E16/E17-b.  It is then lifted only
    spatially to the mid-scale grid.  No HR observable feature contributes to a
    carrier's spectral direction.
    """
    memory_null = model.geometry.project_null(stage2["lr_coefficients"])
    bank_lr = build_lr_curvature_bank(memory_null)  # [N,R,8,Hlr,Wlr]
    n, rank, vectors, lr_h, lr_w = bank_lr.shape
    if vectors != CURVATURE_VECTOR_COUNT:
        raise RuntimeError(
            f"expected {CURVATURE_VECTOR_COUNT} LR curvature vectors, got {vectors}"
        )

    bank = bank_lr.permute(0, 2, 1, 3, 4).contiguous()  # [N,8,R,Hlr,Wlr]
    flat = bank.reshape(n, vectors * rank, lr_h, lr_w)
    lifted = F.interpolate(
        flat,
        size=latent_size,
        mode="bilinear",
        align_corners=False,
    ).reshape(n, vectors, rank, latent_size[0], latent_size[1])

    # Per-sample/per-direction RMS normalization is LR-only and preserves span.
    rms = lifted.double().square().mean(dim=(2, 3, 4), keepdim=True).sqrt()
    rms = rms.clamp_min(float(rms_floor)).to(lifted.dtype)
    normalized = lifted / rms
    return normalized.detach(), rms.detach()


def synthesize_mid_field(
    alpha: torch.Tensor,
    carriers: torch.Tensor,
) -> torch.Tensor:
    """sum_i alpha_i q_i -> [N,R,Hmid,Wmid]."""
    if alpha.ndim != 4 or carriers.ndim != 5:
        raise ValueError("alpha/carriers have invalid dimensions")
    if alpha.size(1) != carriers.size(1):
        raise ValueError("alpha carrier count mismatch")
    return torch.einsum("nivw,nirvw->nrvw", alpha, carriers)


def upsample_mid(field: torch.Tensor, hr_size: Tuple[int, int]) -> torch.Tensor:
    if tuple(field.shape[-2:]) == tuple(hr_size):
        return field
    return F.interpolate(field, size=hr_size, mode="bilinear", align_corners=False)


def upsample_adjoint(
    high_field: torch.Tensor,
    mid_shape: Tuple[int, int, int, int],
) -> torch.Tensor:
    """Exact autograd adjoint of bilinear Hmid->H upsampling."""
    if tuple(mid_shape[-2:]) == tuple(high_field.shape[-2:]):
        return high_field
    with torch.enable_grad():
        mid = torch.zeros(
            mid_shape,
            device=high_field.device,
            dtype=high_field.dtype,
            requires_grad=True,
        )
        high = upsample_mid(mid, high_field.shape[-2:])
        grad = torch.autograd.grad(
            high,
            mid,
            grad_outputs=high_field,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
    return grad.detach()


def apply_operator(
    alpha: torch.Tensor,
    carriers: torch.Tensor,
    curvature_basis: torch.Tensor,
    hr_size: Tuple[int, int],
) -> torch.Tensor:
    mid = synthesize_mid_field(alpha, carriers)
    high = upsample_mid(mid, hr_size)
    return project_to_curvature(curvature_basis, high)


def apply_adjoint(
    high_field: torch.Tensor,
    carriers: torch.Tensor,
    curvature_basis: torch.Tensor,
) -> torch.Tensor:
    # P_curv is self-adjoint.
    projected = project_to_curvature(curvature_basis, high_field)
    n, _, mid_h, mid_w = carriers.shape[0], carriers.shape[2], carriers.shape[3], carriers.shape[4]
    mid = upsample_adjoint(
        projected,
        (n, carriers.size(2), mid_h, mid_w),
    )
    return torch.einsum("nirvw,nrvw->nivw", carriers, mid)


def check_adjoint(
    carriers: torch.Tensor,
    curvature_basis: torch.Tensor,
    hr_size: Tuple[int, int],
) -> float:
    alpha = torch.randn(
        carriers.size(0),
        carriers.size(1),
        carriers.size(3),
        carriers.size(4),
        device=carriers.device,
        dtype=carriers.dtype,
    )
    y = torch.randn(
        curvature_basis.size(0),
        curvature_basis.size(1),
        hr_size[0],
        hr_size[1],
        device=curvature_basis.device,
        dtype=curvature_basis.dtype,
    )
    lhs = _dot(apply_operator(alpha, carriers, curvature_basis, hr_size), y)
    rhs = _dot(alpha, apply_adjoint(y, carriers, curvature_basis))
    denom = max(abs(float(lhs.item())), abs(float(rhs.item())), 1e-30)
    return abs(float(lhs.item() - rhs.item())) / denom


def conjugate_gradient(
    normal_operator,
    rhs: torch.Tensor,
    iterations: int,
    tolerance: float,
    log_interval: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    x = torch.zeros_like(rhs)
    r = rhs - normal_operator(x)
    p = r.clone()
    rr = _dot(r, r)
    rhs_norm = max(_norm(rhs), 1e-30)
    initial_rel = math.sqrt(max(float(rr.item()), 0.0)) / rhs_norm
    final_rel = initial_rel
    steps = 0
    breakdown = False

    for step in range(1, int(iterations) + 1):
        ap = normal_operator(p)
        denom = _dot(p, ap)
        denom_value = float(denom.item())
        if not math.isfinite(denom_value) or denom_value <= 0.0:
            breakdown = True
            break
        alpha_step = rr / denom
        x = x + alpha_step.to(x.dtype) * p
        r = r - alpha_step.to(r.dtype) * ap
        rr_new = _dot(r, r)
        final_rel = math.sqrt(max(float(rr_new.item()), 0.0)) / rhs_norm
        steps = step
        if step == 1 or step % int(log_interval) == 0 or final_rel <= tolerance:
            print(f"[E31] CG {step:4d}: relative normal residual={final_rel:.3e}")
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


def solve_oracle(
    target: torch.Tensor,
    carriers: torch.Tensor,
    curvature_basis: torch.Tensor,
    ridge: float,
    iterations: int,
    tolerance: float,
    log_interval: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    hr_size = target.shape[-2:]
    adjoint_error = check_adjoint(carriers, curvature_basis, hr_size)
    rhs = apply_adjoint(target, carriers, curvature_basis)

    def normal(alpha: torch.Tensor) -> torch.Tensor:
        pred = apply_operator(alpha, carriers, curvature_basis, hr_size)
        result = apply_adjoint(pred, carriers, curvature_basis)
        if ridge > 0:
            result = result + float(ridge) * alpha
        return result

    alpha, cg = conjugate_gradient(
        normal,
        rhs,
        iterations=iterations,
        tolerance=tolerance,
        log_interval=log_interval,
    )
    prediction = apply_operator(alpha, carriers, curvature_basis, hr_size)
    return alpha, prediction, {**cg, "adjoint_relative_error": float(adjoint_error)}


def field_stats(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = prediction.double()
    tgt = target.double()
    pe = float(pred.square().sum().item())
    te = float(tgt.square().sum().item())
    err = float((pred - tgt).square().sum().item())
    dot = float((pred * tgt).sum().item())
    te_safe = max(te, 1e-30)
    pe_safe = max(pe, 1e-30)
    return {
        "curvature_capture": float(1.0 - err / te_safe),
        "cosine": float(dot / math.sqrt(pe_safe * te_safe)),
        "amplitude_ratio": float(math.sqrt(pe_safe / te_safe)),
        "target_energy": float(te),
        "prediction_energy": float(pe),
        "error_energy": float(err),
    }


def alpha_stats(alpha: torch.Tensor) -> Dict[str, float]:
    a = alpha.double()
    abs_a = a.abs()
    return {
        "alpha_rms": float(a.square().mean().sqrt().item()),
        "alpha_abs_mean": float(abs_a.mean().item()),
        "alpha_abs_p95": float(torch.quantile(abs_a.reshape(-1), 0.95).item()),
        "alpha_abs_max": float(abs_a.max().item()),
    }


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    model, foundation, local_epoch, local_best = build_local_model(cfg, info, device)

    if len(test_loader) != 1:
        print(f"[warning] expected one test image, got {len(test_loader)}")

    payload = None
    for batch_index, batch in enumerate(test_loader):
        batch = move_to_device(batch, device)
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
            gt_coeff = foundation.encode(batch["gt"], basis=out["basis"])
            remaining = gt_coeff - out["corrected_coefficients"]
            target_curvature = project_to_curvature(curvature_basis, remaining)

        hr_h, hr_w = target_curvature.shape[-2:]
        if hr_h % cfg.latent_stride != 0 or hr_w % cfg.latent_stride != 0:
            raise ValueError(
                f"latent_stride={cfg.latent_stride} does not divide HR size {(hr_h, hr_w)}"
            )
        latent_size = (hr_h // cfg.latent_stride, hr_w // cfg.latent_stride)
        carriers, carrier_rms = build_midscale_carriers(
            model,
            out,
            latent_size=latent_size,
            rms_floor=cfg.carrier_rms_floor,
        )

        alpha, prediction, solver = solve_oracle(
            target_curvature,
            carriers,
            curvature_basis,
            ridge=cfg.oracle_ridge,
            iterations=cfg.cg_iterations,
            tolerance=cfg.cg_tolerance,
            log_interval=cfg.cg_log_interval,
        )

        with torch.no_grad():
            reconstructed = foundation.decode(
                out["corrected_coefficients"] + prediction,
                basis=out["basis"],
            )
            stage2_metrics = _metrics_plain(
                calc_metrics(out["reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
            )
            oracle_hsi = foundation.decode(
                out["corrected_coefficients"] + target_curvature,
                basis=out["basis"],
            )
            curv_oracle_metrics = _metrics_plain(
                calc_metrics(oracle_hsi, batch["gt"], cfg.scale_ratio)
            )
            e31_metrics = _metrics_plain(
                calc_metrics(reconstructed, batch["gt"], cfg.scale_ratio)
            )

        field = field_stats(prediction, target_curvature)
        astat = alpha_stats(alpha)
        stage2_psnr = stage2_metrics["psnr"]
        full_gain = max(curv_oracle_metrics["psnr"] - stage2_psnr, 1e-12)
        e29_gain = max(float(cfg.e29_stride2_reference_psnr) - stage2_psnr, 1e-12)
        full_realize = (e31_metrics["psnr"] - stage2_psnr) / full_gain
        e29_realize = (e31_metrics["psnr"] - stage2_psnr) / e29_gain

        carrier_rms_flat = carrier_rms.squeeze(-1).squeeze(-1).squeeze(-1)
        carrier_rms_list = [float(x) for x in carrier_rms_flat[0].detach().cpu().tolist()]
        mean_valid_rank = float(curvature_valid.float().sum(dim=1).mean().item())

        if solver["adjoint_relative_error"] > 1e-5:
            decision = "INCONCLUSIVE: adjoint check failed; fix operator before interpreting E31."
        elif solver["breakdown"] > 0.5:
            decision = "INCONCLUSIVE: CG breakdown; do not interpret E31 as a representation result."
        elif e31_metrics["psnr"] >= 46.0:
            decision = (
                "STRONG POSITIVE: LR-HSI-only 8-carrier mid-scale scalar fields retain >=46 dB. "
                "Next step should predict only the 8 scalar fields from legal spatial cues."
            )
        elif e31_metrics["psnr"] >= 45.7:
            decision = (
                "MODERATE: the 8-carrier scalar parameterization retains useful headroom but does "
                "not independently cross 46 dB. Quantify scalar-field identifiability before training."
            )
        else:
            decision = (
                "NEGATIVE: the LR-HSI 8-carrier mid-scale scalar parameterization cannot retain "
                "enough of the E29 stride-2 oracle. Do not build a scalar predictor on this basis."
            )

        print("\n=== E31 LR-HSI CURVATURE-CARRIER x MID-SCALE SCALAR ORACLE ===")
        print(
            f"Stage2       : PSNR={stage2_metrics['psnr']:.4f} "
            f"SAM={stage2_metrics['sam']:.4f}"
        )
        print(
            f"CurvOracle   : PSNR={curv_oracle_metrics['psnr']:.4f} "
            f"SAM={curv_oracle_metrics['sam']:.4f}"
        )
        print(f"E29 s2 ref   : PSNR={cfg.e29_stride2_reference_psnr:.4f}")
        print(
            f"E31 8-carrier: PSNR={e31_metrics['psnr']:.4f} "
            f"SAM={e31_metrics['sam']:.4f}"
        )
        print(
            f"CurvCap={100.0*field['curvature_capture']:.2f}% "
            f"Cos={field['cosine']:.4f} Amp={field['amplitude_ratio']:.4f} | "
            f"FullOracleRealize={100.0*full_realize:.2f}% "
            f"E29Realize={100.0*e29_realize:.2f}%"
        )
        print(
            f"latent={latent_size[0]}x{latent_size[1]} scalarDOF="
            f"{CURVATURE_VECTOR_COUNT*latent_size[0]*latent_size[1]} | "
            f"validR={mean_valid_rank:.3f}"
        )
        print(
            f"CG iter={int(solver['iterations'])} rel={solver['final_relative_residual']:.3e} "
            f"breakdown={int(solver['breakdown'])} "
            f"AdjErr={solver['adjoint_relative_error']:.3e}"
        )
        print(
            f"alphaRMS={astat['alpha_rms']:.4f} "
            f"|alpha|mean={astat['alpha_abs_mean']:.4f} "
            f"p95={astat['alpha_abs_p95']:.4f} max={astat['alpha_abs_max']:.4f}"
        )
        print("Carrier LR-only RMS:", ", ".join(f"{x:.4e}" for x in carrier_rms_list))
        print("Decision:")
        print(decision)

        payload = {
            "experiment": "E31_lr_curvature_carrier_midscale_scalar_oracle",
            "batch_index": int(batch_index),
            "dataset": cfg.dataset,
            "scale_ratio": int(cfg.scale_ratio),
            "image_size": int(cfg.diagnostic_image_size),
            "curvature_rank": int(cfg.curvature_rank),
            "latent_stride": int(cfg.latent_stride),
            "latent_size": [int(latent_size[0]), int(latent_size[1])],
            "carrier_count": CURVATURE_VECTOR_COUNT,
            "carrier_rms": carrier_rms_list,
            "oracle_ridge": float(cfg.oracle_ridge),
            "stage2": stage2_metrics,
            "curvature_oracle": curv_oracle_metrics,
            "e29_stride2_reference_psnr": float(cfg.e29_stride2_reference_psnr),
            "e31": e31_metrics,
            "field": field,
            "alpha": astat,
            "solver": solver,
            "mean_valid_curvature_rank": mean_valid_rank,
            "full_oracle_gain_realization": float(full_realize),
            "e29_gain_realization": float(e29_realize),
            "local_checkpoint_epoch": int(local_epoch),
            "local_checkpoint_best": float(local_best),
            "decision": decision,
        }

    if payload is None:
        raise RuntimeError("test loader produced no batch")

    ensure_dir(os.path.dirname(cfg.output_json) or ".")
    with open(cfg.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved: {cfg.output_json}")


if __name__ == "__main__":
    main()
