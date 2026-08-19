"""E16: GT local-curvature complement oracle for OMN-Net.

This no-training diagnostic tests a new hypothesis for innovation point 2:
second-order curvature observed directly in the LR-HSI coefficient manifold may
span useful directions inside each HR query's tangent-complement space.

Important information boundary:
* curvature directions are computed only from the observed LR-HSI coefficient
  field (no bicubic HR interpolation is used to create them);
* HR query geometry contributes only the already validated local tangent and
  P_comp projector;
* GT is used only to project the true Stage-2 remaining P_comp residual onto
  the LR-HSI-derived curvature subspace and report oracle metrics.

The script evaluates requested curvature ranks (default 1,2,4,6) and masks SVD
columns whose singular values are numerically zero, preventing rank-deficient
SVD null columns from acting as free oracle directions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

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
from utils import ensure_dir, get_device, load_checkpoint, move_to_device, set_seed


def _has_option(arguments: List[str], option: str) -> bool:
    return any(x == option or x.startswith(option + "=") for x in arguments)


def _parse_ranks(text: str) -> List[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or values[0] < 1:
        raise ValueError("curvature_ranks must contain positive integers")
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

    p.add_argument("--diagnostic_image_size", type=int, default=128)
    p.add_argument("--curvature_ranks", type=str, default="1,2,4,6")
    p.add_argument("--curvature_svd_chunk_pixels", type=int, default=1024)
    p.add_argument("--curvature_svd_tolerance", type=float, default=1e-5)
    p.add_argument("--curvature_abs_tolerance", type=float, default=1e-9)

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    cfg.curvature_ranks = _parse_ranks(cfg.curvature_ranks)
    if cfg.curvature_svd_chunk_pixels < 1:
        raise ValueError("curvature_svd_chunk_pixels must be positive")
    if cfg.curvature_svd_tolerance <= 0 or cfg.curvature_abs_tolerance <= 0:
        raise ValueError("curvature SVD tolerances must be positive")
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


def build_lr_curvature_bank(memory_null: torch.Tensor) -> torch.Tensor:
    """Return LR-HSI second-difference vectors [N,R,V,Hlr,Wlr].

    Four orientations are measured at radius 1 and radius 2, giving eight
    observed second-order vectors.  Each difference is divided by ||delta||^2
    so orientation/radius changes affect magnitude less strongly while leaving
    the represented subspace unchanged.
    """
    if memory_null.ndim != 4:
        raise ValueError("memory_null must be [N,R,H,W]")
    if min(memory_null.shape[-2:]) <= 2:
        raise ValueError("LR-HSI field is too small for curvature diagnostic")

    offsets = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
        (0, 2),
        (2, 0),
        (2, 2),
        (2, -2),
    ]
    pad = 2
    vectors = []
    for dy, dx in offsets:
        positive = _shift_reflect(memory_null, dy, dx, pad)
        negative = _shift_reflect(memory_null, -dy, -dx, pad)
        denominator = float(dy * dy + dx * dx)
        curvature = (positive + negative - 2.0 * memory_null) / denominator
        vectors.append(curvature)
    return torch.stack(vectors, dim=2)


def map_lr_bank_to_hr(
    bank: torch.Tensor,
    hr_height: int,
    hr_width: int,
) -> torch.Tensor:
    """Nearest-cell map [N,R,V,Hlr,Wlr] -> [N,Q,V,R]."""
    if bank.ndim != 5:
        raise ValueError("curvature bank must be [N,R,V,Hlr,Wlr]")
    n, rank, vectors, lr_h, lr_w = bank.shape
    q_count = hr_height * hr_width
    linear = torch.arange(q_count, device=bank.device)
    y = torch.div(linear, hr_width, rounding_mode="floor")
    x = linear.remainder(hr_width)
    lr_y = torch.floor((y.float() + 0.5) * lr_h / hr_height).long()
    lr_x = torch.floor((x.float() + 0.5) * lr_w / hr_width).long()
    lr_y = lr_y.clamp_(0, lr_h - 1)
    lr_x = lr_x.clamp_(0, lr_w - 1)
    lr_index = lr_y * lr_w + lr_x

    flat = (
        bank.permute(0, 3, 4, 2, 1)
        .reshape(n, lr_h * lr_w, vectors, rank)
        .contiguous()
    )
    return flat[:, lr_index]


def build_projected_curvature_bank(
    model: LocalNullManifoldNet,
    out: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build query-wise LR-HSI curvature vectors inside P_comp: [N,Q,V,R]."""
    geometry = model.geometry
    memory_null = geometry.project_null(out["lr_coefficients"])
    lr_bank = build_lr_curvature_bank(memory_null)
    _, _, hr_h, hr_w = out["corrected_coefficients"].shape
    mapped = map_lr_bank_to_hr(lr_bank, hr_h, hr_w)
    tangent_flat = flatten_tangent(out["tangent_basis"])

    projected_batches = []
    for b in range(mapped.size(0)):
        projected_batches.append(
            project_complement_vectors(
                mapped[b],
                tangent_flat[b],
                geometry.null_projector,
            )
        )
    return torch.stack(projected_batches, dim=0)


def build_stage2_complement_target(
    model: LocalNullManifoldNet,
    out: Dict[str, torch.Tensor],
    gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GT residual remaining after Stage-2, restricted to P_comp."""
    gt_coeff = model.foundation.encode(gt, basis=out["basis"])
    remaining = gt_coeff - out["corrected_coefficients"]
    remaining_flat = flatten_spatial(remaining)
    tangent_flat = flatten_tangent(out["tangent_basis"])
    targets = []
    for b in range(remaining.size(0)):
        targets.append(
            project_complement_vectors(
                remaining_flat[b],
                tangent_flat[b],
                model.geometry.null_projector,
            )
        )
    target_flat = torch.stack(targets, dim=0)
    target = unflatten_spatial(
        target_flat,
        remaining.size(2),
        remaining.size(3),
    )
    return remaining, target


def curvature_oracle_projections(
    projected_bank: torch.Tensor,
    target_comp: torch.Tensor,
    ranks: Sequence[int],
    chunk_pixels: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, float]]:
    """Project GT P_comp target onto LR-HSI curvature subspaces.

    SVD directions below max(relative_tol*s0, absolute_tol) are masked, so a
    requested rank larger than the local numerical curvature rank cannot use
    arbitrary singular-vector completion directions.
    """
    n, q_count, vector_count, coeff_rank = projected_bank.shape
    target_flat = flatten_spatial(target_comp)
    if target_flat.shape != (n, q_count, coeff_rank):
        raise ValueError("curvature bank and target shapes differ")
    if max(ranks) > vector_count:
        raise ValueError(
            f"requested curvature rank {max(ranks)} exceeds vector bank size {vector_count}"
        )

    outputs = {
        int(r): projected_bank.new_zeros(n, q_count, coeff_rank)
        for r in ranks
    }
    numerical_rank_sum = 0.0
    pixels = 0
    rank_ge = {int(r): 0.0 for r in ranks}
    leading_singular_sum = 0.0
    total_singular_sum = 0.0

    for b in range(n):
        for start in range(0, q_count, chunk_pixels):
            stop = min(start + chunk_pixels, q_count)
            # [Q,V,R] -> [Q,R,V]
            matrix = projected_bank[b, start:stop].transpose(1, 2).float()
            target = target_flat[b, start:stop].float()
            u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
            leading = singular[:, :1]
            threshold = torch.maximum(
                leading * float(relative_tolerance),
                singular.new_full(leading.shape, float(absolute_tolerance)),
            )
            valid = singular > threshold
            numerical_rank = valid.sum(dim=1)
            numerical_rank_sum += float(numerical_rank.float().sum().item())
            pixels += numerical_rank.numel()
            leading_singular_sum += float(leading.sum().item())
            total_singular_sum += float(singular.sum().item())

            for rank in ranks:
                r = int(rank)
                basis = u[:, :, :r]
                valid_r = valid[:, :r].to(basis.dtype)
                coordinates = torch.einsum("qri,qr->qi", basis, target)
                coordinates = coordinates * valid_r
                projection = torch.einsum("qri,qi->qr", basis, coordinates)
                outputs[r][b, start:stop] = projection.to(outputs[r].dtype)
                rank_ge[r] += float((numerical_rank >= r).float().sum().item())

    diagnostics = {
        "curvature_vector_count": float(vector_count),
        "mean_numerical_curvature_rank": numerical_rank_sum / max(pixels, 1),
        "mean_leading_singular_value": leading_singular_sum / max(pixels, 1),
        "mean_sum_singular_values": total_singular_sum / max(pixels, 1),
    }
    for rank in ranks:
        diagnostics[f"pixels_rank_ge_{int(rank)}"] = rank_ge[int(rank)] / max(pixels, 1)

    fields = {
        rank: unflatten_spatial(value, target_comp.size(2), target_comp.size(3))
        for rank, value in outputs.items()
    }
    return fields, diagnostics


def _accumulate_energy(acc: Dict[str, float], key: str, value: torch.Tensor):
    acc[key] = acc.get(key, 0.0) + float(value.double().square().sum().item())


@torch.no_grad()
def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    model, foundation, local_epoch, local_best = build_local_model(
        cfg, info, device
    )

    metric_names = ["stage2", "full_pcomp"] + [
        f"curvature_r{rank}" for rank in cfg.curvature_ranks
    ]
    meters = {name: MetricAverager() for name in metric_names}
    energy: Dict[str, float] = {}
    diag_sum: Dict[str, float] = {}
    batches = 0

    for batch in test_loader:
        batch = move_to_device(batch, device)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        required = [
            "basis",
            "lr_coefficients",
            "tangent_basis",
            "corrected_coefficients",
            "reconstructed_hsi",
        ]
        missing_keys = [key for key in required if key not in out]
        if missing_keys:
            raise KeyError(f"LocalNullManifoldNet output missing keys: {missing_keys}")

        remaining, target_comp = build_stage2_complement_target(
            model, out, batch["gt"]
        )
        projected_bank = build_projected_curvature_bank(model, out)
        oracle_fields, diagnostics = curvature_oracle_projections(
            projected_bank,
            target_comp,
            cfg.curvature_ranks,
            cfg.curvature_svd_chunk_pixels,
            cfg.curvature_svd_tolerance,
            cfg.curvature_abs_tolerance,
        )

        full_pcomp_hsi = foundation.decode(
            out["corrected_coefficients"] + target_comp,
            basis=out["basis"],
        )
        meters["stage2"].update(
            calc_metrics(out["reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
        )
        meters["full_pcomp"].update(
            calc_metrics(full_pcomp_hsi, batch["gt"], cfg.scale_ratio)
        )

        _accumulate_energy(energy, "remaining", remaining)
        _accumulate_energy(energy, "target_comp", target_comp)
        for rank, residual in oracle_fields.items():
            prediction = foundation.decode(
                out["corrected_coefficients"] + residual,
                basis=out["basis"],
            )
            meters[f"curvature_r{rank}"].update(
                calc_metrics(prediction, batch["gt"], cfg.scale_ratio)
            )
            error = residual - target_comp
            _accumulate_energy(energy, f"error_r{rank}", error)
            _accumulate_energy(energy, f"oracle_r{rank}", residual)

        for key, value in diagnostics.items():
            diag_sum[key] = diag_sum.get(key, 0.0) + float(value)
        batches += 1

    result: Dict[str, float] = {
        "local_checkpoint_epoch": float(local_epoch),
        "local_checkpoint_best": float(local_best),
    }
    for name, meter in meters.items():
        for key, value in meter.average().items():
            result[f"{name}_{key.lower()}"] = float(value)

    remaining_energy = max(energy.get("remaining", 0.0), 1e-30)
    target_energy = max(energy.get("target_comp", 0.0), 1e-30)
    result["pcomp_share_of_stage2_remaining"] = target_energy / remaining_energy
    for rank in cfg.curvature_ranks:
        error = energy.get(f"error_r{rank}", target_energy)
        oracle_energy = energy.get(f"oracle_r{rank}", 0.0)
        result[f"curvature_r{rank}_pcomp_capture"] = 1.0 - error / target_energy
        result[f"curvature_r{rank}_oracle_energy_ratio"] = oracle_energy / target_energy
        result[f"curvature_r{rank}_pcomp_rrmse"] = math.sqrt(
            max(error / target_energy, 0.0)
        )

    for key, total in diag_sum.items():
        result[key] = total / max(batches, 1)

    out_dir = os.path.join(cfg.output_root, "diagnostics", cfg.dataset)
    ensure_dir(out_dir)
    output_path = os.path.join(
        out_dir, "local_curvature_complement_oracle.json"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print("=== E16 Local Curvature Complement Oracle ===")
    print(
        f"Stage2: PSNR={result['stage2_psnr']:.4f} "
        f"SAM={result['stage2_sam']:.4f}"
    )
    print(
        f"Full GT Pcomp: PSNR={result['full_pcomp_psnr']:.4f} "
        f"SAM={result['full_pcomp_sam']:.4f}"
    )
    print(
        f"Pcomp share of Stage2 remaining energy="
        f"{100.0*result['pcomp_share_of_stage2_remaining']:.2f}% | "
        f"mean curvature numerical rank="
        f"{result['mean_numerical_curvature_rank']:.2f}"
    )
    for rank in cfg.curvature_ranks:
        print(
            f"rank={rank}: PSNR={result[f'curvature_r{rank}_psnr']:.4f} "
            f"SAM={result[f'curvature_r{rank}_sam']:.4f} "
            f"capture={100.0*result[f'curvature_r{rank}_pcomp_capture']:.2f}% "
            f"pixels_rank>={rank}: "
            f"{100.0*result[f'pixels_rank_ge_{rank}']:.1f}%"
        )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
