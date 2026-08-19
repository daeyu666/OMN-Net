"""E19: curvature principal-rank identifiability diagnostic for OMN-Net.

E16 established that the LR-HSI-derived rank-6 curvature subspace contains a
large fraction of the missing Stage-2 tangent-complement residual. E17/E17-b
showed that a generic proposal predictor cannot reliably choose the correct
combination inside that six-dimensional space. E18 further showed that a
simple observable-curvature witness does not resolve the ambiguity.

E19 therefore asks a different question: are the leading curvature directions
substantially more stable / identifiable than the later directions, such that
OMN-Net should authorize only a low-rank trusted curvature subspace rather than
all numerically valid curvature degrees of freedom?

This script is diagnostic only; it trains no network.

For one common rank-D curvature basis (default D=6), it reports:
* Stage-2, marginal direction, cumulative rank-k, rank-D, and full-Pcomp oracle
  reconstruction metrics;
* GT curvature-coordinate energy carried by each principal direction;
* cumulative energy/capture as rank increases;
* simple rank-wise structure / identifiability statistics using only legal
  LR-HSI / HR-MSI / frozen Stage-2 evidence.

Important detail: local SVD bases are sign-ambiguous. Before any signed
coordinate statistic is computed, each curvature basis vector is canonicalized
by forcing its largest-magnitude coefficient pivot to be positive. This changes
neither P_curv nor any oracle reconstruction; it only makes signed statistics
comparable across pixels.

GT is used only to form diagnostic target coordinates and oracle metrics. No GT
quantity is used to construct the curvature basis or any legal evidence field.
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
from models.local_curvature_extrapolation import build_curvature_basis
from models.local_curvature_extrapolation_e17b import (
    build_signed_projected_curvature_bank,
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
    p.add_argument("--max_curvature_rank", type=int, default=6)
    p.add_argument("--curvature_svd_chunk_pixels", type=int, default=2048)
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
    if cfg.max_curvature_rank < 1 or cfg.max_curvature_rank > 8:
        raise ValueError("max_curvature_rank must be in [1,8]")
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


def canonicalize_curvature_basis(basis: torch.Tensor) -> torch.Tensor:
    """Fix per-pixel SVD signs without changing the represented subspace.

    Args:
        basis: [N,R,D,H,W].
    Returns:
        Canonicalized basis with identical projectors.
    """
    if basis.ndim != 5:
        raise ValueError("curvature basis must be [N,R,D,H,W]")
    # [N,D,H,W], coefficient index of largest absolute pivot.
    pivot_index = basis.abs().argmax(dim=1, keepdim=True)
    pivot = torch.gather(basis, dim=1, index=pivot_index).squeeze(1)
    sign = torch.sign(pivot)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return (basis * sign.unsqueeze(1)).contiguous()


def project_with_prefix(
    basis: torch.Tensor,
    coordinates: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Reconstruct the first ``rank`` canonical curvature coordinates."""
    return torch.einsum(
        "nrdhw,ndhw->nrhw",
        basis[:, :, :rank],
        coordinates[:, :rank],
    )


def project_single_direction(
    basis: torch.Tensor,
    coordinates: torch.Tensor,
    direction: int,
) -> torch.Tensor:
    return (
        basis[:, :, direction]
        * coordinates[:, direction].unsqueeze(1)
    )


@torch.no_grad()
def build_gt_targets(
    model: LocalNullManifoldNet,
    stage2: Dict[str, torch.Tensor],
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
    return {
        "gt_coefficients": gt_coefficients,
        "remaining": remaining,
        "pcomp": pcomp,
    }


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Robust Pearson correlation for 1-D CPU tensors."""
    x = x.double().reshape(-1)
    y = y.double().reshape(-1)
    if x.numel() < 2 or y.numel() != x.numel():
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.square().sum() * y.square().sum())
    if float(denom.item()) <= 1e-30:
        return 0.0
    return float((x * y).sum().div(denom).item())


def _lr_cell_structure_stats(
    values: torch.Tensor,
    valid: torch.Tensor,
    lr_h: int,
    lr_w: int,
) -> Tuple[float, float]:
    """How stable is one GT rank coordinate inside each LR-HSI source cell?

    Returns:
        explained_variance: fraction of pixel variance explained by the
            per-LR-cell mean coordinate (1 is perfectly cell-stable).
        sign_consistency: amplitude-weighted |sum a| / sum |a| inside LR cells
            (1 means no sign cancellation inside cells).
    """
    if values.ndim != 3 or valid.shape != values.shape:
        raise ValueError("values/valid must be [N,H,W]")
    n, h, w = values.shape
    q = h * w
    linear = torch.arange(q, device=values.device)
    y = torch.div(linear, w, rounding_mode="floor")
    x = linear.remainder(w)
    lr_y = torch.floor((y.float() + 0.5) * lr_h / h).long().clamp_(0, lr_h - 1)
    lr_x = torch.floor((x.float() + 0.5) * lr_w / w).long().clamp_(0, lr_w - 1)
    index = lr_y * lr_w + lr_x
    cells = lr_h * lr_w

    total_sse = 0.0
    total_sst = 0.0
    total_signed = 0.0
    total_abs = 0.0

    for b in range(n):
        v = values[b].reshape(-1).double()
        m = valid[b].reshape(-1)
        mf = m.double()

        count = torch.zeros(cells, device=v.device, dtype=torch.double)
        cell_sum = torch.zeros_like(count)
        cell_abs = torch.zeros_like(count)
        count.scatter_add_(0, index, mf)
        cell_sum.scatter_add_(0, index, v * mf)
        cell_abs.scatter_add_(0, index, v.abs() * mf)
        cell_mean = cell_sum / count.clamp_min(1.0)
        pred = cell_mean[index]

        if bool(m.any()):
            mean = (v * mf).sum() / mf.sum().clamp_min(1.0)
            total_sse += float(((v - pred).square() * mf).sum().item())
            total_sst += float(((v - mean).square() * mf).sum().item())
        total_signed += float(cell_sum.abs().sum().item())
        total_abs += float(cell_abs.sum().item())

    explained = 1.0 - total_sse / max(total_sst, 1e-30)
    sign_consistency = total_signed / max(total_abs, 1e-30)
    return float(explained), float(sign_consistency)


def _append_valid(
    store: List[torch.Tensor],
    tensor: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    selected = tensor[valid].detach().float().cpu()
    if selected.numel() > 0:
        store.append(selected)


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    dmax = int(cfg.max_curvature_rank)

    stage2_meter = MetricAverager()
    full_pcomp_meter = MetricAverager()
    cumulative_meters = {k: MetricAverager() for k in range(1, dmax + 1)}
    marginal_meters = {i: MetricAverager() for i in range(dmax)}

    pcomp_energy = 0.0
    prefix_errors = [0.0 for _ in range(dmax)]
    direction_energy = [0.0 for _ in range(dmax)]
    valid_count = [0.0 for _ in range(dmax)]
    total_pixel_count = [0.0 for _ in range(dmax)]

    coord_values: List[List[torch.Tensor]] = [[] for _ in range(dmax)]
    sigma_values: List[List[torch.Tensor]] = [[] for _ in range(dmax)]
    drive_mean_values: List[List[torch.Tensor]] = [[] for _ in range(dmax)]
    drive_max_values: List[List[torch.Tensor]] = [[] for _ in range(dmax)]
    msi_norm_values: List[List[torch.Tensor]] = [[] for _ in range(dmax)]
    tangent_norm_values: List[List[torch.Tensor]] = [[] for _ in range(dmax)]

    lr_cell_sse_num = [0.0 for _ in range(dmax)]
    lr_cell_sst_den = [0.0 for _ in range(dmax)]
    lr_cell_sign_num = [0.0 for _ in range(dmax)]
    lr_cell_sign_den = [0.0 for _ in range(dmax)]

    for batch in loader:
        batch = move_to_device(batch, device)
        stage2 = model(batch["lr_hsi"], batch["hr_msi"])
        curvature_basis, curvature_singular, curvature_valid = build_curvature_basis(
            model,
            stage2,
            curvature_rank=dmax,
            chunk_pixels=cfg.curvature_svd_chunk_pixels,
            relative_tolerance=cfg.curvature_svd_tolerance,
            absolute_tolerance=cfg.curvature_abs_tolerance,
        )
        curvature_basis = canonicalize_curvature_basis(curvature_basis)
        targets = build_gt_targets(model, stage2, batch["gt"])

        coordinates = torch.einsum(
            "nrdhw,nrhw->ndhw",
            curvature_basis,
            targets["pcomp"],
        )

        # Legal LR-HSI signed curvature evidence, expressed in the same
        # canonical principal basis. Shape: [N,8,D,H,W].
        signed_bank = build_signed_projected_curvature_bank(model, stage2)
        bank_coordinates = torch.einsum(
            "nrdhw,nvrhw->nvdhw",
            curvature_basis,
            signed_bank,
        )
        drive_mean = bank_coordinates.mean(dim=1)
        max_index = bank_coordinates.abs().argmax(dim=1, keepdim=True)
        drive_max = torch.gather(
            bank_coordinates, dim=1, index=max_index
        ).squeeze(1)

        msi_norm = stage2["msi_residual"].double().square().sum(dim=1).sqrt().float()
        tangent_norm = (
            stage2["tangent_residual"].double().square().sum(dim=1).sqrt().float()
        )

        stage2_meter.update(
            calc_metrics(stage2["reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
        )
        full_pcomp_hsi = model.foundation.decode(
            stage2["corrected_coefficients"] + targets["pcomp"],
            basis=stage2["basis"],
        )
        full_pcomp_meter.update(
            calc_metrics(full_pcomp_hsi, batch["gt"], cfg.scale_ratio)
        )

        batch_pcomp_energy = targets["pcomp"].double().square().sum()
        pcomp_energy += float(batch_pcomp_energy.item())

        for k in range(1, dmax + 1):
            prefix = project_with_prefix(curvature_basis, coordinates, k)
            prefix_hsi = model.foundation.decode(
                stage2["corrected_coefficients"] + prefix,
                basis=stage2["basis"],
            )
            cumulative_meters[k].update(
                calc_metrics(prefix_hsi, batch["gt"], cfg.scale_ratio)
            )
            prefix_errors[k - 1] += float(
                (prefix.double() - targets["pcomp"].double())
                .square().sum().item()
            )

        _, _, lr_h, lr_w = stage2["lr_coefficients"].shape
        for i in range(dmax):
            marginal = project_single_direction(curvature_basis, coordinates, i)
            marginal_hsi = model.foundation.decode(
                stage2["corrected_coefficients"] + marginal,
                basis=stage2["basis"],
            )
            marginal_meters[i].update(
                calc_metrics(marginal_hsi, batch["gt"], cfg.scale_ratio)
            )

            valid = curvature_valid[:, i]
            coord = coordinates[:, i]
            direction_energy[i] += float(
                (coord.double().square() * valid.double()).sum().item()
            )
            valid_count[i] += float(valid.double().sum().item())
            total_pixel_count[i] += float(valid.numel())

            _append_valid(coord_values[i], coord, valid)
            _append_valid(sigma_values[i], curvature_singular[:, i], valid)
            _append_valid(drive_mean_values[i], drive_mean[:, i], valid)
            _append_valid(drive_max_values[i], drive_max[:, i], valid)
            _append_valid(msi_norm_values[i], msi_norm, valid)
            _append_valid(tangent_norm_values[i], tangent_norm, valid)

            # Accumulate LR-cell structure using sufficient statistics so
            # multiple test samples remain valid.
            values = coord
            n, h, w = values.shape
            q = h * w
            linear = torch.arange(q, device=values.device)
            yy = torch.div(linear, w, rounding_mode="floor")
            xx = linear.remainder(w)
            lry = torch.floor((yy.float() + 0.5) * lr_h / h).long().clamp_(0, lr_h - 1)
            lrx = torch.floor((xx.float() + 0.5) * lr_w / w).long().clamp_(0, lr_w - 1)
            cell_index = lry * lr_w + lrx
            cells = lr_h * lr_w
            for b in range(n):
                v = values[b].reshape(-1).double()
                m = valid[b].reshape(-1)
                mf = m.double()
                count = torch.zeros(cells, device=v.device, dtype=torch.double)
                cell_sum = torch.zeros_like(count)
                cell_abs = torch.zeros_like(count)
                count.scatter_add_(0, cell_index, mf)
                cell_sum.scatter_add_(0, cell_index, v * mf)
                cell_abs.scatter_add_(0, cell_index, v.abs() * mf)
                cell_mean = cell_sum / count.clamp_min(1.0)
                pred = cell_mean[cell_index]
                if bool(m.any()):
                    global_mean = (v * mf).sum() / mf.sum().clamp_min(1.0)
                    lr_cell_sse_num[i] += float(
                        ((v - pred).square() * mf).sum().item()
                    )
                    lr_cell_sst_den[i] += float(
                        ((v - global_mean).square() * mf).sum().item()
                    )
                lr_cell_sign_num[i] += float(cell_sum.abs().sum().item())
                lr_cell_sign_den[i] += float(cell_abs.sum().item())

    pcomp_energy = max(pcomp_energy, 1e-30)
    rank6_energy = max(sum(direction_energy), 1e-30)

    result: Dict[str, object] = {
        "stage2": stage2_meter.average(),
        "full_pcomp": full_pcomp_meter.average(),
        "max_curvature_rank": dmax,
        "ranks": [],
    }

    cumulative_energy = 0.0
    for i in range(dmax):
        rank = i + 1
        cumulative_energy += direction_energy[i]
        coord = torch.cat(coord_values[i], dim=0) if coord_values[i] else torch.zeros(1)
        sigma = torch.cat(sigma_values[i], dim=0) if sigma_values[i] else torch.zeros_like(coord)
        drive_mean_i = (
            torch.cat(drive_mean_values[i], dim=0)
            if drive_mean_values[i]
            else torch.zeros_like(coord)
        )
        drive_max_i = (
            torch.cat(drive_max_values[i], dim=0)
            if drive_max_values[i]
            else torch.zeros_like(coord)
        )
        msi_i = (
            torch.cat(msi_norm_values[i], dim=0)
            if msi_norm_values[i]
            else torch.zeros_like(coord)
        )
        tangent_i = (
            torch.cat(tangent_norm_values[i], dim=0)
            if tangent_norm_values[i]
            else torch.zeros_like(coord)
        )

        positive_fraction = float((coord > 0).float().mean().item())
        abs_mean = float(coord.abs().mean().item())
        std = float(coord.double().std(unbiased=False).item())
        explained = 1.0 - lr_cell_sse_num[i] / max(lr_cell_sst_den[i], 1e-30)
        sign_consistency = lr_cell_sign_num[i] / max(lr_cell_sign_den[i], 1e-30)

        cumulative_metrics = cumulative_meters[rank].average()
        marginal_metrics = marginal_meters[i].average()
        rank_entry = {
            "rank": rank,
            "valid_fraction": valid_count[i] / max(total_pixel_count[i], 1.0),
            "gt_coordinate_abs_mean": abs_mean,
            "gt_coordinate_std": std,
            "gt_coordinate_positive_fraction": positive_fraction,
            "direction_energy": direction_energy[i],
            "direction_energy_share_of_rank_d": direction_energy[i] / rank6_energy,
            "cumulative_energy_share_of_rank_d": cumulative_energy / rank6_energy,
            "cumulative_pcomp_capture": 1.0 - prefix_errors[i] / pcomp_energy,
            "marginal_oracle": marginal_metrics,
            "cumulative_oracle": cumulative_metrics,
            "lr_cell_explained_variance": explained,
            "lr_cell_sign_consistency": sign_consistency,
            "corr_abs_gt_vs_curvature_singular": _pearson(coord.abs(), sigma.abs()),
            "corr_signed_gt_vs_lr_curvature_mean_drive": _pearson(
                coord, drive_mean_i
            ),
            "corr_signed_gt_vs_lr_curvature_max_drive": _pearson(
                coord, drive_max_i
            ),
            "corr_abs_gt_vs_hr_msi_residual_norm": _pearson(
                coord.abs(), msi_i.abs()
            ),
            "corr_abs_gt_vs_stage2_tangent_residual_norm": _pearson(
                coord.abs(), tangent_i.abs()
            ),
        }
        result["ranks"].append(rank_entry)

    return result


def _metric(metrics: Dict[str, float], key: str) -> float:
    if key in metrics:
        return float(metrics[key])
    lower = key.lower()
    for name, value in metrics.items():
        if name.lower() == lower:
            return float(value)
    return float("nan")


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    model, _, local_epoch, local_best = build_local_model(cfg, info, device)

    result = evaluate(model, test_loader, cfg, device)
    result.update({
        "dataset": cfg.dataset,
        "local_checkpoint": cfg.local_checkpoint,
        "local_checkpoint_epoch": local_epoch,
        "local_checkpoint_best": local_best,
        "diagnostic_image_size": cfg.diagnostic_image_size,
        "curvature_svd_tolerance": cfg.curvature_svd_tolerance,
        "curvature_abs_tolerance": cfg.curvature_abs_tolerance,
        "interpretation": (
            "Lower ranks are attractive only if they retain useful cumulative "
            "oracle gain while showing materially stronger coordinate structure "
            "or legal-evidence correlation than later ranks."
        ),
    })

    out_dir = os.path.join(
        cfg.output_root,
        "diagnostics",
        "curvature_rank_identifiability",
        cfg.dataset,
    )
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "curvature_rank_identifiability.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    stage2_psnr = _metric(result["stage2"], "psnr")
    full_psnr = _metric(result["full_pcomp"], "psnr")
    print("E19 curvature principal-rank identifiability")
    print(
        f"Stage2={stage2_psnr:.4f} | FullPcomp={full_psnr:.4f} | "
        f"max_rank={cfg.max_curvature_rank}"
    )
    print(
        "rank | marginalPSNR cumulativePSNR PcompCap energyShare cumEnergy | "
        "cellR2 signCons | corr|a|-sigma corr(a,meanDrive) "
        "corr(a,maxDrive) corr|a|-MSI"
    )
    for entry in result["ranks"]:
        print(
            f"{entry['rank']:>4d} | "
            f"{_metric(entry['marginal_oracle'], 'psnr'):.4f} "
            f"{_metric(entry['cumulative_oracle'], 'psnr'):.4f} "
            f"{100.0*entry['cumulative_pcomp_capture']:.2f}% "
            f"{100.0*entry['direction_energy_share_of_rank_d']:.2f}% "
            f"{100.0*entry['cumulative_energy_share_of_rank_d']:.2f}% | "
            f"{entry['lr_cell_explained_variance']:.3f} "
            f"{entry['lr_cell_sign_consistency']:.3f} | "
            f"{entry['corr_abs_gt_vs_curvature_singular']:+.3f} "
            f"{entry['corr_signed_gt_vs_lr_curvature_mean_drive']:+.3f} "
            f"{entry['corr_signed_gt_vs_lr_curvature_max_drive']:+.3f} "
            f"{entry['corr_abs_gt_vs_hr_msi_residual_norm']:+.3f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
