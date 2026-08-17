"""GT local tangent-manifold oracle for OMN-Net."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import parse_args
from data_loader import build_loaders
from metrics import MetricAverager, calc_metrics
from models import (
    ObservationGeometry,
    build_local_tangent_field,
    build_spectral_response,
    load_foundation_checkpoint,
)
from utils import ensure_dir, get_device, move_to_device, set_seed


def _parse_dims(text: str) -> List[int]:
    dims = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not dims or dims[0] < 1:
        raise ValueError("tangent dimensions must be positive")
    return dims


def parse_specific_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--foundation_checkpoint",
        type=str,
        default="./checkpoints/spectral_foundation/PaviaU/"
        "foundation_for_local_null.pth",
    )
    p.add_argument("--anchor_ridge_ratio", type=float, default=1e-3)
    p.add_argument("--projector_tolerance", type=float, default=1e-6)
    p.add_argument("--tangent_kernel_size", type=int, default=5)
    p.add_argument("--tangent_dilation", type=int, default=2)
    p.add_argument("--tangent_dimensions", type=str, default="2,4,6,8")
    p.add_argument("--tangent_chunk_pixels", type=int, default=2048)
    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    cfg.tangent_dimensions = _parse_dims(cfg.tangent_dimensions)
    if cfg.dataset != "PaviaU" and "PaviaU" in cfg.foundation_checkpoint:
        cfg.foundation_checkpoint = os.path.join(
            cfg.checkpoint_root,
            "spectral_foundation",
            cfg.dataset,
            "foundation_for_local_null.pth",
        )
    return cfg


@torch.no_grad()
def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    foundation, _ = load_foundation_checkpoint(
        cfg.foundation_checkpoint, info["n_bands"], device
    )
    basis = foundation.get_basis().detach()
    geometry = ObservationGeometry(
        basis,
        build_spectral_response(info).to(device),
        cfg.anchor_ridge_ratio,
        cfg.projector_tolerance,
    ).to(device)

    dims = cfg.tangent_dimensions
    max_dim = max(dims)
    meters = {
        "anchor": MetricAverager(),
        "full_null_oracle": MetricAverager(),
        "basis_oracle": MetricAverager(),
        **{f"d{d}": MetricAverager() for d in dims},
    }
    missing_energy = 0.0
    remaining_energy = {d: 0.0 for d in dims}

    for batch in test_loader:
        batch = move_to_device(batch, device)
        lr_hsi, hr_msi, gt = (
            batch["lr_hsi"],
            batch["hr_msi"],
            batch["gt"],
        )
        lr_coeff = foundation.encode(lr_hsi, basis=basis)
        up = F.interpolate(
            lr_coeff,
            size=gt.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        base_hsi = foundation.decode(up, basis=basis)
        base_msi = geometry.hsi_to_msi(base_hsi)
        anchor_coeff = up + geometry.analytical_residual(
            hr_msi - base_msi
        )
        anchor_hsi = foundation.decode(anchor_coeff, basis=basis)

        gt_coeff = foundation.encode(gt, basis=basis)
        gt_null = geometry.project_null(gt_coeff)
        null_seed = geometry.project_null(up)
        missing = gt_null - null_seed

        tangent_basis, _, _ = build_local_tangent_field(
            null_seed,
            max_dim,
            cfg.tangent_kernel_size,
            cfg.tangent_dilation,
            cfg.tangent_chunk_pixels,
        )

        full_null_hsi = foundation.decode(
            anchor_coeff + missing, basis=basis
        )
        basis_hsi = foundation.decode(gt_coeff, basis=basis)
        meters["anchor"].update(
            calc_metrics(anchor_hsi, gt, cfg.scale_ratio)
        )
        meters["full_null_oracle"].update(
            calc_metrics(full_null_hsi, gt, cfg.scale_ratio)
        )
        meters["basis_oracle"].update(
            calc_metrics(basis_hsi, gt, cfg.scale_ratio)
        )
        missing_energy += float(missing.double().square().sum().item())

        for d in dims:
            tangent = tangent_basis[:, :, :d]
            coords = torch.einsum(
                "nrdhw,nrhw->ndhw", tangent, missing
            )
            projected = torch.einsum(
                "nrdhw,ndhw->nrhw", tangent, coords
            )
            projected = geometry.project_null(projected)
            pred = foundation.decode(
                anchor_coeff + projected, basis=basis
            )
            meters[f"d{d}"].update(
                calc_metrics(pred, gt, cfg.scale_ratio)
            )
            remaining = (missing - projected).double()
            remaining_energy[d] += float(remaining.square().sum().item())

    result = {
        "anchor": meters["anchor"].average(),
        "full_null_oracle": meters["full_null_oracle"].average(),
        "basis_oracle": meters["basis_oracle"].average(),
        "tangent": {},
    }
    missing_energy = max(missing_energy, 1e-30)
    for d in dims:
        m = meters[f"d{d}"].average()
        result["tangent"][str(d)] = {
            **m,
            "missing_null_mse_captured": (
                1.0 - remaining_energy[d] / missing_energy
            ),
            "null_relative_rmse": math.sqrt(
                remaining_energy[d] / missing_energy
            ),
        }

    print("=" * 92)
    print("OMN-Net GT Local Tangent-Manifold Oracle")
    print("=" * 92)
    print(
        f"Anchor: PSNR={result['anchor']['PSNR']:.4f} "
        f"SAM={result['anchor']['SAM']:.4f}"
    )
    for d in dims:
        item = result["tangent"][str(d)]
        print(
            f"d={d:<2d}: PSNR={item['PSNR']:.4f} "
            f"SAM={item['SAM']:.4f} | "
            f"missing capture="
            f"{100*item['missing_null_mse_captured']:.2f}%"
        )
    print(
        f"Full null oracle: "
        f"{result['full_null_oracle']['PSNR']:.4f} dB"
    )

    out_dir = os.path.join(
        cfg.output_root, "diagnostics", "local_tangent", cfg.dataset
    )
    ensure_dir(out_dir)
    with open(
        os.path.join(out_dir, "local_tangent_oracle.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
