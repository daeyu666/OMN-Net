"""Inspect the observable/null coefficient ceiling without training."""
from __future__ import annotations

import argparse
import json
import os
import sys

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
    build_spectral_response,
    load_foundation_checkpoint,
)
from utils import ensure_dir, get_device, move_to_device, set_seed


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
    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
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

    meters = {
        name: MetricAverager()
        for name in [
            "anchor",
            "gt_observable_oracle",
            "gt_null_oracle",
            "full_coefficient_oracle",
            "basis_oracle",
        ]
    }
    obs_energy = 0.0
    null_energy = 0.0
    full_vs_basis = 0.0

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
        remaining = gt_coeff - anchor_coeff
        obs = geometry.project_observable(remaining)
        null = geometry.project_null(remaining)

        obs_hsi = foundation.decode(anchor_coeff + obs, basis=basis)
        null_hsi = foundation.decode(anchor_coeff + null, basis=basis)
        full_hsi = foundation.decode(
            anchor_coeff + obs + null, basis=basis
        )
        basis_hsi = foundation.decode(gt_coeff, basis=basis)

        for name, prediction in {
            "anchor": anchor_hsi,
            "gt_observable_oracle": obs_hsi,
            "gt_null_oracle": null_hsi,
            "full_coefficient_oracle": full_hsi,
            "basis_oracle": basis_hsi,
        }.items():
            meters[name].update(
                calc_metrics(prediction, gt, cfg.scale_ratio)
            )
        obs_energy += float(obs.double().square().sum().item())
        null_energy += float(null.double().square().sum().item())
        full_vs_basis = max(
            full_vs_basis,
            float((full_hsi - basis_hsi).abs().max().item()),
        )

    result = {name: meter.average() for name, meter in meters.items()}
    total = max(obs_energy + null_energy, 1e-30)
    result["diagnostics"] = {
        "basis_rank": foundation.basis_rank,
        "observable_rank": int(geometry.observable_rank.item()),
        "null_dimension": (
            foundation.basis_rank - int(geometry.observable_rank.item())
        ),
        "remaining_energy_fraction_observable": obs_energy / total,
        "remaining_energy_fraction_null": null_energy / total,
        "full_vs_basis_max_abs": full_vs_basis,
        **{
            key: float(value.item())
            for key, value in geometry.statistics().items()
        },
    }

    print("=" * 92)
    print("OMN-Net Observability Ceiling")
    print("=" * 92)
    for name in [
        "anchor",
        "gt_observable_oracle",
        "gt_null_oracle",
        "full_coefficient_oracle",
        "basis_oracle",
    ]:
        m = result[name]
        print(
            f"{name:28s}: PSNR={m['PSNR']:.4f} dB, "
            f"SAM={m['SAM']:.4f} deg, RMSE={m['RMSE']:.8f}"
        )
    d = result["diagnostics"]
    print(
        f"rank(S)={d['observable_rank']}/{d['basis_rank']} | "
        f"null_dim={d['null_dimension']} | "
        f"remaining energy obs/null="
        f"{100*d['remaining_energy_fraction_observable']:.2f}%/"
        f"{100*d['remaining_energy_fraction_null']:.2f}%"
    )

    out_dir = os.path.join(
        cfg.output_root, "diagnostics", "observability", cfg.dataset
    )
    ensure_dir(out_dir)
    with open(
        os.path.join(out_dir, "observability_ceiling.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
