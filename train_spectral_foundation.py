"""Train OMN-Net spectral foundation from LR-HSI only."""
from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from models.spectral_foundation import SpectralFoundation
from utils import (
    AverageMeter,
    CSVLogger,
    ensure_dir,
    get_device,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


def first_spectral_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def second_spectral_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


@torch.no_grad()
def collect_lr_spectra(loader: Iterable, max_pixels: int) -> torch.Tensor:
    collected: List[torch.Tensor] = []
    total = 0
    for batch in loader:
        lr_hsi = batch["lr_hsi"].float()
        pixels = lr_hsi.permute(0, 2, 3, 1).reshape(-1, lr_hsi.size(1))
        valid = (
            torch.isfinite(pixels).all(dim=1)
            & (pixels.abs().mean(dim=1) > 1e-6)
        )
        pixels = pixels[valid]
        if pixels.numel() == 0:
            continue
        remaining = max_pixels - total
        if pixels.size(0) > remaining:
            indices = torch.randperm(pixels.size(0))[:remaining]
            pixels = pixels[indices]
        collected.append(pixels.cpu())
        total += pixels.size(0)
        if total >= max_pixels:
            break
    if not collected:
        raise RuntimeError("No valid LR-HSI spectra found for PCA")
    return torch.cat(collected, dim=0)


@torch.no_grad()
def compute_pca_initialization(
    spectra: torch.Tensor, basis_rank: int
) -> Dict[str, torch.Tensor]:
    data = spectra.double()
    mean = data.mean(dim=0)
    centered = data - mean
    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / max(data.size(0) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    basis = eigenvectors[:, :basis_rank]
    retained = eigenvalues[:basis_rank]
    total_variance = eigenvalues.sum()
    return {
        "mean_spectrum": mean.float(),
        "basis": basis.float(),
        "coefficient_scale": retained.sqrt().clamp_min(1e-8).float(),
        "eigenvalues": retained.float(),
        "total_variance": total_variance.float(),
        "explained_variance_ratio": (
            retained.sum() / total_variance.clamp_min(1e-12)
        ).float(),
    }


def parse_specific_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--basis_rank", type=int, default=32)
    parser.add_argument("--basis_init_pixels", type=int, default=100000)
    parser.add_argument("--basis_grad_clip", type=float, default=1.0)
    parser.add_argument("--basis_lambda_sam", type=float, default=0.5)
    parser.add_argument("--basis_lambda_sgrad1", type=float, default=0.1)
    parser.add_argument("--basis_lambda_sgrad2", type=float, default=0.05)
    parser.add_argument("--basis_lambda_anchor", type=float, default=0.001)
    parser.add_argument("--basis_selection_sam", type=float, default=1.0)
    parser.add_argument("--basis_selection_sgrad1", type=float, default=0.2)
    parser.add_argument("--basis_selection_sgrad2", type=float, default=0.1)
    specific, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    return cfg


def compute_losses(model, out, target, sam_loss, cfg):
    rec = out["reconstruction"]
    l1 = F.l1_loss(rec, target)
    mse = F.mse_loss(rec, target)
    sam = sam_loss(rec, target)
    sg1 = F.l1_loss(
        first_spectral_difference(rec),
        first_spectral_difference(target),
    )
    sg2 = F.l1_loss(
        second_spectral_difference(rec),
        second_spectral_difference(target),
    )
    ref = model.get_reference_projector().to(out["projector"])
    anchor = F.mse_loss(out["projector"], ref)
    total = (
        cfg.lambda_l1 * l1
        + cfg.basis_lambda_sam * sam
        + cfg.basis_lambda_sgrad1 * sg1
        + cfg.basis_lambda_sgrad2 * sg2
        + cfg.basis_lambda_anchor * anchor
    )
    return {
        "total": total,
        "l1": l1,
        "mse": mse,
        "sam": sam,
        "sgrad1": sg1,
        "sgrad2": sg2,
        "anchor": anchor,
    }


def run_epoch(model, loader, sam_loss, cfg, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    names = ["total", "l1", "mse", "sam", "sgrad1", "sgrad2", "anchor"]
    meters = {name: AverageMeter() for name in names}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = move_to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            out = model(batch["lr_hsi"])
            losses = compute_losses(
                model, out, batch["lr_hsi"], sam_loss, cfg
            )
            if training:
                losses["total"].backward()
                if cfg.basis_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.basis_grad_clip
                    )
                optimizer.step()
            n = batch["lr_hsi"].size(0)
            for name in names:
                meters[name].update(losses[name].detach().item(), n)
    result = {name: meter.avg for name, meter in meters.items()}
    result["psnr"] = -10.0 * math.log10(max(result["mse"], 1e-12))
    result["sam_deg"] = result["sam"] * 180.0 / math.pi
    result["selection"] = (
        cfg.lambda_l1 * result["l1"]
        + cfg.basis_selection_sam * result["sam"]
        + cfg.basis_selection_sgrad1 * result["sgrad1"]
        + cfg.basis_selection_sgrad2 * result["sgrad2"]
    )
    return result


@torch.no_grad()
def estimate_coefficient_scale(model, loader, device):
    model.eval()
    count = 0
    sum_coeff = torch.zeros(
        model.basis_rank, dtype=torch.float64, device=device
    )
    sum_square = torch.zeros_like(sum_coeff)
    for batch in loader:
        batch = move_to_device(batch, device)
        coeff = model(batch["lr_hsi"])["coefficients"].double()
        flat = coeff.permute(1, 0, 2, 3).reshape(model.basis_rank, -1)
        sum_coeff += flat.sum(dim=1)
        sum_square += flat.square().sum(dim=1)
        count += flat.size(1)
    mean = sum_coeff / max(count, 1)
    var = (sum_square / max(count, 1) - mean.square()).clamp_min(1e-12)
    return var.sqrt().float()


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    model = SpectralFoundation(
        n_bands=info["n_bands"], basis_rank=cfg.basis_rank
    ).to(device)
    init = compute_pca_initialization(
        collect_lr_spectra(train_loader, cfg.basis_init_pixels),
        cfg.basis_rank,
    )
    model.initialize_from_pca(
        init["mean_spectrum"],
        init["basis"],
        init["coefficient_scale"],
        init["eigenvalues"],
        init["total_variance"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05
    )
    sam_loss = SAMLoss()

    root = os.path.join(cfg.checkpoint_root, "spectral_foundation", cfg.dataset)
    ensure_dir(root)
    log_path = os.path.join(
        cfg.log_root, "spectral_foundation", f"{cfg.dataset}.log"
    )
    csv = CSVLogger(
        os.path.join(
            cfg.log_root, "spectral_foundation", f"{cfg.dataset}.csv"
        ),
        ["epoch", "lr", "train_total", "val_psnr", "val_sam", "selection"],
    )

    best_selection = float("inf")
    best_path = os.path.join(root, "spectral_foundation_best.pth")
    for epoch in range(1, cfg.epochs + 1):
        train = run_epoch(
            model, train_loader, sam_loss, cfg, device, optimizer
        )
        val = run_epoch(model, test_loader, sam_loss, cfg, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        csv.write(
            {
                "epoch": epoch,
                "lr": lr,
                "train_total": train["total"],
                "val_psnr": val["psnr"],
                "val_sam": val["sam_deg"],
                "selection": val["selection"],
            }
        )
        write_log(
            log_path,
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"PSNR={val['psnr']:.4f} SAM={val['sam_deg']:.4f} | "
            f"selection={val['selection']:.6f}",
        )
        if val["selection"] < best_selection:
            best_selection = val["selection"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                -best_selection,
                best_path,
                extra={
                    "model_role": "spectral_foundation",
                    "dataset": cfg.dataset,
                    "n_bands": info["n_bands"],
                    "basis_rank": cfg.basis_rank,
                    "pca_explained_variance_ratio": float(
                        init["explained_variance_ratio"].item()
                    ),
                },
            )

    try:
        state = torch.load(
            best_path, map_location=device, weights_only=False
        )
    except TypeError:
        state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.set_coefficient_scale(
        estimate_coefficient_scale(model, train_loader, device)
    )
    downstream_path = os.path.join(root, "foundation_for_local_null.pth")
    save_checkpoint(
        model,
        None,
        state.get("epoch", 0),
        state.get("best_metric", 0.0),
        downstream_path,
        extra={
            "model_role": "spectral_foundation",
            "dataset": cfg.dataset,
            "n_bands": info["n_bands"],
            "basis_rank": cfg.basis_rank,
        },
    )
    print(f"Saved downstream foundation: {downstream_path}")


if __name__ == "__main__":
    main()
