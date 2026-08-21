"""E24: LR-HSI cross-scale internal supervision for curvature adaptation.

This experiment adapts only the E17-b curvature proposal predictor using the
current test scene's *observed* LR-HSI. No HR-HSI ground truth is used by the
optimizer or by checkpoint selection.

For the real x4 task
    LR-HSI + HR-MSI -> HR-HSI,
the observed LR-HSI is treated as a fully spectral pseudo-HR target. A lower
resolution pseudo input is generated with the same HSI degradation operator,
and the pseudo-HR MSI is generated with the same SRF:
    pseudo-GT      = observed LR-HSI
    pseudo-LLR-HSI = D(pseudo-GT)
    pseudo-HR-MSI  = R(pseudo-GT)

The resulting task is spectrally supervised and geometrically isomorphic to the
real fusion task. Stage-1, Stage-2, P_tan and P_curv remain frozen/analytical;
only the existing E17-b proposal predictor is updated.

The real HR-HSI GT is evaluated only as a diagnostic trajectory. It is never
used in the adaptation loss and never used to select the saved checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from typing import Dict, List

import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders, make_lr_hsi
from metrics import calc_metrics
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.local_curvature_extrapolation_e17b import LocalCurvatureExtrapolationE17BNet
from train_local_curvature_extrapolation import build_targets
from utils import (
    ensure_dir,
    get_device,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
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
    p.add_argument(
        "--curvature_checkpoint",
        type=str,
        default=(
            "./checkpoints/local_curvature_extrapolation_e17b/PaviaU/"
            "curvature_e17b_best_psnr.pth"
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

    p.add_argument("--curvature_rank", type=int, default=6)
    p.add_argument("--curvature_svd_chunk_pixels", type=int, default=2048)
    p.add_argument("--curvature_svd_tolerance", type=float, default=1e-5)
    p.add_argument("--curvature_abs_tolerance", type=float, default=1e-9)
    p.add_argument(
        "--curvature_proposal_amplitude_multiplier", type=float, default=8.0
    )
    p.add_argument("--curvature_predictor_hidden", type=int, default=96)
    p.add_argument("--curvature_predictor_blocks", type=int, default=4)
    p.add_argument("--curvature_loss_beta", type=float, default=0.25)
    p.add_argument("--curvature_grad_clip", type=float, default=1.0)

    p.add_argument("--adapt_steps", type=int, default=400)
    p.add_argument("--adapt_lr", type=float, default=2e-4)
    p.add_argument("--adapt_weight_decay", type=float, default=0.0)
    p.add_argument("--adapt_eval_interval", type=int, default=25)
    p.add_argument(
        "--adapt_checkpoint_name",
        type=str,
        default="e24_best_pseudo_loss.pth",
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
    if cfg.adapt_steps < 1 or cfg.adapt_eval_interval < 1:
        raise ValueError("adapt steps/eval interval must be positive")
    if cfg.adapt_lr <= 0 or cfg.curvature_loss_beta <= 0:
        raise ValueError("adapt lr and curvature loss beta must be positive")
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

    model = LocalCurvatureExtrapolationE17BNet(
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
            f"checkpoint rank={checkpoint_rank}, requested rank={cfg.curvature_rank}"
        )
    role = str(metadata["extra"].get("model_role", ""))
    if role and "e17b" not in role:
        raise ValueError(f"Checkpoint role '{role}' does not look like E17-b")

    curvature_epoch, curvature_best = load_checkpoint(
        model,
        cfg.curvature_checkpoint,
        optimizer=None,
        strict=True,
        map_location=device,
        load_optimizer=False,
    )
    model.eval()
    for parameter in model.local_model.parameters():
        parameter.requires_grad_(False)
    return model, local_epoch, local_best, curvature_epoch, curvature_best


def _pseudo_task_from_observed_lr(
    model: LocalCurvatureExtrapolationE17BNet,
    real_batch: Dict[str, torch.Tensor],
    cfg,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Construct a legal x4 internal task from the observed real LR-HSI."""
    observed_lr = real_batch["lr_hsi"].detach()
    if observed_lr.size(0) != 1:
        raise ValueError("E24 v1 expects one test scene per adaptation run")
    h, w = observed_lr.shape[-2:]
    if h % cfg.scale_ratio != 0 or w % cfg.scale_ratio != 0:
        raise ValueError(
            f"Observed LR-HSI size {(h, w)} is not divisible by scale_ratio={cfg.scale_ratio}"
        )
    if min(h // cfg.scale_ratio, w // cfg.scale_ratio) < 5:
        raise ValueError("Pseudo LLR-HSI is too small for the curvature construction")

    # Use the same dataset degradation routine used to create LR-HSI inputs.
    pseudo_gt_np = (
        observed_lr[0].detach().float().cpu().permute(1, 2, 0).contiguous().numpy()
    )
    pseudo_lr_np = make_lr_hsi(pseudo_gt_np, cfg.scale_ratio)
    pseudo_lr = (
        torch.from_numpy(pseudo_lr_np)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(device=device, dtype=observed_lr.dtype)
    )

    # Same SRF as the real task; no hidden spectrum is synthesized.
    pseudo_msi = model.local_model.geometry.hsi_to_msi(observed_lr).detach()
    return {"lr_hsi": pseudo_lr, "hr_msi": pseudo_msi, "gt": observed_lr}


def _normalized_curvature_loss(
    out: Dict[str, torch.Tensor], target: torch.Tensor, beta: float
) -> torch.Tensor:
    scale = out["coefficient_scale"].view(1, -1, 1, 1)
    return F.smooth_l1_loss(
        out["curvature_residual"] / scale,
        target / scale,
        beta=beta,
    )


@torch.no_grad()
def evaluate_task(
    model: LocalCurvatureExtrapolationE17BNet,
    batch: Dict[str, torch.Tensor],
    cfg,
) -> Dict[str, float]:
    model.eval()
    out = model(batch["lr_hsi"], batch["hr_msi"])
    targets = build_targets(model, out, batch["gt"])
    pred = out["curvature_residual"]
    target = targets["curvature"]

    oracle_hsi = model.local_model.foundation.decode(
        out["corrected_coefficients"] + target, basis=out["basis"]
    )
    full_pcomp_hsi = model.local_model.foundation.decode(
        out["corrected_coefficients"] + targets["pcomp"], basis=out["basis"]
    )
    stage2_metric = calc_metrics(out["reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
    pred_metric = calc_metrics(
        out["curvature_reconstructed_hsi"], batch["gt"], cfg.scale_ratio
    )
    oracle_metric = calc_metrics(oracle_hsi, batch["gt"], cfg.scale_ratio)
    pcomp_metric = calc_metrics(full_pcomp_hsi, batch["gt"], cfg.scale_ratio)

    pred64 = pred.double()
    target64 = target.double()
    pcomp64 = targets["pcomp"].double()
    pred_energy = float(pred64.square().sum().item())
    target_energy = max(float(target64.square().sum().item()), 1e-30)
    pcomp_energy = max(float(pcomp64.square().sum().item()), 1e-30)
    dot = float((pred64 * target64).sum().item())
    curv_error = float((pred64 - target64).square().sum().item())
    pcomp_error = float((pred64 - pcomp64).square().sum().item())

    amp = math.sqrt(pred_energy / target_energy) if pred_energy > 1e-30 else 0.0
    cosine = (
        dot / math.sqrt(pred_energy * target_energy)
        if pred_energy > 1e-30
        else 0.0
    )
    stage2_psnr = float(stage2_metric["PSNR"])
    oracle_psnr = float(oracle_metric["PSNR"])
    oracle_span = oracle_psnr - stage2_psnr
    realize = (
        (float(pred_metric["PSNR"]) - stage2_psnr) / oracle_span
        if abs(oracle_span) > 1e-12
        else 0.0
    )

    pixel_dot = (pred64 * target64).sum(dim=1, keepdim=True)
    pixel_pred_energy = pred64.square().sum(dim=1, keepdim=True)
    pixel_alpha = pixel_dot / pixel_pred_energy.clamp_min(1e-30)
    pixel_residual = pixel_alpha.to(pred.dtype) * pred
    pixel_hsi = model.local_model.foundation.decode(
        out["corrected_coefficients"] + pixel_residual, basis=out["basis"]
    )
    pixel_metric = calc_metrics(pixel_hsi, batch["gt"], cfg.scale_ratio)

    return {
        "stage2_psnr": stage2_psnr,
        "pred_psnr": float(pred_metric["PSNR"]),
        "pred_sam": float(pred_metric["SAM"]),
        "pixel_scalar_psnr": float(pixel_metric["PSNR"]),
        "oracle_psnr": oracle_psnr,
        "full_pcomp_psnr": float(pcomp_metric["PSNR"]),
        "loss": float(
            _normalized_curvature_loss(out, target, cfg.curvature_loss_beta).item()
        ),
        "curvature_capture": float(1.0 - curv_error / target_energy),
        "pcomp_capture": float(1.0 - pcomp_error / pcomp_energy),
        "amplitude_ratio": float(amp),
        "cosine": float(cosine),
        "oracle_realization": float(realize),
        "saturation": float(
            (out["normalized_curvature_proposal"].abs() > 0.98)
            .float().mean().item()
        ),
    }


def _print_eval(label: str, step: int, stat: Dict[str, float]):
    print(
        f"{label} step={step:04d} | "
        f"PSNR={stat['pred_psnr']:.4f} SAM={stat['pred_sam']:.4f} "
        f"Stage2={stat['stage2_psnr']:.4f} Oracle={stat['oracle_psnr']:.4f} "
        f"PixelScalar={stat['pixel_scalar_psnr']:.4f} | "
        f"loss={stat['loss']:.6f} CurvCap={100.0*stat['curvature_capture']:.2f}% "
        f"PcompCap={100.0*stat['pcomp_capture']:.2f}% "
        f"Amp={stat['amplitude_ratio']:.3f} Cos={stat['cosine']:.3f} "
        f"OracleRealize={100.0*stat['oracle_realization']:.2f}% "
        f"sat={100.0*stat['saturation']:.2f}%"
    )


def _checkpoint_extra(cfg, source_checkpoint: str, pseudo_shape) -> Dict:
    return {
        "model_role": "internal_curvature_adaptation_e24",
        "experiment": "E24 LR-HSI cross-scale internal curvature adaptation",
        "dataset": cfg.dataset,
        "curvature_rank": int(cfg.curvature_rank),
        "source_checkpoint": source_checkpoint,
        "selection": "lowest evaluated pseudo-task curvature loss; real HR-HSI GT excluded",
        "pseudo_gt_source": "observed real LR-HSI",
        "pseudo_task_shape": list(pseudo_shape),
        "adapt_steps": int(cfg.adapt_steps),
        "adapt_lr": float(cfg.adapt_lr),
    }


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)

    if len(test_loader.dataset) != 1:
        raise ValueError(
            "E24 v1 expects one held-out test scene. "
            f"Found {len(test_loader.dataset)} test samples."
        )
    real_batch = move_to_device(next(iter(test_loader)), device)

    model, local_epoch, local_best, curvature_epoch, curvature_best = build_model(
        cfg, info, device
    )
    pseudo_batch = _pseudo_task_from_observed_lr(model, real_batch, cfg, device)

    print(
        "E24 internal curvature adaptation | "
        f"dataset={cfg.dataset} rank={cfg.curvature_rank} "
        f"checkpoint_epoch={curvature_epoch} checkpoint_best={curvature_best:.4f} "
        f"realLR={tuple(real_batch['lr_hsi'].shape[-2:])} "
        f"pseudoLLR={tuple(pseudo_batch['lr_hsi'].shape[-2:])} "
        f"pseudoGT={tuple(pseudo_batch['gt'].shape[-2:])}"
    )
    print(
        "Information boundary | adaptation uses pseudo task only: "
        "observed LR-HSI -> degraded LLR-HSI + SRF MSI -> observed LR-HSI. "
        "Real HR-HSI GT is diagnostic only."
    )

    pseudo_start = evaluate_task(model, pseudo_batch, cfg)
    real_start = evaluate_task(model, real_batch, cfg)
    _print_eval("Pseudo", 0, pseudo_start)
    _print_eval("Real-diagnostic", 0, real_start)

    optimizer = torch.optim.AdamW(
        model.proposal_predictor.parameters(),
        lr=cfg.adapt_lr,
        weight_decay=cfg.adapt_weight_decay,
    )

    out_dir = os.path.join(cfg.output_root, "internal_curvature_adaptation", cfg.dataset)
    ckpt_dir = os.path.join(
        cfg.checkpoint_root, "internal_curvature_adaptation", cfg.dataset
    )
    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)
    checkpoint_path = os.path.join(ckpt_dir, cfg.adapt_checkpoint_name)

    best_pseudo_loss = float(pseudo_start["loss"])
    best_pseudo_step = 0
    best_state = copy.deepcopy(model.proposal_predictor.state_dict())
    trajectory = [
        {"step": 0, "pseudo": pseudo_start, "real_diagnostic": real_start}
    ]

    save_checkpoint(
        model,
        optimizer=None,
        epoch=0,
        best_metric=-best_pseudo_loss,
        path=checkpoint_path,
        extra=_checkpoint_extra(
            cfg, cfg.curvature_checkpoint, pseudo_batch["gt"].shape
        ),
    )

    for step in range(1, cfg.adapt_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(pseudo_batch["lr_hsi"], pseudo_batch["hr_msi"])
        targets = build_targets(model, out, pseudo_batch["gt"])
        loss = _normalized_curvature_loss(
            out, targets["curvature"], cfg.curvature_loss_beta
        )
        loss.backward()
        if cfg.curvature_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.proposal_predictor.parameters(), cfg.curvature_grad_clip
            )
        optimizer.step()

        if step % cfg.adapt_eval_interval == 0 or step == cfg.adapt_steps:
            # This post-update pseudo loss is the only model-selection signal.
            pseudo_stat = evaluate_task(model, pseudo_batch, cfg)
            real_stat = evaluate_task(model, real_batch, cfg)
            _print_eval("Pseudo", step, pseudo_stat)
            _print_eval("Real-diagnostic", step, real_stat)
            trajectory.append(
                {
                    "step": int(step),
                    "pseudo": pseudo_stat,
                    "real_diagnostic": real_stat,
                }
            )

            if pseudo_stat["loss"] < best_pseudo_loss:
                best_pseudo_loss = float(pseudo_stat["loss"])
                best_pseudo_step = int(step)
                best_state = copy.deepcopy(model.proposal_predictor.state_dict())
                save_checkpoint(
                    model,
                    optimizer=None,
                    epoch=step,
                    best_metric=-best_pseudo_loss,
                    path=checkpoint_path,
                    extra=_checkpoint_extra(
                        cfg, cfg.curvature_checkpoint, pseudo_batch["gt"].shape
                    ),
                )

    # Restore the state selected only by the legal pseudo-task loss.
    model.proposal_predictor.load_state_dict(best_state, strict=True)
    pseudo_selected = evaluate_task(model, pseudo_batch, cfg)
    real_selected = evaluate_task(model, real_batch, cfg)
    print(
        "E24 pseudo-selected summary | "
        f"step={best_pseudo_step} pseudo_loss={best_pseudo_loss:.6f} | "
        f"Real {real_start['pred_psnr']:.4f}->{real_selected['pred_psnr']:.4f} "
        f"({real_selected['pred_psnr']-real_start['pred_psnr']:+.4f} dB) | "
        f"Cos {real_start['cosine']:.3f}->{real_selected['cosine']:.3f} | "
        f"CurvCap {100.0*real_start['curvature_capture']:.2f}%->"
        f"{100.0*real_selected['curvature_capture']:.2f}% | "
        f"OracleRealize {100.0*real_start['oracle_realization']:.2f}%->"
        f"{100.0*real_selected['oracle_realization']:.2f}%"
    )

    result = {
        "experiment": "E24 LR-HSI cross-scale internal curvature adaptation",
        "dataset": cfg.dataset,
        "curvature_rank": int(cfg.curvature_rank),
        "source_checkpoint": cfg.curvature_checkpoint,
        "source_checkpoint_epoch": int(curvature_epoch),
        "source_checkpoint_best": float(curvature_best),
        "local_checkpoint_epoch": int(local_epoch),
        "local_checkpoint_best": float(local_best),
        "information_boundary": {
            "pseudo_gt": "observed real LR-HSI",
            "pseudo_lr": "same x4 HSI degradation applied to observed LR-HSI",
            "pseudo_msi": "same SRF applied to observed LR-HSI",
            "optimized_parameters": "E17-b proposal predictor only",
            "real_hr_hsi_gt_used_for_optimization": False,
            "real_hr_hsi_gt_used_for_checkpoint_selection": False,
        },
        "shapes": {
            "real_lr": list(real_batch["lr_hsi"].shape),
            "pseudo_lr": list(pseudo_batch["lr_hsi"].shape),
            "pseudo_gt": list(pseudo_batch["gt"].shape),
            "pseudo_msi": list(pseudo_batch["hr_msi"].shape),
        },
        "adaptation": {
            "steps": int(cfg.adapt_steps),
            "lr": float(cfg.adapt_lr),
            "weight_decay": float(cfg.adapt_weight_decay),
            "eval_interval": int(cfg.adapt_eval_interval),
            "best_pseudo_step": int(best_pseudo_step),
            "best_pseudo_loss": float(best_pseudo_loss),
            "checkpoint": checkpoint_path,
        },
        "start": {"pseudo": pseudo_start, "real_diagnostic": real_start},
        "pseudo_selected": {
            "pseudo": pseudo_selected,
            "real_diagnostic": real_selected,
        },
        "trajectory": trajectory,
    }
    out_path = os.path.join(out_dir, "e24_internal_curvature_adaptation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result: {out_path}")
    print(f"Saved pseudo-selected checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
