"""E23: single-patch curvature overfit + free-proposal control for OMN-Net.

Purpose
-------
E22 showed two simultaneous bottlenecks for E17-b rank-6 curvature recovery:
(1) train->test generalization loss, and (2) incomplete realization even on the
training region.  E23 removes cross-patch generalization entirely and asks two
capacity questions on one fixed, augmentation-free training patch.

A. Predictor-overfit control
   Continue optimizing the existing E17-b proposal predictor on the same fixed
   train-native patch, with Stage-1/Stage-2 and the LR-HSI-derived P_curv frozen.
   If the CNN can approach the curvature oracle on this patch, the architecture
   has enough local capacity and the main issue is sharing/generalization across
   regions.  If it still saturates far below the oracle, the legal evidence ->
   proposal mapping is itself difficult for the current parameterization.

B. Free-proposal control
   Replace the CNN output with one independently optimizable raw 32-D proposal
   at every HR pixel, while preserving EXACTLY the same
       tanh -> amplitude limit -> P_curv projection
   parameterization and the same normalized SmoothL1 objective.  This isolates
   P_curv, the amplitude bound and the optimization geometry from the predictor.

A direct GT bounded-target control is also reported.  Because the supervised
curvature target already lies in P_curv, if target/limit is within (-1,1), then
atanh(target/limit) is an explicit feasible raw proposal whose projected output
reproduces the target (up to numerical precision).  This is diagnostic only.

No test GT is used and no production model is changed.  All GT use is confined
to this single training-patch capacity diagnostic.
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
from data_loader import build_datasets
from metrics import calc_metrics
from models.local_curvature_extrapolation import project_to_curvature
from train_local_curvature_extrapolation import build_targets
from train_local_curvature_extrapolation_e17b import build_model
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

    p.add_argument("--diagnostic_image_size", type=int, default=128)
    p.add_argument("--patch_index", type=int, default=0)
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

    p.add_argument("--overfit_steps", type=int, default=1200)
    p.add_argument("--overfit_lr", type=float, default=2e-4)
    p.add_argument("--overfit_weight_decay", type=float, default=0.0)
    p.add_argument("--overfit_eval_interval", type=int, default=50)

    p.add_argument("--free_steps", type=int, default=800)
    p.add_argument("--free_lr", type=float, default=5e-2)
    p.add_argument("--free_eval_interval", type=int, default=50)
    p.add_argument(
        "--free_init",
        type=str,
        choices=["zero", "predictor"],
        default="zero",
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
    if cfg.curvature_loss_beta <= 0:
        raise ValueError("curvature_loss_beta must be positive")
    if cfg.overfit_steps < 1 or cfg.free_steps < 1:
        raise ValueError("overfit_steps and free_steps must be positive")
    if cfg.overfit_eval_interval < 1 or cfg.free_eval_interval < 1:
        raise ValueError("evaluation intervals must be positive")
    if cfg.overfit_lr <= 0 or cfg.free_lr <= 0:
        raise ValueError("learning rates must be positive")

    cfg.image_size = cfg.diagnostic_image_size
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


def _batchify(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict:
    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = value
    return move_to_device(batch, device)


def _cached_predictor_features(
    out: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild the exact E17-b predictor input once for the fixed patch."""
    scale = out["coefficient_scale"].view(1, -1, 1, 1).detach()
    normalized_null_seed = out["null_seed_coefficients"].detach() / scale
    normalized_tangent_residual = out["tangent_residual"].detach() / scale
    curvature_projector_diagonal = out[
        "curvature_projector_diagonal"
    ].detach()
    singular = out["curvature_singular_values"].detach()
    singular_scale = singular[:, :1].clamp_min(1e-8)
    normalized_singular = singular / singular_scale
    signed_features = out["normalized_signed_curvature_features"].detach()

    features = torch.cat(
        [
            out["hr_msi"].detach() if "hr_msi" in out else None,
        ],
        dim=1,
    ) if False else None

    # hr_msi is an input rather than an output key; caller inserts it explicitly.
    return (
        normalized_null_seed,
        normalized_tangent_residual,
        torch.cat(
            [
                curvature_projector_diagonal,
                normalized_singular,
                signed_features,
            ],
            dim=1,
        ),
    )


def build_features(
    out: Dict[str, torch.Tensor],
    hr_msi: torch.Tensor,
) -> torch.Tensor:
    normalized_null_seed, normalized_tangent_residual, curvature_features = (
        _cached_predictor_features(out)
    )
    return torch.cat(
        [
            hr_msi.detach(),
            out["base_msi"].detach(),
            out["msi_residual"].detach(),
            normalized_null_seed,
            normalized_tangent_residual,
            curvature_features,
        ],
        dim=1,
    ).detach()


def _smooth_l1(
    residual: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return F.smooth_l1_loss(
        residual / scale,
        target / scale,
        beta=beta,
    )


@torch.no_grad()
def _evaluate_residual(
    model,
    residual: torch.Tensor,
    target: torch.Tensor,
    pcomp: torch.Tensor,
    corrected_coefficients: torch.Tensor,
    basis: torch.Tensor,
    gt: torch.Tensor,
    scale: torch.Tensor,
    cfg,
    stage2_psnr: float,
    curvature_oracle_psnr: float,
    normalized_proposal: torch.Tensor | None = None,
) -> Dict[str, float]:
    hsi = model.local_model.foundation.decode(
        corrected_coefficients + residual,
        basis=basis,
    )
    metric = calc_metrics(hsi, gt, cfg.scale_ratio)

    pred64 = residual.double()
    target64 = target.double()
    pcomp64 = pcomp.double()
    pred_energy = float(pred64.square().sum().item())
    target_energy = max(float(target64.square().sum().item()), 1e-30)
    pcomp_energy = max(float(pcomp64.square().sum().item()), 1e-30)
    dot = float((pred64 * target64).sum().item())
    curv_error = float((pred64 - target64).square().sum().item())
    pcomp_error = float((pred64 - pcomp64).square().sum().item())

    if pred_energy > 1e-30:
        cosine = dot / math.sqrt(pred_energy * target_energy)
        amplitude = math.sqrt(pred_energy / target_energy)
    else:
        cosine = 0.0
        amplitude = 0.0

    pixel_dot = (pred64 * target64).sum(dim=1, keepdim=True)
    pixel_energy = pred64.square().sum(dim=1, keepdim=True)
    pixel_alpha = pixel_dot / pixel_energy.clamp_min(1e-30)
    pixel_residual = pixel_alpha.to(residual.dtype) * residual
    pixel_hsi = model.local_model.foundation.decode(
        corrected_coefficients + pixel_residual,
        basis=basis,
    )
    pixel_metric = calc_metrics(pixel_hsi, gt, cfg.scale_ratio)

    oracle_span = curvature_oracle_psnr - stage2_psnr
    realization = (
        (metric["PSNR"] - stage2_psnr) / oracle_span
        if abs(oracle_span) > 1e-12
        else 0.0
    )

    result = {
        "psnr": float(metric["PSNR"]),
        "sam": float(metric["SAM"]),
        "rmse": float(metric["RMSE"]),
        "loss": float(
            _smooth_l1(residual, target, scale, cfg.curvature_loss_beta).item()
        ),
        "curvature_capture": float(1.0 - curv_error / target_energy),
        "pcomp_capture": float(1.0 - pcomp_error / pcomp_energy),
        "amplitude_ratio": float(amplitude),
        "cosine": float(cosine),
        "oracle_realization": float(realization),
        "pixel_scalar_psnr": float(pixel_metric["PSNR"]),
        "pixel_scalar_sam": float(pixel_metric["SAM"]),
    }
    if normalized_proposal is not None:
        result["saturation"] = float(
            (normalized_proposal.detach().abs() > 0.98).float().mean().item()
        )
    return result


def _print_metrics(prefix: str, record: Dict[str, float], step: int | None = None):
    lead = prefix if step is None else f"{prefix} step={step:04d}"
    sat = (
        f" sat={100.0 * record['saturation']:.2f}%"
        if "saturation" in record
        else ""
    )
    print(
        f"{lead} | PSNR={record['psnr']:.4f} SAM={record['sam']:.4f} "
        f"loss={record['loss']:.6f} "
        f"CurvCap={100.0 * record['curvature_capture']:.2f}% "
        f"PcompCap={100.0 * record['pcomp_capture']:.2f}% "
        f"Amp={record['amplitude_ratio']:.3f} Cos={record['cosine']:.3f} "
        f"OracleRealize={100.0 * record['oracle_realization']:.2f}% "
        f"PixelScalar={record['pixel_scalar_psnr']:.4f}{sat}"
    )


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    train_set, _, info = build_datasets(cfg)
    train_set.augment = False
    if len(train_set) == 0:
        raise RuntimeError("Training set has no patches")
    if cfg.patch_index < 0 or cfg.patch_index >= len(train_set):
        raise IndexError(
            f"patch_index={cfg.patch_index} outside [0,{len(train_set)-1}]"
        )

    patch_coord = tuple(int(v) for v in train_set.coords[cfg.patch_index])
    batch = _batchify(train_set[cfg.patch_index], device)

    model, _, local_epoch, local_best = build_model(cfg, info, device)
    metadata = _read_checkpoint_metadata(cfg.curvature_checkpoint, device)
    checkpoint_rank = metadata["extra"].get("curvature_rank")
    if checkpoint_rank is not None and int(checkpoint_rank) != int(cfg.curvature_rank):
        raise ValueError(
            "Curvature checkpoint rank mismatch: "
            f"checkpoint rank={checkpoint_rank}, requested rank={cfg.curvature_rank}"
        )
    role = str(metadata["extra"].get("model_role", ""))
    if role and "e17b" not in role:
        raise ValueError(
            f"Checkpoint role '{role}' does not look like E17-b."
        )

    curvature_epoch, curvature_best = load_checkpoint(
        model,
        cfg.curvature_checkpoint,
        optimizer=None,
        strict=True,
        map_location=device,
        load_optimizer=False,
    )
    model.eval()

    with torch.no_grad():
        out = model(batch["lr_hsi"], batch["hr_msi"])
        targets = build_targets(model, out, batch["gt"])

    scale = out["coefficient_scale"].view(1, -1, 1, 1).detach()
    limit = cfg.curvature_proposal_amplitude_multiplier * scale
    curvature_basis = out["curvature_basis"].detach()
    target = targets["curvature"].detach()
    pcomp = targets["pcomp"].detach()
    corrected = out["corrected_coefficients"].detach()
    spectral_basis = out["basis"].detach()
    gt = batch["gt"].detach()
    features = build_features(out, batch["hr_msi"])

    with torch.no_grad():
        stage2_metric = calc_metrics(out["reconstructed_hsi"], gt, cfg.scale_ratio)
        oracle_hsi = model.local_model.foundation.decode(
            corrected + target,
            basis=spectral_basis,
        )
        oracle_metric = calc_metrics(oracle_hsi, gt, cfg.scale_ratio)
        full_pcomp_hsi = model.local_model.foundation.decode(
            corrected + pcomp,
            basis=spectral_basis,
        )
        full_pcomp_metric = calc_metrics(full_pcomp_hsi, gt, cfg.scale_ratio)

    stage2_psnr = float(stage2_metric["PSNR"])
    oracle_psnr = float(oracle_metric["PSNR"])

    baseline = _evaluate_residual(
        model,
        out["curvature_residual"].detach(),
        target,
        pcomp,
        corrected,
        spectral_basis,
        gt,
        scale,
        cfg,
        stage2_psnr,
        oracle_psnr,
        normalized_proposal=out["normalized_curvature_proposal"].detach(),
    )

    # Explicit bounded-target feasibility control.
    ratio = target / limit
    exceeds = ratio.abs() >= 1.0
    feasibility = {
        "target_exceeds_limit_fraction": float(exceeds.float().mean().item()),
        "target_peak_limit_ratio": float(ratio.abs().max().item()),
        "target_projection_self_relative_error": float(
            (
                (project_to_curvature(curvature_basis, target) - target)
                .double()
                .square()
                .sum()
                / target.double().square().sum().clamp_min(1e-30)
            ).sqrt().item()
        ),
    }
    direct_normalized = ratio.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    direct_raw = torch.atanh(direct_normalized)
    direct_proposal = torch.tanh(direct_raw) * limit
    direct_residual = project_to_curvature(curvature_basis, direct_proposal)
    direct_control = _evaluate_residual(
        model,
        direct_residual,
        target,
        pcomp,
        corrected,
        spectral_basis,
        gt,
        scale,
        cfg,
        stage2_psnr,
        oracle_psnr,
        normalized_proposal=direct_normalized,
    )

    print(
        "E23 single-patch overfit | "
        f"dataset={cfg.dataset} patch_index={cfg.patch_index} coord={patch_coord} "
        f"checkpoint_epoch={curvature_epoch} checkpoint_best={curvature_best:.4f} "
        f"local_epoch={local_epoch} local_best={local_best:.4f}"
    )
    print(
        f"Reference | Stage2={stage2_psnr:.4f} "
        f"CurvOracle={oracle_psnr:.4f} "
        f"FullPcomp={full_pcomp_metric['PSNR']:.4f}"
    )
    _print_metrics("Checkpoint", baseline)
    print(
        "Bound feasibility | "
        f"exceed={100.0 * feasibility['target_exceeds_limit_fraction']:.4f}% "
        f"peak|target/limit|={feasibility['target_peak_limit_ratio']:.4f} "
        f"Pcurv(target) relerr={feasibility['target_projection_self_relative_error']:.3e}"
    )
    _print_metrics("Direct bounded-target GT control", direct_control)

    # ---------------------------------------------------------------
    # A) Continue overfitting the existing E17-b predictor on one patch.
    # ---------------------------------------------------------------
    model.proposal_predictor.train()
    optimizer = torch.optim.AdamW(
        model.proposal_predictor.parameters(),
        lr=cfg.overfit_lr,
        weight_decay=cfg.overfit_weight_decay,
    )
    overfit_curve = []
    best_overfit = dict(baseline)
    best_overfit_step = 0

    for step in range(1, cfg.overfit_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        raw = model.proposal_predictor(features)
        normalized = torch.tanh(raw)
        proposal = normalized * limit
        residual = project_to_curvature(curvature_basis, proposal)
        loss = _smooth_l1(residual, target, scale, cfg.curvature_loss_beta)
        loss.backward()
        if cfg.curvature_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.proposal_predictor.parameters(), cfg.curvature_grad_clip
            )
        optimizer.step()

        if step % cfg.overfit_eval_interval == 0 or step == cfg.overfit_steps:
            with torch.no_grad():
                raw_eval = model.proposal_predictor(features)
                norm_eval = torch.tanh(raw_eval)
                residual_eval = project_to_curvature(
                    curvature_basis, norm_eval * limit
                )
                stat = _evaluate_residual(
                    model,
                    residual_eval,
                    target,
                    pcomp,
                    corrected,
                    spectral_basis,
                    gt,
                    scale,
                    cfg,
                    stage2_psnr,
                    oracle_psnr,
                    normalized_proposal=norm_eval,
                )
            stat["step"] = int(step)
            overfit_curve.append(stat)
            _print_metrics("Predictor-overfit", stat, step)
            if stat["psnr"] > best_overfit["psnr"]:
                best_overfit = dict(stat)
                best_overfit_step = int(step)

    with torch.no_grad():
        raw_final = model.proposal_predictor(features)
        norm_final = torch.tanh(raw_final)
        residual_final = project_to_curvature(curvature_basis, norm_final * limit)
        final_overfit = _evaluate_residual(
            model,
            residual_final,
            target,
            pcomp,
            corrected,
            spectral_basis,
            gt,
            scale,
            cfg,
            stage2_psnr,
            oracle_psnr,
            normalized_proposal=norm_final,
        )

    # ---------------------------------------------------------------
    # B) Per-pixel free raw proposal, same tanh/limit/P_curv/loss.
    # ---------------------------------------------------------------
    if cfg.free_init == "predictor":
        free_initial = out["raw_curvature_proposal"].detach().clone()
    else:
        free_initial = torch.zeros_like(out["raw_curvature_proposal"]).detach()
    free_raw = torch.nn.Parameter(free_initial)
    free_optimizer = torch.optim.Adam([free_raw], lr=cfg.free_lr)
    free_curve = []
    best_free = None
    best_free_step = 0

    for step in range(1, cfg.free_steps + 1):
        free_optimizer.zero_grad(set_to_none=True)
        free_normalized = torch.tanh(free_raw)
        free_residual = project_to_curvature(
            curvature_basis, free_normalized * limit
        )
        free_loss = _smooth_l1(
            free_residual, target, scale, cfg.curvature_loss_beta
        )
        free_loss.backward()
        free_optimizer.step()

        if step % cfg.free_eval_interval == 0 or step == cfg.free_steps:
            with torch.no_grad():
                norm_eval = torch.tanh(free_raw)
                residual_eval = project_to_curvature(
                    curvature_basis, norm_eval * limit
                )
                stat = _evaluate_residual(
                    model,
                    residual_eval,
                    target,
                    pcomp,
                    corrected,
                    spectral_basis,
                    gt,
                    scale,
                    cfg,
                    stage2_psnr,
                    oracle_psnr,
                    normalized_proposal=norm_eval,
                )
            stat["step"] = int(step)
            free_curve.append(stat)
            _print_metrics("Free-proposal", stat, step)
            if best_free is None or stat["psnr"] > best_free["psnr"]:
                best_free = dict(stat)
                best_free_step = int(step)

    with torch.no_grad():
        free_norm_final = torch.tanh(free_raw)
        free_residual_final = project_to_curvature(
            curvature_basis, free_norm_final * limit
        )
        final_free = _evaluate_residual(
            model,
            free_residual_final,
            target,
            pcomp,
            corrected,
            spectral_basis,
            gt,
            scale,
            cfg,
            stage2_psnr,
            oracle_psnr,
            normalized_proposal=free_norm_final,
        )

    print(
        "E23 summary | "
        f"Checkpoint={baseline['psnr']:.4f}/{100.0*baseline['curvature_capture']:.2f}% "
        f"PredictorBest={best_overfit['psnr']:.4f}/"
        f"{100.0*best_overfit['curvature_capture']:.2f}%@{best_overfit_step} "
        f"FreeBest={best_free['psnr']:.4f}/"
        f"{100.0*best_free['curvature_capture']:.2f}%@{best_free_step} "
        f"DirectBounded={direct_control['psnr']:.4f}/"
        f"{100.0*direct_control['curvature_capture']:.2f}% "
        f"Oracle={oracle_psnr:.4f}"
    )

    result = {
        "experiment": "E23 single-patch predictor overfit + free proposal control",
        "dataset": cfg.dataset,
        "patch_index": int(cfg.patch_index),
        "patch_coord": list(patch_coord),
        "checkpoint": cfg.curvature_checkpoint,
        "checkpoint_epoch": int(curvature_epoch),
        "checkpoint_best": float(curvature_best),
        "curvature_rank": int(cfg.curvature_rank),
        "reference": {
            "stage2": {k.lower(): float(v) for k, v in stage2_metric.items()},
            "curvature_oracle": {
                k.lower(): float(v) for k, v in oracle_metric.items()
            },
            "full_pcomp": {
                k.lower(): float(v) for k, v in full_pcomp_metric.items()
            },
        },
        "feasibility": feasibility,
        "checkpoint_baseline": baseline,
        "direct_bounded_target_control": direct_control,
        "predictor_overfit": {
            "steps": int(cfg.overfit_steps),
            "lr": float(cfg.overfit_lr),
            "best_step": int(best_overfit_step),
            "best": best_overfit,
            "final": final_overfit,
            "curve": overfit_curve,
        },
        "free_proposal": {
            "init": cfg.free_init,
            "steps": int(cfg.free_steps),
            "lr": float(cfg.free_lr),
            "best_step": int(best_free_step),
            "best": best_free,
            "final": final_free,
            "curve": free_curve,
        },
    }

    out_dir = os.path.join(
        cfg.output_root,
        "diagnostics",
        "curvature_single_patch_overfit",
        cfg.dataset,
    )
    ensure_dir(out_dir)
    out_path = os.path.join(
        out_dir,
        f"curvature_single_patch_overfit_patch{cfg.patch_index:03d}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
