"""E26: LR-HSI manifold-state clustered curvature specialists for OMN-Net.

E23 showed that the existing E17-b predictor can fit one fixed patch almost to
100% curvature capture, while E22 showed that one globally shared predictor
realizes much less of the curvature oracle across the scene. E25 further showed
that LR-HSI manifold-state similarity is positively associated with predictor
gradient alignment. E26 therefore tests a discrete state-conditioned upper
bound before introducing any dynamic modulation network.

Protocol
--------
1) Build one legal LR-HSI manifold descriptor for every deterministic
   train-native patch. No GT-dependent quantity is used by clustering/routing.
2) Standardize descriptors across training patches, L2-normalize them and run a
   deterministic spherical k-means with K=4.
3) Initialize K identical E17-b specialists from the same global checkpoint.
   Each specialist is fine-tuned only on patches in its own state cluster.
4) Evaluate all K specialists on the held-out test image.
5) Report two test results:
   * legal nearest-state routing: choose the expert whose training-state center
     is most similar to the test LR-HSI descriptor;
   * GT-only oracle-best expert: diagnostic upper bound over the K specialists.
6) Train a Global-continue control from the same source checkpoint for the same
   number of optimizer updates as one specialist. This distinguishes gains from
   state specialization from gains caused merely by additional training.

The real test HR-HSI GT is never used by clustering, expert routing, expert
training, checkpoint selection or the Global-continue control. It is used only
for final diagnostic metrics and the oracle-best expert report.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_datasets
from metrics import calc_metrics
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.local_curvature_extrapolation_e17b import LocalCurvatureExtrapolationE17BNet
from train_local_curvature_extrapolation import build_targets
from utils import ensure_dir, get_device, load_checkpoint, move_to_device, save_checkpoint, set_seed


def _has_option(arguments: List[str], option: str) -> bool:
    return any(x == option or x.startswith(option + "=") for x in arguments)


def parse_specific_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--foundation_checkpoint", type=str, default="./checkpoints/RAPD-Net/basis_for_stage2.pth")
    p.add_argument(
        "--local_checkpoint",
        type=str,
        default="./checkpoints/local_null_manifold/PaviaU/local_null_best_psnr.pth",
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
    p.add_argument("--curvature_rank", type=int, default=6)
    p.add_argument("--curvature_svd_chunk_pixels", type=int, default=2048)
    p.add_argument("--curvature_svd_tolerance", type=float, default=1e-5)
    p.add_argument("--curvature_abs_tolerance", type=float, default=1e-9)
    p.add_argument("--curvature_proposal_amplitude_multiplier", type=float, default=8.0)
    p.add_argument("--curvature_predictor_hidden", type=int, default=96)
    p.add_argument("--curvature_predictor_blocks", type=int, default=4)
    p.add_argument("--curvature_loss_beta", type=float, default=0.25)
    p.add_argument("--curvature_grad_clip", type=float, default=1.0)

    p.add_argument("--state_clusters", type=int, default=4)
    p.add_argument("--kmeans_iterations", type=int, default=100)
    p.add_argument("--expert_steps", type=int, default=800)
    p.add_argument("--expert_lr", type=float, default=2e-4)
    p.add_argument("--expert_weight_decay", type=float, default=0.0)
    p.add_argument("--expert_eval_interval", type=int, default=100)
    p.add_argument("--global_control_steps", type=int, default=800)
    p.add_argument("--global_control_lr", type=float, default=2e-4)
    p.add_argument("--global_control_eval_interval", type=int, default=100)

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"
    cfg.image_size = cfg.diagnostic_image_size
    if cfg.curvature_rank < 1 or cfg.curvature_rank > 8:
        raise ValueError("curvature_rank must be in [1,8]")
    if cfg.state_clusters < 2:
        raise ValueError("state_clusters must be >= 2")
    if cfg.expert_steps < 1 or cfg.global_control_steps < 1:
        raise ValueError("training steps must be positive")
    if cfg.expert_eval_interval < 1 or cfg.global_control_eval_interval < 1:
        raise ValueError("eval intervals must be positive")
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
    return model, local_epoch, local_best, curvature_epoch, curvature_best


def _batchify(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict:
    batch = {}
    for key, value in sample.items():
        batch[key] = value.unsqueeze(0) if torch.is_tensor(value) else value
    return move_to_device(batch, device)


def _normalized_curvature_loss(
    out: Dict[str, torch.Tensor], target: torch.Tensor, beta: float
) -> torch.Tensor:
    scale = out["coefficient_scale"].view(1, -1, 1, 1)
    return F.smooth_l1_loss(
        out["curvature_residual"] / scale,
        target / scale,
        beta=beta,
    )


def _lr_manifold_descriptor(model, out: Dict[str, torch.Tensor]) -> np.ndarray:
    """Same GT-free LR-HSI manifold descriptor family used by E25."""
    scale = out["coefficient_scale"].view(1, -1, 1, 1).detach()
    global_scale = scale.mean().clamp_min(1e-8)

    lr_coeff = out["lr_coefficients"].detach() / scale
    lr_null = (
        model.local_model.geometry.project_null(out["lr_coefficients"].detach())
        / scale
    )
    tangent_singular = out["tangent_singular_values"].detach() / global_scale
    curvature_singular = out["curvature_singular_values"].detach() / global_scale
    projector_diag = out["curvature_projector_diagonal"].detach()
    signed = out["signed_projected_curvature_bank"].detach() / scale.unsqueeze(1)
    signed_rms = signed.square().mean(dim=(2, 3, 4)).sqrt()

    def mean_std(x: torch.Tensor) -> List[torch.Tensor]:
        return [x.mean(dim=(-2, -1)), x.std(dim=(-2, -1), unbiased=False)]

    parts: List[torch.Tensor] = []
    parts += mean_std(lr_coeff)
    parts += mean_std(lr_null)
    parts += mean_std(tangent_singular)
    parts += mean_std(curvature_singular)
    parts += mean_std(projector_diag)
    parts.append(signed_rms)
    return torch.cat(parts, dim=1).squeeze(0).float().cpu().numpy().astype(np.float64)


def _fit_standardizer(train_desc: np.ndarray):
    mean = train_desc.mean(axis=0, keepdims=True)
    std = train_desc.std(axis=0, keepdims=True)
    keep = std.reshape(-1) > 1e-10
    if not np.any(keep):
        raise RuntimeError("All descriptor dimensions are constant")
    return mean, std, keep


def _transform_descriptor(x: np.ndarray, mean, std, keep) -> np.ndarray:
    z = (x[:, keep] - mean[:, keep]) / std[:, keep]
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norms, 1e-12)


def _spherical_kmeans(x: np.ndarray, k: int, iterations: int):
    if x.ndim != 2 or x.shape[0] < k:
        raise ValueError("not enough samples for requested state_clusters")
    n = x.shape[0]

    global_mean = x.mean(axis=0)
    global_mean /= max(np.linalg.norm(global_mean), 1e-12)
    first = int(np.argmin(x @ global_mean))
    centers = [x[first].copy()]
    chosen = {first}
    while len(centers) < k:
        sims = np.stack([x @ c for c in centers], axis=1)
        nearest_sim = sims.max(axis=1)
        for idx in chosen:
            nearest_sim[idx] = 2.0
        nxt = int(np.argmin(nearest_sim))
        chosen.add(nxt)
        centers.append(x[nxt].copy())
    centers = np.stack(centers, axis=0)

    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(max(iterations, 1)):
        new_labels = np.argmax(x @ centers.T, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        new_centers = []
        for cluster in range(k):
            members = x[labels == cluster]
            if members.shape[0] == 0:
                sims = x @ centers.T
                best_sim = sims.max(axis=1)
                idx = int(np.argmin(best_sim))
                center = x[idx].copy()
            else:
                center = members.mean(axis=0)
            center /= max(np.linalg.norm(center), 1e-12)
            new_centers.append(center)
        centers = np.stack(new_centers, axis=0)
    labels = np.argmax(x @ centers.T, axis=1).astype(np.int64)
    return labels, centers


@torch.no_grad()
def _evaluate(model, batch: Dict[str, torch.Tensor], cfg) -> Dict[str, float]:
    model.eval()
    out = model(batch["lr_hsi"], batch["hr_msi"])
    targets = build_targets(model, out, batch["gt"])
    pred = out["curvature_residual"]
    target = targets["curvature"]

    pred_metric = calc_metrics(out["curvature_reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
    stage2_metric = calc_metrics(out["reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
    oracle_hsi = model.local_model.foundation.decode(
        out["corrected_coefficients"] + target,
        basis=out["basis"],
    )
    oracle_metric = calc_metrics(oracle_hsi, batch["gt"], cfg.scale_ratio)

    pred64 = pred.double()
    target64 = target.double()
    pred_energy = float(pred64.square().sum().item())
    target_energy = max(float(target64.square().sum().item()), 1e-30)
    dot = float((pred64 * target64).sum().item())
    error = float((pred64 - target64).square().sum().item())
    amp = math.sqrt(pred_energy / target_energy) if pred_energy > 1e-30 else 0.0
    cosine = (
        dot / math.sqrt(pred_energy * target_energy)
        if pred_energy > 1e-30 else 0.0
    )
    stage2_psnr = float(stage2_metric["PSNR"])
    oracle_psnr = float(oracle_metric["PSNR"])
    span = oracle_psnr - stage2_psnr
    realize = (
        (float(pred_metric["PSNR"]) - stage2_psnr) / span
        if abs(span) > 1e-12 else 0.0
    )
    return {
        "stage2_psnr": stage2_psnr,
        "pred_psnr": float(pred_metric["PSNR"]),
        "pred_sam": float(pred_metric["SAM"]),
        "oracle_psnr": oracle_psnr,
        "loss": float(_normalized_curvature_loss(out, target, cfg.curvature_loss_beta).item()),
        "curvature_capture": float(1.0 - error / target_energy),
        "amplitude_ratio": float(amp),
        "cosine": float(cosine),
        "oracle_realization": float(realize),
    }


@torch.no_grad()
def _mean_subset_loss(model, dataset, indices: Sequence[int], cfg, device) -> float:
    model.eval()
    total = 0.0
    for index in indices:
        batch = _batchify(dataset[index], device)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        targets = build_targets(model, out, batch["gt"])
        total += float(_normalized_curvature_loss(out, targets["curvature"], cfg.curvature_loss_beta).item())
    return total / max(len(indices), 1)


def _train_subset(
    model,
    dataset,
    indices: Sequence[int],
    steps: int,
    lr: float,
    weight_decay: float,
    eval_interval: int,
    cfg,
    device,
    label: str,
):
    if len(indices) == 0:
        raise ValueError(f"{label} has no training patches")
    optimizer = torch.optim.AdamW(
        model.proposal_predictor.parameters(), lr=lr, weight_decay=weight_decay
    )
    best_state = copy.deepcopy(model.proposal_predictor.state_dict())
    best_loss = _mean_subset_loss(model, dataset, indices, cfg, device)
    best_step = 0
    trajectory = [{"step": 0, "subset_loss": best_loss}]

    for step in range(1, steps + 1):
        model.train()
        index = indices[(step - 1) % len(indices)]
        batch = _batchify(dataset[index], device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        targets = build_targets(model, out, batch["gt"])
        loss = _normalized_curvature_loss(out, targets["curvature"], cfg.curvature_loss_beta)
        loss.backward()
        if cfg.curvature_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.proposal_predictor.parameters(), cfg.curvature_grad_clip
            )
        optimizer.step()

        if step % eval_interval == 0 or step == steps:
            subset_loss = _mean_subset_loss(model, dataset, indices, cfg, device)
            trajectory.append({"step": int(step), "subset_loss": float(subset_loss)})
            print(f"{label} step={step:04d} subset_loss={subset_loss:.6f}")
            if subset_loss < best_loss:
                best_loss = float(subset_loss)
                best_step = int(step)
                best_state = copy.deepcopy(model.proposal_predictor.state_dict())

    model.proposal_predictor.load_state_dict(best_state, strict=True)
    return {
        "best_step": int(best_step),
        "best_subset_loss": float(best_loss),
        "trajectory": trajectory,
    }


def _checkpoint_extra(cfg, role: str, cluster: int | None, members: Sequence[int]):
    return {
        "model_role": role,
        "experiment": "E26 LR-HSI state-cluster curvature specialists",
        "dataset": cfg.dataset,
        "curvature_rank": int(cfg.curvature_rank),
        "source_checkpoint": cfg.curvature_checkpoint,
        "state_clusters": int(cfg.state_clusters),
        "cluster": None if cluster is None else int(cluster),
        "member_indices": [int(v) for v in members],
        "routing": "nearest spherical-kmeans center in standardized LR-HSI manifold descriptor",
        "test_gt_used_for_routing": False,
        "test_gt_used_for_checkpoint_selection": False,
    }


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    train_set, test_set, info = build_datasets(cfg)
    train_set.augment = False
    if len(test_set) != 1:
        raise ValueError("E26 v1 expects exactly one held-out test image")
    if len(train_set) < cfg.state_clusters:
        raise ValueError("Not enough train patches for requested state_clusters")

    model, local_epoch, local_best, curvature_epoch, curvature_best = build_model(cfg, info, device)
    real_batch = _batchify(test_set[0], device)
    global_baseline = _evaluate(model, real_batch, cfg)

    train_descs = []
    with torch.no_grad():
        for index in range(len(train_set)):
            batch = _batchify(train_set[index], device)
            out = model(batch["lr_hsi"], batch["hr_msi"])
            train_descs.append(_lr_manifold_descriptor(model, out))
        real_out = model(real_batch["lr_hsi"], real_batch["hr_msi"])
        test_desc = _lr_manifold_descriptor(model, real_out)

    train_descs = np.stack(train_descs, axis=0)
    test_desc = test_desc.reshape(1, -1)
    mean, std, keep = _fit_standardizer(train_descs)
    train_z = _transform_descriptor(train_descs, mean, std, keep)
    test_z = _transform_descriptor(test_desc, mean, std, keep)
    labels, centers = _spherical_kmeans(
        train_z, cfg.state_clusters, cfg.kmeans_iterations
    )
    cluster_members = [np.where(labels == k)[0].tolist() for k in range(cfg.state_clusters)]
    cluster_sizes = [len(v) for v in cluster_members]
    if any(size == 0 for size in cluster_sizes):
        raise RuntimeError(f"Spherical k-means produced empty cluster: {cluster_sizes}")

    test_center_similarity = (test_z @ centers.T).reshape(-1)
    routed_cluster = int(np.argmax(test_center_similarity))
    sorted_sim = np.sort(test_center_similarity)[::-1]
    routing_margin = float(sorted_sim[0] - sorted_sim[1]) if len(sorted_sim) > 1 else float("nan")

    print(
        "E26 state-cluster experts | "
        f"dataset={cfg.dataset} K={cfg.state_clusters} train_patches={len(train_set)} "
        f"checkpoint_epoch={curvature_epoch} checkpoint_best={curvature_best:.4f}"
    )
    print(
        "Global baseline | "
        f"PSNR={global_baseline['pred_psnr']:.4f} SAM={global_baseline['pred_sam']:.4f} "
        f"CurvCap={100.0*global_baseline['curvature_capture']:.2f}% "
        f"Cos={global_baseline['cosine']:.3f}"
    )
    print(f"Cluster sizes | {cluster_sizes}")
    print(
        "Legal routing | "
        f"test-center similarities={[round(float(v), 4) for v in test_center_similarity]} "
        f"routed_cluster={routed_cluster} margin={routing_margin:.4f}"
    )

    ckpt_dir = os.path.join(
        cfg.checkpoint_root, "state_cluster_curvature_experts", cfg.dataset
    )
    out_dir = os.path.join(
        cfg.output_root, "state_cluster_curvature_experts", cfg.dataset
    )
    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)

    expert_records = []
    routed_result = None

    for cluster, members in enumerate(cluster_members):
        load_checkpoint(
            model,
            cfg.curvature_checkpoint,
            optimizer=None,
            strict=True,
            map_location=device,
            load_optimizer=False,
        )
        model.eval()
        print(f"\n--- Expert {cluster}/{cfg.state_clusters-1} members={len(members)} ---")
        training = _train_subset(
            model,
            train_set,
            members,
            cfg.expert_steps,
            cfg.expert_lr,
            cfg.expert_weight_decay,
            cfg.expert_eval_interval,
            cfg,
            device,
            label=f"Expert{cluster}",
        )
        test_stat = _evaluate(model, real_batch, cfg)
        expert_path = os.path.join(ckpt_dir, f"expert_{cluster}.pth")
        save_checkpoint(
            model,
            optimizer=None,
            epoch=training["best_step"],
            best_metric=-training["best_subset_loss"],
            path=expert_path,
            extra=_checkpoint_extra(cfg, "state_cluster_curvature_expert", cluster, members),
        )
        record = {
            "cluster": int(cluster),
            "members": [int(v) for v in members],
            "size": int(len(members)),
            "center_similarity_to_test": float(test_center_similarity[cluster]),
            "training": training,
            "test_diagnostic": test_stat,
            "checkpoint": expert_path,
        }
        expert_records.append(record)
        print(
            f"Expert{cluster} test | PSNR={test_stat['pred_psnr']:.4f} "
            f"SAM={test_stat['pred_sam']:.4f} CurvCap={100.0*test_stat['curvature_capture']:.2f}% "
            f"Cos={test_stat['cosine']:.3f} OracleRealize={100.0*test_stat['oracle_realization']:.2f}%"
        )
        if cluster == routed_cluster:
            routed_result = test_stat

    load_checkpoint(
        model,
        cfg.curvature_checkpoint,
        optimizer=None,
        strict=True,
        map_location=device,
        load_optimizer=False,
    )
    all_indices = list(range(len(train_set)))
    print("\n--- Global-continue control ---")
    global_training = _train_subset(
        model,
        train_set,
        all_indices,
        cfg.global_control_steps,
        cfg.global_control_lr,
        cfg.expert_weight_decay,
        cfg.global_control_eval_interval,
        cfg,
        device,
        label="GlobalContinue",
    )
    global_continue = _evaluate(model, real_batch, cfg)
    global_path = os.path.join(ckpt_dir, "global_continue.pth")
    save_checkpoint(
        model,
        optimizer=None,
        epoch=global_training["best_step"],
        best_metric=-global_training["best_subset_loss"],
        path=global_path,
        extra=_checkpoint_extra(cfg, "state_cluster_global_continue_control", None, all_indices),
    )

    oracle_record = max(
        expert_records,
        key=lambda item: item["test_diagnostic"]["pred_psnr"],
    )
    oracle_cluster = int(oracle_record["cluster"])
    oracle_result = oracle_record["test_diagnostic"]
    if routed_result is None:
        raise RuntimeError("Routed expert result was not recorded")

    print(
        "E26 Global-continue | "
        f"PSNR={global_continue['pred_psnr']:.4f} "
        f"CurvCap={100.0*global_continue['curvature_capture']:.2f}% "
        f"Cos={global_continue['cosine']:.3f}"
    )
    print(
        "E26 SUMMARY | "
        f"Baseline={global_baseline['pred_psnr']:.4f} | "
        f"GlobalContinue={global_continue['pred_psnr']:.4f} "
        f"({global_continue['pred_psnr']-global_baseline['pred_psnr']:+.4f}) | "
        f"RoutedExpert={routed_result['pred_psnr']:.4f} "
        f"cluster={routed_cluster} "
        f"({routed_result['pred_psnr']-global_baseline['pred_psnr']:+.4f}) | "
        f"OracleExpert={oracle_result['pred_psnr']:.4f} cluster={oracle_cluster} "
        f"({oracle_result['pred_psnr']-global_baseline['pred_psnr']:+.4f}) | "
        f"OracleGap={oracle_result['pred_psnr']-routed_result['pred_psnr']:+.4f}"
    )

    result = {
        "experiment": "E26 LR-HSI state-cluster curvature specialists",
        "dataset": cfg.dataset,
        "source_checkpoint": cfg.curvature_checkpoint,
        "source_checkpoint_epoch": int(curvature_epoch),
        "source_checkpoint_best": float(curvature_best),
        "local_epoch": int(local_epoch),
        "local_best": float(local_best),
        "curvature_rank": int(cfg.curvature_rank),
        "state_clusters": int(cfg.state_clusters),
        "information_boundary": {
            "clustering_descriptor": "LR-HSI manifold only; no GT",
            "routing": "nearest state center; no test GT",
            "expert_training": "train-native cluster GT only",
            "expert_checkpoint_selection": "cluster training loss only",
            "test_gt_used_for_oracle_expert_only": True,
        },
        "cluster_sizes": cluster_sizes,
        "labels": labels.tolist(),
        "test_center_similarity": [float(v) for v in test_center_similarity],
        "routed_cluster": int(routed_cluster),
        "routing_margin": float(routing_margin),
        "global_baseline": global_baseline,
        "global_continue_control": {
            "training": global_training,
            "test_diagnostic": global_continue,
            "checkpoint": global_path,
        },
        "experts": expert_records,
        "routed_expert": {
            "cluster": int(routed_cluster),
            "test_diagnostic": routed_result,
        },
        "oracle_best_expert": {
            "cluster": int(oracle_cluster),
            "test_diagnostic": oracle_result,
        },
        "summary": {
            "baseline_psnr": float(global_baseline["pred_psnr"]),
            "global_continue_psnr": float(global_continue["pred_psnr"]),
            "routed_psnr": float(routed_result["pred_psnr"]),
            "oracle_expert_psnr": float(oracle_result["pred_psnr"]),
            "global_continue_gain": float(global_continue["pred_psnr"] - global_baseline["pred_psnr"]),
            "routed_gain": float(routed_result["pred_psnr"] - global_baseline["pred_psnr"]),
            "oracle_expert_gain": float(oracle_result["pred_psnr"] - global_baseline["pred_psnr"]),
            "oracle_routing_gap": float(oracle_result["pred_psnr"] - routed_result["pred_psnr"]),
        },
    }
    out_path = os.path.join(out_dir, "e26_state_cluster_curvature_experts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
