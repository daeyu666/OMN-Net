"""E25: local patch gradient-conflict diagnostic for curvature extrapolation.

E22/E23 established a striking gap: the fixed E17-b predictor realizes only a
fraction of the curvature oracle when one parameter set is shared across the
scene, while the same predictor can fit one fixed training patch essentially to
100% curvature capture.  E25 tests whether this is caused by local-task
interference.

For every deterministic train-native patch, this script:
1) computes the exact E17-b curvature training loss and its gradient with
   respect to the *entire* proposal predictor;
2) stores a normalized full-predictor gradient on CPU (float16 by default) and
   an exact float32 head-only gradient as a numerical control;
3) builds sign/gauge-invariant patch descriptors from legal LR-HSI-derived
   spectral-manifold quantities; and
4) compares pairwise gradient cosine with manifold-state similarity.

Primary analysis excludes spatially overlapping patch pairs, so a positive
state-similarity/gradient-alignment relation cannot be explained merely by two
patches sharing pixels.  The script also reports all-pair statistics and a
secondary legal-context descriptor that adds HR-MSI residual statistics.

As an auxiliary diagnostic only, the source E17-b checkpoint gradients for the
held-out real task and the E24 lower-scale pseudo task are compared.  Real
HR-HSI GT is used only to define diagnostic gradients; no parameter is updated.

No training or checkpoint mutation occurs in E25.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import parse_args
from data_loader import build_datasets, make_lr_hsi
from models import LocalNullManifoldNet, build_spectral_response, load_foundation_checkpoint
from models.local_curvature_extrapolation_e17b import LocalCurvatureExtrapolationE17BNet
from train_local_curvature_extrapolation import build_targets
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

    p.add_argument(
        "--gradient_storage_dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
    )
    p.add_argument("--gradient_gram_chunk", type=int, default=8)
    p.add_argument(
        "--max_patches",
        type=int,
        default=0,
        help="0 means all train-native patches; otherwise use first N patches.",
    )
    p.add_argument("--pair_quantile", type=float, default=0.20)

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
    if cfg.gradient_gram_chunk < 1:
        raise ValueError("gradient_gram_chunk must be positive")
    if cfg.max_patches < 0:
        raise ValueError("max_patches must be >= 0")
    if not (0.0 < cfg.pair_quantile < 0.5):
        raise ValueError("pair_quantile must lie in (0,0.5)")
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


def _batchify(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict:
    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = value
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


def _flatten_gradients(
    parameters: Sequence[torch.nn.Parameter], dtype: torch.dtype
) -> torch.Tensor:
    pieces = []
    for parameter in parameters:
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            pieces.append(parameter.grad.detach().float().reshape(-1).cpu())
    vector = torch.cat(pieces, dim=0)
    norm = vector.norm().clamp_min(1e-30)
    return (vector / norm).to(dtype=dtype)


def _descriptor_from_output(model, out: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (LR-manifold-only descriptor, full legal-context descriptor)."""
    scale = out["coefficient_scale"].view(1, -1, 1, 1).detach()
    global_scale = scale.mean().clamp_min(1e-8)

    lr_coeff = out["lr_coefficients"].detach() / scale
    lr_null = model.local_model.geometry.project_null(out["lr_coefficients"].detach()) / scale
    tangent_singular = out["tangent_singular_values"].detach() / global_scale
    curvature_singular = out["curvature_singular_values"].detach() / global_scale
    projector_diag = out["curvature_projector_diagonal"].detach()

    signed = out["signed_projected_curvature_bank"].detach() / scale.unsqueeze(1)
    signed_rms = signed.square().mean(dim=(2, 3, 4)).sqrt()  # [1,8]

    def mean_std(x: torch.Tensor) -> List[torch.Tensor]:
        return [x.mean(dim=(-2, -1)), x.std(dim=(-2, -1), unbiased=False)]

    manifold_parts = []
    manifold_parts += mean_std(lr_coeff)
    manifold_parts += mean_std(lr_null)
    manifold_parts += mean_std(tangent_singular)
    manifold_parts += mean_std(curvature_singular)
    manifold_parts += mean_std(projector_diag)
    manifold_parts.append(signed_rms)
    manifold = torch.cat(manifold_parts, dim=1).squeeze(0).float().cpu().numpy()

    # Secondary descriptor: add only legal HR-MSI observable/context statistics.
    msi_residual = out["msi_residual"].detach()
    base_msi = out["base_msi"].detach()
    legal_parts = [torch.from_numpy(manifold).view(1, -1).to(msi_residual.device)]
    legal_parts += mean_std(msi_residual)
    legal_parts += mean_std(base_msi)
    legal = torch.cat(legal_parts, dim=1).squeeze(0).float().cpu().numpy()
    return manifold.astype(np.float64), legal.astype(np.float64)


def _patch_gradient_and_descriptor(model, batch: Dict[str, torch.Tensor], cfg):
    parameters = [p for p in model.proposal_predictor.parameters() if p.requires_grad]
    head_parameters = [
        p for p in model.proposal_predictor.head.parameters() if p.requires_grad
    ]
    model.zero_grad(set_to_none=True)
    out = model(batch["lr_hsi"], batch["hr_msi"])
    targets = build_targets(model, out, batch["gt"])
    loss = _normalized_curvature_loss(out, targets["curvature"], cfg.curvature_loss_beta)
    loss.backward()

    storage_dtype = torch.float16 if cfg.gradient_storage_dtype == "float16" else torch.float32
    full_grad = _flatten_gradients(parameters, storage_dtype)
    head_grad = _flatten_gradients(head_parameters, torch.float32)
    manifold_desc, legal_desc = _descriptor_from_output(model, out)

    with torch.no_grad():
        pred = out["curvature_residual"].double()
        target = targets["curvature"].double()
        pred_energy = float(pred.square().sum().item())
        target_energy = max(float(target.square().sum().item()), 1e-30)
        dot = float((pred * target).sum().item())
        cosine = (
            dot / math.sqrt(pred_energy * target_energy)
            if pred_energy > 1e-30
            else 0.0
        )
        capture = 1.0 - float((pred - target).square().sum().item()) / target_energy
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach().item()),
        "full_grad": full_grad,
        "head_grad": head_grad,
        "manifold_desc": manifold_desc,
        "legal_desc": legal_desc,
        "curvature_cosine": float(cosine),
        "curvature_capture": float(capture),
    }


def _cosine_gram_chunked(matrix: torch.Tensor, chunk: int) -> np.ndarray:
    """Cosine Gram for row-normalized CPU vectors without a full float32 copy."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be [N,D]")
    n = matrix.size(0)
    norms = torch.empty(n, dtype=torch.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = matrix[start:stop].float()
        norms[start:stop] = block.double().square().sum(dim=1).sqrt()
    gram = np.empty((n, n), dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        a = matrix[i0:i1].float()
        for j0 in range(0, n, chunk):
            j1 = min(j0 + chunk, n)
            b = matrix[j0:j1].float()
            dots = (a @ b.t()).double()
            denom = norms[i0:i1].view(-1, 1) * norms[j0:j1].view(1, -1)
            gram[i0:i1, j0:j1] = (dots / denom.clamp_min(1e-30)).cpu().numpy()
    np.fill_diagonal(gram, 1.0)
    return np.clip(gram, -1.0, 1.0)


def _standardized_cosine_similarity(descriptors: np.ndarray) -> np.ndarray:
    x = descriptors.astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    keep = std.reshape(-1) > 1e-10
    if not np.any(keep):
        return np.eye(x.shape[0], dtype=np.float64)
    z = (x[:, keep] - mean[:, keep]) / std[:, keep]
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(norms, 1e-12)
    sim = z @ z.T
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, -1.0, 1.0)


def _rect_overlap_fraction(
    a: Tuple[int, int], b: Tuple[int, int], patch_size: int
) -> float:
    at, al = a
    bt, bl = b
    ab, ar = at + patch_size, al + patch_size
    bb, br = bt + patch_size, bl + patch_size
    ih = max(0, min(ab, bb) - max(at, bt))
    iw = max(0, min(ar, br) - max(al, bl))
    return float(ih * iw) / float(patch_size * patch_size)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _corr(a: np.ndarray, b: np.ndarray, spearman: bool = False) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    x = _rankdata(a) if spearman else a.astype(np.float64)
    y = _rankdata(b) if spearman else b.astype(np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _partial_corr_xy_z(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rxy = _corr(x, y)
    rxz = _corr(x, z)
    ryz = _corr(y, z)
    denom = math.sqrt(max((1.0 - rxz * rxz) * (1.0 - ryz * ryz), 1e-30))
    return float((rxy - rxz * ryz) / denom)


def _pair_arrays(
    grad_cos: np.ndarray,
    state_sim: np.ndarray,
    coords: List[Tuple[int, int]],
    patch_size: int,
    nonoverlap_only: bool,
):
    gs, ss, dd, oo = [], [], [], []
    n = len(coords)
    for i in range(n):
        ci = np.asarray(coords[i], dtype=np.float64) + patch_size / 2.0
        for j in range(i + 1, n):
            overlap = _rect_overlap_fraction(coords[i], coords[j], patch_size)
            if nonoverlap_only and overlap > 0.0:
                continue
            cj = np.asarray(coords[j], dtype=np.float64) + patch_size / 2.0
            gs.append(grad_cos[i, j])
            ss.append(state_sim[i, j])
            dd.append(float(np.linalg.norm(ci - cj)))
            oo.append(overlap)
    return (
        np.asarray(gs, dtype=np.float64),
        np.asarray(ss, dtype=np.float64),
        np.asarray(dd, dtype=np.float64),
        np.asarray(oo, dtype=np.float64),
    )


def _pair_summary(
    grad_cos: np.ndarray,
    state_sim: np.ndarray,
    coords: List[Tuple[int, int]],
    patch_size: int,
    quantile: float,
    nonoverlap_only: bool,
) -> Dict[str, float]:
    g, s, d, overlap = _pair_arrays(
        grad_cos, state_sim, coords, patch_size, nonoverlap_only
    )
    if g.size == 0:
        return {"pair_count": 0}
    q_lo = float(np.quantile(s, quantile))
    q_hi = float(np.quantile(s, 1.0 - quantile))
    low = s <= q_lo
    high = s >= q_hi
    d_norm = d / max(float(d.max()), 1e-12)
    return {
        "pair_count": int(g.size),
        "gradient_cosine_mean": float(g.mean()),
        "gradient_cosine_median": float(np.median(g)),
        "negative_gradient_pair_fraction": float((g < 0.0).mean()),
        "state_gradient_pearson": _corr(s, g, spearman=False),
        "state_gradient_spearman": _corr(s, g, spearman=True),
        "state_gradient_partial_pearson_controlling_distance": _partial_corr_xy_z(s, g, d_norm),
        "spatial_distance_gradient_pearson": _corr(d_norm, g),
        "state_similarity_low_threshold": q_lo,
        "state_similarity_high_threshold": q_hi,
        "farthest_state_quantile_gradient_cosine": float(g[low].mean()),
        "nearest_state_quantile_gradient_cosine": float(g[high].mean()),
        "farthest_state_quantile_negative_fraction": float((g[low] < 0.0).mean()),
        "nearest_state_quantile_negative_fraction": float((g[high] < 0.0).mean()),
        "nearest_minus_farthest_gradient_cosine": float(g[high].mean() - g[low].mean()),
        "mean_overlap_fraction": float(overlap.mean()),
    }


def _pseudo_task_from_real_lr(model, real_batch: Dict[str, torch.Tensor], cfg, device):
    observed_lr = real_batch["lr_hsi"].detach()
    h, w = observed_lr.shape[-2:]
    if h % cfg.scale_ratio != 0 or w % cfg.scale_ratio != 0:
        raise ValueError("real LR-HSI dimensions must be divisible by scale_ratio")
    pseudo_gt_np = observed_lr[0].float().cpu().permute(1, 2, 0).contiguous().numpy()
    pseudo_lr_np = make_lr_hsi(pseudo_gt_np, cfg.scale_ratio)
    pseudo_lr = (
        torch.from_numpy(pseudo_lr_np)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(device=device, dtype=observed_lr.dtype)
    )
    pseudo_msi = model.local_model.geometry.hsi_to_msi(observed_lr).detach()
    return {"lr_hsi": pseudo_lr, "hr_msi": pseudo_msi, "gt": observed_lr}


def _gradient_cosine_for_two_tasks(model, a: Dict, b: Dict, cfg) -> Dict[str, float]:
    parameters = [p for p in model.proposal_predictor.parameters() if p.requires_grad]

    def one(batch):
        model.zero_grad(set_to_none=True)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        targets = build_targets(model, out, batch["gt"])
        loss = _normalized_curvature_loss(out, targets["curvature"], cfg.curvature_loss_beta)
        loss.backward()
        grad = _flatten_gradients(parameters, torch.float32)
        value = float(loss.detach().item())
        model.zero_grad(set_to_none=True)
        return grad, value

    ga, la = one(a)
    gb, lb = one(b)
    cosine = float(torch.dot(ga, gb).item())
    return {"gradient_cosine": cosine, "task_a_loss": la, "task_b_loss": lb}


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    train_set, test_set, info = build_datasets(cfg)
    train_set.augment = False
    if len(train_set) < 2:
        raise RuntimeError("E25 requires at least two training patches")

    patch_indices = list(range(len(train_set)))
    if cfg.max_patches > 0:
        patch_indices = patch_indices[: min(cfg.max_patches, len(patch_indices))]
    if len(patch_indices) < 2:
        raise RuntimeError("Selected patch count must be at least two")

    model, local_epoch, local_best, curvature_epoch, curvature_best = build_model(
        cfg, info, device
    )
    model.eval()

    parameters = [p for p in model.proposal_predictor.parameters() if p.requires_grad]
    full_parameter_count = sum(p.numel() for p in parameters)
    head_parameter_count = sum(p.numel() for p in model.proposal_predictor.head.parameters())
    bytes_per = 2 if cfg.gradient_storage_dtype == "float16" else 4
    estimated_mb = len(patch_indices) * full_parameter_count * bytes_per / (1024.0 ** 2)

    print(
        "E25 patch gradient conflict | "
        f"dataset={cfg.dataset} patches={len(patch_indices)}/{len(train_set)} "
        f"checkpoint_epoch={curvature_epoch} checkpoint_best={curvature_best:.4f} "
        f"predictor_params={full_parameter_count} head_params={head_parameter_count}"
    )
    print(
        "Gradient storage | "
        f"dtype={cfg.gradient_storage_dtype} estimated={estimated_mb:.1f} MiB CPU | "
        "full predictor gradients are normalized before storage"
    )

    full_grads = []
    head_grads = []
    manifold_descs = []
    legal_descs = []
    coords = []
    patch_records = []

    for k, index in enumerate(patch_indices):
        batch = _batchify(train_set[index], device)
        stat = _patch_gradient_and_descriptor(model, batch, cfg)
        full_grads.append(stat.pop("full_grad"))
        head_grads.append(stat.pop("head_grad"))
        manifold_descs.append(stat.pop("manifold_desc"))
        legal_descs.append(stat.pop("legal_desc"))
        coord = tuple(int(v) for v in train_set.coords[index])
        coords.append(coord)
        patch_records.append({"index": int(index), "coord": list(coord), **stat})
        print(
            f"Patch {k+1:03d}/{len(patch_indices):03d} index={index:03d} "
            f"coord={coord} loss={stat['loss']:.6f} "
            f"CurvCap={100.0*stat['curvature_capture']:.2f}% "
            f"Cos={stat['curvature_cosine']:.3f}"
        )

    full_matrix = torch.stack(full_grads, dim=0).contiguous()
    head_matrix = torch.stack(head_grads, dim=0).contiguous()
    manifold_array = np.stack(manifold_descs, axis=0)
    legal_array = np.stack(legal_descs, axis=0)

    print("Computing pairwise full-predictor gradient cosine matrix...")
    full_cos = _cosine_gram_chunked(full_matrix, cfg.gradient_gram_chunk)
    head_cos = _cosine_gram_chunked(head_matrix, max(cfg.gradient_gram_chunk, 16))
    manifold_sim = _standardized_cosine_similarity(manifold_array)
    legal_sim = _standardized_cosine_similarity(legal_array)

    primary = _pair_summary(
        full_cos,
        manifold_sim,
        coords,
        cfg.patch_size,
        cfg.pair_quantile,
        nonoverlap_only=True,
    )
    all_pairs = _pair_summary(
        full_cos,
        manifold_sim,
        coords,
        cfg.patch_size,
        cfg.pair_quantile,
        nonoverlap_only=False,
    )
    legal_primary = _pair_summary(
        full_cos,
        legal_sim,
        coords,
        cfg.patch_size,
        cfg.pair_quantile,
        nonoverlap_only=True,
    )
    head_primary = _pair_summary(
        head_cos,
        manifold_sim,
        coords,
        cfg.patch_size,
        cfg.pair_quantile,
        nonoverlap_only=True,
    )

    tri = np.triu_indices(len(coords), k=1)
    full_head_agreement = _corr(full_cos[tri], head_cos[tri])

    print(
        "E25 PRIMARY non-overlap | "
        f"pairs={primary.get('pair_count', 0)} "
        f"GradCos={primary.get('gradient_cosine_mean', float('nan')):+.3f} "
        f"negative={100.0*primary.get('negative_gradient_pair_fraction', float('nan')):.2f}% | "
        f"state->grad Pearson={primary.get('state_gradient_pearson', float('nan')):+.3f} "
        f"Spearman={primary.get('state_gradient_spearman', float('nan')):+.3f} "
        f"partial(distance)={primary.get('state_gradient_partial_pearson_controlling_distance', float('nan')):+.3f}"
    )
    print(
        "E25 STATE QUANTILES non-overlap | "
        f"nearest {100.0*cfg.pair_quantile:.0f}% GradCos="
        f"{primary.get('nearest_state_quantile_gradient_cosine', float('nan')):+.3f} "
        f"neg={100.0*primary.get('nearest_state_quantile_negative_fraction', float('nan')):.2f}% | "
        f"farthest {100.0*cfg.pair_quantile:.0f}% GradCos="
        f"{primary.get('farthest_state_quantile_gradient_cosine', float('nan')):+.3f} "
        f"neg={100.0*primary.get('farthest_state_quantile_negative_fraction', float('nan')):.2f}% | "
        f"delta={primary.get('nearest_minus_farthest_gradient_cosine', float('nan')):+.3f}"
    )
    print(
        "E25 CONTROLS | "
        f"all-pair GradCos={all_pairs.get('gradient_cosine_mean', float('nan')):+.3f} "
        f"negative={100.0*all_pairs.get('negative_gradient_pair_fraction', float('nan')):.2f}% | "
        f"head/full pair-cos corr={full_head_agreement:+.3f} | "
        f"legal-context state->grad Pearson="
        f"{legal_primary.get('state_gradient_pearson', float('nan')):+.3f}"
    )

    pseudo_real = None
    if len(test_set) == 1:
        real_batch = _batchify(test_set[0], device)
        pseudo_batch = _pseudo_task_from_real_lr(model, real_batch, cfg, device)
        pseudo_real = _gradient_cosine_for_two_tasks(
            model, pseudo_batch, real_batch, cfg
        )
        print(
            "E25 E24 gradient control | "
            f"pseudo-vs-real GradCos={pseudo_real['gradient_cosine']:+.3f} "
            f"pseudo_loss={pseudo_real['task_a_loss']:.6f} "
            f"real_loss={pseudo_real['task_b_loss']:.6f}"
        )

    result = {
        "experiment": "E25 local patch gradient conflict diagnostic",
        "dataset": cfg.dataset,
        "checkpoint": cfg.curvature_checkpoint,
        "checkpoint_epoch": int(curvature_epoch),
        "checkpoint_best": float(curvature_best),
        "local_epoch": int(local_epoch),
        "local_best": float(local_best),
        "curvature_rank": int(cfg.curvature_rank),
        "patch_size": int(cfg.patch_size),
        "stride": int(cfg.stride),
        "patch_count": int(len(patch_indices)),
        "full_predictor_parameter_count": int(full_parameter_count),
        "head_parameter_count": int(head_parameter_count),
        "gradient_storage_dtype": cfg.gradient_storage_dtype,
        "estimated_gradient_storage_mib": float(estimated_mb),
        "descriptor": {
            "primary": (
                "LR coefficients mean/std + LR null coefficients mean/std + "
                "tangent singular mean/std + curvature singular mean/std + "
                "curvature projector diagonal mean/std + signed curvature RMS"
            ),
            "secondary_legal_context": "primary + MSI residual/base-MSI mean/std",
            "standardization": "per descriptor dimension across patches, then row L2 normalization",
        },
        "primary_nonoverlap_lr_manifold": primary,
        "all_pairs_lr_manifold": all_pairs,
        "primary_nonoverlap_legal_context": legal_primary,
        "head_gradient_primary_nonoverlap_lr_manifold": head_primary,
        "full_head_pairwise_cosine_correlation": float(full_head_agreement),
        "pseudo_real_gradient_control": pseudo_real,
        "patches": patch_records,
        "matrices": {
            "full_gradient_cosine": full_cos.tolist(),
            "head_gradient_cosine": head_cos.tolist(),
            "lr_manifold_similarity": manifold_sim.tolist(),
            "legal_context_similarity": legal_sim.tolist(),
        },
    }

    out_dir = os.path.join(
        cfg.output_root,
        "diagnostics",
        "curvature_patch_gradient_conflict",
        cfg.dataset,
    )
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "e25_patch_gradient_conflict.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
