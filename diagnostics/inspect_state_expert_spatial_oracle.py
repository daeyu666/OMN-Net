"""E27: spatial-granularity oracle over E17-b baseline + E26 experts.

E26 showed that whole-image nearest-state routing and even GT oracle selection of
one whole-image expert do not improve over the original E17-b checkpoint. E27
asks the final remaining question for the local-state/expert route: are the E26
experts nevertheless complementary at a finer spatial scale?

No training occurs here. The candidate set is fixed:
    candidate 0: original E17-b baseline checkpoint
    candidates 1..K: E26 state-cluster expert checkpoints

For the held-out test image, GT is used only to construct diagnostic spatial
oracles. At each requested block size, one candidate is selected per block by
minimum summed squared HSI reconstruction error across all spectral bands and
pixels in that block. Block size 1 is the pixel oracle; block size equal to the
image size is the whole-image oracle.

The script reports both baseline+experts and expert-only oracles, together with
candidate selection fractions. This distinguishes genuine expert complementarity
from an oracle that mostly falls back to the original baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import parse_args
from data_loader import build_datasets
from metrics import calc_metrics
from train_state_cluster_curvature_experts import build_model, _batchify
from utils import ensure_dir, get_device, load_checkpoint, set_seed


def _has_option(arguments: List[str], option: str) -> bool:
    return any(x == option or x.startswith(option + "=") for x in arguments)


def _parse_block_sizes(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("oracle block sizes must be positive integers")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one oracle block size is required")
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
    p.add_argument(
        "--expert_root",
        type=str,
        default="./checkpoints/state_cluster_curvature_experts/PaviaU",
    )
    p.add_argument("--expert_count", type=int, default=4)
    p.add_argument("--expert_filename", type=str, default="expert_{index}.pth")
    p.add_argument(
        "--oracle_block_sizes",
        type=str,
        default="1,4,8,16,128",
        help="Comma-separated HR block sizes. 1 is pixel oracle.",
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
    p.add_argument(
        "--curvature_proposal_amplitude_multiplier", type=float, default=8.0
    )
    p.add_argument("--curvature_predictor_hidden", type=int, default=96)
    p.add_argument("--curvature_predictor_blocks", type=int, default=4)

    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"
    cfg.image_size = cfg.diagnostic_image_size
    cfg.oracle_block_sizes = _parse_block_sizes(cfg.oracle_block_sizes)
    if cfg.expert_count < 1:
        raise ValueError("expert_count must be >= 1")
    if cfg.curvature_rank < 1 or cfg.curvature_rank > 8:
        raise ValueError("curvature_rank must be in [1,8]")
    return cfg


@torch.no_grad()
def _prediction(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    out = model(batch["lr_hsi"], batch["hr_msi"])
    return out["curvature_reconstructed_hsi"].detach()


def _selection_fractions(index_map: torch.Tensor, candidate_count: int) -> List[float]:
    total = float(index_map.numel())
    return [
        float((index_map == index).sum().item()) / max(total, 1.0)
        for index in range(candidate_count)
    ]


@torch.no_grad()
def _block_oracle(
    candidates: torch.Tensor,
    gt: torch.Tensor,
    block_size: int,
):
    """Select one candidate per spatial block by spectral-spatial SSE.

    candidates: [K,C,H,W]
    gt:         [1,C,H,W]
    """
    if candidates.ndim != 4 or gt.ndim != 4 or gt.size(0) != 1:
        raise ValueError("invalid candidate/GT shapes")
    k, _, h, w = candidates.shape
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(
            f"block_size={block_size} must divide test size {(h, w)} exactly"
        )

    gt0 = gt[0]
    squared = (candidates.double() - gt0.double().unsqueeze(0)).square()
    # [K,C,H,W] -> [K,H,W] spectral squared error.
    pixel_error = squared.sum(dim=1)

    out = torch.empty_like(candidates[0])
    block_choice = torch.empty(
        h // block_size,
        w // block_size,
        device=candidates.device,
        dtype=torch.long,
    )
    pixel_choice = torch.empty(h, w, device=candidates.device, dtype=torch.long)

    for by, top in enumerate(range(0, h, block_size)):
        for bx, left in enumerate(range(0, w, block_size)):
            block_error = pixel_error[
                :, top : top + block_size, left : left + block_size
            ].sum(dim=(1, 2))
            choice = int(torch.argmin(block_error).item())
            block_choice[by, bx] = choice
            pixel_choice[
                top : top + block_size, left : left + block_size
            ] = choice
            out[:, top : top + block_size, left : left + block_size] = candidates[
                choice, :, top : top + block_size, left : left + block_size
            ]

    return out.unsqueeze(0), block_choice, pixel_choice


def _format_shares(names: Sequence[str], shares: Sequence[float]) -> str:
    return " ".join(
        f"{name}={100.0 * share:.1f}%" for name, share in zip(names, shares)
    )


def main():
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    _, test_set, info = build_datasets(cfg)
    if len(test_set) != 1:
        raise ValueError("E27 expects exactly one held-out test image")
    batch = _batchify(test_set[0], device)

    model, local_epoch, local_best, curvature_epoch, curvature_best = build_model(
        cfg, info, device
    )

    expert_paths = [
        os.path.join(
            cfg.expert_root,
            cfg.expert_filename.format(index=index),
        )
        for index in range(cfg.expert_count)
    ]
    for path in expert_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing E26 expert checkpoint: {path}. Run E26 first or set --expert_root."
            )

    candidate_names = ["baseline"] + [f"expert{index}" for index in range(cfg.expert_count)]
    candidate_paths = [cfg.curvature_checkpoint] + expert_paths
    predictions = []
    individual = []

    for name, path in zip(candidate_names, candidate_paths):
        load_checkpoint(
            model,
            path,
            optimizer=None,
            strict=True,
            map_location=device,
            load_optimizer=False,
        )
        pred = _prediction(model, batch)
        predictions.append(pred[0])
        metric = calc_metrics(pred, batch["gt"], cfg.scale_ratio)
        individual.append(
            {
                "name": name,
                "checkpoint": path,
                "psnr": float(metric["PSNR"]),
                "sam": float(metric["SAM"]),
                "rmse": float(metric["RMSE"]),
            }
        )
        print(
            f"Candidate {name} | PSNR={metric['PSNR']:.4f} "
            f"SAM={metric['SAM']:.4f} RMSE={metric['RMSE']:.6f}"
        )

    candidates = torch.stack(predictions, dim=0)
    _, _, h, w = candidates.shape
    for block_size in cfg.oracle_block_sizes:
        if h % block_size != 0 or w % block_size != 0:
            raise ValueError(
                f"Requested block size {block_size} does not divide test image {(h, w)}"
            )

    baseline_psnr = individual[0]["psnr"]
    best_single = max(individual, key=lambda item: item["psnr"])

    print(
        "E27 spatial expert oracle | "
        f"dataset={cfg.dataset} candidates={candidate_names} test={(h, w)} "
        f"source_checkpoint_epoch={curvature_epoch} source_best={curvature_best:.4f}"
    )

    all_oracles = []
    expert_only_oracles = []
    expert_candidates = candidates[1:]
    expert_names = candidate_names[1:]

    for block_size in cfg.oracle_block_sizes:
        oracle, block_choice, pixel_choice = _block_oracle(
            candidates, batch["gt"], block_size
        )
        metric = calc_metrics(oracle, batch["gt"], cfg.scale_ratio)
        block_shares = _selection_fractions(block_choice, len(candidate_names))
        pixel_shares = _selection_fractions(pixel_choice, len(candidate_names))
        record = {
            "block_size": int(block_size),
            "psnr": float(metric["PSNR"]),
            "sam": float(metric["SAM"]),
            "rmse": float(metric["RMSE"]),
            "gain_over_baseline": float(metric["PSNR"] - baseline_psnr),
            "gain_over_best_single": float(metric["PSNR"] - best_single["psnr"]),
            "block_selection_fraction": block_shares,
            "pixel_coverage_fraction": pixel_shares,
        }
        all_oracles.append(record)
        label = "Pixel" if block_size == 1 else (
            "Whole" if block_size == h and block_size == w else f"{block_size}x{block_size}"
        )
        print(
            f"E27 ALL {label} Oracle | PSNR={metric['PSNR']:.4f} "
            f"SAM={metric['SAM']:.4f} gain={metric['PSNR']-baseline_psnr:+.4f} | "
            f"blocks[{_format_shares(candidate_names, block_shares)}] | "
            f"pixels[{_format_shares(candidate_names, pixel_shares)}]"
        )

        expert_oracle, expert_block_choice, expert_pixel_choice = _block_oracle(
            expert_candidates, batch["gt"], block_size
        )
        expert_metric = calc_metrics(expert_oracle, batch["gt"], cfg.scale_ratio)
        expert_block_shares = _selection_fractions(
            expert_block_choice, len(expert_names)
        )
        expert_pixel_shares = _selection_fractions(
            expert_pixel_choice, len(expert_names)
        )
        expert_record = {
            "block_size": int(block_size),
            "psnr": float(expert_metric["PSNR"]),
            "sam": float(expert_metric["SAM"]),
            "rmse": float(expert_metric["RMSE"]),
            "gain_over_baseline": float(expert_metric["PSNR"] - baseline_psnr),
            "block_selection_fraction": expert_block_shares,
            "pixel_coverage_fraction": expert_pixel_shares,
        }
        expert_only_oracles.append(expert_record)
        print(
            f"E27 EXPERT-ONLY {label} Oracle | PSNR={expert_metric['PSNR']:.4f} "
            f"gain={expert_metric['PSNR']-baseline_psnr:+.4f} | "
            f"pixels[{_format_shares(expert_names, expert_pixel_shares)}]"
        )

    pixel_record = next(
        (record for record in all_oracles if record["block_size"] == 1), None
    )
    lr_cell_record = next(
        (record for record in all_oracles if record["block_size"] == cfg.scale_ratio),
        None,
    )
    whole_record = next(
        (
            record
            for record in all_oracles
            if record["block_size"] == h and record["block_size"] == w
        ),
        None,
    )

    print(
        "E27 SUMMARY | "
        f"Baseline={baseline_psnr:.4f} "
        f"BestSingle={best_single['psnr']:.4f}({best_single['name']}) | "
        f"PixelOracle={pixel_record['psnr']:.4f if pixel_record else float('nan')}"
        if False
        else ""
    )
    # Keep the final summary explicit rather than relying on conditional f-format syntax.
    pixel_text = f"{pixel_record['psnr']:.4f}" if pixel_record is not None else "NA"
    cell_text = f"{lr_cell_record['psnr']:.4f}" if lr_cell_record is not None else "NA"
    whole_text = f"{whole_record['psnr']:.4f}" if whole_record is not None else "NA"
    print(
        "E27 SUMMARY | "
        f"Baseline={baseline_psnr:.4f} | "
        f"BestSingle={best_single['psnr']:.4f}({best_single['name']}) | "
        f"PixelOracle={pixel_text} | "
        f"LRcell{cfg.scale_ratio}x{cfg.scale_ratio}={cell_text} | "
        f"WholeOracle={whole_text}"
    )

    result = {
        "experiment": "E27 spatial-granularity oracle over baseline + E26 experts",
        "dataset": cfg.dataset,
        "curvature_rank": int(cfg.curvature_rank),
        "source_checkpoint": cfg.curvature_checkpoint,
        "source_checkpoint_epoch": int(curvature_epoch),
        "source_checkpoint_best": float(curvature_best),
        "local_epoch": int(local_epoch),
        "local_best": float(local_best),
        "expert_root": cfg.expert_root,
        "candidate_names": candidate_names,
        "individual": individual,
        "block_sizes": [int(v) for v in cfg.oracle_block_sizes],
        "baseline_plus_experts": all_oracles,
        "expert_only": expert_only_oracles,
        "information_boundary": {
            "training": "none",
            "test_gt_used_for_candidate_generation": False,
            "test_gt_used_for_oracle_selection": True,
            "purpose": "diagnostic upper bound only",
        },
        "summary": {
            "baseline_psnr": float(baseline_psnr),
            "best_single_name": best_single["name"],
            "best_single_psnr": float(best_single["psnr"]),
            "pixel_oracle_psnr": None if pixel_record is None else float(pixel_record["psnr"]),
            "lr_cell_oracle_psnr": None if lr_cell_record is None else float(lr_cell_record["psnr"]),
            "whole_oracle_psnr": None if whole_record is None else float(whole_record["psnr"]),
        },
    }

    out_dir = os.path.join(
        cfg.output_root,
        "diagnostics",
        "state_expert_spatial_oracle",
        cfg.dataset,
    )
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "e27_state_expert_spatial_oracle.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
