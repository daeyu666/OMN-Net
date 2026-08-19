"""Prototype-consensus arbitration for OMN-Net innovation point 2.

A wide observable-key recall pool is high-recall but intentionally impure. This
module forms deterministic spectral-state prototypes from the LR-HSI null-state
memory itself, then asks which prototype is unusually enriched inside the
query's observable-compatible candidate pool relative to that prototype's
scene-wide prior frequency.

Only LR-HSI-derived null states provide values. The selected prototype can
modify the reconstruction only through the query-specific P_comp projection.
No GT, uncertainty, trust radius, or trainable parameter is used.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .nonlocal_complement import (
    flatten_spatial,
    flatten_tangent,
    gather_complement_candidates,
    unflatten_spatial,
)


def deterministic_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    iterations: int = 20,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic farthest-point initialized k-means.

    Args:
        x: [M,D].
    Returns:
        labels [M], centers [C,D], prior [C].
    """
    if x.ndim != 2:
        raise ValueError("x must be [M,D]")
    m = x.size(0)
    if m < 2:
        raise ValueError("memory must contain at least two states")
    c = min(int(n_clusters), m)
    if c < 2:
        raise ValueError("n_clusters must be at least two")

    mean = x.mean(dim=0, keepdim=True)
    first = (x - mean).square().sum(dim=1).argmin()
    centers = [x[first]]
    min_dist = (x - centers[0]).square().sum(dim=1)
    for _ in range(1, c):
        index = min_dist.argmax()
        new_center = x[index]
        centers.append(new_center)
        min_dist = torch.minimum(
            min_dist, (x - new_center).square().sum(dim=1)
        )
    centers = torch.stack(centers, dim=0)

    labels = torch.zeros(m, dtype=torch.long, device=x.device)
    for _ in range(max(int(iterations), 1)):
        distance = torch.cdist(x.float(), centers.float(), p=2).square()
        new_labels = distance.argmin(dim=1)
        new_centers = centers.clone()
        for cluster in range(c):
            mask = new_labels == cluster
            if bool(mask.any()):
                new_centers[cluster] = x[mask].mean(dim=0)
        stable = torch.equal(new_labels, labels)
        labels = new_labels
        centers = new_centers
        if stable:
            break

    count = torch.bincount(labels, minlength=c).to(x.dtype)
    prior = count / count.sum().clamp_min(eps)
    return labels, centers, prior


class SpectralPrototypeConsensus(nn.Module):
    """Select a recurrent LR-HSI spectral-state prototype inside a wide recall pool.

    The prototype score is based on query-specific observable retrieval mass and
    enrichment over scene-wide prototype frequency:

        score_c = mass_c / prior_c**prior_exponent

    Thus a rare material can win when it is disproportionately concentrated in
    the query's observable-compatible pool, instead of being suppressed by a
    globally dominant material class.
    """

    def __init__(
        self,
        n_prototypes: int = 32,
        kmeans_iterations: int = 20,
        prior_exponent: float = 0.5,
        min_cluster_candidates: int = 4,
        query_chunk_pixels: int = 64,
        eps: float = 1e-8,
    ):
        super().__init__()
        if n_prototypes < 2:
            raise ValueError("n_prototypes must be at least two")
        if kmeans_iterations < 1:
            raise ValueError("kmeans_iterations must be positive")
        if not 0.0 <= prior_exponent <= 1.0:
            raise ValueError("prior_exponent must be in [0,1]")
        if min_cluster_candidates < 1 or query_chunk_pixels < 1 or eps <= 0:
            raise ValueError("invalid consensus settings")
        self.n_prototypes = int(n_prototypes)
        self.kmeans_iterations = int(kmeans_iterations)
        self.prior_exponent = float(prior_exponent)
        self.min_cluster_candidates = int(min_cluster_candidates)
        self.query_chunk_pixels = int(query_chunk_pixels)
        self.eps = float(eps)

    def _standardize_memory(self, memory: torch.Tensor) -> torch.Tensor:
        mean = memory.mean(dim=0, keepdim=True)
        scale = memory.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        return (memory - mean) / scale

    def forward(
        self,
        topk_indices: torch.Tensor,
        topk_observable_weights: torch.Tensor,
        memory_null: torch.Tensor,
        local_null_state: torch.Tensor,
        tangent_basis: torch.Tensor,
        null_projector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if topk_indices.ndim != 4 or topk_observable_weights.ndim != 4:
            raise ValueError("top-k fields must be [N,K,H,W]")
        if topk_indices.shape != topk_observable_weights.shape:
            raise ValueError("index and weight shapes differ")
        if memory_null.ndim != 4 or local_null_state.ndim != 4:
            raise ValueError("null states must be [N,R,H,W]")
        if tangent_basis.ndim != 5:
            raise ValueError("tangent_basis must be [N,R,D,H,W]")

        n, k, height, width = topk_indices.shape
        if local_null_state.shape[-2:] != (height, width):
            raise ValueError("query spatial sizes differ")
        if tangent_basis.shape[0] != n or tangent_basis.shape[-2:] != (height, width):
            raise ValueError("tangent/query spatial sizes differ")
        if memory_null.size(0) != n:
            raise ValueError("batch sizes differ")

        memory_flat = flatten_spatial(memory_null)
        local_flat = flatten_spatial(local_null_state)
        tangent_flat = flatten_tangent(tangent_basis)
        indices_flat = (
            topk_indices.permute(0, 2, 3, 1)
            .reshape(n, height * width, k)
            .contiguous()
        )
        weights_flat = (
            topk_observable_weights.permute(0, 2, 3, 1)
            .reshape(n, height * width, k)
            .contiguous()
        )

        rank = local_null_state.size(1)
        q_count = height * width
        soft_residual = local_null_state.new_zeros(n, q_count, rank)
        uniform_residual = local_null_state.new_zeros(n, q_count, rank)
        winner_cluster = torch.empty(
            n, q_count, dtype=torch.long, device=topk_indices.device
        )
        winner_mass = local_null_state.new_zeros(n, q_count, 1)
        winner_prior = local_null_state.new_zeros(n, q_count, 1)
        winner_enrichment = local_null_state.new_zeros(n, q_count, 1)
        winner_support = local_null_state.new_zeros(n, q_count, 1)
        winner_mask_flat = torch.zeros(
            n, q_count, k, dtype=torch.bool, device=topk_indices.device
        )
        memory_labels_all = []
        memory_prior_all = []

        for b in range(n):
            memory = memory_flat[b]
            standardized = self._standardize_memory(memory)
            labels, _, prior = deterministic_kmeans(
                standardized,
                n_clusters=self.n_prototypes,
                iterations=self.kmeans_iterations,
                eps=self.eps,
            )
            memory_labels_all.append(labels)
            memory_prior_all.append(prior)
            c = prior.numel()

            for start in range(0, q_count, self.query_chunk_pixels):
                stop = min(start + self.query_chunk_pixels, q_count)
                idx = indices_flat[b, start:stop]
                obs_w = weights_flat[b, start:stop]
                obs_w = obs_w / obs_w.sum(dim=1, keepdim=True).clamp_min(self.eps)
                candidate_labels = labels[idx]

                cluster_mass = obs_w.new_zeros(stop - start, c)
                cluster_count = obs_w.new_zeros(stop - start, c)
                cluster_mass.scatter_add_(1, candidate_labels, obs_w)
                cluster_count.scatter_add_(
                    1, candidate_labels, torch.ones_like(obs_w)
                )
                score = cluster_mass / prior.to(obs_w).clamp_min(self.eps).pow(
                    self.prior_exponent
                ).unsqueeze(0)
                score = score.masked_fill(
                    cluster_count < self.min_cluster_candidates,
                    float("-inf"),
                )
                invalid = ~torch.isfinite(score).any(dim=1)
                if bool(invalid.any()):
                    score[invalid] = cluster_mass[invalid]
                winner = score.argmax(dim=1)
                mask = candidate_labels == winner.unsqueeze(1)

                chosen_w = obs_w * mask.to(obs_w.dtype)
                chosen_w = chosen_w / chosen_w.sum(dim=1, keepdim=True).clamp_min(
                    self.eps
                )
                uniform_w = mask.to(obs_w.dtype)
                uniform_w = uniform_w / uniform_w.sum(
                    dim=1, keepdim=True
                ).clamp_min(self.eps)

                candidates = gather_complement_candidates(
                    memory,
                    local_flat[b, start:stop],
                    tangent_flat[b, start:stop],
                    idx,
                    null_projector,
                )
                soft_residual[b, start:stop] = torch.sum(
                    chosen_w.unsqueeze(-1) * candidates, dim=1
                )
                uniform_residual[b, start:stop] = torch.sum(
                    uniform_w.unsqueeze(-1) * candidates, dim=1
                )
                rows = torch.arange(stop - start, device=winner.device)
                mass = cluster_mass[rows, winner]
                p = prior.to(obs_w)[winner]
                count = cluster_count[rows, winner]

                winner_cluster[b, start:stop] = winner
                winner_mass[b, start:stop, 0] = mass
                winner_prior[b, start:stop, 0] = p
                winner_enrichment[b, start:stop, 0] = mass / p.clamp_min(self.eps)
                winner_support[b, start:stop, 0] = count
                winner_mask_flat[b, start:stop] = mask

        def field_1(x: torch.Tensor) -> torch.Tensor:
            return unflatten_spatial(x, height, width)

        winner_mask = (
            winner_mask_flat.reshape(n, height, width, k)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        winner_cluster_field = winner_cluster.reshape(n, 1, height, width)

        return {
            "soft_residual": field_1(soft_residual),
            "uniform_residual": field_1(uniform_residual),
            "winner_cluster": winner_cluster_field,
            "winner_mask": winner_mask,
            "winner_mass": field_1(winner_mass),
            "winner_prior": field_1(winner_prior),
            "winner_enrichment": field_1(winner_enrichment),
            "winner_support": field_1(winner_support),
            "memory_labels": torch.stack(memory_labels_all, dim=0),
            "memory_prior": torch.stack(memory_prior_all, dim=0),
        }
