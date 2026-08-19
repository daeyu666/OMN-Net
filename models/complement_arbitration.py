"""Deterministic candidate arbitration for OMN-Net innovation point 2.

The wide observable-key recall pool is intentionally high-recall. This module
adds one new, OMN-specific criterion before any trainable Stage-3 network is
introduced: a remote LR-HSI state is useful only if it agrees with the query in
already trusted subspaces. Observable agreement is inherited from the first
retrieval stage; tangent agreement is measured here. Only the remaining
P_comp difference is allowed to update the reconstruction.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .nonlocal_complement import (
    flatten_spatial,
    flatten_tangent,
    gather_complement_candidates,
    unflatten_spatial,
)


class KnownSubspaceConsistencyArbitrator(nn.Module):
    """Rerank a wide recall pool by observable + local-tangent consistency.

    No GT, uncertainty score, trust radius, or trainable parameter is used.
    The module therefore belongs entirely to innovation point 2.
    """

    def __init__(
        self,
        top_m: int = 16,
        tangent_weight: float = 1.0,
        temperature_ratio: float = 1.0,
        query_chunk_pixels: int = 128,
        eps: float = 1e-8,
    ):
        super().__init__()
        if top_m < 1:
            raise ValueError("top_m must be positive")
        if tangent_weight < 0:
            raise ValueError("tangent_weight must be non-negative")
        if temperature_ratio <= 0 or query_chunk_pixels < 1 or eps <= 0:
            raise ValueError("invalid arbitration settings")
        self.top_m = int(top_m)
        self.tangent_weight = float(tangent_weight)
        self.temperature_ratio = float(temperature_ratio)
        self.query_chunk_pixels = int(query_chunk_pixels)
        self.eps = float(eps)

    def _normalize_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Per-query robust normalization without any GT statistics."""
        minimum = x.min(dim=1, keepdim=True).values
        centered = x - minimum
        scale = centered.median(dim=1, keepdim=True).values.clamp_min(self.eps)
        return centered / scale

    def _weights(self, score: torch.Tensor) -> torch.Tensor:
        centered = score - score[:, :1]
        scale = centered.median(dim=1, keepdim=True).values.clamp_min(self.eps)
        scale = scale * self.temperature_ratio
        return torch.softmax(-centered / scale, dim=1)

    def forward(
        self,
        topk_indices: torch.Tensor,
        topk_observable_distances: torch.Tensor,
        memory_null: torch.Tensor,
        local_null_state: torch.Tensor,
        tangent_basis: torch.Tensor,
        null_projector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if topk_indices.ndim != 4 or topk_observable_distances.ndim != 4:
            raise ValueError("top-k fields must be [N,K,H,W]")
        if memory_null.ndim != 4 or local_null_state.ndim != 4:
            raise ValueError("null states must be [N,R,H,W]")
        if tangent_basis.ndim != 5:
            raise ValueError("tangent_basis must be [N,R,D,H,W]")
        if topk_indices.shape != topk_observable_distances.shape:
            raise ValueError("top-k index and distance shapes differ")

        n, k, height, width = topk_indices.shape
        if local_null_state.shape[-2:] != (height, width):
            raise ValueError("query spatial sizes differ")
        if tangent_basis.shape[0] != n or tangent_basis.shape[-2:] != (height, width):
            raise ValueError("tangent/query spatial sizes differ")
        if memory_null.size(0) != n:
            raise ValueError("batch sizes differ")
        top_m = min(self.top_m, k)

        memory_flat = flatten_spatial(memory_null)
        local_flat = flatten_spatial(local_null_state)
        tangent_flat = flatten_tangent(tangent_basis)
        indices_flat = (
            topk_indices.permute(0, 2, 3, 1)
            .reshape(n, height * width, k)
            .contiguous()
        )
        obs_flat = (
            topk_observable_distances.permute(0, 2, 3, 1)
            .reshape(n, height * width, k)
            .contiguous()
        )

        selected_indices = torch.empty(
            n, height * width, top_m, dtype=torch.long, device=topk_indices.device
        )
        selected_scores = local_null_state.new_empty(n, height * width, top_m)
        selected_weights = local_null_state.new_empty(n, height * width, top_m)
        selected_tangent = local_null_state.new_empty(n, height * width, top_m)
        top1_residual = local_null_state.new_zeros(
            n, height * width, local_null_state.size(1)
        )
        soft_residual = local_null_state.new_zeros(
            n, height * width, local_null_state.size(1)
        )

        for batch_index in range(n):
            memory = memory_flat[batch_index]
            for start in range(0, height * width, self.query_chunk_pixels):
                stop = min(start + self.query_chunk_pixels, height * width)
                idx = indices_flat[batch_index, start:stop]
                obs = obs_flat[batch_index, start:stop].to(local_null_state.dtype)
                local = local_flat[batch_index, start:stop]
                tangent = tangent_flat[batch_index, start:stop]

                candidate_states = memory[idx]
                differences = candidate_states - local.unsqueeze(1)
                tangent_coordinates = torch.einsum(
                    "qrd,qkr->qkd", tangent.to(differences), differences
                )
                tangent_mismatch = tangent_coordinates.square().sum(dim=2)

                obs_normalized = self._normalize_distance(obs)
                tangent_normalized = self._normalize_distance(tangent_mismatch)
                score = obs_normalized + self.tangent_weight * tangent_normalized

                score_value, order = torch.topk(
                    score, k=top_m, dim=1, largest=False, sorted=True
                )
                chosen_indices = torch.gather(idx, 1, order)
                chosen_tangent = torch.gather(tangent_mismatch, 1, order)
                weights = self._weights(score_value)
                candidates = gather_complement_candidates(
                    memory,
                    local,
                    tangent,
                    chosen_indices,
                    null_projector,
                )

                selected_indices[batch_index, start:stop] = chosen_indices
                selected_scores[batch_index, start:stop] = score_value
                selected_weights[batch_index, start:stop] = weights
                selected_tangent[batch_index, start:stop] = chosen_tangent
                top1_residual[batch_index, start:stop] = candidates[:, 0]
                soft_residual[batch_index, start:stop] = torch.sum(
                    weights.unsqueeze(-1) * candidates, dim=1
                )

        def field_k(x: torch.Tensor) -> torch.Tensor:
            return (
                x.reshape(n, height, width, top_m)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        return {
            "top1_residual": unflatten_spatial(top1_residual, height, width),
            "soft_residual": unflatten_spatial(soft_residual, height, width),
            "selected_indices": field_k(selected_indices),
            "selected_scores": field_k(selected_scores),
            "selected_weights": field_k(selected_weights),
            "selected_tangent_mismatch": field_k(selected_tangent),
        }
