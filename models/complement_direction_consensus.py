"""Deterministic direction-consensus arbitration for OMN-Net innovation point 2.

The wide observable-key pool is kept intact. Candidate LR-HSI states are first
converted to query-specific tangent-complement residuals. Their unit directions
are then compared through the exact first directional moment

    g = sum_i pi_i u_i,
    s_i = u_i^T g = sum_j pi_j cos(u_i, u_j).

This avoids an explicit KxK affinity matrix while measuring candidate-to-
candidate directional recurrence. No GT, uncertainty gate, or trainable
parameter is used.
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


class ComplementDirectionConsensus(nn.Module):
    """Select a compact candidate set by repeated P_comp correction direction."""

    def __init__(
        self,
        top_m: int = 32,
        prior_mode: str = "uniform",
        temperature_ratio: float = 1.0,
        query_chunk_pixels: int = 64,
        eps: float = 1e-8,
    ):
        super().__init__()
        if top_m < 1:
            raise ValueError("top_m must be positive")
        if prior_mode not in {"uniform", "observable"}:
            raise ValueError("prior_mode must be 'uniform' or 'observable'")
        if temperature_ratio <= 0 or query_chunk_pixels < 1 or eps <= 0:
            raise ValueError("invalid direction-consensus settings")
        self.top_m = int(top_m)
        self.prior_mode = str(prior_mode)
        self.temperature_ratio = float(temperature_ratio)
        self.query_chunk_pixels = int(query_chunk_pixels)
        self.eps = float(eps)

    def _selected_weights(
        self,
        selected_scores: torch.Tensor,
        selected_prior: torch.Tensor,
    ) -> torch.Tensor:
        maximum = selected_scores.max(dim=1, keepdim=True).values
        shifted = selected_scores - maximum
        scale = shifted.abs().median(dim=1, keepdim=True).values.clamp_min(self.eps)
        logits = shifted / (scale * self.temperature_ratio)
        directional = torch.softmax(logits, dim=1)
        if self.prior_mode == "observable":
            directional = directional * selected_prior.clamp_min(self.eps)
            directional = directional / directional.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return directional

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
            raise ValueError("top-k index/weight shapes differ")
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
        top_m = min(self.top_m, k)

        memory_flat = flatten_spatial(memory_null)
        local_flat = flatten_spatial(local_null_state)
        tangent_flat = flatten_tangent(tangent_basis)
        indices_flat = (
            topk_indices.permute(0, 2, 3, 1)
            .reshape(n, height * width, k)
            .contiguous()
        )
        observable_flat = (
            topk_observable_weights.permute(0, 2, 3, 1)
            .reshape(n, height * width, k)
            .contiguous()
        )

        selected_indices = torch.empty(
            n, height * width, top_m, dtype=torch.long, device=topk_indices.device
        )
        selected_scores = local_null_state.new_empty(n, height * width, top_m)
        selected_weights = local_null_state.new_empty(n, height * width, top_m)
        top1_residual = local_null_state.new_zeros(
            n, height * width, local_null_state.size(1)
        )
        soft_residual = local_null_state.new_zeros(
            n, height * width, local_null_state.size(1)
        )
        resultant_norm = local_null_state.new_zeros(n, height * width, 1)
        selected_alignment = local_null_state.new_zeros(n, height * width, 1)

        for b in range(n):
            memory = memory_flat[b]
            for start in range(0, height * width, self.query_chunk_pixels):
                stop = min(start + self.query_chunk_pixels, height * width)
                idx = indices_flat[b, start:stop]
                obs_prior = observable_flat[b, start:stop].to(local_null_state.dtype)
                local = local_flat[b, start:stop]
                tangent = tangent_flat[b, start:stop]

                candidates = gather_complement_candidates(
                    memory,
                    local,
                    tangent,
                    idx,
                    null_projector,
                )
                magnitude = candidates.square().sum(dim=2).sqrt()
                valid = magnitude > self.eps
                unit = candidates / magnitude.clamp_min(self.eps).unsqueeze(-1)
                unit = unit * valid.unsqueeze(-1).to(unit.dtype)

                if self.prior_mode == "observable":
                    prior = obs_prior * valid.to(obs_prior.dtype)
                    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(self.eps)
                else:
                    prior = valid.to(obs_prior.dtype)
                    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(self.eps)

                resultant = torch.sum(prior.unsqueeze(-1) * unit, dim=1)
                score = torch.einsum("qkr,qr->qk", unit, resultant)
                score = score.masked_fill(~valid, -float("inf"))
                score_value, order = torch.topk(
                    score, k=top_m, dim=1, largest=True, sorted=True
                )
                chosen_indices = torch.gather(idx, 1, order)
                chosen_prior = torch.gather(prior, 1, order)
                weights = self._selected_weights(score_value, chosen_prior)
                chosen_candidates = torch.gather(
                    candidates,
                    1,
                    order.unsqueeze(-1).expand(-1, -1, candidates.size(2)),
                )

                selected_indices[b, start:stop] = chosen_indices
                selected_scores[b, start:stop] = score_value
                selected_weights[b, start:stop] = weights
                top1_residual[b, start:stop] = chosen_candidates[:, 0]
                soft_residual[b, start:stop] = torch.sum(
                    weights.unsqueeze(-1) * chosen_candidates, dim=1
                )
                resultant_norm[b, start:stop, 0] = resultant.square().sum(dim=1).sqrt()
                selected_alignment[b, start:stop, 0] = score_value.mean(dim=1)

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
            "resultant_norm": unflatten_spatial(resultant_norm, height, width),
            "selected_alignment": unflatten_spatial(selected_alignment, height, width),
        }
