"""E14 candidate-side LR-HSI memory-context rankability probe.

This module changes exactly one variable relative to E13-v2: each recalled
LR-HSI candidate receives a learned spatial-context descriptor extracted from
its local coefficient neighborhood. Query features, observable K-recall,
P_comp candidate construction, pure learned logits, and GT Frank-Wolfe
supervision remain unchanged.

The memory context is selection-only. It never modifies a candidate residual and
is never written into the reconstructed coefficient field.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nonlocal_complement import (
    flatten_spatial,
    flatten_tangent,
    gather_complement_candidates,
    project_complement_vectors,
)
from .sparse_simplex_complement import CandidateScorer
from .support_distillation_v2 import PureLearnedSupportRanker


class MemoryContextEncoder(nn.Module):
    """Encode a 5x5 LR-HSI coefficient neighborhood for candidate selection."""

    def __init__(
        self,
        coefficient_rank: int,
        hidden_channels: int = 64,
        output_channels: int = 64,
    ):
        super().__init__()
        if hidden_channels < 1 or output_channels < 1:
            raise ValueError("memory context channels must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(coefficient_rank, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, 1),
        )

    def forward(self, normalized_lr_coefficients: torch.Tensor) -> torch.Tensor:
        return self.net(normalized_lr_coefficients)


class MemoryContextSupportRanker(PureLearnedSupportRanker):
    """E13-v2 ranker augmented only with candidate-side LR spatial context."""

    def __init__(
        self,
        local_model,
        top_k: int = 690,
        exclusion_radius_lr: int = 1,
        query_chunk_pixels: int = 64,
        context_channels: int = 64,
        context_blocks: int = 2,
        scorer_hidden: int = 96,
        scorer_blocks: int = 2,
        sparsemax_temperature: float = 1.0,
        key_eps: float = 1e-6,
        scorer_init_std: float = 1e-3,
        memory_hidden_channels: int = 64,
        memory_context_channels: int = 64,
    ):
        super().__init__(
            local_model=local_model,
            top_k=top_k,
            exclusion_radius_lr=exclusion_radius_lr,
            query_chunk_pixels=query_chunk_pixels,
            context_channels=context_channels,
            context_blocks=context_blocks,
            scorer_hidden=scorer_hidden,
            scorer_blocks=scorer_blocks,
            sparsemax_temperature=sparsemax_temperature,
            key_eps=key_eps,
            scorer_init_std=scorer_init_std,
        )
        if memory_hidden_channels < 1 or memory_context_channels < 1:
            raise ValueError("invalid memory context dimensions")

        self.memory_context_channels = int(memory_context_channels)
        rank = local_model.basis_rank
        msi_channels = local_model.msi_channels

        self.memory_context_encoder = MemoryContextEncoder(
            coefficient_rank=rank,
            hidden_channels=memory_hidden_channels,
            output_channels=memory_context_channels,
        )

        # Keep every E13-v2 feature unchanged and concatenate exactly one new
        # candidate-side context vector.
        candidate_feature_dim = msi_channels + 1 + rank + 1 + 1
        scorer_input_dim = (
            context_channels + memory_context_channels + candidate_feature_dim
        )
        self.candidate_scorer = CandidateScorer(
            input_dim=scorer_input_dim,
            hidden_dim=scorer_hidden,
            blocks=scorer_blocks,
        )
        nn.init.normal_(
            self.candidate_scorer.head.weight,
            mean=0.0,
            std=float(scorer_init_std),
        )
        nn.init.zeros_(self.candidate_scorer.head.bias)

    def score_queries(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        query_indices: Optional[torch.Tensor] = None,
        return_candidate_details: bool = False,
        rank_top_m: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            stage2 = self.local_model(lr_hsi, hr_msi)
            stage2["hr_msi"] = hr_msi.detach()
            geometry = self.local_model.geometry
            reduced_response = geometry.reduced_response.to(
                stage2["lr_coefficients"]
            )
            memory_observable = torch.einsum(
                "mr,nrhw->nmhw",
                reduced_response,
                stage2["lr_coefficients"],
            )
            query_observable = torch.einsum(
                "mr,nrhw->nmhw",
                reduced_response,
                stage2["anchor_coefficients"],
            )
            memory_null = geometry.project_null(stage2["lr_coefficients"])
            local_null_state = (
                stage2["null_seed_coefficients"] + stage2["tangent_residual"]
            )
            (
                topk_indices,
                topk_distances,
                query_observable_std,
                memory_observable_std,
            ) = self._retrieve_observable_topk(
                query_observable,
                memory_observable,
            )

        # Query path is identical to E13-v2.
        query_context, raw_probe = self.query_encoder(
            self._query_features(stage2)
        )

        # E14's only added source of trainable information: a 5x5 LR-HSI
        # coefficient neighborhood encoded at the candidate location.
        coefficient_scale_4d = stage2["coefficient_scale"].view(1, -1, 1, 1)
        normalized_lr_coefficients = (
            stage2["lr_coefficients"] / coefficient_scale_4d
        )
        memory_context = self.memory_context_encoder(
            normalized_lr_coefficients
        )

        n, _, height, width = local_null_state.shape
        full_query_count = height * width
        rank = local_null_state.size(1)
        top_k = topk_indices.size(2)

        if query_indices is not None:
            if query_indices.ndim != 2 or query_indices.size(0) != n:
                raise ValueError("query_indices must be [N,Q]")
            if query_indices.numel() == 0:
                raise ValueError("query_indices cannot be empty")
            if int(query_indices.min().item()) < 0:
                raise ValueError("query_indices contain negative values")
            if int(query_indices.max().item()) >= full_query_count:
                raise ValueError("query_indices are outside the HR field")
            effective_query_count = query_indices.size(1)
        else:
            effective_query_count = full_query_count

        if rank_top_m is not None:
            if rank_top_m < 1:
                raise ValueError("rank_top_m must be positive")
            rank_top_m = min(int(rank_top_m), top_k)

        memory_flat = flatten_spatial(memory_null)
        memory_context_flat = flatten_spatial(memory_context)
        local_flat = flatten_spatial(local_null_state)
        tangent_flat = flatten_tangent(stage2["tangent_basis"])
        context_flat = flatten_spatial(query_context)
        probe_flat = flatten_spatial(raw_probe)
        coefficient_scale = stage2["coefficient_scale"].view(1, 1, rank)
        null_projector = geometry.null_projector

        logits_batches = []
        candidate_batches = []
        candidate_index_batches = []
        ranked_batches = []
        ranked_position_batches = []
        observable_batches = []
        logit_std_values = []
        memory_context_norm_values = []

        for b in range(n):
            if query_indices is None:
                selected = torch.arange(
                    full_query_count,
                    device=hr_msi.device,
                    dtype=torch.long,
                )
            else:
                selected = query_indices[b]

            idx_all = topk_indices[b, selected]
            distance_all = topk_distances[b, selected]
            query_obs_all = query_observable_std[b, selected]
            local_all = local_flat[b, selected]
            tangent_all = tangent_flat[b, selected]
            context_all = context_flat[b, selected]
            probe_all = probe_flat[b, selected]

            logits_chunks = []
            candidate_chunks = []
            index_chunks = []
            ranked_chunks = []
            ranked_position_chunks = []
            observable_chunks = []
            memory = memory_flat[b]
            memory_obs = memory_observable_std[b].to(local_null_state.dtype)
            memory_ctx = memory_context_flat[b]

            for start in range(0, effective_query_count, self.query_chunk_pixels):
                stop = min(start + self.query_chunk_pixels, effective_query_count)
                idx = idx_all[start:stop]
                distances = distance_all[start:stop].to(local_null_state.dtype)
                local = local_all[start:stop]
                tangent = tangent_all[start:stop]
                context = context_all[start:stop]
                probe = probe_all[start:stop]

                candidates = gather_complement_candidates(
                    memory,
                    local,
                    tangent,
                    idx,
                    null_projector,
                )
                normalized_candidates = candidates / coefficient_scale
                candidate_magnitude = normalized_candidates.square().mean(
                    dim=2,
                    keepdim=True,
                ).sqrt()

                query_obs = query_obs_all[start:stop].to(
                    local_null_state.dtype
                )
                candidate_obs = memory_obs[idx]
                signed_observable_difference = (
                    query_obs.unsqueeze(1) - candidate_obs
                )

                centered_distance = distances - distances[:, :1]
                distance_scale = centered_distance.median(
                    dim=1,
                    keepdim=True,
                ).values.clamp_min(self.key_eps)
                normalized_distance = centered_distance / distance_scale

                probe_comp = project_complement_vectors(
                    probe,
                    tangent,
                    null_projector,
                )
                probe_unit = probe_comp / probe_comp.square().sum(
                    dim=1,
                    keepdim=True,
                ).sqrt().clamp_min(self.key_eps)
                candidate_unit = candidates / candidates.square().sum(
                    dim=2,
                    keepdim=True,
                ).sqrt().clamp_min(self.key_eps)
                probe_alignment = torch.einsum(
                    "qkr,qr->qk",
                    candidate_unit,
                    probe_unit,
                ).unsqueeze(-1)

                candidate_features = torch.cat(
                    [
                        signed_observable_difference,
                        normalized_distance.unsqueeze(-1),
                        normalized_candidates,
                        candidate_magnitude,
                        probe_alignment,
                    ],
                    dim=2,
                )
                context_expanded = context.unsqueeze(1).expand(
                    -1,
                    top_k,
                    -1,
                )
                candidate_memory_context = memory_ctx[idx]
                scorer_input = torch.cat(
                    [
                        context_expanded,
                        candidate_memory_context,
                        candidate_features,
                    ],
                    dim=2,
                )
                logits = self.candidate_scorer(
                    scorer_input.reshape(-1, scorer_input.size(2))
                ).reshape(stop - start, top_k)
                logit_std_values.append(logits.detach().std(dim=1).mean())
                memory_context_norm_values.append(
                    candidate_memory_context.detach().square().mean(dim=2).sqrt().mean()
                )

                if return_candidate_details:
                    logits_chunks.append(logits)
                    candidate_chunks.append(candidates.detach())
                    index_chunks.append(idx.detach())

                if rank_top_m is not None:
                    order = torch.topk(
                        logits,
                        k=rank_top_m,
                        dim=1,
                        largest=True,
                        sorted=True,
                    ).indices
                    ranked_position_chunks.append(order.detach())
                    ranked_chunks.append(
                        torch.gather(idx, 1, order).detach()
                    )
                    observable_chunks.append(idx[:, :rank_top_m].detach())

            if return_candidate_details:
                logits_batches.append(torch.cat(logits_chunks, dim=0))
                candidate_batches.append(torch.cat(candidate_chunks, dim=0))
                candidate_index_batches.append(torch.cat(index_chunks, dim=0))
            if rank_top_m is not None:
                ranked_batches.append(torch.cat(ranked_chunks, dim=0))
                ranked_position_batches.append(
                    torch.cat(ranked_position_chunks, dim=0)
                )
                observable_batches.append(torch.cat(observable_chunks, dim=0))

        result = {
            "basis": stage2["basis"],
            "coefficient_scale": stage2["coefficient_scale"],
            "lr_coefficients": stage2["lr_coefficients"],
            "anchor_coefficients": stage2["anchor_coefficients"],
            "null_seed_coefficients": stage2["null_seed_coefficients"],
            "tangent_basis": stage2["tangent_basis"],
            "stage2_coefficients": stage2["corrected_coefficients"],
            "stage2_hsi": stage2["reconstructed_hsi"],
            "sampled_query_indices": query_indices,
            "retrieval_top_k": torch.tensor(
                top_k,
                device=hr_msi.device,
                dtype=torch.long,
            ),
            "height": torch.tensor(height, device=hr_msi.device),
            "width": torch.tensor(width, device=hr_msi.device),
            "learned_logit_std": (
                torch.stack(logit_std_values).mean()
                if logit_std_values
                else hr_msi.new_tensor(0.0)
            ),
            "memory_context_norm": (
                torch.stack(memory_context_norm_values).mean()
                if memory_context_norm_values
                else hr_msi.new_tensor(0.0)
            ),
        }
        if return_candidate_details:
            result.update(
                {
                    "candidate_logits_flat": torch.stack(logits_batches, dim=0),
                    "candidate_residuals_flat": torch.stack(candidate_batches, dim=0),
                    "candidate_indices_flat": torch.stack(candidate_index_batches, dim=0),
                }
            )
        if rank_top_m is not None:
            result.update(
                {
                    "ranked_candidate_indices_flat": torch.stack(ranked_batches, dim=0),
                    "ranked_candidate_positions_flat": torch.stack(
                        ranked_position_batches,
                        dim=0,
                    ),
                    "observable_candidate_indices_flat": torch.stack(
                        observable_batches,
                        dim=0,
                    ),
                }
            )
        return result
