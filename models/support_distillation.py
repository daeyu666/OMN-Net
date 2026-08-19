"""GT-support distillation ranker for OMN-Net innovation point 2.

This module is a diagnostic learner, not the final Stage-3 reconstructor. It
keeps Stage-1/Stage-2 frozen and reuses the exact K-candidate observable recall
and candidate features from SparseSimplexComplementNet, but it never applies
sparsemax and never generates a complement residual. It only predicts logits
for the real LR-HSI-derived candidates.

Training supervision is supplied externally by a GT Frank-Wolfe teacher. At
inference/diagnostic time GT is unavailable; the ranker outputs Top-M candidate
indices so that their retained convex-oracle capacity can be measured.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .nonlocal_complement import (
    flatten_spatial,
    flatten_tangent,
    gather_complement_candidates,
    project_complement_vectors,
)
from .sparse_simplex_complement import SparseSimplexComplementNet


class CandidateSupportRanker(SparseSimplexComplementNet):
    """Score all observable-recalled LR-HSI candidates without early sparsity."""

    def score_queries(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        query_indices: Optional[torch.Tensor] = None,
        return_candidate_details: bool = False,
        rank_top_m: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return candidate logits for training or compact ranked indices for eval.

        Args:
            query_indices: optional [N,Q] HR linear indices. Training should
                sample a modest Q so all K candidate logits receive gradients.
            return_candidate_details: when True, return [N,Q,K,R] candidate
                residuals and [N,Q,K] logits for Frank-Wolfe distillation.
            rank_top_m: when set, return only the highest-scoring M indices for
                every effective query; useful for full-field evaluation.
        """
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
                query_observable, memory_observable
            )

        query_context, raw_probe = self.query_encoder(
            self._query_features(stage2)
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
        observable_batches = []

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
            observable_chunks = []
            memory = memory_flat[b]
            memory_obs = memory_observable_std[b].to(local_null_state.dtype)

            for start in range(0, effective_query_count, self.query_chunk_pixels):
                stop = min(start + self.query_chunk_pixels, effective_query_count)
                idx = idx_all[start:stop]
                distances = distance_all[start:stop].to(
                    local_null_state.dtype
                )
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
                    dim=2, keepdim=True
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
                    dim=1, keepdim=True
                ).values.clamp_min(self.key_eps)
                normalized_distance = centered_distance / distance_scale

                probe_comp = project_complement_vectors(
                    probe,
                    tangent,
                    null_projector,
                )
                probe_unit = probe_comp / probe_comp.square().sum(
                    dim=1, keepdim=True
                ).sqrt().clamp_min(self.key_eps)
                candidate_unit = candidates / candidates.square().sum(
                    dim=2, keepdim=True
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
                scorer_input = torch.cat(
                    [context_expanded, candidate_features],
                    dim=2,
                )
                learned_delta = self.candidate_scorer(
                    scorer_input.reshape(-1, scorer_input.size(2))
                ).reshape(stop - start, top_k)
                logits = -normalized_distance + learned_delta

                if return_candidate_details:
                    logits_chunks.append(logits)
                    # Candidate values come from frozen Stage-2/LR-HSI state;
                    # detach makes the information boundary explicit.
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
                    ranked_chunks.append(torch.gather(idx, 1, order).detach())
                    # Observable baseline is the original retrieval ordering.
                    observable_chunks.append(idx[:, :rank_top_m].detach())

            if return_candidate_details:
                logits_batches.append(torch.cat(logits_chunks, dim=0))
                candidate_batches.append(torch.cat(candidate_chunks, dim=0))
                candidate_index_batches.append(torch.cat(index_chunks, dim=0))
            if rank_top_m is not None:
                ranked_batches.append(torch.cat(ranked_chunks, dim=0))
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
        }
        if return_candidate_details:
            result.update(
                {
                    "candidate_logits_flat": torch.stack(
                        logits_batches,
                        dim=0,
                    ),
                    "candidate_residuals_flat": torch.stack(
                        candidate_batches,
                        dim=0,
                    ),
                    "candidate_indices_flat": torch.stack(
                        candidate_index_batches,
                        dim=0,
                    ),
                }
            )
        if rank_top_m is not None:
            result.update(
                {
                    "ranked_candidate_indices_flat": torch.stack(
                        ranked_batches,
                        dim=0,
                    ),
                    "observable_candidate_indices_flat": torch.stack(
                        observable_batches,
                        dim=0,
                    ),
                }
            )
        return result
