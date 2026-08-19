"""Query-conditioned sparse-simplex arbitration for OMN-Net innovation point 2.

The module keeps the validated Stage-1/Stage-2 path frozen. HR-MSI-derived
information is allowed to condition *selection*, but the actual tangent-
complement value must come from an observed LR-HSI spectral state.

For every HR query p:

    d_i(p) = P_comp(p) [C_null^LR(q_i) - C_null^Stage2(p)]
    w(p)   = sparsemax(logits(p))
    DeltaC = sum_i w_i(p) d_i(p)

with w_i >= 0 and sum_i w_i = 1. Therefore the learned correction remains in
the convex hull of real LR-HSI-derived P_comp candidate residuals. A learned
P_comp probe is selection-only: it can score candidates but is never added to
the reconstruction directly.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_null_manifold import LocalNullManifoldNet
from .nonlocal_complement import (
    flatten_spatial,
    flatten_tangent,
    gather_complement_candidates,
    project_complement_vectors,
    unflatten_spatial,
)


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection onto the probability simplex."""
    if logits.numel() == 0:
        return logits
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    k = z.size(dim)
    view = [1] * z.ndim
    view[dim] = k
    range_k = torch.arange(
        1, k + 1, device=z.device, dtype=z.dtype
    ).view(view)
    cumsum = z_sorted.cumsum(dim)
    support = 1.0 + range_k * z_sorted > cumsum
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau_sum = cumsum.gather(dim, support_size - 1)
    tau = (tau_sum - 1.0) / support_size.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


class ResidualContextBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class QueryContextEncoder(nn.Module):
    """Encode local HR observable context and emit a selection-only probe."""

    def __init__(
        self,
        input_channels: int,
        context_channels: int,
        coefficient_rank: int,
        blocks: int = 2,
    ):
        super().__init__()
        groups = min(8, context_channels)
        while context_channels % groups != 0 and groups > 1:
            groups -= 1
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, context_channels, 3, padding=1),
            nn.GroupNorm(groups, context_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualContextBlock(context_channels) for _ in range(blocks)]
        )
        self.context_head = nn.Conv2d(
            context_channels, context_channels, kernel_size=1
        )
        self.probe_head = nn.Conv2d(
            context_channels, coefficient_rank, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.blocks(self.stem(x))
        return self.context_head(hidden), self.probe_head(hidden)


class CandidateScorer(nn.Module):
    """Predict a query-conditioned correction to the observable recall score."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        blocks: int = 2,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(max(blocks - 1, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


class SparseSimplexComplementNet(nn.Module):
    """Frozen Stage2 + trainable query-conditioned LR-HSI candidate arbitration."""

    def __init__(
        self,
        local_model: LocalNullManifoldNet,
        top_k: int = 690,
        exclusion_radius_lr: int = 1,
        query_chunk_pixels: int = 64,
        context_channels: int = 64,
        context_blocks: int = 2,
        scorer_hidden: int = 96,
        scorer_blocks: int = 2,
        sparsemax_temperature: float = 1.0,
        key_eps: float = 1e-6,
    ):
        super().__init__()
        if top_k < 1 or query_chunk_pixels < 1:
            raise ValueError("top_k and query_chunk_pixels must be positive")
        if exclusion_radius_lr < 0:
            raise ValueError("exclusion_radius_lr must be non-negative")
        if sparsemax_temperature <= 0 or key_eps <= 0:
            raise ValueError("temperature and key_eps must be positive")

        self.local_model = local_model
        self.top_k = int(top_k)
        self.exclusion_radius_lr = int(exclusion_radius_lr)
        self.query_chunk_pixels = int(query_chunk_pixels)
        self.sparsemax_temperature = float(sparsemax_temperature)
        self.key_eps = float(key_eps)

        for parameter in self.local_model.parameters():
            parameter.requires_grad_(False)
        self.local_model.eval()

        rank = local_model.basis_rank
        msi_channels = local_model.msi_channels
        tangent_dimension = local_model.tangent_dimension
        query_input_channels = (
            3 * msi_channels + rank + rank + tangent_dimension
        )
        self.query_encoder = QueryContextEncoder(
            input_channels=query_input_channels,
            context_channels=context_channels,
            coefficient_rank=rank,
            blocks=context_blocks,
        )
        candidate_feature_dim = msi_channels + 1 + rank + 1 + 1
        self.candidate_scorer = CandidateScorer(
            input_dim=context_channels + candidate_feature_dim,
            hidden_dim=scorer_hidden,
            blocks=scorer_blocks,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.local_model.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _query_features(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        scale = out["coefficient_scale"].view(1, -1, 1, 1)
        normalized_null_seed = out["null_seed_coefficients"] / scale
        global_scale = out["coefficient_scale"].mean().clamp_min(1e-8)
        normalized_tangent_scale = out["tangent_scale"] / global_scale
        return torch.cat(
            [
                out["hr_msi"],
                out["base_msi"],
                out["msi_residual"],
                normalized_null_seed,
                out["tangent_projector_diagonal"],
                normalized_tangent_scale,
            ],
            dim=1,
        )

    @torch.no_grad()
    def _retrieve_observable_topk(
        self,
        query_observable: torch.Tensor,
        memory_observable: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """High-recall observable retrieval without constructing P_comp values."""
        n, channels, height, width = query_observable.shape
        nm, cm, memory_h, memory_w = memory_observable.shape
        if n != nm or channels != cm:
            raise ValueError("query/memory observable shapes are incompatible")

        query_flat = flatten_spatial(query_observable)
        memory_flat = flatten_spatial(memory_observable)
        q_count = height * width
        memory_count = memory_h * memory_w
        max_excluded = (2 * self.exclusion_radius_lr + 1) ** 2
        top_k = min(self.top_k, memory_count - max_excluded)
        if top_k < 1:
            raise ValueError("no non-local candidate remains")

        indices_batches = []
        distance_batches = []
        query_standardized_batches = []
        memory_standardized_batches = []

        memory_linear = torch.arange(memory_count, device=query_observable.device)
        memory_y = torch.div(memory_linear, memory_w, rounding_mode="floor")
        memory_x = memory_linear.remainder(memory_w)
        query_linear = torch.arange(q_count, device=query_observable.device)
        query_y = torch.div(query_linear, width, rounding_mode="floor")
        query_x = query_linear.remainder(width)
        query_lr_y = torch.floor(
            (query_y.to(torch.float32) + 0.5) * memory_h / height
        ).to(torch.long).clamp_(0, memory_h - 1)
        query_lr_x = torch.floor(
            (query_x.to(torch.float32) + 0.5) * memory_w / width
        ).to(torch.long).clamp_(0, memory_w - 1)

        for b in range(n):
            memory = memory_flat[b].float()
            query = query_flat[b].float()
            mean = memory.mean(dim=0, keepdim=True)
            std = memory.std(dim=0, unbiased=False, keepdim=True).clamp_min(
                self.key_eps
            )
            memory_std = (memory - mean) / std
            query_std = (query - mean) / std
            idx_chunks = []
            dist_chunks = []
            for start in range(0, q_count, self.query_chunk_pixels):
                stop = min(start + self.query_chunk_pixels, q_count)
                q = query_std[start:stop]
                distances = torch.cdist(q, memory_std, p=2).square()
                distances = distances / max(channels, 1)
                cy = query_lr_y[start:stop].unsqueeze(1)
                cx = query_lr_x[start:stop].unsqueeze(1)
                local_mask = (
                    (memory_y.unsqueeze(0) - cy).abs()
                    <= self.exclusion_radius_lr
                ) & (
                    (memory_x.unsqueeze(0) - cx).abs()
                    <= self.exclusion_radius_lr
                )
                distances = distances.masked_fill(local_mask, float("inf"))
                finite_count = torch.isfinite(distances).sum(dim=1)
                if int(finite_count.min().item()) < top_k:
                    raise RuntimeError("not enough candidates after local exclusion")
                top_dist, top_idx = torch.topk(
                    distances, k=top_k, dim=1, largest=False, sorted=True
                )
                idx_chunks.append(top_idx)
                dist_chunks.append(top_dist)
            indices_batches.append(torch.cat(idx_chunks, dim=0))
            distance_batches.append(torch.cat(dist_chunks, dim=0))
            query_standardized_batches.append(query_std)
            memory_standardized_batches.append(memory_std)

        return (
            torch.stack(indices_batches, dim=0),
            torch.stack(distance_batches, dim=0),
            torch.stack(query_standardized_batches, dim=0),
            torch.stack(memory_standardized_batches, dim=0),
        )

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        query_indices: Optional[torch.Tensor] = None,
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
                raise ValueError("query_indices must be [N,Q_sample]")
            if query_indices.numel() == 0:
                raise ValueError("query_indices cannot be empty")
            if int(query_indices.min().item()) < 0 or int(query_indices.max().item()) >= full_query_count:
                raise ValueError("query_indices are outside the HR field")
            effective_query_count = query_indices.size(1)
        else:
            effective_query_count = full_query_count

        memory_flat = flatten_spatial(memory_null)
        local_flat = flatten_spatial(local_null_state)
        tangent_flat = flatten_tangent(stage2["tangent_basis"])
        context_flat = flatten_spatial(query_context)
        probe_flat = flatten_spatial(raw_probe)
        coefficient_scale = stage2["coefficient_scale"].view(1, 1, rank)
        null_projector = geometry.null_projector

        residual_batches = []
        active_batches = []
        max_weight_batches = []
        probe_norm_batches = []

        for b in range(n):
            if query_indices is None:
                selected = torch.arange(
                    full_query_count, device=hr_msi.device, dtype=torch.long
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

            residual_chunks = []
            active_chunks = []
            max_weight_chunks = []
            probe_norm_chunks = []
            memory = memory_flat[b]
            memory_obs = memory_observable_std[b].to(local_null_state.dtype)

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
                    probe, tangent, null_projector
                )
                probe_unit = probe_comp / probe_comp.square().sum(
                    dim=1, keepdim=True
                ).sqrt().clamp_min(self.key_eps)
                candidate_unit = candidates / candidates.square().sum(
                    dim=2, keepdim=True
                ).sqrt().clamp_min(self.key_eps)
                probe_alignment = torch.einsum(
                    "qkr,qr->qk", candidate_unit, probe_unit
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
                    -1, top_k, -1
                )
                scorer_input = torch.cat(
                    [context_expanded, candidate_features], dim=2
                )
                learned_delta = self.candidate_scorer(
                    scorer_input.reshape(-1, scorer_input.size(2))
                ).reshape(stop - start, top_k)

                logits = -normalized_distance + learned_delta
                weights = sparsemax(
                    logits / self.sparsemax_temperature, dim=1
                )
                residual_chunks.append(
                    torch.sum(weights.unsqueeze(-1) * candidates, dim=1)
                )
                active_chunks.append(
                    (weights > 1e-8).to(weights.dtype).sum(
                        dim=1, keepdim=True
                    )
                )
                max_weight_chunks.append(
                    weights.max(dim=1, keepdim=True).values
                )
                probe_norm_chunks.append(
                    probe_comp.square().sum(dim=1, keepdim=True).sqrt()
                )

            residual_batches.append(torch.cat(residual_chunks, dim=0))
            active_batches.append(torch.cat(active_chunks, dim=0))
            max_weight_batches.append(torch.cat(max_weight_chunks, dim=0))
            probe_norm_batches.append(torch.cat(probe_norm_chunks, dim=0))

        complement_flat = torch.stack(residual_batches, dim=0)
        active = torch.stack(active_batches, dim=0)
        max_weight = torch.stack(max_weight_batches, dim=0)
        probe_norm = torch.stack(probe_norm_batches, dim=0)

        result = {
            "basis": stage2["basis"],
            "coefficient_scale": stage2["coefficient_scale"],
            "lr_coefficients": stage2["lr_coefficients"],
            "anchor_coefficients": stage2["anchor_coefficients"],
            "null_seed_coefficients": stage2["null_seed_coefficients"],
            "tangent_basis": stage2["tangent_basis"],
            "tangent_scale": stage2["tangent_scale"],
            "stage2_coefficients": stage2["corrected_coefficients"],
            "stage2_hsi": stage2["reconstructed_hsi"],
            "complement_residual_flat": complement_flat,
            "sampled_query_indices": query_indices,
            "active_candidates_mean": active.detach().mean(),
            "max_weight_mean": max_weight.detach().mean(),
            "probe_norm_mean": probe_norm.detach().mean(),
            "retrieval_top_k": torch.tensor(
                top_k, device=hr_msi.device, dtype=torch.long
            ),
        }

        if query_indices is None:
            complement_residual = unflatten_spatial(
                complement_flat, height, width
            )
            corrected_coefficients = (
                stage2["corrected_coefficients"] + complement_residual
            )
            reconstructed_hsi = self.local_model.foundation.decode(
                corrected_coefficients, basis=stage2["basis"]
            )
            result.update(
                {
                    "complement_residual": complement_residual,
                    "corrected_coefficients": corrected_coefficients,
                    "reconstructed_hsi": reconstructed_hsi,
                }
            )
        return result
