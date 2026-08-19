"""Observable-keyed non-local spectral memory for tangent-complement recovery.

Innovation point 2 in OMN-Net is deliberately asymmetric:

* retrieval keys are built from MSI-observable coefficient information;
* memory values come only from LR-HSI null-space spectral states;
* retrieved state differences are projected into the local tangent complement
  before they are allowed to alter the reconstruction.

The module is deterministic and contains no trainable parameters. It is used
first as a no-training diagnostic; a trainable Stage-3 predictor should only be
added if this restricted memory has enough complement-space support.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def flatten_spatial(x: torch.Tensor) -> torch.Tensor:
    """[N,C,H,W] -> [N,H*W,C]."""
    if x.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W], got {tuple(x.shape)}")
    return x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))


def unflatten_spatial(
    x: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """[N,H*W,C] -> [N,C,H,W]."""
    if x.ndim != 3 or x.size(1) != height * width:
        raise ValueError(
            f"Expected [N,{height * width},C], got {tuple(x.shape)}"
        )
    return (
        x.reshape(x.size(0), height, width, x.size(2))
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def flatten_tangent(tangent_basis: torch.Tensor) -> torch.Tensor:
    """[N,R,D,H,W] -> [N,H*W,R,D]."""
    if tangent_basis.ndim != 5:
        raise ValueError(
            f"Expected tangent [N,R,D,H,W], got {tuple(tangent_basis.shape)}"
        )
    return (
        tangent_basis.permute(0, 3, 4, 1, 2)
        .reshape(
            tangent_basis.size(0),
            tangent_basis.size(3) * tangent_basis.size(4),
            tangent_basis.size(1),
            tangent_basis.size(2),
        )
        .contiguous()
    )


def project_complement_vectors(
    vectors: torch.Tensor,
    tangent_basis: torch.Tensor,
    null_projector: torch.Tensor,
) -> torch.Tensor:
    """Project query-wise candidate vectors into N(S) minus local tangent.

    Args:
        vectors: [Q,K,R] or [Q,R].
        tangent_basis: [Q,R,D].
        null_projector: [R,R].

    The robust numerical form is P_null (I - P_tan) P_null instead of relying
    on the exact identity P_comp = P_null - P_tan.
    """
    squeeze = False
    if vectors.ndim == 2:
        vectors = vectors.unsqueeze(1)
        squeeze = True
    if vectors.ndim != 3 or tangent_basis.ndim != 3:
        raise ValueError("vectors must be [Q,K,R] and tangent [Q,R,D]")
    if vectors.size(0) != tangent_basis.size(0):
        raise ValueError("query counts differ between vectors and tangent basis")
    if vectors.size(2) != tangent_basis.size(1):
        raise ValueError("coefficient ranks differ")
    if null_projector.shape != (vectors.size(2), vectors.size(2)):
        raise ValueError("invalid null projector shape")

    projector = null_projector.to(vectors)
    null_vectors = torch.einsum("rs,qks->qkr", projector, vectors)
    coordinates = torch.einsum(
        "qrd,qkr->qkd", tangent_basis.to(null_vectors), null_vectors
    )
    tangent_part = torch.einsum(
        "qrd,qkd->qkr", tangent_basis.to(null_vectors), coordinates
    )
    complement = torch.einsum(
        "rs,qks->qkr", projector, null_vectors - tangent_part
    )
    return complement[:, 0] if squeeze else complement


def gather_complement_candidates(
    memory_null: torch.Tensor,
    local_null_state: torch.Tensor,
    tangent_basis: torch.Tensor,
    candidate_indices: torch.Tensor,
    null_projector: torch.Tensor,
) -> torch.Tensor:
    """Gather LR-HSI memory states and convert them to P_comp residuals.

    Args:
        memory_null: [M,R].
        local_null_state: [Q,R].
        tangent_basis: [Q,R,D].
        candidate_indices: [Q,K].
        null_projector: [R,R].
    Returns:
        candidate complement residuals [Q,K,R].
    """
    if memory_null.ndim != 2 or local_null_state.ndim != 2:
        raise ValueError("memory_null and local_null_state must be matrices")
    if candidate_indices.ndim != 2:
        raise ValueError("candidate_indices must be [Q,K]")
    if local_null_state.size(0) != candidate_indices.size(0):
        raise ValueError("query counts differ")

    candidates = memory_null[candidate_indices]
    differences = candidates - local_null_state.unsqueeze(1)
    return project_complement_vectors(
        differences, tangent_basis, null_projector
    )


class ObservableKeyedComplementMemory(nn.Module):
    """Retrieve LR-HSI null states by observable keys and output P_comp residuals.

    HR-MSI-derived observable information is allowed to decide *where to look*.
    It is never used as a complement-space value. All candidate values come
    from LR-HSI coefficients and every candidate is projected by the query's
    local tangent-complement operator before aggregation.
    """

    def __init__(
        self,
        top_k: int = 32,
        exclusion_radius_lr: int = 1,
        query_chunk_pixels: int = 256,
        temperature_ratio: float = 1.0,
        key_eps: float = 1e-6,
    ):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if exclusion_radius_lr < 0:
            raise ValueError("exclusion_radius_lr must be non-negative")
        if query_chunk_pixels < 1:
            raise ValueError("query_chunk_pixels must be positive")
        if temperature_ratio <= 0 or key_eps <= 0:
            raise ValueError("temperature_ratio and key_eps must be positive")
        self.top_k = int(top_k)
        self.exclusion_radius_lr = int(exclusion_radius_lr)
        self.query_chunk_pixels = int(query_chunk_pixels)
        self.temperature_ratio = float(temperature_ratio)
        self.key_eps = float(key_eps)

    def _standardize_keys(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standardize with LR-memory statistics only (no GT statistics)."""
        mean = memory.mean(dim=0, keepdim=True)
        scale = memory.std(dim=0, unbiased=False, keepdim=True)
        scale = scale.clamp_min(self.key_eps)
        return (query - mean) / scale, (memory - mean) / scale

    def _weights(self, distances: torch.Tensor) -> torch.Tensor:
        """Adaptive softmax weights with a per-query distance scale."""
        if distances.ndim != 2:
            raise ValueError("distances must be [Q,K]")
        centered = distances - distances[:, :1]
        scale = distances.median(dim=1, keepdim=True).values
        scale = scale.clamp_min(self.key_eps) * self.temperature_ratio
        return torch.softmax(-centered / scale, dim=1)

    def forward(
        self,
        query_observable: torch.Tensor,
        memory_observable: torch.Tensor,
        memory_null: torch.Tensor,
        local_null_state: torch.Tensor,
        tangent_basis: torch.Tensor,
        null_projector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if query_observable.ndim != 4 or memory_observable.ndim != 4:
            raise ValueError("observable keys must be [N,C,H,W]")
        if memory_null.ndim != 4 or local_null_state.ndim != 4:
            raise ValueError("null states must be [N,R,H,W]")
        if tangent_basis.ndim != 5:
            raise ValueError("tangent_basis must be [N,R,D,H,W]")

        n, _, height, width = query_observable.shape
        nm, _, memory_h, memory_w = memory_observable.shape
        if n != nm or memory_null.size(0) != n or local_null_state.size(0) != n:
            raise ValueError("batch sizes differ")
        if local_null_state.shape[-2:] != (height, width):
            raise ValueError("local null state and query spatial sizes differ")
        if tangent_basis.shape[0] != n or tangent_basis.shape[-2:] != (height, width):
            raise ValueError("tangent field and query spatial sizes differ")
        if memory_null.shape[-2:] != (memory_h, memory_w):
            raise ValueError("memory observable/null spatial sizes differ")
        if memory_null.size(1) != local_null_state.size(1):
            raise ValueError("memory/local coefficient ranks differ")
        if tangent_basis.size(1) != local_null_state.size(1):
            raise ValueError("tangent/local coefficient ranks differ")

        q_flat = flatten_spatial(query_observable)
        k_flat = flatten_spatial(memory_observable)
        v_flat = flatten_spatial(memory_null)
        ref_flat = flatten_spatial(local_null_state)
        tangent_flat = flatten_tangent(tangent_basis)

        q_count = height * width
        memory_count = memory_h * memory_w
        max_excluded = (2 * self.exclusion_radius_lr + 1) ** 2
        if memory_count <= max_excluded:
            raise ValueError("memory is too small for the requested exclusion radius")
        top_k = min(self.top_k, memory_count - max_excluded)
        if top_k < 1:
            raise ValueError("no non-local candidates remain")

        all_indices = torch.empty(
            n, q_count, top_k, dtype=torch.long, device=query_observable.device
        )
        all_distances = query_observable.new_empty(n, q_count, top_k)
        all_weights = query_observable.new_empty(n, q_count, top_k)
        all_residual = local_null_state.new_zeros(
            n, q_count, local_null_state.size(1)
        )
        all_support = query_observable.new_zeros(n, q_count, 1)
        all_consensus = query_observable.new_zeros(n, q_count, 1)

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

        for batch_index in range(n):
            query_keys, memory_keys = self._standardize_keys(
                q_flat[batch_index], k_flat[batch_index]
            )
            memory_values = v_flat[batch_index]
            local_values = ref_flat[batch_index]
            tangents = tangent_flat[batch_index]

            for start in range(0, q_count, self.query_chunk_pixels):
                stop = min(start + self.query_chunk_pixels, q_count)
                qk = query_keys[start:stop].float()
                mk = memory_keys.float()
                distances = torch.cdist(qk, mk, p=2).square()
                distances = distances / max(qk.size(1), 1)

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
                    raise RuntimeError("not enough non-local candidates for top-k")

                top_dist, top_idx = torch.topk(
                    distances, k=top_k, dim=1, largest=False, sorted=True
                )
                weights = self._weights(top_dist.to(query_observable.dtype))
                candidates = gather_complement_candidates(
                    memory_values,
                    local_values[start:stop],
                    tangents[start:stop],
                    top_idx,
                    null_projector,
                )
                residual = torch.sum(weights.unsqueeze(-1) * candidates, dim=1)
                disagreement = candidates - residual.unsqueeze(1)
                consensus = torch.sum(
                    weights.unsqueeze(-1) * disagreement.square(), dim=(1, 2)
                ) / max(candidates.size(2), 1)
                support = 1.0 / weights.square().sum(dim=1).clamp_min(self.key_eps)

                all_indices[batch_index, start:stop] = top_idx
                all_distances[batch_index, start:stop] = top_dist.to(
                    all_distances.dtype
                )
                all_weights[batch_index, start:stop] = weights
                all_residual[batch_index, start:stop] = residual
                all_support[batch_index, start:stop, 0] = support
                all_consensus[batch_index, start:stop, 0] = consensus

        residual_field = unflatten_spatial(all_residual, height, width)
        indices_field = (
            all_indices.reshape(n, height, width, top_k)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        distances_field = (
            all_distances.reshape(n, height, width, top_k)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        weights_field = (
            all_weights.reshape(n, height, width, top_k)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        support_field = unflatten_spatial(all_support, height, width)
        consensus_field = unflatten_spatial(all_consensus, height, width)

        return {
            "complement_residual": residual_field,
            "topk_indices": indices_field,
            "topk_distances": distances_field,
            "topk_weights": weights_field,
            "effective_support": support_field,
            "consensus_variance": consensus_field,
        }
