"""E17-b: signed LR-HSI curvature-aware complement extrapolation for OMN-Net.

E17 established that the LR-HSI-derived curvature subspace is useful but only
weakly identifiable when the predictor sees P_curv diagonal and singular-value
summaries.  E17-b changes one variable only: the eight signed, P_comp-projected
LR-HSI second-difference vectors are additionally exposed to the proposal
predictor.

The information boundary is unchanged:
* curvature evidence comes only from LR-HSI;
* HR-MSI is used only as observable/context input;
* the predictor emits a global coefficient proposal;
* the actual correction is still strictly P_curv r, never a free P_comp residual.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .local_curvature_extrapolation import (
    build_curvature_basis,
    build_lr_curvature_bank,
    map_lr_bank_to_hr,
    project_to_curvature,
)
from .local_null_manifold import GlobalProposalPredictor, LocalNullManifoldNet
from .nonlocal_complement import flatten_tangent, project_complement_vectors


CURVATURE_VECTOR_COUNT = 8


def build_signed_projected_curvature_bank(
    local_model: LocalNullManifoldNet,
    stage2: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return fixed-order signed curvature evidence [N,8,R,H,W].

    The eight vectors correspond to four orientations at radius 1 followed by
    the same four orientations at radius 2, exactly matching
    ``build_lr_curvature_bank``.  Every vector is projected into the query-wise
    tangent complement and null space before it is exposed to the predictor.
    """
    geometry = local_model.geometry
    memory_null = geometry.project_null(stage2["lr_coefficients"])
    bank_lr = build_lr_curvature_bank(memory_null)
    _, _, h, w = stage2["corrected_coefficients"].shape
    mapped = map_lr_bank_to_hr(bank_lr, h, w)  # [N,Q,8,R]
    tangent = flatten_tangent(stage2["tangent_basis"])

    projected_batches = []
    for b in range(mapped.size(0)):
        projected_batches.append(
            project_complement_vectors(
                mapped[b],
                tangent[b],
                geometry.null_projector,
            )
        )
    projected = torch.stack(projected_batches, dim=0)  # [N,Q,8,R]
    n, _, vectors, rank = projected.shape
    if vectors != CURVATURE_VECTOR_COUNT:
        raise RuntimeError(
            f"expected {CURVATURE_VECTOR_COUNT} curvature vectors, got {vectors}"
        )
    return (
        projected.reshape(n, h, w, vectors, rank)
        .permute(0, 3, 4, 1, 2)
        .contiguous()
        .detach()
    )


class LocalCurvatureExtrapolationE17BNet(nn.Module):
    """Frozen Stage-2 plus signed-curvature-aware P_curv proposal predictor."""

    def __init__(
        self,
        local_model: LocalNullManifoldNet,
        curvature_rank: int = 6,
        curvature_svd_chunk_pixels: int = 2048,
        curvature_svd_tolerance: float = 1e-5,
        curvature_abs_tolerance: float = 1e-9,
        proposal_amplitude_multiplier: float = 8.0,
        predictor_hidden_channels: int = 96,
        predictor_blocks: int = 4,
    ):
        super().__init__()
        self.local_model = local_model
        self.curvature_rank = int(curvature_rank)
        self.curvature_svd_chunk_pixels = int(curvature_svd_chunk_pixels)
        self.curvature_svd_tolerance = float(curvature_svd_tolerance)
        self.curvature_abs_tolerance = float(curvature_abs_tolerance)
        self.proposal_amplitude_multiplier = float(proposal_amplitude_multiplier)

        for parameter in self.local_model.parameters():
            parameter.requires_grad_(False)
        self.local_model.eval()

        rank = local_model.basis_rank
        msi = local_model.msi_channels
        input_channels = (
            msi                            # hr_msi
            + msi                          # base_msi
            + msi                          # msi_residual
            + rank                         # normalized null seed
            + rank                         # normalized tangent residual
            + rank                         # curvature projector diagonal
            + self.curvature_rank          # normalized singular values
            + CURVATURE_VECTOR_COUNT * rank  # signed projected curvature bank
        )
        self.proposal_predictor = GlobalProposalPredictor(
            input_channels=input_channels,
            coefficient_channels=rank,
            hidden_channels=predictor_hidden_channels,
            blocks=predictor_blocks,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.local_model.eval()
        return self

    def trainable_parameters(self):
        return self.proposal_predictor.parameters()

    def forward(
        self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            stage2 = self.local_model(lr_hsi, hr_msi)
            curvature_basis, curvature_singular, curvature_valid = (
                build_curvature_basis(
                    self.local_model,
                    stage2,
                    curvature_rank=self.curvature_rank,
                    chunk_pixels=self.curvature_svd_chunk_pixels,
                    relative_tolerance=self.curvature_svd_tolerance,
                    absolute_tolerance=self.curvature_abs_tolerance,
                )
            )
            signed_bank = build_signed_projected_curvature_bank(
                self.local_model, stage2
            )

        scale = stage2["coefficient_scale"].view(1, -1, 1, 1)
        normalized_null_seed = stage2["null_seed_coefficients"] / scale
        normalized_tangent_residual = stage2["tangent_residual"] / scale
        curvature_projector_diagonal = curvature_basis.square().sum(dim=2)
        singular_scale = curvature_singular[:, :1].clamp_min(1e-8)
        normalized_singular = curvature_singular / singular_scale

        # Preserve signed direction, spatial orientation and radius identity.
        # [N,8,R,H,W] -> [N,8*R,H,W], each direction's R coefficients contiguous.
        normalized_signed_bank = signed_bank / scale.unsqueeze(1)
        n, vectors, rank, h, w = normalized_signed_bank.shape
        signed_features = normalized_signed_bank.reshape(
            n, vectors * rank, h, w
        )

        features = torch.cat(
            [
                hr_msi,
                stage2["base_msi"],
                stage2["msi_residual"],
                normalized_null_seed,
                normalized_tangent_residual,
                curvature_projector_diagonal,
                normalized_singular,
                signed_features,
            ],
            dim=1,
        )
        raw = self.proposal_predictor(features)
        normalized = torch.tanh(raw)
        limit = self.proposal_amplitude_multiplier * scale
        proposal = normalized * limit
        curvature_residual = project_to_curvature(curvature_basis, proposal)
        corrected = stage2["corrected_coefficients"] + curvature_residual
        reconstructed = self.local_model.foundation.decode(
            corrected, basis=stage2["basis"]
        )

        return {
            **stage2,
            "curvature_basis": curvature_basis,
            "curvature_singular_values": curvature_singular,
            "curvature_valid_mask": curvature_valid,
            "curvature_projector_diagonal": curvature_projector_diagonal,
            "signed_projected_curvature_bank": signed_bank,
            "normalized_signed_curvature_features": signed_features,
            "raw_curvature_proposal": raw,
            "normalized_curvature_proposal": normalized,
            "curvature_global_proposal": proposal,
            "curvature_residual": curvature_residual,
            "curvature_corrected_coefficients": corrected,
            "curvature_reconstructed_hsi": reconstructed,
        }
