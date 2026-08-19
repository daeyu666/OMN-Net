"""LR-HSI curvature-constrained complement extrapolation for OMN-Net.

Innovation point 2 candidate after E16:
* curvature directions are built only from the observed LR-HSI null-coefficient field;
* the LR curvature bank is mapped to each HR query and projected into P_comp;
* a global coefficient proposal is predicted from legal Stage-2/HR-MSI context;
* only the projection of that proposal onto the LR-HSI curvature subspace may
  modify the reconstruction.

The predictor never generates a free P_comp residual.  It only predicts a
proposal that is subsequently authorized by P_curv = U_curv U_curv^T.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_null_manifold import GlobalProposalPredictor, LocalNullManifoldNet
from .nonlocal_complement import (
    flatten_spatial,
    flatten_tangent,
    project_complement_vectors,
    unflatten_spatial,
)


def _shift_reflect(x: torch.Tensor, dy: int, dx: int, pad: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y0 = pad + int(dy)
    x0 = pad + int(dx)
    return padded[:, :, y0:y0 + h, x0:x0 + w]


def build_lr_curvature_bank(memory_null: torch.Tensor) -> torch.Tensor:
    """Observed LR-HSI second differences [N,R,8,Hlr,Wlr]."""
    offsets = [
        (0, 1), (1, 0), (1, 1), (1, -1),
        (0, 2), (2, 0), (2, 2), (2, -2),
    ]
    vectors = []
    for dy, dx in offsets:
        positive = _shift_reflect(memory_null, dy, dx, 2)
        negative = _shift_reflect(memory_null, -dy, -dx, 2)
        denom = float(dy * dy + dx * dx)
        vectors.append((positive + negative - 2.0 * memory_null) / denom)
    return torch.stack(vectors, dim=2)


def map_lr_bank_to_hr(bank: torch.Tensor, hr_h: int, hr_w: int) -> torch.Tensor:
    """Nearest LR-cell assignment: [N,R,V,Hlr,Wlr] -> [N,Q,V,R]."""
    n, rank, vectors, lr_h, lr_w = bank.shape
    q = hr_h * hr_w
    linear = torch.arange(q, device=bank.device)
    y = torch.div(linear, hr_w, rounding_mode="floor")
    x = linear.remainder(hr_w)
    lr_y = torch.floor((y.float() + 0.5) * lr_h / hr_h).long().clamp_(0, lr_h - 1)
    lr_x = torch.floor((x.float() + 0.5) * lr_w / hr_w).long().clamp_(0, lr_w - 1)
    index = lr_y * lr_w + lr_x
    flat = bank.permute(0, 3, 4, 2, 1).reshape(
        n, lr_h * lr_w, vectors, rank
    )
    return flat[:, index]


def build_curvature_basis(
    local_model: LocalNullManifoldNet,
    stage2: Dict[str, torch.Tensor],
    curvature_rank: int = 6,
    chunk_pixels: int = 2048,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return basis [N,R,D,H,W], singular values [N,D,H,W], valid mask."""
    if curvature_rank < 1 or curvature_rank > 8:
        raise ValueError("curvature_rank must be in [1,8]")
    geometry = local_model.geometry
    memory_null = geometry.project_null(stage2["lr_coefficients"])
    bank_lr = build_lr_curvature_bank(memory_null)
    _, _, h, w = stage2["corrected_coefficients"].shape
    mapped = map_lr_bank_to_hr(bank_lr, h, w)
    tangent = flatten_tangent(stage2["tangent_basis"])

    n, q, _, coeff_rank = mapped.shape
    basis_out = mapped.new_zeros(n, q, coeff_rank, curvature_rank)
    singular_out = mapped.new_zeros(n, q, curvature_rank)
    valid_out = torch.zeros(
        n, q, curvature_rank, device=mapped.device, dtype=torch.bool
    )

    for b in range(n):
        for start in range(0, q, chunk_pixels):
            stop = min(start + chunk_pixels, q)
            projected = project_complement_vectors(
                mapped[b, start:stop],
                tangent[b, start:stop],
                geometry.null_projector,
            )
            matrix = projected.transpose(1, 2).float()  # [Q,R,V]
            u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
            leading = singular[:, :1]
            threshold = torch.maximum(
                leading * float(relative_tolerance),
                singular.new_full(
                    leading.shape, float(absolute_tolerance)
                ),
            )
            valid = singular[:, :curvature_rank] > threshold
            selected = (
                u[:, :, :curvature_rank]
                * valid.unsqueeze(1).to(u.dtype)
            )
            basis_out[b, start:stop] = selected.to(basis_out.dtype)
            singular_out[b, start:stop] = singular[:, :curvature_rank].to(
                singular_out.dtype
            )
            valid_out[b, start:stop] = valid

    basis = (
        basis_out.reshape(n, h, w, coeff_rank, curvature_rank)
        .permute(0, 3, 4, 1, 2)
        .contiguous()
    )
    singular = (
        singular_out.reshape(n, h, w, curvature_rank)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    valid = (
        valid_out.reshape(n, h, w, curvature_rank)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return basis.detach(), singular.detach(), valid.detach()


def project_to_curvature(
    basis: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    coordinates = torch.einsum("nrdhw,nrhw->ndhw", basis, vector)
    return torch.einsum("nrdhw,ndhw->nrhw", basis, coordinates)


class LocalCurvatureExtrapolationNet(nn.Module):
    """Frozen Stage-2 plus LR-curvature-authorized global proposal predictor."""

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
            msi                    # hr_msi
            + msi                  # base_msi
            + msi                  # msi_residual
            + rank                 # normalized null seed
            + rank                 # normalized tangent residual
            + rank                 # curvature projector diagonal
            + self.curvature_rank  # normalized singular values
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

        scale = stage2["coefficient_scale"].view(1, -1, 1, 1)
        normalized_null_seed = stage2["null_seed_coefficients"] / scale
        normalized_tangent_residual = stage2["tangent_residual"] / scale
        curvature_projector_diagonal = curvature_basis.square().sum(dim=2)
        singular_scale = curvature_singular[:, :1].clamp_min(1e-8)
        normalized_singular = curvature_singular / singular_scale

        features = torch.cat(
            [
                hr_msi,
                stage2["base_msi"],
                stage2["msi_residual"],
                normalized_null_seed,
                normalized_tangent_residual,
                curvature_projector_diagonal,
                normalized_singular,
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
            "raw_curvature_proposal": raw,
            "normalized_curvature_proposal": normalized,
            "curvature_global_proposal": proposal,
            "curvature_residual": curvature_residual,
            "curvature_corrected_coefficients": corrected,
            "curvature_reconstructed_hsi": reconstructed,
        }
