"""Local null-manifold extrapolation for OMN-Net.

The HR-MSI observable component is handled analytically. LR-HSI defines a local
spectral tangent space T_p, while the predictor proposes a residual in the fixed
global coefficient coordinates. Only T_p T_p^T r_tilde is allowed to alter the
null-space reconstruction.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .observation_geometry import ObservationGeometry
from .spectral_foundation import SpectralFoundation


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.norm1(self.conv1(x)))
        residual = self.norm2(self.conv2(residual))
        return self.act(x + residual)


@torch.no_grad()
def build_local_tangent_field(
    null_seed: torch.Tensor,
    dimension: int,
    kernel_size: int = 5,
    dilation: int = 2,
    chunk_pixels: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build LR-null local tangent bases by SVD of neighbor-minus-center states."""
    if null_seed.ndim != 4:
        raise ValueError("null_seed must be [N,C,H,W]")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and >= 3")
    if dilation < 1 or chunk_pixels < 1:
        raise ValueError("dilation and chunk_pixels must be positive")

    n, channels, height, width = null_seed.shape
    elements = kernel_size * kernel_size
    max_dimension = min(channels, elements)
    if dimension > max_dimension:
        raise ValueError(
            f"dimension={dimension} exceeds local rank bound {max_dimension}"
        )
    radius = dilation * (kernel_size - 1) // 2
    if height <= radius or width <= radius:
        raise ValueError("spatial size is too small for requested tangent radius")

    padded = F.pad(
        null_seed,
        (radius, radius, radius, radius),
        mode="reflect",
    )
    patches = F.unfold(
        padded,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=1,
    )
    patches = patches.view(n, channels, elements, height, width)
    differences = patches - null_seed.unsqueeze(2)
    matrices = (
        differences.permute(0, 3, 4, 1, 2)
        .reshape(n * height * width, channels, elements)
        .contiguous()
    )

    tangent_flat = null_seed.new_zeros(
        n * height * width, channels, dimension
    )
    singular_flat = null_seed.new_zeros(n * height * width, dimension)

    for start in range(0, matrices.size(0), chunk_pixels):
        stop = min(start + chunk_pixels, matrices.size(0))
        u, singular_values, _ = torch.linalg.svd(
            matrices[start:stop].float(), full_matrices=False
        )
        tangent = u[:, :, :dimension]
        singular = singular_values[:, :dimension]

        max_indices = tangent.abs().argmax(dim=1, keepdim=True)
        pivots = torch.gather(tangent, dim=1, index=max_indices).squeeze(1)
        signs = torch.sign(pivots)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        tangent = tangent * signs.unsqueeze(1)

        tangent_flat[start:stop] = tangent.to(tangent_flat.dtype)
        singular_flat[start:stop] = singular.to(singular_flat.dtype)

    tangent_basis = (
        tangent_flat.reshape(n, height, width, channels, dimension)
        .permute(0, 3, 4, 1, 2)
        .contiguous()
    )
    singular_field = (
        singular_flat.reshape(n, height, width, dimension)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    tangent_scale = singular_field / math.sqrt(max(elements - 1, 1))
    return (
        tangent_basis.detach(),
        tangent_scale.detach(),
        singular_field.detach(),
    )


class GlobalProposalPredictor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        coefficient_channels: int,
        hidden_channels: int = 96,
        blocks: int = 4,
    ):
        super().__init__()
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(blocks)]
        )
        self.head = nn.Conv2d(
            hidden_channels, coefficient_channels, 3, padding=1
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class LocalNullManifoldNet(nn.Module):
    """Analytical observable anchor plus local null-manifold extrapolation."""

    def __init__(
        self,
        foundation: SpectralFoundation,
        spectral_response: torch.Tensor,
        anchor_ridge_ratio: float = 1e-3,
        projector_tolerance: float = 1e-6,
        tangent_dimension: int = 4,
        tangent_kernel_size: int = 5,
        tangent_dilation: int = 2,
        tangent_chunk_pixels: int = 2048,
        proposal_amplitude_multiplier: float = 8.0,
        predictor_hidden_channels: int = 96,
        predictor_blocks: int = 4,
    ):
        super().__init__()
        if proposal_amplitude_multiplier <= 0:
            raise ValueError("proposal_amplitude_multiplier must be positive")

        self.foundation = foundation
        self.n_bands = int(foundation.n_bands)
        self.basis_rank = int(foundation.basis_rank)
        self.msi_channels = int(spectral_response.size(0))
        self.tangent_dimension = int(tangent_dimension)
        self.tangent_kernel_size = int(tangent_kernel_size)
        self.tangent_dilation = int(tangent_dilation)
        self.tangent_chunk_pixels = int(tangent_chunk_pixels)
        self.proposal_amplitude_multiplier = float(
            proposal_amplitude_multiplier
        )

        for parameter in self.foundation.parameters():
            parameter.requires_grad_(False)
        self.foundation.eval()

        with torch.no_grad():
            basis = self.foundation.get_basis().detach()
        self.geometry = ObservationGeometry(
            basis=basis,
            spectral_response=spectral_response,
            anchor_ridge_ratio=anchor_ridge_ratio,
            projector_tolerance=projector_tolerance,
        )

        input_channels = (
            3 * self.msi_channels
            + self.basis_rank
            + self.basis_rank
            + self.tangent_dimension
        )
        self.proposal_predictor = GlobalProposalPredictor(
            input_channels=input_channels,
            coefficient_channels=self.basis_rank,
            hidden_channels=predictor_hidden_channels,
            blocks=predictor_blocks,
        )

    @staticmethod
    def tangent_project(
        tangent_basis: torch.Tensor, proposal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coordinates = torch.einsum(
            "nrdhw,nrhw->ndhw", tangent_basis, proposal
        )
        projected = torch.einsum(
            "nrdhw,ndhw->nrhw", tangent_basis, coordinates
        )
        return projected, coordinates

    def coefficient_scale(self) -> torch.Tensor:
        return self.foundation.coefficient_scale.detach().clamp_min(1e-8)

    def forward(
        self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            basis = self.foundation.get_basis().detach()
            lr_coefficients = self.foundation.encode(
                lr_hsi, basis=basis
            ).detach()

        target_size = hr_msi.shape[-2:]
        bicubic_coefficients = F.interpolate(
            lr_coefficients,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
        base_hsi = self.foundation.decode(
            bicubic_coefficients, basis=basis
        )
        base_msi = self.geometry.hsi_to_msi(base_hsi)
        msi_residual = hr_msi - base_msi

        analytic_residual = self.geometry.analytical_residual(msi_residual)
        anchor_coefficients = bicubic_coefficients + analytic_residual
        anchor_hsi = self.foundation.decode(
            anchor_coefficients, basis=basis
        )

        null_seed = self.geometry.project_null(bicubic_coefficients)
        tangent_basis, tangent_scale, tangent_singular_values = (
            build_local_tangent_field(
                null_seed=null_seed,
                dimension=self.tangent_dimension,
                kernel_size=self.tangent_kernel_size,
                dilation=self.tangent_dilation,
                chunk_pixels=self.tangent_chunk_pixels,
            )
        )

        coefficient_scale = self.coefficient_scale()
        normalized_null_seed = null_seed / coefficient_scale.view(
            1, -1, 1, 1
        )
        global_scale = coefficient_scale.mean().clamp_min(1e-8)
        normalized_tangent_scale = tangent_scale / global_scale
        tangent_projector_diagonal = tangent_basis.square().sum(dim=2)

        predictor_input = torch.cat(
            [
                hr_msi,
                base_msi,
                msi_residual,
                normalized_null_seed,
                tangent_projector_diagonal,
                normalized_tangent_scale,
            ],
            dim=1,
        )
        raw_proposal = self.proposal_predictor(predictor_input)
        normalized_proposal = torch.tanh(raw_proposal)
        proposal_limit = (
            self.proposal_amplitude_multiplier
            * coefficient_scale.view(1, -1, 1, 1)
        )
        proposal = normalized_proposal * proposal_limit

        tangent_projected, tangent_coordinates = self.tangent_project(
            tangent_basis, proposal
        )
        tangent_residual = self.geometry.project_null(tangent_projected)
        off_tangent_proposal = proposal - tangent_projected

        proposal_energy = proposal.double().square().sum()
        tangent_energy = tangent_projected.double().square().sum()
        off_energy = off_tangent_proposal.double().square().sum()

        corrected_coefficients = anchor_coefficients + tangent_residual
        reconstructed_hsi = self.foundation.decode(
            corrected_coefficients, basis=basis
        )

        return {
            "basis": basis,
            "coefficient_scale": coefficient_scale,
            "lr_coefficients": lr_coefficients,
            "bicubic_coefficients": bicubic_coefficients,
            "base_hsi": base_hsi,
            "base_msi": base_msi,
            "msi_residual": msi_residual,
            "analytic_coefficient_residual": analytic_residual,
            "anchor_coefficients": anchor_coefficients,
            "anchor_hsi": anchor_hsi,
            "null_seed_coefficients": null_seed,
            "tangent_basis": tangent_basis,
            "tangent_scale": tangent_scale,
            "tangent_singular_values": tangent_singular_values,
            "tangent_projector_diagonal": tangent_projector_diagonal,
            "raw_global_coefficient_proposal": raw_proposal,
            "normalized_global_coefficient_proposal": normalized_proposal,
            "global_coefficient_proposal": proposal,
            "proposal_limit": proposal_limit,
            "proposal_tangent_coordinates": tangent_coordinates,
            "tangent_projected_proposal": tangent_projected,
            "off_tangent_proposal": off_tangent_proposal,
            "tangent_residual": tangent_residual,
            "tangent_projection_energy_ratio": (
                tangent_energy / proposal_energy.clamp_min(1e-30)
            ).to(proposal.dtype),
            "off_tangent_energy_ratio": (
                off_energy / proposal_energy.clamp_min(1e-30)
            ).to(proposal.dtype),
            "proposal_saturation_ratio": (
                normalized_proposal.detach().abs() > 0.98
            ).float().mean(),
            "corrected_coefficients": corrected_coefficients,
            "reconstructed_hsi": reconstructed_hsi,
            "projected_msi": self.geometry.hsi_to_msi(reconstructed_hsi),
            "observable_rank": self.geometry.observable_rank.to(hr_msi.device),
            **self.geometry.statistics(),
        }
