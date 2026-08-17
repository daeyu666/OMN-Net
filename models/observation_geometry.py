"""Observation geometry and fixed physical operators for OMN-Net.

The analytical SRF anchor handles the directly MSI-observable coefficient
component. The exact row-space / null-space split is constructed from S = R U.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_foundation import SpectralFoundation


def build_spectral_response(info: dict) -> torch.Tensor:
    n_bands = int(info["n_bands"])
    n_msi = int(info["n_select_bands"])
    srf = info.get("srf_weights")
    if srf is not None:
        response = torch.from_numpy(np.asarray(srf, dtype=np.float32))
    else:
        indices = np.linspace(0, n_bands - 1, n_msi).round().astype(np.int64)
        response = torch.zeros(n_msi, n_bands, dtype=torch.float32)
        response[torch.arange(n_msi), torch.from_numpy(indices)] = 1.0
    if response.shape != (n_msi, n_bands):
        raise ValueError(
            f"Invalid spectral response {tuple(response.shape)}, expected "
            f"{(n_msi, n_bands)}"
        )
    return response


def project_coefficients(
    projector: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("rk,nkhw->nrhw", projector, coefficients)


class ObservationGeometry(nn.Module):
    """Exact observable/null projectors plus analytical SRF backprojection."""

    def __init__(
        self,
        basis: torch.Tensor,
        spectral_response: torch.Tensor,
        anchor_ridge_ratio: float = 1e-3,
        projector_tolerance: float = 1e-6,
    ):
        super().__init__()
        if basis.ndim != 2 or spectral_response.ndim != 2:
            raise ValueError("basis and spectral_response must be matrices")
        if spectral_response.size(1) != basis.size(0):
            raise ValueError("spectral response and basis band counts differ")
        if anchor_ridge_ratio <= 0 or projector_tolerance <= 0:
            raise ValueError("ridge ratio and tolerance must be positive")

        reduced = spectral_response.float() @ basis.float()
        _, singular_values, vh = torch.linalg.svd(reduced, full_matrices=True)
        threshold = projector_tolerance * singular_values.max().clamp_min(1e-12)
        rank = int((singular_values > threshold).sum().item())

        row_basis = vh[:rank].transpose(0, 1).contiguous()
        observable = row_basis @ row_basis.transpose(0, 1)
        identity = torch.eye(
            basis.size(1), dtype=observable.dtype, device=observable.device
        )
        null = identity - observable

        gram = reduced @ reduced.transpose(0, 1)
        gram_scale = torch.trace(gram) / max(reduced.size(0), 1)
        actual_ridge = anchor_ridge_ratio * gram_scale
        regularized = gram + actual_ridge * torch.eye(
            reduced.size(0), dtype=gram.dtype, device=gram.device
        )
        inverse_rhs = torch.eye(
            reduced.size(0), dtype=gram.dtype, device=gram.device
        )
        backprojector = reduced.transpose(0, 1) @ torch.linalg.solve(
            regularized, inverse_rhs
        )

        self.register_buffer("spectral_response", spectral_response.float())
        self.register_buffer("reduced_response", reduced.contiguous())
        self.register_buffer(
            "observable_projector", observable.detach().contiguous()
        )
        self.register_buffer("null_projector", null.detach().contiguous())
        self.register_buffer(
            "coefficient_backprojector", backprojector.detach().contiguous()
        )
        self.register_buffer(
            "observable_singular_values", singular_values.detach().contiguous()
        )
        self.register_buffer(
            "observable_rank", torch.tensor(rank, dtype=torch.int64)
        )
        self.register_buffer("actual_anchor_ridge", actual_ridge.reshape(()))

    def hsi_to_msi(self, hsi: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "mb,nbhw->nmhw", self.spectral_response.to(hsi), hsi
        )

    def analytical_residual(self, msi_residual: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "rm,nmhw->nrhw",
            self.coefficient_backprojector.to(msi_residual),
            msi_residual,
        )

    def project_observable(self, coefficients: torch.Tensor) -> torch.Tensor:
        return project_coefficients(
            self.observable_projector.to(coefficients), coefficients
        )

    def project_null(self, coefficients: torch.Tensor) -> torch.Tensor:
        return project_coefficients(
            self.null_projector.to(coefficients), coefficients
        )

    def statistics(self) -> Dict[str, torch.Tensor]:
        obs = self.observable_projector
        null = self.null_projector
        identity = torch.eye(obs.size(0), dtype=obs.dtype, device=obs.device)
        return {
            "observable_idempotence_error": (obs @ obs - obs).abs().max(),
            "null_idempotence_error": (null @ null - null).abs().max(),
            "projector_complement_error": (obs + null - identity).abs().max(),
            "projector_orthogonality_error": (obs @ null).abs().max(),
            "reduced_response_null_leakage": (
                self.reduced_response @ null
            ).abs().max(),
        }


class FixedSpatialDegradation(nn.Module):
    """Fixed Gaussian blur followed by bicubic resize."""

    def __init__(
        self, channels: int, kernel_size: int = 5, sigma: float = 2.0
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        coordinates = torch.arange(kernel_size, dtype=torch.float32)
        coordinates = coordinates - (kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
        self.register_buffer(
            "kernel", kernel.repeat(channels, 1, 1, 1), persistent=False
        )
        self.channels = int(channels)
        self.padding = kernel_size // 2

    def forward(
        self, x: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != self.channels:
            raise ValueError(
                f"Expected [N,{self.channels},H,W], got {tuple(x.shape)}"
            )
        padded = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
            mode="reflect",
        )
        blurred = F.conv2d(
            padded, self.kernel.to(dtype=x.dtype), groups=self.channels
        )
        return F.interpolate(
            blurred, size=target_size, mode="bicubic", align_corners=False
        )


def load_foundation_checkpoint(
    path: str, expected_n_bands: int, device: torch.device
) -> Tuple[SpectralFoundation, dict]:
    """Load an OMN foundation or a compatible RAPD Stage-1 checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Foundation checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)

    model_state = state.get("model", state)
    if "raw_basis" not in model_state:
        raise KeyError("Checkpoint does not contain raw_basis")
    raw_basis = model_state["raw_basis"]
    extra = state.get("extra", {})
    n_bands = int(extra.get("n_bands", raw_basis.shape[0]))
    basis_rank = int(extra.get("basis_rank", raw_basis.shape[1]))
    if n_bands != expected_n_bands:
        raise ValueError(
            f"Checkpoint bands={n_bands}, dataset bands={expected_n_bands}"
        )

    model = SpectralFoundation(n_bands=n_bands, basis_rank=basis_rank).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state
