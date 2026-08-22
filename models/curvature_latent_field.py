"""E30: half-resolution curvature latent-field recovery for OMN-Net.

E29 showed that the rank-6 curvature-authorized GT residual can still reach
46+ dB when it is parameterized by a continuous half-resolution latent field,
while an LR-grid latent field is too coarse.  This module therefore replaces
per-HR-pixel curvature prediction with

    z_mid -> bilinear upsample -> P_curv -> Delta C.

The latent field is represented in the fixed 32-D global coefficient basis.
HR-MSI may guide the spatial latent field, but it can never write a free
MSI-unobservable spectrum: every predicted correction is projected through the
LR-HSI-derived P_curv authorization field before reconstruction.

Two predictor-input variants are supported:
* msi_only: E30-A identifiability baseline.  The predictor sees only a lossless
  PixelUnshuffle representation of HR-MSI at the latent resolution.
* fusion: E30-B.  The same predictor additionally receives legal LR-HSI / frozen
  Stage-2 state descriptors at the latent resolution.

No GT latent coordinate is part of the model.  Training should supervise the
unique authorized residual P_curv Up(z), avoiding arbitrary regression to one
possibly non-unique latent least-squares solution.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_curvature_extrapolation import build_curvature_basis, project_to_curvature
from .local_null_manifold import LocalNullManifoldNet, ResidualBlock


class CurvatureLatentPredictor(nn.Module):
    """Simple fully-convolutional predictor operating only at latent scale."""

    def __init__(
        self,
        input_channels: int,
        coefficient_channels: int,
        hidden_channels: int = 96,
        blocks: int = 4,
    ):
        super().__init__()
        if input_channels < 1 or coefficient_channels < 1:
            raise ValueError("input/coefficient channels must be positive")
        if hidden_channels < 1 or blocks < 1:
            raise ValueError("hidden_channels and blocks must be positive")
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
            hidden_channels, coefficient_channels, kernel_size=3, padding=1
        )
        # Zero initialization makes epoch-0 exactly Stage-2, matching the E17
        # experimental convention and preventing an arbitrary initial writeback.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class CurvatureLatentFieldNet(nn.Module):
    """Frozen Stage-2 plus an E29-inspired spatially compressed latent field."""

    VALID_VARIANTS = {"msi_only", "fusion"}

    def __init__(
        self,
        local_model: LocalNullManifoldNet,
        variant: str = "fusion",
        latent_stride: int = 2,
        curvature_rank: int = 6,
        curvature_svd_chunk_pixels: int = 2048,
        curvature_svd_tolerance: float = 1e-5,
        curvature_abs_tolerance: float = 1e-9,
        latent_amplitude_multiplier: float = 8.0,
        predictor_hidden_channels: int = 96,
        predictor_blocks: int = 4,
    ):
        super().__init__()
        variant = str(variant).lower()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(self.VALID_VARIANTS)}, got {variant}"
            )
        if latent_stride < 1:
            raise ValueError("latent_stride must be positive")
        if curvature_rank < 1 or curvature_rank > 8:
            raise ValueError("curvature_rank must be in [1,8]")
        if latent_amplitude_multiplier <= 0:
            raise ValueError("latent_amplitude_multiplier must be positive")

        self.local_model = local_model
        self.variant = variant
        self.latent_stride = int(latent_stride)
        self.curvature_rank = int(curvature_rank)
        self.curvature_svd_chunk_pixels = int(curvature_svd_chunk_pixels)
        self.curvature_svd_tolerance = float(curvature_svd_tolerance)
        self.curvature_abs_tolerance = float(curvature_abs_tolerance)
        self.latent_amplitude_multiplier = float(latent_amplitude_multiplier)

        for parameter in self.local_model.parameters():
            parameter.requires_grad_(False)
        self.local_model.eval()

        rank = int(local_model.basis_rank)
        msi = int(local_model.msi_channels)
        tiled_msi_channels = msi * self.latent_stride * self.latent_stride

        input_channels = tiled_msi_channels
        if self.variant == "fusion":
            # HR MSI tiles + MSI residual tiles + LR spectral state + frozen
            # Stage-2 tangent residual + P_curv diagonal + curvature singulars.
            input_channels += (
                tiled_msi_channels
                + rank
                + rank
                + rank
                + self.curvature_rank
            )

        self.predictor = CurvatureLatentPredictor(
            input_channels=input_channels,
            coefficient_channels=rank,
            hidden_channels=predictor_hidden_channels,
            blocks=predictor_blocks,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Stage-2 and the spectral foundation remain frozen/eval even while the
        # new latent predictor is trained.
        self.local_model.eval()
        return self

    def trainable_parameters(self):
        return self.predictor.parameters()

    @staticmethod
    def _resize_to_latent(
        field: torch.Tensor,
        latent_size: Tuple[int, int],
        mode: str,
    ) -> torch.Tensor:
        if tuple(field.shape[-2:]) == tuple(latent_size):
            return field
        if mode == "area":
            return F.interpolate(field, size=latent_size, mode="area")
        return F.interpolate(
            field,
            size=latent_size,
            mode=mode,
            align_corners=False,
        )

    def _pixel_unshuffle(self, field: torch.Tensor) -> torch.Tensor:
        stride = self.latent_stride
        h, w = field.shape[-2:]
        if h % stride != 0 or w % stride != 0:
            raise ValueError(
                f"HR size {(h, w)} must be divisible by latent_stride={stride}"
            )
        if stride == 1:
            return field
        return F.pixel_unshuffle(field, downscale_factor=stride)

    def _build_predictor_features(
        self,
        stage2: Dict[str, torch.Tensor],
        hr_msi: torch.Tensor,
        curvature_basis: torch.Tensor,
        curvature_singular: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h, w = hr_msi.shape[-2:]
        latent_size = (h // self.latent_stride, w // self.latent_stride)
        msi_tiles = self._pixel_unshuffle(hr_msi)
        features: List[torch.Tensor] = [msi_tiles]
        diagnostics: Dict[str, torch.Tensor] = {
            "latent_msi_tiles": msi_tiles,
        }

        if self.variant == "fusion":
            scale = stage2["coefficient_scale"].view(1, -1, 1, 1)
            normalized_lr = stage2["lr_coefficients"] / scale
            lr_state = self._resize_to_latent(
                normalized_lr, latent_size, mode="bilinear"
            )

            tangent_state = self._resize_to_latent(
                stage2["tangent_residual"] / scale,
                latent_size,
                mode="area",
            )

            projector_diagonal = curvature_basis.square().sum(dim=2)
            projector_state = self._resize_to_latent(
                projector_diagonal, latent_size, mode="area"
            )

            singular_scale = curvature_singular[:, :1].clamp_min(1e-8)
            normalized_singular = curvature_singular / singular_scale
            singular_state = self._resize_to_latent(
                normalized_singular, latent_size, mode="area"
            )

            residual_tiles = self._pixel_unshuffle(stage2["msi_residual"])
            features.extend(
                [
                    residual_tiles,
                    lr_state,
                    tangent_state,
                    projector_state,
                    singular_state,
                ]
            )
            diagnostics.update(
                {
                    "latent_msi_residual_tiles": residual_tiles,
                    "latent_lr_spectral_state": lr_state,
                    "latent_tangent_state": tangent_state,
                    "latent_curvature_projector_diagonal": projector_state,
                    "latent_curvature_singular_state": singular_state,
                }
            )

        return torch.cat(features, dim=1), diagnostics

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
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

        predictor_input, feature_diagnostics = self._build_predictor_features(
            stage2,
            hr_msi,
            curvature_basis,
            curvature_singular,
        )
        raw_latent = self.predictor(predictor_input)
        normalized_latent = torch.tanh(raw_latent)

        coefficient_scale = stage2["coefficient_scale"].view(1, -1, 1, 1)
        latent_limit = self.latent_amplitude_multiplier * coefficient_scale
        latent = normalized_latent * latent_limit

        latent_hr = F.interpolate(
            latent,
            size=stage2["corrected_coefficients"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        curvature_residual = project_to_curvature(curvature_basis, latent_hr)
        corrected = stage2["corrected_coefficients"] + curvature_residual
        reconstructed = self.local_model.foundation.decode(
            corrected, basis=stage2["basis"]
        )

        return {
            **stage2,
            "e30_variant": self.variant,
            "curvature_basis": curvature_basis,
            "curvature_singular_values": curvature_singular,
            "curvature_valid_mask": curvature_valid,
            "curvature_projector_diagonal": curvature_basis.square().sum(dim=2),
            "latent_predictor_input": predictor_input,
            **feature_diagnostics,
            "raw_curvature_latent": raw_latent,
            "normalized_curvature_latent": normalized_latent,
            "curvature_latent": latent,
            "curvature_latent_limit": latent_limit,
            "curvature_latent_hr": latent_hr,
            "curvature_residual": curvature_residual,
            "curvature_corrected_coefficients": corrected,
            "curvature_reconstructed_hsi": reconstructed,
        }
