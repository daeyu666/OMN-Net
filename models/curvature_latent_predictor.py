import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.net(x)


class CurvatureLatentPredictor(nn.Module):
    """
    E30:
    Predict 64x64 curvature latent field instead of HR curvature residual.

    Input:
      msi_feature:      HR MSI feature
      lr_null_feature:  LR-HSI null coefficient feature lifted to 64x64
      curvature_cond:   curvature basis/singular-value condition

    Output:
      z64: latent curvature coefficient field
    """
    def __init__(self, msi_channels, lr_channels, cond_channels,
                 latent_channels=32, hidden=96, blocks=4):
        super().__init__()

        self.msi_encoder = nn.Sequential(
            nn.Conv2d(msi_channels, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GELU()
        )

        self.lr_encoder = nn.Sequential(
            nn.Conv2d(lr_channels, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GELU()
        )

        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, 1),
            nn.GELU()
        )

        self.fusion = nn.Conv2d(hidden * 3, hidden, 1)
        self.body = nn.Sequential(*[ResidualBlock(hidden) for _ in range(blocks)])
        self.head = nn.Conv2d(hidden, latent_channels, 3, 1, 1)

    def forward(self, msi_feature, lr_null_feature, curvature_cond):
        if lr_null_feature.shape[-2:] != msi_feature.shape[-2:]:
            lr_null_feature = F.interpolate(
                lr_null_feature,
                size=msi_feature.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        if curvature_cond.shape[-2:] != msi_feature.shape[-2:]:
            curvature_cond = F.interpolate(
                curvature_cond,
                size=msi_feature.shape[-2:],
                mode="nearest"
            )

        a = self.msi_encoder(msi_feature)
        b = self.lr_encoder(lr_null_feature)
        c = self.cond_encoder(curvature_cond)

        x = self.fusion(torch.cat([a, b, c], dim=1))
        x = self.body(x)
        return self.head(x)


def curvature_project(z_hr, p_curv):
    """Apply curvature authorization projector."""
    return torch.einsum('bijhw,bjhw->bihw', p_curv, z_hr)
