import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class CurvatureLatentPredictor(nn.Module):
    def __init__(self, msi_channels, lr_channels, cond_channels, latent_channels=32, hidden=96, blocks=4):
        super().__init__()
        self.msi_encoder = nn.Sequential(
            nn.Conv2d(msi_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.lr_encoder = nn.Sequential(
            nn.Conv2d(lr_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, 1), nn.GELU())
        self.fusion = nn.Conv2d(hidden * 3, hidden, 1)
        self.body = nn.Sequential(*[ResidualBlock(hidden) for _ in range(blocks)])
        self.head = nn.Conv2d(hidden, latent_channels, 3, padding=1)

    def forward(self, msi_feature, lr_null_feature, curvature_cond):
        target = msi_feature.shape[-2:]
        lr_null_feature = F.interpolate(lr_null_feature, target, mode='bilinear', align_corners=False)
        curvature_cond = F.interpolate(curvature_cond, target, mode='nearest')
        x = torch.cat([
            self.msi_encoder(msi_feature),
            self.lr_encoder(lr_null_feature),
            self.cond_encoder(curvature_cond)
        ], dim=1)
        return self.head(self.body(self.fusion(x)))


def apply_curvature_projection(z64, p_curv):
    z_hr = F.interpolate(z64, scale_factor=2, mode='bilinear', align_corners=False)
    return torch.einsum('bijhw,bjhw->bihw', p_curv, z_hr)
