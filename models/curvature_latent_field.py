import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class CurvatureLatentField(nn.Module):
    """
    E30-A baseline:
    Predict 64x64 curvature latent field instead of pixel-wise HR curvature residual.

    Input:
        msi:      [B,Cm,128,128]
        lr_hsi:   [B,Ch,32,32]
        optional stage2 feature can be concatenated by caller

    Output:
        z64:      [B,latent_dim,64,64]
    """
    def __init__(self, msi_channels, hsi_channels, latent_dim=32, hidden=64):
        super().__init__()

        self.msi_encoder = nn.Sequential(
            ConvBlock(msi_channels, hidden),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            ConvBlock(hidden, hidden),
        )

        self.hsi_encoder = nn.Sequential(
            ConvBlock(hsi_channels, hidden),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            ConvBlock(hidden, hidden),
        )

        self.fusion = nn.Sequential(
            ConvBlock(hidden * 2, hidden),
            nn.Conv2d(hidden, latent_dim, 1)
        )

    def forward(self, msi, lr_hsi):
        f_msi = self.msi_encoder(msi)
        f_hsi = F.interpolate(lr_hsi, size=f_msi.shape[-2:], mode='bilinear', align_corners=False)
        f_hsi = self.hsi_encoder(f_hsi)
        z64 = self.fusion(torch.cat([f_msi, f_hsi], dim=1))
        return z64


def curvature_project(z64, p_curv, output_size=(128,128)):
    """
    Geometry authorization stage.
    z64 is first lifted to HR and then projected into allowed curvature subspace.
    """
    z_hr = F.interpolate(z64, size=output_size, mode='bilinear', align_corners=False)
    return torch.einsum('bijhw,bjhw->bihw', p_curv, z_hr)
