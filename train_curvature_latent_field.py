import argparse
import torch
import torch.nn.functional as F
from models.curvature_latent_field import CurvatureLatentField, curvature_project


def parse_args():
    parser = argparse.ArgumentParser('E30-A curvature latent field predictor')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def curvature_loss(pred, target):
    return F.smooth_l1_loss(pred, target)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset/checkpoint binding will reuse E29 data pipeline.
    # This stage intentionally trains only the latent predictor.
    model = CurvatureLatentField(
        msi_channels=6,
        hsi_channels=32,
        latent_dim=args.latent_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    model.train()
    for epoch in range(args.epochs):
        # Placeholder batch interface:
        # msi, lr_hsi, p_curv, target_curvature
        # provided by OMN-Net E29 loader.
        raise RuntimeError(
            'E30-A model initialized. Bind E29 curvature latent dataset before training.'
        )


if __name__ == '__main__':
    main()
