import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.curvature_latent_predictor import CurvatureLatentPredictor


def curvature_loss(pred_z, gt_z, pred_residual, gt_residual, beta=0.25):
    latent = F.smooth_l1_loss(pred_z, gt_z)
    residual = F.smooth_l1_loss(pred_residual, gt_residual)
    return latent + beta * residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--curvature_latent_gt', required=True)
    args = parser.parse_args()

    device = torch.device(args.device)

    model = CurvatureLatentPredictor(
        msi_channels=6,
        lr_channels=32,
        cond_channels=6,
        latent_channels=args.latent_channels
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0
    )

    # Dataset binding is intentionally separated from the predictor.
    # Existing OMN-Net loaders can provide:
    # msi_feature, lr_null_feature, curvature_condition,
    # gt_latent64, gt_curvature_residual
    checkpoint = torch.load(args.curvature_latent_gt, map_location='cpu')

    train_loader = DataLoader(checkpoint['dataset'], batch_size=1, shuffle=True)

    best = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            msi, lr_null, cond, z_gt, c_gt = [x.to(device) for x in batch]

            z_pred = model(msi, lr_null, cond)
            loss = curvature_loss(z_pred, z_gt, z_pred, z_gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f'Epoch {epoch+1:03d}/{args.epochs} loss={loss.item():.6f}')

        if loss.item() < best or best == 0:
            best = loss.item()
            torch.save(model.state_dict(), 'curvature_latent_e30_best.pth')


if __name__ == '__main__':
    main()
