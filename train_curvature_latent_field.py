import argparse
import os
import torch
import torch.nn.functional as F
from models.curvature_latent_field import CurvatureLatentField, curvature_project


def parse_args():
    p = argparse.ArgumentParser('E30-A curvature latent field predictor')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--device', default='cuda')
    p.add_argument('--save_dir', default='checkpoints/e30_curvature_latent_field/PaviaU')
    return p.parse_args()


def curvature_loss(pred, target):
    return F.smooth_l1_loss(pred, target)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = 0.0
    for batch in loader:
        msi = batch['msi'].to(device)
        lr_hsi = batch['lr_hsi'].to(device)
        p_curv = batch['p_curv'].to(device)
        target = batch['curvature_target'].to(device)

        z64 = model(msi, lr_hsi)
        pred = curvature_project(z64, p_curv)
        loss = curvature_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()

    return loss_sum / max(len(loader), 1)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = CurvatureLatentField(
        msi_channels=6,
        hsi_channels=32,
        latent_dim=args.latent_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Reuse E29 dataset builder.
    # Required batch keys:
    # msi, lr_hsi, p_curv, curvature_target
    from e29_curvature_dataset import build_e29_curvature_loader
    loader = build_e29_curvature_loader()

    os.makedirs(args.save_dir, exist_ok=True)
    best = 1e9

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f'Epoch {epoch:03d}/{args.epochs} loss={loss:.6f}')

        if loss < best:
            best = loss
            torch.save({'model': model.state_dict()},
                       os.path.join(args.save_dir, 'curvature_latent_best.pth'))


if __name__ == '__main__':
    main()
