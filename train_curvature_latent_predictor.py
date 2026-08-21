import argparse
import torch
import torch.nn.functional as F
from models.curvature_latent_predictor import CurvatureLatentPredictor, apply_curvature_projection


def curvature_loss(pred_z, gt_z, pred_residual, gt_residual, beta=0.25):
    latent_loss = F.smooth_l1_loss(pred_z, gt_z)
    residual_loss = F.smooth_l1_loss(pred_residual, gt_residual)
    return latent_loss + beta * residual_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--curvature_latent_gt', required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = CurvatureLatentPredictor(6, 32, 6, latent_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    data = torch.load(args.curvature_latent_gt, map_location='cpu')
    loader = data['loader']

    best = float('inf')
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            msi, lr_null, cond, z_gt, p_curv, c_gt = [x.to(device) for x in batch]
            z_pred = model(msi, lr_null, cond)
            c_pred = apply_curvature_projection(z_pred, p_curv)
            loss = curvature_loss(z_pred, z_gt, c_pred, c_gt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if loss.item() < best:
            best = loss.item()
            torch.save(model.state_dict(), 'curvature_latent_e30_best.pth')
        print(epoch, loss.item())


if __name__ == '__main__':
    main()
