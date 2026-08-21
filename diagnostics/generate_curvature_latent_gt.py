import torch


def generate_curvature_latent_target(curvature_residual, p_curv, solver):
    """Generate compressed 64x64 curvature latent target from E29 inversion."""
    target = solver(curvature_residual, p_curv)
    return target


if __name__ == '__main__':
    print('E30 curvature latent GT generator ready')
