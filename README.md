# OMN-Net

**OMN-Net** is a clean HSI-MSI fusion / hyperspectral super-resolution research
codebase centered on **observation geometry and MSI-null spectral recovery**.

The repository was split from the RAPD-Net exploration code after the
observability and tangent-manifold diagnostics showed that the main bottleneck
is not the directly MSI-observable coefficient component, but the HR spatial
detail hidden in the MSI-null space.

## Current formulation

### 1. Spectral foundation

LR-HSI defines an affine orthogonal scene subspace

\[
C_{lr}=U_r^\top(Y_{lr}-\mu),\qquad
\hat Y_{lr}=\mu+U_rC_{lr}.
\]

`U_r` is PCA initialized and QR orthogonalized. Coefficients are signed
coordinates, not abundances.

### 2. Analytical observation geometry

With MSI spectral response `R`,

\[
S=RU_r.
\]

OMN-Net constructs the exact row-space and null-space projectors of `S`.
The directly observable coefficient correction is supplied by the analytical
SRF anchor instead of a learned observable branch.

### 3. Local null-manifold extrapolation

The LR-HSI null coefficient field

\[
C_{\text{null}}^0=P_{\text{null}}\,\uparrow C_{lr}
\]

defines a local tangent basis `T_p` through SVD of neighboring null-state
differences. A network predicts a residual proposal in the fixed global
coefficient coordinates, but only

\[
\Delta C_{\text{null}}(p)=T_pT_p^\top\widetilde r(p)
\]

is allowed to affect reconstruction. This is invariant to sign flips,
permutations, and rotations of the local SVD basis.

### 4. Tangent-complement non-local spectral recurrence diagnostic

Innovation point 2 is tested before any trainable Stage-3 backbone is added.
The residual subspace is the local tangent complement inside the MSI null space,
implemented numerically as

\[
P_{\text{comp}}(p)=P_{\text{null}}(I-T_pT_p^\top)P_{\text{null}}.
\]

The diagnostic uses an asymmetric non-local memory:

- the retrieval key is the MSI-observable reduced-response state `S C`;
- the memory value is the LR-HSI-derived null state `P_null C_lr`;
- a retrieved remote state can affect reconstruction only after projection into
  the query pixel's `P_comp` subspace.

This lets the oracle separate three possible bottlenecks: LR-HSI memory value
support, observable-key retrieval, and non-GT aggregation.

## Repository structure

```text
OMN-Net/
├── models/
│   ├── spectral_foundation.py
│   ├── observation_geometry.py
│   ├── local_null_manifold.py
│   └── nonlocal_complement.py
├── diagnostics/
│   ├── inspect_observability.py
│   ├── inspect_local_tangent_oracle.py
│   └── inspect_nonlocal_complement_oracle.py
├── train_spectral_foundation.py
├── train_local_null_manifold.py
├── data_loader.py
├── losses.py
├── metrics.py
├── srf_utils.py
├── utils.py
└── MIGRATION.md
```

RAPD-Net remains the archive for rejected and ablation branches. OMN-Net only
contains mechanisms that are still on the active research path.

## Quick start

Train the LR-HSI spectral foundation:

```bash
python train_spectral_foundation.py \
  --dataset PaviaU \
  --msi_mode srf \
  --srf_band_set wv2_visible6 \
  --basis_rank 32
```

Inspect the observable/null ceiling:

```bash
python diagnostics/inspect_observability.py \
  --dataset PaviaU \
  --msi_mode srf \
  --srf_band_set wv2_visible6
```

Inspect the local tangent oracle:

```bash
python diagnostics/inspect_local_tangent_oracle.py \
  --dataset PaviaU \
  --msi_mode srf \
  --srf_band_set wv2_visible6 \
  --tangent_dimensions 2,4,6,8
```

Train the current local-null mainline:

```bash
python train_local_null_manifold.py \
  --dataset PaviaU \
  --epochs 800 \
  --batch_size 1 \
  --lr 0.0002 \
  --msi_mode srf \
  --srf_band_set wv2_visible6 \
  --tangent_dimension 4 \
  --tangent_kernel_size 5 \
  --tangent_dilation 2 \
  --proposal_amplitude_multiplier 8
```

Inspect innovation point 2 after the Stage-2 checkpoint has been reproduced:

```bash
python diagnostics/inspect_nonlocal_complement_oracle.py \
  --dataset PaviaU \
  --foundation_checkpoint checkpoints/RAPD-Net/basis_for_stage2.pth \
  --local_checkpoint checkpoints/local_null_manifold/PaviaU/local_null_best_psnr.pth \
  --msi_mode srf \
  --srf_band_set wv2_visible6 \
  --tangent_dimension 4 \
  --tangent_kernel_size 5 \
  --tangent_dilation 2 \
  --nonlocal_top_k 32 \
  --nonlocal_exclusion_radius_lr 1
```

The diagnostic writes
`outputs/diagnostics/nonlocal_complement/PaviaU/nonlocal_complement_oracle.json`.
It reports the actual Stage-2 result, the GT `P_comp` ceiling, global LR-state
hard/convex oracles, observable-key Top-K hard/convex oracles, and a fully
non-GT soft-retrieval result.

`models/observation_geometry.py::load_foundation_checkpoint` can also load a
compatible RAPD-Net Stage-1 spectral-basis checkpoint, so the validated old
foundation can be reused without retraining if its `.pth` file is copied into
the new project.

## Next research step

The no-training innovation-point-2 diagnostic is implemented, but a trainable
Stage-3 complement predictor is still intentionally absent. It should be added
only if the LR-HSI non-local state oracle shows meaningful recovery inside
`P_comp`.

If global LR-state oracles are strong but observable-key Top-K oracles are weak,
the next work is retrieval-key design. If Top-K oracles are strong but soft
aggregation is weak, the next work is candidate weighting / residual prediction.
If even the global LR-state oracle is weak, the non-local complement route
should be stopped instead of being rescued by a larger backbone.
