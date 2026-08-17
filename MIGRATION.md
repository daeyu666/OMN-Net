# RAPD-Net → OMN-Net migration

OMN-Net intentionally starts from a clean codebase. This migration keeps only
the components that survived the RAPD-Net diagnostic and ablation process.

## Migrated and retained

- **Spectral foundation**: affine orthogonal LR-HSI basis with PCA
  initialization and signed coefficients.
- **Observation geometry**: `S = R U_r`, exact observable/null projectors, and
  the analytical SRF anchor.
- **Local null manifold**: LR-null local SVD tangent field and the
  basis-invariant `T T^T` projected global coefficient proposal.
- **Physical degradation operator**: fixed Gaussian blur + bicubic resize for
  LR consistency.
- **No-training diagnostics**:
  - observable/null ceiling;
  - local tangent-manifold oracle.

The local-null trainable implementation corresponds to the best validated
RAPD-Net formulation before the repository split: global coefficient proposal
followed by the LR-derived tangent projector. Old Stage-1 checkpoints remain
loadable because the foundation state keys are kept compatible.

## Intentionally not migrated

The following RAPD-Net branches remain historical experiments and are not part
of the OMN-Net mainline:

- NSP / frequency reliability screening;
- adaptive frequency routing;
- shared-structure decoupling;
- observable-to-null cross routing;
- MSI-only local transport;
- masked-null variants;
- Symmetric-Frequency guidance as a mainline dependency;
- old Stage-3 observable/null deterministic duplication;
- diffusion-based uncertainty refinement.

Their files stay in RAPD-Net as an experiment archive.

## OMN-Net research boundary

Implemented now:

1. spectral foundation;
2. analytical observable anchor;
3. local null-manifold extrapolation.

Planned only after new oracle validation:

4. non-local tangent-complement recovery;
5. consistency-calibrated residual trust region.

No unvalidated Stage-3 mechanism is included in the mainline yet.
