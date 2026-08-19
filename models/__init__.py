"""Core OMN-Net model components."""

from .spectral_foundation import SpectralFoundation
from .observation_geometry import (
    FixedSpatialDegradation,
    ObservationGeometry,
    build_spectral_response,
    load_foundation_checkpoint,
    project_coefficients,
)
from .local_null_manifold import (
    LocalNullManifoldNet,
    build_local_tangent_field,
)
from .nonlocal_complement import (
    ObservableKeyedComplementMemory,
    flatten_spatial,
    flatten_tangent,
    gather_complement_candidates,
    project_complement_vectors,
    unflatten_spatial,
)

__all__ = [
    "SpectralFoundation",
    "ObservationGeometry",
    "FixedSpatialDegradation",
    "build_spectral_response",
    "load_foundation_checkpoint",
    "project_coefficients",
    "LocalNullManifoldNet",
    "build_local_tangent_field",
    "ObservableKeyedComplementMemory",
    "flatten_spatial",
    "flatten_tangent",
    "gather_complement_candidates",
    "project_complement_vectors",
    "unflatten_spatial",
]
