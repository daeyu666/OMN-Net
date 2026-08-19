"""Compatibility-fixed entry point for E15 global complement transport.

The original E15 diagnostic was written against the Stage-3 wrapper naming
(`stage2_coefficients`, `stage2_hsi`) while it directly calls
LocalNullManifoldNet, whose native keys are `corrected_coefficients` and
`reconstructed_hsi`.  This entry point keeps Stage-2 untouched and adds the two
aliases only inside the E15 process before delegating to the original script.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.local_null_manifold import LocalNullManifoldNet


_ORIGINAL_FORWARD = LocalNullManifoldNet.forward


def _e15_compatible_forward(self, *args, **kwargs):
    out = _ORIGINAL_FORWARD(self, *args, **kwargs)
    required = {
        "corrected_coefficients",
        "reconstructed_hsi",
        "basis",
        "coefficient_scale",
        "lr_coefficients",
        "anchor_coefficients",
        "null_seed_coefficients",
        "tangent_basis",
        "tangent_residual",
    }
    missing = sorted(required.difference(out.keys()))
    if missing:
        raise KeyError(
            "E15 LocalNullManifoldNet output is missing required keys: "
            + ", ".join(missing)
        )

    # E15 was originally written with Stage-3 wrapper aliases.  Add them only
    # for this diagnostic process; the validated Stage-2 implementation and its
    # native return contract remain unchanged.
    out["stage2_coefficients"] = out["corrected_coefficients"]
    out["stage2_hsi"] = out["reconstructed_hsi"]
    return out


LocalNullManifoldNet.forward = _e15_compatible_forward

from diagnostics.inspect_degradation_closed_global_transport import main


if __name__ == "__main__":
    main()
