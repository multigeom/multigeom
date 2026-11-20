"""
Analysis (core: torch): measures that compare the geometries of two encoders, from the
eigenvectors and eigenvalues that `multigeom.geometry` returns: the subspace alignment
of two sets of eigenvectors, the Bures-Wasserstein distance between two reconstructed
metrics, and the matching of two sets of eigenvectors for display; and the analyses of
the estimates at each noise scale (`per_scale`).
"""

from .eigenvectors import match_eigenvectors
from .metrics import (
    bures_wasserstein,
    bures_wasserstein_from_overlap,
    subspace_alignment,
)
from .per_scale import (
    align_signs,
    compute_band,
    compute_bootstrap,
    compute_information,
    compute_normalized,
)

__all__ = [
    "subspace_alignment",
    "bures_wasserstein",
    "bures_wasserstein_from_overlap",
    "match_eigenvectors",
    "align_signs",
    "compute_normalized",
    "compute_information",
    "compute_bootstrap",
    "compute_band",
]
