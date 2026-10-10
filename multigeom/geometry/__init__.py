"""
Geometry (core: torch, numpy, scipy): the Fisher information J(x), the multi-scale Fisher metric G(x) of the
conditional-pair estimator, similarity measures between metrics, and the local metric field of Fig. 5.
"""
from .estimator import EstimatorOutput, conditional_pair_estimate
from .fisher import FisherOutput, fisher_eigendecomposition
from .metrics import bures_wasserstein, bures_wasserstein_from_overlap, subspace_alignment

__all__ = [
    "EstimatorOutput", "conditional_pair_estimate", "FisherOutput", "fisher_eigendecomposition",
    "bures_wasserstein", "bures_wasserstein_from_overlap", "subspace_alignment",
]
