"""
Geometry (core: torch, numpy, scipy): the Fisher information J(x) (`compute_fisher`),
the multi-scale Fisher metric G(x) (`compute_g`, with the conditional-pair estimator by
default; other methods are plugged in with `register_method`), and the local metric
field of Fig. 5. Measures that compare metrics are in
`multigeom.analysis`.
"""

from .estimator import EstimatorOutput, conditional_pair_estimate, estimate_g
from .estimator import estimate_g as compute_g
from .methods import available_methods, register_method
from .fisher import FisherOutput, fisher_eigendecomposition
from .fisher import fisher_eigendecomposition as compute_fisher

__all__ = [
    "compute_fisher",
    "compute_g",
    "FisherOutput",
    "EstimatorOutput",
    "fisher_eigendecomposition",
    "conditional_pair_estimate",
    "estimate_g",
    "available_methods",
    "register_method",
]
