"""
Geometry (core: torch, numpy, scipy): the Fisher information J(x) (`compute_fisher`), the multi-scale Fisher metric G(x) of the
conditional-pair estimator (`compute_g`), and the local metric field of Fig. 5. Measures that compare metrics are in
`multigeom.analysis`.
"""
from .estimator import EstimatorOutput, conditional_pair_estimate
from .estimator import conditional_pair_estimate as compute_g
from .fisher import FisherOutput, fisher_eigendecomposition
from .fisher import fisher_eigendecomposition as compute_fisher

__all__ = ["compute_fisher", "compute_g", "FisherOutput", "EstimatorOutput", "fisher_eigendecomposition",
           "conditional_pair_estimate"]
