"""
Geometry (core: torch, numpy, scipy): the Fisher information J(x) (`compute_fisher`),
the multi-scale Fisher metric G(x) (`compute_g`, with the conditional-pair estimator by
default; other methods are plugged in with `register_method`), its integrand at each
noise level (`compute_g_per_scale`), the stimulus features at each noise level
(`compute_env_per_scale`), the envelope and coding spectra (`compute_env_spectrum`,
`compute_coding_spectrum`), and the local metric field of Fig. 5. Measures that compare
metrics are in `multigeom.analysis`.
"""

from .estimator import (
    EstimatorOutput,
    PerScaleOutput,
    conditional_pair_estimate,
    estimate_g,
    estimate_g_per_scale,
)
from .estimator import estimate_g as compute_g
from .estimator import estimate_g_per_scale as compute_g_per_scale
from .env import estimate_env_per_scale
from .env import estimate_env_per_scale as compute_env_per_scale
from .spectrum import (
    CodingSpectrumOutput,
    SpectrumOutput,
    estimate_coding_spectrum,
    estimate_env_spectrum,
)
from .spectrum import estimate_coding_spectrum as compute_coding_spectrum
from .spectrum import estimate_env_spectrum as compute_env_spectrum
from .methods import available_methods, register_method
from .fisher import FisherOutput, fisher_eigendecomposition
from .fisher import fisher_eigendecomposition as compute_fisher

__all__ = [
    "compute_fisher",
    "compute_g",
    "compute_g_per_scale",
    "compute_env_per_scale",
    "compute_env_spectrum",
    "compute_coding_spectrum",
    "FisherOutput",
    "EstimatorOutput",
    "PerScaleOutput",
    "SpectrumOutput",
    "CodingSpectrumOutput",
    "fisher_eigendecomposition",
    "conditional_pair_estimate",
    "estimate_g",
    "estimate_g_per_scale",
    "estimate_env_per_scale",
    "estimate_env_spectrum",
    "estimate_coding_spectrum",
    "available_methods",
    "register_method",
]
