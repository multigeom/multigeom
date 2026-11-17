"""
Low-dimensional stimuli (core: torch, numpy): the Fisher information J(x), the
multi-scale Fisher information G(x), the local information i(x) = G(x) / 2 and the
mutual information, computed exactly on grids, without a diffusion model.

Two settings. One neuron with Gaussian response noise, r = f(x) + sigma z, and a
one-dimensional stimulus on a uniform grid, by deterministic quadrature: the tuning
curve, the prior density and the blur levels are given as arrays on the grid. And the
linear Gaussian code in any dimension, in closed form, with the nearest-neighbour
estimate of the identity I(R; X) = (1/2) E[Tr G].

All functions take tensors or NumPy arrays, return tensors with the dtype and device
of the input tensors (float64 by default), and report information in nats
(`to_bits` converts). The definitions are those of Azeglio et al. (NeurIPS 2025) for
the local information and of Azeglio, Laquitaine, Ferrari, Chalk (NeurIPS 2026, arXiv
2605.06304) for the multi-scale Fisher information.
"""

from .length import cumulative_distance, distance
from .fisher import compute_fisher, fisher_from_jacobian
from .gaussian import gaussian_blurred_fisher, gaussian_g, gaussian_information
from .information import (
    LN2,
    entropy,
    mutual_information,
    posterior,
    response_density,
    response_grid,
    specific_surprise,
    stimulus_specific_information,
    to_bits,
)
from .multiscale import (
    BlurredCode,
    MultiscaleOutput,
    blur_levels,
    blurred_code,
    compute_g,
)
from .neighbours import nearest_neighbour_directions, nearest_neighbour_information

__all__ = [
    "compute_fisher",
    "compute_g",
    "MultiscaleOutput",
    "blur_levels",
    "blurred_code",
    "BlurredCode",
    "fisher_from_jacobian",
    "mutual_information",
    "specific_surprise",
    "stimulus_specific_information",
    "posterior",
    "entropy",
    "response_density",
    "response_grid",
    "to_bits",
    "LN2",
    "cumulative_distance",
    "distance",
    "gaussian_blurred_fisher",
    "gaussian_g",
    "gaussian_information",
    "nearest_neighbour_directions",
    "nearest_neighbour_information",
]
