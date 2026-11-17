"""
The Fisher information J(x) of a code with Gaussian response noise of standard
deviation sigma: J(x) = f'(x)^2 / sigma^2 for one neuron and a one-dimensional
stimulus, and J(x) = sum_i grad f_i(x) grad f_i(x)^T / sigma^2 for several neurons and
a stimulus in d dimensions. Units: 1 / (stimulus unit)^2.
"""

from __future__ import annotations

import torch

from ._grid import as_tensors, check_on_grid, grid_step


def compute_fisher(x, tuning, sigma: float) -> torch.Tensor:
    """
    The Fisher information J(x) = f'(x)^2 / sigma^2 of one neuron on a stimulus grid.

    The derivative is the central difference of `numpy.gradient` (one-sided at the two
    ends of the grid).

    Args:
        x: the uniform stimulus grid [n_x].
        tuning: the tuning curve f(x) [n_x].
        sigma: the standard deviation of the response noise.

    Returns:
        J(x) [n_x].
    """
    x, tuning = as_tensors(x, tuning)
    check_on_grid(x, tuning=tuning)
    (slope,) = torch.gradient(tuning, spacing=grid_step(x))
    return slope**2 / sigma**2


def fisher_from_jacobian(jacobian, sigma: float) -> torch.Tensor:
    """
    The Fisher matrix J = A^T A / sigma^2 from the Jacobian A of the tuning curves.

    Args:
        jacobian: the Jacobian A [..., n_neurons, d], A_ij = d f_i / d x_j, with any
            leading batch dimensions (for example one per stimulus).
        sigma: the standard deviation of the response noise of each neuron.

    Returns:
        J [..., d, d].
    """
    (jacobian,) = as_tensors(jacobian)
    return jacobian.mT @ jacobian / sigma**2
