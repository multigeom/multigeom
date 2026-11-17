"""
Information of one neuron about a one-dimensional stimulus, by quadrature on grids.

The code is r = f(x) + sigma z with z standard Gaussian, and the stimulus has the
prior density p(x) on a uniform grid. The response density p(r | x) = N(r; f(x),
sigma^2) is evaluated on a uniform response grid that covers f(x) +- r_pad sigma, and
every integral is a Riemann sum on the two grids. All quantities are in nats (natural
logarithms). `to_bits` converts them to bits.

The specific surprise and the stimulus-specific information are the stimulus-specific
measures compared with the local information in Laquitaine, Azeglio et al. (NeurIPS
2025).
"""

from __future__ import annotations

import math

import torch

from ._grid import as_tensors, check_on_grid, gaussian, grid_step, tiny

LN2 = math.log(2.0)


def to_bits(nats):
    """
    Convert an information from nats to bits.

    Args:
        nats: a value in nats (a number or a tensor).

    Returns:
        The value in bits, nats / ln 2.
    """
    return nats / LN2


def response_grid(
    tuning, sigma: float, n_r: int = 361, r_pad: float = 6.0
) -> torch.Tensor:
    """
    The uniform response grid used by the functions of this module.

    Args:
        tuning: the tuning curve f(x) on the stimulus grid [n_x].
        sigma: the standard deviation of the response noise.
        n_r: the number of response values.
        r_pad: the grid covers [min f - r_pad sigma, max f + r_pad sigma].

    Returns:
        The response grid [n_r], with the dtype and device of `tuning`.
    """
    (tuning,) = as_tensors(tuning)
    lo = tuning.min().item() - r_pad * sigma
    hi = tuning.max().item() + r_pad * sigma
    return torch.linspace(lo, hi, n_r, dtype=tuning.dtype, device=tuning.device)


def response_density(tuning, sigma: float, r) -> torch.Tensor:
    """
    The response density p(r | x) = N(r; f(x), sigma^2).

    Args:
        tuning: the tuning curve f(x) on the stimulus grid [n_x].
        sigma: the standard deviation of the response noise.
        r: the responses [n_r].

    Returns:
        p(r | x) [n_x, n_r].
    """
    tuning, r = as_tensors(tuning, r)
    return gaussian(r[None, :], tuning[:, None], sigma)


def _code(x, tuning, prior, sigma, n_r, r_pad):
    x, tuning, prior = as_tensors(x, tuning, prior)
    dx = grid_step(x)
    check_on_grid(x, tuning=tuning, prior=prior)
    r = response_grid(tuning, sigma, n_r, r_pad)
    p_r_x = response_density(tuning, sigma, r)
    p_r = (prior[:, None] * p_r_x).sum(0) * dx
    return x, prior, dx, r, (r[1] - r[0]).item(), p_r_x, p_r


def posterior(x, tuning, prior, sigma: float, r) -> torch.Tensor:
    """
    The posterior p(x | r) = p(r | x) p(x) / p(r) on the stimulus grid.

    Args:
        x: the uniform stimulus grid [n_x].
        tuning: the tuning curve f(x) [n_x].
        prior: the prior density p(x) [n_x], normalised on the grid.
        sigma: the standard deviation of the response noise.
        r: the responses [n_r].

    Returns:
        p(x | r) [n_x, n_r], one density over the stimulus grid per response.
    """
    x, tuning, prior, r = as_tensors(x, tuning, prior, r)
    dx = grid_step(x)
    check_on_grid(x, tuning=tuning, prior=prior)
    joint = response_density(tuning, sigma, r) * prior[:, None]
    return joint / (joint.sum(0) * dx).clamp_min(tiny(joint.dtype))


def entropy(density, step: float, dim: int = 0) -> torch.Tensor:
    """
    The differential entropy -sum p ln p * step of densities on a uniform grid, in nats.

    Args:
        density: the densities, sampled on a grid along `dim`.
        step: the grid step.
        dim: the dimension of the grid.

    Returns:
        The entropies, with `dim` removed.
    """
    (density,) = as_tensors(density)
    logp = density.clamp_min(tiny(density.dtype)).log()
    return -(density * logp).sum(dim) * step


def mutual_information(
    x, tuning, prior, sigma: float, n_r: int = 361, r_pad: float = 6.0
) -> torch.Tensor:
    """
    The mutual information I(R; X) between the response and the stimulus, in nats.

    Args:
        x: the uniform stimulus grid [n_x].
        tuning: the tuning curve f(x) [n_x].
        prior: the prior density p(x) [n_x], normalised on the grid.
        sigma: the standard deviation of the response noise.
        n_r: the number of points of the response grid (see `response_grid`).
        r_pad: the margin of the response grid, in units of sigma.

    Returns:
        I(R; X), a scalar tensor.
    """
    _, prior, dx, _, dr, p_r_x, p_r = _code(x, tuning, prior, sigma, n_r, r_pad)
    return (prior[:, None] * _log_ratio_terms(p_r_x, p_r)).sum() * dx * dr


def _log_ratio_terms(p_r_x, p_r):
    floor = tiny(p_r_x.dtype)
    return p_r_x * torch.log(p_r_x.clamp_min(floor) / p_r.clamp_min(floor))


def specific_surprise(
    x, tuning, prior, sigma: float, n_r: int = 361, r_pad: float = 6.0
) -> torch.Tensor:
    """
    The specific surprise D_KL(p(r | x) || p(r)) of each stimulus, in nats.

    Its prior average is the mutual information.

    Args:
        x: the uniform stimulus grid [n_x].
        tuning: the tuning curve f(x) [n_x].
        prior: the prior density p(x) [n_x], normalised on the grid.
        sigma: the standard deviation of the response noise.
        n_r: the number of points of the response grid (see `response_grid`).
        r_pad: the margin of the response grid, in units of sigma.

    Returns:
        The specific surprise on the stimulus grid [n_x].
    """
    _, _, _, _, dr, p_r_x, p_r = _code(x, tuning, prior, sigma, n_r, r_pad)
    return _log_ratio_terms(p_r_x, p_r).sum(1) * dr


def stimulus_specific_information(
    x, tuning, prior, sigma: float, n_r: int = 361, r_pad: float = 6.0
) -> torch.Tensor:
    """
    The stimulus-specific information H(X) - E_{r | x} H(X | r) of each stimulus, in
    nats.

    Its prior average is the mutual information.

    Args:
        x: the uniform stimulus grid [n_x].
        tuning: the tuning curve f(x) [n_x].
        prior: the prior density p(x) [n_x], normalised on the grid.
        sigma: the standard deviation of the response noise.
        n_r: the number of points of the response grid (see `response_grid`).
        r_pad: the margin of the response grid, in units of sigma.

    Returns:
        The stimulus-specific information on the stimulus grid [n_x].
    """
    _, prior, dx, _, dr, p_r_x, p_r = _code(x, tuning, prior, sigma, n_r, r_pad)
    post = p_r_x * prior[:, None] / p_r.clamp_min(tiny(p_r.dtype))[None, :]
    h_post = entropy(post, dx, dim=0)
    return entropy(prior, dx) - (p_r_x * h_post[None, :]).sum(1) * dr
