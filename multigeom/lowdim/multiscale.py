"""
The multi-scale Fisher information G(x) of one neuron and a one-dimensional stimulus,
by quadrature on grids (Azeglio, Laquitaine, Ferrari, Chalk, NeurIPS 2026, arXiv
2605.06304).

The code is r = f(x) + sigma z, and the stimulus has the prior density p(x). At the
blur level t (a variance, sqrt(t) is the blur width) the blurred stimulus is
x_t = x + sqrt(t) z', and the blurred code is

    p_t(r | x_t) = int p(x | x_t) N(r; f(x), sigma^2) dx,

with Fisher information J_t(x_t). Blurring J_t back with the Gaussian phi_t of
variance t gives the per-scale term (phi_t * J_t)(x), and

    G(x) = int_0^inf (phi_t * J_t)(x) dt,    i(x) = G(x) / 2,

where i(x) is the local information of Azeglio et al. (NeurIPS 2025). Its prior
average is the mutual information I(R; X) in nats. J and J_t are in 1 / (stimulus
unit)^2 and t in (stimulus unit)^2, so G has no unit: it is twice the local
information, in nats, and a change of the stimulus unit leaves it unchanged.

Discretisation. p(r | x) is evaluated on the response grid of
`multigeom.lowdim.information.response_grid`. At each blur level, x_t runs on a grid
that extends the stimulus grid by 6 sqrt(t) on each side, with step max(dx,
sqrt(t) / 12). J_t is the response-grid sum of p_t (d ln p_t / d x_t)^2 with the
derivative by central differences. The back-blur phi_t * J_t is E_z[J_t(x + sqrt(t)
z)] by 40-point Gauss-Hermite quadrature with J_t interpolated linearly, which stays
accurate when sqrt(t) is below the grid step. The t integral is the trapezoid rule
over the given blur levels, accumulated one level at a time, so that memory does not
grow with the number of levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ._grid import as_tensors, check_on_grid, gaussian, grid_step, interp, tiny
from .fisher import compute_fisher
from .information import response_density, response_grid

# Nodes and weights of E over N(0, 1), computed once (torch has no Gauss-Hermite rule).
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(40)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


@dataclass(frozen=True)
class MultiscaleOutput:
    """
    The multi-scale Fisher information of a one-dimensional code, returned by
    `compute_g`. Values on the stimulus grid: J in 1 / (stimulus unit)^2, G and the
    local information in nats.
    """

    g: torch.Tensor  # [n_x] G(x)
    fisher: torch.Tensor  # [n_x] J(x), the per-scale term at t = 0
    local_information: torch.Tensor  # [n_x] i(x) = G(x) / 2
    t: torch.Tensor  # [n_t] the blur levels
    terms: torch.Tensor | None = None  # [n_t, n_x] (phi_t * J_t)(x), if requested


@dataclass(frozen=True)
class BlurredCode:
    """
    The blurred code p_t(r | x_t) at one blur level, returned by `blurred_code`.
    """

    x: torch.Tensor  # [n_q] the points x_t
    mean: torch.Tensor  # [n_q] E[r | x_t]
    std: torch.Tensor  # [n_q] standard deviation of p_t(r | x_t)
    fisher: torch.Tensor  # [n_q] J_t(x_t)


def blur_levels(
    t_min: float = 1e-6,
    t_max: float = 1e4,
    per_decade: int = 100,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Blur levels for `compute_g`: 0, then log-spaced levels from t_min to t_max.

    The number of log-spaced levels is round(per_decade * log10(t_max / t_min)). The
    trapezoid rule over these levels misses the part of the integral above t_max,
    which is small once sqrt(t_max) is well beyond the width of the prior, and
    replaces the part below t_min by one trapezoid between J and the per-scale term
    at t_min.

    Args:
        t_min: the smallest positive blur level (a variance).
        t_max: the largest blur level.
        per_decade: the number of levels per decade.
        dtype: the dtype of the result.
        device: the device of the result.

    Returns:
        The blur levels [n_t], increasing, the first one 0.
    """
    n = round(per_decade * math.log10(t_max / t_min))
    t = torch.logspace(
        math.log10(t_min), math.log10(t_max), n, dtype=dtype, device=device
    )
    return torch.cat([t.new_zeros(1), t])


def _blur_posterior(x, prior, dx, t):
    """p(x | x_t) [n_y, n_x] on the grid of x_t [n_y] at one blur level t > 0."""
    s = math.sqrt(t)
    start, stop = x[0].item() - 6 * s, x[-1].item() + 6 * s + 1e-9
    step = max(dx, s / 12)
    n_y = math.ceil((stop - start) / step)
    y = start + torch.arange(n_y, dtype=x.dtype, device=x.device) * step
    weights = gaussian(y[:, None], x[None, :], s) * prior[None, :] * dx
    return y, weights / weights.sum(1, keepdim=True).clamp_min(tiny(x.dtype))


def _blurred_fisher(y, post, p_r_x, dr):
    """J_t on the grid of x_t, from p(x | x_t) and p(r | x)."""
    p_t = post @ p_r_x
    (score,) = torch.gradient(
        p_t.clamp_min(tiny(p_t.dtype)).log(), spacing=(y[1] - y[0]).item(), dim=0
    )
    return (p_t * score**2).sum(1) * dr


def _back_blur(x, y, fisher_t, t):
    """(phi_t * J_t)(x) = E_z[J_t(x + sqrt(t) z)] by Gauss-Hermite quadrature."""
    nodes = torch.as_tensor(_GH_NODES, dtype=x.dtype, device=x.device)
    weights = torch.as_tensor(_GH_WEIGHTS, dtype=x.dtype, device=x.device)
    return weights @ interp(x[None, :] + math.sqrt(t) * nodes[:, None], y, fisher_t)


def _checked(x, tuning, prior, sigma, n_r, r_pad):
    x, tuning, prior = as_tensors(x, tuning, prior)
    dx = grid_step(x)
    check_on_grid(x, tuning=tuning, prior=prior)
    r = response_grid(tuning, sigma, n_r, r_pad)
    return x, tuning, prior, dx, response_density(tuning, sigma, r), r


def compute_g(
    x,
    tuning,
    prior,
    sigma: float,
    t,
    n_r: int = 361,
    r_pad: float = 6.0,
    keep_terms: bool = False,
) -> MultiscaleOutput:
    """
    The multi-scale Fisher information G(x) = int_0^inf (phi_t * J_t)(x) dt of one
    neuron with Gaussian response noise (see the module docstring for the method).

    Args:
        x: the uniform stimulus grid [n_x].
        tuning: the tuning curve f(x) [n_x].
        prior: the prior density p(x) [n_x], normalised on the grid.
        sigma: the standard deviation of the response noise.
        t: the blur levels [n_t], increasing, the first one 0 (see `blur_levels`).
        n_r: the number of points of the response grid.
        r_pad: the margin of the response grid, in units of sigma.
        keep_terms: also return the per-scale terms (phi_t * J_t)(x) [n_t, n_x].

    Returns:
        MultiscaleOutput with G, J and the local information i = G / 2 [n_x], the
        blur levels [n_t], and the per-scale terms if `keep_terms`.
    """
    x, tuning, prior, dx, p_r_x, r = _checked(x, tuning, prior, sigma, n_r, r_pad)
    (t,) = as_tensors(t)
    t = t.to(x)
    if t.dim() != 1 or t.numel() < 2 or t[0] != 0 or (t.diff() <= 0).any():
        raise ValueError("t must be 1-D and increasing, with t[0] = 0")
    dr = (r[1] - r[0]).item()
    fisher = compute_fisher(x, tuning, sigma)
    previous, g = fisher, torch.zeros_like(x)
    terms = [fisher] if keep_terms else None
    levels = t.tolist()
    for t0, t1 in zip(levels[:-1], levels[1:]):
        y, post = _blur_posterior(x, prior, dx, t1)
        term = _back_blur(x, y, _blurred_fisher(y, post, p_r_x, dr), t1)
        g = g + 0.5 * (t1 - t0) * (previous + term)
        previous = term
        if keep_terms:
            terms.append(term)
    return MultiscaleOutput(
        g=g,
        fisher=fisher,
        local_information=0.5 * g,
        t=t,
        terms=torch.stack(terms) if keep_terms else None,
    )


def blurred_code(
    x,
    tuning,
    prior,
    sigma: float,
    t: float,
    query=None,
    n_r: int = 361,
    r_pad: float = 6.0,
) -> BlurredCode:
    """
    The blurred code p_t(r | x_t) at one blur level: its mean, its standard deviation
    and its Fisher information J_t, interpolated linearly at the query points.

    The mean and the variance follow from p(x | x_t) by the law of total variance:
    var = sigma^2 + E[f^2 | x_t] - E[f | x_t]^2. At t = 0 the code is not blurred: the
    mean is f, the standard deviation sigma and J_t the Fisher information J.

    Args:
        x: the uniform stimulus grid [n_x].
        tuning: the tuning curve f(x) [n_x].
        prior: the prior density p(x) [n_x], normalised on the grid.
        sigma: the standard deviation of the response noise.
        t: the blur level (a variance), t >= 0.
        query: the points x_t [n_q]. Defaults to the stimulus grid.
        n_r: the number of points of the response grid.
        r_pad: the margin of the response grid, in units of sigma.

    Returns:
        BlurredCode with the query points, the mean, the standard deviation and J_t,
        each [n_q].
    """
    x, tuning, prior, dx, p_r_x, r = _checked(x, tuning, prior, sigma, n_r, r_pad)
    query = x if query is None else as_tensors(query)[0].to(x)
    if t < 0:
        raise ValueError(f"the blur level must be >= 0, got {t}")
    if t == 0:
        return BlurredCode(
            x=query,
            mean=interp(query, x, tuning),
            std=torch.full_like(query, sigma),
            fisher=interp(query, x, compute_fisher(x, tuning, sigma)),
        )
    y, post = _blur_posterior(x, prior, dx, t)
    fisher_t = _blurred_fisher(y, post, p_r_x, (r[1] - r[0]).item())
    mean = post @ tuning
    std = (sigma**2 + post @ tuning**2 - mean**2).sqrt()
    return BlurredCode(
        x=query,
        mean=interp(query, y, mean),
        std=interp(query, y, std),
        fisher=interp(query, y, fisher_t),
    )
