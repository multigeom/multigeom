"""
Lengths under a one-dimensional metric: the distance between a and b is
int_a^b sqrt(metric(x)) dx, computed with the trapezoid rule on the grid of the metric.
With the Fisher information J as the metric this is the Fisher distance, with the
multi-scale Fisher information G the G distance. J is in 1 / (stimulus unit)^2, so
the Fisher distance has no unit. G has no unit, so the G distance is in the stimulus
unit.
"""

from __future__ import annotations

import torch

from ._grid import as_tensors, check_on_grid, interp


def cumulative_distance(x, metric) -> torch.Tensor:
    """
    The distance from the first grid point, int_{x_0}^{x} sqrt(metric) dx.

    Args:
        x: the increasing grid [n_x] (it need not be uniform).
        metric: the metric on the grid [n_x], non-negative.

    Returns:
        The distance from x_0 to each grid point [n_x], 0 at x_0.
    """
    x, metric = as_tensors(x, metric)
    check_on_grid(x, metric=metric)
    if (metric < 0).any():
        raise ValueError("the metric must be non-negative")
    steps = torch.cumulative_trapezoid(metric.sqrt(), x=x)
    return torch.cat([steps.new_zeros(1), steps])


def distance(x, metric, a, b) -> torch.Tensor:
    """
    The distance between a and b, the absolute value of int_a^b sqrt(metric) dx.

    The cumulative distance is interpolated linearly between grid points.

    Args:
        x: the increasing grid [n_x].
        metric: the metric on the grid [n_x], non-negative.
        a: the first points, a number or a tensor.
        b: the second points, broadcastable with `a`.

    Returns:
        The distances, with the broadcast shape of `a` and `b`.
    """
    x, metric = as_tensors(x, metric)
    a, b = (torch.as_tensor(v, dtype=x.dtype, device=x.device) for v in (a, b))
    a, b = torch.broadcast_tensors(a, b)
    cumulative = cumulative_distance(x, metric)
    return (interp(b, x, cumulative) - interp(a, x, cumulative)).abs()
