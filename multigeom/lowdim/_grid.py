"""
Shared helpers of `multigeom.lowdim`: inputs as tensors, the checks of a stimulus grid
and linear interpolation.
"""

from __future__ import annotations

import math

import torch


def as_tensors(*arrays) -> tuple[torch.Tensor, ...]:
    """
    Convert arrays to tensors of one dtype on one device.

    The first floating-point tensor among the inputs sets the dtype and the device.
    Without one, the result is float64 on the CPU.

    Args:
        *arrays: tensors, NumPy arrays, lists or numbers.

    Returns:
        The inputs as tensors, in the same order.
    """
    like = next(
        (a for a in arrays if isinstance(a, torch.Tensor) and a.is_floating_point()),
        None,
    )
    dtype = torch.float64 if like is None else like.dtype
    device = None if like is None else like.device
    return tuple(torch.as_tensor(a, dtype=dtype, device=device) for a in arrays)


def tiny(dtype: torch.dtype) -> float:
    """The floor applied to densities before a logarithm or a division."""
    return max(1e-300, torch.finfo(dtype).tiny)


def grid_step(x: torch.Tensor) -> float:
    """
    The step of a uniform, increasing one-dimensional grid.

    Args:
        x: the grid [n_x], at least 3 points.

    Returns:
        The step x[1] - x[0].

    Raises:
        ValueError: if the grid is not one-dimensional, increasing and uniform.
    """
    if x.dim() != 1 or x.numel() < 3:
        raise ValueError(f"the grid must be 1-D with 3 points or more, got {x.shape}")
    steps = x.diff()
    step = steps[0].item()
    # 1e-3 of a step leaves room for the rounding of a float32 linspace
    if step <= 0 or (steps - step).abs().max().item() > 1e-3 * step:
        raise ValueError("the grid must be uniform and increasing")
    return step


def check_on_grid(x: torch.Tensor, **arrays: torch.Tensor) -> None:
    """
    Check that every array has the shape of the grid.

    Args:
        x: the grid [n_x].
        **arrays: the arrays to check, by name.

    Raises:
        ValueError: if an array does not have the shape [n_x].
    """
    for name, a in arrays.items():
        if a.shape != x.shape:
            raise ValueError(f"{name} has shape {tuple(a.shape)}, the grid {x.shape}")


def interp(xq: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation, as `numpy.interp`: the end values are kept outside the grid.

    Args:
        xq: the query points, any shape.
        xp: the increasing grid [n].
        fp: the values on the grid [n].

    Returns:
        The interpolated values, with the shape of `xq`.
    """
    q = xq.reshape(-1).contiguous()
    i = torch.searchsorted(xp, q).clamp(1, xp.numel() - 1)
    x0, x1 = xp[i - 1], xp[i]
    w = ((q - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return (fp[i - 1] + w * (fp[i] - fp[i - 1])).reshape(xq.shape)


def gaussian(a: torch.Tensor, mean: torch.Tensor, sd: float) -> torch.Tensor:
    """The density of N(mean, sd^2) at a, broadcast."""
    return torch.exp(-0.5 * ((a - mean) / sd) ** 2) / (math.sqrt(2 * math.pi) * sd)
