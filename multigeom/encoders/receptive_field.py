"""Where in the image a neuron of an encoder looks: the receptive-field regions of a Gaussian readout."""
from __future__ import annotations

import torch

IMAGE_SIZE = 64


def readout_receptive_field(state: dict, resolution: int = 256, feature_side: int | None = None):
    """
    Union of the neurons' half-maximum readout regions, sampled on a `resolution` x `resolution` grid over the 64 x 64 image.

    The Gaussian readout weights feature-map cell (h, w) by exp(-||(u_w, u_h) - mu_n||^2 / (2 sigma_n^2)), with the S cells
    placed at u = linspace(-1, 1, S); mu_n[0] is the horizontal (column) and mu_n[1] the vertical (row) coordinate, and
    sigma_n is clamped at 0.05 as in the forward pass. Feature cell s covers the image pixels [s * 64 / S, (s + 1) * 64 / S),
    so a point at pixel coordinate p (pixel k spans [k, k + 1)) has u(p) = 2 (p * S / 64 - 1/2) / (S - 1) - 1. This maps
    cell centres to pixel positions only: the backbone's own receptive field, which widens each region further, is not
    included.

    Args:
        state: the state dict of the encoder, with `readout.mu` [N, 2], `readout.sigma` [N] and, if the encoder has it,
            `readout.grid_x` (its length is the side S of the feature map).
        resolution: side of the sampling grid.
        feature_side: S, when `state` has no `readout.grid_x`.

    Returns:
        (field [resolution, resolution] = max over the neurons of the normalised Gaussians, pixel coordinates [resolution]
        of the samples).
    """
    mu = torch.as_tensor(state["readout.mu"]).double()                        # [N, 2]
    sigma = torch.as_tensor(state["readout.sigma"]).double().clamp(min=0.05)   # [N]
    grid = state.get("readout.grid_x")
    if grid is not None:
        feature_side = int(grid.numel())
    if feature_side is None:
        raise ValueError("the state has no readout.grid_x: give feature_side, the side of the feature map")
    p = (torch.arange(resolution, dtype=torch.float64) + 0.5) * IMAGE_SIZE / resolution
    u = 2.0 * (p * feature_side / IMAGE_SIZE - 0.5) / (feature_side - 1) - 1.0
    dx2 = (u.unsqueeze(0) - mu[:, 0:1]) ** 2                                   # [N, R] columns
    dy2 = (u.unsqueeze(0) - mu[:, 1:2]) ** 2                                   # [N, R] rows
    field = torch.exp(-(dy2.unsqueeze(2) + dx2.unsqueeze(1)) / (2.0 * sigma.view(-1, 1, 1) ** 2))     # [N, R, R]
    return field.max(dim=0).values.numpy(), p.numpy()
