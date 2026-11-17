"""
The nearest-neighbour estimate of the mutual information from the multi-scale Fisher
information (Azeglio, Laquitaine, Ferrari, Chalk, NeurIPS 2026, arXiv 2605.06304).

The identity I(R; X) = (1/2) E_x[Tr G(x)] is estimated from N stimuli x_i drawn from
the prior and G at each of them. For each stimulus, u_i is the unit vector towards its
nearest neighbour in Euclidean distance. As N grows the nearest-neighbour directions
become uniform on the sphere, E[u u^T] = I / d, and

    I(R; X) ~ (d / 2) mean_i u_i^T G(x_i) u_i     (in nats).
"""

from __future__ import annotations

import torch

from ._grid import as_tensors


def nearest_neighbour_directions(samples, chunk_size: int = 2048) -> torch.Tensor:
    """
    The unit vector from each sample towards its nearest neighbour (Euclidean).

    The distances are computed in blocks of `chunk_size` samples, so memory grows as
    chunk_size * N, not N^2.

    Args:
        samples: the stimuli [N, d], N >= 2, no two equal.
        chunk_size: the number of samples per block.

    Returns:
        The directions [N, d].
    """
    (samples,) = as_tensors(samples)
    if samples.dim() != 2 or samples.shape[0] < 2:
        raise ValueError(f"samples must be [N, d] with N >= 2, got {samples.shape}")
    nearest = []
    for start in range(0, samples.shape[0], chunk_size):
        block = samples[start : start + chunk_size]
        dist = torch.cdist(block, samples, compute_mode="donot_use_mm_for_euclid_dist")
        rows = torch.arange(block.shape[0], device=samples.device)
        dist[rows, rows + start] = torch.inf
        nearest.append(dist.argmin(1))
    offsets = samples[torch.cat(nearest)] - samples
    return offsets / offsets.norm(dim=1, keepdim=True)


def nearest_neighbour_information(samples, g, chunk_size: int = 2048) -> torch.Tensor:
    """
    The nearest-neighbour estimate (d / 2) mean_i u_i^T G(x_i) u_i of I(R; X), in nats.

    Args:
        samples: the stimuli x_i [N, d], drawn from the prior.
        g: G at the samples [N, d, d], or one G [d, d] shared by all of them.
        chunk_size: the number of samples per block of distances.

    Returns:
        The estimate of I(R; X), a scalar tensor.
    """
    samples, g = as_tensors(samples, g)
    u = nearest_neighbour_directions(samples, chunk_size)
    quadratic = torch.einsum("ni,nij,nj->n", u, g.expand(len(u), -1, -1), u)
    return 0.5 * samples.shape[1] * quadratic.mean()
