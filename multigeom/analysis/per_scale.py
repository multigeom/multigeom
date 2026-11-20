"""
Analyses of the estimates at each noise scale (`compute_g_per_scale`,
`compute_env_per_scale`, `compute_coding_spectrum`): the signs of the eigenvectors for
display, curves normalised to compare their shapes, the information under a coding
spectrum, and a paired bootstrap of the mean curves of several areas with its 95% band.
"""

from __future__ import annotations

import numpy as np
import torch


def align_signs(eigvecs: torch.Tensor) -> torch.Tensor:
    """
    The eigenvectors [J, k, P] of the J noise levels, each with the sign that agrees
    with the same eigenvector at the next coarser level (for display: the sign of an
    eigenvector is arbitrary); those of the coarsest level have a positive sum.
    """
    # a copy, to leave the results untouched
    aligned = eigvecs.clone()

    # the coarsest level (the last one): a positive sum over the pixels
    aligned[-1] *= torch.where(aligned[-1].sum(dim=1, keepdim=True) < 0, -1.0, 1.0)

    # then level by level towards the finest: the sign of the coarser neighbour
    for level in range(aligned.shape[0] - 2, -1, -1):
        overlap = (aligned[level] * aligned[level + 1]).sum(dim=1, keepdim=True)
        aligned[level] *= torch.where(overlap < 0, -1.0, 1.0)
    return aligned


def compute_normalized(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    `values` divided by the largest value of `reference`: a curve over its own maximum,
    or the eigenvalues of a matrix over its largest leading eigenvalue, so that curves
    of different sizes can be compared by their shape.
    """
    return values / reference.max()


def compute_information(
    curves: torch.Tensor, log_t: torch.Tensor
) -> tuple[float, float]:
    """
    ½ ∫ ρ dτ (trapezoids over log t) of the curve of each image [n, J]: the mean over
    the images, which estimates I(R; X) in nats, and its standard error.
    """
    # one integral per image, then their mean and standard error
    integrals = 0.5 * torch.trapezoid(curves.double(), log_t.double(), dim=1)
    return float(integrals.mean()), float(integrals.std() / len(integrals) ** 0.5)


def compute_bootstrap(curves: dict, n_bootstrap: int, seed: int) -> dict:
    """
    Paired bootstrap of the mean curves of the areas. `curves` maps each area to its
    curves per image [n, J], the same images in the same order; each of the
    `n_bootstrap` resamples draws n images with replacement, the same ones for every
    area, so that the areas can be compared resample by resample. Returns, per area,
    the resampled mean curves [n_bootstrap, J] and the same over their own maximum.
    """
    # how many times each image is drawn in each resample [n_bootstrap, n]
    n_images = len(next(iter(curves.values())))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_images, size=(n_bootstrap, n_images))
    counts = np.zeros((n_bootstrap, n_images))
    np.add.at(counts, (np.arange(n_bootstrap)[:, None], draws), 1)

    # the mean curve of each resample, raw and over its own maximum, for every area
    bootstrap = {}
    for area, values in curves.items():
        means = counts @ values / n_images
        bootstrap[area] = {
            "mean": means,
            "normalized": means / means.max(axis=1, keepdims=True),
        }
    return bootstrap


def compute_band(samples: np.ndarray) -> np.ndarray:
    """
    The 95% band of bootstrapped curves [n_bootstrap, J]: their 2.5th and 97.5th
    percentiles at each point [2, J].
    """
    return np.percentile(samples, [2.5, 97.5], axis=0)
