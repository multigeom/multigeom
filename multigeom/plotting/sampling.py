"""
The figure of images generated from neural responses: for one presented image, the
noisy image and the estimate x̂(x_t, r) of each step of the backward diffusion process
conditioned on the responses r of several populations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from .style import as_numpy


def plot_posterior_samples(
    blocks: dict, params: dict, title: str = ""
) -> tuple[plt.Figure, dict]:
    """
    For one presented image, one block of rows per population of `blocks` (name →
    (image [1, 1, H, W], {log t: (x_t [n, 1, H, W], x̂ [n, 1, H, W])})): the noisy image
    x_t of the first sample (top row), then the estimate x̂(x_t, r) of each of the n
    samples, one row each, at each step from the coarsest noise scale (left) to the
    finest (right, the generated image). The left column holds the presented image, in
    each population's own units. The presented image and the estimates share the gray
    scale of the presented image (its 0.5th to 99.5th percentiles); each noisy image
    has its own. For the populations in `params["remove_mean"]`, each estimate is drawn
    with its mean removed, on the gray scale of that sample's generated image (the
    0.5th to 99.5th percentiles), for models whose samples drift in overall
    brightness. A red line separates two populations. Returns the figure and the
    images drawn ({name: {"noisy": [J, H, W], "estimates": [n, J, H, W]}}).
    """
    # the rows: per population, the noisy row and one row per sample, an empty row
    # between two populations
    names = list(blocks)
    scales = list(next(iter(blocks.values()))[1])
    n_samples = len(next(iter(blocks.values()))[1][scales[0]][1])
    gap, heights, places = 0.25, [], []
    for k, name in enumerate(names):
        if k > 0:
            heights.append(gap)
        places.append([len(heights) + i for i in range(1 + n_samples)])
        heights += [1.0] * (1 + n_samples)
    n_columns = len(scales) + 1
    fig = plt.figure(
        figsize=(1.1 * n_columns + 0.8, 1.1 * sum(heights) + 0.8),
        dpi=params["screen_dpi"],
    )
    grid = fig.add_gridspec(
        len(heights),
        n_columns,
        height_ratios=heights,
        hspace=0.06,
        wspace=0.06,
        left=0.09,
        right=0.995,
        bottom=0.01,
        top=0.93,
    )

    # each population: the presented image, the noisy row, the estimates
    drawn, all_axes = {}, []
    for name, rows in zip(names, places):
        image, trajectory = blocks[name]
        clean = as_numpy(image[0, 0].double())
        low, high = np.percentile(clean, [0.5, 99.5])
        axes = [[fig.add_subplot(grid[r, c]) for c in range(n_columns)] for r in rows]
        all_axes.append(axes)
        axes[0][0].imshow(clean, cmap="gray", vmin=low, vmax=high)
        axes[0][0].set_title("presented", fontsize=8)
        for r in range(1, len(rows)):
            axes[r][0].axis("off")
        noisy = np.stack([as_numpy(trajectory[s][0][0, 0].double()) for s in scales])
        estimates = np.stack(
            [
                np.stack([as_numpy(trajectory[s][1][i, 0].double()) for s in scales])
                for i in range(n_samples)
            ]
        )
        centred = name in params.get("remove_mean", ())
        if centred:
            estimates = estimates - estimates.mean(axis=(2, 3), keepdims=True)
        drawn[name] = {"noisy": noisy, "estimates": estimates}
        for j in range(len(scales)):
            limit = np.percentile(np.abs(noisy[j]), 99.5)
            axes[0][j + 1].imshow(noisy[j], cmap="gray", vmin=-limit, vmax=limit)
        for i in range(n_samples):
            scale = (low, high)
            if centred:
                scale = tuple(np.percentile(estimates[i, -1], [0.5, 99.5]))
            for j in range(len(scales)):
                axes[i + 1][j + 1].imshow(
                    estimates[i, j], cmap="gray", vmin=scale[0], vmax=scale[1]
                )

        # the labels of the rows, and the name of the population
        axes[0][1].set_ylabel(r"$x_t$", fontsize=9)
        for i in range(n_samples):
            suffix = "\n(mean removed)" if centred else ""
            axes[i + 1][1].set_ylabel(
                rf"$\hat x(x_t, r)$" + f"\nsample {i + 1}{suffix}", fontsize=8
            )
        top = axes[0][0].get_position().y1
        bottom = axes[-1][0].get_position().y0
        fig.text(
            0.01,
            (top + bottom) / 2,
            name,
            rotation=90,
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # no axes; the noise scale of each column, above the first population
    for axes in all_axes:
        for row in axes:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
    for j, scale in enumerate(scales):
        label = rf"$\tau$ = {round(scale, 1) + 0.0:.1f}"
        last = j == len(scales) - 1
        all_axes[0][0][j + 1].set_title(
            label + ("\n(generated)" if last else ""), fontsize=8
        )
    if title:
        fig.suptitle(title, fontsize=10)

    # a red line between two populations, in the middle of the gap
    for above, below in zip(all_axes[:-1], all_axes[1:]):
        middle = (above[-1][0].get_position().y0 + below[0][0].get_position().y1) / 2
        fig.add_artist(
            plt.Line2D([0.01, 0.995], [middle, middle], color="red", linewidth=1.5)
        )
    return fig, drawn


def get_trajectory_scales(trajectory: dict, alphas_cumprod: torch.Tensor) -> dict:
    """The trajectory of `sample_images` ({step: (x_t, x̂)}) keyed by the noise scale
    τ = log t of each step instead, in the same order (coarsest first)."""
    return {
        float(((1 - alphas_cumprod[s]) / alphas_cumprod[s]).log()): pair
        for s, pair in trajectory.items()
    }
