"""
The figures of the retina models, in the layout of open-retina's information-geometry
figures (`run_example._plot_cell_summary`), so that they can be compared with them.

    plot_cell_summary   one row per image: the image, then the leading eigenvectors of
                        a metric (G(x) or J(x)), each with its share of the eigenvalues
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .style import plt


def get_summary_data(images: torch.Tensor, eigenpairs: list, n_eig: int) -> dict:
    """
    The data of `plot_cell_summary`: the images [n, H, W], and per image the leading
    `n_eig` eigenvectors [n_eig, H, W], each over its largest |value|, and their shares
    of the sum of the eigenvalues given [n_eig].
    """
    side = images.shape[-1]
    data = {"images": images[:, 0].cpu().numpy(), "vectors": [], "shares": []}
    for pair in eigenpairs:
        vectors = pair.eigenvectors[:n_eig].reshape(n_eig, side, side).cpu().numpy()
        vectors = vectors / (np.abs(vectors).max(axis=(1, 2), keepdims=True) + 1e-9)
        values = pair.eigenvalues.cpu().double()
        data["vectors"].append(vectors)
        data["shares"].append((values / values.sum()).numpy()[:n_eig])
    return data


def plot_cell_summary(
    images: torch.Tensor,
    eigenpairs: list,
    title: str,
    labels: list | None = None,
    n_eig: int = 5,
) -> tuple[plt.Figure, dict]:
    """
    One row per image of `images` [n, 1, H, W]: the image (grey), then the leading
    `n_eig` eigenvectors of its metric (`eigenpairs`, one output with `eigenvalues` and
    `eigenvectors` per image), each over its largest |value| (red positive, blue
    negative) with its share of the sum of the eigenvalues given; `labels` on the left
    of the rows. Returns the figure and the data it shows.
    """
    # the data drawn, and one panel per image and column
    data = get_summary_data(images, eigenpairs, n_eig)
    n_images = len(data["images"])
    fig, axes = plt.subplots(
        n_images,
        n_eig + 1,
        figsize=(2.0 * (n_eig + 1), 2.0 * n_images),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    # each row: the image, then the eigenvectors with their shares
    for row in range(n_images):
        axes[row, 0].imshow(data["images"][row], cmap="gray")
        axes[row, 0].set_ylabel(labels[row] if labels else f"idx{row}", fontsize=8)
        for k in range(n_eig):
            ax = axes[row, k + 1]
            ax.imshow(data["vectors"][row][k], cmap="RdBu_r", vmin=-1, vmax=1)
            ax.text(
                0.04,
                0.95,
                f"{data['shares'][row][k] * 100:.0f}%",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    # the column titles, no ticks, and the axes frames
    axes[0, 0].set_title("stimulus", fontsize=8)
    for k in range(n_eig):
        axes[0, k + 1].set_title(f"e{k + 1}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)

    # the colour scale, shared, and the title
    bar = fig.colorbar(
        ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu_r"),
        ax=axes,
        fraction=0.02,
        pad=0.01,
        ticks=[-1, 0, 1],
    )
    bar.ax.set_yticklabels(["−", "0", "+"])
    bar.set_label("eigenvector weight (per-panel normalised)", fontsize=8)
    fig.suptitle(title, fontsize=10)
    return fig, data
