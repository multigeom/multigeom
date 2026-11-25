"""
The style of the figures of the paper: the matplotlib settings (set when this module is
imported), the colours of the areas, and the drawing of an eigenvector and of a panel
letter.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

IMAGE_SIZE = 64
AREA_COLORS = {"V1": "#0072B2", "V4": "#E69F00"}  # Fig. 3B-C scatter

# the colours of the other populations (e.g. retina cell types), in the order drawn
OTHER_COLORS = ("#009E73", "#D55E00", "#CC79A7", "#56B4E9")
RF_COLORS = {"V1": "#56B4E9", "V4": "#E69F00"}  # Fig. 3D outlines (blue, yellow)
EIGVEC_CMAP = "RdBu_r"

# the matplotlib settings of the figures
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(STYLE)


def as_numpy(x) -> np.ndarray:
    """A tensor or array as a NumPy array."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def show_eigenvector(ax: plt.Axes, v, cmap: str = EIGVEC_CMAP) -> None:
    """
    Draw one eigenvector as a square image (64 x 64 for V1 and V4) with a symmetric
    diverging colour scale, clipped at the 99th percentile of abs(v) so a few extreme
    pixels do not wash out the pattern.
    """
    values = as_numpy(v)
    side = int(round(values.size**0.5))
    img = values.reshape(side, side)
    vmax = float(np.percentile(np.abs(img), 99.0)) + 1e-12
    ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_label(
    fig: plt.Figure, ax: plt.Axes, letter: str, dx: float = -0.02, dy: float = 0.01
) -> None:
    """Bold panel letter at the top-left corner of `ax` (figure coordinates)."""
    box = ax.get_position()
    fig.text(
        box.x0 + dx,
        box.y1 + dy,
        letter,
        fontsize=14,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def get_color(colors: dict, name: str, index: int) -> str:
    """
    The colour of population `name` in `colors`, or, for a population it does not
    name, the `index`-th colour of OTHER_COLORS (`index`: its place among those drawn).
    """
    return colors.get(name, OTHER_COLORS[index % len(OTHER_COLORS)])
