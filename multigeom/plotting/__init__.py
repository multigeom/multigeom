"""
Plotting (needs matplotlib): the figures of the estimates at each noise scale
(`per_scale`) and of the whitening of the images (`whitening`), in the style of the
paper (`style`). Importing this subpackage sets the matplotlib style of the paper
(fonts, sizes, axes without top and right spines); the rest of `multigeom` does not
import matplotlib.
"""

from .per_scale import plot_coding_spectrum, plot_env_spectrum, plot_per_scale
from .retina import plot_cell_counts, plot_cell_responses
from .sampling import get_trajectory_scales, plot_posterior_samples
from .whitening import (
    plot_denoised,
    plot_training_loss,
    plot_recovered,
    plot_whitened_vs_standard,
    plot_samples,
    plot_trajectories,
    plot_whitening_examples,
    plot_whitening_kernel,
    plot_whitening_spectra,
)

__all__ = [
    "plot_per_scale",
    "plot_env_spectrum",
    "plot_coding_spectrum",
    "plot_whitening_spectra",
    "plot_whitening_kernel",
    "plot_whitening_examples",
    "plot_recovered",
    "plot_whitened_vs_standard",
    "plot_samples",
    "plot_trajectories",
    "plot_training_loss",
    "plot_denoised",
    "plot_posterior_samples",
    "get_trajectory_scales",
    "plot_cell_counts",
    "plot_cell_responses",
]
