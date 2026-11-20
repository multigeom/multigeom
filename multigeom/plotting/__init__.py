"""
Plotting (needs matplotlib): the figures of the estimates at each noise scale
(`per_scale`), in the style of the paper (`style`). Importing this subpackage sets the
matplotlib style of the paper (fonts, sizes, axes without top and right spines); the
rest of `multigeom` does not import matplotlib.
"""

from .per_scale import plot_coding_spectrum, plot_env_spectrum, plot_per_scale

__all__ = ["plot_per_scale", "plot_env_spectrum", "plot_coding_spectrum"]
