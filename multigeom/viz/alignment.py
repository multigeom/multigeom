"""Figures of the comparison between configurations, from the Stage-5 JSON: subspace alignment (Fig. 4D-E) and the
Bures-Wasserstein distance (Fig. 6)."""
from __future__ import annotations

import json
from pathlib import Path


def plot_alignment(metrics, figsize=(10, 3.8)):
    """Fig. 4D (G(x)) and 4E (J(x)): subspace alignment vs the number k of eigenvectors, from the Stage-5 JSON."""
    import matplotlib.pyplot as plt

    from .fig4 import panel_alignment
    from .style import panel_label

    if not isinstance(metrics, dict):
        with open(metrics) as f:
            metrics = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, metric, letter in zip(axes, ("G", "J"), "DE"):
        panel_alignment(ax, metrics, metric)
        panel_label(fig, ax, letter)
    return fig


def plot_bures_wasserstein(metrics):
    """Fig. 6 from the Stage-5 JSON (a dict, or the path to it): d_BW against k for G(x) (left) and J(x) (right)."""
    from .fig_app_bw import make_figure
    if not isinstance(metrics, dict):
        metrics = json.loads(Path(metrics).read_text())
    return make_figure(metrics)
