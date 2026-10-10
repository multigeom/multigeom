"""
Appendix figure fig:app-bw: Bures-Wasserstein distance between reconstructed
geometries as a function of k.

d_BW(A, B) between the rank-k reconstructions V Λ Vᵀ of two metrics (paper
Eq. (eq:bures)), averaged across test images with s.e.m. bands, for four
comparisons: V1 vs V4 within Inception and within ResNet-50 (same model,
different areas; dashed) and Inception vs ResNet-50 within V1 and within V4
(same area, different models; solid). Left: G(x). Right: J(x).

Input: runs/metrics/cross_encoder_metrics.json (Stage 5).

Usage:
    python -m multigeom.viz.fig_app_bw --runs-dir runs --out-dir figures/out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


from .style import cross_encoder_metrics, plot_comparisons, plt, save_figure

SCRIPT = "fig_app_bw"
TITLES = {"G": "Multi-scale inf. geom.", "J": "Information geom."}


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="figures/out")
    return p.parse_args(argv)


def make_figure(metrics: dict):
    """Fig. 6 from the Stage-5 JSON (as a dict): d_BW against k for G(x) (left) and J(x) (right); returns the figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric in zip(axes, ("G", "J")):
        n = plot_comparisons(ax, metrics, metric, "bures_wasserstein", SCRIPT)
        ax.set_title(TITLES[metric], fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of eigenvectors  $k$", fontsize=12)
        ax.set_xlim(left=0)
        if n == 0:
            ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", color="0.5")
    axes[0].set_ylabel("Bures-Wasserstein  $d_{BW}$", fontsize=12)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    metrics = cross_encoder_metrics(Path(args.runs_dir), SCRIPT)
    if metrics is None:
        return []
    return save_figure(make_figure(metrics), Path(args.out_dir), "fig_app_bw")


if __name__ == "__main__":
    main()
