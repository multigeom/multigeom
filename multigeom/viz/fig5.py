"""
Figure 5: local metric field projected onto the leading eigenvectors of the
reference model's metric.

  A  G(x): ellipses of the 2 × 2 projected metric Vᵀ G(x(a, b)) V of the
     reference (Inception, solid blue) and comparison (ResNet-50, dashed
     orange) encoders, at every image x(a, b) of the Stage-6 grid; V1 left,
     V4 right.
  B  same as A for J(x).

Geometry of the drawing. Each ellipse is centred at its grid coefficients
(a, b) ∈ {−h, …, h}², so neighbouring centres are one unit apart along e₁
(= v₁, horizontal) and e₂ (= v₂, vertical) whatever the step sizes ε₁, ε₂.
Ellipse shape follows multigeom/geometry/metric_field.py: full axes 2√μ_max and 2√μ_min of
the 2 × 2 matrix, major axis at the stored angle from e₁. Within each
(metric, area) plot every ellipse of both encoders is multiplied by one
common factor, chosen so that the longest major axis in that plot is
--fill (default 0.9) grid units. Sizes are therefore comparable across grid
points and between the two encoders inside a plot, but not between plots.

Input: <runs-dir>/metric_field/img{idx:04d}/{G,J}_{V1,V4}.pt (Stage 6).

Usage:
    python -m multigeom.viz.fig5 --runs-dir runs --out-dir figures/out --img-idx 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


from .style import (
    AREAS, ENCODER_COLORS, ENCODER_LABELS, IMAGE_SIZE, METRIC_TITLES, as_numpy, blank,
    load_torch, panel_label, plt, save_figure,
)
from ..geometry.metric_field import ellipse_grid

SCRIPT = "fig5"
LINESTYLES = ("-", (0, (4, 2)))          # reference solid, comparison dashed


def parse_args(argv: list[str] | None = None):
    """Command-line arguments of `python -m multigeom.viz.fig5`."""
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="figures/out")
    p.add_argument("--img-idx", type=int, default=0)
    p.add_argument("--metric-field-dir", default=None,
                   help="default: <runs-dir>/metric_field/img{idx:04d}")
    p.add_argument("--fill", type=float, default=0.9,
                   help="longest major axis in each plot, in grid units")
    p.add_argument("--show-x0", action="store_true",
                   help="add a thumbnail of the reference image")
    return p.parse_args(argv)


def draw_field(ax, rec: dict, fill: float) -> list[str]:
    """Draw both encoders' ellipses for one (metric, area) record; returns encoders drawn."""
    coeffs = as_numpy(rec["grid_coefficients"]).astype(float)
    encoders = list(rec["encoders"])
    ell = {enc: ellipse_grid(as_numpy(rec["matrices_2x2"][enc])) for enc in encoders}
    longest = max(float(e[..., 0].max()) for e in ell.values())
    scale = fill / longest if longest > 0 else 1.0
    for enc, ls in zip(encoders, LINESTYLES):
        for i, a in enumerate(coeffs):
            for j, b in enumerate(coeffs):
                w, h, ang = ell[enc][i, j]
                ax.add_patch(Ellipse((a, b), w * scale, h * scale, angle=ang, fill=False,
                                     edgecolor=ENCODER_COLORS[enc], linestyle=ls, lw=1.5))
    h = coeffs.max()
    ax.set_xlim(-h - 0.9, h + 0.6)
    ax.set_ylim(-h - 0.9, h + 0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
    arrow = dict(arrowstyle="->", lw=1.0, color="black")
    x0, y0 = -h - 0.75, -h - 0.75
    ax.annotate("", xy=(h + 0.4, y0), xytext=(x0, y0), arrowprops=arrow)
    ax.annotate("", xy=(x0, h + 0.4), xytext=(x0, y0), arrowprops=arrow)
    ax.text(h + 0.4, y0 + 0.08, r"$\mathbf{e}_1$", ha="right", va="bottom", fontsize=11)
    ax.text(x0 + 0.08, h + 0.4, r"$\mathbf{e}_2$", ha="left", va="top", fontsize=11)
    return encoders


def make_figure(records: dict, fill: float = 0.9, show_x0: bool = False):
    """
    The figure from Stage-6 records keyed by (metric, area), metric in {"G", "J"}: G(x) in panel A and J(x) in
    panel B, V1 left and V4 right inside each; missing records are left blank. Returns the matplotlib figure.
    """
    fig = plt.figure(figsize=(16, 4.4))
    outer = fig.add_gridspec(1, 2, wspace=0.08)
    first_axes, encoders_drawn = {}, []
    for c, metric in enumerate(("G", "J")):
        inner = outer[c].subgridspec(1, 2, wspace=0.05)
        for k, area in enumerate(AREAS):
            ax = fig.add_subplot(inner[k])
            first_axes.setdefault(metric, ax)
            rec = records.get((metric, area))
            if rec is None:
                blank(ax)
            else:
                drawn = draw_field(ax, rec, fill)
                encoders_drawn = encoders_drawn or drawn
            ax.set_title(area, fontsize=12)
        box_l = first_axes[metric].get_position()
        fig.text(outer[c].get_position(fig).x0 + outer[c].get_position(fig).width / 2,
                 box_l.y1 + 0.08, METRIC_TITLES[metric], ha="center", va="bottom",
                 fontsize=13)
        panel_label(fig, first_axes[metric], "AB"[c], dy=0.06)

    if encoders_drawn:
        handles = [Line2D([0], [0], color=ENCODER_COLORS[enc], ls=ls, lw=2,
                          label=ENCODER_LABELS[enc])
                   for enc, ls in zip(encoders_drawn, LINESTYLES)]
        fig.legend(handles=handles, loc="lower right", ncol=2, frameon=False,
                   bbox_to_anchor=(0.9, -0.06))

    if show_x0:
        x0 = next(iter(records.values()))["x0"]
        ax_img = fig.add_axes([0.02, 0.05, 0.08, 0.3])
        ax_img.imshow(as_numpy(x0).reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray",
                      vmin=-1, vmax=1)
        ax_img.set_title("$x_0$", fontsize=10)
        ax_img.axis("off")
    return fig


def main(argv: list[str] | None = None) -> list[Path]:
    """Draw Figure 5 from the results of the local metric field and write it (command-line entry point)."""
    args = parse_args(argv)
    field_dir = Path(args.metric_field_dir) if args.metric_field_dir else \
        Path(args.runs_dir) / "metric_field" / f"img{args.img_idx:04d}"
    records = {}
    for metric in ("G", "J"):
        for area in AREAS:
            rec = load_torch(field_dir / f"{metric}_{area}.pt", SCRIPT,
                             f"{metric}/{area} metric field (run Stage 6)")
            if rec is not None:
                records[(metric, area)] = rec
    if not records:
        print(f"[{SCRIPT}] no Stage-6 outputs in {field_dir}; nothing to draw.")
        return []

    fig = make_figure(records, args.fill, args.show_x0)
    return save_figure(fig, Path(args.out_dir), f"fig5_img{args.img_idx:04d}")


if __name__ == "__main__":
    main()
